#include "clusterlm/node/node_worker.hpp"

#include <algorithm>
#include <condition_variable>
#include <map>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "clusterlm/common/clock.hpp"
#include "clusterlm/common/log.hpp"
#include "clusterlm/domain/backend_adapter.hpp"
#include "clusterlm/lease/lease_store.hpp"
#include "clusterlm/platform/fs_safety.hpp"
#include "clusterlm/protocol/wire.hpp"

namespace clusterlm::node {

using namespace std::chrono_literals;
using protocol::Channel;
using protocol::Message;
using protocol::MessageStream;

std::string_view to_string(NodeState s) {
  switch (s) {
    case NodeState::kBusy: return "Busy";
    case NodeState::kAvailable: return "Available";
    case NodeState::kPreparing: return "Preparing";
    case NodeState::kReady: return "Ready";
    case NodeState::kInferencing: return "Inferencing";
    case NodeState::kReleasing: return "Releasing";
    case NodeState::kCleanupPending: return "CleanupPending";
  }
  return "?";
}

namespace {

constexpr auto kPollInterval = 100ms;
constexpr auto kHelloTimeout = 5s;

// Resolves a domain's objects from the lease store. Every pointer it hands out is into this process's own
// lease-owned memory; there is no fallback to Father or to any other source.
class LeaseObjectResolver final : public objects::ObjectResolver {
 public:
  LeaseObjectResolver(const protocol::PreparePlan& plan, const lease::LeaseStore& store) : plan_(plan), store_(store) {
    for (const auto& a : plan_.assignments) {
      const auto& obj = plan_.manifest.objects[a.object_index];
      by_name_.emplace(obj.name, a);
    }
  }

  Result<objects::ProvisionedObject> resolve(std::string_view name) const override {
    auto it = by_name_.find(std::string(name));
    if (it == by_name_.end())
      return make_error(ErrorCode::kNotFound, "object " + std::string(name) + " is not assigned to this Node");
    CLM_ASSIGN_OR_RETURN(auto bytes, store_.sealed_bytes(it->second.object_index));
    objects::ProvisionedObject out;
    out.entry = &plan_.manifest.objects[it->second.object_index];
    out.bytes = bytes;
    out.target = it->second.target;
    return out;
  }

 private:
  const protocol::PreparePlan& plan_;
  const lease::LeaseStore& store_;
  std::unordered_map<std::string, protocol::ObjectAssignment> by_name_;
};

Message error_reply(const Status& s, protocol::MessageType in_reply_to) {
  return protocol::ErrorMessage{s.code(), s.message(), in_reply_to};
}

}  // namespace

struct NodeWorker::Impl {
  NodeConfig cfg;
  std::unique_ptr<transport::Listener> listener;
  std::unique_ptr<domain::BackendAdapter> backend;
  std::shared_ptr<transport::SimulatedLink> egress_link;
  std::thread accept_thread;
  std::atomic<bool> stopping{false};

  mutable std::mutex mu;  // guards the lease/plan/stream state below
  std::condition_variable cv;
  NodeState state = NodeState::kBusy;
  LeaseGeneration lease{0};
  NodeStatus counters;
  std::unique_ptr<lease::LeaseStore> store;
  std::optional<protocol::PreparePlan> plan;
  std::size_t sealed_count = 0;
  std::uint64_t prepare_started_ns = 0;
  std::unique_ptr<LeaseObjectResolver> resolver;
  std::map<std::uint32_t, std::unique_ptr<domain::ExecutionDomain>> domains;  // by stage id
  std::shared_ptr<MessageStream> control;
  std::shared_ptr<MessageStream> father_activation;
  std::shared_ptr<MessageStream> provision;
  std::vector<std::shared_ptr<MessageStream>> peer_inbound;
  std::optional<protocol::AuthorizePeer> downstream;  // we forward to this peer
  std::shared_ptr<MessageStream> downstream_stream;
  std::vector<protocol::AuthorizePeer> inbound_allowed;  // peers allowed to forward to us
  std::vector<std::thread> handlers;

  std::mutex exec_mu;  // serializes domain execution/commit/abort/release

  // ---- helpers ---------------------------------------------------------------------------------------

  // Lifecycle phases are reported through the lease store so its crash hook (and the bench's
  // --crash-at fault injection) sees every phase exactly once per occurrence.
  void phase(std::string_view name) {
    if (store) store->notify_phase(name);
  }

  std::unique_ptr<transport::Connection> wrap(std::unique_ptr<transport::Connection> c) {
    if (!cfg.impairment && !cfg.faults) return c;
    return transport::impair(std::move(c), cfg.impairment.value_or(transport::NetworkConditions{}), egress_link,
                             cfg.faults);
  }

  void set_state(NodeState s) {
    if (state != s) log::info("node_state", {{"from", std::string(to_string(state))}, {"to", std::string(to_string(s))}});
    state = s;
    counters.state = s;
  }

  protocol::OfferResources offer_locked() const {
    protocol::OfferResources offer;
    offer.lease = lease;
    offer.safe_ram_bytes = cfg.ram_allowance;
    offer.safe_vram_bytes = cfg.vram_allowance;
    offer.staging_disk_bytes = cfg.disk_allowance;
    offer.cpu_summary = "reported-by-clusterlm-bench";
    offer.gpu_summary = backend->info().supports_gpu ? "gpu" : "none";
    return offer;
  }

  // ---- connection acceptance -------------------------------------------------------------------------

  void accept_loop() {
    while (!stopping.load()) {
      auto conn = listener->accept(kPollInterval);
      if (!conn.is_ok()) {
        if (conn.status().code() == ErrorCode::kDeadlineExceeded) continue;
        if (stopping.load()) break;
        log::warn("accept_failed", {{"error", conn.status().to_string()}});
        continue;
      }
      std::lock_guard lock(mu);
      handlers.emplace_back([this, c = wrap(std::move(conn).value())]() mutable { handle_connection(std::move(c)); });
    }
  }

  void handle_connection(std::unique_ptr<transport::Connection> conn) {
    auto frame = conn->receive(kHelloTimeout);
    if (!frame.is_ok() || frame->type != static_cast<std::uint16_t>(protocol::MessageType::kHello)) {
      conn->close();
      return;
    }
    auto decoded = protocol::decode(protocol::MessageType::kHello, frame->payload);
    if (!decoded.is_ok()) {
      conn->close();
      return;
    }
    const auto hello = std::get<protocol::Hello>(decoded.value());
    if (hello.protocol_version != protocol::kProtocolVersion) {
      conn->close();
      return;
    }
    auto stream = std::make_shared<MessageStream>(std::move(conn), hello.channel);
    switch (hello.channel) {
      case Channel::kControl: serve_control(stream, hello); break;
      case Channel::kActivation: serve_activation(stream, hello); break;
      case Channel::kProvision: serve_provision(stream, hello); break;
    }
    stream->close();
  }

  Status send_hello_ack(MessageStream& s) {
    protocol::HelloAck ack;
    // Insecure loopback mode has no cryptographic identity; the configured name stands in for it.
    ack.device_id = cfg.security.identity ? cfg.security.identity->fingerprint() : cfg.name;
    {
      std::lock_guard lock(mu);
      ack.lease = lease;
    }
    return s.send(ack);
  }

  // ---- control channel (Father) ------------------------------------------------------------------------

  void serve_control(const std::shared_ptr<MessageStream>& stream, const protocol::Hello& hello) {
    if (hello.role != protocol::NodeRole::kFather) return;
    {
      std::lock_guard lock(mu);
      if (control) {
        (void)stream->send(error_reply(make_error(ErrorCode::kFailedPrecondition, "a Father is already connected"),
                                       protocol::MessageType::kHello));
        return;
      }
      control = stream;
    }
    if (!send_hello_ack(*stream).is_ok()) return drop_control(stream);
    {
      std::lock_guard lock(mu);
      if (state == NodeState::kAvailable) (void)stream->send(offer_locked());
    }
    while (!stopping.load()) {
      auto received = stream->receive(kPollInterval);
      if (!received.is_ok()) {
        if (received.status().code() == ErrorCode::kDeadlineExceeded) continue;
        break;
      }
      handle_control(*stream, received->message, received->correlation);
    }
    drop_control(stream);
  }

  // Father loss: the lease cannot outlive its coordinator. Release everything.
  void drop_control(const std::shared_ptr<MessageStream>& stream) {
    bool had_lease;
    {
      std::lock_guard lock(mu);
      if (control != stream) return;
      control.reset();
      had_lease = plan.has_value() || state == NodeState::kPreparing || state == NodeState::kReady;
    }
    if (had_lease && !stopping.load()) {
      log::warn("father_lost_releasing");
      release(protocol::ReleaseReason::kFault, /*notify=*/false);
    }
  }

  void handle_control(MessageStream& s, const Message& m, std::uint64_t corr) {
    if (auto* p = std::get_if<protocol::PreparePlan>(&m)) {
      auto r = prepare_plan(*p);
      (void)s.send(r.is_ok() ? Message(r.value()) : error_reply(r.status(), protocol::MessageType::kPreparePlan), corr);
    } else if (auto* a = std::get_if<protocol::AuthorizePeer>(&m)) {
      auto st = authorize_peer(*a);
      (void)s.send(st.is_ok() ? Message(protocol::Pong{0}) : error_reply(st, protocol::MessageType::kAuthorizePeer),
                   corr);
    } else if (auto* o = std::get_if<protocol::OpenSession>(&m)) {
      auto st = open_session(*o);
      (void)s.send(st.is_ok() ? Message(protocol::SessionOpened{o->epoch, o->session})
                              : error_reply(st, protocol::MessageType::kOpenSession),
                   corr);
    } else if (auto* c = std::get_if<protocol::CommitWindow>(&m)) {
      auto r = commit(*c);
      (void)s.send(r.is_ok() ? Message(protocol::CommitAckMessage{c->stage, r.value()})
                             : error_reply(r.status(), protocol::MessageType::kCommitWindow),
                   corr);
    } else if (auto* ab = std::get_if<protocol::AbortSession>(&m)) {
      auto st = abort(*ab);
      (void)s.send(st.is_ok() ? Message(protocol::SessionOpened{ab->epoch, ab->session})
                              : error_reply(st, protocol::MessageType::kAbortSession),
                   corr);
    } else if (auto* rl = std::get_if<protocol::ReleaseLease>(&m)) {
      LeaseGeneration current;
      {
        std::lock_guard lock(mu);
        current = lease;
      }
      if (rl->lease != current) {
        (void)s.send(error_reply(make_error(ErrorCode::kStaleEpoch, "release names a stale lease"),
                                 protocol::MessageType::kReleaseLease),
                     corr);
        return;
      }
      auto report = release(rl->reason, /*notify=*/false);
      (void)s.send(report, corr);
      std::lock_guard lock(mu);
      if (state == NodeState::kAvailable) (void)s.send(offer_locked());
    } else if (auto* ping = std::get_if<protocol::Ping>(&m)) {
      (void)s.send(protocol::Pong{ping->nonce}, corr);
    } else {
      (void)s.send(error_reply(make_error(ErrorCode::kProtocolError, "unexpected control message"), protocol::type_of(m)),
                   corr);
    }
  }

  Result<protocol::PlanAccepted> prepare_plan(const protocol::PreparePlan& p) {
    std::lock_guard lock(mu);
    if (p.lease != lease) return make_error(ErrorCode::kStaleEpoch, "PreparePlan names lease " + p.lease.str());
    if (state != NodeState::kAvailable)
      return make_error(ErrorCode::kFailedPrecondition, "Node is " + std::string(to_string(state)));
    if (p.backend_build != backend->info().build_hash)
      return make_error(ErrorCode::kVersionMismatch, "backend build mismatch: Node has " + backend->info().build_hash);
    CLM_RETURN_IF_ERROR(p.manifest.validate());

    std::uint64_t ram = 0, vram = 0, disk = 0;
    for (const auto& a : p.assignments) {
      if (a.object_index >= p.manifest.objects.size())
        return make_error(ErrorCode::kOutOfRange, "assignment names a missing object");
      const auto& obj = p.manifest.objects[a.object_index];
      if (obj.father_only()) return make_error(ErrorCode::kPermissionDenied, "plan assigns Father-only object");
      switch (a.target) {
        case objects::AllocationTarget::kCpuResident: ram += obj.byte_size; break;
        // GPU-resident objects land in bounded RAM staging, are uploaded, and the staging is discarded. The
        // reference backend has no GPU, so it keeps them in RAM and accounts them against both budgets.
        case objects::AllocationTarget::kGpuResident: vram += obj.byte_size; ram += obj.byte_size; break;
        case objects::AllocationTarget::kTemporaryBacking: disk += obj.byte_size; break;
        case objects::AllocationTarget::kFatherOnly:
          return make_error(ErrorCode::kPermissionDenied, "plan assigns Father-only target");
      }
    }
    if (ram > std::min(cfg.ram_allowance, p.ram_cap_bytes ? p.ram_cap_bytes : cfg.ram_allowance))
      return make_error(ErrorCode::kResourceExhausted, "plan RAM " + std::to_string(ram) + " exceeds allowance");
    if (vram > cfg.vram_allowance)
      return make_error(ErrorCode::kResourceExhausted, "plan VRAM exceeds allowance");
    if (disk > cfg.disk_allowance)
      return make_error(ErrorCode::kResourceExhausted, "plan requires disk staging beyond allowance");

    CLM_RETURN_IF_ERROR(store->begin_lease(lease, lease::LeaseBudget{ram, disk}));
    for (const auto& a : p.assignments) {
      const auto& obj = p.manifest.objects[a.object_index];
      const auto placement = a.target == objects::AllocationTarget::kTemporaryBacking ? lease::Placement::kDisk
                                                                                       : lease::Placement::kRam;
      auto created = store->create_object(a.object_index, obj.byte_size, obj.object_digest, placement);
      if (!created.is_ok()) {
        (void)store->release();
        lease = lease.next();
        return created.status();
      }
    }
    plan = p;
    sealed_count = 0;
    prepare_started_ns = monotonic_ns();
    set_state(NodeState::kPreparing);
    log::info("plan_accepted", {{"lease", lease.str()}, {"objects", std::to_string(p.assignments.size())},
                                {"ram_bytes", std::to_string(ram)}, {"disk_bytes", std::to_string(disk)}});
    return protocol::PlanAccepted{lease, p.plan_hash, ram, vram, disk};
  }

  Status authorize_peer(const protocol::AuthorizePeer& a) {
    std::lock_guard lock(mu);
    if (a.lease != lease || !plan) return make_error(ErrorCode::kStaleEpoch, "AuthorizePeer for a stale lease");
    if (a.plan_hash != plan->plan_hash) return make_error(ErrorCode::kPermissionDenied, "plan hash mismatch");
    auto owns = [&](StageId s) {
      return std::any_of(plan->stages.begin(), plan->stages.end(), [&](const auto& st) { return st.stage == s; });
    };
    // The peer's paired identity is trusted only for this lease; release() revokes it.
    if (cfg.security.mode == transport::SecurityConfig::Mode::kMutualTls) cfg.security.trust(a.peer_device_id);
    if (owns(a.from_stage)) {
      downstream = a;
      downstream_stream.reset();
      return Status::ok();
    }
    if (owns(a.to_stage)) {
      inbound_allowed.push_back(a);
      return Status::ok();
    }
    return make_error(ErrorCode::kPermissionDenied, "AuthorizePeer names stages this Node does not own");
  }

  Status open_session(const protocol::OpenSession& o) {
    std::lock_guard exec(exec_mu);
    std::lock_guard lock(mu);
    if (state != NodeState::kReady) return make_error(ErrorCode::kFailedPrecondition, "Node is not Ready");
    for (auto& [stage, d] : domains) CLM_RETURN_IF_ERROR(d->open_session(o.epoch, o.session));
    return Status::ok();
  }

  Result<domain::CommitAck> commit(const protocol::CommitWindow& c) {
    std::lock_guard exec(exec_mu);
    domain::ExecutionDomain* d;
    {
      std::lock_guard lock(mu);
      auto it = domains.find(c.stage.value);
      if (it == domains.end()) return make_error(ErrorCode::kNotFound, "stage not hosted here");
      d = it->second.get();
    }
    return d->commit_window(c.request);
  }

  Status abort(const protocol::AbortSession& a) {
    std::lock_guard exec(exec_mu);
    std::lock_guard lock(mu);
    Status first;
    for (auto& [stage, d] : domains) {
      auto st = d->abort_session(a.epoch, a.session);
      if (!st.is_ok() && first.is_ok() && st.code() != ErrorCode::kNotFound) first = st;
    }
    return first;
  }

  // ---- provision channel -------------------------------------------------------------------------------

  void serve_provision(const std::shared_ptr<MessageStream>& stream, const protocol::Hello& hello) {
    {
      std::lock_guard lock(mu);
      if (hello.role != protocol::NodeRole::kFather || hello.lease != lease) {
        (void)stream->send(error_reply(make_error(ErrorCode::kStaleEpoch, "provision channel for stale lease"),
                                       protocol::MessageType::kHello));
        return;
      }
      provision = stream;
    }
    if (!send_hello_ack(*stream).is_ok()) return;
    while (!stopping.load()) {
      auto received = stream->receive(kPollInterval);
      if (!received.is_ok()) {
        if (received.status().code() == ErrorCode::kDeadlineExceeded) continue;
        break;
      }
      if (auto* chunk = std::get_if<protocol::ProvisionChunk>(&received->message)) {
        auto st = provision_chunk(*chunk);
        // Chunks are acknowledged only on failure; flow control is the transport's bounded buffers.
        if (!st.is_ok()) (void)stream->send(error_reply(st, protocol::MessageType::kProvisionChunk), received->correlation);
      } else if (auto* seal = std::get_if<protocol::SealObject>(&received->message)) {
        auto st = seal_object(*seal);
        (void)stream->send(st.is_ok() ? Message(protocol::ObjectSealed{seal->lease, seal->object_index})
                                      : error_reply(st, protocol::MessageType::kSealObject),
                           received->correlation);
      }
    }
    std::lock_guard lock(mu);
    if (provision == stream) provision.reset();
  }

  Status provision_chunk(const protocol::ProvisionChunk& c) {
    phase("transfer");
    std::lock_guard lock(mu);
    if (c.lease != lease || state != NodeState::kPreparing)
      return make_error(ErrorCode::kStaleEpoch, "chunk for a stale or inactive lease");
    auto* writer = store->find_object(c.object_index);
    if (writer == nullptr) return make_error(ErrorCode::kNotFound, "chunk for an object not in the plan");
    CLM_RETURN_IF_ERROR(writer->write_chunk(c.offset, c.data, c.chunk_digest));
    counters.provisioned_bytes += c.data.size();
    return Status::ok();
  }

  Status seal_object(const protocol::SealObject& s) {
    phase("hashing");
    bool complete = false;
    {
      std::lock_guard lock(mu);
      if (s.lease != lease || state != NodeState::kPreparing || !plan)
        return make_error(ErrorCode::kStaleEpoch, "seal for a stale or inactive lease");
      const auto& obj = plan->manifest.objects.at(s.object_index);
      if (s.total_length != obj.byte_size || s.object_digest != obj.object_digest)
        return make_error(ErrorCode::kDataLoss, "seal does not match the plan manifest");
      auto* writer = store->find_object(s.object_index);
      if (writer == nullptr) return make_error(ErrorCode::kNotFound, "seal for an object not in the plan");
      CLM_RETURN_IF_ERROR(writer->seal());
      complete = ++sealed_count == plan->assignments.size();
    }
    if (complete) finish_prepare();
    return Status::ok();
  }

  // All objects validated: build domains, bind objects, allocate state, run a synthetic execution check.
  void finish_prepare() {
    phase("allocation");
    Status st = build_domains();
    std::shared_ptr<MessageStream> ctl;
    protocol::PlanReady ready;
    {
      std::lock_guard lock(mu);
      ctl = control;
      if (st.is_ok()) st = store->mark_ready();
      if (st.is_ok()) {
        set_state(NodeState::kReady);
        ready.lease = lease;
        ready.plan_hash = plan->plan_hash;
        ready.resident_bytes = store->ram_used() + store->disk_used();
        ready.prepare_ns = monotonic_ns() - prepare_started_ns;
      }
    }
    phase("ready");
    if (!st.is_ok()) {
      log::error("prepare_failed", {{"error", st.to_string()}});
      if (ctl) (void)ctl->send(error_reply(st, protocol::MessageType::kPlanReady));
      release(protocol::ReleaseReason::kFault, /*notify=*/false);
      return;
    }
    log::info("plan_ready", {{"lease", ready.lease.str()}, {"resident_bytes", std::to_string(ready.resident_bytes)},
                             {"prepare_ms", std::to_string(ready.prepare_ns / 1'000'000)}});
    if (ctl) (void)ctl->send(ready);
  }

  Status build_domains() {
    std::lock_guard exec(exec_mu);
    std::lock_guard lock(mu);
    resolver = std::make_unique<LeaseObjectResolver>(*plan, *store);
    phase("mapping");
    for (const auto& sa : plan->stages) {
      domain::DomainSpec spec;
      spec.stage = sa.stage;
      spec.role = sa.role;
      spec.layers = sa.layers;
      spec.max_context = sa.max_context;
      spec.max_window = sa.max_window;
      if (sa.role != domain::StageRole::kMiddle)
        return make_error(ErrorCode::kPermissionDenied, "Nodes host token-free middle stages only");
      CLM_ASSIGN_OR_RETURN(auto d, backend->create_domain(plan->manifest, spec));
      CLM_RETURN_IF_ERROR(d->prepare(*resolver));
      // Synthetic execution check on a scratch session: proves the plan executes before reporting Ready.
      const Epoch check_epoch{0};
      const SessionId check_session{~std::uint64_t{0}};
      CLM_RETURN_IF_ERROR(d->open_session(check_epoch, check_session));
      domain::WindowRequest req{check_epoch, check_session, WindowId{1}, 0, StateVersion{0}, 1};
      domain::StageActivations probe;
      probe.layout = d->boundary();
      probe.positions = 1;
      probe.data.assign(probe.layout.floats_per_position(), 0.0f);
      CLM_RETURN_IF_ERROR(d->run_window(req, probe).status());
      CLM_RETURN_IF_ERROR(d->abort_session(check_epoch, check_session));
      domains.emplace(sa.stage.value, std::move(d));
    }
    return Status::ok();
  }

  // ---- activation channel ------------------------------------------------------------------------------

  void serve_activation(const std::shared_ptr<MessageStream>& stream, const protocol::Hello& hello) {
    bool from_father = false;
    {
      std::lock_guard lock(mu);
      if (hello.lease != lease || !plan) {
        (void)stream->send(error_reply(make_error(ErrorCode::kStaleEpoch, "activation channel for stale lease"),
                                       protocol::MessageType::kHello));
        return;
      }
      if (hello.role == protocol::NodeRole::kFather) {
        father_activation = stream;
        from_father = true;
      } else {
        const auto peer_id = stream->peer().authenticated ? stream->peer().device_id : hello.device_id;
        const bool allowed = std::any_of(inbound_allowed.begin(), inbound_allowed.end(),
                                         [&](const auto& a) { return a.peer_device_id == peer_id; });
        if (!allowed) {
          (void)stream->send(error_reply(make_error(ErrorCode::kPermissionDenied, "peer not authorized for this plan"),
                                         protocol::MessageType::kHello));
          return;
        }
        peer_inbound.push_back(stream);
      }
    }
    if (!send_hello_ack(*stream).is_ok()) return;
    while (!stopping.load()) {
      auto received = stream->receive(kPollInterval);
      if (!received.is_ok()) {
        if (received.status().code() == ErrorCode::kDeadlineExceeded) continue;
        break;
      }
      if (auto* run = std::get_if<protocol::RunWindow>(&received->message)) {
        run_window(*run, received->correlation);
      }
    }
    std::lock_guard lock(mu);
    if (from_father && father_activation == stream) father_activation.reset();
  }

  void run_window(const protocol::RunWindow& run, std::uint64_t corr) {
    protocol::StageResult result;
    result.epoch = run.request.epoch;
    result.session = run.request.session;
    result.window = run.request.window;
    result.stage = run.stage;
    result.timings = run.upstream_timings;

    domain::ExecutionDomain* d = nullptr;
    std::shared_ptr<MessageStream> to_father;
    {
      std::lock_guard lock(mu);
      to_father = father_activation;
      if (run.lease != lease) {
        ++counters.stale_rejections;
        result.status = ErrorCode::kStaleEpoch;
        result.error_message = "RunWindow names a stale lease";
      } else if (state != NodeState::kReady) {
        result.status = ErrorCode::kFailedPrecondition;
        result.error_message = "Node is " + std::string(to_string(state));
      } else if (auto it = domains.find(run.stage.value); it == domains.end()) {
        result.status = ErrorCode::kNotFound;
        result.error_message = "stage not hosted here";
      } else {
        d = it->second.get();
        set_state(NodeState::kInferencing);
      }
    }
    if (d != nullptr) {
      phase("inference");
      Stopwatch sw;
      Result<domain::StageActivations> out = [&] {
        std::lock_guard exec(exec_mu);
        return d->run_window(run.request, run.activations);
      }();
      domain::StageTiming timing;
      timing.compute_ns = sw.elapsed_ns();
      result.timings.push_back(timing);
      {
        std::lock_guard lock(mu);
        if (state == NodeState::kInferencing) set_state(NodeState::kReady);
        ++counters.windows_executed;
      }
      if (!out.is_ok()) {
        result.status = out.status().code();
        result.error_message = out.status().message();
        if (out.status().code() == ErrorCode::kStaleEpoch) {
          std::lock_guard lock(mu);
          ++counters.stale_rejections;
        }
      } else if (run.forward_to_peer && has_downstream(run.stage)) {
        // Direct peer path: the activations go to the next stage, not back through Father. A failed
        // forward invalidates the window; Father learns of it here and aborts the session.
        auto fwd = forward(run, std::move(out).value(), result.timings, corr);
        if (fwd.is_ok()) return;
        result.status = fwd.code();
        result.error_message = "forward to peer failed: " + fwd.message();
      } else {
        result.activations = std::move(out).value();
      }
    }
    if (to_father) (void)to_father->send(result, corr);
  }

  bool has_downstream(StageId stage) const {
    std::lock_guard lock(mu);
    return downstream && downstream->from_stage == stage;
  }

  // Forward this stage's output directly to the authorized downstream peer.
  Status forward(const protocol::RunWindow& run, domain::StageActivations&& acts,
                 const std::vector<domain::StageTiming>& timings, std::uint64_t corr) {
    std::shared_ptr<MessageStream> peer;
    protocol::AuthorizePeer auth;
    {
      std::lock_guard lock(mu);
      if (!downstream) return make_error(ErrorCode::kFailedPrecondition, "downstream peer revoked");
      auth = *downstream;
      peer = downstream_stream;
    }
    if (!peer) {
      CLM_ASSIGN_OR_RETURN(peer, connect_peer(auth));
      std::lock_guard lock(mu);
      downstream_stream = peer;
    }
    protocol::RunWindow next;
    next.lease = auth.peer_lease;
    next.request = run.request;
    next.stage = auth.to_stage;
    next.forward_to_peer = true;
    next.upstream_timings = timings;
    next.activations = std::move(acts);
    if (auto st = peer->send(next, corr); !st.is_ok()) {
      std::lock_guard lock(mu);
      downstream_stream.reset();
      return st;
    }
    std::lock_guard lock(mu);
    ++counters.windows_forwarded;
    return Status::ok();
  }

  Result<std::shared_ptr<MessageStream>> connect_peer(const protocol::AuthorizePeer& auth) {
    CLM_ASSIGN_OR_RETURN(auto endpoint, transport::Endpoint::parse(auth.peer_endpoint));
    std::optional<std::string> expected;
    if (cfg.security.mode == transport::SecurityConfig::Mode::kMutualTls) expected = auth.peer_device_id;
    CLM_ASSIGN_OR_RETURN(auto conn, transport::connect(endpoint, cfg.security, expected, 5s));
    auto stream = std::make_shared<MessageStream>(wrap(std::move(conn)), Channel::kActivation);
    protocol::Hello hello;
    hello.role = protocol::NodeRole::kNode;
    hello.channel = Channel::kActivation;
    hello.device_id = cfg.security.identity ? cfg.security.identity->fingerprint() : cfg.name;
    hello.backend_build = backend->info().build_hash;
    hello.lease = auth.peer_lease;
    CLM_RETURN_IF_ERROR(stream->send(hello));
    CLM_RETURN_IF_ERROR(stream->expect<protocol::HelloAck>(5s).status());
    // The peer never sends anything else on this stream; results go to Father.
    return stream;
  }

  // ---- release -----------------------------------------------------------------------------------------

  protocol::ReleaseComplete release(protocol::ReleaseReason reason, bool notify) {
    Stopwatch sw;
    protocol::ReleaseComplete report;
    std::vector<std::shared_ptr<MessageStream>> to_close;
    {
      std::lock_guard lock(mu);
      report.lease = lease;
      set_state(NodeState::kReleasing);
      // Stop accepting new work and provisioning first.
      if (provision) to_close.push_back(provision);
      if (father_activation) to_close.push_back(father_activation);
      if (downstream_stream) to_close.push_back(downstream_stream);
      for (auto& p : peer_inbound) to_close.push_back(p);
      provision.reset();
      father_activation.reset();
      downstream_stream.reset();
      peer_inbound.clear();
      if (downstream) cfg.security.revoke(downstream->peer_device_id);
      for (const auto& auth : inbound_allowed) cfg.security.revoke(auth.peer_device_id);
      downstream.reset();
      inbound_allowed.clear();
    }
    for (auto& s : to_close) s->close();
    phase("cleanup");
    {
      // Domain execution is bounded; waiting on exec_mu is the cooperative safe point. The supervising
      // service enforces the hard cancellation deadline by terminating this worker's job object.
      std::lock_guard exec(exec_mu);
      std::lock_guard lock(mu);
      for (auto& [stage, d] : domains) (void)d->release();
      domains.clear();
      resolver.reset();
      plan.reset();
      sealed_count = 0;
      report.resources_released = true;
      auto rr = store->release();
      report.storage_cleaned = rr.storage_cleaned;
      report.residual_bytes = rr.residual_bytes;
      report.errors = rr.errors;
      lease = lease.next();
      counters.lease_generation = lease.value;
      ++counters.leases_released;
      counters.last_storage_cleaned = rr.storage_cleaned;
      report.release_ns = sw.elapsed_ns();
      counters.last_release_ns = report.release_ns;
      if (!rr.storage_cleaned) {
        set_state(NodeState::kCleanupPending);
      } else {
        const bool busy = reason == protocol::ReleaseReason::kLocalActivity || reason == protocol::ReleaseReason::kPause;
        set_state(busy ? NodeState::kBusy : NodeState::kAvailable);
      }
      log::info("lease_released", {{"lease", report.lease.str()}, {"storage_cleaned", rr.storage_cleaned ? "1" : "0"},
                                   {"residual_bytes", std::to_string(rr.residual_bytes)},
                                   {"release_us", std::to_string(report.release_ns / 1000)}});
      if (notify && control) (void)control->send(report);
    }
    return report;
  }
};

// ---- public API ------------------------------------------------------------------------------------------

NodeWorker::NodeWorker(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

Result<std::unique_ptr<NodeWorker>> NodeWorker::start(NodeConfig config) {
  auto impl = std::make_unique<Impl>();
  impl->cfg = std::move(config);
  log::info("node_starting", {{"name", impl->cfg.name}});
  if (impl->cfg.backend != "reference")
    return make_error(ErrorCode::kUnimplemented, "backend '" + impl->cfg.backend + "' is not available in this build");
  impl->backend = domain::make_reference_backend();
  if (impl->cfg.impairment) impl->egress_link = std::make_shared<transport::SimulatedLink>(*impl->cfg.impairment);

  // Orphan recovery runs before any lease can be accepted.
  lease::LeaseStoreOptions opts;
  if (impl->cfg.phase_hook) opts.crash_hook = impl->cfg.phase_hook;
  CLM_ASSIGN_OR_RETURN(impl->store, lease::LeaseStore::open(impl->cfg.staging_root, opts));

  CLM_ASSIGN_OR_RETURN(impl->listener, transport::listen(impl->cfg.listen, impl->cfg.security));
  // Lease generations strictly increase across restarts (the journal remembers the highest one), so a
  // message naming any earlier lease is stale even after a crash.
  impl->lease = LeaseGeneration{impl->store->last_generation() + 1};
  impl->counters.lease_generation = impl->lease.value;
  impl->set_state(impl->cfg.start_busy ? NodeState::kBusy : NodeState::kAvailable);
  Impl* raw = impl.get();
  impl->accept_thread = std::thread([raw] { raw->accept_loop(); });
  return std::unique_ptr<NodeWorker>(new NodeWorker(std::move(impl)));
}

NodeWorker::~NodeWorker() { stop(); }

transport::Endpoint NodeWorker::endpoint() const { return impl_->listener->local_endpoint(); }

std::string NodeWorker::device_id() const {
  return impl_->cfg.security.identity ? impl_->cfg.security.identity->fingerprint() : impl_->cfg.name;
}

NodeStatus NodeWorker::status() const {
  std::lock_guard lock(impl_->mu);
  NodeStatus s = impl_->counters;
  s.state = impl_->state;
  auto census = platform::allocated_bytes_under(impl_->store->root());
  s.staging_census_bytes = census.is_ok() ? census.value() : ~std::uint64_t{0};
  return s;
}

void NodeWorker::on_local_activity() {
  NodeState s;
  {
    std::lock_guard lock(impl_->mu);
    s = impl_->state;
  }
  log::info("local_activity");
  if (s == NodeState::kPreparing || s == NodeState::kReady || s == NodeState::kInferencing) {
    impl_->release(protocol::ReleaseReason::kLocalActivity, /*notify=*/true);
  } else {
    std::lock_guard lock(impl_->mu);
    if (s == NodeState::kAvailable) impl_->set_state(NodeState::kBusy);
  }
}

void NodeWorker::on_local_idle() {
  std::lock_guard lock(impl_->mu);
  if (impl_->state == NodeState::kCleanupPending) {
    auto rr = impl_->store->retry_cleanup();
    if (!rr.storage_cleaned) return;
  } else if (impl_->state != NodeState::kBusy) {
    return;
  }
  impl_->set_state(NodeState::kAvailable);
  if (impl_->control) (void)impl_->control->send(impl_->offer_locked());
}

void NodeWorker::stop() {
  if (!impl_ || impl_->stopping.exchange(true)) return;
  impl_->listener->close();
  if (impl_->accept_thread.joinable()) impl_->accept_thread.join();
  std::vector<std::shared_ptr<MessageStream>> streams;
  {
    std::lock_guard lock(impl_->mu);
    for (auto* s : {&impl_->control, &impl_->father_activation, &impl_->provision, &impl_->downstream_stream})
      if (*s) streams.push_back(*s);
    for (auto& p : impl_->peer_inbound) streams.push_back(p);
  }
  for (auto& s : streams) s->close();
  std::vector<std::thread> handlers;
  {
    std::lock_guard lock(impl_->mu);
    handlers.swap(impl_->handlers);
  }
  for (auto& t : handlers)
    if (t.joinable()) t.join();
  bool leased;
  {
    std::lock_guard lock(impl_->mu);
    leased = impl_->plan.has_value() || impl_->state == NodeState::kReady || impl_->state == NodeState::kPreparing;
  }
  if (leased) impl_->release(protocol::ReleaseReason::kShutdown, /*notify=*/false);
}

}  // namespace clusterlm::node
