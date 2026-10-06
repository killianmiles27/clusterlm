#include "clusterlm/coordinator/coordinator.hpp"

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <map>
#include <mutex>
#include <sstream>
#include <thread>

#include "clusterlm/common/clock.hpp"
#include "clusterlm/common/log.hpp"
#include "clusterlm/domain/backend_adapter.hpp"
#include "clusterlm/domain/drafter.hpp"
#include "clusterlm/objects/canonical_store.hpp"
#include "clusterlm/protocol/wire.hpp"

namespace clusterlm::coordinator {

using namespace std::chrono_literals;
using protocol::Channel;
using protocol::Message;
using protocol::MessageStream;
using protocol::ReceivedMessage;

std::int32_t argmax(std::span<const float> logits) {
  std::size_t best = 0;
  for (std::size_t i = 1; i < logits.size(); ++i)
    if (logits[i] > logits[best]) best = i;
  return static_cast<std::int32_t>(best);
}

// ---- ClusterPlan -----------------------------------------------------------------------------------------

Result<ClusterPlan> ClusterPlan::parse(std::string_view text, std::uint32_t n_layers) {
  ClusterPlan plan;
  std::stringstream ss{std::string(text)};
  std::string item;
  std::vector<std::pair<objects::LayerRange, int>> parts;
  while (std::getline(ss, item, ',')) {
    const auto at = item.find('@');
    const auto dash = item.find('-');
    if (at == std::string::npos || dash == std::string::npos || dash > at)
      return make_error(ErrorCode::kInvalidArgument, "plan item '" + item + "' is not BEGIN-END@OWNER");
    objects::LayerRange range;
    try {
      range.begin = static_cast<std::uint32_t>(std::stoul(item.substr(0, dash)));
      range.end = static_cast<std::uint32_t>(std::stoul(item.substr(dash + 1, at - dash - 1)));
    } catch (...) {
      return make_error(ErrorCode::kInvalidArgument, "bad layer range in '" + item + "'");
    }
    const std::string owner = item.substr(at + 1);
    int domain = kFatherDomain;
    if (owner != "father") {
      try {
        domain = std::stoi(owner);
      } catch (...) {
        return make_error(ErrorCode::kInvalidArgument, "bad owner in '" + item + "'");
      }
    }
    parts.emplace_back(range, domain);
  }
  if (parts.size() < 2) return make_error(ErrorCode::kInvalidArgument, "a plan needs at least a prefix and a tail");
  for (std::size_t i = 0; i < parts.size(); ++i) {
    StagePlan sp;
    sp.stage = StageId{static_cast<std::uint32_t>(i)};
    sp.layers = parts[i].first;
    sp.domain = parts[i].second;
    sp.role = i == 0 ? domain::StageRole::kPrefix
                     : (i + 1 == parts.size() ? domain::StageRole::kTail : domain::StageRole::kMiddle);
    plan.stages.push_back(sp);
  }
  if (plan.stages.back().layers.end != n_layers)
    return make_error(ErrorCode::kInvalidArgument, "plan must end at layer " + std::to_string(n_layers));
  return plan;
}

Status ClusterPlan::validate(const objects::ModelGeometry& g, std::size_t node_count) const {
  if (stages.size() < 2) return make_error(ErrorCode::kInvalidArgument, "plan needs prefix and tail");
  std::uint32_t expect = 0;
  for (std::size_t i = 0; i < stages.size(); ++i) {
    const auto& s = stages[i];
    if (s.layers.begin != expect || s.layers.empty())
      return make_error(ErrorCode::kInvalidArgument, "stage ranges must be contiguous and non-empty");
    expect = s.layers.end;
    const bool father = s.domain == kFatherDomain;
    if ((s.role == domain::StageRole::kPrefix || s.role == domain::StageRole::kTail) && !father)
      return make_error(ErrorCode::kInvalidArgument, "prefix and tail stages must run on Father");
    if (s.role == domain::StageRole::kMiddle && father)
      return make_error(ErrorCode::kInvalidArgument, "middle stages run on Nodes (merge Father ranges instead)");
    if (!father && (s.domain < 0 || static_cast<std::size_t>(s.domain) >= node_count))
      return make_error(ErrorCode::kInvalidArgument, "stage names an unknown Node index");
    if (s.role != domain::StageRole::kPrefix && s.layers.contains(g.ple_layer))
      return make_error(ErrorCode::kInvalidArgument, "the PLE layer must stay in Father's prefix");
  }
  if (expect != g.n_layers) return make_error(ErrorCode::kInvalidArgument, "plan does not cover every layer");
  if (stages.front().layers.begin != 0 || !stages.front().layers.contains(g.ple_layer))
    return make_error(ErrorCode::kInvalidArgument, "prefix must start at layer 0 and contain the PLE layer");
  std::vector<int> seen;
  for (const auto& s : stages) {
    if (s.domain == kFatherDomain) continue;
    if (std::find(seen.begin(), seen.end(), s.domain) != seen.end())
      return make_error(ErrorCode::kInvalidArgument, "a Node owns at most one contiguous stage in this version");
    seen.push_back(s.domain);
  }
  return Status::ok();
}

Digest256 ClusterPlan::hash(const Digest256& model_root) const {
  ByteWriter w;
  w.str("clusterlm-plan-v1");
  w.raw(model_root.bytes);
  w.u32(max_context);
  w.u32(max_window);
  w.u32(static_cast<std::uint32_t>(stages.size()));
  for (const auto& s : stages) {
    w.u32(s.stage.value);
    w.u8(static_cast<std::uint8_t>(s.role));
    w.u32(s.layers.begin);
    w.u32(s.layers.end);
    w.i32(s.domain);
  }
  auto sorted = targets;
  std::sort(sorted.begin(), sorted.end());
  w.u32(static_cast<std::uint32_t>(sorted.size()));
  for (const auto& [name, target] : sorted) {
    w.str(name);
    w.u8(static_cast<std::uint8_t>(target));
  }
  return Sha256::of(w.bytes());
}

std::string ClusterPlan::describe() const {
  std::string out;
  for (const auto& s : stages) {
    if (!out.empty()) out += " -> ";
    out += std::string(domain::to_string(s.role)) + "[" + std::to_string(s.layers.begin) + "," +
           std::to_string(s.layers.end) + ")@" + (s.domain == kFatherDomain ? "father" : "node" + std::to_string(s.domain));
  }
  return out;
}

// ---- RemoteNode --------------------------------------------------------------------------------------------

namespace {

// Thread-safe inbox fed by one stream's reader thread. Replies are matched by correlation id; messages with
// correlation 0 are unsolicited events (offers, lease release notifications, PlanReady).
class Inbox {
 public:
  void push(ReceivedMessage m) {
    std::lock_guard lock(mu_);
    items_.push_back(std::move(m));
    cv_.notify_all();
  }
  void close(Status why) {
    std::lock_guard lock(mu_);
    closed_ = std::move(why);
    cv_.notify_all();
  }
  // Wait for the first message satisfying `pred`.
  template <typename Pred>
  Result<ReceivedMessage> wait(Pred pred, std::chrono::milliseconds timeout) {
    std::unique_lock lock(mu_);
    const auto deadline = SteadyClock::now() + timeout;
    while (true) {
      for (auto it = items_.begin(); it != items_.end(); ++it) {
        if (pred(*it)) {
          ReceivedMessage m = std::move(*it);
          items_.erase(it);
          return m;
        }
      }
      if (closed_) return *closed_;
      if (cv_.wait_until(lock, deadline) == std::cv_status::timeout)
        return make_error(ErrorCode::kDeadlineExceeded, "timed out waiting for reply");
    }
  }

 private:
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<ReceivedMessage> items_;
  std::optional<Status> closed_;
};

struct StreamWithInbox {
  std::shared_ptr<MessageStream> stream;
  Inbox inbox;
  std::thread reader;
  std::atomic<bool> stop{false};

  void start() {
    reader = std::thread([this] {
      while (!stop.load()) {
        auto r = stream->receive(100ms);
        if (!r.is_ok()) {
          if (r.status().code() == ErrorCode::kDeadlineExceeded) continue;
          inbox.close(r.status());
          return;
        }
        inbox.push(std::move(r).value());
      }
      inbox.close(make_error(ErrorCode::kUnavailable, "stream closed"));
    });
  }
  void shutdown() {
    stop.store(true);
    if (stream) stream->close();
    if (reader.joinable()) reader.join();
  }
  ~StreamWithInbox() { shutdown(); }

  // Send a request and wait for the reply with the same correlation id.
  template <typename Reply>
  Result<Reply> call(const Message& m, std::chrono::milliseconds timeout) {
    const auto corr = stream->next_correlation();
    CLM_RETURN_IF_ERROR(stream->send(m, corr));
    CLM_ASSIGN_OR_RETURN(auto reply, inbox.wait([corr](const ReceivedMessage& r) { return r.correlation == corr; },
                                                 timeout));
    if (auto* v = std::get_if<Reply>(&reply.message)) return std::move(*v);
    if (auto* e = std::get_if<protocol::ErrorMessage>(&reply.message)) return make_error(e->code, e->message);
    return make_error(ErrorCode::kProtocolError, "unexpected reply " +
                                                     std::string(protocol::to_string(protocol::type_of(reply.message))));
  }
};

}  // namespace

struct RemoteNode {
  NodeEndpoint endpoint;
  std::string device_id;  // as reported by the Node (fingerprint, or name in insecure mode)
  LeaseGeneration lease{0};
  protocol::OfferResources offer;
  std::unique_ptr<StreamWithInbox> control;
  std::unique_ptr<StreamWithInbox> activation;
  std::unique_ptr<StreamWithInbox> provision;
  std::optional<StagePlan> stage;
  NodeProvisionReport provision_report;
};

struct Coordinator::Impl {
  CoordinatorConfig cfg;
  std::unique_ptr<objects::CanonicalModelStore> store;
  std::unique_ptr<domain::BackendAdapter> backend;
  std::shared_ptr<transport::SimulatedLink> egress_link;  // Father's single NIC, shared by all Node links
  std::vector<std::unique_ptr<RemoteNode>> nodes;
  std::optional<ClusterPlan> plan;
  Digest256 plan_hash;
  std::map<std::uint32_t, std::unique_ptr<domain::ExecutionDomain>> local;  // Father prefix/tail by stage id
  Epoch epoch{0};
  SessionId next_session{1};
  bool prepared = false;

  std::unique_ptr<transport::Connection> wrap(std::unique_ptr<transport::Connection> c) {
    if (!cfg.impairment && !cfg.faults) return c;
    return transport::impair(std::move(c), cfg.impairment.value_or(transport::NetworkConditions{}), egress_link,
                             cfg.faults);
  }

  Result<std::unique_ptr<StreamWithInbox>> open_channel(RemoteNode& n, Channel channel, bool start_reader) {
    std::optional<std::string> expected;
    if (!n.endpoint.device_id.empty()) expected = n.endpoint.device_id;
    CLM_ASSIGN_OR_RETURN(auto conn, transport::connect(n.endpoint.endpoint, cfg.security, expected, 5s));
    auto s = std::make_unique<StreamWithInbox>();
    s->stream = std::make_shared<MessageStream>(wrap(std::move(conn)), channel);
    protocol::Hello hello;
    hello.role = protocol::NodeRole::kFather;
    hello.channel = channel;
    hello.device_id = cfg.security.identity ? cfg.security.identity->fingerprint() : "father";
    hello.backend_build = backend->info().build_hash;
    hello.lease = n.lease;
    CLM_RETURN_IF_ERROR(s->stream->send(hello));
    CLM_ASSIGN_OR_RETURN(auto ack, s->stream->expect<protocol::HelloAck>(cfg.request_timeout));
    if (ack.protocol_version != protocol::kProtocolVersion)
      return make_error(ErrorCode::kVersionMismatch, "Node speaks protocol " + std::to_string(ack.protocol_version));
    n.device_id = ack.device_id;
    n.lease = ack.lease;
    if (start_reader) s->start();
    return s;
  }

  RemoteNode& node_for(const StagePlan& s) { return *nodes.at(static_cast<std::size_t>(s.domain)); }

  std::vector<const StagePlan*> remote_stages() const {
    std::vector<const StagePlan*> out;
    for (const auto& s : plan->stages)
      if (s.domain != kFatherDomain) out.push_back(&s);
    return out;
  }

  objects::AllocationTarget target_for(const std::string& name) const {
    for (const auto& [n, t] : plan->targets)
      if (n == name) return t;
    return objects::AllocationTarget::kCpuResident;
  }

  // ---- preparation -------------------------------------------------------------------------------------

  Result<protocol::PreparePlan> plan_message(RemoteNode& n, const StagePlan& s) {
    const auto& m = store->manifest();
    protocol::PreparePlan p;
    p.lease = n.lease;
    p.model_root = m.root_hash();
    p.backend_build = backend->info().build_hash;
    p.plan_hash = plan_hash;
    p.stages.push_back({s.stage, s.role, s.layers, plan->max_context, plan->max_window});
    p.ram_cap_bytes = n.offer.safe_ram_bytes;
    p.vram_cap_bytes = n.offer.safe_vram_bytes;
    p.staging_cap_bytes = n.offer.staging_disk_bytes;
    // Plan-scoped manifest: geometry, shard identities and only this Node's layer objects. No embedding,
    // lookup, head, MTP, tokenizer or unrelated layers.
    p.manifest.artifact_id = m.artifact_id;
    p.manifest.license = m.license;
    p.manifest.geometry = m.geometry;
    p.manifest.shards = m.shards;
    for (const auto* obj : m.layer_objects(s.layers)) {
      p.assignments.push_back({static_cast<std::uint32_t>(p.manifest.objects.size()), target_for(obj->name)});
      p.manifest.objects.push_back(*obj);
    }
    return p;
  }

  // Stream every assigned object over the provision channel in bounded, individually digested chunks.
  Status provision_node(RemoteNode& n, const protocol::PreparePlan& p) {
    for (const auto& a : p.assignments) {
      const auto& obj = p.manifest.objects[a.object_index];
      CLM_ASSIGN_OR_RETURN(Bytes bytes, store->read_object_bytes(obj.name));
      for (std::uint64_t off = 0; off < bytes.size(); off += cfg.provision_chunk_bytes) {
        const auto len = std::min<std::uint64_t>(cfg.provision_chunk_bytes, bytes.size() - off);
        protocol::ProvisionChunk chunk;
        chunk.lease = n.lease;
        chunk.object_index = a.object_index;
        chunk.offset = off;
        chunk.data.assign(bytes.begin() + static_cast<std::ptrdiff_t>(off),
                          bytes.begin() + static_cast<std::ptrdiff_t>(off + len));
        chunk.chunk_digest = Sha256::of(chunk.data);
        CLM_RETURN_IF_ERROR(n.provision->stream->send(chunk));
      }
      protocol::SealObject seal{n.lease, a.object_index, obj.byte_size, obj.object_digest};
      CLM_RETURN_IF_ERROR(n.provision->stream->send(seal, n.provision->stream->next_correlation()));
      n.provision_report.bytes += bytes.size();
      ++n.provision_report.objects;
    }
    // Every seal must be acknowledged; any chunk/seal error arrives as an ErrorMessage.
    for (std::size_t i = 0; i < p.assignments.size(); ++i) {
      CLM_ASSIGN_OR_RETURN(auto reply, n.provision->inbox.wait([](const ReceivedMessage&) { return true; },
                                                               cfg.prepare_timeout));
      if (auto* e = std::get_if<protocol::ErrorMessage>(&reply.message)) return make_error(e->code, e->message);
      if (!std::holds_alternative<protocol::ObjectSealed>(reply.message))
        return make_error(ErrorCode::kProtocolError, "unexpected message on provision channel");
    }
    return Status::ok();
  }

  Status prepare_node(RemoteNode& n, const StagePlan& s) {
    Stopwatch sw;
    n.stage = s;
    n.provision_report = {};
    n.provision_report.node = n.endpoint.name;
    CLM_ASSIGN_OR_RETURN(auto p, plan_message(n, s));
    CLM_RETURN_IF_ERROR(n.control->call<protocol::PlanAccepted>(p, cfg.request_timeout).status());
    CLM_ASSIGN_OR_RETURN(n.provision, open_channel(n, Channel::kProvision, true));
    CLM_RETURN_IF_ERROR(provision_node(n, p));
    // PlanReady (or a prepare failure) arrives unsolicited on the control channel.
    CLM_ASSIGN_OR_RETURN(auto ready, n.control->inbox.wait(
                                         [](const ReceivedMessage& r) {
                                           return std::holds_alternative<protocol::PlanReady>(r.message) ||
                                                  (r.correlation == 0 &&
                                                   std::holds_alternative<protocol::ErrorMessage>(r.message));
                                         },
                                         cfg.prepare_timeout));
    if (auto* e = std::get_if<protocol::ErrorMessage>(&ready.message)) return make_error(e->code, e->message);
    const auto& pr = std::get<protocol::PlanReady>(ready.message);
    if (pr.plan_hash != plan_hash || pr.lease != n.lease)
      return make_error(ErrorCode::kStaleEpoch, "PlanReady for a different plan or lease");
    n.provision_report.node_prepare_ns = pr.prepare_ns;
    // Provisioning is over; the bulk channel is not kept open during inference.
    n.provision->shutdown();
    n.provision.reset();
    CLM_ASSIGN_OR_RETURN(n.activation, open_channel(n, Channel::kActivation, true));
    n.provision_report.prepare_ms = sw.elapsed_ms();
    return Status::ok();
  }

  Status authorize_peers() {
    const auto remotes = remote_stages();
    for (std::size_t i = 0; i + 1 < remotes.size(); ++i) {
      auto& from = node_for(*remotes[i]);
      auto& to = node_for(*remotes[i + 1]);
      protocol::AuthorizePeer out;  // to the upstream Node: forward to `to`
      out.lease = from.lease;
      out.plan_hash = plan_hash;
      out.from_stage = remotes[i]->stage;
      out.to_stage = remotes[i + 1]->stage;
      out.peer_device_id = to.device_id;
      out.peer_endpoint = to.endpoint.endpoint.str();
      out.peer_lease = to.lease;
      protocol::AuthorizePeer in = out;  // to the downstream Node: accept `from`
      in.lease = to.lease;
      in.peer_device_id = from.device_id;
      in.peer_endpoint.clear();
      in.peer_lease = from.lease;
      CLM_RETURN_IF_ERROR(to.control->call<protocol::Pong>(in, cfg.request_timeout).status());
      CLM_RETURN_IF_ERROR(from.control->call<protocol::Pong>(out, cfg.request_timeout).status());
    }
    return Status::ok();
  }

  // ---- execution -------------------------------------------------------------------------------------

  void abort_everywhere(SessionId session, const std::string& reason) {
    for (auto& [id, d] : local) (void)d->abort_session(epoch, session);
    for (const auto* s : remote_stages()) {
      auto& n = node_for(*s);
      if (n.control) (void)n.control->call<protocol::SessionOpened>(protocol::AbortSession{epoch, session, reason}, 2s);
    }
    log::warn("session_aborted", {{"epoch", epoch.str()}, {"session", session.str()}, {"reason", reason}});
    epoch = epoch.next();
  }

  // Run one window through every stage. Returns the tail logits.
  Result<domain::Logits> run_round(const domain::WindowRequest& req, std::span<const std::int32_t> tokens,
                                   RoundTrace& trace) {
    const auto& stages = plan->stages;
    Stopwatch sw;
    CLM_ASSIGN_OR_RETURN(auto acts, local.at(stages.front().stage.value)->run_prefix(req, tokens));
    trace.prefix_ms = sw.elapsed_ms();

    const auto remotes = remote_stages();
    sw.reset();
    if (!remotes.empty()) {
      const bool chain = cfg.direct_peer && remotes.size() > 1;
      for (std::size_t i = 0; i < remotes.size(); ++i) {
        auto& n = node_for(*remotes[i]);
        protocol::RunWindow run;
        run.lease = n.lease;
        run.request = req;
        run.stage = remotes[i]->stage;
        run.forward_to_peer = chain;
        run.activations = std::move(acts);
        const auto corr = n.activation->stream->next_correlation();
        const auto payload = protocol::encode(run).size();
        CLM_RETURN_IF_ERROR(n.activation->stream->send(run, corr));
        ++trace.boundary_messages;
        trace.boundary_payload_bytes += payload;
        // In chain mode the result comes back from the LAST Node; otherwise from this one.
        auto& reply_node = chain ? node_for(*remotes.back()) : n;
        CLM_ASSIGN_OR_RETURN(auto reply, reply_node.activation->inbox.wait(
                                             [&](const ReceivedMessage& r) {
                                               auto* sr = std::get_if<protocol::StageResult>(&r.message);
                                               return sr != nullptr && sr->window == req.window &&
                                                      sr->session == req.session;
                                             },
                                             cfg.window_timeout));
        auto& result = std::get<protocol::StageResult>(reply.message);
        ++trace.boundary_messages;
        trace.boundary_payload_bytes += reply.wire_bytes;
        if (result.status != ErrorCode::kOk)
          return make_error(result.status, "stage " + result.stage.str() + ": " + result.error_message);
        if (result.epoch != req.epoch) return make_error(ErrorCode::kStaleEpoch, "StageResult from another epoch");
        trace.remote_timings = result.timings;
        acts = std::move(result.activations);
        if (chain) break;
      }
    }
    trace.remote_ms = sw.elapsed_ms();
    sw.reset();
    CLM_ASSIGN_OR_RETURN(auto logits, local.at(stages.back().stage.value)->run_tail(req, acts));
    trace.tail_ms = sw.elapsed_ms();
    return logits;
  }

  Status commit_round(const domain::WindowRequest& req, std::uint32_t accepted, RoundTrace& trace) {
    Stopwatch sw;
    domain::CommitRequest c{req.epoch, req.session, req.window, accepted, req.expected_state};
    // Send every remote commit first so Nodes commit in parallel, then commit locally, then collect acks.
    std::vector<std::pair<RemoteNode*, std::uint64_t>> pending;
    for (const auto* s : remote_stages()) {
      auto& n = node_for(*s);
      const auto corr = n.control->stream->next_correlation();
      CLM_RETURN_IF_ERROR(n.control->stream->send(protocol::CommitWindow{c, s->stage}, corr));
      ++trace.control_messages;
      pending.emplace_back(&n, corr);
    }
    std::optional<domain::CommitAck> reference;
    for (auto& [id, d] : local) {
      CLM_ASSIGN_OR_RETURN(auto ack, d->commit_window(c));
      if (!reference) reference = ack;
    }
    for (auto& [n, corr] : pending) {
      const auto want = corr;
      CLM_ASSIGN_OR_RETURN(auto reply, n->control->inbox.wait(
                                           [want](const ReceivedMessage& r) { return r.correlation == want; },
                                           cfg.request_timeout));
      ++trace.control_messages;
      if (auto* e = std::get_if<protocol::ErrorMessage>(&reply.message)) return make_error(e->code, e->message);
      auto* ack = std::get_if<protocol::CommitAckMessage>(&reply.message);
      if (ack == nullptr) return make_error(ErrorCode::kProtocolError, "expected CommitAck");
      // Every domain must agree on the committed position and state version.
      if (reference && (ack->ack.committed_position != reference->committed_position ||
                        ack->ack.state != reference->state))
        return make_error(ErrorCode::kAborted, "domains disagree on committed state");
    }
    trace.commit_ms = sw.elapsed_ms();
    return Status::ok();
  }
};

// ---- Coordinator ---------------------------------------------------------------------------------------

Coordinator::Coordinator(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

Coordinator::~Coordinator() {
  if (impl_ && impl_->prepared) (void)release();
}

Result<std::unique_ptr<Coordinator>> Coordinator::create(CoordinatorConfig config) {
  auto impl = std::make_unique<Impl>();
  impl->cfg = std::move(config);
  CLM_ASSIGN_OR_RETURN(impl->store, objects::CanonicalModelStore::open(impl->cfg.model_dir));
  impl->backend = domain::make_reference_backend();
  if (impl->cfg.impairment) impl->egress_link = std::make_shared<transport::SimulatedLink>(*impl->cfg.impairment);
  for (const auto& ep : impl->cfg.nodes) {
    auto n = std::make_unique<RemoteNode>();
    n->endpoint = ep;
    impl->nodes.push_back(std::move(n));
  }
  return std::unique_ptr<Coordinator>(new Coordinator(std::move(impl)));
}

const objects::ModelManifest& Coordinator::manifest() const { return impl_->store->manifest(); }

Status Coordinator::connect() {
  for (auto& n : impl_->nodes) {
    if (n->control) continue;
    CLM_ASSIGN_OR_RETURN(n->control, impl_->open_channel(*n, Channel::kControl, true));
    // An Available Node offers resources immediately; a Busy Node offers later.
    CLM_ASSIGN_OR_RETURN(auto offer, n->control->inbox.wait(
                                         [](const ReceivedMessage& r) {
                                           return std::holds_alternative<protocol::OfferResources>(r.message);
                                         },
                                         impl_->cfg.request_timeout));
    n->offer = std::get<protocol::OfferResources>(offer.message);
    n->lease = n->offer.lease;
    log::info("node_connected", {{"node", n->endpoint.name}, {"lease", n->lease.str()},
                                 {"safe_ram", std::to_string(n->offer.safe_ram_bytes)}});
  }
  return Status::ok();
}

Result<PrepareReport> Coordinator::prepare(const ClusterPlan& plan) {
  auto& im = *impl_;
  const auto& m = im.store->manifest();
  CLM_RETURN_IF_ERROR(plan.validate(m.geometry, im.nodes.size()));
  if (im.prepared) return make_error(ErrorCode::kFailedPrecondition, "release the current plan first");
  Stopwatch sw;
  im.plan = plan;
  im.plan_hash = plan.hash(m.root_hash());
  log::info("prepare_plan", {{"plan", plan.describe()}, {"plan_hash", im.plan_hash.hex().substr(0, 16)}});

  // Father-local prefix and tail domains, bound to the canonical store (which loads only their objects).
  for (const auto& s : plan.stages) {
    if (s.domain != kFatherDomain) continue;
    domain::DomainSpec spec{s.stage, s.role, s.layers, plan.max_context, plan.max_window, 1};
    CLM_ASSIGN_OR_RETURN(auto d, im.backend->create_domain(m, spec));
    CLM_RETURN_IF_ERROR(d->prepare(*im.store));
    im.local.emplace(s.stage.value, std::move(d));
  }

  // Provision Nodes concurrently. Their transfers share Father's single egress link, so concurrency overlaps
  // per-Node hashing/loading rather than multiplying bandwidth.
  std::vector<Status> results(plan.stages.size());
  std::vector<std::thread> workers;
  for (std::size_t i = 0; i < plan.stages.size(); ++i) {
    const auto& s = plan.stages[i];
    if (s.domain == kFatherDomain) continue;
    workers.emplace_back([&, i, s] { results[i] = im.prepare_node(im.node_for(s), s); });
  }
  for (auto& t : workers) t.join();
  for (const auto& st : results) {
    if (!st.is_ok()) {
      log::error("prepare_failed", {{"error", st.to_string()}});
      im.prepared = true;  // so release() cleans up partial state
      (void)release();
      return st;
    }
  }
  if (im.cfg.direct_peer) CLM_RETURN_IF_ERROR(im.authorize_peers());
  im.prepared = true;

  PrepareReport report;
  report.plan_hash = im.plan_hash;
  for (const auto* s : im.remote_stages()) report.nodes.push_back(im.node_for(*s).provision_report);
  report.father_resident_bytes = im.store->loaded_bytes();
  report.total_ms = sw.elapsed_ms();
  return report;
}

bool Coordinator::ready() const { return impl_->prepared; }

Result<GenerationResult> Coordinator::generate(const GenerationRequest& request) {
  auto& im = *impl_;
  if (!im.prepared) return make_error(ErrorCode::kFailedPrecondition, "no prepared plan");
  if (request.prompt.empty()) return make_error(ErrorCode::kInvalidArgument, "empty prompt");
  if (request.q == 0 || request.q > im.plan->max_window)
    return make_error(ErrorCode::kInvalidArgument, "q must be in [1, max_window]");
  if (request.q > 1 && !request.drafter) return make_error(ErrorCode::kInvalidArgument, "q > 1 requires a drafter");
  const std::uint32_t chunk = std::min(request.prefill_chunk, im.plan->max_window);
  const auto vocab = im.store->manifest().geometry.vocab_size;

  GenerationResult out;
  im.epoch = im.epoch.next();
  out.epoch = im.epoch;
  const SessionId session = im.next_session;
  im.next_session = im.next_session.next();

  // Open the session on every domain.
  for (auto& [id, d] : im.local) CLM_RETURN_IF_ERROR(d->open_session(im.epoch, session));
  for (const auto* s : im.remote_stages()) {
    auto st = im.node_for(*s).control->call<protocol::SessionOpened>(protocol::OpenSession{im.epoch, session},
                                                                     im.cfg.request_timeout);
    if (!st.is_ok()) {
      im.abort_everywhere(session, "open failed");
      return st.status();
    }
  }

  std::uint64_t position = 0;
  StateVersion state{0};
  WindowId window{0};
  std::vector<std::int32_t> committed;  // every token whose state is committed (Father-local history)
  std::int32_t next_token = -1;
  Stopwatch total;

  auto fail = [&](const Status& st) -> Status {
    im.abort_everywhere(session, st.message());
    return make_error(st.code() == ErrorCode::kOk ? ErrorCode::kAborted : st.code(),
                      "distributed session invalidated: " + st.to_string());
  };

  // Prefill in bounded chunks.
  for (std::size_t off = 0; off < request.prompt.size(); off += chunk) {
    const auto n = static_cast<std::uint32_t>(std::min<std::size_t>(chunk, request.prompt.size() - off));
    window = window.next();
    domain::WindowRequest req{im.epoch, session, window, position, state, n};
    RoundTrace trace;
    trace.prefill = true;
    trace.positions = n;
    Stopwatch rsw;
    auto logits = im.run_round(req, std::span(request.prompt).subspan(off, n), trace);
    if (!logits.is_ok()) return fail(logits.status());
    if (auto st = im.commit_round(req, n, trace); !st.is_ok()) return fail(st);
    trace.accepted = n;
    trace.total_ms = rsw.elapsed_ms();
    position += n;
    state = state.next();
    committed.insert(committed.end(), request.prompt.begin() + static_cast<std::ptrdiff_t>(off),
                     request.prompt.begin() + static_cast<std::ptrdiff_t>(off + n));
    if (off + n == request.prompt.size())
      next_token = argmax(std::span<const float>(logits->data).subspan(std::size_t{n - 1} * vocab, vocab));
    out.rounds.push_back(std::move(trace));
  }
  out.prefill_ms = total.elapsed_ms();
  out.tokens.push_back(next_token);
  out.first_token_ms = out.prefill_ms;

  // Decode: speculative windows of q positions [next_token, d1..d_{q-1}].
  Stopwatch decode;
  while (out.tokens.size() < request.max_new_tokens) {
    const std::uint32_t q = request.q;
    RoundTrace trace;
    Stopwatch rsw;
    std::vector<std::int32_t> window_tokens{next_token};
    if (q > 1) {
      Stopwatch dsw;
      auto drafts = request.drafter->draft(committed, next_token, q - 1);
      trace.draft_ms = dsw.elapsed_ms();
      if (drafts.size() != q - 1) return fail(make_error(ErrorCode::kInternal, "drafter returned wrong count"));
      window_tokens.insert(window_tokens.end(), drafts.begin(), drafts.end());
    }
    window = window.next();
    domain::WindowRequest req{im.epoch, session, window, position, state, q};
    trace.positions = q;
    auto logits = im.run_round(req, window_tokens, trace);
    if (!logits.is_ok()) return fail(logits.status());
    // Accept the longest draft prefix the target model agrees with (greedy verification).
    std::uint32_t accepted = 1;
    while (accepted < q &&
           argmax(std::span<const float>(logits->data).subspan(std::size_t{accepted - 1} * vocab, vocab)) ==
               window_tokens[accepted])
      ++accepted;
    if (auto st = im.commit_round(req, accepted, trace); !st.is_ok()) return fail(st);
    position += accepted;
    state = state.next();
    committed.insert(committed.end(), window_tokens.begin(), window_tokens.begin() + accepted);
    for (std::uint32_t i = 1; i < accepted; ++i) out.tokens.push_back(window_tokens[i]);
    next_token = argmax(std::span<const float>(logits->data).subspan(std::size_t{accepted - 1} * vocab, vocab));
    out.tokens.push_back(next_token);
    trace.accepted = accepted;
    trace.total_ms = rsw.elapsed_ms();
    out.rounds.push_back(std::move(trace));
    ++out.decode_rounds;
  }
  if (out.tokens.size() > request.max_new_tokens) out.tokens.resize(request.max_new_tokens);
  out.decode_ms = decode.elapsed_ms();
  out.decode_tokens = static_cast<std::uint32_t>(out.tokens.size() - 1);

  // The session is complete; free its sequence state everywhere (the lease and weights stay Ready).
  for (auto& [id, d] : im.local) (void)d->abort_session(im.epoch, session);
  for (const auto* s : im.remote_stages())
    (void)im.node_for(*s).control->call<protocol::SessionOpened>(
        protocol::AbortSession{im.epoch, session, "complete"}, im.cfg.request_timeout);
  return out;
}

Result<ReleaseReport> Coordinator::release() {
  auto& im = *impl_;
  ReleaseReport report;
  for (auto& n : im.nodes) {
    if (!n->control) continue;
    Stopwatch sw;
    if (n->activation) n->activation->shutdown();
    n->activation.reset();
    if (n->provision) n->provision->shutdown();
    n->provision.reset();
    auto rc = n->control->call<protocol::ReleaseComplete>(
        protocol::ReleaseLease{n->lease, protocol::ReleaseReason::kFatherRequest}, im.cfg.request_timeout);
    ReleaseReport::NodeRelease nr;
    nr.node = n->endpoint.name;
    nr.release_ms = sw.elapsed_ms();
    if (rc.is_ok()) {
      nr.resources_released = rc->resources_released;
      nr.storage_cleaned = rc->storage_cleaned;
      nr.residual_bytes = rc->residual_bytes;
      nr.errors = rc->errors;
    } else {
      nr.errors = rc.status().to_string();
    }
    // The Node re-offers under its next lease generation.
    auto offer = n->control->inbox.wait(
        [](const ReceivedMessage& r) { return std::holds_alternative<protocol::OfferResources>(r.message); }, 2s);
    if (offer.is_ok()) {
      n->offer = std::get<protocol::OfferResources>(offer->message);
      n->lease = n->offer.lease;
    }
    n->stage.reset();
    report.nodes.push_back(std::move(nr));
  }
  for (auto& [id, d] : im.local) (void)d->release();
  im.local.clear();
  im.plan.reset();
  im.prepared = false;
  return report;
}

}  // namespace clusterlm::coordinator
