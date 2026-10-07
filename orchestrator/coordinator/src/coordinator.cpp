#include "clusterlm/coordinator/coordinator.hpp"

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <thread>

#include "clusterlm/common/clock.hpp"
#include "clusterlm/common/log.hpp"
#include "clusterlm/backends/backend_factory.hpp"
#include "clusterlm/domain/backend_adapter.hpp"
#include "clusterlm/domain/drafter.hpp"
#include "clusterlm/objects/canonical_store.hpp"
#include "clusterlm/protocol/wire.hpp"

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#endif

namespace clusterlm::coordinator {

using namespace std::chrono_literals;
using protocol::Channel;
using protocol::Message;
using protocol::MessageStream;
using protocol::ReceivedMessage;

std::string_view to_string(PreparePhase p) {
  switch (p) {
    case PreparePhase::kFatherDomains: return "father-domains";
    case PreparePhase::kProvisioning: return "provisioning";
    case PreparePhase::kNodePreparing: return "node-preparing";
    case PreparePhase::kNodeReady: return "node-ready";
    case PreparePhase::kAuthorizing: return "authorizing";
    case PreparePhase::kDone: return "done";
  }
  return "?";
}

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
    // Only the tail may be empty: a head-only tail follows a plan whose last layers run elsewhere.
    if (s.layers.begin != expect || (s.layers.empty() && s.role != domain::StageRole::kTail))
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
    if (m.correlation != 0 && std::find(discarded_.begin(), discarded_.end(), m.correlation) != discarded_.end())
      return;
    items_.push_back(std::move(m));
    cv_.notify_all();
  }
  void close(Status why) {
    std::lock_guard lock(mu_);
    closed_ = std::move(why);
    cv_.notify_all();
  }
  // Remember a correlation whose reply (if it ever arrives) is no longer wanted.
  void discard_correlation(std::uint64_t corr) {
    std::lock_guard lock(mu_);
    items_.erase(std::remove_if(items_.begin(), items_.end(),
                                [corr](const ReceivedMessage& m) { return m.correlation == corr; }),
                 items_.end());
    discarded_.push_back(corr);
    if (discarded_.size() > 64) discarded_.erase(discarded_.begin());
  }
  // Drop everything queued (results and loss notices of a previous plan's streams).
  void clear() {
    std::lock_guard lock(mu_);
    items_.clear();
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
  std::vector<std::uint64_t> discarded_;
  std::optional<Status> closed_;
};

// Correlation id marking a synthetic "this stream failed" message pushed into a shared inbox.
constexpr std::uint64_t kStreamLost = ~std::uint64_t{0};

struct StreamWithInbox {
  std::shared_ptr<MessageStream> stream;
  Inbox inbox;
  // When set, received messages go to this shared inbox instead (activation results from every Node feed one
  // event queue so the window pipeline can react to whichever stage finishes first). Stream failure is then
  // reported as an ErrorMessage with correlation kStreamLost instead of closing the shared inbox.
  Inbox* sink = nullptr;
  std::string name;
  std::thread reader;
  std::atomic<bool> stop{false};

  void start() {
    reader = std::thread([this] {
      while (!stop.load()) {
        auto r = stream->receive(100ms);
        if (!r.is_ok()) {
          if (r.status().code() == ErrorCode::kDeadlineExceeded) continue;
          if (stop.load()) break;
          fail(r.status());
          return;
        }
        (sink ? *sink : inbox).push(std::move(r).value());
      }
      if (!sink) inbox.close(make_error(ErrorCode::kUnavailable, "stream closed"));
    });
  }
  void fail(const Status& why) {
    if (!sink) {
      inbox.close(why);
      return;
    }
    ReceivedMessage m;
    m.message = protocol::ErrorMessage{ErrorCode::kUnavailable, name + " activation channel lost: " + why.message(),
                                       protocol::MessageType::kStageResult};
    m.correlation = kStreamLost;
    sink->push(std::move(m));
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
  // Set when the Node released its lease on its own (local activity, fault) — Father learns it from an
  // unsolicited ReleaseComplete. The plan is no longer executable on this Node.
  std::optional<protocol::ReleaseComplete> self_released;
};

// Collects per-Node progress from the (concurrent) provisioning workers and hands consistent snapshots to the sink.
class ProgressTracker {
 public:
  void begin(PrepareProgressSink sink, const std::vector<std::pair<const RemoteNode*, std::string>>& nodes) {
    std::lock_guard lk(mu_);
    sink_ = std::move(sink);
    slots_.clear();
    state_ = PrepareProgress{};
    for (const auto& [ptr, name] : nodes) {
      slots_[ptr] = state_.nodes.size();
      NodePrepareProgress np;
      np.node = name;
      state_.nodes.push_back(std::move(np));
    }
    last_emit_ = {};
  }
  void end() {
    std::lock_guard lk(mu_);
    sink_ = nullptr;
  }
  bool active() const {
    std::lock_guard lk(mu_);
    return static_cast<bool>(sink_);
  }
  void set_overall(PreparePhase p) {
    std::lock_guard lk(mu_);
    if (!sink_) return;
    state_.phase = p;
    emit_locked(true);
  }
  // `change` edits one Node's entry; `force` = a phase change or a sealed object (never throttled).
  template <typename F>
  void update(const RemoteNode* node, F&& change, bool force) {
    std::lock_guard lk(mu_);
    if (!sink_) return;
    auto it = slots_.find(node);
    if (it == slots_.end()) return;
    change(state_.nodes[it->second]);
    emit_locked(force);
  }

 private:
  void emit_locked(bool force) {
    const auto now = SteadyClock::now();
    if (!force && last_emit_ != SteadyClock::time_point{} && now - last_emit_ < std::chrono::milliseconds(50)) return;
    last_emit_ = now;
    state_.bytes_sent = state_.bytes_total = 0;
    state_.objects_sealed = state_.objects_total = 0;
    for (const auto& n : state_.nodes) {
      state_.bytes_sent += n.bytes_sent;
      state_.bytes_total += n.bytes_total;
      state_.objects_sealed += n.objects_sealed;
      state_.objects_total += n.objects_total;
    }
    sink_(state_);
  }

  mutable std::mutex mu_;
  PrepareProgressSink sink_;
  std::map<const RemoteNode*, std::size_t> slots_;
  PrepareProgress state_;
  SteadyClock::time_point last_emit_{};
};

struct Coordinator::Impl {
  // Declared first so it outlives every Node stream whose reader thread pushes into it.
  Inbox results;  // StageResults from every Node's activation channel
  CoordinatorConfig cfg;
  std::unique_ptr<objects::CanonicalModelStore> store;
  std::shared_ptr<domain::BackendAdapter> backend;
  std::shared_ptr<transport::SimulatedLink> egress_link;  // Father's single NIC, shared by all Node links
  std::vector<std::unique_ptr<RemoteNode>> nodes;
  std::optional<ClusterPlan> plan;
  Digest256 plan_hash;
  std::map<std::uint32_t, std::unique_ptr<domain::ExecutionDomain>> local;  // Father prefix/tail by stage id
  // Largest sequence-state size each local domain reported during this lease (HQ-PERF-02), by stage id.
  std::map<std::uint32_t, DomainStateReport> local_state_peak;
  void sample_local_state() {
    for (auto& [id, d] : local) {
      auto& peak = local_state_peak[id];
      peak.stage = id;
      const auto m = d->read_metrics();
      peak.state_bytes_peak = std::max(peak.state_bytes_peak, m.state_bytes);
      peak.window_bytes_peak = std::max(peak.window_bytes_peak, m.window_bytes);
    }
  }
  Epoch epoch{0};
  SessionId next_session{1};
  bool prepared = false;

  std::unique_ptr<transport::Connection> wrap(std::unique_ptr<transport::Connection> c) {
    if (!cfg.impairment && !cfg.faults) return c;
    return transport::impair(std::move(c), cfg.impairment.value_or(transport::NetworkConditions{}), egress_link,
                             cfg.faults);
  }

  Result<std::unique_ptr<StreamWithInbox>> open_channel(RemoteNode& n, Channel channel, bool start_reader,
                                                        Inbox* sink = nullptr) {
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
    s->sink = sink;
    s->name = n.endpoint.name;
    if (start_reader) s->start();
    return s;
  }

  // Apply unsolicited Node events: new resource offers (next lease generation) and self-initiated releases.
  void drain_events(RemoteNode& n) {
    if (!n.control) return;
    while (true) {
      auto ev = n.control->inbox.wait(
          [](const ReceivedMessage& r) {
            return r.correlation == 0 && (std::holds_alternative<protocol::OfferResources>(r.message) ||
                                          std::holds_alternative<protocol::ReleaseComplete>(r.message));
          },
          0ms);
      if (!ev.is_ok()) return;
      if (auto* offer = std::get_if<protocol::OfferResources>(&ev->message)) {
        n.offer = *offer;
        n.lease = offer->lease;
      } else if (auto* rc = std::get_if<protocol::ReleaseComplete>(&ev->message)) {
        log::warn("node_released_lease", {{"node", n.endpoint.name}, {"lease", rc->lease.str()},
                                           {"storage_cleaned", rc->storage_cleaned ? "1" : "0"}});
        n.self_released = *rc;
      }
    }
  }

  Status check_nodes_ready() {
    for (const auto* s : remote_stages()) {
      auto& n = node_for(*s);
      drain_events(n);
      if (n.self_released)
        return make_error(ErrorCode::kUnavailable, "Node " + n.endpoint.name + " released its lease");
      if (!n.control || !n.activation) return make_error(ErrorCode::kUnavailable, "Node " + n.endpoint.name + " lost");
    }
    return Status::ok();
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

  // Stream every assigned object over the provision channel in bounded, individually digested chunks. If the
  // bulk connection breaks while the lease is still valid, reconnect and resume: the Node reports which objects
  // are already sealed (ProvisionStatus) and only the rest are sent again, whole.
  Status provision_node(RemoteNode& n, const protocol::PreparePlan& p) {
    std::vector<bool> sealed(p.manifest.objects.size(), false);
    Status last;
    for (int attempt = 0; attempt <= cfg.provision_retries; ++attempt) {
      if (cancel_prepare.load()) return make_error(ErrorCode::kCancelled, "preparation cancelled");
      if (attempt > 0) {
        log::warn("provision_resume", {{"node", n.endpoint.name}, {"attempt", std::to_string(attempt)},
                                       {"after", last.to_string()}});
        if (n.provision) n.provision->shutdown();
        auto ch = open_channel(n, Channel::kProvision, true);
        if (!ch.is_ok()) {
          last = ch.status();
          continue;
        }
        n.provision = std::move(ch).value();
        ++n.provision_report.resumes;
      }
      auto status = n.provision->inbox.wait(
          [](const ReceivedMessage& r) { return std::holds_alternative<protocol::ProvisionStatus>(r.message); },
          cfg.request_timeout);
      if (!status.is_ok()) {
        last = status.status();
        continue;
      }
      for (auto idx : std::get<protocol::ProvisionStatus>(status->message).sealed_objects)
        if (idx < sealed.size()) sealed[idx] = true;
      last = provision_pass(n, p, sealed);
      if (last.is_ok() || last.code() != ErrorCode::kUnavailable) return last;
    }
    return last;
  }

  Status provision_pass(RemoteNode& n, const protocol::PreparePlan& p, std::vector<bool>& sealed) {
    {
      // Objects the Node already holds (a resumed transfer) count as sent; the rest are streamed again, whole.
      std::uint64_t done_bytes = 0;
      std::uint32_t done_objects = 0;
      for (const auto& a : p.assignments)
        if (sealed[a.object_index]) {
          done_bytes += p.manifest.objects[a.object_index].byte_size;
          ++done_objects;
        }
      progress.update(&n, [&](NodePrepareProgress& np) {
        np.bytes_sent = done_bytes;
        np.objects_sealed = done_objects;
      }, true);
    }
    std::size_t outstanding = 0;
    for (const auto& a : p.assignments) {
      if (sealed[a.object_index]) continue;
      if (cancel_prepare.load()) return make_error(ErrorCode::kCancelled, "preparation cancelled");
      const auto& obj = p.manifest.objects[a.object_index];
      // Bounded streaming read: Father never holds more than one chunk of an object in memory. The wall time of the
      // whole call minus the time spent inside the callback is the source read (disk, page cache); the callback's
      // digest and send are accounted separately (HQ-PROV-01: disk read vs. transfer).
      std::uint64_t callback_ns = 0;
      Stopwatch stream_clock;
      const Status streamed = store->stream_object(
          obj.name, cfg.provision_chunk_bytes, [&](std::uint64_t offset, ByteSpan data) -> Status {
            const std::uint64_t cb_start = monotonic_ns();
            protocol::ProvisionChunk chunk;
            chunk.lease = n.lease;
            chunk.object_index = a.object_index;
            chunk.offset = offset;
            chunk.data.assign(data.begin(), data.end());
            chunk.chunk_digest = Sha256::of(chunk.data);
            const std::uint64_t digested = monotonic_ns();
            n.provision_report.father_chunk_digest_ns += digested - cb_start;
            Status sent = n.provision->stream->send(chunk);
            const std::uint64_t done = monotonic_ns();
            n.provision_report.father_send_ns += done - digested;
            callback_ns += done - cb_start;
            if (!sent.is_ok()) return sent;
            progress.update(&n, [&](NodePrepareProgress& np) { np.bytes_sent += data.size(); }, false);
            return Status::ok();
          });
      const std::uint64_t stream_ns = stream_clock.elapsed_ns();
      n.provision_report.father_source_read_ns += stream_ns > callback_ns ? stream_ns - callback_ns : 0;
      CLM_RETURN_IF_ERROR(streamed);
      protocol::SealObject seal{n.lease, a.object_index, obj.byte_size, obj.object_digest};
      CLM_RETURN_IF_ERROR(n.provision->stream->send(seal, n.provision->stream->next_correlation()));
      n.provision_report.bytes += obj.byte_size;
      ++n.provision_report.objects;
      ++outstanding;
    }
    // Every seal must be acknowledged; any chunk/seal error arrives as an ErrorMessage.
    for (; outstanding > 0; --outstanding) {
      CLM_ASSIGN_OR_RETURN(auto reply, n.provision->inbox.wait([](const ReceivedMessage&) { return true; },
                                                               cfg.prepare_timeout));
      if (auto* e = std::get_if<protocol::ErrorMessage>(&reply.message)) return make_error(e->code, e->message);
      auto* ok = std::get_if<protocol::ObjectSealed>(&reply.message);
      if (ok == nullptr) return make_error(ErrorCode::kProtocolError, "unexpected message on provision channel");
      if (ok->object_index < sealed.size() && !sealed[ok->object_index]) {
        sealed[ok->object_index] = true;
        progress.update(&n, [&](NodePrepareProgress& np) { ++np.objects_sealed; }, true);
      }
    }
    return Status::ok();
  }

  Status prepare_node(RemoteNode& n, const StagePlan& s) {
    Stopwatch sw;
    n.stage = s;
    n.provision_report = {};
    n.provision_report.node = n.endpoint.name;
    CLM_ASSIGN_OR_RETURN(auto p, plan_message(n, s));
    {
      std::uint64_t total = 0;
      for (const auto& a : p.assignments) total += p.manifest.objects[a.object_index].byte_size;
      progress.update(&n, [&](NodePrepareProgress& np) {
        np.phase = PreparePhase::kProvisioning;
        np.bytes_total = total;
        np.bytes_sent = 0;
        np.objects_total = static_cast<std::uint32_t>(p.assignments.size());
        np.objects_sealed = 0;
      }, true);
    }
    CLM_RETURN_IF_ERROR(n.control->call<protocol::PlanAccepted>(p, cfg.request_timeout).status());
    CLM_ASSIGN_OR_RETURN(n.provision, open_channel(n, Channel::kProvision, true));
    if (Status pst = provision_node(n, p); !pst.is_ok()) {
      // A Node that fails to prepare (e.g. a backend without its hardware) reports the real reason on the control
      // channel and releases its lease; the provision channel then only sees a stale lease. Prefer the reported reason.
      auto reported = n.control->inbox.wait(
          [](const ReceivedMessage& r) {
            return r.correlation == 0 && std::holds_alternative<protocol::ErrorMessage>(r.message);
          },
          std::chrono::milliseconds(500));
      if (reported.is_ok()) {
        const auto& e = std::get<protocol::ErrorMessage>(reported->message);
        return make_error(e.code, e.message);
      }
      return pst;
    }
    progress.update(&n, [&](NodePrepareProgress& np) {
      np.phase = PreparePhase::kNodePreparing;
      np.bytes_sent = np.bytes_total;
      np.objects_sealed = np.objects_total;
    }, true);
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
    n.provision_report.node_chunk_write_ns = pr.chunk_write_ns;
    n.provision_report.node_seal_hash_ns = pr.seal_hash_ns;
    n.provision_report.node_build_ns = pr.build_ns;
    progress.update(&n, [](NodePrepareProgress& np) { np.phase = PreparePhase::kNodeReady; }, true);
    // Provisioning is over; the bulk channel is not kept open during inference.
    n.provision->shutdown();
    n.provision.reset();
    CLM_ASSIGN_OR_RETURN(n.activation, open_channel(n, Channel::kActivation, true, &results));
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
  //
  // Windows move through the stages as events: Father runs the prefix, sends the activations to the first Node
  // (which forwards along the authorized peer chain in direct mode), and every StageResult from any Node lands
  // in one shared queue (`results`). In relay mode Father forwards each result to the next Node itself. The
  // tail runs when the last remote stage's result arrives. Prefill keeps several auto-committed chunks in flight
  // (bounded); decode has exactly one window in flight.

  std::atomic<bool> cancel_prepare{false};
  ProgressTracker progress;

  struct Flight {
    domain::WindowRequest req;
    bool auto_commit = false;
    std::size_t hop = 0;  // index into remote_stages() of the stage currently computing it
    Stopwatch launched;
    RoundTrace* trace = nullptr;
  };

  static bool cancelled(const GenerationRequest& r) { return r.cancel && r.cancel->load(); }

  void abort_everywhere(SessionId session, const std::string& reason) {
    for (auto& [id, d] : local) (void)d->abort_session(epoch, session);
    for (const auto* s : remote_stages()) {
      auto& n = node_for(*s);
      if (n.control) (void)n.control->call<protocol::SessionOpened>(protocol::AbortSession{epoch, session, reason}, 2s);
    }
    log::warn("session_aborted", {{"epoch", epoch.str()}, {"session", session.str()}, {"reason", reason}});
    epoch = epoch.next();
  }

  // Discard one outstanding window everywhere; sessions keep their committed state. Domains that never saw the
  // window (it had not reached them yet) treat it as a no-op; ones that are still computing it discard it after.
  Status abort_window_everywhere(const domain::WindowRequest& req) {
    for (auto& [id, d] : local) {
      auto r = d->abort_window(req.epoch, req.session, req.window);
      // A local domain that has not admitted the window yet rejects it as never admitted: nothing to discard.
      if (!r.is_ok() && r.status().code() != ErrorCode::kFailedPrecondition) return r.status();
    }
    for (const auto* s : remote_stages()) {
      auto& n = node_for(*s);
      auto r = n.control->call<protocol::WindowAborted>(
          protocol::AbortWindow{req.epoch, req.session, req.window, s->stage}, cfg.request_timeout);
      if (!r.is_ok() && r.status().code() != ErrorCode::kFailedPrecondition) return r.status();
    }
    return Status::ok();
  }

  // Send a window's activations to remote stage `hop`.
  Status send_hop(Flight& f, std::size_t hop, domain::StageActivations&& acts) {
    const auto remotes = remote_stages();
    auto& n = node_for(*remotes[hop]);
    protocol::RunWindow run;
    run.lease = n.lease;
    run.request = f.req;
    run.stage = remotes[hop]->stage;
    run.forward_to_peer = cfg.direct_peer && hop + 1 < remotes.size();
    run.auto_commit = f.auto_commit;
    run.activations = std::move(acts);
    const auto corr = n.activation->stream->next_correlation();
    const auto payload = protocol::encode(run).size();
    CLM_RETURN_IF_ERROR(n.activation->stream->send(run, corr));
    f.hop = hop;
    ++f.trace->boundary_messages;
    f.trace->boundary_payload_bytes += payload;
    return Status::ok();
  }

  // Prefix + first hop (or straight to the tail when every stage is local).
  Result<std::optional<domain::Logits>> launch(Flight& f, std::span<const std::int32_t> tokens) {
    Stopwatch sw;
    auto& prefix = *local.at(plan->stages.front().stage.value);
    CLM_ASSIGN_OR_RETURN(auto acts, prefix.run_prefix(f.req, tokens));
    if (f.auto_commit) CLM_RETURN_IF_ERROR(local_commit(prefix, f.req, f.req.positions).status());
    f.trace->prefix_ms = sw.elapsed_ms();
    if (remote_stages().empty()) {
      CLM_ASSIGN_OR_RETURN(auto logits, run_tail(f, acts));
      return std::optional<domain::Logits>(std::move(logits));
    }
    f.launched.reset();
    CLM_RETURN_IF_ERROR(send_hop(f, 0, std::move(acts)));
    return std::optional<domain::Logits>();
  }

  Result<domain::Logits> run_tail(Flight& f, const domain::StageActivations& acts) {
    Stopwatch sw;
    auto& tail = *local.at(plan->stages.back().stage.value);
    CLM_ASSIGN_OR_RETURN(auto logits, tail.run_tail(f.req, acts));
    if (f.auto_commit) CLM_RETURN_IF_ERROR(local_commit(tail, f.req, f.req.positions).status());
    f.trace->tail_ms = sw.elapsed_ms();
    return logits;
  }

  static Result<domain::CommitAck> local_commit(domain::ExecutionDomain& d, const domain::WindowRequest& req,
                                                std::uint32_t accepted) {
    return d.commit_window({req.epoch, req.session, req.window, accepted, req.expected_state});
  }

  // Wait for the next StageResult of any in-flight window and advance it. Returns the completed window's
  // index into `flights` and its logits when it left the tail, nullopt when it only moved one hop.
  Result<std::optional<std::pair<std::size_t, domain::Logits>>> advance(std::vector<Flight>& flights,
                                                                        const GenerationRequest& request,
                                                                        double& wait_ms) {
    Stopwatch waited;
    const auto deadline = SteadyClock::now() + cfg.window_timeout;
    Result<ReceivedMessage> got = make_error(ErrorCode::kDeadlineExceeded, "no result");
    while (true) {
      got = results.wait([](const ReceivedMessage&) { return true; }, 50ms);
      if (got.is_ok() || got.status().code() != ErrorCode::kDeadlineExceeded) break;
      if (SteadyClock::now() >= deadline)
        return make_error(ErrorCode::kDeadlineExceeded, "timed out waiting for a stage result");
      if (cancelled(request) && !flights.empty() && !flights.front().auto_commit)
        return make_error(ErrorCode::kCancelled, "cancelled while waiting for verification");
    }
    wait_ms += waited.elapsed_ms();
    if (!got.is_ok()) return got.status();
    if (auto* e = std::get_if<protocol::ErrorMessage>(&got->message)) return make_error(e->code, e->message);
    auto* sr = std::get_if<protocol::StageResult>(&got->message);
    if (sr == nullptr) return make_error(ErrorCode::kProtocolError, "unexpected message on an activation channel");
    std::size_t idx = flights.size();
    for (std::size_t i = 0; i < flights.size(); ++i)
      if (flights[i].req.window == sr->window && flights[i].req.session == sr->session && flights[i].req.epoch == sr->epoch)
        idx = i;
    // A result for a window that is no longer in flight (aborted window, previous epoch) is a straggler.
    if (idx == flights.size()) return std::optional<std::pair<std::size_t, domain::Logits>>();
    Flight& f = flights[idx];
    if (sr->status != ErrorCode::kOk)
      return make_error(sr->status, "stage " + sr->stage.str() + ": " + sr->error_message);
    ++f.trace->boundary_messages;
    f.trace->boundary_payload_bytes += got->wire_bytes;
    const auto remotes = remote_stages();
    std::size_t done_hop = remotes.size();
    for (std::size_t h = 0; h < remotes.size(); ++h)
      if (remotes[h]->stage == sr->stage) done_hop = h;
    if (done_hop == remotes.size()) return make_error(ErrorCode::kProtocolError, "StageResult from an unknown stage");
    f.trace->remote_timings.insert(f.trace->remote_timings.end(), sr->timings.begin(), sr->timings.end());
    if (done_hop + 1 < remotes.size()) {
      // Relay mode: Father forwards to the next Node. (In direct mode only the last Node reports to Father.)
      if (cfg.direct_peer) return make_error(ErrorCode::kProtocolError, "intermediate result in direct-peer mode");
      CLM_RETURN_IF_ERROR(send_hop(f, done_hop + 1, std::move(sr->activations)));
      return std::optional<std::pair<std::size_t, domain::Logits>>();
    }
    f.trace->remote_ms = f.launched.elapsed_ms();
    CLM_ASSIGN_OR_RETURN(auto logits, run_tail(f, sr->activations));
    return std::optional<std::pair<std::size_t, domain::Logits>>(std::make_pair(idx, std::move(logits)));
  }

  // Commit `accepted` positions of a decode window on every domain and wait for every acknowledgement. A lost
  // acknowledgement is retried with the identical (idempotent) CommitWindow; the Node replays its ack.
  Status commit_round(const domain::WindowRequest& req, std::uint32_t accepted, RoundTrace& trace) {
    Stopwatch sw;
    domain::CommitRequest c{req.epoch, req.session, req.window, accepted, req.expected_state};
    struct Pending {
      RemoteNode* node;
      StageId stage;
      std::uint64_t corr;
    };
    std::vector<Pending> pending;
    for (const auto* s : remote_stages()) {
      auto& n = node_for(*s);
      const auto corr = n.control->stream->next_correlation();
      CLM_RETURN_IF_ERROR(n.control->stream->send(protocol::CommitWindow{c, s->stage}, corr));
      ++trace.control_messages;
      pending.push_back({&n, s->stage, corr});
    }
    std::optional<domain::CommitAck> reference;
    for (auto& [id, d] : local) {
      CLM_ASSIGN_OR_RETURN(auto ack, d->commit_window(c));
      if (!reference) reference = ack;
    }
    for (auto& p : pending) {
      Result<ReceivedMessage> reply = make_error(ErrorCode::kDeadlineExceeded, "no ack");
      std::vector<std::uint64_t> corrs{p.corr};
      auto timeout = cfg.request_timeout;
      for (int attempt = 0;; ++attempt) {
        // Either the original commit's ack (merely delayed) or a replay's ack settles the outcome.
        reply = p.node->control->inbox.wait(
            [&corrs](const ReceivedMessage& r) {
              return std::find(corrs.begin(), corrs.end(), r.correlation) != corrs.end();
            },
            timeout);
        if (reply.is_ok() || reply.status().code() != ErrorCode::kDeadlineExceeded || attempt >= cfg.commit_retries)
          break;
        // Unknown commit outcome: resend the same commit (idempotent on the Node) and wait longer.
        log::warn("commit_ack_timeout_retry", {{"node", p.node->endpoint.name}, {"window", req.window.str()}});
        corrs.push_back(p.node->control->stream->next_correlation());
        CLM_RETURN_IF_ERROR(p.node->control->stream->send(protocol::CommitWindow{c, p.stage}, corrs.back()));
        ++trace.commit_retries;
        ++trace.control_messages;
        timeout *= 2;
      }
      // Drop whichever duplicate ack arrives later for this commit.
      for (auto corr : corrs) p.node->control->inbox.discard_correlation(corr);
      if (!reply.is_ok()) return reply.status();
      ++trace.control_messages;
      if (auto* e = std::get_if<protocol::ErrorMessage>(&reply->message)) return make_error(e->code, e->message);
      auto* ack = std::get_if<protocol::CommitAckMessage>(&reply->message);
      if (ack == nullptr) return make_error(ErrorCode::kProtocolError, "expected CommitAck");
      // Every domain must agree on the committed position and state version.
      if (reference && (ack->ack.committed_position != reference->committed_position ||
                        ack->ack.state != reference->state))
        return make_error(ErrorCode::kAborted, "domains disagree on committed state");
    }
    trace.commit_ms = sw.elapsed_ms();
    return Status::ok();
  }

  // Prefill `tokens` into the conversation as auto-committed chunks with at most `inflight` in the pipeline.
  // Returns the logits of the last chunk's last position (the distribution of the next token).
  Result<std::vector<float>> prefill(ConversationState& cs, std::span<const std::int32_t> tokens,
                                     const GenerationRequest& request, GenerationResult& out) {
    const std::uint32_t chunk = std::max<std::uint32_t>(1, std::min(request.prefill_chunk, plan->max_window));
    const std::uint32_t inflight = std::max<std::uint32_t>(1, request.prefill_inflight);
    const auto vocab = store->manifest().geometry.vocab_size;
    std::vector<Flight> flights;
    std::deque<RoundTrace> traces;  // stable addresses while windows are in flight
    std::vector<float> last_logits;
    std::size_t next = 0;      // next token offset to launch
    bool stop_launching = false;
    Stopwatch sw;
    // Future positions/states are known in advance: each chunk commits fully.
    std::uint64_t launch_position = cs.position;
    StateVersion launch_state = cs.state;
    auto finish = [&](std::size_t idx, domain::Logits&& logits) {
      Flight& f = flights[idx];
      f.trace->total_ms = f.launched.elapsed_ms() + f.trace->prefix_ms;
      cs.position += f.req.positions;
      cs.state = cs.state.next();
      last_logits.assign(logits.data.end() - static_cast<std::ptrdiff_t>(vocab), logits.data.end());
      flights.erase(flights.begin() + static_cast<std::ptrdiff_t>(idx));
    };
    while (next < tokens.size() || !flights.empty()) {
      if (!stop_launching && cancelled(request)) stop_launching = true;  // drain what is in flight, launch no more
      if (!stop_launching && next < tokens.size() && flights.size() < inflight) {
        const auto n = static_cast<std::uint32_t>(std::min<std::size_t>(chunk, tokens.size() - next));
        cs.window = cs.window.next();
        Flight f;
        f.req = {cs.epoch, cs.session, cs.window, launch_position, launch_state, n};
        f.auto_commit = true;
        traces.emplace_back();
        f.trace = &traces.back();
        f.trace->prefill = true;
        f.trace->positions = n;
        f.trace->accepted = n;
        f.trace->in_flight = static_cast<std::uint32_t>(flights.size());
        out.prefill_max_in_flight = std::max<std::uint32_t>(out.prefill_max_in_flight, f.trace->in_flight + 1);
        CLM_ASSIGN_OR_RETURN(auto done, launch(f, tokens.subspan(next, n)));
        cs.committed.insert(cs.committed.end(), tokens.begin() + static_cast<std::ptrdiff_t>(next),
                            tokens.begin() + static_cast<std::ptrdiff_t>(next + n));
        launch_position += n;
        launch_state = launch_state.next();
        next += n;
        ++out.prefill_chunks;
        out.prefill_tokens += n;
        flights.push_back(std::move(f));
        if (done) finish(flights.size() - 1, std::move(*done));
        continue;
      }
      if (flights.empty()) break;  // cancelled with nothing in flight
      double wait = 0;
      CLM_ASSIGN_OR_RETURN(auto adv, advance(flights, request, wait));
      out.prefill_wait_ms += wait;
      if (adv) finish(adv->first, std::move(adv->second));
    }
    const double ms = sw.elapsed_ms();
    out.prefill_ms += ms;
    if (ms > 0) out.prefill_tok_s = 1000.0 * out.prefill_tokens / ms;
    for (auto& t : traces) out.rounds.push_back(std::move(t));
    // Chunks that never launched are simply not part of the conversation; every launched chunk was drained, so
    // all domains agree on the committed position and the conversation stays usable.
    if (stop_launching && next < tokens.size()) return make_error(ErrorCode::kCancelled, "cancelled during prefill");
    return last_logits;
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
  impl->backend = impl->cfg.backend ? impl->cfg.backend : std::shared_ptr<domain::BackendAdapter>(domain::make_reference_backend());
  if (impl->cfg.impairment) impl->egress_link = std::make_shared<transport::SimulatedLink>(impl->cfg.impairment->bandwidth_bytes_per_s);
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

Result<PrepareReport> Coordinator::prepare(const ClusterPlan& plan, PrepareProgressSink progress_sink) {
  auto& im = *impl_;
  const auto& m = im.store->manifest();
  CLM_RETURN_IF_ERROR(plan.validate(m.geometry, im.nodes.size()));
  if (im.prepared) return make_error(ErrorCode::kFailedPrecondition, "release the current plan first");
  for (auto& n : im.nodes) {
    im.drain_events(*n);
    n->self_released.reset();
  }
  im.results.clear();
  Stopwatch sw;
  im.plan = plan;
  im.plan_hash = plan.hash(m.root_hash());
  log::info("prepare_plan", {{"plan", plan.describe()}, {"plan_hash", im.plan_hash.hex().substr(0, 16)}});

  {
    std::vector<std::pair<const RemoteNode*, std::string>> remote;
    for (const auto& s : plan.stages)
      if (s.domain != kFatherDomain) remote.emplace_back(&im.node_for(s), im.node_for(s).endpoint.name);
    im.progress.begin(std::move(progress_sink), remote);
  }
  struct EndProgress {
    ProgressTracker& t;
    ~EndProgress() { t.end(); }
  } end_progress{im.progress};
  im.progress.set_overall(PreparePhase::kFatherDomains);

  // Father-local prefix and tail domains, bound to the canonical store (which loads only their objects).
  for (const auto& s : plan.stages) {
    if (s.domain != kFatherDomain) continue;
    domain::DomainSpec spec{s.stage, s.role, s.layers, plan.max_context, plan.max_window, 1};
    // The backend's real error (e.g. kHardwareUnavailable from the Strata CUDA engine) is returned as is; domains
    // already prepared are released so a failed prepare leaves nothing behind (no Node was contacted yet).
    auto fail_local = [&](const Status& st) {
      log::error("prepare_failed", {{"error", st.to_string()}});
      for (auto& [id, ld] : im.local) (void)ld->release();
      im.local.clear();
      im.local_state_peak.clear();
      im.plan.reset();
      return st;
    };
    auto d = im.backend->create_domain(m, spec);
    if (!d.is_ok()) return fail_local(d.status());
    if (Status st = d.value()->prepare(*im.store); !st.is_ok()) return fail_local(st);
    im.local.emplace(s.stage.value, std::move(d).value());
  }

  // Provision Nodes concurrently. Their transfers share Father's single egress link, so concurrency overlaps
  // per-Node hashing/loading rather than multiplying bandwidth.
  im.progress.set_overall(PreparePhase::kProvisioning);
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
  if (im.cfg.direct_peer) {
    im.progress.set_overall(PreparePhase::kAuthorizing);
    if (Status st = im.authorize_peers(); !st.is_ok()) {
      im.prepared = true;  // so release() cleans up the provisioned Nodes and local domains
      (void)release();
      return st;
    }
  }
  im.prepared = true;
  im.progress.set_overall(PreparePhase::kDone);

  PrepareReport report;
  report.plan_hash = im.plan_hash;
  for (const auto* s : im.remote_stages()) report.nodes.push_back(im.node_for(*s).provision_report);
  report.father_resident_bytes = im.store->loaded_bytes();
  report.total_ms = sw.elapsed_ms();
  return report;
}

bool Coordinator::ready() const { return impl_->prepared; }

namespace {

std::uint64_t peak_rss_bytes() {
#ifdef _WIN32
  PROCESS_MEMORY_COUNTERS pmc{};
  if (K32GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc)) return pmc.PeakWorkingSetSize;
  return 0;
#else
  std::ifstream f("/proc/self/status");
  std::string line;
  while (std::getline(f, line))
    if (line.rfind("VmHWM:", 0) == 0) return std::stoull(line.substr(6)) * 1024;
  return 0;
#endif
}

}  // namespace

Result<std::shared_ptr<domain::Drafter>> Coordinator::make_drafter() {
  auto& im = *impl_;
  if (!im.prepared || !im.plan) return make_error(ErrorCode::kFailedPrecondition, "no prepared plan");
  if (im.backend->info().name != "strata") {
    CLM_ASSIGN_OR_RETURN(auto d, domain::MtpFixtureDrafter::create(im.store->manifest(), *im.store));
    return std::shared_ptr<domain::Drafter>(std::move(d));
  }
  auto it = im.local.find(im.plan->stages.back().stage.value);
  if (it == im.local.end() || it->second->spec().role != domain::StageRole::kTail)
    return make_error(ErrorCode::kFailedPrecondition, "the plan has no Father tail domain");
  CLM_ASSIGN_OR_RETURN(auto d, backends::make_backend_drafter(im.backend->info().name, *it->second));
  return std::shared_ptr<domain::Drafter>(std::move(d));
}

Result<std::shared_ptr<Conversation>> Coordinator::open_conversation() {
  auto& im = *impl_;
  if (!im.prepared) return make_error(ErrorCode::kFailedPrecondition, "no prepared plan");
  CLM_RETURN_IF_ERROR(im.check_nodes_ready());
  auto conv = std::make_shared<Conversation>();
  auto& cs = conv->coordinator_state();
  im.epoch = im.epoch.next();
  cs.epoch = im.epoch;
  cs.session = im.next_session;
  im.next_session = im.next_session.next();
  for (auto& [id, d] : im.local) CLM_RETURN_IF_ERROR(d->open_session(cs.epoch, cs.session));
  im.sample_local_state();
  for (const auto* s : im.remote_stages()) {
    auto st = im.node_for(*s).control->call<protocol::SessionOpened>(protocol::OpenSession{cs.epoch, cs.session},
                                                                     im.cfg.request_timeout);
    if (!st.is_ok()) {
      im.abort_everywhere(cs.session, "open failed");
      return st.status();
    }
  }
  cs.valid = true;
  return conv;
}

Status Coordinator::close_conversation(Conversation& conversation) {
  auto& im = *impl_;
  auto& cs = conversation.coordinator_state();
  if (!cs.valid) return Status::ok();
  cs.valid = false;
  for (auto& [id, d] : im.local) (void)d->abort_session(cs.epoch, cs.session);
  for (const auto* s : im.remote_stages()) {
    auto& n = im.node_for(*s);
    if (n.control)
      (void)n.control->call<protocol::SessionOpened>(protocol::AbortSession{cs.epoch, cs.session, "closed"},
                                                     im.cfg.request_timeout);
  }
  return Status::ok();
}

Result<GenerationResult> Coordinator::generate(const GenerationRequest& request) {
  auto& im = *impl_;
  if (!im.prepared) return make_error(ErrorCode::kFailedPrecondition, "no prepared plan");
  if (request.q == 0 || request.q > im.plan->max_window)
    return make_error(ErrorCode::kInvalidArgument, "q must be in [1, max_window]");
  if (request.q > 1 && !request.drafter) return make_error(ErrorCode::kInvalidArgument, "q > 1 requires a drafter");
  if (request.max_new_tokens == 0) return make_error(ErrorCode::kInvalidArgument, "max_new_tokens must be positive");
  const auto vocab = im.store->manifest().geometry.vocab_size;
  for (auto t : request.prompt)
    if (t < 0 || static_cast<std::uint32_t>(t) >= vocab) return make_error(ErrorCode::kInvalidArgument, "token outside the vocabulary");
  CLM_RETURN_IF_ERROR(im.check_nodes_ready());

  std::shared_ptr<Conversation> conv = request.conversation;
  const bool one_shot = !conv;
  if (one_shot) {
    CLM_ASSIGN_OR_RETURN(conv, open_conversation());
  } else if (!conv->valid()) {
    return make_error(ErrorCode::kFailedPrecondition, "conversation session is no longer valid; open a new one");
  }
  auto& cs = conv->coordinator_state();
  // This turn feeds the previous turn's unfed prediction first, then the new tokens.
  std::vector<std::int32_t> feed = cs.pending;
  feed.insert(feed.end(), request.prompt.begin(), request.prompt.end());
  if (feed.empty()) return make_error(ErrorCode::kInvalidArgument, "nothing to prefill");
  if (cs.position + feed.size() + request.max_new_tokens + request.q > im.plan->max_context)
    return make_error(ErrorCode::kResourceExhausted, "turn would exceed the plan's context reservation");
  cs.pending.clear();

  GenerationResult out;
  out.epoch = cs.epoch;
  out.session = cs.session;
  // A non-zero seed makes the turn reproducible; seed 0 derives one from the session (still deterministic).
  domain::Rng rng(request.sampling.seed != 0 ? request.sampling.seed : domain::mix_seed(cs.session.value, cs.position));
  Stopwatch total;

  // Distributed failures invalidate the session everywhere; Father keeps its history for a re-prefill.
  auto fail = [&](const Status& st) -> Status {
    cs.valid = false;
    im.abort_everywhere(cs.session, st.message());
    return make_error(st.code(), "distributed session invalidated: " + st.message());
  };

  auto prefilled = im.prefill(cs, feed, request, out);
  if (!prefilled.is_ok()) {
    if (prefilled.status().code() == ErrorCode::kCancelled) {
      out.cancelled = true;
      out.peak_rss_bytes = peak_rss_bytes();
      if (one_shot) (void)close_conversation(*conv);
      return out;
    }
    return fail(prefilled.status());
  }
  auto emit = [&](std::span<const std::int32_t> toks) {
    std::size_t n = 0;
    for (auto t : toks) {
      if (out.tokens.size() >= request.max_new_tokens || out.stopped_on_token) break;
      out.tokens.push_back(t);
      ++n;
      if (std::find(request.stop_tokens.begin(), request.stop_tokens.end(), t) != request.stop_tokens.end())
        out.stopped_on_token = true;
    }
    if (n > 0 && request.on_tokens) request.on_tokens(std::span(out.tokens).last(n));
    return n;
  };
  auto next_from = [&](std::span<const float> logits) {
    if (request.sampling.greedy()) return domain::argmax_token(logits);
    auto p = domain::distribution(logits, request.sampling);
    return domain::sample_from(p, rng);
  };
  std::int32_t next_token = next_from(prefilled.value());
  emit(std::span(&next_token, 1));
  out.first_token_ms = total.elapsed_ms();

  // Decode: speculative windows [next_token, d1..d_{q-1}]. Drafts are never output unless accepted.
  Stopwatch decode;
  while (out.tokens.size() < request.max_new_tokens && !out.stopped_on_token) {
    if (Impl::cancelled(request)) {
      out.cancelled = true;
      break;
    }
    // Never verify more positions than the output budget can show: every committed position must correspond to
    // a token the caller receives (or the one pending prediction).
    const auto remaining = static_cast<std::uint32_t>(request.max_new_tokens - out.tokens.size());
    const std::uint32_t q = std::min(request.q, remaining);
    RoundTrace trace;
    Stopwatch rsw;
    std::vector<std::int32_t> window_tokens{next_token};
    domain::DraftProposal proposal;
    if (q > 1) {
      Stopwatch dsw;
      proposal = request.drafter->propose(cs.committed, next_token, q - 1, request.sampling, rng);
      trace.draft_ms = dsw.elapsed_ms();
      if (proposal.tokens.size() != q - 1) return fail(make_error(ErrorCode::kInternal, "drafter returned wrong count"));
      window_tokens.insert(window_tokens.end(), proposal.tokens.begin(), proposal.tokens.end());
    }
    cs.window = cs.window.next();
    Impl::Flight f;
    f.req = {cs.epoch, cs.session, cs.window, cs.position, cs.state, q};
    f.trace = &trace;
    trace.positions = q;
    trace.proposed = q - 1;
    const auto window_req = f.req;  // f is moved into `flights` below; the commit needs the request afterwards
    std::vector<Impl::Flight> flights;
    auto launched = im.launch(f, window_tokens);
    if (!launched.is_ok()) return fail(launched.status());
    domain::Logits logits;
    if (launched.value()) {
      logits = std::move(*launched.value());
    } else {
      flights.push_back(std::move(f));
      while (true) {
        auto adv = im.advance(flights, request, trace.wait_ms);
        if (!adv.is_ok()) {
          if (adv.status().code() == ErrorCode::kCancelled) {
            // Discard the outstanding window everywhere; the conversation keeps its committed state and the
            // unfed prediction stays pending for the next turn.
            if (auto st = im.abort_window_everywhere(flights.front().req); !st.is_ok()) return fail(st);
            out.cancelled = true;
            break;
          }
          return fail(adv.status());
        }
        if (adv.value()) {
          logits = std::move(adv.value()->second);
          break;
        }
      }
      if (out.cancelled) break;
    }
    // Verify on Father: greedy prefix match, or exact stochastic speculative sampling.
    std::uint32_t accepted_drafts = 0;
    std::int32_t following = -1;
    auto at = [&](std::uint32_t i) { return std::span<const float>(logits.data).subspan(std::size_t{i} * vocab, vocab); };
    if (request.sampling.greedy()) {
      while (accepted_drafts + 1 < q && domain::argmax_token(at(accepted_drafts)) == window_tokens[accepted_drafts + 1])
        ++accepted_drafts;
      following = domain::argmax_token(at(accepted_drafts));
    } else {
      std::vector<std::vector<float>> target;
      for (std::uint32_t i = 0; i < q; ++i) target.push_back(domain::distribution(at(i), request.sampling));
      auto outcome = domain::verify_speculative(target, proposal.tokens, proposal.probs, rng);
      accepted_drafts = outcome.accepted_drafts;
      following = outcome.next_token;
    }
    // A stop token among the accepted drafts ends the turn there: commit through it and nothing after.
    for (std::uint32_t i = 0; i < accepted_drafts; ++i) {
      const auto t = window_tokens[i + 1];
      if (std::find(request.stop_tokens.begin(), request.stop_tokens.end(), t) != request.stop_tokens.end()) {
        accepted_drafts = i;
        following = t;
        break;
      }
    }
    const std::uint32_t accepted = accepted_drafts + 1;  // positions committed: next_token + accepted drafts
    if (auto st = im.commit_round(window_req, accepted, trace); !st.is_ok()) return fail(st);
    cs.position += accepted;
    cs.state = cs.state.next();
    cs.committed.insert(cs.committed.end(), window_tokens.begin(), window_tokens.begin() + accepted);
    std::vector<std::int32_t> new_tokens(window_tokens.begin() + 1, window_tokens.begin() + accepted);
    new_tokens.push_back(following);
    trace.accepted = accepted;
    trace.accepted_drafts = accepted_drafts;
    trace.emitted = static_cast<std::uint32_t>(emit(new_tokens));
    next_token = following;
    trace.total_ms = rsw.elapsed_ms();
    out.proposed_positions += trace.proposed;
    out.accepted_drafts += accepted_drafts;
    out.draft_ms_total += trace.draft_ms;
    out.verify_ms_total += trace.prefix_ms + trace.remote_ms + trace.tail_ms;
    out.commit_ms_total += trace.commit_ms;
    out.rounds.push_back(std::move(trace));
    ++out.decode_rounds;
  }
  // The last emitted token has not been fed yet; the next turn feeds it first.
  cs.pending.push_back(next_token);
  out.decode_ms = decode.elapsed_ms();
  out.decode_tokens = out.tokens.empty() ? 0 : static_cast<std::uint32_t>(out.tokens.size() - 1);
  out.peak_rss_bytes = peak_rss_bytes();
  if (one_shot) (void)close_conversation(*conv);
  return out;
}

void Coordinator::cancel_prepare() { impl_->cancel_prepare.store(true); }

Status Coordinator::enable_routing_aggregation(bool on) {
  if (!impl_->prepared) return make_error(ErrorCode::kFailedPrecondition, "no prepared plan");
  for (auto& [id, d] : impl_->local) CLM_RETURN_IF_ERROR(d->enable_routing_aggregation(on));
  return Status::ok();
}

Result<std::vector<domain::RoutingAggregate>> Coordinator::routing_aggregates() const {
  std::vector<domain::RoutingAggregate> out;
  for (const auto& [id, d] : impl_->local) {
    CLM_ASSIGN_OR_RETURN(auto a, d->routing_aggregate());
    if (!a.counts.empty()) out.push_back(std::move(a));
  }
  return out;
}

Result<ReleaseReport> Coordinator::release() {
  auto& im = *impl_;
  ReleaseReport report;
  for (auto& n : im.nodes) {
    if (!n->control || !n->stage) continue;
    im.drain_events(*n);
    Stopwatch sw;
    if (n->activation) n->activation->shutdown();
    n->activation.reset();
    if (n->provision) n->provision->shutdown();
    n->provision.reset();
    Result<protocol::ReleaseComplete> rc =
        n->self_released ? Result<protocol::ReleaseComplete>(*n->self_released)
                         : n->control->call<protocol::ReleaseComplete>(
                               protocol::ReleaseLease{n->lease, protocol::ReleaseReason::kFatherRequest},
                               im.cfg.request_timeout);
    ReleaseReport::NodeRelease nr;
    nr.node = n->endpoint.name;
    nr.release_ms = sw.elapsed_ms();
    if (rc.is_ok()) {
      nr.resources_released = rc->resources_released;
      nr.storage_cleaned = rc->storage_cleaned;
      nr.residual_bytes = rc->residual_bytes;
      nr.errors = rc->errors;
      for (const auto& d : rc->domain_state) nr.domain_state.push_back({d.stage.value, d.state_bytes_peak, d.window_bytes_peak});
    } else {
      nr.errors = rc.status().to_string();
      if (rc.status().code() == ErrorCode::kUnavailable) {
        // The Node is gone (crash / Father-side link loss). Its own restart runs orphan recovery; Father
        // reconnects on the next connect().
        n->control->shutdown();
        n->control.reset();
      }
    }
    // An Available Node re-offers under its next lease generation (a Busy one offers later).
    if (!n->self_released && n->control) {
      auto offer = n->control->inbox.wait(
          [](const ReceivedMessage& r) { return std::holds_alternative<protocol::OfferResources>(r.message); }, 2s);
      if (offer.is_ok()) {
        n->offer = std::get<protocol::OfferResources>(offer->message);
        n->lease = n->offer.lease;
      }
    }
    n->self_released.reset();
    n->stage.reset();
    report.nodes.push_back(std::move(nr));
  }
  im.sample_local_state();
  for (const auto& [id, peak] : im.local_state_peak) report.father_domain_state.push_back(peak);
  im.local_state_peak.clear();
  for (auto& [id, d] : im.local) (void)d->release();
  im.local.clear();
  im.results.clear();
  im.plan.reset();
  im.prepared = false;
  return report;
}

}  // namespace clusterlm::coordinator
