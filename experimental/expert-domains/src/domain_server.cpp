#include "clusterlm/expert_domains/domain_server.hpp"

#include <algorithm>
#include <chrono>

#include "clusterlm/common/clock.hpp"
#include "clusterlm/objects/manifest.hpp"
#include "expert_kernel.hpp"

namespace clusterlm::expert_domains {

using namespace std::chrono_literals;

struct ExpertDomainServer::Impl {
  objects::ModelGeometry geometry;
  std::uint32_t first_layer = 0, end_layer = 0;
  std::vector<std::vector<ExpertWeights>> experts;  // [layer - first_layer][local expert]
  ExpertScratch scratch;
  std::unique_ptr<QuantExperts> quant;  // set when config.kernel is quantized (then `experts` stays empty)
  std::mutex mu;
  std::uint64_t last_window = 0;
  std::uint64_t served = 0;  // batches answered (fault-injection counter)
  ExpertDomainMetrics metrics;
};

ExpertDomainServer::ExpertDomainServer(const objects::ModelManifest& manifest, ExpertDomainConfig config)
    : impl_(std::make_unique<Impl>()), config_(std::move(config)) {
  impl_->geometry = manifest.geometry;
  epoch_.store(config_.epoch.value);
}

ExpertDomainServer::~ExpertDomainServer() { stop(); }

Result<std::unique_ptr<ExpertDomainServer>> ExpertDomainServer::create(const objects::ModelManifest& manifest,
                                                                       const objects::ObjectResolver& resolver,
                                                                       ExpertDomainConfig config) {
  CLM_RETURN_IF_ERROR(manifest.validate());
  const objects::ModelGeometry& g = manifest.geometry;
  if (config.end_layer == 0) config.end_layer = g.n_layers;
  if (config.first_layer >= config.end_layer || config.end_layer > g.n_layers)
    return make_error(ErrorCode::kInvalidArgument, "expert domain layer range outside the model");
  if (config.owned_experts.empty() || !std::is_sorted(config.owned_experts.begin(), config.owned_experts.end()) ||
      std::adjacent_find(config.owned_experts.begin(), config.owned_experts.end()) != config.owned_experts.end() ||
      config.owned_experts.back() >= g.n_experts)
    return make_error(ErrorCode::kInvalidArgument, "owned_experts must be a non-empty ascending set of valid ids");
  if (config.owned_experts.size() > config.limits.max_local_experts)
    return make_error(ErrorCode::kInvalidArgument, "more owned experts than the wire limit allows");
  if (config.limits.max_hidden < g.hidden_size || config.limits.max_layer < g.n_layers)
    return make_error(ErrorCode::kInvalidArgument, "decode limits smaller than the model geometry");
  std::unique_ptr<ExpertDomainServer> s(new ExpertDomainServer(manifest, std::move(config)));
  CLM_RETURN_IF_ERROR(s->load(resolver));
  return s;
}

Status ExpertDomainServer::load(const objects::ObjectResolver& resolver) {
  Impl& m = *impl_;
  const objects::ModelGeometry& g = m.geometry;
  m.first_layer = config_.first_layer;
  m.end_layer = config_.end_layer;
  m.experts.clear();
  std::uint64_t bytes = 0;
  if (config_.kernel.quantized()) {
    CLM_ASSIGN_OR_RETURN(m.quant, QuantExperts::create(config_.kernel, g.hidden_size, g.expert_ff, m.first_layer, m.end_layer,
                                                       static_cast<std::uint32_t>(config_.owned_experts.size()),
                                                       config_.owned_experts.front()));
    m.metrics.resident_bytes = m.quant->resident_bytes();
    return Status::ok();
  }
  for (std::uint32_t L = m.first_layer; L < m.end_layer; ++L) {
    std::vector<ExpertWeights> layer;
    layer.reserve(config_.owned_experts.size());
    for (std::uint32_t e : config_.owned_experts) {
      CLM_ASSIGN_OR_RETURN(objects::ProvisionedObject obj, resolver.resolve(objects::expert_object_name(L, e)));
      ExpertWeights w;
      CLM_RETURN_IF_ERROR(load_expert_weights(obj, g.hidden_size, g.expert_ff, w));
      bytes += (w.gate.size() + w.up.size() + w.down.size()) * sizeof(float);
      layer.push_back(std::move(w));
    }
    m.experts.push_back(std::move(layer));
  }
  m.scratch.init(g.hidden_size, g.expert_ff);
  m.metrics.resident_bytes = bytes;
  return Status::ok();
}

void ExpertDomainServer::release() {
  std::lock_guard<std::mutex> lk(impl_->mu);
  impl_->experts.clear();
  impl_->experts.shrink_to_fit();
  impl_->quant.reset();
  impl_->metrics.resident_bytes = 0;
}

std::string ExpertDomainServer::kernel_path() const {
  std::lock_guard<std::mutex> lk(impl_->mu);
  return impl_->quant ? impl_->quant->kernel_path() : std::string();
}

ExpertDomainMetrics ExpertDomainServer::metrics() const {
  std::lock_guard<std::mutex> lk(impl_->mu);
  return impl_->metrics;
}

Result<ExpertResult> ExpertDomainServer::execute(const ExpertBatch& batch) {
  std::lock_guard<std::mutex> lk(impl_->mu);
  auto r = execute_locked(batch);
  if (!r.is_ok()) ++impl_->metrics.rejected;
  return r;
}

Result<ExpertResult> ExpertDomainServer::execute_locked(const ExpertBatch& batch) {
  Impl& m = *impl_;
  const std::size_t H = m.geometry.hidden_size, ff = m.geometry.expert_ff;
  if (m.experts.empty() && !m.quant) return make_error(ErrorCode::kFailedPrecondition, "domain released");
  if (batch.epoch.value != epoch_.load())
    return make_error(ErrorCode::kStaleEpoch, "batch names a stale epoch");
  if (batch.window.value < m.last_window)
    return make_error(ErrorCode::kFailedPrecondition, "window id moved backwards");
  if (batch.layer < m.first_layer || batch.layer >= m.end_layer)
    return make_error(ErrorCode::kOutOfRange, "layer not served by this domain");
  if (batch.hidden != H) return make_error(ErrorCode::kInvalidArgument, "hidden size mismatch");
  if (batch.positions == 0 || batch.positions > config_.limits.max_positions ||
      batch.activations.size() != std::size_t{batch.positions} * H || batch.routes.size() != batch.positions)
    return make_error(ErrorCode::kInvalidArgument, "malformed batch dimensions");
  const std::size_t n_local = config_.owned_experts.size();
  for (const auto& per : batch.routes)
    for (const ExpertRoute& r : per)
      if (r.local_expert >= n_local) return make_error(ErrorCode::kOutOfRange, "local expert id out of range");
  static const std::vector<ExpertWeights> kNoWeights;
  const std::vector<ExpertWeights>& layer = m.quant ? kNoWeights : m.experts[batch.layer - m.first_layer];

  m.last_window = batch.window.value;
  ExpertResult out;
  out.epoch = batch.epoch;
  out.window = batch.window;
  out.layer = batch.layer;
  out.positions = batch.positions;
  out.hidden = batch.hidden;
  out.partial.assign(std::size_t{batch.positions} * H, 0.0f);
  const std::uint64_t t0 = monotonic_ns();
  for (std::uint32_t p = 0; p < batch.positions; ++p) {
    float* y = out.partial.data() + std::size_t{p} * H;
    const float* h = batch.activations.data() + std::size_t{p} * H;
    // Routes are strictly ascending on the wire, so the summation order is fixed by the message itself.
    for (const ExpertRoute& r : batch.routes[p]) {
      if (m.quant) {
        const Status qs = m.quant->accumulate(batch.layer, r.local_expert, h, r.weight, y);
        if (!qs.is_ok()) return qs;
      } else {
        accumulate_expert(layer[r.local_expert], H, ff, h, r.weight, m.scratch, y);
      }
      ++out.experts_executed;
    }
  }
  out.compute_ns = monotonic_ns() - t0;
  ++m.metrics.batches;
  m.metrics.executions += out.experts_executed;
  m.metrics.compute_ns += out.compute_ns;
  return out;
}

Status ExpertDomainServer::serve(std::unique_ptr<transport::Connection> connection) {
  if (running_.load() || thread_.joinable()) return make_error(ErrorCode::kFailedPrecondition, "already serving");
  if (!connection) return make_error(ErrorCode::kInvalidArgument, "null connection");
  connection->set_max_payload(config_.limits.max_payload_bytes());
  {
    std::lock_guard<std::mutex> lk(impl_->mu);
    impl_->last_window = 0;
    impl_->served = 0;
  }
  connection_ = std::shared_ptr<transport::Connection>(std::move(connection));
  stop_.store(false);
  running_.store(true);
  thread_ = std::thread([this, c = connection_] { run(c); });
  return Status::ok();
}

void ExpertDomainServer::stop() {
  stop_.store(true);
  if (connection_) connection_->close();
  if (thread_.joinable()) thread_.join();
  connection_.reset();
  running_.store(false);
}

void ExpertDomainServer::run(std::shared_ptr<transport::Connection> conn) {
  auto send_frame = [&](transport::Frame f) {
    {
      std::lock_guard<std::mutex> lk(impl_->mu);
      impl_->metrics.bytes_sent += f.payload.size();
    }
    return conn->send(f);
  };
  auto reply_error = [&](const Status& st, std::uint64_t corr) {
    ExpertError e;
    e.code = st.code();
    e.message = st.message().substr(0, config_.limits.max_error_message);
    return send_frame(to_frame(e, corr));
  };
  auto count_rejected = [&] {
    std::lock_guard<std::mutex> lk(impl_->mu);
    ++impl_->metrics.rejected;
  };
  while (!stop_.load()) {
    auto frame = conn->receive(100ms);
    if (!frame.is_ok()) {
      if (frame.status().code() == ErrorCode::kDeadlineExceeded) continue;
      break;  // closed or protocol violation: the transport already dropped the connection
    }
    {
      std::lock_guard<std::mutex> lk(impl_->mu);
      impl_->metrics.bytes_received += frame->payload.size();
    }
    if (frame->type != kMsgExpertBatch) {
      count_rejected();
      if (!reply_error(make_error(ErrorCode::kProtocolError, "unexpected message type"), frame->correlation).is_ok()) break;
      continue;
    }
    auto batch = decode_batch(frame->payload, config_.limits);
    if (!batch.is_ok()) {
      count_rejected();
      if (!reply_error(batch.status(), frame->correlation).is_ok()) break;
      continue;
    }
    bool die = false;
    {
      std::lock_guard<std::mutex> lk(impl_->mu);
      die = config_.die_after_batches != 0 && impl_->served >= config_.die_after_batches;
      if (!die) ++impl_->served;
    }
    if (die) {
      conn->close();  // test fault: vanish without a reply
      break;
    }
    auto result = execute(*batch);
    const Status sent = result.is_ok() ? send_frame(to_frame(*result, frame->correlation))
                                       : reply_error(result.status(), frame->correlation);
    if (!sent.is_ok()) break;
  }
  running_.store(false);
}

Result<std::unique_ptr<objects::InMemoryResolver>> provision_expert_objects(
    const objects::CanonicalModelStore& father_store, const std::vector<std::uint32_t>& owned_experts,
    std::uint32_t first_layer, std::uint32_t end_layer) {
  auto r = std::make_unique<objects::InMemoryResolver>(father_store.manifest());
  for (std::uint32_t L = first_layer; L < end_layer; ++L)
    for (std::uint32_t e : owned_experts) {
      const std::string name = objects::expert_object_name(L, e);
      CLM_ASSIGN_OR_RETURN(Bytes bytes, father_store.read_object_bytes(name));
      CLM_RETURN_IF_ERROR(r->add(name, std::move(bytes)));
    }
  return r;
}

}  // namespace clusterlm::expert_domains
