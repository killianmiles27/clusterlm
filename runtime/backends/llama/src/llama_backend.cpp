#include "clusterlm/backends/llama_backend.hpp"

#include <algorithm>
#include <mutex>
#include <utility>

#include "clusterlm/common/clock.hpp"
#include "clusterlm/windows/window_ledger.hpp"
#include "llama_engine.hpp"

#ifndef CLUSTERLM_LLAMA_PIN
#define CLUSTERLM_LLAMA_PIN "unknown"  // set by CMake from third_party/upstream.json
#endif

namespace clusterlm::backends {
namespace {

using namespace domain;

Status precondition(std::string m) { return make_error(ErrorCode::kFailedPrecondition, std::move(m)); }
Status unimplemented(std::string m) { return make_error(ErrorCode::kUnimplemented, std::move(m)); }

// The "activations" a llama prefix domain hands to its tail are an opaque Father-local handle: one 3-float zero
// record (hc = 1, H = 1) per position. No activation values exist outside llama.cpp, and nothing here is ever
// serialized: a llama plan has no remote stage (ADR 0300).
BoundaryLayout handle_layout() {
  BoundaryLayout l;
  l.residual_streams = 1;
  l.hidden_size = 1;
  return l;
}

// State shared by the two Father domains of one model.
struct Shared {
  objects::ModelManifest manifest;
  LlamaBackendOptions options;
  LlamaModelReport report;
  std::shared_ptr<LlamaEngine> engine;
  std::vector<std::filesystem::path> files;
  DomainSpec prefix_spec;
};

class LlamaDomainBase : public LlamaLocalDomain {
 public:
  LlamaDomainBase(std::shared_ptr<Shared> shared, DomainSpec spec) : shared_(std::move(shared)), spec_(spec) {}

  const DomainSpec& spec() const override { return spec_; }
  BoundaryLayout boundary() const override { return handle_layout(); }
  const LlamaModelReport& model_report() const override { return shared_->report; }
  std::string llama_pin() const override { return CLUSTERLM_LLAMA_PIN; }
  LlamaDomainStats llama_stats() const override { return shared_->engine->stats(); }

  Result<StageActivations> run_window(const WindowRequest&, const StageActivations&) override {
    return unimplemented(
        "llama backend: the Fast tier runs the whole model on Father; there is no activation-boundary middle stage "
        "(docs/backends/llama-local.md)");
  }

  Result<DomainRequirements> describe_requirements() const override {
    const auto& g = shared_->manifest.geometry;
    const auto& o = shared_->options;
    DomainRequirements r;
    if (spec_.role == StageRole::kPrefix) {
      // Estimates from the manifest; llama.cpp reports the exact figures in its own load log. No required
      // objects: the backend maps Father's GGUF files directly instead of resolving objects.
      std::uint64_t layer_bytes = 0, other_bytes = 0;
      for (const auto& obj : shared_->manifest.objects) (obj.layer ? layer_bytes : other_bytes) += obj.byte_size;
      const std::uint64_t all = layer_bytes + other_bytes;
      if (o.n_gpu_layers == 0) {
        r.cpu_weight_bytes = all;
      } else if (o.n_gpu_layers < 0 || static_cast<std::uint32_t>(o.n_gpu_layers) >= g.n_layers) {
        r.gpu_weight_bytes = all;  // llama.cpp keeps the (mmapped) embedding table on the host; counted as GPU here
      } else {
        r.gpu_weight_bytes = layer_bytes / g.n_layers * static_cast<std::uint64_t>(o.n_gpu_layers);
        r.cpu_weight_bytes = all - r.gpu_weight_bytes;
      }
      std::uint64_t attn_layers = 0;
      for (auto k : g.layer_kinds) attn_layers += k == objects::LayerKind::kFullAttention ? 1 : 0;
      // FP16 K and V per attention layer. Recurrent layers keep a small fixed state that is not included.
      r.state_bytes = attn_layers * 2 * std::uint64_t{g.n_kv_heads} * g.head_dim * spec_.max_context * 2 *
                      std::max<std::uint32_t>(spec_.max_sessions, 1);
      r.window_bytes = std::uint64_t{spec_.max_window} * g.vocab_size * sizeof(float);  // per-window logits
      const std::uint32_t local = spec_.max_local_batch ? spec_.max_local_batch : spec_.max_window;
      r.scratch_bytes = std::uint64_t{std::min<std::uint32_t>(local, 512)} * (g.vocab_size + 8ull * g.hidden_size) * sizeof(float);
    }
    return r;
  }

  Status abort_session(Epoch epoch, SessionId session) override {
    CLM_RETURN_IF_ERROR(ledger_.abort_session(epoch, session));
    on_session_closed(session);
    return Status::ok();
  }

  DomainMetrics read_metrics() const override {
    DomainMetrics m = metrics_;
    m.stale_rejections = ledger_.stale_rejections();
    return m;
  }

 protected:
  virtual void on_session_closed(SessionId) {}

  // Common admission checks for a window; on failure the ledger entry is dropped.
  Status admit(const WindowRequest& req) {
    if (!prepared_) return precondition("domain not prepared");
    CLM_RETURN_IF_ERROR(ledger_.begin_window(req));
    Status st = Status::ok();
    if (req.positions > spec_.max_window)
      st = make_error(ErrorCode::kResourceExhausted, "window exceeds max_window");
    else if (req.base_position + req.positions > spec_.max_context)
      st = make_error(ErrorCode::kResourceExhausted, "window exceeds max_context");
    if (!st.is_ok()) ledger_.fail_window(req.session);
    return st;
  }

  std::shared_ptr<Shared> shared_;
  DomainSpec spec_;
  bool prepared_ = false;
  windows::WindowLedger ledger_;
  DomainMetrics metrics_;
};

class PrefixDomain final : public LlamaDomainBase {
 public:
  using LlamaDomainBase::LlamaDomainBase;

  Status prepare(const objects::ObjectResolver&) override {
    if (prepared_) return precondition("domain already prepared; release() first");
    CLM_RETURN_IF_ERROR(shared_->engine->load());
    prepared_ = true;
    return Status::ok();
  }

  Status open_session(Epoch epoch, SessionId session) override {
    if (!prepared_) return precondition("domain not prepared");
    CLM_RETURN_IF_ERROR(ledger_.open_session(epoch, session));
    Status st = shared_->engine->open_session(session);
    if (!st.is_ok()) (void)ledger_.abort_session(epoch, session);
    return st;
  }

  Result<StageActivations> run_prefix(const WindowRequest& req, std::span<const std::int32_t> tokens) override {
    CLM_RETURN_IF_ERROR(admit(req));
    if (tokens.size() != req.positions) {
      ledger_.fail_window(req.session);
      return make_error(ErrorCode::kInvalidArgument, "tokens.size() != request.positions");
    }
    Stopwatch sw;
    Status st = shared_->engine->run_window(req.session, req.window, req.base_position, tokens);
    if (!st.is_ok()) {
      ledger_.fail_window(req.session);
      return st;
    }
    ++metrics_.windows_run;
    metrics_.compute_ns_total += sw.elapsed_ns();
    StageActivations out;
    out.layout = handle_layout();
    out.first_position = req.base_position;
    out.positions = req.positions;
    out.data.assign(std::size_t{req.positions} * out.layout.floats_per_position(), 0.0f);
    return out;
  }

  Result<Logits> run_tail(const WindowRequest&, const StageActivations&) override {
    return precondition("run_tail is only valid on the tail domain");
  }

  Result<CommitAck> commit_window(const CommitRequest& request) override {
    if (!prepared_) return precondition("domain not prepared");
    CLM_ASSIGN_OR_RETURN(windows::WindowLedger::CommitDecision d, ledger_.begin_commit(request));
    if (d.kind == windows::WindowLedger::CommitDecision::Kind::kReplay) return d.ack;
    Status st = shared_->engine->commit(request.session, request.window, request.accepted);
    if (!st.is_ok()) return st;
    ++metrics_.windows_committed;
    metrics_.positions_committed += request.accepted;
    return ledger_.finish_commit(request);
  }

  Result<WindowAbortAck> abort_window(Epoch epoch, SessionId session, WindowId window) override {
    CLM_ASSIGN_OR_RETURN(WindowAbortAck ack, ledger_.abort_window(epoch, session, window));
    CLM_RETURN_IF_ERROR(shared_->engine->abort_window(session, window));
    ++metrics_.windows_aborted;
    return ack;
  }

  Status release() override {
    ledger_.clear();
    shared_->engine->release();
    prepared_ = false;
    return Status::ok();
  }

  DomainMetrics read_metrics() const override {
    DomainMetrics m = LlamaDomainBase::read_metrics();
    m.resident_weight_bytes = shared_->engine->model_bytes();
    m.state_bytes = describe_state_bytes();
    return m;
  }

 private:
  std::uint64_t describe_state_bytes() const {
    if (shared_->engine->open_sessions() == 0) return 0;
    auto r = describe_requirements();
    return r.is_ok() ? r.value().state_bytes * shared_->engine->open_sessions() /
                           std::max<std::uint32_t>(spec_.max_sessions, 1)
                     : 0;
  }

  void on_session_closed(SessionId session) override { shared_->engine->close_session(session); }
};

class TailDomain final : public LlamaDomainBase {
 public:
  using LlamaDomainBase::LlamaDomainBase;

  Status prepare(const objects::ObjectResolver&) override {
    if (prepared_) return precondition("domain already prepared; release() first");
    if (!shared_->engine->loaded()) return precondition("the llama tail domain is prepared after its prefix domain");
    prepared_ = true;
    return Status::ok();
  }

  Status open_session(Epoch epoch, SessionId session) override {
    if (!prepared_) return precondition("domain not prepared");
    return ledger_.open_session(epoch, session);
  }

  Result<StageActivations> run_prefix(const WindowRequest&, std::span<const std::int32_t>) override {
    return precondition("run_prefix is only valid on the prefix domain");
  }

  Result<Logits> run_tail(const WindowRequest& req, const StageActivations& input) override {
    CLM_RETURN_IF_ERROR(admit(req));
    auto reject = [&](Status st) -> Result<Logits> {
      ledger_.fail_window(req.session);
      return st;
    };
    if (input.layout != handle_layout() || input.positions != req.positions || input.first_position != req.base_position)
      return reject(make_error(ErrorCode::kInvalidArgument, "tail input is not the prefix domain's handle for this window"));
    Stopwatch sw;
    auto logits = shared_->engine->take_logits(req.session, req.window, req.positions);
    if (!logits.is_ok()) return reject(logits.status());
    ++metrics_.windows_run;
    metrics_.compute_ns_total += sw.elapsed_ns();
    return std::move(logits).value();
  }

  Result<CommitAck> commit_window(const CommitRequest& request) override {
    if (!prepared_) return precondition("domain not prepared");
    CLM_ASSIGN_OR_RETURN(windows::WindowLedger::CommitDecision d, ledger_.begin_commit(request));
    if (d.kind == windows::WindowLedger::CommitDecision::Kind::kReplay) return d.ack;
    ++metrics_.windows_committed;
    metrics_.positions_committed += request.accepted;
    return ledger_.finish_commit(request);  // the KV change was applied once, by the prefix domain
  }

  Result<WindowAbortAck> abort_window(Epoch epoch, SessionId session, WindowId window) override {
    CLM_ASSIGN_OR_RETURN(WindowAbortAck ack, ledger_.abort_window(epoch, session, window));
    ++metrics_.windows_aborted;
    return ack;
  }

  Status release() override {
    ledger_.clear();
    prepared_ = false;
    return Status::ok();
  }
};

class LlamaBackend final : public BackendAdapter {
 public:
  explicit LlamaBackend(LlamaBackendOptions options) : options_(std::move(options)) {}

  BackendInfo info() const override {
    BackendInfo i;
    i.name = "llama";
    i.build_hash = std::string("llama.cpp@") + CLUSTERLM_LLAMA_PIN;
#ifdef CLUSTERLM_LLAMA_CUDA
    i.build_hash += "+cuda";
    i.supports_gpu = true;
#endif
    // The CPU build of the pinned llama.cpp runs here; GPU execution is only claimed by a CUDA build, and its
    // availability on a given machine is a hardware question (HQ-TIER-01), not something this build can assert.
    i.hardware_available = true;
    return i;
  }

  Result<std::unique_ptr<ExecutionDomain>> create_domain(const objects::ModelManifest& manifest,
                                                         const DomainSpec& spec) override {
    CLM_RETURN_IF_ERROR(manifest.validate());
    if (spec.role == StageRole::kMiddle)
      return unimplemented(
          "llama backend: the Fast tier is Father-only; middle stages cannot be hosted (docs/backends/llama-local.md)");
    const std::uint32_t L = manifest.geometry.n_layers;
    if (spec.max_context == 0 || spec.max_window == 0 || spec.max_sessions == 0 || spec.max_window > spec.max_context)
      return make_error(ErrorCode::kInvalidArgument, "domain spec needs max_context >= max_window >= 1 and max_sessions >= 1");
    std::lock_guard<std::mutex> lk(mu_);
    if (spec.role == StageRole::kPrefix) {
      if (spec.layers.begin != 0 || spec.layers.end != L)
        return unimplemented("llama backend: the prefix domain must cover every layer [0, " + std::to_string(L) +
                             "); partial layer ranges are not supported");
      auto shared = std::make_shared<Shared>();
      shared->manifest = manifest;
      shared->options = options_;
      shared->prefix_spec = spec;
      for (const auto& shard : manifest.shards) shared->files.push_back(options_.model_dir / shard.file_name);
      for (std::size_t i = 0; i < shared->files.size(); ++i) {
        std::error_code ec;
        const auto size = std::filesystem::file_size(shared->files[i], ec);
        if (ec) return make_error(ErrorCode::kNotFound, "model file missing: " + manifest.shards[i].file_name);
        if (size != manifest.shards[i].byte_size)
          return make_error(ErrorCode::kDataLoss, "model file '" + manifest.shards[i].file_name + "' has a different size than the manifest records");
      }
      // The report reads the GGUF directories only (no tensor data), so it works before prepare().
      auto rebuilt = build_llama_manifest(shared->files, {}, &shared->report);
      if (!rebuilt.is_ok()) return rebuilt.status();
      LlamaEngine::Params p;
      p.files = shared->files;
      p.options = options_;
      p.spec = spec;
      p.manifest_vocab = manifest.geometry.vocab_size;
      shared->engine = std::make_shared<LlamaEngine>(std::move(p));
      pending_ = shared;
      return std::unique_ptr<ExecutionDomain>(std::make_unique<PrefixDomain>(std::move(shared), spec));
    }
    // kTail: head-only, paired with the prefix domain created just before it.
    if (!spec.layers.empty() || spec.layers.begin != L)
      return unimplemented("llama backend: the tail domain is head-only ([L, L)); the whole model runs in the prefix domain");
    auto shared = pending_.lock();
    if (!shared) return precondition("llama tail domain requires its prefix domain to be created first");
    if (shared->manifest.root_hash() != manifest.root_hash())
      return precondition("llama tail domain was created for a different model than its prefix domain");
    return std::unique_ptr<ExecutionDomain>(std::make_unique<TailDomain>(std::move(shared), spec));
  }

 private:
  LlamaBackendOptions options_;
  std::mutex mu_;
  std::weak_ptr<Shared> pending_;
};

}  // namespace

std::unique_ptr<domain::BackendAdapter> make_llama_backend(const LlamaBackendOptions& options) {
  return std::make_unique<LlamaBackend>(options);
}

}  // namespace clusterlm::backends
