#include "llama_engine.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>
#include <span>
#include <thread>

#include "clusterlm/common/clock.hpp"
#include "clusterlm/common/log.hpp"
#include "llama.h"

namespace clusterlm::backends {
namespace {

std::atomic<std::int64_t> g_models{0};
std::atomic<std::int64_t> g_contexts{0};

// llama.cpp prints to stderr by default. Forward errors (and warnings at debug level) to the project logger;
// llama.cpp's messages describe the model and the runtime, never prompts or activations.
void forward_llama_log(ggml_log_level level, const char* text, void*) {
  if (text == nullptr || (level != GGML_LOG_LEVEL_ERROR && level != GGML_LOG_LEVEL_WARN)) return;
  std::string msg(text);
  while (!msg.empty() && (msg.back() == '\n' || msg.back() == '\r')) msg.pop_back();
  if (msg.size() > 240) msg.resize(240);
  if (level == GGML_LOG_LEVEL_ERROR)
    log::error("llama_log", {{"msg", msg}});
  else
    log::debug("llama_log", {{"msg", msg}});  // llama.cpp warns about benign model properties on every load
}

void ensure_backend() {
  static std::once_flag once;
  std::call_once(once, [] {
    llama_log_set(forward_llama_log, nullptr);
    llama_backend_init();  // process lifetime; llama_backend_free is deliberately never called
  });
}

Status internal(std::string m) { return make_error(ErrorCode::kInternal, std::move(m)); }
Status precondition(std::string m) { return make_error(ErrorCode::kFailedPrecondition, std::move(m)); }

}  // namespace

LlamaLiveCounts llama_live_counts() { return {g_models.load(), g_contexts.load()}; }

struct LlamaEngine::Impl {
  struct Window {
    WindowId id;
    std::uint64_t base = 0;
    std::uint32_t q = 0;
  };
  struct Stash {
    WindowId window;
    domain::Logits logits;
  };
  struct Session {
    llama_seq_id seq = 0;
    std::vector<std::int32_t> history;  // committed tokens, plus the outstanding window's while it is open
    std::uint64_t committed = 0;
    std::optional<Window> window;
    std::optional<Stash> stash;
  };

  Params params;
  mutable std::mutex mu;
  llama_model* model = nullptr;
  llama_context* ctx = nullptr;
  llama_batch batch{};
  bool batch_allocated = false;
  std::uint32_t n_batch = 0;
  std::uint32_t n_vocab = 0;
  std::uint64_t model_size = 0;
  std::vector<bool> slot_used;
  std::unordered_map<SessionId, Session> sessions;
  LlamaDomainStats stats;
  std::uint64_t windows_run = 0;
  std::uint64_t compute_ns = 0;

  llama_memory_t memory() const { return llama_get_memory(ctx); }

  void release_unlocked() {
    sessions.clear();
    slot_used.clear();
    if (batch_allocated) {
      llama_batch_free(batch);
      batch = llama_batch{};
      batch_allocated = false;
    }
    if (ctx != nullptr) {
      llama_free(ctx);
      ctx = nullptr;
      --g_contexts;
    }
    if (model != nullptr) {
      llama_model_free(model);
      model = nullptr;
      --g_models;
    }
    model_size = 0;
    n_vocab = 0;
  }

  // Feeds `tokens` at positions [base, base+n) of `seq`. `logit_rows` selects which rows get logits: all of them,
  // only the last, or none (the last row is still marked: llama.cpp needs at least one output per decode).
  Status decode_chunk(llama_seq_id seq, std::uint64_t base, std::span<const std::int32_t> tokens, bool all_logits) {
    const auto n = static_cast<std::int32_t>(tokens.size());
    batch.n_tokens = n;
    for (std::int32_t i = 0; i < n; ++i) {
      batch.token[i] = tokens[static_cast<std::size_t>(i)];
      batch.pos[i] = static_cast<llama_pos>(base + static_cast<std::uint64_t>(i));
      batch.n_seq_id[i] = 1;
      batch.seq_id[i][0] = seq;
      batch.logits[i] = all_logits || i + 1 == n ? 1 : 0;
    }
    ++stats.decode_calls;
    stats.positions_decoded += tokens.size();
    const std::int32_t rc = llama_decode(ctx, batch);
    if (rc == 0) return Status::ok();
    if (rc == 1) return make_error(ErrorCode::kResourceExhausted, "llama.cpp found no KV slot for the window");
    if (rc == 2) return make_error(ErrorCode::kCancelled, "llama.cpp decode aborted");
    return internal("llama_decode failed with code " + std::to_string(rc));
  }

  // Removes everything at or after `keep` from the session's KV. A refusal (recurrent state that cannot roll
  // back to that point) is repaired by clearing the sequence and re-decoding the committed prefix.
  Status trim(Session& s, std::uint64_t keep) {
    s.history.resize(static_cast<std::size_t>(keep));
    const llama_pos pmax = llama_memory_seq_pos_max(memory(), s.seq);
    if (pmax < 0 ? keep == 0 : static_cast<std::uint64_t>(pmax) + 1 == keep) return Status::ok();  // nothing to remove
    ++stats.kv_trims;
    if (llama_memory_seq_rm(memory(), s.seq, static_cast<llama_pos>(keep), -1)) return Status::ok();
    ++stats.recompute_fallbacks;
    (void)llama_memory_seq_rm(memory(), s.seq, -1, -1);  // removing a whole sequence never fails
    for (std::uint64_t at = 0; at < keep;) {
      const std::size_t n = std::min<std::uint64_t>(n_batch, keep - at);
      CLM_RETURN_IF_ERROR(decode_chunk(s.seq, at, std::span<const std::int32_t>(s.history).subspan(at, n), false));
      at += n;
    }
    return Status::ok();
  }
};

LlamaEngine::LlamaEngine(Params params) : impl_(std::make_unique<Impl>()) { impl_->params = std::move(params); }

LlamaEngine::~LlamaEngine() { release(); }

Status LlamaEngine::load() {
  Impl& im = *impl_;
  std::lock_guard<std::mutex> lk(im.mu);
  if (im.model != nullptr) return Status::ok();
  if (im.params.files.empty()) return make_error(ErrorCode::kInvalidArgument, "llama engine: no model files");
  for (const auto& f : im.params.files) {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(f, ec))
      return make_error(ErrorCode::kNotFound, "llama engine: model file missing: " + f.filename().string());
  }
  ensure_backend();

  llama_model_params mp = llama_model_default_params();
  mp.n_gpu_layers = im.params.options.n_gpu_layers;
  mp.load_mode = im.params.options.use_mmap ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE;
  std::vector<std::string> paths;
  for (const auto& f : im.params.files) paths.push_back(f.string());
  if (paths.size() == 1) {
    im.model = llama_model_load_from_file(paths[0].c_str(), mp);
  } else {
    std::vector<const char*> c;
    for (const auto& p : paths) c.push_back(p.c_str());
    im.model = llama_model_load_from_splits(c.data(), c.size(), mp);
  }
  if (im.model == nullptr) return make_error(ErrorCode::kUnavailable, "llama.cpp could not load the model");
  ++g_models;

  const llama_vocab* vocab = llama_model_get_vocab(im.model);
  im.n_vocab = static_cast<std::uint32_t>(llama_vocab_n_tokens(vocab));
  im.model_size = llama_model_size(im.model);
  if (im.params.manifest_vocab != 0 && im.n_vocab != im.params.manifest_vocab) {
    const auto got = im.n_vocab;
    im.release_unlocked();
    return precondition("the loaded model has vocabulary " + std::to_string(got) + " but the manifest says " +
                        std::to_string(im.params.manifest_vocab));
  }

  const domain::DomainSpec& spec = im.params.spec;
  const std::uint32_t max_window = std::max<std::uint32_t>(spec.max_window, 1);
  const std::uint32_t sessions = std::max<std::uint32_t>(spec.max_sessions, 1);
  llama_context_params cp = llama_context_default_params();
  cp.n_ctx = spec.max_context * sessions;
  cp.n_batch = max_window;
  const std::uint32_t local = spec.max_local_batch != 0 ? spec.max_local_batch : max_window;
  cp.n_ubatch = std::min<std::uint32_t>(std::min<std::uint32_t>(local, max_window), 512);
  cp.n_seq_max = sessions;
  cp.n_rs_seq = max_window;  // recurrent-state rollback depth; llama.cpp clamps it to 0 for architectures without
  cp.n_threads = im.params.options.n_threads > 0
                     ? im.params.options.n_threads
                     : static_cast<std::int32_t>(std::max(1u, std::thread::hardware_concurrency() / 2));
  cp.n_threads_batch = im.params.options.n_threads_batch > 0 ? im.params.options.n_threads_batch : cp.n_threads;
  cp.offload_kqv = im.params.options.n_gpu_layers != 0;
  cp.kv_unified = true;
  cp.no_perf = true;
  im.ctx = llama_init_from_model(im.model, cp);
  if (im.ctx == nullptr) {
    im.release_unlocked();
    return make_error(ErrorCode::kResourceExhausted, "llama.cpp could not create a context for max_context " +
                                                          std::to_string(spec.max_context));
  }
  ++g_contexts;
  im.n_batch = cp.n_batch;
  im.batch = llama_batch_init(static_cast<std::int32_t>(im.n_batch), 0, 1);
  im.batch_allocated = true;
  im.slot_used.assign(sessions, false);
  return Status::ok();
}

void LlamaEngine::release() {
  std::lock_guard<std::mutex> lk(impl_->mu);
  impl_->release_unlocked();
}

bool LlamaEngine::loaded() const {
  std::lock_guard<std::mutex> lk(impl_->mu);
  return impl_->ctx != nullptr;
}

std::uint64_t LlamaEngine::model_bytes() const {
  std::lock_guard<std::mutex> lk(impl_->mu);
  return impl_->model_size;
}

std::uint32_t LlamaEngine::vocab() const {
  std::lock_guard<std::mutex> lk(impl_->mu);
  return impl_->n_vocab;
}

Status LlamaEngine::open_session(SessionId session) {
  Impl& im = *impl_;
  std::lock_guard<std::mutex> lk(im.mu);
  if (im.ctx == nullptr) return precondition("llama engine is not loaded");
  // Re-opening a session (a newer epoch re-establishes it) starts from empty state.
  if (auto it = im.sessions.find(session); it != im.sessions.end()) {
    (void)llama_memory_seq_rm(im.memory(), it->second.seq, -1, -1);
    im.slot_used[static_cast<std::size_t>(it->second.seq)] = false;
    im.sessions.erase(it);
  }
  const auto free_slot = std::find(im.slot_used.begin(), im.slot_used.end(), false);
  if (free_slot == im.slot_used.end()) return make_error(ErrorCode::kResourceExhausted, "max_sessions reached");
  *free_slot = true;
  Impl::Session s;
  s.seq = static_cast<llama_seq_id>(free_slot - im.slot_used.begin());
  (void)llama_memory_seq_rm(im.memory(), s.seq, -1, -1);
  im.sessions.emplace(session, std::move(s));
  return Status::ok();
}

void LlamaEngine::close_session(SessionId session) {
  Impl& im = *impl_;
  std::lock_guard<std::mutex> lk(im.mu);
  auto it = im.sessions.find(session);
  if (it == im.sessions.end()) return;
  if (im.ctx != nullptr) (void)llama_memory_seq_rm(im.memory(), it->second.seq, -1, -1);
  if (static_cast<std::size_t>(it->second.seq) < im.slot_used.size()) im.slot_used[static_cast<std::size_t>(it->second.seq)] = false;
  im.sessions.erase(it);
}

std::size_t LlamaEngine::open_sessions() const {
  std::lock_guard<std::mutex> lk(impl_->mu);
  return impl_->sessions.size();
}

Status LlamaEngine::run_window(SessionId session, WindowId window, std::uint64_t base,
                               std::span<const std::int32_t> tokens) {
  Impl& im = *impl_;
  std::lock_guard<std::mutex> lk(im.mu);
  if (im.ctx == nullptr) return precondition("llama engine is not loaded");
  auto it = im.sessions.find(session);
  if (it == im.sessions.end()) return make_error(ErrorCode::kNotFound, "session not open in the llama engine");
  Impl::Session& s = it->second;
  if (s.window) return precondition("a window is already outstanding in the llama engine");
  if (s.committed != base) return precondition("llama engine session is at a different committed position");
  if (tokens.empty() || tokens.size() > im.n_batch) return make_error(ErrorCode::kInvalidArgument, "window size outside [1, max_window]");
  for (std::int32_t t : tokens)
    if (t < 0 || static_cast<std::uint32_t>(t) >= im.n_vocab) return make_error(ErrorCode::kInvalidArgument, "token outside the vocabulary");
  const llama_pos pmax = llama_memory_seq_pos_max(im.memory(), s.seq);
  if (static_cast<std::uint64_t>(pmax + 1) != base) return internal("llama KV position disagrees with the committed position");

  const std::uint32_t q = static_cast<std::uint32_t>(tokens.size());
  const bool all = q <= im.params.options.full_logits_max_positions;
  Stopwatch sw;
  s.history.insert(s.history.end(), tokens.begin(), tokens.end());
  Status st = im.decode_chunk(s.seq, base, tokens, all);
  if (!st.is_ok()) {
    // Whatever part of the window llama.cpp processed must not stay in the cache.
    (void)llama_memory_seq_rm(im.memory(), s.seq, static_cast<llama_pos>(base), -1);
    s.history.resize(static_cast<std::size_t>(base));
    return st;
  }
  Impl::Stash stash;
  stash.window = window;
  stash.logits.vocab = im.n_vocab;
  stash.logits.positions = all ? q : 1;
  stash.logits.data.resize(std::size_t{stash.logits.positions} * im.n_vocab);
  for (std::uint32_t i = 0; i < stash.logits.positions; ++i) {
    const float* row = llama_get_logits_ith(im.ctx, static_cast<std::int32_t>(all ? i : q - 1));
    if (row == nullptr) {
      (void)llama_memory_seq_rm(im.memory(), s.seq, static_cast<llama_pos>(base), -1);
      s.history.resize(static_cast<std::size_t>(base));
      return internal("llama.cpp returned no logits for a requested position");
    }
    std::memcpy(stash.logits.data.data() + std::size_t{i} * im.n_vocab, row, std::size_t{im.n_vocab} * sizeof(float));
  }
  s.stash = std::move(stash);
  s.window = Impl::Window{window, base, q};
  ++im.windows_run;
  im.compute_ns += sw.elapsed_ns();
  return Status::ok();
}

Status LlamaEngine::commit(SessionId session, WindowId window, std::uint32_t accepted) {
  Impl& im = *impl_;
  std::lock_guard<std::mutex> lk(im.mu);
  if (im.ctx == nullptr) return precondition("llama engine is not loaded");
  auto it = im.sessions.find(session);
  if (it == im.sessions.end()) return make_error(ErrorCode::kNotFound, "session not open in the llama engine");
  Impl::Session& s = it->second;
  if (!s.window || s.window->id != window) return precondition("no matching outstanding window in the llama engine");
  if (accepted == 0 || accepted > s.window->q) return make_error(ErrorCode::kInvalidArgument, "accepted outside [1, q]");
  const std::uint64_t keep = s.window->base + accepted;
  s.window.reset();
  Status st = im.trim(s, keep);
  s.committed = keep;
  return st;
}

Status LlamaEngine::abort_window(SessionId session, WindowId window) {
  Impl& im = *impl_;
  std::lock_guard<std::mutex> lk(im.mu);
  if (im.ctx == nullptr) return Status::ok();
  auto it = im.sessions.find(session);
  if (it == im.sessions.end()) return Status::ok();
  Impl::Session& s = it->second;
  if (!s.window || s.window->id != window) return Status::ok();
  const std::uint64_t base = s.window->base;
  s.window.reset();
  s.stash.reset();
  return im.trim(s, base);
}

Result<domain::Logits> LlamaEngine::take_logits(SessionId session, WindowId window, std::uint32_t positions) {
  Impl& im = *impl_;
  std::lock_guard<std::mutex> lk(im.mu);
  auto it = im.sessions.find(session);
  if (it == im.sessions.end()) return make_error(ErrorCode::kNotFound, "session not open in the llama engine");
  Impl::Session& s = it->second;
  if (!s.stash || s.stash->window != window)
    return make_error(ErrorCode::kNotFound, "no logits for this window (the prefix domain has not run it, or the tail already took them)");
  const bool prefill_chunk = s.stash->logits.positions == 1 && positions > 1;
  if (s.stash->logits.positions != positions && !prefill_chunk)
    return make_error(ErrorCode::kInvalidArgument, "tail window size differs from the prefix window size");
  domain::Logits out = std::move(s.stash->logits);
  s.stash.reset();
  return out;
}

LlamaDomainStats LlamaEngine::stats() const {
  std::lock_guard<std::mutex> lk(impl_->mu);
  return impl_->stats;
}
std::uint64_t LlamaEngine::windows_run() const {
  std::lock_guard<std::mutex> lk(impl_->mu);
  return impl_->windows_run;
}
std::uint64_t LlamaEngine::compute_ns() const {
  std::lock_guard<std::mutex> lk(impl_->mu);
  return impl_->compute_ns;
}

}  // namespace clusterlm::backends
