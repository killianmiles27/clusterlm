#pragma once
// Shared helpers for the llama backend tests: a tiny GGUF model directory with its manifest, and llama.cpp's own
// straightforward decode (the reference the backend's logits and tokens are compared against).
#include <doctest/doctest.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <memory>
#include <vector>

#include "clusterlm/backends/llama_backend.hpp"
#include "clusterlm/backends/llama_manifest.hpp"
#include "clusterlm/backends/llama_tiny_model.hpp"
#include "clusterlm/coordinator/coordinator.hpp"
#include "clusterlm/objects/canonical_store.hpp"
#include "llama.h"

namespace clusterlm::llamatest {

inline std::filesystem::path unique_dir(const std::string& name) {
  static std::atomic<int> counter{0};
  auto p = std::filesystem::temp_directory_path() /
           ("clm-llama-" + name + "-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-" +
            std::to_string(counter++));
  std::filesystem::create_directories(p);
  return p;
}

// A temporary model directory holding one tiny GGUF and the manifest.json the Coordinator opens.
struct TinyModel {
  std::filesystem::path dir;
  std::filesystem::path gguf;
  objects::ModelManifest manifest;

  explicit TinyModel(const std::string& name, backends::TinyLlamaSpec spec = {}) : dir(unique_dir(name)) {
    gguf = dir / "tiny.gguf";
    auto st = backends::write_tiny_llama_gguf(gguf, spec);
    REQUIRE_MESSAGE(st.is_ok(), st.to_string());
    auto m = backends::write_llama_model_dir({gguf});
    REQUIRE_MESSAGE(m.is_ok(), m.status().to_string());
    manifest = m.value();
  }
  ~TinyModel() {
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
  }
  std::uint32_t layers() const { return manifest.geometry.n_layers; }
  std::uint32_t vocab() const { return manifest.geometry.vocab_size; }
  // "0-L@father,L-L@father": the prefix domain holds the model, the tail is head-only.
  std::string plan_text() const { return "0-" + std::to_string(layers()) + "@father," + std::to_string(layers()) + "-" + std::to_string(layers()) + "@father"; }
};

inline backends::LlamaBackendOptions options_for(const TinyModel& m, std::int32_t threads = 2) {
  backends::LlamaBackendOptions o;
  o.model_dir = m.dir;
  o.n_threads = threads;
  return o;
}

inline std::shared_ptr<domain::BackendAdapter> make_backend(const TinyModel& m, std::int32_t threads = 2) {
  return std::shared_ptr<domain::BackendAdapter>(backends::make_llama_backend(options_for(m, threads)));
}

inline std::unique_ptr<coordinator::Coordinator> make_father(const TinyModel& m, std::uint32_t max_context = 256,
                                                             std::uint32_t max_window = 128) {
  coordinator::CoordinatorConfig cfg;
  cfg.model_dir = m.dir;
  cfg.security.mode = transport::SecurityConfig::Mode::kInsecureLoopbackOnly;
  cfg.backend = make_backend(m);
  auto c = coordinator::Coordinator::create(std::move(cfg));
  REQUIRE_MESSAGE(c.is_ok(), c.status().to_string());
  auto plan = coordinator::ClusterPlan::parse(m.plan_text(), m.layers());
  REQUIRE_MESSAGE(plan.is_ok(), plan.status().to_string());
  plan.value().max_context = max_context;
  plan.value().max_window = max_window;
  auto prep = c.value()->prepare(plan.value());
  REQUIRE_MESSAGE(prep.is_ok(), prep.status().to_string());
  return std::move(c).value();
}

inline std::vector<std::int32_t> prompt_of(std::size_t n, std::uint32_t vocab, std::uint32_t salt = 0) {
  std::vector<std::int32_t> p;
  for (std::size_t i = 0; i < n; ++i) p.push_back(static_cast<std::int32_t>((i * 7919u + 13u + salt * 31u) % vocab));
  return p;
}

// llama.cpp used directly, without ClusterLM: one context, one sequence, the whole token list in one batch.
class RefLlama {
 public:
  RefLlama(const std::filesystem::path& gguf, std::uint32_t n_ctx, std::int32_t threads = 2) {
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    model_ = llama_model_load_from_file(gguf.string().c_str(), mp);
    REQUIRE(model_ != nullptr);
    vocab_ = static_cast<std::uint32_t>(llama_vocab_n_tokens(llama_model_get_vocab(model_)));
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = n_ctx;
    cp.n_ubatch = std::min<std::uint32_t>(n_ctx, 512);
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.no_perf = true;
    ctx_ = llama_init_from_model(model_, cp);
    REQUIRE(ctx_ != nullptr);
  }
  ~RefLlama() {
    if (ctx_) llama_free(ctx_);
    if (model_) llama_model_free(model_);
  }
  RefLlama(const RefLlama&) = delete;
  RefLlama& operator=(const RefLlama&) = delete;
  std::uint32_t vocab() const { return vocab_; }

  // Logits for every token of `tokens` (positions 0..n-1) from a fresh sequence in one batch.
  std::vector<float> logits_all(const std::vector<std::int32_t>& tokens) {
    llama_memory_clear(llama_get_memory(ctx_), true);
    std::vector<float> out;
    decode(tokens, 0, true, &out);
    return out;
  }

  // Logits of the last token only, from a fresh sequence decoded as one batch with a single output row.
  std::vector<float> logits_last(const std::vector<std::int32_t>& tokens) {
    llama_memory_clear(llama_get_memory(ctx_), true);
    std::vector<float> out;
    decode(tokens, 0, false, &out);
    return out;
  }

  // Greedy continuation (ties to the lowest id, as the Coordinator's argmax), one token per decode call.
  std::vector<std::int32_t> greedy(const std::vector<std::int32_t>& prompt, std::uint32_t n_new) {
    llama_memory_clear(llama_get_memory(ctx_), true);
    std::vector<float> last;
    decode(prompt, 0, false, &last);
    std::vector<std::int32_t> out;
    std::uint64_t pos = prompt.size();
    for (std::uint32_t i = 0; i < n_new; ++i) {
      const auto next = argmax(last);
      out.push_back(next);
      if (i + 1 == n_new) break;
      last.clear();
      decode({next}, pos++, false, &last);
    }
    return out;
  }

 private:
  static std::int32_t argmax(const std::vector<float>& v) {
    std::size_t best = 0;
    for (std::size_t i = 1; i < v.size(); ++i)
      if (v[i] > v[best]) best = i;
    return static_cast<std::int32_t>(best);
  }

  void decode(const std::vector<std::int32_t>& tokens, std::uint64_t base, bool all, std::vector<float>* out) {
    llama_batch b = llama_batch_init(static_cast<std::int32_t>(tokens.size()), 0, 1);
    b.n_tokens = static_cast<std::int32_t>(tokens.size());
    for (std::int32_t i = 0; i < b.n_tokens; ++i) {
      b.token[i] = tokens[static_cast<std::size_t>(i)];
      b.pos[i] = static_cast<llama_pos>(base + static_cast<std::uint64_t>(i));
      b.n_seq_id[i] = 1;
      b.seq_id[i][0] = 0;
      b.logits[i] = all || i + 1 == b.n_tokens;
    }
    REQUIRE(llama_decode(ctx_, b) == 0);
    const std::int32_t rows = all ? b.n_tokens : 1;
    for (std::int32_t i = 0; i < rows; ++i) {
      const float* l = llama_get_logits_ith(ctx_, all ? i : b.n_tokens - 1);
      REQUIRE(l != nullptr);
      out->insert(out->end(), l, l + vocab_);
    }
    llama_batch_free(b);
  }

  llama_model* model_ = nullptr;
  llama_context* ctx_ = nullptr;
  std::uint32_t vocab_ = 0;
};

// Two Father domains driven directly, the way the Coordinator drives them.
struct Pair {
  std::unique_ptr<domain::ExecutionDomain> prefix, tail;
  std::unique_ptr<objects::CanonicalModelStore> store;
  Epoch epoch{1};
  SessionId session{1};
  std::uint64_t window = 0;
  std::uint64_t pos = 0;
  StateVersion state{0};
  domain::WindowRequest last;

  Pair(const TinyModel& m, std::uint32_t max_context = 256, std::uint32_t max_window = 64, std::uint32_t sessions = 1,
       backends::LlamaBackendOptions opts = {}) {
    if (opts.model_dir.empty()) opts = options_for(m);
    auto backend = backends::make_llama_backend(opts);
    const std::uint32_t L = m.layers();
    domain::DomainSpec ps{StageId{0}, domain::StageRole::kPrefix, {0, L}, max_context, max_window, sessions};
    domain::DomainSpec ts{StageId{1}, domain::StageRole::kTail, {L, L}, max_context, max_window, sessions};
    auto p = backend->create_domain(m.manifest, ps);
    REQUIRE_MESSAGE(p.is_ok(), p.status().to_string());
    auto t = backend->create_domain(m.manifest, ts);
    REQUIRE_MESSAGE(t.is_ok(), t.status().to_string());
    prefix = std::move(p).value();
    tail = std::move(t).value();
    auto s = objects::CanonicalModelStore::open(m.dir);
    REQUIRE_MESSAGE(s.is_ok(), s.status().to_string());
    store = std::move(s).value();
    REQUIRE(prefix->prepare(*store).is_ok());
    REQUIRE(tail->prepare(*store).is_ok());
    REQUIRE(prefix->open_session(epoch, session).is_ok());
    REQUIRE(tail->open_session(epoch, session).is_ok());
  }

  domain::WindowRequest next_request(std::uint32_t positions) {
    domain::WindowRequest r;
    r.epoch = epoch;
    r.session = session;
    r.window = WindowId{++window};
    r.base_position = pos;
    r.expected_state = state;
    r.positions = positions;
    last = r;
    return r;
  }

  // Runs a window; the caller commits or aborts it.
  Result<domain::Logits> run(const std::vector<std::int32_t>& tokens) {
    const auto req = next_request(static_cast<std::uint32_t>(tokens.size()));
    auto acts = prefix->run_prefix(req, tokens);
    if (!acts.is_ok()) return acts.status();
    return tail->run_tail(req, acts.value());
  }

  Status commit(std::uint32_t accepted) {
    domain::CommitRequest c{last.epoch, last.session, last.window, accepted, last.expected_state};
    auto a = prefix->commit_window(c);
    if (!a.is_ok()) return a.status();
    auto b = tail->commit_window(c);
    if (!b.is_ok()) return b.status();
    REQUIRE(a.value() == b.value());
    pos = a->committed_position;
    state = a->state;
    return Status::ok();
  }
};

inline float max_abs_diff(const float* a, const float* b, std::size_t n) {
  float d = 0;
  for (std::size_t i = 0; i < n; ++i) d = std::max(d, std::fabs(a[i] - b[i]));
  return d;
}


}  // namespace clusterlm::llamatest
