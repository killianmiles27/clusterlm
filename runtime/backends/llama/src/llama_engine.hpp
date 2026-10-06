#pragma once
// LlamaEngine: one loaded llama.cpp model + context shared by the Father prefix and tail domains of the llama
// backend. All llama.cpp calls live in llama_engine.cpp; this header exposes no llama.cpp types.
//
// Thread-safe (one mutex): the Coordinator drives both domains from one thread, but the engine does not rely
// on it. Sessions map to llama sequence slots; a session's committed token history is kept (Father-local, it is
// the same token IDs the Coordinator already holds) only so a refused partial KV removal can be repaired by
// re-decoding the committed prefix.
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

#include "clusterlm/backends/llama_backend.hpp"
#include "clusterlm/common/ids.hpp"
#include "clusterlm/domain/execution_domain.hpp"

namespace clusterlm::backends {

class LlamaEngine {
 public:
  struct Params {
    std::vector<std::filesystem::path> files;  // GGUF shards in order
    LlamaBackendOptions options;
    domain::DomainSpec spec;                   // the prefix spec: max_context / max_window / max_sessions
    std::uint32_t manifest_vocab = 0;          // the loaded model must agree
  };

  explicit LlamaEngine(Params params);
  ~LlamaEngine();
  LlamaEngine(const LlamaEngine&) = delete;
  LlamaEngine& operator=(const LlamaEngine&) = delete;

  // Loads the model and creates the context. Idempotent while loaded.
  Status load();
  // Frees the context and the model (and every session). Safe to call repeatedly.
  void release();
  bool loaded() const;
  std::uint64_t model_bytes() const;  // llama_model_size of the loaded model (0 when released)
  std::uint32_t vocab() const;

  Status open_session(SessionId session);
  // Frees the sequence slot and its KV; unknown sessions are ignored.
  void close_session(SessionId session);
  std::size_t open_sessions() const;

  // Decodes tokens at positions [base, base+n) of the session as one temporary window and keeps the per-position
  // logits for take_logits(). The session must be at exactly `base` committed positions.
  Status run_window(SessionId session, WindowId window, std::uint64_t base, std::span<const std::int32_t> tokens);
  // Keeps the first `accepted` positions of the outstanding window; the rest leave the KV cache.
  Status commit(SessionId session, WindowId window, std::uint32_t accepted);
  // Discards the outstanding window if it is `window` (no-op otherwise).
  Status abort_window(SessionId session, WindowId window);
  // Moves the window's logits out. kNotFound when they were never produced or were already taken.
  Result<domain::Logits> take_logits(SessionId session, WindowId window, std::uint32_t positions);

  LlamaDomainStats stats() const;
  std::uint64_t windows_run() const;
  std::uint64_t compute_ns() const;

  struct Impl;

 private:
  std::unique_ptr<Impl> impl_;
};

}  // namespace clusterlm::backends
