# 0300: Fast tier runs as two Father domains over one llama.cpp context

## Context
The Fast tier (`llama-local`) runs a whole model on Father with the pinned llama.cpp. The Coordinator requires a plan
with a prefix and a tail stage, both on Father. llama.cpp cannot expose a mid-model activation boundary, and has no
PLE, gated residual streams or experts in the ClusterLM sense.

## Decision
- `LlamaLocalBackend` creates a `kPrefix` domain over `[0, L)` and a head-only `kTail` over `[L, L)` (the plan
  `0-L@father,L-L@father`, already valid for the Coordinator). The prefix runs the full llama.cpp forward pass and keeps
  the window's logits Father-local; it returns an opaque 3-float-per-position zero "handle" (hc=1, H=1) as its
  activations, never serialized. The tail returns the stored logits. Only the prefix touches the KV cache; both keep a
  `WindowLedger`, so state versions, epochs and commit replay follow the shared rules.
- Middle stages and partial layer ranges are refused (`kUnimplemented`); `run_window` is `kUnimplemented`.
- `CoordinatorConfig::backend` (optional, default reference backend) is the only Coordinator change.
- Weights are mapped from Father's GGUF files named by the manifest (`LlamaBackendOptions::model_dir`); the resolver is
  unused and nothing is requantized. The manifest of a llama model is built from the GGUF itself
  (`build_llama_manifest`): dense models map to a degenerate geometry (1 expert, hc=1, placeholder PLE).
- Windows longer than `full_logits_max_positions` (32) are prefill chunks and return only the last position's logits
  (`Logits::positions == 1`), which is all the Coordinator reads from prefill; this avoids the output head for every
  prompt position. Verification windows return every position.
- commit(n) is `llama_memory_seq_rm(seq, base+n, -1)`; abort is the same from the window base. If llama.cpp refuses a
  partial removal (recurrent state), the sequence is cleared and the committed prefix re-decoded from Father's own token
  history; `n_rs_seq = max_window` asks llama.cpp for rollback snapshots where the architecture supports them.

## Consequences
Coordinator, protocol and catalog are unchanged. A Fast plan cannot include Nodes. Requirements are estimates from the
manifest; exact allocations are only in llama.cpp's load log (the runtime allocation ledger remains open work). The
recompute fallback is implemented but not exercised by the tiny attention-only test model.
