# 0408: Shipped backend descriptors, llama.cpp breadth evidence, Strata type admission

## Context
Backend-capability-v1 §6 handed F the real descriptors, the loader decision and the qualification-evidence record. An Opus
review of the Strata descriptor against the code and a test campaign on the pinned llama.cpp (CPU build) produced facts the
provisional examples did not have.

## Decisions
1. **Descriptors are embedded JSON.** `runtime/backends/<backend>/descriptor/*.descriptor.json` is embedded into the backend
   library at build time (`cmake/ClusterLMEmbed.cmake`; byte array, no MSVC literal limit) and exposed as
   `llama_local_descriptor_json()` / `strata_hybrid_descriptor_json()`. A missing file can never change what is advertised.
   Parsing into `BackendDescriptor` and the registry stay with A's type (CMR-0001); until it lands the JSON is validated by
   `scripts/check_backend_descriptors.py` (schema + F's rules) and by tests that compare it with what the code exercises.
2. **llama-local lists only what ran.** `tensor_formats` = the 28 weight types the pinned CPU build loaded and decoded
   identically to llama.cpp (`test_llama_tensor_types`; includes NVFP4, Q1_0, Q2_0 absent from the provisional list).
   Families = the 6 architectures with fixture tests (llama incl. MoE and tied head, qwen2, qwen3, gemma, phi3, mamba)
   plus the legacy label-only Fast entry. Label: "Supported, awaiting hardware qualification"; this is mechanics evidence on
   random-weight CPU fixtures and says nothing about real checkpoints. The other 147 architectures the pinned build names
   (`llama_arch_table.inc`, generated) are Experimental; names it does not know are Unsupported
   (`llama_architecture_support`). `max_sessions_per_domain` stays 1; recurrent rollback by recompute is now tested (Mamba).
3. **Strata descriptor from the Opus review** (`strata-hybrid`): 21 tensor formats (Q2_K, IQ1_S removed: no GPU path),
   `cpu: false`, exact Flash-Next families Supported-awaiting, the `*Flash-Next*` glob Experimental, `validated_max_workers`
   2 (only a fake-engine 2-Worker test), `max_sessions_per_domain` 1 (MTP bound to slot 0), `max_q` 4.
4. **Strata type admission.** `expert_format()` now returns `kVersionMismatch` for routed-expert types outside the pinned GPU
   kernels instead of letting the engine `std::exit(1)` on a Node. Fixture-tested; CUDA path unrun.
5. **Qualification evidence** = `qualification/evidence/*.evidence.json` referencing a benchmark result with
   `provenance: Qualified` and a matching hash; enforced by the checker. None exist.

## Consequences
`docs/interfaces/examples/profile-generic-requirements.example.json` (family `Example-Flash-Next-Finetune`, one worker slot)
fails against the shipped Strata descriptor because the glob is Experimental (CMR-0008). The Fast family entry has no
architecture because the model file was never inspected (HQ-FAST-01); the checker allowlists exactly that entry.
Known risks not fixed (need a GPU or Strata patch): MTP VRAM missing from `describe_requirements`, silent embedding VRAM
fallback, advisory `check_vram`, whole-device sync in `release()`, no Windows build of Strata. See the handoff.
