#pragma once
// Strata backend adapter: ExecutionDomains backed by the pinned, patched Strata engine (CUDA Verifier, VRAM expert
// tier, CPU expert pool). See docs/backends/strata-port.md and ADRs 0200-0203.
//
// Compiled only with CLUSTERLM_ENABLE_STRATA (CUDA toolkit + the fetched, patched checkout:
// `python3 scripts/fetch_upstream.py --apply-patches strata strata-ggml`). This declaration is dependency-free so
// callers can reference it unconditionally; without the option the factory is simply not linked.
//
// What each domain owns (nothing is shared between two domains of one process - the Father prefix and tail are
// independent): its layers' canonical dense weights in one device arena (Strata WeightTable, loaded from the
// provisioned strata-dense objects - no pack directory), the native GGUF projections (NativeDense), the head
// (NativeHead, tail) and embedding (NativeEmbed, prefix / MTP tail), the GPU-resident routed experts in an
// ExpertCache, the CPU-resident complement read in place from the provisioned objects by a CPU ExpertPool, and per
// session: a SessionState carved for its layers only (GDN recurrent + conv state, QSA K/V + indexer, PLE history),
// a Verifier (speculative-window scratch, captured graphs) and mapped hand-off buffers.
//
// Without a usable CUDA device prepare() reports kHardwareUnavailable; describe_requirements() (Strata's own sizing
// arithmetic) works anywhere.
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>

#include "clusterlm/domain/backend_adapter.hpp"
#include "clusterlm/domain/drafter.hpp"

namespace clusterlm::backends {

struct StrataBackendOptions {
  int cuda_device = 0;
  // Device memory left free for the OS/desktop (Windows WDDM); prepare() refuses a plan that would cut into it.
  std::uint32_t vram_reserve_mib = 1024;
  // Verification width ceiling; Strata's kVerifyMaxT is 8 at the pinned commit. Larger transport windows (prefill
  // chunks) run in local sub-batches of at most this many positions.
  std::uint32_t max_window_cap = 8;
  // CPU expert pool workers (0 = Strata's default: the physical cores).
  std::uint32_t cpu_threads = 0;
  // Father prefix only: the model GGUF that holds the PLE n-gram table (Father-local, read in place with direct I/O;
  // never provisioned). Empty: the prefix refuses to prepare (running without the PLE is a different model).
  std::filesystem::path ple_table_gguf;
  // Father tail only: the MTP drafter's runtime directory (tools/mtp_rt.py output, Father-local). Empty: no MTP.
  std::filesystem::path mtp_dir;
  // Test hook: called with the exact token pointer and window length handed to Strata's Verifier::run (nullptr on
  // every token-free domain). Never set in production.
  std::function<void(const std::int32_t* tokens, int positions)> verifier_call_observer;
};

// Never returns null. info().hardware_available is false when no CUDA device can be opened.
std::unique_ptr<domain::BackendAdapter> make_strata_backend(const StrataBackendOptions& options);

// The MTP drafter bound to a Strata TAIL domain created by make_strata_backend with options.mtp_dir set (Father
// only: it consumes token IDs and the tail's final residual rows locally). The domain must outlive the drafter.
Result<std::unique_ptr<domain::Drafter>> make_strata_mtp_drafter(domain::ExecutionDomain& tail);

}  // namespace clusterlm::backends
