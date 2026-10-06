#pragma once
// Strata backend adapter: ExecutionDomains backed by the pinned Strata kernels (CUDA, Verifier, expert pool).
//
// Compiled only with CLUSTERLM_ENABLE_STRATA (needs the CUDA toolkit and the fetched checkout, see
// scripts/fetch_upstream.py). The declaration is dependency-free so callers can reference it unconditionally;
// without the option the factory simply is not linked.
//
// Status: SKELETON. Structure, admission checks and error paths are real; every operation that needs the CUDA
// Verifier is QUALIFICATION-PENDING and returns kUnimplemented/kHardwareUnavailable. See
// docs/backends/strata-port.md for the seams and the findings the real implementation must address
// (hand-off layout transpose, PLE-on-Father constraint, no abort-without-commit, non-idempotent commit).
#include <cstdint>
#include <memory>
#include <string>

#include "clusterlm/domain/backend_adapter.hpp"

namespace clusterlm::backends {

struct StrataBackendOptions {
  int cuda_device = 0;
  // Device memory left free for the OS/desktop (Windows WDDM); the domain's budget is free VRAM minus this.
  std::uint32_t vram_reserve_mib = 1024;
  // Verification width ceiling; Strata's kVerifyMaxT is 8 at the pinned commit.
  std::uint32_t max_window_cap = 8;
};

// Never returns null. The adapter reports hardware_available=false when no CUDA device can be opened.
std::unique_ptr<domain::BackendAdapter> make_strata_backend(const StrataBackendOptions& options);

}  // namespace clusterlm::backends
