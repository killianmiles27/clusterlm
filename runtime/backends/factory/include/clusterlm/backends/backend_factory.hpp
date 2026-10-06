#pragma once
// Backend selection by name: the one place that maps "reference" / "strata" to a BackendAdapter, shared by the Node
// worker and the Coordinator (Father prefix/tail) so `--backend <name>` means the same thing on every machine.
//
//   "reference"  the deterministic CPU backend (runtime/domain), always available;
//   "strata"     make_strata_backend() when this build has CLUSTERLM_ENABLE_STRATA, otherwise kUnimplemented
//                ("built without CLUSTERLM_ENABLE_STRATA"). On a host without a usable CUDA device the adapter exists
//                but its domains refuse prepare() with kHardwareUnavailable - nothing here pretends otherwise.
//
// An unknown name is kInvalidArgument: a process asked for a backend this code does not know must refuse to start,
// never fall back to another backend (Father and Nodes must agree on the backend build hash anyway).
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "clusterlm/backends/strata_backend.hpp"
#include "clusterlm/common/status.hpp"
#include "clusterlm/domain/backend_adapter.hpp"
#include "clusterlm/domain/drafter.hpp"

namespace clusterlm::backends {

struct BackendOptions {
  std::string name = "reference";
  // Used only by "strata". Father-only fields (ple_table_gguf, mtp_dir) are ignored on Nodes by the adapter itself.
  StrataBackendOptions strata;
};

// Names this code knows ("reference", "strata"), whether or not they are built in.
std::vector<std::string> known_backend_names();
// True when the named backend is compiled into this binary.
bool backend_built(std::string_view name);

// kInvalidArgument for an unknown name, kUnimplemented for a known one that is not built, ok otherwise.
Status check_backend_name(std::string_view name);
Result<std::unique_ptr<domain::BackendAdapter>> make_backend(const BackendOptions& options);

// The MTP drafter of the named backend over a Father TAIL domain created by that backend (see
// make_strata_mtp_drafter). "reference" has no drafter bound to a domain: kUnimplemented (the reference MTP fixture
// drafter is created from the model store by the Coordinator).
Result<std::unique_ptr<domain::Drafter>> make_backend_drafter(std::string_view name, domain::ExecutionDomain& tail);

}  // namespace clusterlm::backends
