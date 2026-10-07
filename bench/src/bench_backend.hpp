#pragma once
// Backend selection for the runtime commands (`cluster`, `faults`, `placement-inputs`, `placement-validate`): the same
// `--backend reference|llama|strata` and Strata/llama option flags as clusterlm-father / clusterlm-node (parsed by
// apps/common/backend_cli.hpp), resolved BEFORE anything is spawned so a backend this binary does not have fails with
// a clear message instead of a half-started cluster.
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "cli.hpp"
#include "clusterlm/common/status.hpp"
#include "clusterlm/domain/backend_adapter.hpp"

namespace clusterlm::bench {

struct BackendChoice {
  std::string name = "reference";
  // Father's prefix/tail backend (CoordinatorConfig::backend); null selects the reference backend. Shared by every
  // Coordinator of the run - runs are sequential, so only one holds device memory at a time.
  std::shared_ptr<domain::BackendAdapter> father;
  // clusterlm-node flags that make each spawned Node host the same backend (see cli::node_backend_args).
  std::vector<std::string> node_args;
  std::string build_hash = "reference";  // BackendInfo::build_hash of Father's adapter (recorded in the result)
  bool real() const { return name != "reference"; }
};

// kInvalidArgument for an unknown name, kUnimplemented for a known backend this binary was built without (the message
// names the CMake option), kInvalidArgument for llama with Nodes (the llama backend is Father-only), otherwise the
// factory's own error. `remote_stages`: how many plan stages run on Nodes. `model_dir` is the llama backend's model.
Result<BackendChoice> resolve_backend(const cli::Args& args, const std::filesystem::path& model_dir, std::size_t remote_stages);

// The flags `resolve_backend` reads, for the command specs.
const std::vector<std::string>& backend_flags();

}  // namespace clusterlm::bench
