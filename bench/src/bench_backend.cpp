#include "bench_backend.hpp"

#include "backend_cli.hpp"
#include "clusterlm/backends/backend_factory.hpp"

namespace clusterlm::bench {

Result<BackendChoice> resolve_backend(const cli::Args& args, const std::filesystem::path& model_dir, std::size_t remote_stages) {
  BackendChoice choice;
  const backends::BackendOptions options = cli::backend_options_from_args(args, model_dir);
  choice.name = options.name;
  CLM_RETURN_IF_ERROR(backends::check_backend_name(choice.name));
  if (choice.name == "llama" && remote_stages > 0)
    return make_error(ErrorCode::kInvalidArgument,
                      "the llama backend is Father-only (the Fast tier): use --tier fast or a plan whose stages are all @father");
  choice.node_args = cli::node_backend_args(args);
  if (choice.name == "reference") return choice;
  CLM_ASSIGN_OR_RETURN(auto adapter, backends::make_backend(options));
  choice.build_hash = adapter->info().build_hash;
  choice.father = std::shared_ptr<domain::BackendAdapter>(std::move(adapter));
  return choice;
}

const std::vector<std::string>& backend_flags() { return cli::backend_flag_names(); }

}  // namespace clusterlm::bench
