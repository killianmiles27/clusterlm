#pragma once
// Backend-selection command-line flags shared by clusterlm-father and clusterlm-bench (so `--backend strata
// --cuda-device 0 ...` means the same thing in both), and the matching flags clusterlm-node takes. Header-only; the
// including target links clusterlm_backend_factory.
//
//   --backend reference|llama|strata   Father's prefix/tail backend (Nodes get the same, see node_backend_args)
//   --strata-ple-gguf FILE             Father prefix: the GGUF that holds the PLE n-gram table
//   --strata-mtp-dir DIR               Father tail: the MTP drafter runtime directory
//   --cuda-device N, --vram-reserve-mib N, --strata-cpu-threads N     Strata engine options (Father and Nodes)
//   --llama-gpu-layers N               llama backend only (present in a CLUSTERLM_ENABLE_LLAMA build)
#include <filesystem>
#include <string>
#include <vector>

#include "cli.hpp"
#include "clusterlm/backends/backend_factory.hpp"

namespace clusterlm::cli {

// Every flag this header reads (usage text, bench command spec).
inline const std::vector<std::string>& backend_flag_names() {
  static const std::vector<std::string> names = {"backend",          "strata-ple-gguf",    "strata-mtp-dir",   "cuda-device",
                                                 "vram-reserve-mib", "strata-cpu-threads", "llama-gpu-layers"};
  return names;
}

// Options for Father's own backend. `model_dir` is the llama backend's model directory (ignored otherwise).
inline backends::BackendOptions backend_options_from_args(const Args& args,
                                                          [[maybe_unused]] const std::filesystem::path& model_dir) {
  backends::BackendOptions bo;
  bo.name = args.get("backend", "reference");
  bo.strata.cuda_device = static_cast<int>(args.integer("cuda-device", 0));
  bo.strata.vram_reserve_mib = static_cast<std::uint32_t>(args.integer("vram-reserve-mib", 1024));
  bo.strata.cpu_threads = static_cast<std::uint32_t>(args.integer("strata-cpu-threads", 0));
  bo.strata.ple_table_gguf = args.get("strata-ple-gguf");
  bo.strata.mtp_dir = args.get("strata-mtp-dir");
#if defined(CLUSTERLM_FACTORY_HAS_LLAMA)
  bo.llama.model_dir = model_dir;
  bo.llama.n_gpu_layers = static_cast<std::int32_t>(args.integer("llama-gpu-layers", 0));
#endif
  return bo;
}

// The clusterlm-node flags that make a Node host the same backend as Father (the build hashes must match). Only flags
// the caller gave are forwarded, so the Node's own defaults apply otherwise. Father-only options (PLE table, MTP
// directory, llama) are never forwarded: token IDs and the Father tail never reach a Node.
inline std::vector<std::string> node_backend_args(const Args& args) {
  std::vector<std::string> out;
  if (args.has("backend")) out.insert(out.end(), {"--backend", args.get("backend")});
  for (const char* flag : {"cuda-device", "vram-reserve-mib", "strata-cpu-threads"})
    if (args.has(flag)) out.insert(out.end(), {std::string("--") + flag, args.get(flag)});
  return out;
}

}  // namespace clusterlm::cli
