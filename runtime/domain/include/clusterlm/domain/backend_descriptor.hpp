#pragma once
// Backend capability descriptors (docs/interfaces/backend-capability-v1.md, frozen v1.1): the DECLARATIVE half of a
// backend. BackendAdapter / BackendInfo stay the runtime half. A descriptor says honestly what a backend can run and
// how sure we are; profiles, the model library, the UI, the API and the scheduler ask it without instantiating anything.
//
// Rules enforced here: every capability is explicit (the parser requires every capability field; a missing capability is
// written false/0/none), unknown fields are errors, arrays are bounded, tensor formats are canonical ggml names, and a
// descriptor is built into the binary (embedded) so a missing file can never advertise a backend.
//
// check_model() computes the user-facing compatibility label from descriptor + model facts. Labels are NEVER typed by a
// user and never raised by fixture or reference-backend results.
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "clusterlm/common/status.hpp"

namespace clusterlm::domain {

// The four user-facing labels, strongest first. Compare with `<=`/`>=` through rank().
enum class CompatLabel : std::uint8_t { kSupportedQualified, kSupportedAwaitingQualification, kExperimental, kUnsupported };
std::string_view to_string(CompatLabel l) noexcept;  // the exact label text
std::optional<CompatLabel> compat_label_from_string(std::string_view s);
inline int rank(CompatLabel l) { return static_cast<int>(l); }  // lower = stronger
inline bool is_supported(CompatLabel l) { return l == CompatLabel::kSupportedQualified || l == CompatLabel::kSupportedAwaitingQualification; }

struct FamilyEntry {
  std::string family_match;  // exact or '*' glob over the library family label
  std::optional<std::string> architecture;  // GGUF general.architecture, exact
  CompatLabel status = CompatLabel::kUnsupported;
  std::vector<std::string> evidence;
  friend bool operator==(const FamilyEntry&, const FamilyEntry&) = default;
};

struct OptionSpec {
  enum class Type : std::uint8_t { kInteger, kBoolean, kString } type = Type::kInteger;
  std::optional<std::int64_t> minimum, maximum;
  std::vector<std::string> allowed;  // enum
  std::string description;
  friend bool operator==(const OptionSpec&, const OptionSpec&) = default;
};

struct BackendDescriptor {
  std::string id;            // llama-local | strata-hybrid | reference
  std::string display_name;
  std::string factory_name;  // llama | strata | reference
  bool development_only = false;

  struct ModelSupport {
    std::vector<std::string> containers;       // gguf | gguf-split | manifest-native
    std::vector<std::string> tensor_formats;   // canonical ggml names, upper case
    std::vector<FamilyEntry> families;
    CompatLabel unlisted_family_status = CompatLabel::kUnsupported;  // Experimental | Unsupported only
    friend bool operator==(const ModelSupport&, const ModelSupport&) = default;
  } model_support;

  struct Devices {
    bool cpu = false;
    std::vector<std::string> gpu_vendors;
    std::optional<std::string> min_compute_capability;
    std::vector<std::string> requires_runtime;
    friend bool operator==(const Devices&, const Devices&) = default;
  } devices;

  struct Execution {
    bool single_host = false, cross_machine = false;
    std::uint32_t validated_max_workers = 0;
    std::vector<std::string> validated_topologies;
    std::string layer_partitioning = "none";  // none | contiguous-layers
    bool cpu_gpu_hybrid = false, token_free_middle_stages = false, manual_layer_ranges = false;
    std::vector<std::string> worker_roles;  // [] or ["middle"]
    std::vector<std::string> evidence;
    friend bool operator==(const Execution&, const Execution&) = default;
  } execution;

  struct State {
    std::vector<std::string> kinds;  // kv | recurrent | indexer | ple-history
    std::string supports_rollback = "none";  // native | recompute | none
    std::uint32_t max_sessions_per_domain = 1;
    bool per_client_isolation = false;
    friend bool operator==(const State&, const State&) = default;
  } state;

  struct Speculation {
    bool window_commit_abort = false;
    std::string mtp = "none";  // none | one-hot-proposals | full-distribution
    std::uint32_t max_q = 1;
    friend bool operator==(const Speculation&, const Speculation&) = default;
  } speculation;

  struct Serving {
    bool full_logits_for_sampling = false, constrained_decoding = false, logprobs = false;
    friend bool operator==(const Serving&, const Serving&) = default;
  } serving;

  std::vector<std::string> limitations;
  std::vector<std::pair<std::string, OptionSpec>> options;  // sorted by name

  struct Qualification {
    CompatLabel label = CompatLabel::kUnsupported;  // ceiling for every model on this backend
    std::vector<std::string> hardware_experiments;  // HQ ids
    std::string notes;
    friend bool operator==(const Qualification&, const Qualification&) = default;
  } qualification;

  friend bool operator==(const BackendDescriptor&, const BackendDescriptor&) = default;
};

Status validate(const BackendDescriptor& d);
Result<BackendDescriptor> backend_descriptor_from_json(std::string_view json);
std::string to_json(const BackendDescriptor& d);

// ---- runtime half: probed on THIS machine ---------------------------------------------------------------------

struct BackendRuntimeStatus {
  bool built = false;             // backends::backend_built(factory_name)
  bool runtime_present = false;   // CUDA runtime/driver, DLLs, ...
  bool hardware_available = false;  // == BackendInfo::hardware_available
  std::string build_hash;         // == BackendInfo::build_hash; Host and Workers must agree
  std::vector<std::string> reasons;  // human sentences when any bool is false
};

// ---- model facts and compatibility ------------------------------------------------------------------------------

// What check_model() needs to know about a model; filled from a library ModelRecord (kept free of orchestrator types so
// runtime/domain stays below orchestrator/).
struct ModelFacts {
  std::string container;        // gguf (single file) | gguf-split | manifest-native
  std::string family;           // library family label
  std::string architecture;     // GGUF general.architecture ("" if absent)
  std::vector<std::string> tensor_types;  // EVERY tensor type present, canonical upper-case names
  bool experimental_opt_in = false;       // the user's per-model opt-in (library record)
};

struct Finding {
  enum class Severity : std::uint8_t { kBlocker, kWarning, kNote } severity = Severity::kNote;
  std::string code;     // stable machine code
  std::string message;  // human sentence
  friend bool operator==(const Finding&, const Finding&) = default;
};

struct CompatReport {
  CompatLabel label = CompatLabel::kUnsupported;
  std::vector<Finding> findings;
  bool hosts_alone = false;
  bool distributable = false;
  std::uint32_t max_workers = 0;  // distributable ? validated_max_workers : 0
  bool has_blocker() const;
};

CompatReport check_model(const BackendDescriptor& d, const ModelFacts& m, const BackendRuntimeStatus& rt);

// ---- registry ---------------------------------------------------------------------------------------------------

// The descriptors compiled into this binary. Read-only after construction.
class BackendRegistry {
 public:
  // reference, llama-local, strata-hybrid as shipped (provisional, F reviews: docs/interfaces/backend-capability-v1.md §6).
  static const BackendRegistry& builtin();
  // For tests and for F's loader: a registry of explicit descriptors (each validated).
  static Result<BackendRegistry> of(std::vector<BackendDescriptor> descriptors);

  std::vector<const BackendDescriptor*> list() const;
  const BackendDescriptor* find(std::string_view id) const;

 private:
  std::vector<BackendDescriptor> descriptors_;
};

// Glob match with '*' only (any run of characters), case-insensitive. Pattern and text are bounded (<= 120).
bool family_glob_match(std::string_view pattern, std::string_view text);

}  // namespace clusterlm::domain
