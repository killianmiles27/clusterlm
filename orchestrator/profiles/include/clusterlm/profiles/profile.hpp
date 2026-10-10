#pragma once
// Execution profiles (docs/interfaces/profile-schema-v1.md, frozen v1.1) and routing aliases.
//
// A profile says WHAT to run (exact model identity + quantization), HOW (backend, context, speculation), WHERE (topology
// slots Workers may fill), and what to do when things change. It never holds measurements, qualification claims, paths,
// credentials or fingerprints on export. The parser is the C++ mirror of schemas/profile-v1.schema.json:
//   * strict: an unknown field at any depth, a wrong type or an out-of-range value is an error (never silently dropped);
//   * bounded: 1 MiB document, depth 16 (kMaxDocumentBytes / kMaxJsonDepth);
//   * the writer omits every optional field equal to its default, so a document that does not use a later optional
//     addition stays readable by an older reader.
// Cross-object rules (unique api_model_id, fallback chains, worker-count rule, ...) live in validate.hpp.
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "clusterlm/common/status.hpp"

namespace clusterlm::profiles {

inline constexpr std::size_t kMaxDocumentBytes = 1u << 20;
inline constexpr int kMaxJsonDepth = 16;
inline constexpr std::size_t kMaxWorkerSlots = 8;

// ---- model identity ---------------------------------------------------------------------------------------------

struct ExpectedFile {
  std::string role;
  std::optional<std::string> name;
  std::optional<std::uint64_t> approx_bytes;
  friend bool operator==(const ExpectedFile&, const ExpectedFile&) = default;
};

struct ModelIdentity {
  std::string family;
  std::string display_name;
  std::string quant;
  std::optional<std::string> artifact_id;
  std::optional<std::string> expected_root_hash;  // null = unpinned
  std::vector<ExpectedFile> expected_files;
  friend bool operator==(const ModelIdentity&, const ModelIdentity&) = default;
};

struct ModelRef {
  ModelIdentity identity;
  std::optional<std::string> library_id;  // local only; removed on export
  friend bool operator==(const ModelRef&, const ModelRef&) = default;
};

// ---- backend ----------------------------------------------------------------------------------------------------

struct BackendRef {
  std::string id;
  std::uint32_t min_contract_version = 1;
  // Scalar options only (integer, boolean or string), at most 16, key pattern [a-z][a-z0-9_]{0,31}.
  struct Option {
    std::string key;
    std::variant<std::int64_t, bool, std::string> value;
    friend bool operator==(const Option&, const Option&) = default;
  };
  std::vector<Option> options;  // sorted by key
  friend bool operator==(const BackendRef&, const BackendRef&) = default;
};

// ---- context ----------------------------------------------------------------------------------------------------

struct ContextPolicy {
  std::uint32_t default_tokens = 4096;
  std::uint32_t max_tokens = 4096;
  std::vector<std::uint32_t> offered_profiles;  // ascending; empty = placement workload defaults clipped to max_tokens
  friend bool operator==(const ContextPolicy&, const ContextPolicy&) = default;
};

// ---- topology ---------------------------------------------------------------------------------------------------

struct Requirements {
  std::vector<std::string> gpu_vendor;  // nvidia|amd|intel; empty = any
  std::optional<std::uint32_t> min_vram_mib, min_ram_mib;
  std::optional<std::string> min_compute_capability;  // "8.6"
  std::optional<std::uint32_t> min_link_mbit;
  bool cpu_only_ok = false;
  friend bool operator==(const Requirements&, const Requirements&) = default;
};

struct Selector {
  enum class Mode : std::uint8_t { kBinding, kMachine, kRequirements } mode = Mode::kBinding;
  std::string binding;  // kBinding
  std::string machine;  // kMachine: device fingerprint (local only)
  std::optional<Requirements> requirements;  // kRequirements (required), kBinding (optional extra checks)
  friend bool operator==(const Selector&, const Selector&) = default;
};

struct Slot {
  enum class Kind : std::uint8_t { kHost, kWorker } kind = Kind::kWorker;
  std::string slot;
  std::string label;
  bool optional = false;
  std::optional<Selector> select;  // worker slots only
  friend bool operator==(const Slot&, const Slot&) = default;
};

struct Topology {
  std::vector<Slot> slots;
  std::optional<std::uint32_t> min_workers, max_workers;  // defaults: see worker_count_bounds()
  friend bool operator==(const Topology&, const Topology&) = default;
};

// Effective [min,max] worker count after defaults: min = non-optional worker slots, max = worker slots.
struct WorkerBounds { std::uint32_t min = 0, max = 0; };
WorkerBounds worker_count_bounds(const Topology& t);
std::size_t worker_slot_count(const Topology& t);

// ---- remaining sections -----------------------------------------------------------------------------------------

struct Resources {
  enum class Device : std::uint8_t { kAuto, kGpuFirst, kGpuOnly, kCpuOnly } device_preference = Device::kAuto;
  std::optional<std::uint32_t> host_vram_margin_mib, host_ram_margin_mib, worker_vram_margin_mib, worker_ram_margin_mib,
      max_worker_temp_storage_mib;
  friend bool operator==(const Resources&, const Resources&) = default;
};

struct ManualStage {
  std::string slot;
  std::uint32_t first_layer = 0, end_layer = 1;
  friend bool operator==(const ManualStage&, const ManualStage&) = default;
};
struct Placement {
  enum class Mode : std::uint8_t { kAuto, kManual } mode = Mode::kAuto;
  enum class Objective : std::uint8_t { kBalanced, kLatency, kPrepCost } objective = Objective::kBalanced;
  std::vector<ManualStage> manual;  // iff mode == kManual
  friend bool operator==(const Placement&, const Placement&) = default;
};

struct Speculation {
  bool enabled = true;
  std::uint32_t max_q = 1;
  friend bool operator==(const Speculation&, const Speculation&) = default;
};

struct Lifecycle {
  enum class Preparation : std::uint8_t { kOnDemand, kKeepReady, kManual } preparation = Preparation::kOnDemand;
  std::uint32_t release_after_idle_seconds = 60;
  bool prepare_when_available = false;
  friend bool operator==(const Lifecycle&, const Lifecycle&) = default;
};

struct OnWorkerLoss {
  enum class Then : std::uint8_t { kStop, kReplan, kFallbackProfile } then = Then::kStop;
  std::uint32_t max_retries = 1;
  std::optional<std::string> fallback_profile_id;
  bool preserve_conversation = true;
  friend bool operator==(const OnWorkerLoss&, const OnWorkerLoss&) = default;
};

struct Exposure {
  bool api = false;
  std::optional<std::string> api_model_id;
  bool allow_lan = false;
  friend bool operator==(const Exposure&, const Exposure&) = default;
};

struct Goal {
  std::string metric;
  double value = 0;
  std::optional<std::string> experiment;
  // status is always pending_qualification in v1 (the schema allows nothing else).
  friend bool operator==(const Goal&, const Goal&) = default;
};

struct Profile {
  std::string id;  // prof_*
  std::string name;
  std::string description;
  std::uint64_t revision = 1;
  std::optional<std::string> example_of;
  std::vector<std::string> qualification_experiments;
  ModelRef model;
  BackendRef backend;
  ContextPolicy context;
  Topology topology;
  Resources resources;
  Placement placement;
  Speculation speculation;
  std::string chat_template_source = "model";
  Lifecycle lifecycle;
  OnWorkerLoss on_worker_loss;
  Exposure exposure;
  std::vector<Goal> goals;
  friend bool operator==(const Profile&, const Profile&) = default;
};

// ---- routing alias ----------------------------------------------------------------------------------------------

struct AliasCandidate {
  std::string profile_id;
  bool has_when = false;
  std::vector<std::string> workers_available;
  std::optional<std::uint32_t> min_workers_available;
  bool require_ready = false;
  bool avoid_on_battery = true;
  friend bool operator==(const AliasCandidate&, const AliasCandidate&) = default;
};

struct RoutingAlias {
  enum class Policy : std::uint8_t { kFirstViable, kBestReady } policy = Policy::kFirstViable;
  std::string id;  // alias_*
  std::string name;
  std::string api_model_id;
  bool allow_lan = false;
  std::vector<AliasCandidate> candidates;
  bool allow_prepare = false;
  friend bool operator==(const RoutingAlias&, const RoutingAlias&) = default;
};

// ---- parse / serialize ------------------------------------------------------------------------------------------

Result<Profile> profile_from_json(std::string_view json);
std::string to_json(const Profile& p);
Result<RoutingAlias> alias_from_json(std::string_view json);
std::string to_json(const RoutingAlias& a);

// Field-level validation of one document (the schema half): patterns, ranges, per-object semantic rules that need no
// other object. Run by the parsers; exposed for hand-built values.
Status validate(const Profile& p);
Status validate(const RoutingAlias& a);

// Id helpers.
bool is_profile_id(std::string_view s);
bool is_alias_id(std::string_view s);
bool is_api_model_id(std::string_view s);

}  // namespace clusterlm::profiles
