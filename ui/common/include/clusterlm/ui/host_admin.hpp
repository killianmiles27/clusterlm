#pragma once
// HostAdminClient: everything the Host management views (Models, Profiles, Machines, Connections, Performance,
// Settings, first-run wizard) may ask of the Host side, as an abstract interface plus the plain data it exchanges.
//
// The data types are UI-facing views of the frozen contracts in docs/interfaces/ (profile-schema-v1,
// readiness-state-machine-v1, backend-capability-v1, auth-scopes-v1). They carry no behaviour. Implementations:
//   * ScriptedHostAdminClient (scripted_host_admin.hpp): in-memory, seeded with the Fast/Strong/Ultra example set;
//     unit tests and `--demo`. Every number it produces is labelled Synthetic.
//   * the IPC client for profile/library/server ops arrives with workstreams A/B/C (docs/cross-module-requests.md,
//     CMR-0008). Until then an operation the Host cannot do answers kUnimplemented and the view says so in words.
//
// Honesty rules carried by the types:
//   * readiness is the seven-state function output of the Host; the UI never derives "ready" itself;
//   * a Quantity always names its provenance (Synthetic < Measured < Qualified); a dry run is Synthetic unless the
//     Host says otherwise and is never presented as a benchmark;
//   * compatibility labels are exactly the four in the spec and arrive from the Host (computed from descriptors and
//     qualification evidence); the UI never lets a user type one;
//   * secrets (API key values) appear exactly once, in the reply that creates them.
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "clusterlm/common/status.hpp"

namespace clusterlm::ui {

// ---- shared -------------------------------------------------------------------------------------------------

enum class Provenance : std::uint8_t { kSynthetic, kMeasured, kQualified };
std::string_view to_string(Provenance p);  // "Synthetic" | "Measured" | "Qualified"

struct Quantity {
  double value = 0;
  std::string unit;  // "bytes", "tok/s", "s", ...
  Provenance provenance = Provenance::kSynthetic;
  std::string source;  // where it came from, in words ("placement search over advertised capability")
};

// Spec compatibility labels (docs/spec/02). Never typed by a user.
enum class CompatLabel : std::uint8_t {
  kSupportedQualified,
  kSupportedAwaitingQualification,
  kExperimental,
  kUnsupported,
};
std::string_view to_string(CompatLabel l);  // exact spec wording

// ---- models -------------------------------------------------------------------------------------------------

struct ModelCompat {
  std::string backend_id;    // "llama-local" | "strata-hybrid" | ...
  std::string backend_name;  // human name
  CompatLabel label = CompatLabel::kUnsupported;
  bool backend_built = true;     // this build contains it
  bool runtime_present = true;   // its runtime (e.g. CUDA) is present on the Host
  std::uint32_t max_workers = 0; // distributable up to; 0 = Host only
  std::vector<std::string> findings;  // human sentences: why this label
};

struct ModelView {
  std::string id;  // mdl_*
  std::string name, family, architecture, quant;
  std::vector<std::string> quant_mix;  // every tensor type present, so a mixed quantization is never shown as one
  std::string dir;                     // Host-local; shown to the owner on the Host, never exported
  std::uint32_t file_count = 0;
  bool split = false;
  std::uint64_t total_bytes = 0;
  std::optional<std::uint32_t> block_count, context_length, expert_count;
  std::optional<std::string> root_hash;    // verified manifest root
  std::optional<std::string> pinned_root;  // explicit user confirmation
  std::vector<ModelCompat> compat;
  std::vector<std::string> used_by;  // names of profiles that reference this record
};

struct ScanIssue {
  std::string file, message;
};
struct ScanReport {
  std::uint32_t found = 0;
  std::vector<ScanIssue> issues;
};

// ---- profiles -----------------------------------------------------------------------------------------------

// readiness-state-machine-v1 §1
enum class ReadyState : std::uint8_t { kUnavailable, kInstalled, kCompatible, kLoadable, kPreparing, kReady, kBusy };
std::string_view to_string(ReadyState s);  // "unavailable" ...

struct ReadinessView {
  ReadyState state = ReadyState::kUnavailable;
  ReadyState reached = ReadyState::kUnavailable;  // for unavailable: how far it got (kUnavailable | kInstalled | kCompatible)
  std::vector<std::string> reasons;               // first is the headline
  std::vector<std::string> blockers;              // machine codes
  std::optional<double> percent;                  // preparing
  std::optional<double> eta_seconds;              // preparing; always shown as an estimate
  std::string last_error;
  std::vector<std::string> suggested_alternatives;  // profile ids that are ready/loadable
};

struct SlotView {
  std::string name;   // "host", "w1"
  bool host = false;
  bool optional = false;
  std::string selector;  // human sentence: "assigned to G14", "any Worker with 12 GB+ graphics memory", "not assigned"
  std::string binding;   // binding name for selector mode "binding" (empty otherwise)
  std::string bound_machine;  // display name of the machine filling it right now, empty when unfilled
};

struct ProfileView {
  std::string id;  // prof_*
  std::string name;
  std::uint32_t revision = 1;
  std::string example_of;  // "tier:fast" or empty
  std::string model_text;  // "Qwen3 30B-A3B, IQ3_S"
  std::string model_library_id;  // local; empty if unresolved
  std::string backend_id;
  std::uint32_t default_context = 4096, max_context = 4096;
  std::vector<std::uint32_t> offered_contexts;
  std::vector<SlotView> slots;
  std::uint32_t min_workers = 0, max_workers = 0;
  std::string preparation;  // "on-demand" | "keep-ready" | "manual"
  bool prepare_when_available = false;
  std::uint32_t release_after_idle_seconds = 0;
  std::string worker_loss_text;  // "stop" | "use <profile> instead (you will see a notice)"
  std::string fallback_profile_id;
  bool api_enabled = false, allow_lan = false;
  std::string api_model_id;
  std::vector<std::string> goals;  // non-binding targets, always "pending qualification"
  ReadinessView readiness;
  bool selected = false;
};

// What the editor can change. Layer numbers and GGUF details are deliberately absent.
struct ProfileDraft {
  std::string name;
  std::string model_library_id;  // chosen from the library
  std::string backend_id;
  std::uint32_t default_context = 4096;
  std::uint32_t max_context = 4096;
  std::vector<std::string> worker_bindings;  // one entry per worker slot: binding name
  std::uint32_t min_workers = 0;
  std::string preparation = "on-demand";
  std::uint32_t release_after_idle_seconds = 60;
  std::string fallback_profile_id;
  bool api_enabled = false, allow_lan = false;
  std::string api_model_id;
};

struct Finding {
  std::uint8_t level = 1;  // 1 schema/semantic, 2 compatibility, 3 dry run
  std::string code;        // stable machine code
  std::string message;     // human sentence
  bool blocking = true;
};

struct MachineShare {
  std::string machine;  // display name
  Quantity cpu_bytes, gpu_bytes, state_bytes, scratch_bytes;
};
struct DryRunReport {
  bool feasible = false;
  std::uint32_t context_tokens = 0;
  std::vector<MachineShare> per_machine;
  Quantity preparation_bytes;           // data that would be sent to Workers
  std::vector<std::string> bottlenecks;
  std::string plan_summary;
  std::vector<Quantity> quantities;     // e.g. predicted decode rate; Synthetic unless a benchmark backs it
};

enum class ImportChoice : std::uint8_t { kAsk, kReplace, kDuplicate };
struct ImportResult {
  std::string profile_id;
  bool conflict = false;                    // same id, different content: caller must choose Replace/Duplicate
  std::vector<std::string> todo;            // "Import the model ...", "Assign the Worker slot w1"
  std::vector<Finding> findings;
};

// ---- machines -----------------------------------------------------------------------------------------------

struct MachineView {
  std::string id;    // device fingerprint (short form shown)
  std::string name;  // human name the user sees everywhere
  bool is_host = false;
  std::string state;           // "Available" | "In use by someone at the PC" | "Offline" | ...
  std::string state_detail;
  std::string os, cpu;
  std::uint64_t ram_bytes = 0, free_ram_bytes = 0;
  std::vector<std::string> gpus;           // "NVIDIA RTX 3060, 12 GB"
  std::uint64_t vram_bytes = 0, free_vram_bytes = 0;
  std::string backend_build;               // advertised backend build hash, short
  bool on_battery = false, paused = false;
  std::string last_seen;  // "just now" / "2 min ago" / "no recent status"
  std::vector<std::string> used_in;  // profiles currently placed on it
  std::vector<std::string> bindings; // binding names pointing at it
};

struct BindingView {
  std::string name;     // "node:laptop-class" (migrated) or "worker:g14"
  std::string machine;  // machine id; empty = not assigned
  std::vector<std::string> used_by;  // profile names
};

// ---- connections (serving API) -----------------------------------------------------------------------------

struct ServerSettings {
  bool enabled = false;
  bool allow_lan = false;  // false = this PC only. auth-scopes-v1 section 1: needs TLS; the Host refuses plain HTTP off loopback
  std::uint16_t port = 11434;
  bool tls = false;
  bool no_key_on_loopback = false;  // owner option: unauthenticated loopback requests get the `inference` scope only
};
struct ServerStatus {
  bool available = false;  // this build contains the serving layer
  bool listening = false;
  std::string address;     // "127.0.0.1:11434"
  std::string note;        // why unavailable / what is missing
  std::string tls_fingerprint;  // server certificate fingerprint to compare on a client; empty without TLS
  ServerSettings settings;
  std::vector<std::string> exposed_models;  // api_model_ids
};

struct ApiKeyView {
  std::string id, name;
  std::vector<std::string> scopes;          // auth-scopes-v1 section 3: inference, status:read, profiles:prepare, ...
  std::vector<std::string> allowed_models;  // "*" for all
  bool lan_allowed = false;                 // network: lan (default is loopback-only)
  std::string created, last_used;
  bool revoked = false;
};
struct NewKey {
  ApiKeyView key;
  std::string secret;  // shown exactly once
};

// ---- performance ---------------------------------------------------------------------------------------------

struct RunStat {
  std::string profile_id, profile_name, model;
  std::string when;
  double tok_s = 0, ttft_ms = 0;
  std::uint32_t tokens = 0;
  std::string reason;      // completed | cancelled | failed
  Provenance provenance = Provenance::kSynthetic;
  std::vector<std::string> machines;
};

// ---- the client ----------------------------------------------------------------------------------------------

struct HostCapabilities {
  bool models = false, profiles = false, machines = false, server = false, keys = false, dry_run = false,
       backup = false;
};

struct HostSummary {
  std::string host_name;
  bool dev_fixture = false;  // the Host runs the development fixture model: nothing it reports is a measurement
};

class HostAdminClient {
 public:
  virtual ~HostAdminClient() = default;
  virtual Result<HostSummary> summary() = 0;
  virtual HostCapabilities capabilities() = 0;

  // models
  virtual Result<std::vector<ModelView>> list_models() = 0;
  virtual Result<std::vector<std::string>> scan_roots() = 0;
  virtual Status add_scan_root(std::string_view dir) = 0;
  virtual Status remove_scan_root(std::string_view dir) = 0;
  virtual Result<ScanReport> rescan() = 0;
  // Imports one file (or any shard of a split set) from a path on the Host.
  virtual Result<ModelView> import_model(std::string_view path) = 0;
  virtual Status pin_model_root(std::string_view model_id, std::string_view root_hash) = 0;
  virtual Status remove_model(std::string_view model_id) = 0;  // removes the record, never deletes files

  // profiles
  virtual Result<std::vector<ProfileView>> list_profiles(std::uint32_t context_tokens) = 0;
  virtual Result<std::string> create_profile(const ProfileDraft& d) = 0;
  virtual Status update_profile(std::string_view id, const ProfileDraft& d) = 0;
  virtual Status delete_profile(std::string_view id) = 0;
  virtual Result<std::string> duplicate_profile(std::string_view id) = 0;
  virtual Result<std::string> export_profile(std::string_view id) = 0;
  virtual Result<ImportResult> import_profile(std::string_view document, ImportChoice choice) = 0;
  virtual Result<std::vector<Finding>> validate_profile(std::string_view id) = 0;
  virtual Result<std::vector<Finding>> validate_draft(const ProfileDraft& d) = 0;
  virtual Result<DryRunReport> dry_run(std::string_view id, std::uint32_t context_tokens) = 0;
  virtual Status select_profile(std::string_view id) = 0;
  virtual Status prepare_profile(std::string_view id, std::uint32_t context_tokens) = 0;
  virtual Status release_profile(std::string_view id) = 0;

  // machines
  virtual Result<std::vector<MachineView>> list_machines() = 0;
  virtual Result<std::vector<BindingView>> list_bindings() = 0;
  virtual Status set_binding(std::string_view name, std::string_view machine_id) = 0;  // "" unbinds

  // connections
  virtual Result<ServerStatus> server_status() = 0;
  virtual Status set_server(const ServerSettings& s) = 0;
  virtual Result<std::vector<ApiKeyView>> list_keys() = 0;
  virtual Result<NewKey> create_key(std::string_view name, const std::vector<std::string>& scopes,
                                    const std::vector<std::string>& allowed_models, bool lan_allowed) = 0;
  virtual Status revoke_key(std::string_view key_id) = 0;

  // performance
  virtual Result<std::vector<RunStat>> recent_runs() = 0;

  // settings: configuration and library metadata only; never weights, keys or conversations (R-37)
  virtual Result<std::string> export_backup() = 0;
  virtual Status import_backup(std::string_view document) = 0;
};

// Human wording, shared by the views.
std::string describe_ready_state(ReadyState s);  // "Ready", "Can be prepared", ...
std::string describe_compat(CompatLabel l);       // identical to to_string(CompatLabel) today; one place to change

}  // namespace clusterlm::ui
