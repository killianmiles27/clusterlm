#pragma once
// Persistent product configuration: versioned JSON documents, one for Father and one for a Node.
//
// Father document (per user, under platform::default_paths().father_root), version 2 (profile-schema-v1.md §5): paired
// Nodes (the persistent paired-device store), machine bindings (v1 "assignments"), the execution profiles and routing
// aliases, the Host model library, the selected profile, context preference and advanced options. The tier-era values
// (selected tier, keep-ready policy, model directory per tier, user-confirmed roots) stay in the struct as the LEGACY
// TIER VIEW and are persisted under "legacy_v1" so the existing tier code keeps working unchanged and nothing a v1
// installation held is lost; the profile layer is authoritative for new code (see docs/adr/0408). Node document (under node_root): the paired Father, the
// participation policy (allow when idle, idle seconds, AC only, temporary storage limit, paused, startup) and
// resource caps.
//
// A document never contains secrets (identities live in their own owner-only key files), prompts, text or
// tokens. Schema rules: `version` (integer) is required; unknown fields are ignored (forward compatible within
// a version); every known field is range checked. See store.hpp for loading, saving, recovery and migration.
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "clusterlm/common/status.hpp"
#include "clusterlm/library/library.hpp"
#include "clusterlm/profiles/profile.hpp"

namespace clusterlm::config {

inline constexpr int kFatherSettingsVersion = 2;
inline constexpr int kNodeSettingsVersion = 1;

// A device this machine has paired with. The fingerprint (SHA-256 of the device certificate, lowercase hex) is
// the identity that is pinned; name/address/role are labels and routing hints, never trust inputs.
struct PairedDevice {
  std::string fingerprint;           // 64 lowercase hex chars
  std::string name;                  // user-visible label ("G14")
  std::string address;               // Father->Node: the Node's data endpoint "host:port"; empty for a paired Father
  std::string role;                  // "node" or "father"
  std::int64_t paired_at_unix = 0;   // seconds; informational
  friend bool operator==(const PairedDevice&, const PairedDevice&) = default;
};

struct KeepReadyPolicy {
  bool enabled = false;            // keep the selected tier prepared between requests
  std::uint32_t release_after_idle_minutes = 30;  // release leases after this much inactivity (0 = never)
  friend bool operator==(const KeepReadyPolicy&, const KeepReadyPolicy&) = default;
};

struct AdvancedOptions {
  std::string bench_results_dir;     // where Measured profiles are looked up ("" = none)
  std::string log_level = "info";    // debug | info | warn | error
  bool direct_peer = true;           // Node->Node activation forwarding instead of Father relay
  std::uint32_t default_q = 1;       // speculative verification width
  std::uint32_t prefill_chunk = 128;
  friend bool operator==(const AdvancedOptions&, const AdvancedOptions&) = default;
};

// A profile / alias / library document that failed validation on load. Kept verbatim and never rewritten or deleted
// (profile-schema-v1.md §1a): it can never make the whole settings file count as corrupt and lose the pairing records.
struct QuarantinedDocument {
  std::string kind;      // "profile" | "alias" | "library"
  std::string reason;    // short, no document content
  std::string document;  // the original text
  friend bool operator==(const QuarantinedDocument&, const QuarantinedDocument&) = default;
};

inline constexpr std::size_t kMaxQuarantined = 64;
inline constexpr std::size_t kMaxPreservedBytes = 64u << 10;

struct FatherSettings {
  std::vector<PairedDevice> paired_nodes;
  // v2 "machine_bindings": binding name (v1 catalog role such as "node:laptop-class") -> device fingerprint. The field
  // keeps its v1 name so existing code compiles unchanged.
  std::map<std::string, std::string> assignments;
  std::string selected_profile_id;  // "" until the example set is seeded; otherwise an existing profile id
  std::vector<profiles::Profile> profiles;
  std::vector<profiles::RoutingAlias> aliases;
  std::vector<QuarantinedDocument> quarantine;
  library::ModelLibrary library;
  bool examples_seeded = false;  // true once the Fast/Strong/Ultra example set was added (never re-added after a delete)
  // Unknown top-level fields of the document, preserved verbatim (raw JSON text per key) so a newer or sibling feature's
  // data (e.g. the serving section) survives a rewrite by this build.
  std::map<std::string, std::string> preserved;
  // ---- legacy tier view (persisted under "legacy_v1") ----
  std::string selected_tier = "fast";
  std::uint32_t context_tokens = 4096;
  KeepReadyPolicy keep_ready;
  std::map<std::string, std::string> model_dirs;   // tier id -> directory holding the manifest + shards
  // tier id -> manifest root hash (hex) the user confirmed after inspection (catalog models start unpinned)
  std::map<std::string, std::string> confirmed_model_roots;
  AdvancedOptions advanced;

  const PairedDevice* find_node(std::string_view fingerprint) const;
  // Adds or replaces (by fingerprint).
  void upsert_node(PairedDevice device);
  // Removes the node and every assignment naming it. False if unknown.
  bool remove_node(std::string_view fingerprint);
  // Selects a tier through both views: the legacy tier name and, when the example profile exists, the profile id.
  void select_tier(const std::string& tier);
  const profiles::Profile* find_profile(std::string_view id) const;
  friend bool operator==(const FatherSettings&, const FatherSettings&) = default;
};

struct ResourceCaps {
  std::uint32_t ram_gib = 4;       // memory the Node may commit to leases
  std::uint32_t vram_gib = 0;
  std::uint32_t threads = 0;       // 0 = automatic. Recorded; the worker does not enforce a thread cap yet.
  friend bool operator==(const ResourceCaps&, const ResourceCaps&) = default;
};

struct NodeSettings {
  std::string name = "node";
  std::optional<PairedDevice> paired_father;
  bool allow_when_idle = true;       // participate when the machine is idle
  std::uint32_t idle_seconds = 300;
  bool ac_only = true;
  std::uint32_t temp_storage_limit_gib = 0;  // 0 = no explicit cap (the lease admission still applies)
  bool paused = false;
  bool start_with_system = true;     // service startup
  ResourceCaps caps;
  friend bool operator==(const NodeSettings&, const NodeSettings&) = default;
};

// Structural and range validation. These run on load (a failing document is treated as corrupt) and before
// every save (a failing update is refused and nothing is written).
bool is_fingerprint(std::string_view text);  // 64 lowercase hex characters
Status validate(const PairedDevice& device);
Status validate(const FatherSettings& settings);
Status validate(const NodeSettings& settings);

std::string to_json(const FatherSettings& settings);
std::string to_json(const NodeSettings& settings);
// Parse + validate. The version is checked by the store; here a missing or wrong-typed field is an error.
Result<FatherSettings> father_settings_from_json(std::string_view json);
Result<NodeSettings> node_settings_from_json(std::string_view json);

// Adds the Fast/Strong/Ultra example set to a settings value that never had it (fresh install), selects the profile for
// `selected_tier`, applies the legacy keep-ready policy to the lifecycles and sets examples_seeded. A no-op when already
// seeded. Pure: no I/O.
void ensure_example_set(FatherSettings& settings);

}  // namespace clusterlm::config
