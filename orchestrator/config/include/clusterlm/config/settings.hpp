#pragma once
// Persistent product configuration: versioned JSON documents, one for Father and one for a Node.
//
// Father document (per user, under platform::default_paths().father_root): paired Nodes (the persistent
// paired-device store), role -> machine assignments, selected tier, context preference, keep-ready policy,
// model directory per tier and advanced options. Node document (under node_root): the paired Father, the
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

namespace clusterlm::config {

inline constexpr int kFatherSettingsVersion = 1;
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

struct FatherSettings {
  std::vector<PairedDevice> paired_nodes;
  std::map<std::string, std::string> assignments;  // catalog role ("node:laptop-class") -> device fingerprint
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

}  // namespace clusterlm::config
