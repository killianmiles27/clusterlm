#pragma once
// Dynamic topology: turns a profile's slots + selectors into concrete Workers, given the Host's machine bindings and the
// Workers' last advertised capabilities (profile-schema-v1.md §2). A pure function: no I/O, no clocks, deterministic.
//
// Rules (frozen):
//   * a Worker that is Busy, Offline, Paused, Releasing, or on battery (unless allowed) does NOT satisfy any selector
//     ("a busy machine is not available compute");
//   * one machine fills at most one slot of a profile;
//   * binding / machine slots resolve first, in slot order; requirements slots then resolve over the machines not yet
//     chosen, picking the Worker with the most free VRAM, then the lowest fingerprint (deterministic);
//   * a binding with no entry in the Host's bindings is "not assigned": never auto-bound;
//   * an optional slot may stay empty while the filled count stays within [min_workers, max_workers];
//   * capability values carry provenance; unknown values never satisfy a requirement ("link speed unknown").
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "clusterlm/profiles/profile.hpp"

namespace clusterlm::profiles {

struct GpuInfo {
  std::string vendor;  // nvidia | amd | intel | other
  std::string name;
  std::uint32_t vram_mib = 0;
  std::string compute_capability;  // "8.6"; empty when not applicable / unknown
  friend bool operator==(const GpuInfo&, const GpuInfo&) = default;
};

enum class WorkerAvailability : std::uint8_t { kAvailable, kBusy, kOffline, kPaused, kReleasing };
std::string_view to_string(WorkerAvailability a) noexcept;

// What a Worker advertises about itself. Capability data is declared by the Worker (provenance "declared") or taken from a
// bench run on it ("measured"); it is never a promise that a model fits (placement decides that).
struct WorkerCapability {
  std::string fingerprint;  // 64 lowercase hex; the pinned identity
  std::string name;         // display label
  std::uint32_t ram_mib = 0;
  std::uint32_t cpu_threads = 0;
  std::vector<GpuInfo> gpus;
  std::uint32_t link_mbit = 0;        // Host<->Worker link; 0 = unknown
  std::vector<std::string> backends;  // descriptor ids built into the Worker and with their runtime present
  std::string build_hash;             // must equal the Host's for a plan to be admitted
  std::string provenance = "declared";  // declared | measured
  std::string source;                   // e.g. "worker advertisement", "bench:Node-g14.json"
  WorkerAvailability availability = WorkerAvailability::kAvailable;
  bool on_battery = false;
  friend bool operator==(const WorkerCapability&, const WorkerCapability&) = default;
};

Status validate(const WorkerCapability& c);
std::string to_json(const WorkerCapability& c);
Result<WorkerCapability> worker_capability_from_json(std::string_view json);

using BindingMap = std::map<std::string, std::string>;  // binding name -> device fingerprint

struct ResolveOptions {
  std::string backend_id;            // when set, a Worker must advertise this backend
  bool allow_on_battery = false;     // Worker power policy normally forbids battery (node ac_only)
};

struct SlotResolution {
  enum class State : std::uint8_t { kFilled, kEmptyOptional, kUnsatisfied } state = State::kUnsatisfied;
  std::string slot;
  std::string label;
  std::optional<std::string> fingerprint;  // kFilled
  std::string reason;                      // why not filled (user-readable); empty when filled
};

struct TopologyResolution {
  std::vector<SlotResolution> slots;  // one per worker slot, in slot order
  std::uint32_t filled = 0;
  bool satisfiable = false;           // required slots filled and filled within [min,max]
  std::string headline;               // first blocker, empty when satisfiable
  std::vector<std::string> fingerprints() const;  // chosen Workers in stage order
};

// Does `cap` meet `req`? On failure `why` names the first unmet requirement.
bool meets_requirements(const WorkerCapability& cap, const Requirements& req, std::string* why = nullptr);
// Numeric "8.6" >= "7.5" comparison; false for malformed input.
bool compute_capability_at_least(std::string_view have, std::string_view need);

TopologyResolution resolve_topology(const Topology& topology, const BindingMap& bindings,
                                    const std::vector<WorkerCapability>& workers, const ResolveOptions& options = {});

}  // namespace clusterlm::profiles
