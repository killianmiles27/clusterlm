#pragma once
// Tier readiness: a pure function of observed inputs. It is deliberately conservative:
//   Ready only if the model is verified and pinned/confirmed, the backend can run, every required machine is
//   paired, powered and in a usable state, a feasible placement exists for the requested context, the current
//   plan/lease is ready, and EVERY required domain is Ready under that plan. Anything less is Preparing,
//   Available (could be prepared) or Unavailable, with a user-readable reason for each blocker.
// ETAs are estimates and say so; a missing rate yields no ETA rather than a guess.
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "clusterlm/catalog/catalog.hpp"

namespace clusterlm::catalog {

// Mirrors node::NodeState plus Offline (unreachable / unpaired). Kept here so the catalog layer does not link
// the Node worker.
enum class MachineState : std::uint8_t {
  kBusy, kAvailable, kPreparing, kReady, kInferencing, kReleasing, kCleanupPending, kOffline,
};
std::string_view to_string(MachineState s) noexcept;

enum class TierState : std::uint8_t { kUnavailable, kAvailable, kPreparing, kReady };
std::string_view to_string(TierState s) noexcept;

struct ModelAvailability {
  bool manifest_present = false;
  bool hashes_verified = false;
  std::string manifest_root_hex;        // root hash of the manifest found locally ("" if none)
  std::string user_confirmed_root_hex;  // root the user confirmed after inspection ("" if none)
};

struct BackendAvailability {
  std::string name;
  bool hardware_available = false;  // BackendInfo::hardware_available
};

struct PowerState {
  bool on_ac = true;
  bool battery_saver = false;
};

struct MachineInputs {
  std::string role;           // catalog role this machine fills
  std::string machine_id;     // concrete machine from the TierAssignment; shown to the user
  bool paired = true;
  MachineState state = MachineState::kOffline;
  PowerState power;
};

struct ProvisioningProgress {
  std::uint64_t bytes_done = 0;
  std::uint64_t bytes_total = 0;
  std::optional<double> rate_bytes_per_s;  // measured or estimated transfer rate
  bool rate_is_measured = false;
};

struct PlanReadiness {
  bool feasible_for_context = false;  // a placement exists for the requested context
  bool plan_ready = false;            // current lease/plan is prepared and current for every domain
};

struct ReadinessInputs {
  std::uint32_t context_tokens = 4096;
  ModelAvailability model;
  BackendAvailability backend;
  std::vector<MachineInputs> machines;
  std::optional<ProvisioningProgress> provisioning;
  PlanReadiness plan;
};

struct PrepareProgress {
  double percent = 0;                        // 0..100
  std::optional<double> eta_seconds;         // always an estimate
  bool eta_is_estimate = true;
};

struct TierReadiness {
  std::string tier_id;
  std::string model_name;                    // catalog identity of the model this tier would run
  TierState state = TierState::kUnavailable;
  std::vector<std::string> reasons;          // user-readable; first is the headline
  std::vector<std::string> notes;            // non-blocking (e.g. context not yet qualified)
  std::optional<PrepareProgress> progress;   // Preparing only
  std::vector<std::string> suggested_fallback;  // viable lower tiers (Ready or Available), catalog order
  std::string headline() const { return reasons.empty() ? std::string() : reasons.front(); }
};

TierReadiness evaluate(const TierEntry& tier, const ReadinessInputs& inputs);

// Evaluates every tier and fills `suggested_fallback` from the catalog order. `inputs` maps tier id -> inputs;
// a tier without inputs is Unavailable("no readiness information").
TierReadiness evaluate_with_fallback(const Catalog& catalog, const TierEntry& tier,
                                     const std::vector<std::pair<std::string, ReadinessInputs>>& inputs);

// "about 3 min", "under a minute", "about 1 h 5 min".
std::string format_duration(double seconds);

}  // namespace clusterlm::catalog
