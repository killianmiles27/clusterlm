#pragma once
// Production providers for FatherService: they turn persisted configuration (paired Nodes, role assignments,
// model directories, advanced options) and observation of the real world into the two seams the service uses.
//
//   ConfigDeploymentProvider  tier + assignment + settings + placement search -> Deployment (coordinator config
//                             with the paired Nodes' endpoints and PINNED identities, plus the ClusterPlan)
//   LiveReadinessSource       model directory / manifest / hash verification, backend availability, Node state
//                             through the control channel (offers), power -> catalog::ReadinessInputs
//
// Honesty rules enforced here:
//   * Placement uses the best available profiles: Measured files from the bench results directory when present,
//     else the Synthetic development fixtures. Which one was used is recorded on the shared DetailsBoard and
//     surfaced next to the tier ("placement uses SYNTHETIC profiles ...").
//   * No real inference backend (Strata/llama) is built into this binary, so a tier is never Ready unless the
//     explicit development override (--dev-fixture-model, reference backend on a fixture model) is on, and then
//     the tier says so in its details.
#include <atomic>
#include <chrono>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "clusterlm/config/store.hpp"
#include "clusterlm/father/service.hpp"
#include "clusterlm/transport/security.hpp"

namespace clusterlm::father {

// Human-readable facts about a tier that are not part of catalog readiness (profile provenance, backend notes).
class DetailsBoard {
 public:
  void set(const std::string& tier_id, std::string key, std::string text);
  std::vector<std::string> get(const std::string& tier_id) const;

 private:
  mutable std::mutex mu_;
  std::map<std::string, std::map<std::string, std::string>> notes_;
};

// What the Father service is doing with its Nodes; a probe must never disturb an active session, and a Node that is
// part of one is reported in the matching state instead of being probed.
enum class SessionPhase : std::uint8_t { kNone, kPreparing, kReadyIdle, kInferencing };

// The latest preparation progress per tier. It is the production wiring between the service (which receives the
// Coordinator's progress while a tier is being prepared) and LiveReadinessSource (which reports a tier Preparing with
// a percentage and an ETA while bytes are still going out):
//
//   auto board = std::make_shared<ProvisioningBoard>();
//   ProductionOptions po;  po.provisioning = board->provider();
//   ServiceDeps deps;      deps.prepare_observer = board->observer();
//
// The rate is measured on this run (bytes sent over elapsed time once at least half a second of data exists); before
// that no rate is reported and the UI shows no ETA. An entry is removed when the prepare ends, whatever the outcome,
// so a failed prepare can never leave a tier stuck in Preparing. Thread-safe.
class ProvisioningBoard {
 public:
  void update(const std::string& tier_id, const coordinator::PrepareProgress& progress);
  void clear(const std::string& tier_id);
  std::optional<catalog::ProvisioningProgress> get(const std::string& tier_id) const;
  PrepareObserver observer();
  std::function<std::optional<catalog::ProvisioningProgress>(const std::string&)> provider();

 private:
  struct Entry {
    catalog::ProvisioningProgress progress;
    std::chrono::steady_clock::time_point started{};
    std::uint64_t bytes_at_start = 0;
    bool started_set = false;
  };
  mutable std::mutex mu_;
  std::map<std::string, Entry> entries_;
};

struct ProductionOptions {
  std::shared_ptr<config::FatherSettingsStore> settings;
  std::shared_ptr<const transport::DeviceIdentity> identity;
  std::filesystem::path profiles_dir;  // Synthetic fallback fixtures (Father-*.json, Node-*.json, network-*.json)
  // DEVELOPMENT ONLY: the model directory configured for a tier holds a fixture model that the reference
  // backend can run. Marks the backend available, auto-confirms the unpinned model and labels everything.
  bool dev_fixture_model = false;
  std::shared_ptr<DetailsBoard> details = std::make_shared<DetailsBoard>();
  std::function<SessionPhase()> session_phase = [] { return SessionPhase::kNone; };
  std::function<catalog::PowerState()> father_power = [] { return catalog::PowerState{}; };
  std::function<std::optional<catalog::ProvisioningProgress>(const std::string& tier_id)> provisioning;
  std::chrono::milliseconds probe_timeout{600};
  std::chrono::milliseconds probe_ttl{1500};
};

class ConfigDeploymentProvider final : public DeploymentProvider {
 public:
  explicit ConfigDeploymentProvider(ProductionOptions options) : opt_(std::move(options)) {}
  Result<Deployment> resolve(const catalog::TierEntry& tier, std::uint32_t context_tokens) override;

 private:
  ProductionOptions opt_;
};

class LiveReadinessSource final : public ReadinessSource {
 public:
  explicit LiveReadinessSource(ProductionOptions options) : opt_(std::move(options)) {}
  ~LiveReadinessSource() override;
  catalog::ReadinessInputs observe(const catalog::TierEntry& tier, const catalog::TierAssignment& assignment,
                                   std::uint32_t context_tokens) override;

 private:
  struct Verify {
    bool done = false, ok = false;
    std::string root_hex;
  };
  struct Probe {
    catalog::MachineState state = catalog::MachineState::kOffline;
    bool on_ac = true;
    std::chrono::steady_clock::time_point at{};
    bool valid = false;
  };
  Verify verification(const std::filesystem::path& dir, const std::string& root_hex);
  Probe probe_node(const config::PairedDevice& node);

  ProductionOptions opt_;
  std::mutex mu_;
  std::map<std::string, Verify> verified_;
  std::map<std::string, std::thread> verifiers_;
  std::map<std::string, Probe> probes_;
  std::map<std::string, bool> feasible_;  // tier|ctx -> last placement feasibility (computed lazily)
  std::atomic<bool> stopping_{false};
};

}  // namespace clusterlm::father
