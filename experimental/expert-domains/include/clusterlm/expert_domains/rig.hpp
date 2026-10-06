#pragma once
// GroupedRig: an in-process grouped expert-domain cluster for tests and the simulation bench.
//
// Father's FatherExecutor plus R ExpertDomainServer threads, connected by REAL transport connections (loopback
// TCP, insecure mode — the same `transport::Connection` a LAN/TLS link would give) optionally wrapped in the
// network impairment layer. Each domain's resolver is plan-scoped: it holds only that domain's routed experts.
// Father's egress NIC is one SimulatedLink shared by every Father->domain connection, as in the real cluster.
#include <memory>
#include <optional>
#include <vector>

#include "clusterlm/expert_domains/domain_server.hpp"
#include "clusterlm/expert_domains/father_executor.hpp"
#include "clusterlm/transport/impairment.hpp"

namespace clusterlm::expert_domains {

struct RigOptions {
  const objects::CanonicalModelStore* store = nullptr;  // Father's canonical store (required)
  std::uint32_t remote_domains = 2;
  std::vector<std::uint32_t> shares;  // per owner (Father first); empty = equal
  bool strided = false;               // round-robin ownership instead of contiguous ranges
  std::vector<std::uint32_t> explicit_owner_of;  // if non-empty: owner per expert (overrides shares/strided)
  std::optional<transport::NetworkConditions> network;  // SIMULATION ONLY; empty = raw loopback
  std::uint32_t max_context = 256;
  std::uint32_t max_window = 8;
  bool overlap_local = true;
  std::chrono::milliseconds layer_timeout{5000};
  std::vector<std::uint32_t> die_after_batches;  // per remote domain (test fault injection); empty = never
  Epoch epoch{1};
};

// A connected loopback TCP pair (insecure mode): {Father side, domain side}. Tests and the rig use it; a LAN
// deployment would use transport::connect/listen with mutual TLS instead.
Result<std::pair<std::unique_ptr<transport::Connection>, std::unique_ptr<transport::Connection>>> make_loopback_pair();

class GroupedRig {
 public:
  static Result<std::unique_ptr<GroupedRig>> create(const RigOptions& options);
  ~GroupedRig();

  FatherExecutor& executor() { return *executor_; }
  ExpertDomainServer& server(std::size_t i) { return *servers_[i]; }
  std::size_t remote_count() const { return servers_.size(); }
  const ExpertAssignment& assignment() const { return assignment_; }
  // Stops every domain thread and closes Father's links (idempotent).
  void shutdown();

 private:
  GroupedRig() = default;
  ExpertAssignment assignment_;
  std::vector<std::unique_ptr<objects::InMemoryResolver>> resolvers_;
  std::vector<std::unique_ptr<ExpertDomainServer>> servers_;
  std::unique_ptr<FatherExecutor> executor_;
};

}  // namespace clusterlm::expert_domains
