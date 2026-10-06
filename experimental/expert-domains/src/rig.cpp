#include "clusterlm/expert_domains/rig.hpp"

#include <algorithm>

namespace clusterlm::expert_domains {

using namespace std::chrono_literals;

namespace {

struct ConnectionPair {
  std::unique_ptr<transport::Connection> father_side;
  std::unique_ptr<transport::Connection> domain_side;
};

}  // namespace

Result<std::pair<std::unique_ptr<transport::Connection>, std::unique_ptr<transport::Connection>>> make_loopback_pair() {
  transport::SecurityConfig sec;
  sec.mode = transport::SecurityConfig::Mode::kInsecureLoopbackOnly;
  CLM_ASSIGN_OR_RETURN(auto listener, transport::listen(transport::Endpoint{"127.0.0.1", 0}, sec));
  // A bound listener accepts the TCP handshake in the kernel, so connect-then-accept needs no helper thread.
  CLM_ASSIGN_OR_RETURN(auto client, transport::connect(listener->local_endpoint(), sec, std::nullopt, 5s));
  CLM_ASSIGN_OR_RETURN(auto server, listener->accept(5s));
  listener->close();
  return std::make_pair(std::move(client), std::move(server));
}

Result<std::unique_ptr<GroupedRig>> GroupedRig::create(const RigOptions& o) {
  if (o.store == nullptr) return make_error(ErrorCode::kInvalidArgument, "RigOptions::store is required");
  if (o.remote_domains == 0) return make_error(ErrorCode::kInvalidArgument, "at least one remote domain");
  const objects::ModelManifest& manifest = o.store->manifest();
  const std::uint32_t owners = o.remote_domains + 1;
  std::unique_ptr<GroupedRig> rig(new GroupedRig());
  if (!o.explicit_owner_of.empty()) {
    CLM_ASSIGN_OR_RETURN(rig->assignment_, ExpertAssignment::from_owners(o.explicit_owner_of, owners));
  } else if (o.strided) {
    CLM_ASSIGN_OR_RETURN(rig->assignment_, ExpertAssignment::strided(manifest.geometry.n_experts, owners));
  } else {
    std::vector<std::uint32_t> shares = o.shares.empty() ? std::vector<std::uint32_t>(owners, 1) : o.shares;
    if (shares.size() != owners) return make_error(ErrorCode::kInvalidArgument, "one share per owner is required");
    CLM_ASSIGN_OR_RETURN(rig->assignment_, ExpertAssignment::ranges(manifest.geometry.n_experts, shares));
  }

  ExpertDecodeLimits limits;
  limits.max_positions = std::max<std::uint32_t>(limits.max_positions, o.max_window);
  limits.max_hidden = std::max(limits.max_hidden, manifest.geometry.hidden_size);
  limits.max_layer = std::max(limits.max_layer, manifest.geometry.n_layers);

  // Father's egress NIC: one shared link for every Father -> domain connection.
  std::shared_ptr<transport::SimulatedLink> egress;
  if (o.network) egress = std::make_shared<transport::SimulatedLink>(o.network->bandwidth_bytes_per_s);

  std::vector<RemoteLink> links;
  for (std::uint32_t r = 0; r < o.remote_domains; ++r) {
    ExpertDomainConfig dc;
    dc.name = "domain" + std::to_string(r + 1);
    dc.owned_experts = rig->assignment_.owned[r + 1];
    dc.epoch = o.epoch;
    dc.limits = limits;
    if (r < o.die_after_batches.size()) dc.die_after_batches = o.die_after_batches[r];
    CLM_ASSIGN_OR_RETURN(auto resolver, provision_expert_objects(*o.store, dc.owned_experts, 0, manifest.geometry.n_layers));
    CLM_ASSIGN_OR_RETURN(auto server, ExpertDomainServer::create(manifest, *resolver, dc));
    CLM_ASSIGN_OR_RETURN(auto raw, make_loopback_pair());
    ConnectionPair pair{std::move(raw.first), std::move(raw.second)};
    if (o.network) {
      transport::NetworkConditions fc = *o.network, dcnd = *o.network;
      fc.seed += 2 * r;       // independent jitter streams per direction
      dcnd.seed += 2 * r + 1;
      pair.father_side = transport::impair(std::move(pair.father_side), fc, egress);
      pair.domain_side = transport::impair(std::move(pair.domain_side), dcnd);  // the domain's own NIC
    }
    CLM_RETURN_IF_ERROR(server->serve(std::move(pair.domain_side)));
    links.push_back(RemoteLink{dc.name, std::move(pair.father_side)});
    rig->resolvers_.push_back(std::move(resolver));
    rig->servers_.push_back(std::move(server));
  }

  FatherExecutorConfig fc;
  fc.assignment = rig->assignment_;
  fc.max_context = o.max_context;
  fc.max_window = o.max_window;
  fc.epoch = o.epoch;
  fc.limits = limits;
  fc.layer_timeout = o.layer_timeout;
  fc.overlap_local = o.overlap_local;
  CLM_ASSIGN_OR_RETURN(rig->executor_, FatherExecutor::create(manifest, *o.store, std::move(fc), std::move(links)));
  return rig;
}

void GroupedRig::shutdown() {
  if (executor_) executor_->close();
  for (auto& s : servers_) s->stop();
}

GroupedRig::~GroupedRig() { shutdown(); }

}  // namespace clusterlm::expert_domains
