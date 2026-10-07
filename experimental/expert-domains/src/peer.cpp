#include "clusterlm/expert_domains/peer.hpp"

#include <algorithm>
#include <thread>

#include "clusterlm/common/log.hpp"
#include "clusterlm/expert_domains/domain_server.hpp"
#include "clusterlm/expert_domains/rig.hpp"
#include "clusterlm/objects/canonical_store.hpp"

namespace clusterlm::expert_domains {

using namespace std::chrono_literals;

Result<PeerSpec> parse_peer(const std::string& text) {
  const auto eq = text.find('=');
  if (eq == std::string::npos || eq == 0) return make_error(ErrorCode::kInvalidArgument, "peer must be NAME=HOST:PORT[@FINGERPRINT]");
  PeerSpec p;
  p.name = text.substr(0, eq);
  std::string rest = text.substr(eq + 1);
  if (const auto at = rest.find('@'); at != std::string::npos) {
    p.expected_id = rest.substr(at + 1);
    rest.resize(at);
    if (p.expected_id->empty()) return make_error(ErrorCode::kInvalidArgument, "empty fingerprint after '@'");
  }
  CLM_ASSIGN_OR_RETURN(p.endpoint, transport::Endpoint::parse(rest));
  return p;
}

Result<std::vector<RemoteLink>> connect_peers(const std::vector<PeerSpec>& peers, const transport::SecurityConfig& security,
                                              std::chrono::milliseconds timeout) {
  std::vector<RemoteLink> links;
  for (const PeerSpec& p : peers) {
    auto c = transport::connect(p.endpoint, security, p.expected_id, timeout);
    if (!c.is_ok()) return make_error(c.status().code(), "expert domain peer '" + p.name + "' (" + p.endpoint.str() + "): " + c.status().message());
    links.push_back(RemoteLink{p.name, std::move(c).value()});
  }
  return links;
}

Status serve_peer(const PeerServeOptions& o) {
  if (o.domain_index == 0 || o.domain_index > o.remote_domains)
    return make_error(ErrorCode::kInvalidArgument, "--domain-index must be in 1..remote-domains");
  std::filesystem::path model_dir = o.model_dir;
  if (model_dir.empty()) {
    if (o.work_dir.empty()) return make_error(ErrorCode::kInvalidArgument, "need a model directory or a work directory");
    model_dir = o.work_dir / "fixture-model";
    CLM_RETURN_IF_ERROR(objects::write_fixture_model(o.fixture, model_dir).status());
  }
  CLM_ASSIGN_OR_RETURN(auto store, objects::CanonicalModelStore::open(model_dir));
  const objects::ModelManifest& manifest = store->manifest();
  CLM_ASSIGN_OR_RETURN(auto assignment, make_expert_assignment(manifest.geometry.n_experts, o.remote_domains, o.strided, o.shares));

  // The decode limits must admit every batch Father can send; they follow the same rule as GroupedRig.
  ExpertDomainConfig dc;
  dc.name = "domain" + std::to_string(o.domain_index);
  dc.owned_experts = assignment.owned[o.domain_index];
  dc.epoch = o.epoch;
  dc.kernel = o.kernel;
  dc.limits.max_positions = std::max<std::uint32_t>(dc.limits.max_positions, o.max_window);
  dc.limits.max_hidden = std::max(dc.limits.max_hidden, manifest.geometry.hidden_size);
  dc.limits.max_layer = std::max(dc.limits.max_layer, manifest.geometry.n_layers);
  CLM_ASSIGN_OR_RETURN(auto resolver,
                       provision_expert_objects(*store, dc.owned_experts, 0, manifest.geometry.n_layers));
  CLM_ASSIGN_OR_RETURN(auto server, ExpertDomainServer::create(manifest, *resolver, dc));
  CLM_ASSIGN_OR_RETURN(auto listener, transport::listen(o.listen, o.security));
  if (o.on_listening)
    o.on_listening(listener->local_endpoint(), o.security.identity ? o.security.identity->fingerprint() : std::string());

  std::uint32_t sessions = 0;
  while ((o.stop == nullptr || !o.stop->load()) && (o.max_sessions == 0 || sessions < o.max_sessions)) {
    auto conn = listener->accept(200ms);
    if (!conn.is_ok()) {
      // A timeout or a peer that failed authentication: keep listening. Anything else ends the service.
      const ErrorCode c = conn.status().code();
      if (c == ErrorCode::kDeadlineExceeded || c == ErrorCode::kUnauthenticated) continue;
      return conn.status();
    }
    CLM_RETURN_IF_ERROR(server->serve(std::move(conn).value()));
    while (server->serving() && (o.stop == nullptr || !o.stop->load())) std::this_thread::sleep_for(20ms);
    server->stop();
    ++sessions;
  }
  listener->close();
  return Status::ok();
}

}  // namespace clusterlm::expert_domains
