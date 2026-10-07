#pragma once
// LocalCluster: Node worker processes on 127.0.0.1 standing in for remote Nodes ("localhost today, LAN
// tomorrow"). Each Node is a real clusterlm-node process reached only through the real protocol and transport;
// nothing calls across the boundary in-process. Used by ClusterLM Bench and the integration tests.
#include <chrono>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "clusterlm/coordinator/coordinator.hpp"
#include "clusterlm/platform/process.hpp"
#include "clusterlm/transport/security.hpp"

namespace clusterlm::bench {

struct LocalNodeOptions {
  std::string name;
  double ram_gib = 2.0;
  double disk_gib = 0.0;
  std::optional<std::string> impair;   // network preset applied on the Node side
  std::vector<std::string> faults;     // transport fault rules
  std::optional<std::string> crash_at; // lifecycle phase at which the process exits abruptly
};

struct LocalClusterOptions {
  std::filesystem::path work_dir;      // staging roots and identities live here
  std::filesystem::path node_binary;   // defaults to clusterlm-node next to this executable
  bool tls = true;
  std::vector<LocalNodeOptions> nodes;
};

class LocalCluster {
 public:
  static Result<std::unique_ptr<LocalCluster>> start(LocalClusterOptions options);
  ~LocalCluster();

  // Security config + endpoints for a Father Coordinator talking to this cluster.
  transport::SecurityConfig father_security() const { return father_security_; }
  std::vector<coordinator::NodeEndpoint> endpoints() const;

  std::size_t size() const { return nodes_.size(); }
  // Process id of Node i's worker (changes across restart()); 0 when it is not running.
  std::int64_t pid(std::size_t i) const;
  std::filesystem::path staging_root(std::size_t i) const;
  // stdin commands to the Node process.
  Status local_activity(std::size_t i);
  Status local_idle(std::size_t i);
  struct NodeStatusLine {
    std::string state;
    std::uint64_t lease = 0, windows = 0, forwarded = 0, stale = 0, census_bytes = 0;
    bool storage_cleaned = false;
  };
  Result<NodeStatusLine> status(std::size_t i);
  // Abrupt termination (crash emulation).
  void kill(std::size_t i);
  Result<int> wait_exit(std::size_t i, std::chrono::milliseconds timeout);
  // Restart a (dead) Node on the same staging root, identity and port, without its transport fault rules;
  // orphan recovery runs on start.
  Status restart(std::size_t i, std::optional<std::string> crash_at = std::nullopt);

 private:
  struct Node {
    LocalNodeOptions options;
    std::unique_ptr<platform::ChildProcess> process;
    transport::Endpoint endpoint;
    std::string device_id;
    std::filesystem::path identity_dir;
  };
  Status launch(Node& n, const std::string& listen);

  LocalClusterOptions options_;
  std::vector<Node> nodes_;
  std::shared_ptr<const transport::DeviceIdentity> father_identity_;
  transport::SecurityConfig father_security_;
};

}  // namespace clusterlm::bench
