#pragma once
// LocalCluster: Node worker processes on 127.0.0.1 standing in for remote Nodes ("localhost today, LAN
// tomorrow"). Each Node is a real clusterlm-node process reached only through the real protocol and transport;
// nothing calls across the boundary in-process. Used by ClusterLM Bench and the integration tests.
//
// "Tomorrow" is `external`: the Nodes are already running (clusterlm-node or the Node service on the other machines of the
// LAN) and the cluster only holds their endpoints and Father's identity. No process is started or controlled then, and
// everything that needs one (local activity, status line, kill, restart) reports kFailedPrecondition.
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
  double vram_gib = 0.0;   // VRAM allowance the Node offers (GPU-resident objects need it; the reference backend has no GPU)
  double disk_gib = 0.0;
  std::optional<std::string> impair;   // network preset applied on the Node side
  std::vector<std::string> faults;     // transport fault rules
  std::optional<std::string> crash_at; // lifecycle phase at which the process exits abruptly
  std::optional<std::string> hang_at;  // lifecycle phase at which the worker blocks forever (a stuck driver)
};

struct LocalClusterOptions {
  std::filesystem::path work_dir;      // staging roots and identities live here
  std::filesystem::path node_binary;   // defaults to clusterlm-node next to this executable
  bool tls = true;
  std::vector<LocalNodeOptions> nodes;
  // Extra clusterlm-node flags for every Node (backend selection: cli::node_backend_args).
  std::vector<std::string> node_args;
  // Supervised mode (HQ-REL-01): each Node is a clusterlm-node-service process (console mode, simulated activity) that
  // supervises its clusterlm-node worker, as the Windows service does with a Job Object. Local activity is then the
  // service's stdin, a worker that misses the cooperative deadline is force-terminated and relaunched by the service,
  // and the bench counts the service's events. status()/kill()/restart() are not available (the service owns the worker).
  bool supervised = false;
  std::filesystem::path service_binary;  // defaults to clusterlm-node-service next to this executable
  std::chrono::milliseconds cooperative_deadline{1500};
  // External Nodes (see above): when non-empty, `nodes` is ignored and nothing is spawned. With TLS every endpoint must
  // carry the Node's device id (its certificate fingerprint, pinned here); Father's identity is loaded from (or generated
  // into) `father_identity_dir` - keep it persistent so the Nodes can pin it too. Without TLS every endpoint must be loopback.
  std::vector<coordinator::NodeEndpoint> external;
  std::filesystem::path father_identity_dir;  // default: <work_dir>/father-id
};

// One supervisor event line of a clusterlm-node-service: worker_started | offered | revoked | forced_termination |
// worker_restarted.
struct ServiceEvent {
  std::string kind;
  double latency_ms = 0;            // revoked: activity observed -> worker Busy and clean
  std::uint64_t residual_bytes = 0; // staged bytes the worker reported at that point
};

class LocalCluster {
 public:
  static Result<std::unique_ptr<LocalCluster>> start(LocalClusterOptions options);
  ~LocalCluster();

  // Security config + endpoints for a Father Coordinator talking to this cluster.
  transport::SecurityConfig father_security() const { return father_security_; }
  std::vector<coordinator::NodeEndpoint> endpoints() const;

  std::size_t size() const { return nodes_.size(); }
  // The Nodes are processes somebody else runs (LocalClusterOptions::external).
  bool external() const { return !options_.external.empty(); }
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

  // ---- supervised mode only ----
  bool supervised() const { return options_.supervised; }
  // Reads the service's output until Node i's log holds at least `nth` events of `kind` (counting from service start), or
  // the timeout passes (kDeadlineExceeded). Events read on the way are logged too.
  Result<ServiceEvent> wait_service_event(std::size_t i, const std::string& kind, std::size_t nth, std::chrono::milliseconds timeout);
  std::size_t service_event_count(std::size_t i, const std::string& kind) const;
  const std::vector<ServiceEvent>& service_events(std::size_t i) const { return nodes_.at(i).events; }

 private:
  struct Node {
    LocalNodeOptions options;
    std::unique_ptr<platform::ChildProcess> process;
    transport::Endpoint endpoint;
    std::string device_id;
    std::filesystem::path identity_dir;
    std::uint16_t port = 0;           // supervised: the worker's fixed listen port (it is relaunched on it)
    std::vector<ServiceEvent> events; // supervised: every service event read so far
    std::filesystem::path ipc_fallback;  // supervised: short socket directory outside the work dir, removed at the end
  };
  Status launch(Node& n, const std::string& listen);
  Status launch_supervised(Node& n);
  std::vector<std::string> worker_args(const Node& n, const std::string& listen) const;
  static void log_service_line(Node& n, const std::string& line);
  static std::size_t count_events(const Node& n, const std::string& kind);
  static Result<ServiceEvent> wait_event(Node& n, const std::string& kind, std::size_t nth, std::chrono::milliseconds timeout);

  LocalClusterOptions options_;
  std::vector<Node> nodes_;
  std::shared_ptr<const transport::DeviceIdentity> father_identity_;
  transport::SecurityConfig father_security_;
};

}  // namespace clusterlm::bench
