#include "local_cluster.hpp"

#include <sstream>
#include <thread>

namespace clusterlm::bench {

using namespace std::chrono_literals;

namespace {
std::string field(const std::string& line, const std::string& key) {
  std::istringstream ss(line);
  std::string tok;
  while (ss >> tok)
    if (tok.rfind(key + "=", 0) == 0) return tok.substr(key.size() + 1);
  return {};
}
}  // namespace

Result<std::unique_ptr<LocalCluster>> LocalCluster::start(LocalClusterOptions options) {
  auto c = std::unique_ptr<LocalCluster>(new LocalCluster());
  c->options_ = std::move(options);
  if (c->options_.node_binary.empty()) {
#ifdef _WIN32
    c->options_.node_binary = platform::executable_dir() / "clusterlm-node.exe";
#else
    c->options_.node_binary = platform::executable_dir() / "clusterlm-node";
#endif
  }
  std::error_code ec;
  std::filesystem::create_directories(c->options_.work_dir, ec);
  if (c->options_.tls) {
    CLM_ASSIGN_OR_RETURN(auto id, transport::DeviceIdentity::load_or_generate(c->options_.work_dir / "father-id", "father"));
    c->father_identity_ = std::make_shared<const transport::DeviceIdentity>(std::move(id));
    c->father_security_.mode = transport::SecurityConfig::Mode::kMutualTls;
    c->father_security_.identity = c->father_identity_;
  } else {
    c->father_security_.mode = transport::SecurityConfig::Mode::kInsecureLoopbackOnly;
  }
  for (auto& opt : c->options_.nodes) {
    Node n;
    n.options = opt;
    n.identity_dir = c->options_.work_dir / (opt.name + "-id");
    CLM_RETURN_IF_ERROR(c->launch(n, "127.0.0.1:0"));
    // Pairing: Father pins each Node identity.
    if (c->options_.tls) c->father_security_.trusted_peers.push_back(n.device_id);
    c->nodes_.push_back(std::move(n));
  }
  return c;
}

Status LocalCluster::launch(Node& n, const std::string& listen) {
  std::vector<std::string> args = {"--name", n.options.name, "--listen", listen, "--staging",
                                   (options_.work_dir / (n.options.name + "-staging")).string(), "--ram-gib",
                                   std::to_string(n.options.ram_gib), "--disk-gib", std::to_string(n.options.disk_gib),
                                   "--log", "warn"};
  if (options_.tls) {
    args.insert(args.end(), {"--identity", n.identity_dir.string(), "--trust", father_identity_->fingerprint()});
  } else {
    args.push_back("--insecure-loopback");
  }
  if (n.options.impair) args.insert(args.end(), {"--impair", *n.options.impair});
  for (const auto& f : n.options.faults) args.insert(args.end(), {"--fault", f});
  if (n.options.crash_at) args.insert(args.end(), {"--crash-at", *n.options.crash_at});
  CLM_ASSIGN_OR_RETURN(n.process, platform::ChildProcess::spawn(options_.node_binary, args));
  CLM_ASSIGN_OR_RETURN(auto line, n.process->read_until("CLUSTERLM_NODE_LISTENING", 10s));
  CLM_ASSIGN_OR_RETURN(n.endpoint, transport::Endpoint::parse(field(line, "endpoint")));
  n.device_id = field(line, "device_id");
  return Status::ok();
}

LocalCluster::~LocalCluster() {
  for (auto& n : nodes_)
    if (n.process) (void)n.process->write_line("quit");
}

std::vector<coordinator::NodeEndpoint> LocalCluster::endpoints() const {
  std::vector<coordinator::NodeEndpoint> out;
  for (const auto& n : nodes_)
    out.push_back({n.options.name, n.endpoint, options_.tls ? n.device_id : std::string()});
  return out;
}

std::filesystem::path LocalCluster::staging_root(std::size_t i) const {
  return options_.work_dir / (nodes_.at(i).options.name + "-staging");
}

std::int64_t LocalCluster::pid(std::size_t i) const {
  return i < nodes_.size() && nodes_[i].process ? nodes_[i].process->pid() : 0;
}

Status LocalCluster::local_activity(std::size_t i) { return nodes_.at(i).process->write_line("activity"); }
Status LocalCluster::local_idle(std::size_t i) { return nodes_.at(i).process->write_line("idle"); }

Result<LocalCluster::NodeStatusLine> LocalCluster::status(std::size_t i) {
  auto& p = *nodes_.at(i).process;
  CLM_RETURN_IF_ERROR(p.write_line("status"));
  CLM_ASSIGN_OR_RETURN(auto line, p.read_until("CLUSTERLM_NODE_STATUS", 5s));
  NodeStatusLine s;
  s.state = field(line, "state");
  s.lease = std::stoull(field(line, "lease"));
  s.windows = std::stoull(field(line, "windows"));
  s.forwarded = std::stoull(field(line, "forwarded"));
  s.stale = std::stoull(field(line, "stale"));
  s.census_bytes = std::stoull(field(line, "census_bytes"));
  s.storage_cleaned = field(line, "storage_cleaned") == "1";
  return s;
}

void LocalCluster::kill(std::size_t i) { nodes_.at(i).process->kill(); }

Result<int> LocalCluster::wait_exit(std::size_t i, std::chrono::milliseconds timeout) {
  return nodes_.at(i).process->wait(timeout);
}

Status LocalCluster::restart(std::size_t i, std::optional<std::string> crash_at) {
  auto& n = nodes_.at(i);
  if (n.process) {
    n.process->kill();
    (void)n.process->wait(5s);
    n.process.reset();
  }
  n.options.crash_at = std::move(crash_at);
  n.options.faults.clear();  // a restarted Node is healthy unless the caller asks for a crash phase
  const auto port = n.endpoint.port;
  // The previous listener may linger briefly in the kernel; retry the same port.
  Status last;
  for (int attempt = 0; attempt < 50; ++attempt) {
    last = launch(n, "127.0.0.1:" + std::to_string(port));
    if (last.is_ok()) return last;
    std::this_thread::sleep_for(100ms);
  }
  return last;
}

}  // namespace clusterlm::bench
