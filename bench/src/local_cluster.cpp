#include "local_cluster.hpp"

#include <cstdlib>
#include <sstream>
#include <thread>

#include "clusterlm/common/clock.hpp"
#include "clusterlm/transport/transport.hpp"

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

void set_env(const char* key, const std::string& value) {
#ifdef _WIN32
  _putenv_s(key, value.c_str());
#else
  ::setenv(key, value.c_str(), 1);
#endif
}

// A loopback port nobody listens on right now. The supervised worker is relaunched on the same port, so it is fixed up front.
Result<std::uint16_t> free_port() {
  transport::SecurityConfig sec;
  sec.mode = transport::SecurityConfig::Mode::kInsecureLoopbackOnly;
  CLM_ASSIGN_OR_RETURN(auto listener, transport::listen({"127.0.0.1", 0}, sec));
  const auto port = listener->local_endpoint().port;
  listener->close();
  return port;
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
  if (c->options_.service_binary.empty()) {
#ifdef _WIN32
    c->options_.service_binary = platform::executable_dir() / "clusterlm-node-service.exe";
#else
    c->options_.service_binary = platform::executable_dir() / "clusterlm-node-service";
#endif
  }
  std::error_code ec;
  std::filesystem::create_directories(c->options_.work_dir, ec);
  if (c->options_.supervised) {
    // The service derives its per-user directories from the environment: keep them inside the work directory.
    set_env("XDG_STATE_HOME", (c->options_.work_dir / "xdg-state").string());
    set_env("XDG_DATA_HOME", (c->options_.work_dir / "xdg-data").string());
  }
  if (c->options_.tls) {
    const auto identity_dir = c->options_.father_identity_dir.empty() ? c->options_.work_dir / "father-id" : c->options_.father_identity_dir;
    CLM_ASSIGN_OR_RETURN(auto id, transport::DeviceIdentity::load_or_generate(identity_dir, "father"));
    c->father_identity_ = std::make_shared<const transport::DeviceIdentity>(std::move(id));
    c->father_security_.mode = transport::SecurityConfig::Mode::kMutualTls;
    c->father_security_.identity = c->father_identity_;
  } else {
    c->father_security_.mode = transport::SecurityConfig::Mode::kInsecureLoopbackOnly;
  }
  for (const auto& ep : c->options_.external) {
    if (c->options_.tls && ep.device_id.empty())
      return make_error(ErrorCode::kInvalidArgument, "Node '" + ep.name + "': with TLS the endpoint needs the Node's device id (NAME=HOST:PORT@FINGERPRINT)");
    if (!c->options_.tls && ep.endpoint.host != "127.0.0.1" && ep.endpoint.host != "localhost" && ep.endpoint.host != "::1")
      return make_error(ErrorCode::kInvalidArgument, "Node '" + ep.name + "': --insecure only works on loopback; use TLS for a real link");
    Node n;
    n.options.name = ep.name;
    n.endpoint = ep.endpoint;
    n.device_id = ep.device_id;
    if (c->options_.tls) c->father_security_.trusted_peers.push_back(n.device_id);
    c->nodes_.push_back(std::move(n));
  }
  if (!c->options_.external.empty()) return c;
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

std::vector<std::string> LocalCluster::worker_args(const Node& n, const std::string& listen) const {
  std::vector<std::string> args = {"--name", n.options.name, "--listen", listen, "--staging",
                                   (options_.work_dir / (n.options.name + "-staging")).string(), "--ram-gib",
                                   std::to_string(n.options.ram_gib), "--vram-gib", std::to_string(n.options.vram_gib), "--disk-gib",
                                   std::to_string(n.options.disk_gib),
                                   "--log", "warn"};
  if (options_.tls) {
    args.insert(args.end(), {"--identity", n.identity_dir.string(), "--trust", father_identity_->fingerprint()});
  } else {
    args.push_back("--insecure-loopback");
  }
  if (n.options.impair) args.insert(args.end(), {"--impair", *n.options.impair});
  for (const auto& f : n.options.faults) args.insert(args.end(), {"--fault", f});
  if (n.options.crash_at) args.insert(args.end(), {"--crash-at", *n.options.crash_at});
  if (n.options.hang_at) args.insert(args.end(), {"--hang-at", *n.options.hang_at});
  args.insert(args.end(), options_.node_args.begin(), options_.node_args.end());
  return args;
}

Status LocalCluster::launch(Node& n, const std::string& listen) {
  if (options_.supervised) return launch_supervised(n);
  CLM_ASSIGN_OR_RETURN(n.process, platform::ChildProcess::spawn(options_.node_binary, worker_args(n, listen)));
  CLM_ASSIGN_OR_RETURN(auto line, n.process->read_until("CLUSTERLM_NODE_LISTENING", 10s));
  CLM_ASSIGN_OR_RETURN(n.endpoint, transport::Endpoint::parse(field(line, "endpoint")));
  n.device_id = field(line, "device_id");
  return Status::ok();
}

void LocalCluster::log_service_line(Node& n, const std::string& line) {
  if (line.rfind("CLUSTERLM_NODE_SERVICE_EVENT", 0) != 0) return;
  ServiceEvent e;
  e.kind = field(line, "kind");
  try {
    e.latency_ms = std::stod(field(line, "latency_ms"));
    e.residual_bytes = std::stoull(field(line, "residual_bytes"));
  } catch (const std::exception&) {
    // A malformed number leaves the default; the kind is what is counted.
  }
  n.events.push_back(std::move(e));
}

std::size_t LocalCluster::count_events(const Node& n, const std::string& kind) {
  std::size_t count = 0;
  for (const auto& e : n.events) count += e.kind == kind;
  return count;
}

Result<ServiceEvent> LocalCluster::wait_event(Node& n, const std::string& kind, std::size_t nth, std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  for (;;) {
    std::size_t seen = 0;
    for (const auto& e : n.events)
      if (e.kind == kind && ++seen == nth) return e;
    const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now());
    if (left <= std::chrono::milliseconds(0))
      return make_error(ErrorCode::kDeadlineExceeded, "no '" + kind + "' event from " + n.options.name);
    auto line = n.process->read_line(left);
    if (!line.is_ok()) return line.status();
    log_service_line(n, line.value());
  }
}

Status LocalCluster::launch_supervised(Node& n) {
  if (n.port == 0) {
    CLM_ASSIGN_OR_RETURN(n.port, free_port());
  }
  const auto dir = options_.work_dir / (n.options.name + "-service");
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  // Windows pipe names are machine-global: side-by-side services need their own helper pipe.
  const std::string tag = std::to_string(monotonic_ns() % 1000000);
  const std::string pipe = "clmb" + tag + "-" + n.options.name;
  // POSIX: the helper socket lives in --ipc-dir and sockaddr_un paths are short (108 bytes). A deep work directory falls
  // back to a short per-run directory under /tmp (removed with the cluster); Windows names its pipe instead.
  std::filesystem::path ipc = dir / "ipc";
#ifndef _WIN32
  if (ipc.string().size() + pipe.size() + 8 > 100) {
    ipc = std::filesystem::path("/tmp") / ("clmb" + tag);
    n.ipc_fallback = ipc;
  }
#endif
  std::vector<std::string> args = {"--console",       "--simulate-activity", "--name",        n.options.name,
                                   "--settings",      (dir / "node-settings.json").string(),
                                   "--ipc-dir",       ipc.string(), "--helper-pipe", pipe,
                                   "--idle-seconds",  "1",                    "--deadline-ms",
                                   std::to_string(options_.cooperative_deadline.count()),
                                   "--worker",        options_.node_binary.string(), "--"};
  const auto wargs = worker_args(n, "127.0.0.1:" + std::to_string(n.port));
  args.insert(args.end(), wargs.begin(), wargs.end());
  CLM_ASSIGN_OR_RETURN(n.process, platform::ChildProcess::spawn(options_.service_binary, args));
  // The service prints its supervisor events as they happen; the line naming the worker's endpoint comes once it runs.
  for (;;) {
    CLM_ASSIGN_OR_RETURN(auto line, n.process->read_line(20s));
    log_service_line(n, line);
    if (line.rfind("CLUSTERLM_NODE_SERVICE ", 0) == 0) {
      CLM_ASSIGN_OR_RETURN(n.endpoint, transport::Endpoint::parse(field(line, "worker_endpoint")));
      n.device_id = field(line, "device_id");
      break;
    }
  }
  // The worker starts Busy (simulated user activity); an idle report makes the policy offer it to Father.
  CLM_RETURN_IF_ERROR(n.process->write_line("idle"));
  return wait_event(n, "offered", count_events(n, "offered") + 1, 20s).status();
}

Result<ServiceEvent> LocalCluster::wait_service_event(std::size_t i, const std::string& kind, std::size_t nth,
                                                      std::chrono::milliseconds timeout) {
  if (!options_.supervised) return make_error(ErrorCode::kFailedPrecondition, "not a supervised cluster");
  return wait_event(nodes_.at(i), kind, nth, timeout);
}

std::size_t LocalCluster::service_event_count(std::size_t i, const std::string& kind) const {
  return count_events(nodes_.at(i), kind);
}

LocalCluster::~LocalCluster() {
  for (auto& n : nodes_) {
    if (!n.process) continue;
    (void)n.process->write_line("quit");
    if (options_.supervised) {
      // The service stops (and cleans up after) its worker; do not leave it running if it is slow to exit.
      n.process->close_stdin();
      if (!n.process->wait(10s).is_ok()) n.process->kill();
    }
  }
  for (auto& n : nodes_)
    if (!n.ipc_fallback.empty()) {
      std::error_code ec;
      std::filesystem::remove_all(n.ipc_fallback, ec);
    }
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

namespace {
Status not_ours(const std::string& what) {
  return make_error(ErrorCode::kFailedPrecondition, what + ": the Node is an external process this cluster does not control");
}
}  // namespace

Status LocalCluster::local_activity(std::size_t i) {
  if (!nodes_.at(i).process) return not_ours("local activity");
  return nodes_.at(i).process->write_line("activity");
}
Status LocalCluster::local_idle(std::size_t i) {
  if (!nodes_.at(i).process) return not_ours("local idle");
  return nodes_.at(i).process->write_line("idle");
}

Result<LocalCluster::NodeStatusLine> LocalCluster::status(std::size_t i) {
  if (options_.supervised)
    return make_error(ErrorCode::kUnimplemented, "a supervised Node's worker is owned by its service: use the service events");
  if (!nodes_.at(i).process) return not_ours("status");
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

void LocalCluster::kill(std::size_t i) {
  if (nodes_.at(i).process) nodes_.at(i).process->kill();
}

Result<int> LocalCluster::wait_exit(std::size_t i, std::chrono::milliseconds timeout) {
  if (!nodes_.at(i).process) return not_ours("wait_exit");
  return nodes_.at(i).process->wait(timeout);
}

Status LocalCluster::restart(std::size_t i, std::optional<std::string> crash_at) {
  if (options_.supervised) return make_error(ErrorCode::kUnimplemented, "a supervised Node is relaunched by its service");
  if (!nodes_.at(i).process && external()) return not_ours("restart");
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
