#include "clusterlm/node/supervisor.hpp"

#include <algorithm>
#include <sstream>
#include <thread>

#include "clusterlm/common/clock.hpp"
#include "clusterlm/common/log.hpp"

namespace clusterlm::node {

using namespace std::chrono_literals;

std::string_view to_string(SupervisorEventKind k) {
  switch (k) {
    case SupervisorEventKind::kWorkerStarted: return "worker_started";
    case SupervisorEventKind::kOffered: return "offered";
    case SupervisorEventKind::kRevoked: return "revoked";
    case SupervisorEventKind::kForcedTermination: return "forced_termination";
    case SupervisorEventKind::kWorkerRestarted: return "worker_restarted";
  }
  return "?";
}

std::vector<std::string> with_worker_caps(std::vector<std::string> args, const WorkerCaps& caps) {
  auto drop = [&](const std::string& flag) {
    for (std::size_t i = 0; i < args.size();) {
      if (args[i] == flag) {
        args.erase(args.begin() + static_cast<std::ptrdiff_t>(i), args.begin() + static_cast<std::ptrdiff_t>(std::min(i + 2, args.size())));
      } else {
        ++i;
      }
    }
  };
  auto put = [&](const std::string& flag, std::uint32_t value) {
    drop(flag);
    args.push_back(flag);
    args.push_back(std::to_string(value));
  };
  put("--ram-gib", caps.ram_gib);
  put("--vram-gib", caps.vram_gib);
  if (caps.disk_gib > 0) put("--disk-gib", caps.disk_gib); else drop("--disk-gib");
  if (caps.threads > 0) put("--threads", caps.threads); else drop("--threads");
  return args;
}

namespace {
std::string field(const std::string& line, const std::string& key) {
  std::istringstream ss(line);
  std::string tok;
  while (ss >> tok)
    if (tok.rfind(key + "=", 0) == 0) return tok.substr(key.size() + 1);
  return {};
}
}  // namespace

NodeSupervisor::NodeSupervisor(SupervisorConfig config, platform::ActivityMonitor& activity,
                               platform::PowerMonitor& power, std::unique_ptr<platform::ProcessJob> job)
    : cfg_(std::move(config)), activity_(activity), power_(power), job_(std::move(job)) {}

NodeSupervisor::~NodeSupervisor() { stop(); }

Status NodeSupervisor::launch() {
  auto args = cfg_.worker_args;
  // The worker always starts Busy: no resources are offered until local policy says so.
  args.push_back("--start-busy");
  for (const auto& fp : cfg_.trusted_peers) {
    args.push_back("--trust");
    args.push_back(fp);
  }
  CLM_ASSIGN_OR_RETURN(worker_, platform::ChildProcess::spawn(cfg_.worker_binary, args));
  if (job_) {
    platform::JobLimits limits;
    limits.process_memory_limit_bytes = cfg_.worker_memory_limit_bytes;
    CLM_RETURN_IF_ERROR(job_->set_limits(limits));
    CLM_RETURN_IF_ERROR(job_->assign_process(static_cast<std::uint32_t>(worker_->pid())));
  }
  lease_view_ = {};
  last_lease_poll_ = {};
  CLM_ASSIGN_OR_RETURN(auto line, read_worker("CLUSTERLM_NODE_LISTENING", 15s));
  endpoint_ = field(line, "endpoint");
  device_id_ = field(line, "device_id");
  eligible_ = false;
  return Status::ok();
}

Status NodeSupervisor::start() { return launch(); }

void NodeSupervisor::stop() {
  if (!worker_) return;
  (void)worker_->write_line("quit");
  if (!worker_->wait(cfg_.cooperative_deadline).is_ok()) {
    if (job_) (void)job_->terminate_all(1);
    worker_->kill();
    (void)worker_->wait(2s);
  }
  worker_.reset();
}

Result<bool> NodeSupervisor::policy_eligible() {
  CLM_ASSIGN_OR_RETURN(auto a, activity_.sample());
  CLM_ASSIGN_OR_RETURN(auto p, power_.sample());
  const bool idle = a.idle_seconds >= cfg_.policy.idle_seconds_required ||
                    (a.session_locked && cfg_.policy.participate_when_locked);
  const bool power_ok = (p.on_ac_power || !cfg_.policy.require_ac_power) &&
                        (!p.battery_saver || cfg_.policy.allow_battery_saver);
  return idle && power_ok;
}

Result<std::string> NodeSupervisor::read_worker(const std::string& prefix, std::chrono::milliseconds timeout) {
  const auto deadline = SteadyClock::now() + timeout;
  while (true) {
    const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - SteadyClock::now());
    if (left.count() <= 0) return make_error(ErrorCode::kDeadlineExceeded, "timed out waiting for " + prefix);
    CLM_ASSIGN_OR_RETURN(std::string line, worker_->read_line(left));
    if (line.rfind(prefix, 0) == 0) return line;
    if (line.rfind("CLUSTERLM_NODE_UNPAIR_NOTICE", 0) == 0) unpair_notice_ = field(line, "peer");
  }
}

std::optional<std::string> NodeSupervisor::take_unpair_notice() {
  auto out = std::move(unpair_notice_);
  unpair_notice_.reset();
  return out;
}

Result<NodeSupervisor::WorkerStatus> NodeSupervisor::worker_status(std::chrono::milliseconds timeout) {
  CLM_RETURN_IF_ERROR(worker_->write_line("status"));
  CLM_ASSIGN_OR_RETURN(auto line, read_worker("CLUSTERLM_NODE_STATUS", timeout));
  WorkerStatus s;
  s.state = field(line, "state");
  s.census_bytes = std::stoull(field(line, "census_bytes"));
  const auto sealed = field(line, "sealed"), planned = field(line, "planned");
  if (!sealed.empty()) s.sealed = static_cast<std::uint32_t>(std::stoul(sealed));
  if (!planned.empty()) s.planned = static_cast<std::uint32_t>(std::stoul(planned));
  return s;
}

void NodeSupervisor::poll_lease() {
  const auto now = SteadyClock::now();
  if (cfg_.lease_poll_interval.count() > 0 && last_lease_poll_ != std::chrono::steady_clock::time_point{} &&
      now - last_lease_poll_ < cfg_.lease_poll_interval)
    return;
  last_lease_poll_ = now;
  auto st = worker_status(500ms);
  if (!st.is_ok()) {
    lease_view_ = {};  // unresponsive: say nothing rather than a stale state
    return;
  }
  lease_view_ = WorkerLeaseView{true, st->state, st->sealed, st->planned};
}

Result<SupervisorEvent> NodeSupervisor::force_and_relaunch(const std::string& why) {
  log::warn("worker_forced_termination", {{"reason", why}});
  if (job_) (void)job_->terminate_all(137);
  worker_->kill();
  (void)worker_->wait(5s);
  worker_.reset();
  CLM_RETURN_IF_ERROR(launch());  // orphan recovery runs inside the relaunched worker before it listens
  CLM_ASSIGN_OR_RETURN(auto st, worker_status(5s));
  SupervisorEvent ev{SupervisorEventKind::kForcedTermination, 0, st.census_bytes, why};
  return ev;
}

Result<SupervisorEvent> NodeSupervisor::revoke() {
  Stopwatch sw;
  CLM_RETURN_IF_ERROR(worker_->write_line("activity"));
  const auto deadline = SteadyClock::now() + cfg_.cooperative_deadline;
  while (SteadyClock::now() < deadline) {
    const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - SteadyClock::now());
    auto st = worker_status(std::max(left, 1ms));
    if (!st.is_ok()) break;  // unresponsive worker: fall through to forced termination
    if (st->state == "Busy" && st->census_bytes == 0)
      return SupervisorEvent{SupervisorEventKind::kRevoked, sw.elapsed_ms(), 0, ""};
    std::this_thread::sleep_for(cfg_.status_poll);
  }
  CLM_ASSIGN_OR_RETURN(auto ev, force_and_relaunch("cooperative release deadline exceeded"));
  ev.latency_ms = sw.elapsed_ms();
  return ev;
}

Result<std::vector<SupervisorEvent>> NodeSupervisor::tick() {
  std::vector<SupervisorEvent> events;
  if (!worker_ || !worker_->running()) {
    worker_.reset();
    CLM_RETURN_IF_ERROR(launch());
    events.push_back({SupervisorEventKind::kWorkerRestarted, 0, 0, "worker exited"});
  }
  // Fail closed: if policy cannot be read, treat the machine as in use. A suspended machine is never eligible.
  auto eligible = policy_eligible();
  const bool now_eligible = !suspended_ && eligible.is_ok() && eligible.value();
  if (now_eligible && !eligible_) {
    CLM_RETURN_IF_ERROR(worker_->write_line("idle"));
    eligible_ = true;
    events.push_back({SupervisorEventKind::kOffered, 0, 0, ""});
  } else if (!now_eligible && eligible_) {
    eligible_ = false;
    CLM_ASSIGN_OR_RETURN(auto ev, revoke());
    events.push_back(std::move(ev));
  }
  poll_lease();
  return events;
}

Status NodeSupervisor::restart_worker() {
  stop();
  return launch();
}

Result<std::vector<SupervisorEvent>> NodeSupervisor::set_trusted_peers(std::vector<std::string> fingerprints) {
  std::vector<SupervisorEvent> events;
  cfg_.trusted_peers = std::move(fingerprints);
  if (!worker_) return events;  // not started yet: the list applies at launch
  if (worker_->running() && eligible_) {
    eligible_ = false;
    CLM_ASSIGN_OR_RETURN(auto ev, revoke());
    events.push_back(std::move(ev));
  }
  CLM_RETURN_IF_ERROR(restart_worker());
  events.push_back({SupervisorEventKind::kWorkerRestarted, 0, 0, "paired device list changed"});
  return events;
}

Result<std::vector<SupervisorEvent>> NodeSupervisor::reconfigure(std::optional<std::vector<std::string>> worker_args,
                                                                 std::optional<IdlePolicy> policy) {
  std::vector<SupervisorEvent> events;
  if (policy) cfg_.policy = *policy;  // takes effect at the next tick, which may revoke
  if (!worker_args || *worker_args == cfg_.worker_args) return events;
  cfg_.worker_args = std::move(*worker_args);
  if (!worker_) return events;  // not started yet: applies at launch
  if (worker_->running() && eligible_) {
    eligible_ = false;
    CLM_ASSIGN_OR_RETURN(auto ev, revoke());
    events.push_back(std::move(ev));
  }
  CLM_RETURN_IF_ERROR(restart_worker());
  events.push_back({SupervisorEventKind::kWorkerRestarted, 0, 0, "resource caps changed"});
  return events;
}

Result<std::vector<SupervisorEvent>> NodeSupervisor::on_suspend() {
  std::vector<SupervisorEvent> events;
  suspended_ = true;
  // No worker to revoke (never started, or died and not relaunched yet): nothing can be offered anyway.
  if (!worker_ || !worker_->running()) {
    eligible_ = false;
    return events;
  }
  if (eligible_) {
    eligible_ = false;
    CLM_ASSIGN_OR_RETURN(auto ev, revoke());
    events.push_back(std::move(ev));
  }
  return events;
}

Result<std::vector<SupervisorEvent>> NodeSupervisor::on_resume() {
  std::vector<SupervisorEvent> events;
  suspended_ = false;
  eligible_ = false;
  // Whatever happened around the sleep, make sure the worker is Busy; "activity" is idempotent on a Busy worker.
  if (worker_ && worker_->running()) CLM_RETURN_IF_ERROR(worker_->write_line("activity"));
  return events;
}

}  // namespace clusterlm::node
