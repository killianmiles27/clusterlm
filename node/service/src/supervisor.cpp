#include "clusterlm/node/supervisor.hpp"

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
  CLM_ASSIGN_OR_RETURN(worker_, platform::ChildProcess::spawn(cfg_.worker_binary, args));
  if (job_) {
    platform::JobLimits limits;
    limits.process_memory_limit_bytes = cfg_.worker_memory_limit_bytes;
    CLM_RETURN_IF_ERROR(job_->set_limits(limits));
    CLM_RETURN_IF_ERROR(job_->assign_process(static_cast<std::uint32_t>(worker_->pid())));
  }
  CLM_ASSIGN_OR_RETURN(auto line, worker_->read_until("CLUSTERLM_NODE_LISTENING", 15s));
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

Result<NodeSupervisor::WorkerStatus> NodeSupervisor::worker_status(std::chrono::milliseconds timeout) {
  CLM_RETURN_IF_ERROR(worker_->write_line("status"));
  CLM_ASSIGN_OR_RETURN(auto line, worker_->read_until("CLUSTERLM_NODE_STATUS", timeout));
  WorkerStatus s;
  s.state = field(line, "state");
  s.census_bytes = std::stoull(field(line, "census_bytes"));
  return s;
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
  // Fail closed: if policy cannot be read, treat the machine as in use.
  auto eligible = policy_eligible();
  const bool now_eligible = eligible.is_ok() && eligible.value();
  if (now_eligible && !eligible_) {
    CLM_RETURN_IF_ERROR(worker_->write_line("idle"));
    eligible_ = true;
    events.push_back({SupervisorEventKind::kOffered, 0, 0, ""});
  } else if (!now_eligible && eligible_) {
    eligible_ = false;
    CLM_ASSIGN_OR_RETURN(auto ev, revoke());
    events.push_back(std::move(ev));
  }
  return events;
}

}  // namespace clusterlm::node
