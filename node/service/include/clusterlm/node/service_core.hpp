#pragma once
// ServiceCore: everything the Node service does apart from the OS service host.
//
//   helper pipe ---> HelperActivityMonitor ---> NodeSupervisor ---> worker (Job Object)
//   SCM power/session events ----------------^
//
// It owns the helper IPC server, the helper-fed (fail-closed) activity monitor, helper startup reconciliation and
// the supervisor. The Windows service app (apps/node-service) and the console/dev mode both drive this class, so
// the whole policy path runs on Linux CI with mock adapters and a real worker process.
//
// Threading: tick/suspend/resume/session handling serialise on one mutex around the (single-threaded)
// NodeSupervisor. A suspend notification therefore revokes synchronously inside handle_power_event: when it
// returns, the worker is Busy with zero staged bytes (or has been terminated).
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "clusterlm/node/supervisor.hpp"
#include "clusterlm/platform/helper_activity.hpp"
#include "clusterlm/platform/helper_startup.hpp"
#include "clusterlm/platform/ipc.hpp"
#include "clusterlm/platform/ipc_messages.hpp"
#include "clusterlm/platform/ipc_server_loop.hpp"
#include "clusterlm/platform/platform_events.hpp"

namespace clusterlm::node {

struct ServiceCoreConfig {
  SupervisorConfig supervisor;
  ipc::Endpoint helper_endpoint{ipc::kNodeHelperPipeName, {}};
  ipc::PipeAccess helper_pipe_access = ipc::PipeAccess::kOwnerSystemAndInteractiveUsers;
  ipc::AuthPolicy helper_auth;                // who may speak on the helper pipe
  bool enforce_report_session_match = true;   // an ActivityReport must name the peer's own session
  std::chrono::milliseconds helper_report_stale_after{5000};
  std::chrono::milliseconds tick_interval{100};
  std::filesystem::path staging_root;         // storage census for StatusReply (may be empty)
  std::string paired_father;                  // short device-id prefix shown in status; empty = unpaired
  // Optional: session enumeration and helper launch (Windows: WindowsHelperHost). Null = no helper management.
  std::shared_ptr<platform::HelperHost> helper_host;
  platform::HelperStartupConfig helper_startup;
};

class ServiceCore {
 public:
  using EventSink = std::function<void(const SupervisorEvent&)>;

  ServiceCore(ServiceCoreConfig config, platform::PowerMonitor& power,
              std::unique_ptr<platform::ProcessJob> job = nullptr);
  ~ServiceCore();
  ServiceCore(const ServiceCore&) = delete;
  ServiceCore& operator=(const ServiceCore&) = delete;

  void set_event_sink(EventSink sink);

  // Opens the helper pipe, launches the worker (Busy), starts accepting helpers.
  Status start();
  // One scheduling step (what run() repeats). Exposed so tests can drive time themselves.
  Result<std::vector<SupervisorEvent>> run_once();
  // Loops run_once() every tick_interval until request_stop(); then stops IPC and the worker.
  void run();
  void request_stop();
  void stop();

  // Thread-safe notifications from the SCM / tests.
  void handle_power_event(const platform::PowerEvent& event);
  void handle_session_event(const platform::SessionEvent& event);

  platform::HelperActivityMonitor& helper_activity() { return activity_; }
  ipc::NodeState state() const;
  ipc::StatusReply status() const;
  std::string worker_endpoint() const;
  std::string worker_device_id() const;

 private:
  void serve(const std::shared_ptr<ipc::Connection>& conn, const std::atomic<bool>& stop);
  ipc::Envelope handle_message(const ipc::Envelope& request, const ipc::PeerCredentials& peer);
  void refresh_sessions_locked();
  void emit(const std::vector<SupervisorEvent>& events);

  ServiceCoreConfig cfg_;
  platform::HelperActivityMonitor activity_;
  NodeSupervisor supervisor_;
  platform::HelperStartupState helper_state_;
  std::unique_ptr<ipc::ServerLoop> ipc_loop_;

  mutable std::mutex mu_;  // guards supervisor_ and helper_state_
  std::mutex sink_mu_;
  EventSink sink_;
  std::atomic<bool> started_{false};
  std::atomic<bool> stopping_{false};
  std::mutex wake_mu_;
  std::condition_variable wake_;
};

}  // namespace clusterlm::node
