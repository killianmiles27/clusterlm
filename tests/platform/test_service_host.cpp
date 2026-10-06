#include <doctest/doctest.h>

#include <atomic>
#include <csignal>
#include <thread>

#include "clusterlm/platform/service_host.hpp"

using namespace clusterlm;
using namespace clusterlm::platform;

namespace {
class FakeApp final : public ServiceApp {
 public:
  Status on_start() override {
    started = true;
    return start_status;
  }
  int run() override {
    while (!stop.load()) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    return 7;
  }
  void on_stop(StopReason r) override {
    reason = r;
    stop.store(true);
  }
  bool started = false;
  Status start_status;
  std::atomic<bool> stop{false};
  StopReason reason = StopReason::kStop;
};
}  // namespace

TEST_CASE("console mode runs the app and maps SIGINT to a console-interrupt stop") {
  FakeApp app;
  std::thread interrupter([] {
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    std::raise(SIGINT);
  });
  auto rc = run_in_console(app, {});
  interrupter.join();
  REQUIRE(rc.is_ok());
  CHECK(rc.value() == 7);
  CHECK(app.started);
  CHECK(app.reason == StopReason::kConsoleInterrupt);
}

TEST_CASE("a failing on_start fails console start without running the app") {
  FakeApp app;
  app.start_status = make_error(ErrorCode::kUnavailable, "boom");
  auto rc = run_in_console(app, {});
  CHECK_FALSE(rc.is_ok());
  CHECK(rc.status().code() == ErrorCode::kUnavailable);
}

#ifndef _WIN32
TEST_CASE("the SCM host is unavailable off Windows") {
  FakeApp app;
  auto rc = run_as_service(app, {});
  REQUIRE_FALSE(rc.is_ok());
  CHECK(rc.status().code() == ErrorCode::kUnimplemented);
}
#endif

TEST_CASE("node service install spec: delayed auto start, LocalService, restart recovery, preshutdown") {
  const auto s = node_service_install_spec();
  CHECK(s.name == "ClusterLMNode");
  CHECK(s.start_type == ServiceStartType::kAutoDelayed);
  CHECK(s.account == "NT AUTHORITY\\LocalService");
  CHECK(s.unrestricted_service_sid);
  REQUIRE(s.recovery.size() == 3);
  for (const auto& a : s.recovery) CHECK(a.kind == ServiceRecoveryAction::Kind::kRestart);
  CHECK(s.recovery[0].delay < s.recovery[2].delay);
  CHECK(s.preshutdown_timeout.count() > 0);
}
