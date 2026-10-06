#pragma once
// Mock adapters: deterministic, settable implementations for tests. Values are whatever the test sets;
// nothing here models real hardware.
#include <algorithm>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "clusterlm/platform/adapters.hpp"

namespace clusterlm::platform {

class MockActivityMonitor final : public ActivityMonitor {
 public:
  ActivitySample current;
  std::optional<Status> fail_with;
  Result<ActivitySample> sample() override {
    if (fail_with) return *fail_with;
    return current;
  }
};

class MockPowerMonitor final : public PowerMonitor {
 public:
  PowerSample current;
  std::optional<Status> fail_with;
  Result<PowerSample> sample() override {
    if (fail_with) return *fail_with;
    return current;
  }
};

class MockProcessJob final : public ProcessJob {
 public:
  JobLimits limits;
  std::set<std::uint32_t> pids;
  std::vector<std::uint32_t> killed_pids;  // pids killed by terminate_all/close, in order
  std::uint32_t last_exit_code = 0;
  bool closed = false;

  Status set_limits(const JobLimits& l) override {
    if (closed) return make_error(ErrorCode::kFailedPrecondition, "job closed");
    limits = l;
    return Status::ok();
  }
  Status assign_process(std::uint32_t pid) override {
    if (closed) return make_error(ErrorCode::kFailedPrecondition, "job closed");
    pids.insert(pid);
    return Status::ok();
  }
  Result<std::uint32_t> active_process_count() override { return static_cast<std::uint32_t>(pids.size()); }
  Status terminate_all(std::uint32_t exit_code) override {
    kill_all(exit_code);
    return Status::ok();
  }
  Status close() override {
    if (closed) return Status::ok();
    closed = true;
    if (limits.kill_on_close) kill_all(1);
    return Status::ok();
  }

 private:
  void kill_all(std::uint32_t code) {
    last_exit_code = code;
    for (auto pid : pids) killed_pids.push_back(pid);
    pids.clear();
  }
};

class MockGpuBudgetProbe final : public GpuBudgetProbe {
 public:
  std::vector<GpuAdapterBudget> adapters;
  std::optional<Status> fail_with;
  Result<std::vector<GpuAdapterBudget>> query() override {
    if (fail_with) return *fail_with;
    return adapters;
  }
};

}  // namespace clusterlm::platform
