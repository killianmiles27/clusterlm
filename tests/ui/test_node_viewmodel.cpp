// NodeViewModel: state mapping, pause/resume, settings validation and the honest "not saved" path.
#include <doctest/doctest.h>

#include "clusterlm/ui/node_viewmodel.hpp"

using namespace clusterlm;
using namespace clusterlm::ui;

namespace {

struct FakeNode final : NodeClient {
  NodeStatus current;
  Status pause_result, resume_result, set_result;
  int pauses = 0, resumes = 0, sets = 0;
  NodeSettings last_set;
  Result<NodeStatus> status() override { return current; }
  Status pause() override {
    ++pauses;
    if (pause_result.is_ok()) current.state = NodeUiState::kPaused;
    return pause_result;
  }
  Status resume() override {
    ++resumes;
    if (resume_result.is_ok()) current.state = NodeUiState::kAvailable;
    return resume_result;
  }
  Result<NodeSettings> get_settings() override { return NodeSettings{}; }
  Status set_settings(const NodeSettings& s) override {
    ++sets;
    last_set = s;
    return set_result;
  }
};

}  // namespace

TEST_CASE("ipc node states map to the words a user sees") {
  CHECK(from_ipc(ipc::NodeState::kOffering) == NodeUiState::kAvailable);
  CHECK(from_ipc(ipc::NodeState::kBusy) == NodeUiState::kBusy);
  CHECK(from_ipc(ipc::NodeState::kSuspended) == NodeUiState::kBusy);
  CHECK(from_ipc(ipc::NodeState::kPaused) == NodeUiState::kPaused);
  CHECK(from_ipc(ipc::NodeState::kStarting) == NodeUiState::kStarting);
  CHECK(from_ipc(ipc::NodeState::kStopping) == NodeUiState::kStopping);

  using M = catalog::MachineState;
  CHECK(from_machine_state(M::kAvailable) == NodeUiState::kAvailable);
  CHECK(from_machine_state(M::kPreparing) == NodeUiState::kPreparing);
  CHECK(from_machine_state(M::kReady) == NodeUiState::kReady);
  CHECK(from_machine_state(M::kInferencing) == NodeUiState::kInUse);
  CHECK(from_machine_state(M::kBusy) == NodeUiState::kBusy);
  CHECK(from_machine_state(M::kCleanupPending) == NodeUiState::kCleanupNeeded);
  CHECK(from_machine_state(M::kReleasing) == NodeUiState::kCleanupNeeded);
  CHECK(from_machine_state(M::kOffline) == NodeUiState::kUnreachable);

  CHECK(to_string(NodeUiState::kInUse) == "In use");
  CHECK(to_string(NodeUiState::kCleanupNeeded) == "Cleanup needed");
  for (auto s : {NodeUiState::kUnreachable, NodeUiState::kStarting, NodeUiState::kAvailable, NodeUiState::kPreparing,
                 NodeUiState::kReady, NodeUiState::kInUse, NodeUiState::kBusy, NodeUiState::kPaused,
                 NodeUiState::kCleanupNeeded, NodeUiState::kStopping}) {
    CHECK_FALSE(describe(s).empty());
    // No model/GGUF vocabulary on the Node screen.
    for (const char* banned : {"GGUF", "gguf", "model", "token", "prompt"}) CHECK(describe(s).find(banned) == std::string_view::npos);
  }
}

TEST_CASE("status polling fills labels, father, storage and tray tooltip") {
  FakeNode n;
  n.current = {NodeUiState::kAvailable, "ab12cd34", 1'500'000'000ull, true, ""};
  NodeViewModel vm(n);
  vm.tick(true);
  const auto& s = vm.state();
  CHECK(s.state_label == "Available");
  CHECK(s.father_label == "Paired with Father ab12cd34");
  CHECK(s.storage_label == "Temporary storage in use: 1.5 GB of 64 GB");
  CHECK(s.tooltip == "ClusterLM Node - Available");
  CHECK(s.can_pause);
  CHECK_FALSE(s.can_resume);
}

TEST_CASE("an unpaired Node says so and an unreachable service is a state with a reason") {
  FakeNode n;
  n.current = {NodeUiState::kUnreachable, "", 0, false, "connection refused"};
  NodeViewModel vm(n);
  vm.tick(true);
  CHECK(vm.state().father_label == "Not paired with a Father yet");
  CHECK(vm.state().state_label == "Not running");
  CHECK(vm.state().description.find("connection refused") != std::string::npos);
  CHECK_FALSE(vm.state().can_pause);
  CHECK_FALSE(vm.state().can_resume);
}

TEST_CASE("pause and resume go to the service and the state follows") {
  FakeNode n;
  n.current.state = NodeUiState::kAvailable;
  NodeViewModel vm(n);
  vm.tick(true);
  REQUIRE(vm.pause().is_ok());
  CHECK(n.pauses == 1);
  CHECK(vm.state().state_label == "Paused");
  CHECK(vm.state().can_resume);
  CHECK_FALSE(vm.state().can_pause);
  REQUIRE(vm.resume().is_ok());
  CHECK(vm.state().state_label == "Available");
}

TEST_CASE("a failed pause is shown, not swallowed") {
  FakeNode n;
  n.current.state = NodeUiState::kAvailable;
  n.pause_result = make_error(ErrorCode::kPermissionDenied, "not allowed");
  NodeViewModel vm(n);
  vm.tick(true);
  CHECK_FALSE(vm.pause().is_ok());
  CHECK(vm.state().banner == "Could not pause: not allowed (PERMISSION_DENIED)");
  CHECK(vm.state().state_label == "Available");
  vm.dismiss_banner();
  CHECK(vm.state().banner.empty());
}

TEST_CASE("settings validation names the field and nothing invalid is sent") {
  FakeNode n;
  NodeViewModel vm(n);
  NodeSettings s;
  s.temp_storage_limit_gb = 0;
  CHECK(vm.apply_settings(s).code() == ErrorCode::kInvalidArgument);
  CHECK(vm.state().settings_error == "Temporary storage limit must be between 1 and 4096 GB.");
  s = NodeSettings{};
  s.cpu_cap_percent = 5;
  CHECK_FALSE(vm.apply_settings(s).is_ok());
  CHECK(vm.state().settings_error.find("CPU limit") != std::string::npos);
  s = NodeSettings{};
  s.gpu_memory_cap_percent = 101;
  CHECK_FALSE(vm.apply_settings(s).is_ok());
  CHECK(vm.state().settings_error.find("Graphics memory") != std::string::npos);
  s = NodeSettings{};
  s.ram_cap_gb = 2000;
  CHECK_FALSE(vm.apply_settings(s).is_ok());
  CHECK(vm.state().settings_error.find("Memory limit") != std::string::npos);
  CHECK(n.sets == 0);

  s = NodeSettings{};
  s.allow_when_idle = false;
  s.ac_power_only = false;
  s.temp_storage_limit_gb = 200;
  s.start_with_windows = false;
  s.cpu_cap_percent = 50;
  REQUIRE(vm.apply_settings(s).is_ok());
  CHECK(n.sets == 1);
  CHECK(n.last_set.temp_storage_limit_gb == 200);
  CHECK(vm.state().settings_error.empty());
  CHECK(vm.state().settings_persisted);
}

TEST_CASE("unimplemented settings storage is stated plainly and the values stay in the window") {
  FakeNode n;
  n.set_result = make_error(ErrorCode::kUnimplemented, "This build cannot save Node settings yet.");
  NodeViewModel vm(n);
  NodeSettings s;
  s.temp_storage_limit_gb = 128;
  CHECK(vm.apply_settings(s).code() == ErrorCode::kUnimplemented);
  CHECK_FALSE(vm.state().settings_persisted);
  CHECK(vm.state().settings.temp_storage_limit_gb == 128);
  CHECK(vm.state().settings_note.find("cannot save Node settings") != std::string::npos);
}
