// NodeViewModel: state mapping (including the worker's lease state), pause/resume, settings read and save through the
// service, validation, and the pairing button.
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
  Result<NodeSettings> get_settings() override {
    ++gets;
    if (!get_result.is_ok()) return get_result;
    return stored;
  }
  Status set_settings(const NodeSettings& s) override {
    ++sets;
    last_set = s;
    if (set_result.is_ok()) stored = s;
    return set_result;
  }
  Result<NodePairingInfo> enter_pairing_mode() override {
    ++pairings;
    if (!pair_result.is_ok()) return pair_result;
    return NodePairingInfo{"ABCD-EFGH", "192.168.1.20:47601", "ab12-cd34-ef56", 300};
  }
  Status get_result;
  Status pair_result;
  int gets = 0, pairings = 0;
  NodeSettings stored;
};

}  // namespace

TEST_CASE("ipc node states map to the words a user sees") {
  CHECK(from_ipc(ipc::NodeState::kOffering) == NodeUiState::kAvailable);
  CHECK(from_ipc(ipc::NodeState::kBusy) == NodeUiState::kBusy);
  CHECK(from_ipc(ipc::NodeState::kSuspended) == NodeUiState::kBusy);
  CHECK(from_ipc(ipc::NodeState::kPaused) == NodeUiState::kPaused);
  CHECK(from_ipc(ipc::NodeState::kStarting) == NodeUiState::kStarting);
  CHECK(from_ipc(ipc::NodeState::kStopping) == NodeUiState::kStopping);

  // The worker's lease state shows through only while the Node is offering; anything else stays out of the way.
  using L = ipc::LeaseState;
  CHECK(from_ipc(ipc::NodeState::kOffering, L::kNone) == NodeUiState::kAvailable);
  CHECK(from_ipc(ipc::NodeState::kOffering, L::kPreparing) == NodeUiState::kPreparing);
  CHECK(from_ipc(ipc::NodeState::kOffering, L::kReady) == NodeUiState::kReady);
  CHECK(from_ipc(ipc::NodeState::kOffering, L::kInferencing) == NodeUiState::kInUse);
  CHECK(from_ipc(ipc::NodeState::kOffering, L::kReleasing) == NodeUiState::kCleanupNeeded);
  CHECK(from_ipc(ipc::NodeState::kOffering, L::kCleanupPending) == NodeUiState::kCleanupNeeded);
  for (auto lease : {L::kPreparing, L::kReady, L::kInferencing, L::kReleasing, L::kCleanupPending}) {
    CHECK(from_ipc(ipc::NodeState::kBusy, lease) == NodeUiState::kBusy);
    CHECK(from_ipc(ipc::NodeState::kPaused, lease) == NodeUiState::kPaused);
    CHECK(from_ipc(ipc::NodeState::kSuspended, lease) == NodeUiState::kBusy);
    CHECK(from_ipc(ipc::NodeState::kStopping, lease) == NodeUiState::kStopping);
  }

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
  n.current.state = NodeUiState::kAvailable;
  n.current.paired_father = "ab12cd34";
  n.current.storage_bytes = 1'500'000'000ull;
  n.current.fresh = true;
  n.stored.temp_storage_limit_gb = 64;
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
  n.current.state = NodeUiState::kUnreachable;
  n.current.detail = "connection refused";
  n.get_result = make_error(ErrorCode::kUnavailable, "connection refused");
  NodeViewModel vm(n);
  vm.tick(true);
  CHECK(vm.state().father_label == "Not paired with a Father yet");
  CHECK(vm.state().state_label == "Not running");
  CHECK(vm.state().description.find("connection refused") != std::string::npos);
  CHECK_FALSE(vm.state().can_pause);
  CHECK_FALSE(vm.state().can_resume);
}

TEST_CASE("the lease is shown in counts and plain words, only while the Node is helping") {
  FakeNode n;
  n.current.state = NodeUiState::kPreparing;
  n.current.lease = ipc::LeaseState::kPreparing;
  n.current.lease_parts_done = 3;
  n.current.lease_parts_total = 12;
  NodeViewModel vm(n);
  vm.tick(true);
  CHECK(vm.state().state_label == "Preparing");
  CHECK(vm.state().lease_label == "Receiving temporary files: 3 of 12");
  n.current.state = NodeUiState::kReady;
  n.current.lease = ipc::LeaseState::kReady;
  vm.tick(true);
  CHECK(vm.state().lease_label == "Holding 12 temporary files");
  n.current.state = NodeUiState::kCleanupNeeded;
  n.current.lease = ipc::LeaseState::kCleanupPending;
  vm.tick(true);
  CHECK(vm.state().lease_label == "Removing temporary files");
  // A PC the user is using shows no lease even if the worker still has one for a moment.
  n.current.state = NodeUiState::kBusy;
  vm.tick(true);
  CHECK(vm.state().lease_label.empty());
  // No vocabulary of the model world leaks into these sentences.
  for (const char* banned : {"GGUF", "model", "token", "prompt", "object", "shard"}) {
    n.current.state = NodeUiState::kPreparing;
    n.current.lease = ipc::LeaseState::kPreparing;
    vm.tick(true);
    CHECK(vm.state().lease_label.find(banned) == std::string::npos);
  }
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
  s.temp_storage_limit_gb = 5000;
  CHECK(vm.apply_settings(s).code() == ErrorCode::kInvalidArgument);
  CHECK(vm.state().settings_error == "Temporary storage limit must be 0 (no limit) or between 1 and 4096 GB.");
  s = NodeSettings{};
  s.cpu_cap_percent = 5;
  CHECK_FALSE(vm.apply_settings(s).is_ok());
  CHECK(vm.state().settings_error.find("CPU limit") != std::string::npos);
  s = NodeSettings{};
  s.gpu_memory_gb = 2000;
  CHECK_FALSE(vm.apply_settings(s).is_ok());
  CHECK(vm.state().settings_error.find("Graphics memory") != std::string::npos);
  s = NodeSettings{};
  s.ram_gb = 0;
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

TEST_CASE("settings are read from the service and a save re-reads what it stored") {
  FakeNode n;
  n.stored.temp_storage_limit_gb = 200;
  n.stored.ram_gb = 8;
  NodeViewModel vm(n);
  CHECK(vm.state().settings.temp_storage_limit_gb == 200);
  CHECK(vm.state().settings.ram_gb == 8);
  const auto rev = vm.state().settings_revision;
  NodeSettings s = vm.state().settings;
  s.gpu_memory_gb = 6;
  REQUIRE(vm.apply_settings(s).is_ok());
  CHECK(vm.state().settings_persisted);
  CHECK(vm.state().settings_note.empty());
  CHECK(vm.state().settings.gpu_memory_gb == 6);
  CHECK(vm.state().settings_revision > rev);  // the controls reload from the stored values
  CHECK(vm.state().storage_label.find("of 200 GB") != std::string::npos);
  s.temp_storage_limit_gb = 0;
  REQUIRE(vm.apply_settings(s).is_ok());
  CHECK(vm.state().storage_label.find("no limit set") != std::string::npos);
}

TEST_CASE("a service that cannot save says so and nothing is shown as saved") {
  FakeNode n;
  n.set_result = make_error(ErrorCode::kResourceExhausted, "settings were just changed; try again in a moment");
  NodeViewModel vm(n);
  NodeSettings s;
  s.temp_storage_limit_gb = 128;
  CHECK(vm.apply_settings(s).code() == ErrorCode::kResourceExhausted);
  CHECK_FALSE(vm.state().settings_persisted);
  CHECK(vm.state().settings.temp_storage_limit_gb == 128);  // the edit stays in the window
  CHECK(vm.state().settings_note.find("Settings not saved: settings were just changed") != std::string::npos);
  CHECK(vm.state().settings_note.find("cannot save") == std::string::npos);  // the old "cannot save yet" path is gone
}

TEST_CASE("settings that could not be read at start are read once the service answers") {
  FakeNode n;
  n.current.state = NodeUiState::kUnreachable;
  n.get_result = make_error(ErrorCode::kUnavailable, "not connected");
  n.stored.ram_gb = 16;
  NodeViewModel vm(n);
  CHECK(vm.state().settings_note.find("Could not read the Node's settings") != std::string::npos);
  n.current.state = NodeUiState::kAvailable;
  n.get_result = Status::ok();
  vm.tick(true);
  CHECK(vm.state().settings.ram_gb == 16);
  CHECK(vm.state().settings_note.empty());
}

TEST_CASE("the pairing button shows the code line and drops it when a new request fails") {
  FakeNode n;
  n.current.state = NodeUiState::kAvailable;
  NodeViewModel vm(n);
  vm.tick(true);
  CHECK(vm.state().can_pair);
  REQUIRE(vm.enter_pairing_mode().is_ok());
  CHECK(vm.state().pairing_line.find("ABCD-EFGH") != std::string::npos);
  CHECK(vm.state().pairing_line.find("192.168.1.20:47601") != std::string::npos);
  CHECK(vm.state().pairing_line.find("ab12-cd34-ef56") != std::string::npos);
  CHECK(vm.state().pairing_line.find("5 minutes") != std::string::npos);
  n.pair_result = make_error(ErrorCode::kResourceExhausted, "pairing mode was just started; wait a few seconds");
  CHECK_FALSE(vm.enter_pairing_mode().is_ok());
  CHECK(vm.state().pairing_line.empty());  // a stale code is never left on screen
  CHECK(vm.state().pairing_error.find("wait a few seconds") != std::string::npos);

  n.current.state = NodeUiState::kUnreachable;
  vm.tick(true);
  CHECK_FALSE(vm.state().can_pair);
}

TEST_CASE("settings convert to and from the service's form; the CPU limit travels as a thread count") {
  ipc::NodeSettingsView v;
  v.idle_seconds = 900;  // not edited here: must survive a save
  v.threads = 0;
  CHECK(from_view(v, 16).cpu_cap_percent == 100);
  v.threads = 8;
  CHECK(from_view(v, 16).cpu_cap_percent == 50);
  v.threads = 99;
  CHECK(from_view(v, 16).cpu_cap_percent == 100);
  NodeSettings s;
  s.cpu_cap_percent = 50;
  s.gpu_memory_gb = 6;
  s.ram_gb = 12;
  s.temp_storage_limit_gb = 40;
  s.allow_when_idle = false;
  const auto out = to_view(s, v, 16);
  CHECK(out.threads == 8);
  CHECK(out.vram_gib == 6);
  CHECK(out.ram_gib == 12);
  CHECK(out.temp_storage_limit_gib == 40);
  CHECK_FALSE(out.allow_when_idle);
  CHECK(out.idle_seconds == 900);
  s.cpu_cap_percent = 100;
  CHECK(to_view(s, v, 16).threads == 0);
  s.cpu_cap_percent = 10;
  CHECK(to_view(s, v, 4).threads == 1);  // never below one thread
}
