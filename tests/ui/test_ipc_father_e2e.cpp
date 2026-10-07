// IpcFatherClient against a real clusterlm-father-agent process running the development fixture model
// (--dev-fixture-model; launched like tests/father_agent/test_dev_e2e). The Father view-model drives prepare, chat,
// cancel-free streaming, diagnostics redaction and pairing errors over the real local pipe API.
// Everything runs the reference backend on a small fixture model; no number here is a measurement.
#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <thread>

#include "clusterlm/objects/fixture_model.hpp"
#include "clusterlm/platform/ipc.hpp"
#include "clusterlm/platform/ipc_messages.hpp"
#include "clusterlm/platform/process.hpp"
#include "clusterlm/ui/clients.hpp"
#include "clusterlm/ui/father_viewmodel.hpp"

// Defined by CMake; the fallbacks only let the Windows syntax check compile this file.
#ifndef CLUSTERLM_FATHER_AGENT_BINARY
#define CLUSTERLM_FATHER_AGENT_BINARY "clusterlm-father-agent.exe"
#endif
#ifndef CLUSTERLM_SOURCE_DIR
#define CLUSTERLM_SOURCE_DIR "."
#endif

using namespace clusterlm;
using namespace std::chrono_literals;
using nlohmann::json;
namespace fs = std::filesystem;

namespace {

void set_env(const char* k, const std::string& v) {
#ifdef _WIN32
  _putenv_s(k, v.c_str());
#else
  ::setenv(k, v.c_str(), 1);
#endif
}

struct Agent {
  fs::path root, dir;
  std::unique_ptr<platform::ChildProcess> proc;
  ipc::Endpoint endpoint;

  explicit Agent(const char* name) {
    root = fs::temp_directory_path() / (std::string("clm-uie2e-") + name);
    { std::error_code ec_rm; fs::remove_all(root, ec_rm); }
    fs::create_directories(root);
    auto manifest = objects::write_fixture_model(objects::FixtureSpec{}, root / "model");
    REQUIRE_MESSAGE(manifest.is_ok(), manifest.status().to_string());
    dir = root / "father";
    fs::create_directories(dir / "run");
    set_env("XDG_DATA_HOME", (dir / "data").string());
    set_env("XDG_STATE_HOME", (dir / "state").string());
    set_env("XDG_RUNTIME_DIR", (dir / "run").string());
    std::vector<std::string> args = {"--settings", (dir / "father.json").string(), "--identity", (dir / "identity").string(),
                                     "--catalog", std::string(CLUSTERLM_SOURCE_DIR) + "/fixtures/catalog/clusterlm-catalog.json",
                                     "--profiles-dir", std::string(CLUSTERLM_SOURCE_DIR) + "/fixtures/profiles",
                                     "--user-tag", "uie2e", "--ipc-dir", (dir / "ipc").string(), "--exit-on-stdin-eof",
                                     "--dev-fixture-model"};
    auto p = platform::ChildProcess::spawn(CLUSTERLM_FATHER_AGENT_BINARY, args);
    REQUIRE_MESSAGE(p.is_ok(), p.status().to_string());
    proc = std::move(p).value();
    auto line = proc->read_until("CLUSTERLM_FATHER_AGENT", 30s);
    REQUIRE_MESSAGE(line.is_ok(), line.status().to_string());
    endpoint.name = ipc::father_ui_pipe_name("uie2e");
    endpoint.socket_dir = dir / "ipc";
  }
  ~Agent() {
    if (proc) {
      proc->close_stdin();
      if (!proc->wait(15s).is_ok()) proc->kill();
    }
    std::error_code ec;
    fs::remove_all(root, ec);
  }

  // One raw request on its own short-lived connection (set-up that the UI client intentionally has no call for).
  json raw(json req) {
    auto c = ipc::connect(endpoint, {}, 5s);
    REQUIRE_MESSAGE(c.is_ok(), c.status().to_string());
    req["id"] = 1;
    const std::string text = req.dump();
    REQUIRE(c.value()->send(ipc::father_envelope(ipc::MessageKind::kFatherRequest, Bytes(text.begin(), text.end())), 5s).is_ok());
    for (;;) {
      auto e = c.value()->receive(60s);
      REQUIRE(e.is_ok());
      if (e->kind == static_cast<std::uint16_t>(ipc::MessageKind::kFatherReply)) return json::parse(e->payload.begin(), e->payload.end());
    }
  }
};

template <typename Pred>
bool wait_for(ui::FatherViewModel& vm, Pred pred, std::chrono::seconds timeout = 90s) {
  const auto end = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < end) {
    vm.tick(true);
    if (pred(vm.snapshot())) return true;
    std::this_thread::sleep_for(50ms);
  }
  return false;
}

const ui::TierRow* row(const ui::FatherViewState& s, const char* id) {
  for (const auto& r : s.tiers)
    if (r.id == id) return &r;
  return nullptr;
}

}  // namespace

TEST_CASE("IpcFatherClient reports an absent agent in words and recovers when it appears") {
  ui::IpcFatherClient::Options o;
  o.endpoint.name = ipc::father_ui_pipe_name("nobody-home");
  o.endpoint.socket_dir = fs::temp_directory_path() / "clm-uie2e-absent";
  ui::IpcFatherClient c(o);
  auto t = c.list_tiers(4096);
  CHECK_FALSE(t.is_ok());
  CHECK(t.status().code() == ErrorCode::kUnavailable);
  CHECK(t.status().message() == ui::IpcFatherClient::not_running_message());
  CHECK(c.send_chat({}).status().code() == ErrorCode::kUnavailable);
  ui::FatherViewModel vm(c);
  vm.tick(true);
  CHECK(vm.snapshot().connection_message == ui::IpcFatherClient::not_running_message());
  CHECK_FALSE(vm.snapshot().can_send);
}

TEST_CASE("view-model over IpcFatherClient and a real dev-fixture agent: tiers, prepare, chat, diagnostics, pairing error") {
  Agent agent("full");
  REQUIRE(agent.raw({{"op", "settings.set"}, {"patch", {{"model_dirs", {{"fast", (agent.root / "model").string()}}}}}})["ok"] == true);
  REQUIRE(agent.raw({{"op", "model.confirm"}, {"tier_id", "fast"}})["ok"] == true);

  ui::IpcFatherClient::Options o;
  o.endpoint = agent.endpoint;
  ui::IpcFatherClient client(o);
  ui::FatherViewModel vm(client);

  // Tiers come from the real service; provenance is Synthetic because the agent runs the fixture model.
  REQUIRE(wait_for(vm, [](const ui::FatherViewState& s) { return row(s, "fast") && row(s, "fast")->state_label == "Available"; }));
  auto s = vm.snapshot();
  REQUIRE(s.tiers.size() == 3);
  CHECK(row(s, "ultra")->state_label == "Unavailable");  // no nodes paired
  CHECK(row(s, "ultra")->participants.size() == 3);       // roles resolved through pairing.list
  CHECK(row(s, "fast")->participants == std::vector<std::string>{"This PC"});
  CHECK(client.stats_provenance() == "Synthetic");
  bool dev_note = false;
  for (const auto& n : row(s, "fast")->notes) dev_note = dev_note || n.find("DEVELOPMENT FIXTURE MODEL") != std::string::npos;
  CHECK(dev_note);  // the agent's provenance details reach the tier card

  REQUIRE(vm.select_tier("fast").is_ok());
  REQUIRE(vm.prepare_tier("fast").is_ok());
  REQUIRE(wait_for(vm, [](const ui::FatherViewState& v) { return row(v, "fast")->state_label == "Ready" && !v.prepare.active; }));

  ui::FatherSettings fs;
  fs.max_new_tokens = 12;
  REQUIRE(vm.apply_settings(fs).is_ok());
  REQUIRE(vm.send("TOP-SECRET-PROMPT hello").is_ok());
  REQUIRE(wait_for(vm, [](const ui::FatherViewState& v) { return !v.chat_busy && v.last_answer.valid; }));
  s = vm.snapshot();
  REQUIRE(s.entries.size() == 2);
  CHECK(s.entries[0].text == "TOP-SECRET-PROMPT hello");
  CHECK_FALSE(s.entries[1].full_text().empty());
  CHECK(s.entries[1].attribution.rfind("Answered by Fast (", 0) == 0);
  CHECK(s.last_answer.tokens == 12);
  CHECK(s.last_answer.reason == "completed");
  CHECK(s.last_answer.provenance == "Synthetic");  // never presented as a measurement
  CHECK(s.context.label.find("at least") == 0);    // the agent cannot report exact context use

  // Diagnostics stay redacted unless the user opts in for the export.
  vm.set_diagnostics_open(true);
  s = vm.snapshot();
  CHECK(s.diagnostics.snapshot_text.find("TOP-SECRET") == std::string::npos);
  CHECK(s.diagnostics.plan_summary.empty());
  vm.set_exporter(ui::make_file_exporter((fs::path(agent.root) / "export").string()));
  REQUIRE(vm.export_diagnostics().is_ok());
  vm.set_include_conversation(true);
  auto with_text = client.export_diagnostics(true);
  REQUIRE(with_text.is_ok());
  CHECK(with_text->includes_conversation);
  CHECK(with_text->text.find("TOP-SECRET-PROMPT hello") != std::string::npos);
  auto without = client.export_diagnostics(false);
  REQUIRE(without.is_ok());
  CHECK(without->text.find("TOP-SECRET") == std::string::npos);

  // Pairing: a bad address is an error in words, and nothing becomes paired.
  CHECK_FALSE(vm.start_pairing({"127.0.0.1:1", "ZZZZ-2222", ""}).is_ok());
  CHECK_FALSE(vm.snapshot().pairing.message.empty());
  CHECK(vm.snapshot().pairing.machines.empty());
  CHECK(vm.unpair("nobody").code() == ErrorCode::kNotFound);

  // New conversation and release round-trip.
  REQUIRE(vm.new_conversation().is_ok());
  CHECK(vm.snapshot().entries.empty());
  REQUIRE(vm.release().is_ok());
  CHECK(wait_for(vm, [](const ui::FatherViewState& v) { return row(v, "fast")->state_label == "Available"; }));
}

TEST_CASE("IpcFatherClient reconnects after the agent restarts") {
  ui::IpcFatherClient::Options o;
  {
    Agent first("restart");
    o.endpoint = first.endpoint;
    ui::IpcFatherClient c(o);
    REQUIRE(c.list_tiers(4096).is_ok());
  }
  ui::IpcFatherClient c(o);
  CHECK_FALSE(c.list_tiers(4096).is_ok());  // agent gone
  Agent second("restart");
  o.endpoint = second.endpoint;
  ui::IpcFatherClient c2(o);
  CHECK(c2.list_tiers(4096).is_ok());
}
