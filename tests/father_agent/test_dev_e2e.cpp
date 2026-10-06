// Dev end-to-end on Linux, real processes and real TLS: two clusterlm-node-service processes (console mode,
// simulated activity) + clusterlm-father-agent + a scripted IPC client.
//
//   pair G14 and 3060 (SPAKE2 over TLS) -> assign tier roles -> point the Ultra tier at the FIXTURE model
//   (explicit --dev-fixture-model override) -> prepare -> chat -> streamed tokens -> unpair.
//
// Everything here runs the reference backend on a small fixture model; no number it prints is a measurement.
#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <thread>

#include "clusterlm/objects/fixture_model.hpp"
#include "clusterlm/platform/ipc.hpp"
#include "clusterlm/platform/ipc_messages.hpp"
#include "clusterlm/platform/process.hpp"
#include "clusterlm/transport/transport.hpp"

// Defined by CMake; the fallbacks only let the Windows syntax check (scripts/check_windows_compile.sh) compile this file.
#ifndef CLUSTERLM_NODE_SERVICE_BINARY
#define CLUSTERLM_NODE_SERVICE_BINARY "clusterlm-node-service.exe"
#endif
#ifndef CLUSTERLM_FATHER_AGENT_BINARY
#define CLUSTERLM_FATHER_AGENT_BINARY "clusterlm-father-agent.exe"
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

fs::path temp_dir(const char* name) {
  auto p = fs::temp_directory_path() / (std::string("clm-e2e-") + name + "-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  fs::remove_all(p);
  fs::create_directories(p);
  return p;
}

std::uint16_t free_port() {
  transport::SecurityConfig sec;
  sec.mode = transport::SecurityConfig::Mode::kInsecureLoopbackOnly;
  auto l = transport::listen({"127.0.0.1", 0}, sec);
  REQUIRE(l.is_ok());
  const auto port = l.value()->local_endpoint().port;
  l.value()->close();
  return port;
}

std::string field(const std::string& line, const std::string& key) {
  std::istringstream ss(line);
  std::string tok;
  while (ss >> tok)
    if (tok.rfind(key + "=", 0) == 0) return tok.substr(key.size() + 1);
  return {};
}

struct NodeProc {
  std::string name;
  fs::path dir;
  std::unique_ptr<platform::ChildProcess> proc;
  std::string code, pair_endpoint, fingerprint_short;
  std::uint16_t port = 0;

  void start(const std::string& n, const fs::path& root) {
    name = n;
    dir = root / n;
    fs::create_directories(dir);
    port = free_port();
    set_env("XDG_STATE_HOME", (dir / "state").string());  // node_root etc. stay inside the test directory
    set_env("XDG_DATA_HOME", (dir / "data").string());
    auto p = platform::ChildProcess::spawn(
        CLUSTERLM_NODE_SERVICE_BINARY,
        {"--console", "--simulate-activity", "--pair", "--pair-port", "0", "--name", n, "--port", std::to_string(port),
         "--staging", (dir / "staging").string(), "--identity", (dir / "identity").string(), "--settings", (dir / "node.json").string(),
         "--ipc-dir", (dir / "ipc").string(), "--idle-seconds", "1", "--ram-gib", "2", "--vram-gib", "1", "--worker", CLUSTERLM_NODE_BINARY, "--deadline-ms", "3000"});
    REQUIRE_MESSAGE(p.is_ok(), p.status().to_string());
    proc = std::move(p).value();
    auto line = proc->read_until("CLUSTERLM_NODE_PAIRING ", 30s);
    REQUIRE_MESSAGE(line.is_ok(), line.status().to_string());
    code = field(line.value(), "code");
    const auto ep = field(line.value(), "endpoint");
    pair_endpoint = "127.0.0.1:" + ep.substr(ep.rfind(':') + 1);
    fingerprint_short = field(line.value(), "fingerprint");
    REQUIRE_FALSE(code.empty());
  }
  ~NodeProc() {
    if (proc) {
      (void)proc->write_line("quit");
      proc->close_stdin();
      if (!proc->wait(10s).is_ok()) proc->kill();
    }
  }
};

class UiClient {
 public:
  explicit UiClient(std::unique_ptr<ipc::Connection> c) : conn_(std::move(c)) {}

  // Sends one request, collects events that arrive before its reply.
  json call(json req) {
    req["id"] = ++id_;
    const std::string text = req.dump();
    REQUIRE(conn_->send(ipc::father_envelope(ipc::MessageKind::kFatherRequest, Bytes(text.begin(), text.end())), 5s).is_ok());
    for (;;) {
      auto e = conn_->receive(60s);
      const std::string why = req["op"].get<std::string>() + ": " + e.status().to_string();
      REQUIRE_MESSAGE(e.is_ok(), why);
      json j = json::parse(e->payload.begin(), e->payload.end(), nullptr, false);
      if (e->kind == static_cast<std::uint16_t>(ipc::MessageKind::kFatherEvent)) {
        events.push_back(j);
        continue;
      }
      REQUIRE(e->kind == static_cast<std::uint16_t>(ipc::MessageKind::kFatherReply));
      REQUIRE(j["id"] == id_);
      return j;
    }
  }

  // Waits until `pred` holds for some event (already collected or newly arriving).
  template <typename Pred>
  bool wait_event(Pred pred, std::chrono::seconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    std::size_t seen = 0;
    for (;;) {
      for (; seen < events.size(); ++seen)
        if (pred(events[seen])) return true;
      if (std::chrono::steady_clock::now() > deadline) return false;
      auto e = conn_->receive(500ms);
      if (e.is_ok() && e->kind == static_cast<std::uint16_t>(ipc::MessageKind::kFatherEvent))
        events.push_back(json::parse(e->payload.begin(), e->payload.end(), nullptr, false));
    }
  }
  std::vector<json> events;

 private:
  std::unique_ptr<ipc::Connection> conn_;
  std::uint64_t id_ = 0;
};

struct AgentProc {
  fs::path dir;
  std::unique_ptr<platform::ChildProcess> proc;
  std::unique_ptr<UiClient> ui;

  void start(const fs::path& root, bool dev) {
    dir = root / "father";
    fs::create_directories(dir);
    set_env("XDG_DATA_HOME", (dir / "data").string());
    set_env("XDG_STATE_HOME", (dir / "state").string());
    set_env("XDG_RUNTIME_DIR", (dir / "run").string());
    fs::create_directories(dir / "run");
    std::vector<std::string> args = {"--settings", (dir / "father.json").string(), "--identity", (dir / "identity").string(),
                                     "--catalog", std::string(CLUSTERLM_SOURCE_DIR) + "/fixtures/catalog/clusterlm-catalog.json",
                                     "--profiles-dir", std::string(CLUSTERLM_SOURCE_DIR) + "/fixtures/profiles",
                                     "--user-tag", "e2e", "--ipc-dir", (dir / "ipc").string(), "--exit-on-stdin-eof"};
    if (dev) args.push_back("--dev-fixture-model");
    auto p = platform::ChildProcess::spawn(CLUSTERLM_FATHER_AGENT_BINARY, args);
    REQUIRE_MESSAGE(p.is_ok(), p.status().to_string());
    proc = std::move(p).value();
    auto line = proc->read_until("CLUSTERLM_FATHER_AGENT", 30s);
    REQUIRE_MESSAGE(line.is_ok(), line.status().to_string());
    ipc::Endpoint ep;
    ep.name = ipc::father_ui_pipe_name("e2e");
    ep.socket_dir = dir / "ipc";
    auto c = ipc::connect(ep, {}, 5s);
    REQUIRE_MESSAGE(c.is_ok(), c.status().to_string());
    ui = std::make_unique<UiClient>(std::move(c).value());
  }
  ~AgentProc() {
    ui.reset();
    if (proc) {
      proc->close_stdin();
      if (!proc->wait(15s).is_ok()) proc->kill();
    }
  }
};

json tier_of(const json& list, const char* id) {
  for (const auto& t : list["result"]["tiers"])
    if (t["tier_id"] == id) return t;
  return json::object();
}

}  // namespace

TEST_CASE("dev end to end: pair two nodes, assign tiers, prepare Ultra on the fixture model, chat, unpair") {
  const fs::path root = temp_dir("full");
  {
    auto manifest = objects::write_fixture_model(objects::FixtureSpec{}, root / "model");
    REQUIRE(manifest.is_ok());

    NodeProc g14, n3060;
    g14.start("g14", root);
    n3060.start("n3060", root);
    AgentProc agent;
    agent.start(root, /*dev=*/true);
    auto& ui = *agent.ui;

    auto hello = ui.call({{"op", "hello"}});
    REQUIRE(hello["ok"] == true);
    CHECK(hello["result"]["dev_fixture_model"] == true);

    // ---- pairing ----
    auto wrong = ui.call({{"op", "pairing.start"}, {"address", g14.pair_endpoint}, {"code", "ZZZZ-2222"}});
    CHECK(wrong["ok"] == false);
    CHECK(wrong["error"]["code"] == "UNAUTHENTICATED");
    auto p1 = ui.call({{"op", "pairing.start"}, {"address", g14.pair_endpoint}, {"code", g14.code}});
    REQUIRE_MESSAGE(p1["ok"] == true, p1.dump());
    auto p2 = ui.call({{"op", "pairing.start"}, {"address", n3060.pair_endpoint}, {"code", n3060.code}});
    REQUIRE_MESSAGE(p2["ok"] == true, p2.dump());
    const std::string fp_g14 = p1["result"]["device"]["fingerprint"];
    const std::string fp_3060 = p2["result"]["device"]["fingerprint"];
    CHECK(p1["result"]["device"]["short"].get<std::string>() == g14.fingerprint_short);  // user can compare both screens
    CHECK(p1["result"]["device"]["address"] == "127.0.0.1:" + std::to_string(g14.port));
    REQUIRE(g14.proc->read_until("CLUSTERLM_NODE_PAIRED", 30s).is_ok());
    REQUIRE(n3060.proc->read_until("CLUSTERLM_NODE_PAIRED", 30s).is_ok());
    auto listed = ui.call({{"op", "pairing.list"}});
    CHECK(listed["result"]["devices"].size() == 2);

    // ---- assignment + dev model ----
    auto assigned = ui.call({{"op", "assign.set"}, {"assignments", {{"node:laptop-class", fp_g14}, {"node:designated-3060", fp_3060}}}});
    REQUIRE_MESSAGE(assigned["ok"] == true, assigned.dump());
    auto set = ui.call({{"op", "settings.set"}, {"patch", {{"model_dirs", {{"ultra", (root / "model").string()}}}}}});
    REQUIRE_MESSAGE(set["ok"] == true, set.dump());

    // Nodes go idle (simulated): the paired, restarted workers are offered again.
    REQUIRE(g14.proc->write_line("idle").is_ok());
    REQUIRE(n3060.proc->write_line("idle").is_ok());

    // ---- readiness is honest: Available once model verification + both nodes are in order ----
    json ultra;
    for (int i = 0; i < 120; ++i) {
      ultra = tier_of(ui.call({{"op", "tiers.list"}}), "ultra");
      if (ultra["state"] == "Available") break;
      std::this_thread::sleep_for(500ms);
    }
    REQUIRE_MESSAGE(ultra["state"] == "Available", ultra.dump());
    bool says_dev = false, says_synthetic = false;
    for (const auto& d : ultra["details"]) {
      says_dev = says_dev || d.get<std::string>().find("DEVELOPMENT FIXTURE MODEL") != std::string::npos;
      says_synthetic = says_synthetic || d.get<std::string>().find("SYNTHETIC") != std::string::npos;
    }
    CHECK(says_dev);
    CHECK(says_synthetic);
    // Fast has no model directory: never Ready.
    CHECK(tier_of(ui.call({{"op", "tiers.list"}}), "fast")["state"] != "Ready");

    // ---- prepare -> chat ----
    REQUIRE(ui.call({{"op", "tiers.select"}, {"tier_id", "ultra"}})["ok"] == true);
    auto prep = ui.call({{"op", "tiers.prepare"}, {"tier_id", "ultra"}, {"context_tokens", 4096}});
    REQUIRE_MESSAGE(prep["ok"] == true, prep.dump());
    REQUIRE(ui.wait_event([](const json& e) { return e["event"] == "tier_ready" || e["event"] == "error"; }, 120s));
    bool ready = false;
    for (const auto& e : ui.events) ready = ready || e["event"] == "tier_ready";
    REQUIRE_MESSAGE(ready, "tier did not become ready");

    auto chat = ui.call({{"op", "chat.send"}, {"message", "hello"}, {"max_new_tokens", 12}, {"context_tokens", 4096}});
    REQUIRE_MESSAGE(chat["ok"] == true, chat.dump());
    const auto rid = chat["result"]["request_id"].get<std::uint64_t>();
    REQUIRE(ui.wait_event([&](const json& e) { return e["event"] == "finished" && e["request_id"] == rid; }, 120s));
    std::size_t token_events = 0, tokens = 0;
    std::string finish_reason;
    for (const auto& e : ui.events) {
      if (e["event"] == "tokens") {
        ++token_events;
        tokens += e["token_count"].get<std::size_t>();
        CHECK(e["tier_id"] == "ultra");
        CHECK(e["model_name"].get<std::string>().find("Qwen3.8-Flash-Next") != std::string::npos);
      }
      if (e["event"] == "finished") finish_reason = e["reason"];
    }
    CHECK(token_events >= 1);
    CHECK(tokens == 12);
    CHECK(finish_reason == "completed");

    // Redacted diagnostics by default; text only on request.
    auto diag = ui.call({{"op", "diagnostics.get"}});
    CHECK(diag["result"]["text_included"] == false);
    CHECK(diag["result"].contains("conversation") == false);

    // ---- release, then unpair: the device and its assignment disappear ----
    REQUIRE(ui.call({{"op", "session.release"}})["ok"] == true);
    auto un = ui.call({{"op", "pairing.unpair"}, {"fingerprint", fp_g14}});
    REQUIRE_MESSAGE(un["ok"] == true, un.dump());
    auto after = ui.call({{"op", "pairing.list"}});
    CHECK(after["result"]["devices"].size() == 1);
    CHECK(after["result"]["assignments"].contains("node:laptop-class") == false);
    auto gone = tier_of(ui.call({{"op", "tiers.list"}}), "ultra");
    CHECK(gone["state"] == "Unavailable");
  }
  std::error_code ec;
  fs::remove_all(root, ec);
}

TEST_CASE("without the dev override no tier is ever Ready: the backend is not available in this build") {
  const fs::path root = temp_dir("nodev");
  {
    auto manifest = objects::write_fixture_model(objects::FixtureSpec{}, root / "model");
    REQUIRE_MESSAGE(manifest.is_ok(), manifest.status().to_string());
    AgentProc agent;
    agent.start(root, /*dev=*/false);
    auto& ui = *agent.ui;
    REQUIRE(ui.call({{"op", "settings.set"}, {"patch", {{"model_dirs", {{"fast", (root / "model").string()}}}}}})["ok"] == true);
    REQUIRE(ui.call({{"op", "model.confirm"}, {"tier_id", "fast"}})["ok"] == true);
    json fast;
    for (int i = 0; i < 40; ++i) {
      fast = tier_of(ui.call({{"op", "tiers.list"}}), "fast");
      std::this_thread::sleep_for(250ms);
    }
    CHECK(fast["state"] == "Unavailable");
    bool backend_blocker = false;
    for (const auto& r : fast["reasons"]) backend_blocker = backend_blocker || r.get<std::string>().find("backend not available in this build") != std::string::npos;
    CHECK(backend_blocker);
    auto prep = ui.call({{"op", "tiers.prepare"}, {"tier_id", "fast"}});
    // Accepted asynchronously; the refusal arrives as an error event and the tier never becomes ready.
    CHECK(ui.wait_event([](const json& e) { return e["event"] == "error"; }, 30s));
    for (const auto& e : ui.events) CHECK(e["event"] != "tier_ready");
  }
  std::error_code ec;
  fs::remove_all(root, ec);
}
