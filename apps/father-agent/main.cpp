// clusterlm-father-agent: per-user background agent for ClusterLM Father.
//
// Runs FatherService with the production providers (persisted settings, live readiness, config-driven
// placement) and serves the Father UI over a local named pipe (Windows) / Unix socket (development). The IPC
// protocol is documented in docs/father-ipc.md.
//
//   clusterlm-father-agent [--settings FILE] [--identity DIR] [--catalog FILE] [--profiles-dir DIR]
//                          (--catalog default: <exe dir>/../catalog/clusterlm-catalog.json when installed, else
//                           <exe dir>/clusterlm-catalog.json)
//                          [--user-tag TAG] [--ipc-dir DIR] [--exit-on-stdin-eof]
//                          [--dev-fixture-model]
//
// --dev-fixture-model is a DEVELOPMENT override: the model directory configured for a tier holds the small
// fixture model, which the reference backend can run. It marks the backend available, auto-confirms the unpinned
// model and labels every affected tier. Without it no tier is ever Ready in this build ("backend not available
// in this build").
//
// Stops on Ctrl-C / SIGTERM (console) or when its stdin closes with --exit-on-stdin-eof (harnesses).
#include <atomic>
#include <cstdio>
#include <iostream>
#include <thread>

#include "cli.hpp"
#include "clusterlm/common/log.hpp"
#include "clusterlm/father/father_agent.hpp"
#include "clusterlm/father/father_service_api.hpp"
#include "clusterlm/platform/adapters.hpp"
#include "clusterlm/platform/mock_adapters.hpp"
#include "clusterlm/platform/paths.hpp"
#include "clusterlm/platform/process.hpp"
#include "clusterlm/platform/service_host.hpp"

using namespace clusterlm;

namespace {

class AgentApp final : public platform::ServiceApp {
 public:
  AgentApp(father::FatherAgent& agent, bool exit_on_eof) : agent_(agent), exit_on_eof_(exit_on_eof) {}
  Status on_start() override {
    CLM_RETURN_IF_ERROR(agent_.start());
    std::printf("CLUSTERLM_FATHER_AGENT pipe=%s\n", agent_.endpoint().name.c_str());
    std::fflush(stdout);
    if (exit_on_eof_)
      watcher_ = std::thread([this] {
        std::string line;
        while (std::getline(std::cin, line)) {
        }
        stop_.store(true);
      });
    return Status::ok();
  }
  int run() override {
    while (!stop_.load()) std::this_thread::sleep_for(std::chrono::milliseconds(50));
    agent_.stop();
    if (watcher_.joinable()) watcher_.detach();
    return 0;
  }
  void on_stop(platform::StopReason) override { stop_.store(true); }

 private:
  father::FatherAgent& agent_;
  bool exit_on_eof_;
  std::atomic<bool> stop_{false};
  std::thread watcher_;
};

int fail(const Status& s) {
  std::fprintf(stderr, "error: %s\n", s.to_string().c_str());
  return 1;
}

}  // namespace

int main(int argc, char** argv) {
  cli::Args args(argc, argv);
  log::set_component("father-agent");
  auto paths = platform::default_paths();
  if (!paths.is_ok()) return fail(paths.status());

  auto settings = config::FatherSettingsStore::open(args.get("settings", config::father_settings_path(paths.value()).string()));
  if (!settings.is_ok()) return fail(settings.status());
  if (settings.value()->report().outcome == config::LoadReport::Outcome::kRecoveredCorrupt)
    std::fprintf(stderr, "warning: %s\n", settings.value()->report().note.c_str());
  std::shared_ptr<config::FatherSettingsStore> settings_shared(std::move(settings).value());

  const std::filesystem::path identity_dir = args.get("identity", paths->father_identity.string());
  auto ident = transport::DeviceIdentity::load_or_generate(identity_dir, "clusterlm-father");
  if (!ident.is_ok()) return fail(ident.status());
  auto identity = std::make_shared<const transport::DeviceIdentity>(std::move(ident).value());

  // Installed layout first (<exe dir>/../catalog/), then the development copy next to the executable.
  const std::filesystem::path catalog_path =
      args.has("catalog") ? std::filesystem::path(args.get("catalog")) : father::default_catalog_path(platform::executable_dir());
  auto cat = catalog::Catalog::load(catalog_path.string());
  if (!cat.is_ok()) return fail(cat.status());

  const bool dev = args.has("dev-fixture-model");
  auto api_slot = std::make_shared<std::atomic<father::FatherServiceApi*>>(nullptr);
  father::ProductionOptions po;
  po.settings = settings_shared;
  po.identity = identity;
  po.profiles_dir = args.get("profiles-dir", (platform::executable_dir() / "profiles").string());
  po.dev_fixture_model = dev;
  // Preparation progress (bytes, objects, phase) from the Coordinator reaches readiness observation and the UI.
  auto provisioning_board = std::make_shared<father::ProvisioningBoard>();
  po.provisioning = provisioning_board->provider();
  po.session_phase = [api_slot] {
    auto* a = api_slot->load();
    return a ? a->session_phase() : father::SessionPhase::kNone;
  };
#ifdef _WIN32
  {
    std::shared_ptr<platform::PowerMonitor> power = platform::make_windows_power_monitor();
    po.father_power = [power] {
      catalog::PowerState st;
      auto s = power->sample();
      if (s.is_ok()) {
        st.on_ac = s->on_ac_power;
        st.battery_saver = s->battery_saver;
      }
      return st;
    };
  }
#endif

  father::FatherApiConfig ac;
  ac.catalog = std::move(cat).value();
  ac.settings = settings_shared;
  ac.identity = identity;
  ac.readiness = std::make_shared<father::LiveReadinessSource>(po);
  ac.deployments = std::make_shared<father::ConfigDeploymentProvider>(po);
  ac.details = po.details;
  ac.prepare_observer = provisioning_board->observer();
  ac.dev_fixture_model = dev;
  if (dev) {
    auto tok = father::FixtureByteTokenizer::create(256);
    if (!tok.is_ok()) return fail(tok.status());
    ac.tokenizer = tok.value();
  }
  auto api = father::FatherServiceApi::create(std::move(ac));
  if (!api.is_ok()) return fail(api.status());
  api_slot->store(api.value().get());

  father::FatherAgentConfig cfg;
  auto user = ipc::current_user_id();
  cfg.user_tag = args.get("user-tag", user.is_ok() ? user.value() : "default");
  cfg.ipc_dir = args.get("ipc-dir", paths->ipc_dir.string());
  auto agent = father::FatherAgent::create(std::move(cfg), *api.value());
  if (!agent.is_ok()) return fail(agent.status());
  api.value()->set_event_push([a = agent.value().get()](Bytes b) { a->broadcast(std::move(b)); });

  AgentApp app(*agent.value(), args.has("exit-on-stdin-eof"));
  auto rc = platform::run_in_console(app, {});  // a per-user agent is an ordinary process, not an SCM service
  api_slot->store(nullptr);
  if (!rc.is_ok()) return fail(rc.status());
  return rc.value();
}
