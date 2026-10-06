// clusterlm-father-agent: per-user background agent for ClusterLM Father.
//
// Hosts the Coordinator configuration and serves the Father UI over a local named pipe (Windows) / Unix socket
// (development). The Father service API itself is another workstream: this process currently answers every UI
// request with kUnimplemented, but the transport, access control and lifecycle are final.
//
//   clusterlm-father-agent [--model DIR] [--user-tag TAG] [--ipc-dir DIR] [--node NAME=HOST:PORT]...
//
// Stops on Ctrl-C / SIGTERM (console) or when its stdin closes with --exit-on-stdin-eof (harnesses).
#include <atomic>
#include <cstdio>
#include <iostream>
#include <thread>

#include "cli.hpp"
#include "clusterlm/common/log.hpp"
#include "clusterlm/platform/paths.hpp"
#include "clusterlm/platform/service_host.hpp"
#include "clusterlm/father/father_agent.hpp"

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

}  // namespace

int main(int argc, char** argv) {
  cli::Args args(argc, argv);
  log::set_component("father-agent");
  father::FatherAgentConfig cfg;
  cfg.coordinator.model_dir = args.get("model");
  for (const auto& spec : args.all("node")) {
    const auto eq = spec.find('=');
    if (eq == std::string::npos) continue;
    auto ep = transport::Endpoint::parse(spec.substr(eq + 1));
    if (!ep.is_ok()) {
      std::fprintf(stderr, "error: %s\n", ep.status().to_string().c_str());
      return 2;
    }
    cfg.coordinator.nodes.push_back({spec.substr(0, eq), ep.value(), ""});
  }
  auto user = ipc::current_user_id();
  cfg.user_tag = args.get("user-tag", user.is_ok() ? user.value() : "default");
  auto paths = platform::default_paths();
  cfg.ipc_dir = args.get("ipc-dir", paths.is_ok() ? paths->ipc_dir.string() : std::string());

  father::UnimplementedFatherApi api;
  auto agent = father::FatherAgent::create(std::move(cfg), api);
  if (!agent.is_ok()) {
    std::fprintf(stderr, "error: %s\n", agent.status().to_string().c_str());
    return 1;
  }
  AgentApp app(*agent.value(), args.has("exit-on-stdin-eof"));
  auto rc = platform::run_in_console(app, {});  // a per-user agent is an ordinary process, not an SCM service
  if (!rc.is_ok()) {
    std::fprintf(stderr, "error: %s\n", rc.status().to_string().c_str());
    return 1;
  }
  return rc.value();
}
