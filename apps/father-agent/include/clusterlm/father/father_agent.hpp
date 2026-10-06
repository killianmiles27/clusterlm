#pragma once
// FatherAgent: the per-user background process that hosts the Coordinator and serves the Father UI over a local
// pipe. This file is the seam to the Father service workstream:
//
//   UI process <-- local pipe (ipc::Envelope) --> FatherAgent --> FatherApiHandler --> Coordinator / planner
//
// The agent owns only the transport, the access control (the pipe is reachable by the current user and SYSTEM
// only, and the peer's user must equal the agent's) and the CoordinatorConfig. What a request means is entirely
// the FatherApiHandler's business. Protocol, all envelopes version kFatherUiProtocolVersion:
//   UI -> agent   kFatherRequest {opaque payload}
//   agent -> UI   kFatherReply   {opaque payload}   success
//                 kAck {code != kOk, short message} failure (also: unsupported version / kind)
//   agent -> UI   kFatherEvent   {opaque payload}   unsolicited, via FatherAgent::broadcast
#include <atomic>
#include <filesystem>
#include <memory>
#include <string>

#include "clusterlm/coordinator/coordinator.hpp"
#include "clusterlm/platform/ipc.hpp"
#include "clusterlm/platform/ipc_messages.hpp"
#include "clusterlm/platform/ipc_server_loop.hpp"

namespace clusterlm::father {

// Where the shipped tier catalog is when --catalog is not given. Installed layout (cmake/ClusterLMInstall.cmake):
// <prefix>/bin/clusterlm-father-agent.exe next to <prefix>/catalog/clusterlm-catalog.json, so
// `<exe dir>/../catalog/clusterlm-catalog.json` comes first; the development layout (the catalog copied next to the
// executables) is the fallback, and is also what is returned, for the error message, when neither file exists.
std::filesystem::path default_catalog_path(const std::filesystem::path& exe_dir);

class FatherApiHandler {
 public:
  virtual ~FatherApiHandler() = default;
  // Called on a per-connection thread for every kFatherRequest. Return the reply envelope (kind kFatherReply) or an
  // error Status, which the agent maps to an Ack. Must not log or retain payload contents.
  virtual Result<ipc::Envelope> handle(const ipc::Envelope& request, const ipc::PeerCredentials& peer) = 0;
};

// Placeholder until the Father service API lands: every request fails with kUnimplemented.
class UnimplementedFatherApi final : public FatherApiHandler {
 public:
  Result<ipc::Envelope> handle(const ipc::Envelope&, const ipc::PeerCredentials&) override {
    return make_error(ErrorCode::kUnimplemented, "the Father service API is not implemented yet");
  }
};

struct FatherAgentConfig {
  coordinator::CoordinatorConfig coordinator;  // owned configuration; the handler decides when to create the Coordinator
  std::string user_tag;                        // distinguishes per-user pipes: usually the user name or SID
  std::filesystem::path ipc_dir;               // POSIX only
  std::uint32_t max_frame_bytes = ipc::kDefaultMaxFrameBytes;
};

class FatherAgent {
 public:
  static Result<std::unique_ptr<FatherAgent>> create(FatherAgentConfig config, FatherApiHandler& handler);
  ~FatherAgent();

  Status start();
  void stop();
  // Sends a kFatherEvent to every connected UI. Slow or dead clients are dropped, never waited on for long.
  void broadcast(Bytes payload);

  const coordinator::CoordinatorConfig& coordinator_config() const { return cfg_.coordinator; }
  const ipc::Endpoint& endpoint() const { return endpoint_; }

 private:
  FatherAgent(FatherAgentConfig config, FatherApiHandler& handler, ipc::Endpoint endpoint)
      : cfg_(std::move(config)), handler_(handler), endpoint_(std::move(endpoint)) {}
  void serve(const std::shared_ptr<ipc::Connection>& conn, const std::atomic<bool>& stop);

  FatherAgentConfig cfg_;
  FatherApiHandler& handler_;
  ipc::Endpoint endpoint_;
  std::unique_ptr<ipc::ServerLoop> loop_;
};

}  // namespace clusterlm::father
