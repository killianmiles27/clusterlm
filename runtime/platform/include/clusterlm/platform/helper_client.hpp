#pragma once
// HelperClient: the session helper's side of the helper <-> Node service protocol.
//
// Samples a local ActivityMonitor (Windows: GetLastInputInfo + lock state) and reports it to the service, one
// ActivityReport per call. Connection loss is handled by reconnecting on the next call; while the service is
// unreachable nothing is cached or replayed (the service fails closed on missing reports, which is the point).
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>

#include "clusterlm/platform/adapters.hpp"
#include "clusterlm/platform/ipc.hpp"
#include "clusterlm/platform/ipc_messages.hpp"

namespace clusterlm::platform {

class HelperClient {
 public:
  struct Options {
    ipc::Endpoint endpoint{ipc::kNodeHelperPipeName, {}};
    ipc::ClientOptions client;
    std::uint32_t session_id = 0;  // the helper's own terminal-services session
    std::chrono::milliseconds connect_timeout{300};
    std::chrono::milliseconds io_timeout{2000};
  };

  HelperClient(Options options, ActivityMonitor& source);

  // Samples and reports once. If sampling fails the report says "in use" (idle 0, not locked): the helper fails
  // closed too. Returns the service's verdict; an Ack with a non-OK code is returned as that error.
  Status report_once();
  Result<ipc::StatusReply> status();
  Status pause(std::chrono::seconds duration = std::chrono::seconds(0));  // 0 = until resume()
  Status resume();
  void disconnect();
  bool connected() const;

 private:
  Status ensure_connected();
  Result<ipc::Envelope> call(const ipc::Envelope& request);
  Status expect_ack(const ipc::Envelope& reply);

  Options opts_;
  ActivityMonitor& source_;
  mutable std::mutex mu_;
  std::unique_ptr<ipc::Connection> conn_;
};

}  // namespace clusterlm::platform
