#pragma once
// ServerLoop: accept clients on an ipc::Server and run one handler thread per connection (bounded).
// Rejected peers are dropped (and counted), never logged by identity. stop() closes the server and every live
// connection, then joins all threads, so a handler only has to return when `stop` is set or the connection ends.
#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "clusterlm/platform/ipc.hpp"

namespace clusterlm::ipc {

class ServerLoop {
 public:
  // The handler runs on its own thread for the life of the connection.
  using Handler = std::function<void(const std::shared_ptr<Connection>& conn, const std::atomic<bool>& stop)>;

  ServerLoop(std::unique_ptr<Server> server, Handler handler, std::size_t max_connections = 8);
  ~ServerLoop();
  ServerLoop(const ServerLoop&) = delete;
  ServerLoop& operator=(const ServerLoop&) = delete;

  void start();
  void stop();
  // Currently open connections (e.g. to broadcast an event).
  std::vector<std::shared_ptr<Connection>> connections();
  std::uint64_t rejected_peers() const { return rejected_.load(); }

 private:
  void accept_loop();
  struct Entry {
    std::thread thread;
    std::shared_ptr<Connection> conn;
    std::shared_ptr<std::atomic<bool>> done;
  };

  std::unique_ptr<Server> server_;
  Handler handler_;
  std::size_t max_connections_;
  std::atomic<bool> stop_{false};
  std::atomic<std::uint64_t> rejected_{0};
  std::thread accept_thread_;
  std::mutex mu_;
  std::vector<Entry> entries_;
};

}  // namespace clusterlm::ipc
