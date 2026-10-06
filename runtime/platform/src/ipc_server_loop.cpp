#include "clusterlm/platform/ipc_server_loop.hpp"

#include "clusterlm/common/log.hpp"

namespace clusterlm::ipc {

using namespace std::chrono_literals;

ServerLoop::ServerLoop(std::unique_ptr<Server> server, Handler handler, std::size_t max_connections)
    : server_(std::move(server)), handler_(std::move(handler)), max_connections_(max_connections) {}

ServerLoop::~ServerLoop() { stop(); }

void ServerLoop::start() {
  if (!accept_thread_.joinable()) accept_thread_ = std::thread([this] { accept_loop(); });
}

void ServerLoop::stop() {
  stop_.store(true);
  if (server_) server_->close();
  if (accept_thread_.joinable()) accept_thread_.join();
  std::vector<Entry> entries;
  {
    std::lock_guard lock(mu_);
    entries.swap(entries_);
  }
  for (auto& e : entries) e.conn->close();
  for (auto& e : entries)
    if (e.thread.joinable()) e.thread.join();
}

std::vector<std::shared_ptr<Connection>> ServerLoop::connections() {
  std::lock_guard lock(mu_);
  std::vector<std::shared_ptr<Connection>> out;
  for (auto& e : entries_)
    if (!e.done->load() && e.conn->is_open()) out.push_back(e.conn);
  return out;
}

void ServerLoop::accept_loop() {
  while (!stop_.load()) {
    auto c = server_->accept(200ms);
    {  // reap finished handlers
      std::lock_guard lock(mu_);
      for (auto it = entries_.begin(); it != entries_.end();) {
        if (it->done->load()) {
          it->thread.join();
          it = entries_.erase(it);
        } else {
          ++it;
        }
      }
    }
    if (!c.is_ok()) {
      const auto code = c.status().code();
      if (code == ErrorCode::kDeadlineExceeded) continue;
      if (code == ErrorCode::kCancelled) break;
      rejected_.fetch_add(1);
      log::warn("ipc_connection_rejected", {{"code", std::string(to_string(code))}});  // the peer's identity is not logged
      if (code != ErrorCode::kPermissionDenied && code != ErrorCode::kUnauthenticated) std::this_thread::sleep_for(100ms);
      continue;
    }
    std::shared_ptr<Connection> conn(std::move(c).value());
    std::lock_guard lock(mu_);
    if (entries_.size() >= max_connections_) {
      conn->close();
      rejected_.fetch_add(1);
      continue;
    }
    auto done = std::make_shared<std::atomic<bool>>(false);
    entries_.push_back({std::thread([this, conn, done] {
                          handler_(conn, stop_);
                          conn->close();
                          done->store(true);
                        }),
                        conn, done});
  }
}

}  // namespace clusterlm::ipc
