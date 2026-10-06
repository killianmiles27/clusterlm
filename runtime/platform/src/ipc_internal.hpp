#pragma once
// Byte-stream layer under ipc::Connection/Server. One implementation per OS (ipc_posix.cpp, windows/ipc_win.cpp);
// framing, bounds and authorization live above it in ipc.cpp so they are identical on every platform.
#include <chrono>
#include <memory>

#include "clusterlm/platform/ipc.hpp"

namespace clusterlm::ipc::detail {

class RawChannel {
 public:
  virtual ~RawChannel() = default;
  // Reads 1..n bytes. Returns 0 on timeout (nothing consumed). kUnavailable at EOF/peer gone, kCancelled after close().
  virtual Result<std::size_t> read_some(std::uint8_t* buf, std::size_t n, std::chrono::milliseconds timeout) = 0;
  virtual Status write_all(const std::uint8_t* buf, std::size_t n, std::chrono::milliseconds timeout) = 0;
  // Idempotent; wakes blocked reads/writes from other threads.
  virtual void close() = 0;
  virtual bool is_open() const = 0;
  virtual const PeerCredentials& peer() const = 0;
};

class RawListener {
 public:
  virtual ~RawListener() = default;
  // kDeadlineExceeded on timeout. A client that fails the OS-level check yields kPermissionDenied.
  virtual Result<std::unique_ptr<RawChannel>> accept(std::chrono::milliseconds timeout) = 0;
  virtual void close() = 0;
};

Result<std::unique_ptr<RawListener>> make_listener(const ServerOptions& options);
Result<std::unique_ptr<RawChannel>> make_client(const Endpoint& endpoint, const ClientOptions& options,
                                                std::chrono::milliseconds timeout);

}  // namespace clusterlm::ipc::detail
