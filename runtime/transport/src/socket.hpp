#pragma once
// Internal socket layer: POSIX sockets on Linux, Winsock2 on Windows. All sockets are non-blocking and all
// waits are poll-based in short slices so a cancel flag is honoured promptly on every platform.
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>

#include "clusterlm/common/clock.hpp"
#include "clusterlm/common/status.hpp"

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#endif

namespace clusterlm::transport::net {

#ifdef _WIN32
using Handle = SOCKET;
inline const Handle kInvalidHandle = INVALID_SOCKET;
#else
using Handle = int;
inline constexpr Handle kInvalidHandle = -1;
#endif

using Deadline = SteadyClock::time_point;
inline constexpr std::chrono::milliseconds kPollSlice{20};

// `now + timeout`, saturating so huge timeouts mean "never".
Deadline deadline_after(std::chrono::milliseconds timeout);
inline Deadline no_deadline() { return Deadline::max(); }

// Owns a socket handle. Not copyable; the handle is closed only in the destructor/reset so that shutdown()
// from another thread can never race with fd reuse.
class Socket {
 public:
  Socket() = default;
  explicit Socket(Handle h) : h_(h) {}
  ~Socket() { reset(); }
  Socket(Socket&& o) noexcept : h_(o.h_) { o.h_ = kInvalidHandle; }
  Socket& operator=(Socket&& o) noexcept {
    if (this != &o) {
      reset();
      h_ = o.h_;
      o.h_ = kInvalidHandle;
    }
    return *this;
  }
  Socket(const Socket&) = delete;
  Socket& operator=(const Socket&) = delete;

  bool valid() const { return h_ != kInvalidHandle; }
  Handle handle() const { return h_; }
  void reset();
  // shutdown(SHUT_RDWR): wakes threads blocked in poll/recv on this socket.
  void shutdown_both() const;

 private:
  Handle h_ = kInvalidHandle;
};

enum class WaitResult : std::uint8_t { kReady, kTimeout, kCancelled, kError };
// Waits for readability/writability until `deadline`, or until `*cancel` becomes true.
WaitResult wait_ready(Handle h, bool for_write, Deadline deadline, const std::atomic<bool>* cancel);

enum class IoState : std::uint8_t { kOk, kWouldBlock, kClosed, kError };
struct IoResult {
  IoState state = IoState::kError;
  std::size_t n = 0;
};
IoResult raw_recv(Handle h, void* buf, std::size_t n);
IoResult raw_send(Handle h, const void* buf, std::size_t n);

// Connect to host:port. With loopback_only, every resolved address must be loopback (else kPermissionDenied).
Result<Socket> tcp_connect(const std::string& host, std::uint16_t port, bool loopback_only, Deadline deadline);

struct ListenSocket {
  Socket socket;
  std::uint16_t port = 0;
};
Result<ListenSocket> tcp_listen(const std::string& host, std::uint16_t port, bool loopback_only);

struct Accepted {
  Socket socket;
  std::string peer_address;  // "ip:port"
  bool peer_is_loopback = false;
};
// Non-blocking accept on a listening socket whose readiness was already established. Returns kUnavailable with
// message "again" when there is nothing to accept.
Result<Accepted> tcp_accept(Handle listener);

// True for literal loopback host names/addresses (no DNS).
bool is_loopback_host(const std::string& host);

}  // namespace clusterlm::transport::net
