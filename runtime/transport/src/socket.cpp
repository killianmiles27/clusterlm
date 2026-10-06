#include "socket.hpp"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstring>
#include <memory>

#ifndef _WIN32
#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace clusterlm::transport::net {
namespace {

#ifdef _WIN32
int last_error() { return WSAGetLastError(); }
bool err_would_block(int e) { return e == WSAEWOULDBLOCK; }
bool err_in_progress(int e) { return e == WSAEWOULDBLOCK || e == WSAEINPROGRESS; }
bool err_interrupted(int) { return false; }
constexpr int kSockError = SOCKET_ERROR;

// WSAStartup exactly once, before any socket call.
struct WinsockInit {
  WinsockInit() {
    WSADATA data;
    status = WSAStartup(MAKEWORD(2, 2), &data);
  }
  int status = 0;
};
Status ensure_winsock() {
  static const WinsockInit init;
  if (init.status != 0) return make_error(ErrorCode::kInternal, "WSAStartup failed");
  return Status::ok();
}
#else
int last_error() { return errno; }
bool err_would_block(int e) { return e == EAGAIN || e == EWOULDBLOCK; }
bool err_in_progress(int e) { return e == EINPROGRESS; }
bool err_interrupted(int e) { return e == EINTR; }
constexpr int kSockError = -1;
Status ensure_winsock() { return Status::ok(); }
#endif

std::string err_text(int e) {
#ifdef _WIN32
  return "winsock error " + std::to_string(e);
#else
  return std::strerror(e);
#endif
}

Status set_nonblocking(Handle h) {
#ifdef _WIN32
  u_long mode = 1;
  if (ioctlsocket(h, FIONBIO, &mode) != 0) return make_error(ErrorCode::kInternal, "ioctlsocket(FIONBIO) failed");
#else
  const int flags = fcntl(h, F_GETFL, 0);
  if (flags < 0 || fcntl(h, F_SETFL, flags | O_NONBLOCK) < 0)
    return make_error(ErrorCode::kInternal, "fcntl(O_NONBLOCK): " + err_text(errno));
#endif
  return Status::ok();
}

void set_nodelay(Handle h) {
  const int one = 1;
  (void)setsockopt(h, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&one), sizeof one);
}

void close_handle(Handle h) {
#ifdef _WIN32
  closesocket(h);
#else
  ::close(h);
#endif
}

struct AddrInfoDeleter {
  void operator()(addrinfo* p) const { freeaddrinfo(p); }
};
using AddrInfoPtr = std::unique_ptr<addrinfo, AddrInfoDeleter>;

Result<AddrInfoPtr> resolve(const std::string& host, std::uint16_t port, bool passive) {
  CLM_RETURN_IF_ERROR(ensure_winsock());
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_protocol = IPPROTO_TCP;
  hints.ai_flags = AI_NUMERICSERV | (passive ? AI_PASSIVE : 0);
  addrinfo* res = nullptr;
  const std::string service = std::to_string(port);
  const int rc = getaddrinfo(host.empty() ? nullptr : host.c_str(), service.c_str(), &hints, &res);
  if (rc != 0 || res == nullptr) {
    return make_error(ErrorCode::kUnavailable, "cannot resolve '" + host + "'");
  }
  return AddrInfoPtr(res);
}

bool sockaddr_is_loopback(const sockaddr* sa) {
  if (sa->sa_family == AF_INET) {
    const auto* in = reinterpret_cast<const sockaddr_in*>(sa);
    return (ntohl(in->sin_addr.s_addr) >> 24) == 127;
  }
  if (sa->sa_family == AF_INET6) {
    const auto* in6 = reinterpret_cast<const sockaddr_in6*>(sa);
    if (IN6_IS_ADDR_LOOPBACK(&in6->sin6_addr)) return true;
    if (IN6_IS_ADDR_V4MAPPED(&in6->sin6_addr)) return in6->sin6_addr.s6_addr[12] == 127;
  }
  return false;
}

std::string sockaddr_text(const sockaddr* sa, socklen_t len) {
  char host[NI_MAXHOST] = {0};
  char serv[NI_MAXSERV] = {0};
  if (getnameinfo(sa, len, host, sizeof host, serv, sizeof serv, NI_NUMERICHOST | NI_NUMERICSERV) != 0)
    return "unknown";
  const std::string h = host;
  return (h.find(':') != std::string::npos ? "[" + h + "]" : h) + ":" + serv;
}

}  // namespace

Deadline deadline_after(std::chrono::milliseconds timeout) {
  if (timeout.count() < 0) timeout = std::chrono::milliseconds(0);
  if (timeout > std::chrono::hours(24 * 365)) return no_deadline();
  return SteadyClock::now() + timeout;
}

void Socket::reset() {
  if (h_ != kInvalidHandle) close_handle(h_);
  h_ = kInvalidHandle;
}

void Socket::shutdown_both() const {
  if (h_ == kInvalidHandle) return;
#ifdef _WIN32
  ::shutdown(h_, SD_BOTH);
#else
  ::shutdown(h_, SHUT_RDWR);
#endif
}

WaitResult wait_ready(Handle h, bool for_write, Deadline deadline, const std::atomic<bool>* cancel) {
  for (;;) {
    if (cancel != nullptr && cancel->load(std::memory_order_acquire)) return WaitResult::kCancelled;
    const auto now = SteadyClock::now();
    if (now >= deadline) return WaitResult::kTimeout;
    const auto slice = std::min<SteadyClock::duration>(deadline - now, kPollSlice);
    // Round up so a sub-millisecond remainder does not spin.
    const auto ms = static_cast<int>(
        std::chrono::ceil<std::chrono::milliseconds>(slice).count());
#ifdef _WIN32
    WSAPOLLFD p{};
    p.fd = h;
    p.events = for_write ? POLLWRNORM : POLLRDNORM;
    const int rc = WSAPoll(&p, 1, ms);
#else
    pollfd p{};
    p.fd = h;
    p.events = for_write ? POLLOUT : POLLIN;
    const int rc = poll(&p, 1, ms);
#endif
    if (rc > 0) return WaitResult::kReady;  // POLLHUP/POLLERR also: the next I/O call reports the failure
    if (rc < 0 && !err_interrupted(last_error())) return WaitResult::kError;
  }
}

IoResult raw_recv(Handle h, void* buf, std::size_t n) {
  for (;;) {
#ifdef _WIN32
    const int rc = ::recv(h, static_cast<char*>(buf), static_cast<int>(std::min<std::size_t>(n, 1u << 30)), 0);
#else
    const auto rc = ::recv(h, buf, n, 0);
#endif
    if (rc > 0) return {IoState::kOk, static_cast<std::size_t>(rc)};
    if (rc == 0) return {IoState::kClosed, 0};
    const int e = last_error();
    if (err_interrupted(e)) continue;
    if (err_would_block(e)) return {IoState::kWouldBlock, 0};
    return {IoState::kError, 0};
  }
}

IoResult raw_send(Handle h, const void* buf, std::size_t n) {
  for (;;) {
#ifdef _WIN32
    const int rc = ::send(h, static_cast<const char*>(buf), static_cast<int>(std::min<std::size_t>(n, 1u << 30)), 0);
#else
    // MSG_NOSIGNAL: a vanished peer must surface as an error, never as SIGPIPE.
    const auto rc = ::send(h, buf, n, MSG_NOSIGNAL);
#endif
    if (rc >= 0) return {IoState::kOk, static_cast<std::size_t>(rc)};
    const int e = last_error();
    if (err_interrupted(e)) continue;
    if (err_would_block(e)) return {IoState::kWouldBlock, 0};
    return {IoState::kError, 0};
  }
}

Result<Socket> tcp_connect(const std::string& host, std::uint16_t port, bool loopback_only, Deadline deadline) {
  CLM_ASSIGN_OR_RETURN(AddrInfoPtr addrs, resolve(host, port, false));
  if (loopback_only) {
    for (const addrinfo* ai = addrs.get(); ai != nullptr; ai = ai->ai_next) {
      if (!sockaddr_is_loopback(ai->ai_addr))
        return make_error(ErrorCode::kPermissionDenied, "insecure mode refuses non-loopback address for '" + host + "'");
    }
  }
  Status last = make_error(ErrorCode::kUnavailable, "no addresses for '" + host + "'");
  for (const addrinfo* ai = addrs.get(); ai != nullptr; ai = ai->ai_next) {
    Socket s(::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol));
    if (!s.valid()) {
      last = make_error(ErrorCode::kUnavailable, "socket(): " + err_text(last_error()));
      continue;
    }
    CLM_RETURN_IF_ERROR(set_nonblocking(s.handle()));
    set_nodelay(s.handle());
    int rc = ::connect(s.handle(), ai->ai_addr, static_cast<socklen_t>(ai->ai_addrlen));
    if (rc == kSockError) {
      const int e = last_error();
      if (!err_in_progress(e) && !err_interrupted(e)) {
        last = make_error(ErrorCode::kUnavailable, "connect to " + host + ": " + err_text(e));
        continue;
      }
      const WaitResult w = wait_ready(s.handle(), true, deadline, nullptr);
      if (w == WaitResult::kTimeout) return make_error(ErrorCode::kDeadlineExceeded, "connect timed out");
      if (w != WaitResult::kReady) {
        last = make_error(ErrorCode::kUnavailable, "connect wait failed");
        continue;
      }
      int so_error = 0;
      socklen_t len = sizeof so_error;
      if (getsockopt(s.handle(), SOL_SOCKET, SO_ERROR, reinterpret_cast<char*>(&so_error), &len) != 0 ||
          so_error != 0) {
        last = make_error(ErrorCode::kUnavailable, "connect to " + host + ": " + err_text(so_error));
        continue;
      }
    }
    return s;
  }
  return last;
}

Result<ListenSocket> tcp_listen(const std::string& host, std::uint16_t port, bool loopback_only) {
  CLM_ASSIGN_OR_RETURN(AddrInfoPtr addrs, resolve(host, port, true));
  if (loopback_only) {
    for (const addrinfo* ai = addrs.get(); ai != nullptr; ai = ai->ai_next) {
      if (!sockaddr_is_loopback(ai->ai_addr))
        return make_error(ErrorCode::kPermissionDenied, "insecure mode refuses non-loopback address for '" + host + "'");
    }
  }
  Status last = make_error(ErrorCode::kUnavailable, "no addresses for '" + host + "'");
  for (const addrinfo* ai = addrs.get(); ai != nullptr; ai = ai->ai_next) {
    Socket s(::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol));
    if (!s.valid()) {
      last = make_error(ErrorCode::kUnavailable, "socket(): " + err_text(last_error()));
      continue;
    }
#ifndef _WIN32
    // POSIX: allow quick rebinding. On Windows SO_REUSEADDR would permit port hijacking, so it is not set.
    const int one = 1;
    (void)setsockopt(s.handle(), SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
#endif
    if (::bind(s.handle(), ai->ai_addr, static_cast<socklen_t>(ai->ai_addrlen)) != 0 ||
        ::listen(s.handle(), 64) != 0) {
      last = make_error(ErrorCode::kUnavailable, "bind/listen " + host + ": " + err_text(last_error()));
      continue;
    }
    CLM_RETURN_IF_ERROR(set_nonblocking(s.handle()));
    sockaddr_storage ss{};
    socklen_t len = sizeof ss;
    if (getsockname(s.handle(), reinterpret_cast<sockaddr*>(&ss), &len) != 0)
      return make_error(ErrorCode::kInternal, "getsockname failed");
    std::uint16_t bound = 0;
    if (ss.ss_family == AF_INET) bound = ntohs(reinterpret_cast<sockaddr_in*>(&ss)->sin_port);
    else if (ss.ss_family == AF_INET6) bound = ntohs(reinterpret_cast<sockaddr_in6*>(&ss)->sin6_port);
    return ListenSocket{std::move(s), bound};
  }
  return last;
}

Result<Accepted> tcp_accept(Handle listener) {
  sockaddr_storage ss{};
  socklen_t len = sizeof ss;
  const Handle h = ::accept(listener, reinterpret_cast<sockaddr*>(&ss), &len);
  if (h == kInvalidHandle) {
    const int e = last_error();
    if (err_would_block(e) || err_interrupted(e)) return make_error(ErrorCode::kUnavailable, "again");
#ifndef _WIN32
    if (e == ECONNABORTED) return make_error(ErrorCode::kUnavailable, "again");
#endif
    return make_error(ErrorCode::kUnavailable, "accept(): " + err_text(e));
  }
  Accepted a;
  a.socket = Socket(h);
  CLM_RETURN_IF_ERROR(set_nonblocking(h));
  set_nodelay(h);
  a.peer_is_loopback = sockaddr_is_loopback(reinterpret_cast<sockaddr*>(&ss));
  a.peer_address = sockaddr_text(reinterpret_cast<sockaddr*>(&ss), len);
  return a;
}

bool is_loopback_host(const std::string& host_in) {
  std::string host = host_in;
  if (host.size() >= 2 && host.front() == '[' && host.back() == ']') host = host.substr(1, host.size() - 2);
  std::string lower = host;
  std::transform(lower.begin(), lower.end(), lower.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (lower == "localhost") return true;
  if (!ensure_winsock().is_ok()) return false;
  in_addr a4{};
  if (inet_pton(AF_INET, host.c_str(), &a4) == 1) return (ntohl(a4.s_addr) >> 24) == 127;
  in6_addr a6{};
  if (inet_pton(AF_INET6, host.c_str(), &a6) == 1)
    return IN6_IS_ADDR_LOOPBACK(&a6) || (IN6_IS_ADDR_V4MAPPED(&a6) && a6.s6_addr[12] == 127);
  return false;
}

}  // namespace clusterlm::transport::net
