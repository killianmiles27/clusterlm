// POSIX implementation of the local IPC byte stream: Unix-domain sockets.
//
// Security model: the socket lives in a directory owned by the server's user with no group/other access, the
// socket file itself is chmod 0600, and every accepted connection's SO_PEERCRED uid must be the server's own
// euid (or root). The Windows named-pipe implementation is in windows/ipc_win.cpp.
#ifndef _WIN32

#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <mutex>

#include "internal_errors.hpp"
#include "ipc_internal.hpp"

namespace clusterlm::ipc {
namespace detail {
namespace {

using platform::detail::errno_status;
using std::chrono::milliseconds;

std::string uid_id(uid_t uid) { return "uid:" + std::to_string(static_cast<unsigned long>(uid)); }

// poll() one fd; returns >0 ready, 0 timeout, <0 error (errno set). Retries EINTR within the budget.
int poll_fd(int fd, short events, milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  for (;;) {
    const auto left = std::chrono::duration_cast<milliseconds>(deadline - std::chrono::steady_clock::now());
    pollfd p{fd, events, 0};
    const int rc = ::poll(&p, 1, static_cast<int>(std::max<long long>(0, left.count())));
    if (rc < 0 && errno == EINTR) continue;
    return rc;
  }
}

PeerCredentials peer_of(int fd) {
  PeerCredentials pc;
#ifdef __linux__
  ucred cred{};
  socklen_t len = sizeof cred;
  if (::getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &cred, &len) == 0) {
    pc.pid = static_cast<std::uint32_t>(cred.pid);
    pc.user_id = uid_id(cred.uid);
  }
#else
  uid_t uid = 0;
  gid_t gid = 0;
  if (::getpeereid(fd, &uid, &gid) == 0) pc.user_id = uid_id(uid);
#endif
  return pc;  // session_known stays false: POSIX has no terminal-services sessions
}

class PosixChannel final : public RawChannel {
 public:
  PosixChannel(int fd, PeerCredentials peer) : fd_(fd), peer_(std::move(peer)) {}
  ~PosixChannel() override {
    close();
    const int fd = fd_.exchange(-1);
    if (fd >= 0) ::close(fd);
  }

  Result<std::size_t> read_some(std::uint8_t* buf, std::size_t n, milliseconds timeout) override {
    const int fd = fd_.load();
    if (fd < 0 || shut_.load()) return make_error(ErrorCode::kCancelled, "ipc channel closed");
    const int rc = poll_fd(fd, POLLIN, timeout);
    if (rc < 0) return errno_status("poll");
    if (rc == 0) return std::size_t{0};
    if (shut_.load()) return make_error(ErrorCode::kCancelled, "ipc channel closed");
    const ssize_t r = ::recv(fd, buf, n, 0);
    if (r < 0) {
      if (errno == EINTR || errno == EAGAIN) return std::size_t{0};
      return errno_status("recv");
    }
    if (r == 0) return make_error(ErrorCode::kUnavailable, "ipc peer closed the connection");
    return static_cast<std::size_t>(r);
  }

  Status write_all(const std::uint8_t* buf, std::size_t n, milliseconds timeout) override {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    std::size_t sent = 0;
    while (sent < n) {
      const int fd = fd_.load();
      if (fd < 0 || shut_.load()) return make_error(ErrorCode::kCancelled, "ipc channel closed");
      const auto left = std::chrono::duration_cast<milliseconds>(deadline - std::chrono::steady_clock::now());
      if (left.count() <= 0) return make_error(ErrorCode::kDeadlineExceeded, "ipc send timed out");
      const int rc = poll_fd(fd, POLLOUT, left);
      if (rc < 0) return errno_status("poll");
      if (rc == 0) return make_error(ErrorCode::kDeadlineExceeded, "ipc send timed out");
      const ssize_t w = ::send(fd, buf + sent, n - sent, MSG_NOSIGNAL);
      if (w < 0) {
        if (errno == EINTR || errno == EAGAIN) continue;
        if (errno == EPIPE || errno == ECONNRESET) return make_error(ErrorCode::kUnavailable, "ipc peer closed");
        return errno_status("send");
      }
      sent += static_cast<std::size_t>(w);
    }
    return Status::ok();
  }

  // shutdown() wakes any thread blocked in poll() without freeing the descriptor under it; the fd itself is
  // closed in the destructor, when no other thread can be using this object.
  void close() override {
    shut_.store(true);
    const int fd = fd_.load();
    if (fd >= 0) ::shutdown(fd, SHUT_RDWR);
  }
  bool is_open() const override { return !shut_.load() && fd_.load() >= 0; }
  const PeerCredentials& peer() const override { return peer_; }

 private:
  std::atomic<int> fd_;
  std::atomic<bool> shut_{false};
  PeerCredentials peer_;
};

class PosixListener final : public RawListener {
 public:
  PosixListener(int fd, std::string path) : fd_(fd), path_(std::move(path)) {}
  ~PosixListener() override {
    close();
    const int fd = fd_.exchange(-1);
    if (fd >= 0) ::close(fd);
  }

  Result<std::unique_ptr<RawChannel>> accept(milliseconds timeout) override {
    const int lfd = fd_.load();
    if (lfd < 0 || closed_.load()) return make_error(ErrorCode::kCancelled, "ipc listener closed");
    const int rc = poll_fd(lfd, POLLIN, timeout);
    if (rc < 0) return errno_status("poll");
    if (rc == 0) return make_error(ErrorCode::kDeadlineExceeded, "ipc accept timed out");
    if (closed_.load()) return make_error(ErrorCode::kCancelled, "ipc listener closed");
    const int cfd = ::accept4(lfd, nullptr, nullptr, SOCK_CLOEXEC);
    if (cfd < 0) {
      if (errno == EINTR || errno == EAGAIN || errno == ECONNABORTED) return make_error(ErrorCode::kDeadlineExceeded, "ipc accept interrupted");
      return errno_status("accept");
    }
    PeerCredentials pc = peer_of(cfd);
    const std::string self = uid_id(::geteuid());
    if (pc.user_id != self && pc.user_id != "uid:0") {
      ::close(cfd);
      return make_error(ErrorCode::kPermissionDenied, "ipc peer is a different user");
    }
    return std::unique_ptr<RawChannel>(new PosixChannel(cfd, std::move(pc)));
  }

  void close() override {
    if (closed_.exchange(true)) return;
    const int fd = fd_.load();
    if (fd >= 0) ::shutdown(fd, SHUT_RDWR);
    ::unlink(path_.c_str());
  }

 private:
  std::atomic<int> fd_;
  std::atomic<bool> closed_{false};
  std::string path_;
};

Status fill_addr(const std::string& path, sockaddr_un& addr) {
  std::memset(&addr, 0, sizeof addr);
  addr.sun_family = AF_UNIX;
  if (path.size() >= sizeof addr.sun_path)
    return make_error(ErrorCode::kInvalidArgument, "ipc socket path too long for sockaddr_un");
  std::memcpy(addr.sun_path, path.c_str(), path.size() + 1);
  return Status::ok();
}

// The directory must be ours; it is created 0700 if absent and tightened to 0700 if it is looser.
Status prepare_dir(const std::filesystem::path& dir) {
  struct stat st {};
  if (::stat(dir.c_str(), &st) != 0) {
    if (errno != ENOENT) return errno_status("stat " + dir.string());
    if (::mkdir(dir.c_str(), 0700) != 0 && errno != EEXIST) return errno_status("mkdir " + dir.string());
    if (::stat(dir.c_str(), &st) != 0) return errno_status("stat " + dir.string());
  }
  if (!S_ISDIR(st.st_mode)) return make_error(ErrorCode::kInvalidArgument, "ipc socket_dir is not a directory");
  if (st.st_uid != ::geteuid()) return make_error(ErrorCode::kPermissionDenied, "ipc socket_dir is owned by another user");
  // The directory is dedicated to IPC sockets: a loose mode on a directory we own is tightened, not trusted.
  if ((st.st_mode & 077) != 0 && ::chmod(dir.c_str(), 0700) != 0) return errno_status("chmod " + dir.string());
  return Status::ok();
}

}  // namespace

Result<std::unique_ptr<RawListener>> make_listener(const ServerOptions& options) {
  CLM_ASSIGN_OR_RETURN(auto path, options.endpoint.native_path());
  CLM_RETURN_IF_ERROR(prepare_dir(options.endpoint.socket_dir));
  sockaddr_un addr;
  CLM_RETURN_IF_ERROR(fill_addr(path, addr));
  // Remove a stale socket from a previous run, but never anything that is not a socket we own.
  struct stat st {};
  if (::lstat(path.c_str(), &st) == 0) {
    if (!S_ISSOCK(st.st_mode) || st.st_uid != ::geteuid())
      return make_error(ErrorCode::kAlreadyExists, "ipc endpoint path exists and is not our socket");
    ::unlink(path.c_str());
  }
  const int fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (fd < 0) return errno_status("socket");
  if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof addr) != 0) {
    const Status s = errno_status("bind " + path);
    ::close(fd);
    return s;
  }
  (void)::chmod(path.c_str(), 0600);
  if (::listen(fd, 8) != 0) {
    const Status s = errno_status("listen");
    ::close(fd);
    ::unlink(path.c_str());
    return s;
  }
  return std::unique_ptr<RawListener>(new PosixListener(fd, path));
}

Result<std::unique_ptr<RawChannel>> make_client(const Endpoint& endpoint, const ClientOptions& options,
                                                milliseconds timeout) {
  CLM_ASSIGN_OR_RETURN(auto path, endpoint.native_path());
  sockaddr_un addr;
  CLM_RETURN_IF_ERROR(fill_addr(path, addr));
  const int fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (fd < 0) return errno_status("socket");
  // Unix-domain connect() completes (or fails) immediately; `timeout` bounds retries of a full backlog.
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  for (;;) {
    if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof addr) == 0) break;
    if ((errno == EAGAIN || errno == EINTR) && std::chrono::steady_clock::now() < deadline) {
      ::usleep(5000);
      continue;
    }
    const int e = errno;
    ::close(fd);
    if (e == ENOENT || e == ECONNREFUSED) return make_error(ErrorCode::kUnavailable, "ipc server is not running");
    return errno_status("connect", e);
  }
  PeerCredentials pc = peer_of(fd);  // the server's credentials
  if (!options.expected_server_user_ids.empty()) {
    bool ok = false;
    for (const auto& u : options.expected_server_user_ids) ok = ok || u == pc.user_id;
    if (!ok) {
      ::close(fd);
      return make_error(ErrorCode::kUnauthenticated, "ipc server is not running as an expected user");
    }
  }
  return std::unique_ptr<RawChannel>(new PosixChannel(fd, std::move(pc)));
}

}  // namespace detail

Result<std::string> current_user_id() { return detail::uid_id(::geteuid()); }

}  // namespace clusterlm::ipc

#endif  // !_WIN32
