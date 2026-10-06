// Windows implementation of the local IPC byte stream: named pipes (ADR 0130).
//
//  * Pipe path \\.\pipe\<name>, byte mode, PIPE_REJECT_REMOTE_CLIENTS, FILE_FLAG_FIRST_PIPE_INSTANCE on the first
//    instance (a squatter that created the name earlier makes creation fail instead of being served).
//  * Explicit protected DACL (never the default): SYSTEM + the creating account (+ interactive users, read/write
//    only, for the helper pipe) and an explicit deny for network logons.
//  * Clients open with SECURITY_IDENTIFICATION so the server can identify but never impersonate them.
//  * Overlapped I/O with a per-channel cancel event: every wait honours a timeout and close() from any thread.
//  * Credentials of the peer come from the OS: GetNamedPipeClientProcessId, GetNamedPipeClientSessionId and the
//    client's token (via ImpersonateNamedPipeClient, which Windows only permits after the server has read from
//    the pipe: the client therefore sends a one-byte hello right after connecting).
#ifdef _WIN32

#include <windows.h>
#include <sddl.h>

#include <algorithm>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <string>

#include "../internal_errors.hpp"
#include "../ipc_internal.hpp"

namespace clusterlm::ipc {
namespace detail {
namespace {

using platform::detail::win_status;
using std::chrono::milliseconds;

constexpr std::uint8_t kHelloByte = 0x43;
constexpr milliseconds kHelloTimeout{2000};
constexpr DWORD kPipeBuffer = 64 * 1024;

std::wstring widen(const std::string& s) {
  if (s.empty()) return {};
  const int n = ::MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
  std::wstring w(static_cast<std::size_t>(n), L'\0');
  ::MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), w.data(), n);
  return w;
}

std::string narrow(const std::wstring& w) {
  if (w.empty()) return {};
  const int n = ::WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), nullptr, 0, nullptr, nullptr);
  std::string s(static_cast<std::size_t>(n), '\0');
  ::WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), s.data(), n, nullptr, nullptr);
  return s;
}

struct HandleCloser {
  void operator()(HANDLE h) const {
    if (h != nullptr && h != INVALID_HANDLE_VALUE) ::CloseHandle(h);
  }
};
using UniqueHandle = std::unique_ptr<void, HandleCloser>;

DWORD ms(milliseconds t) { return static_cast<DWORD>(std::clamp<long long>(t.count(), 0, 0x7fffffff)); }

Result<std::string> token_user_sid(HANDLE token) {
  DWORD need = 0;
  ::GetTokenInformation(token, TokenUser, nullptr, 0, &need);
  if (need == 0) return win_status("GetTokenInformation(size)");
  std::vector<std::uint8_t> buf(need);
  if (!::GetTokenInformation(token, TokenUser, buf.data(), need, &need)) return win_status("GetTokenInformation(TokenUser)");
  LPWSTR sid_str = nullptr;
  if (!::ConvertSidToStringSidW(reinterpret_cast<TOKEN_USER*>(buf.data())->User.Sid, &sid_str))
    return win_status("ConvertSidToStringSid");
  std::string out = narrow(sid_str);
  ::LocalFree(sid_str);
  return out;
}

Result<std::string> process_user_sid(DWORD pid) {
  UniqueHandle proc(::OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid));
  if (!proc) return win_status("OpenProcess");
  HANDLE tok = nullptr;
  if (!::OpenProcessToken(proc.get(), TOKEN_QUERY, &tok)) return win_status("OpenProcessToken");
  UniqueHandle token(tok);
  return token_user_sid(token.get());
}

// Identification-level impersonation of the pipe client just to read its token user. Always reverts.
std::string client_user_sid(HANDLE pipe) {
  if (!::ImpersonateNamedPipeClient(pipe)) return {};
  std::string sid;
  HANDLE tok = nullptr;
  if (::OpenThreadToken(::GetCurrentThread(), TOKEN_QUERY, TRUE, &tok)) {
    UniqueHandle token(tok);
    auto r = token_user_sid(token.get());
    if (r.is_ok()) sid = r.value();
  }
  ::RevertToSelf();
  return sid;
}

class WinChannel final : public RawChannel {
 public:
  WinChannel(HANDLE h, PeerCredentials peer) : h_(h), peer_(std::move(peer)) {
    read_ev_ = ::CreateEventW(nullptr, TRUE, FALSE, nullptr);
    write_ev_ = ::CreateEventW(nullptr, TRUE, FALSE, nullptr);
    cancel_ev_ = ::CreateEventW(nullptr, TRUE, FALSE, nullptr);
  }
  ~WinChannel() override {
    close();
    HandleCloser{}(read_ev_);
    HandleCloser{}(write_ev_);
    HandleCloser{}(cancel_ev_);
  }

  Result<std::size_t> read_some(std::uint8_t* buf, std::size_t n, milliseconds timeout) override {
    std::shared_lock hold(h_mu_);  // keeps close() from freeing the handle under an in-flight read
    if (closed_.load()) return make_error(ErrorCode::kCancelled, "ipc channel closed");
    OVERLAPPED ov{};
    ov.hEvent = read_ev_;
    ::ResetEvent(read_ev_);
    DWORD got = 0;
    const DWORD want = static_cast<DWORD>(std::min<std::size_t>(n, 1u << 20));
    if (!::ReadFile(h_, buf, want, &got, &ov)) {
      const DWORD err = ::GetLastError();
      if (err != ERROR_IO_PENDING) return map_io_error("ReadFile", err);
      HANDLE evs[2] = {read_ev_, cancel_ev_};
      const DWORD w = ::WaitForMultipleObjects(2, evs, FALSE, ms(timeout));
      if (w != WAIT_OBJECT_0) {
        ::CancelIoEx(h_, &ov);
        ::GetOverlappedResult(h_, &ov, &got, TRUE);  // bytes may have landed before the cancel took effect
        if (got > 0) return static_cast<std::size_t>(got);
        if (w == WAIT_OBJECT_0 + 1) return make_error(ErrorCode::kCancelled, "ipc channel closed");
        return std::size_t{0};
      }
      if (!::GetOverlappedResult(h_, &ov, &got, FALSE)) return map_io_error("ReadFile", ::GetLastError());
    }
    if (got == 0) return make_error(ErrorCode::kUnavailable, "ipc peer closed the connection");
    return static_cast<std::size_t>(got);
  }

  Status write_all(const std::uint8_t* buf, std::size_t n, milliseconds timeout) override {
    std::shared_lock hold(h_mu_);
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    std::size_t sent = 0;
    while (sent < n) {
      if (closed_.load()) return make_error(ErrorCode::kCancelled, "ipc channel closed");
      OVERLAPPED ov{};
      ov.hEvent = write_ev_;
      ::ResetEvent(write_ev_);
      DWORD put = 0;
      const DWORD want = static_cast<DWORD>(std::min<std::size_t>(n - sent, 1u << 20));
      if (!::WriteFile(h_, buf + sent, want, &put, &ov)) {
        const DWORD err = ::GetLastError();
        if (err != ERROR_IO_PENDING) return map_io_error("WriteFile", err).status();
        const auto left = std::chrono::duration_cast<milliseconds>(deadline - std::chrono::steady_clock::now());
        HANDLE evs[2] = {write_ev_, cancel_ev_};
        const DWORD w = ::WaitForMultipleObjects(2, evs, FALSE, ms(left));
        if (w != WAIT_OBJECT_0) {
          ::CancelIoEx(h_, &ov);
          ::GetOverlappedResult(h_, &ov, &put, TRUE);
          sent += put;
          if (w == WAIT_OBJECT_0 + 1) return make_error(ErrorCode::kCancelled, "ipc channel closed");
          return make_error(ErrorCode::kDeadlineExceeded, "ipc send timed out");
        }
        if (!::GetOverlappedResult(h_, &ov, &put, FALSE)) return map_io_error("WriteFile", ::GetLastError()).status();
      }
      sent += put;
    }
    return Status::ok();
  }

  void close() override {
    if (closed_.exchange(true)) return;
    ::SetEvent(cancel_ev_);  // wakes every wait; they return promptly and drop their shared lock
    ::CancelIoEx(h_, nullptr);
    std::unique_lock exclusive(h_mu_);
    HandleCloser{}(h_);  // the peer sees EOF after it has drained anything already written
    h_ = INVALID_HANDLE_VALUE;
  }
  bool is_open() const override { return !closed_.load(); }
  const PeerCredentials& peer() const override { return peer_; }

  void set_peer(PeerCredentials p) { peer_ = std::move(p); }

 private:
  static Result<std::size_t> map_io_error(const char* what, DWORD err) {
    if (err == ERROR_BROKEN_PIPE || err == ERROR_PIPE_NOT_CONNECTED || err == ERROR_NO_DATA || err == ERROR_OPERATION_ABORTED)
      return make_error(ErrorCode::kUnavailable, "ipc peer closed the connection");
    return win_status(what, err);
  }

  HANDLE h_;
  std::shared_mutex h_mu_;
  HANDLE read_ev_ = nullptr, write_ev_ = nullptr, cancel_ev_ = nullptr;
  std::atomic<bool> closed_{false};
  PeerCredentials peer_;
};

std::wstring build_sddl(PipeAccess access) {
  auto self = current_user_id();
  // Deny network logons outright, then allow SYSTEM and the creating account full control. Interactive users get
  // read/write but not FILE_CREATE_PIPE_INSTANCE (0x12019b = FILE_GENERIC_READ|FILE_GENERIC_WRITE minus 0x4), so
  // they cannot add instances to the server's pipe.
  std::wstring sddl = L"D:P(D;;GA;;;NU)(A;;GA;;;SY)";
  if (self.is_ok()) sddl += L"(A;;GA;;;" + widen(self.value()) + L")";
  if (access == PipeAccess::kOwnerSystemAndInteractiveUsers) sddl += L"(A;;0x12019b;;;IU)";
  return sddl;
}

class WinListener final : public RawListener {
 public:
  WinListener(std::wstring path, PSECURITY_DESCRIPTOR sd) : path_(std::move(path)), sd_(sd) {
    connect_ev_ = ::CreateEventW(nullptr, TRUE, FALSE, nullptr);
    cancel_ev_ = ::CreateEventW(nullptr, TRUE, FALSE, nullptr);
  }
  ~WinListener() override {
    close();
    HandleCloser{}(connect_ev_);
    HandleCloser{}(cancel_ev_);
    if (sd_ != nullptr) ::LocalFree(sd_);
  }

  Status create_instance() {
    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof sa;
    sa.lpSecurityDescriptor = sd_;
    sa.bInheritHandle = FALSE;
    DWORD open_mode = PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED;
    if (first_) open_mode |= FILE_FLAG_FIRST_PIPE_INSTANCE;
    HANDLE h = ::CreateNamedPipeW(path_.c_str(), open_mode,
                                  PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS,
                                  PIPE_UNLIMITED_INSTANCES, kPipeBuffer, kPipeBuffer, 0, &sa);
    if (h == INVALID_HANDLE_VALUE) return win_status("CreateNamedPipeW");
    first_ = false;
    pending_ = h;
    return Status::ok();
  }

  Result<std::unique_ptr<RawChannel>> accept(milliseconds timeout) override {
    std::lock_guard lock(mu_);
    if (closed_.load()) return make_error(ErrorCode::kCancelled, "ipc listener closed");
    if (pending_ == INVALID_HANDLE_VALUE) CLM_RETURN_IF_ERROR(create_instance());
    bool connected = false;
    if (!connecting_) {
      ov_ = OVERLAPPED{};
      ov_.hEvent = connect_ev_;
      ::ResetEvent(connect_ev_);
      if (::ConnectNamedPipe(pending_, &ov_)) {
        connected = true;
      } else {
        const DWORD err = ::GetLastError();
        if (err == ERROR_PIPE_CONNECTED) {
          connected = true;
        } else if (err == ERROR_IO_PENDING) {
          connecting_ = true;
        } else {
          return win_status("ConnectNamedPipe", err);
        }
      }
    }
    if (!connected) {
      HANDLE evs[2] = {connect_ev_, cancel_ev_};
      const DWORD w = ::WaitForMultipleObjects(2, evs, FALSE, ms(timeout));
      if (w == WAIT_TIMEOUT) return make_error(ErrorCode::kDeadlineExceeded, "ipc accept timed out");
      if (w != WAIT_OBJECT_0) return make_error(ErrorCode::kCancelled, "ipc listener closed");
      DWORD ignored = 0;
      connecting_ = false;
      if (!::GetOverlappedResult(pending_, &ov_, &ignored, FALSE)) {
        // The client vanished between connect and completion: recycle the instance.
        ::CloseHandle(pending_);
        pending_ = INVALID_HANDLE_VALUE;
        return make_error(ErrorCode::kDeadlineExceeded, "ipc client disconnected during accept");
      }
    }
    HANDLE client = pending_;
    pending_ = INVALID_HANDLE_VALUE;
    (void)create_instance();  // next instance right away so a second client never sees ERROR_PIPE_BUSY for long

    auto channel = std::make_unique<WinChannel>(client, PeerCredentials{});
    // Hello byte: required so ImpersonateNamedPipeClient is allowed (the server must have read once).
    std::uint8_t hello = 0;
    auto r = channel->read_some(&hello, 1, kHelloTimeout);
    if (!r.is_ok() || r.value() != 1 || hello != kHelloByte)
      return make_error(ErrorCode::kPermissionDenied, "ipc client did not complete the hello");
    PeerCredentials pc;
    ULONG pid = 0, session = 0;
    if (::GetNamedPipeClientProcessId(client, &pid)) pc.pid = pid;
    if (::GetNamedPipeClientSessionId(client, &session)) {
      pc.session_known = true;
      pc.session_id = session;
    }
    pc.user_id = client_user_sid(client);
    channel->set_peer(std::move(pc));
    return std::unique_ptr<RawChannel>(std::move(channel));
  }

  void close() override {
    if (closed_.exchange(true)) return;
    ::SetEvent(cancel_ev_);
    std::lock_guard lock(mu_);
    if (pending_ != INVALID_HANDLE_VALUE) {
      if (connecting_) ::CancelIoEx(pending_, &ov_);
      ::CloseHandle(pending_);
      pending_ = INVALID_HANDLE_VALUE;
    }
  }

 private:
  std::wstring path_;
  PSECURITY_DESCRIPTOR sd_;
  std::mutex mu_;
  HANDLE pending_ = INVALID_HANDLE_VALUE;
  HANDLE connect_ev_ = nullptr, cancel_ev_ = nullptr;
  OVERLAPPED ov_{};
  bool connecting_ = false;
  bool first_ = true;
  std::atomic<bool> closed_{false};
};

}  // namespace

Result<std::unique_ptr<RawListener>> make_listener(const ServerOptions& options) {
  CLM_ASSIGN_OR_RETURN(auto path, options.endpoint.native_path());
  PSECURITY_DESCRIPTOR sd = nullptr;
  if (!::ConvertStringSecurityDescriptorToSecurityDescriptorW(build_sddl(options.access).c_str(), SDDL_REVISION_1, &sd, nullptr))
    return win_status("ConvertStringSecurityDescriptorToSecurityDescriptor");
  auto listener = std::make_unique<WinListener>(widen(path), sd);
  CLM_RETURN_IF_ERROR(listener->create_instance());  // fails if the name is already taken (first-instance flag)
  return std::unique_ptr<RawListener>(std::move(listener));
}

Result<std::unique_ptr<RawChannel>> make_client(const Endpoint& endpoint, const ClientOptions& options,
                                                milliseconds timeout) {
  CLM_ASSIGN_OR_RETURN(auto path, endpoint.native_path());
  const std::wstring wpath = widen(path);
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  HANDLE h = INVALID_HANDLE_VALUE;
  for (;;) {
    h = ::CreateFileW(wpath.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING,
                      FILE_FLAG_OVERLAPPED | SECURITY_SQOS_PRESENT | SECURITY_IDENTIFICATION, nullptr);
    if (h != INVALID_HANDLE_VALUE) break;
    const DWORD err = ::GetLastError();
    const auto left = std::chrono::duration_cast<milliseconds>(deadline - std::chrono::steady_clock::now());
    if (err == ERROR_PIPE_BUSY && left.count() > 0) {
      ::WaitNamedPipeW(wpath.c_str(), ms(left));
      continue;
    }
    if (err == ERROR_FILE_NOT_FOUND) return make_error(ErrorCode::kUnavailable, "ipc server is not running");
    return win_status("CreateFileW(pipe)", err);
  }
  PeerCredentials server;
  ULONG spid = 0;
  if (::GetNamedPipeServerProcessId(h, &spid)) {
    server.pid = spid;
    auto sid = process_user_sid(spid);
    if (sid.is_ok()) server.user_id = sid.value();
  }
  if (!options.expected_server_user_ids.empty()) {
    const auto& ok = options.expected_server_user_ids;
    if (server.user_id.empty() || std::find(ok.begin(), ok.end(), server.user_id) == ok.end()) {
      ::CloseHandle(h);
      return make_error(ErrorCode::kUnauthenticated, "ipc server is not running as an expected account");
    }
  }
  auto channel = std::make_unique<WinChannel>(h, std::move(server));
  const std::uint8_t hello = kHelloByte;
  CLM_RETURN_IF_ERROR(channel->write_all(&hello, 1, kHelloTimeout));
  return std::unique_ptr<RawChannel>(std::move(channel));
}

}  // namespace detail

Result<std::string> current_user_id() {
  HANDLE tok = nullptr;
  if (!::OpenProcessToken(::GetCurrentProcess(), TOKEN_QUERY, &tok)) return platform::detail::win_status("OpenProcessToken");
  detail::UniqueHandle token(tok);
  return detail::token_user_sid(token.get());
}

}  // namespace clusterlm::ipc

#endif  // _WIN32
