#include "clusterlm/platform/process.hpp"

#include <thread>

#include "clusterlm/common/clock.hpp"

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
extern char** environ;
#endif

namespace clusterlm::platform {

using namespace std::chrono_literals;

#ifdef _WIN32

namespace {
std::wstring widen(const std::string& s) {
  if (s.empty()) return {};
  int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
  std::wstring w(static_cast<std::size_t>(n), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), w.data(), n);
  return w;
}
// Quote one argument per the CommandLineToArgvW rules.
std::wstring quote(const std::wstring& arg) {
  if (!arg.empty() && arg.find_first_of(L" \t\"") == std::wstring::npos) return arg;
  std::wstring out = L"\"";
  std::size_t backslashes = 0;
  for (wchar_t c : arg) {
    if (c == L'\\') {
      ++backslashes;
    } else if (c == L'"') {
      out.append(backslashes * 2 + 1, L'\\');
      out.push_back(c);
      backslashes = 0;
    } else {
      out.append(backslashes, L'\\');
      out.push_back(c);
      backslashes = 0;
    }
  }
  out.append(backslashes * 2, L'\\');
  out.push_back(L'"');
  return out;
}
}  // namespace

struct ChildProcess::Impl {
  PROCESS_INFORMATION pi{};
  HANDLE stdin_write = nullptr;
  HANDLE stdout_read = nullptr;
  std::string buffer;
  bool eof = false;
  ~Impl() {
    if (stdin_write) CloseHandle(stdin_write);
    if (stdout_read) CloseHandle(stdout_read);
    if (pi.hProcess) CloseHandle(pi.hProcess);
    if (pi.hThread) CloseHandle(pi.hThread);
  }
};

Result<std::unique_ptr<ChildProcess>> ChildProcess::spawn(const std::filesystem::path& executable,
                                                          const std::vector<std::string>& args) {
  auto impl = std::make_unique<Impl>();
  SECURITY_ATTRIBUTES sa{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
  HANDLE in_read = nullptr, out_write = nullptr;
  if (!CreatePipe(&in_read, &impl->stdin_write, &sa, 0) || !CreatePipe(&impl->stdout_read, &out_write, &sa, 0))
    return make_error(ErrorCode::kInternal, "CreatePipe failed");
  SetHandleInformation(impl->stdin_write, HANDLE_FLAG_INHERIT, 0);
  SetHandleInformation(impl->stdout_read, HANDLE_FLAG_INHERIT, 0);
  std::wstring cmd = quote(executable.wstring());
  for (const auto& a : args) cmd += L" " + quote(widen(a));
  STARTUPINFOW si{};
  si.cb = sizeof si;
  si.dwFlags = STARTF_USESTDHANDLES;
  si.hStdInput = in_read;
  si.hStdOutput = out_write;
  si.hStdError = GetStdHandle(STD_ERROR_HANDLE);
  BOOL ok = CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr, nullptr, &si,
                           &impl->pi);
  CloseHandle(in_read);
  CloseHandle(out_write);
  if (!ok) return make_error(ErrorCode::kNotFound, "CreateProcessW failed for " + executable.string());
  return std::unique_ptr<ChildProcess>(new ChildProcess(std::move(impl)));
}

Status ChildProcess::write_line(const std::string& line) {
  std::string data = line + "\n";
  DWORD written = 0;
  if (!impl_->stdin_write || !WriteFile(impl_->stdin_write, data.data(), static_cast<DWORD>(data.size()), &written, nullptr))
    return make_error(ErrorCode::kUnavailable, "child stdin closed");
  return Status::ok();
}

Result<std::string> ChildProcess::read_line(std::chrono::milliseconds timeout) {
  Stopwatch sw;
  while (true) {
    if (auto nl = impl_->buffer.find('\n'); nl != std::string::npos) {
      std::string line = impl_->buffer.substr(0, nl);
      impl_->buffer.erase(0, nl + 1);
      if (!line.empty() && line.back() == '\r') line.pop_back();
      return line;
    }
    if (impl_->eof) return make_error(ErrorCode::kUnavailable, "child stdout closed");
    DWORD avail = 0;
    if (!PeekNamedPipe(impl_->stdout_read, nullptr, 0, nullptr, &avail, nullptr)) {
      impl_->eof = true;
      continue;
    }
    if (avail == 0) {
      if (sw.elapsed_ms() >= static_cast<double>(timeout.count()))
        return make_error(ErrorCode::kDeadlineExceeded, "timed out reading child stdout");
      std::this_thread::sleep_for(5ms);
      continue;
    }
    char buf[4096];
    DWORD got = 0;
    if (!ReadFile(impl_->stdout_read, buf, std::min<DWORD>(avail, sizeof buf), &got, nullptr) || got == 0) {
      impl_->eof = true;
      continue;
    }
    impl_->buffer.append(buf, got);
  }
}

void ChildProcess::close_stdin() {
  if (impl_->stdin_write) {
    CloseHandle(impl_->stdin_write);
    impl_->stdin_write = nullptr;
  }
}

void ChildProcess::kill() { TerminateProcess(impl_->pi.hProcess, 137); }

Result<int> ChildProcess::wait(std::chrono::milliseconds timeout) {
  if (WaitForSingleObject(impl_->pi.hProcess, static_cast<DWORD>(timeout.count())) != WAIT_OBJECT_0)
    return make_error(ErrorCode::kDeadlineExceeded, "child still running");
  DWORD code = 0;
  GetExitCodeProcess(impl_->pi.hProcess, &code);
  return static_cast<int>(code);
}

bool ChildProcess::running() { return WaitForSingleObject(impl_->pi.hProcess, 0) == WAIT_TIMEOUT; }
std::int64_t ChildProcess::pid() const { return static_cast<std::int64_t>(impl_->pi.dwProcessId); }

std::filesystem::path executable_dir() {
  std::wstring buf(32768, L'\0');
  DWORD n = GetModuleFileNameW(nullptr, buf.data(), static_cast<DWORD>(buf.size()));
  buf.resize(n);
  return std::filesystem::path(buf).parent_path();
}

#else  // POSIX

struct ChildProcess::Impl {
  pid_t pid = -1;
  int stdin_fd = -1;
  int stdout_fd = -1;
  std::string buffer;
  bool eof = false;
  std::optional<int> exit_code;
  ~Impl() {
    if (stdin_fd >= 0) ::close(stdin_fd);
    if (stdout_fd >= 0) ::close(stdout_fd);
  }
};

Result<std::unique_ptr<ChildProcess>> ChildProcess::spawn(const std::filesystem::path& executable,
                                                          const std::vector<std::string>& args) {
  int in_pipe[2], out_pipe[2];
  if (::pipe(in_pipe) != 0) return make_error(ErrorCode::kInternal, "pipe failed");
  if (::pipe(out_pipe) != 0) {
    ::close(in_pipe[0]);
    ::close(in_pipe[1]);
    return make_error(ErrorCode::kInternal, "pipe failed");
  }
  // Parent ends must not leak into other children.
  ::fcntl(in_pipe[1], F_SETFD, FD_CLOEXEC);
  ::fcntl(out_pipe[0], F_SETFD, FD_CLOEXEC);
  posix_spawn_file_actions_t actions;
  posix_spawn_file_actions_init(&actions);
  posix_spawn_file_actions_adddup2(&actions, in_pipe[0], STDIN_FILENO);
  posix_spawn_file_actions_adddup2(&actions, out_pipe[1], STDOUT_FILENO);
  posix_spawn_file_actions_addclose(&actions, in_pipe[0]);
  posix_spawn_file_actions_addclose(&actions, out_pipe[1]);
  std::vector<std::string> storage;
  storage.push_back(executable.string());
  storage.insert(storage.end(), args.begin(), args.end());
  std::vector<char*> argv;
  for (auto& s : storage) argv.push_back(s.data());
  argv.push_back(nullptr);
  auto impl = std::make_unique<Impl>();
  int rc = posix_spawn(&impl->pid, storage[0].c_str(), &actions, nullptr, argv.data(), environ);
  posix_spawn_file_actions_destroy(&actions);
  ::close(in_pipe[0]);
  ::close(out_pipe[1]);
  impl->stdin_fd = in_pipe[1];
  impl->stdout_fd = out_pipe[0];
  if (rc != 0) return make_error(ErrorCode::kNotFound, "posix_spawn " + storage[0] + ": " + std::strerror(rc));
  // Writing to a child that already exited must not kill the harness.
  ::signal(SIGPIPE, SIG_IGN);
  return std::unique_ptr<ChildProcess>(new ChildProcess(std::move(impl)));
}

Status ChildProcess::write_line(const std::string& line) {
  std::string data = line + "\n";
  std::size_t off = 0;
  while (off < data.size()) {
    if (impl_->stdin_fd < 0) return make_error(ErrorCode::kUnavailable, "child stdin closed");
    ssize_t n = ::write(impl_->stdin_fd, data.data() + off, data.size() - off);
    if (n < 0) {
      if (errno == EINTR) continue;
      return make_error(ErrorCode::kUnavailable, "child stdin closed");
    }
    off += static_cast<std::size_t>(n);
  }
  return Status::ok();
}

Result<std::string> ChildProcess::read_line(std::chrono::milliseconds timeout) {
  const auto deadline = SteadyClock::now() + timeout;
  while (true) {
    if (auto nl = impl_->buffer.find('\n'); nl != std::string::npos) {
      std::string line = impl_->buffer.substr(0, nl);
      impl_->buffer.erase(0, nl + 1);
      return line;
    }
    if (impl_->eof) return make_error(ErrorCode::kUnavailable, "child stdout closed");
    const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - SteadyClock::now()).count();
    if (left <= 0) return make_error(ErrorCode::kDeadlineExceeded, "timed out reading child stdout");
    pollfd p{impl_->stdout_fd, POLLIN, 0};
    int r = ::poll(&p, 1, static_cast<int>(left));
    if (r < 0 && errno == EINTR) continue;
    if (r <= 0) continue;
    char buf[4096];
    ssize_t n = ::read(impl_->stdout_fd, buf, sizeof buf);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) {
      impl_->eof = true;
      continue;
    }
    impl_->buffer.append(buf, static_cast<std::size_t>(n));
  }
}

void ChildProcess::close_stdin() {
  if (impl_->stdin_fd >= 0) {
    ::close(impl_->stdin_fd);
    impl_->stdin_fd = -1;
  }
}

void ChildProcess::kill() {
  if (impl_->pid > 0 && !impl_->exit_code) ::kill(impl_->pid, SIGKILL);
}

Result<int> ChildProcess::wait(std::chrono::milliseconds timeout) {
  if (impl_->exit_code) return *impl_->exit_code;
  const auto deadline = SteadyClock::now() + timeout;
  while (true) {
    int status = 0;
    pid_t r = ::waitpid(impl_->pid, &status, WNOHANG);
    if (r == impl_->pid) {
      impl_->exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -WTERMSIG(status);
      return *impl_->exit_code;
    }
    if (r < 0 && errno != EINTR) return make_error(ErrorCode::kInternal, "waitpid failed");
    if (SteadyClock::now() >= deadline) return make_error(ErrorCode::kDeadlineExceeded, "child still running");
    std::this_thread::sleep_for(5ms);
  }
}

bool ChildProcess::running() { return !wait(0ms).is_ok(); }
std::int64_t ChildProcess::pid() const { return impl_->pid; }

std::filesystem::path executable_dir() {
  std::error_code ec;
  auto p = std::filesystem::read_symlink("/proc/self/exe", ec);
  return ec ? std::filesystem::current_path() : p.parent_path();
}

#endif

ChildProcess::ChildProcess(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

ChildProcess::~ChildProcess() {
  if (!impl_) return;
  close_stdin();
  if (!wait(2000ms).is_ok()) {
    kill();
    (void)wait(2000ms);
  }
}

Result<std::string> ChildProcess::read_until(const std::string& prefix, std::chrono::milliseconds timeout) {
  const auto deadline = SteadyClock::now() + timeout;
  while (true) {
    const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - SteadyClock::now());
    if (left.count() <= 0) return make_error(ErrorCode::kDeadlineExceeded, "timed out waiting for " + prefix);
    CLM_ASSIGN_OR_RETURN(std::string line, read_line(left));
    if (line.rfind(prefix, 0) == 0) return line;
  }
}

}  // namespace clusterlm::platform
