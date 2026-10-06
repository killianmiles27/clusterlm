#pragma once
// ChildProcess: launch a worker process with piped stdin/stdout, used by ClusterLM Bench and integration tests
// to run Father/Node execution domains as separate processes on one machine.
//
// stdin is a pipe owned by the parent; closing it (or the parent dying) is the child's signal to exit. stderr
// is inherited. kill() terminates abruptly (SIGKILL / TerminateProcess) to emulate a crash or a forced worker
// deadline. In the product the Node service additionally places workers in a Job Object (ProcessJob adapter).
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "clusterlm/common/status.hpp"

namespace clusterlm::platform {

class ChildProcess {
 public:
  static Result<std::unique_ptr<ChildProcess>> spawn(const std::filesystem::path& executable,
                                                     const std::vector<std::string>& args);
  ~ChildProcess();
  ChildProcess(const ChildProcess&) = delete;
  ChildProcess& operator=(const ChildProcess&) = delete;

  // Write one line (a newline is appended) to the child's stdin.
  Status write_line(const std::string& line);
  // Read one line from the child's stdout (without the newline). kDeadlineExceeded on timeout,
  // kUnavailable at EOF.
  Result<std::string> read_line(std::chrono::milliseconds timeout);
  // Read lines until one starts with `prefix`; returns that line.
  Result<std::string> read_until(const std::string& prefix, std::chrono::milliseconds timeout);
  void close_stdin();
  // Abrupt termination (crash emulation / forced deadline).
  void kill();
  // Wait for exit; returns the exit code (negative signal number on POSIX), or kDeadlineExceeded.
  Result<int> wait(std::chrono::milliseconds timeout);
  bool running();
  std::int64_t pid() const;

  struct Impl;

 private:
  explicit ChildProcess(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

// Directory containing the running executable (to locate sibling binaries such as clusterlm-node).
std::filesystem::path executable_dir();

}  // namespace clusterlm::platform
