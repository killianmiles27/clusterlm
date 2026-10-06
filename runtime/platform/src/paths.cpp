// Default paths (portable layout) and the POSIX implementation of the owner-only helpers. The Windows
// implementation of the OS-facing functions is in windows/paths_win.cpp.
#include "clusterlm/platform/paths.hpp"

#include <cstdlib>

#include "internal_errors.hpp"

#ifndef _WIN32
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace clusterlm::platform {

namespace fs = std::filesystem;

DefaultPaths default_paths_from(const PathRoots& roots, PathStyle style) {
  DefaultPaths p;
  if (style == PathStyle::kWindows) {
    p.node_root = roots.program_data / "ClusterLM" / "Node";
    p.father_root = roots.local_app_data / "ClusterLM";
    p.father_models = p.father_root / "models";
    p.ipc_dir = roots.runtime_dir;  // unused: named pipes
  } else {
    p.node_root = roots.program_data / "clusterlm" / "node";
    p.father_root = roots.local_app_data / "clusterlm" / "father";
    p.father_models = p.father_root / "models";
    p.ipc_dir = roots.runtime_dir / "clusterlm";
  }
  p.node_staging = p.node_root / "staging";
  p.node_identity = p.node_root / "identity";
  p.node_logs = p.node_root / "logs";
  p.father_identity = p.father_root / "identity";
  p.father_logs = p.father_root / "logs";
  return p;
}

Result<DefaultPaths> default_paths() {
  CLM_ASSIGN_OR_RETURN(auto roots, system_path_roots());
#ifdef _WIN32
  return default_paths_from(roots, PathStyle::kWindows);
#else
  return default_paths_from(roots, PathStyle::kPosix);
#endif
}

#ifndef _WIN32

using detail::errno_status;

Result<PathRoots> system_path_roots() {
  auto env = [](const char* name) -> std::string {
    const char* v = std::getenv(name);
    return v != nullptr ? std::string(v) : std::string();
  };
  const std::string home = env("HOME");
  auto xdg = [&](const char* var, const char* fallback) -> Result<fs::path> {
    const std::string v = env(var);
    if (!v.empty() && fs::path(v).is_absolute()) return fs::path(v);  // XDG: relative values must be ignored
    if (home.empty()) return make_error(ErrorCode::kFailedPrecondition, std::string("neither ") + var + " nor HOME is set");
    return fs::path(home) / fallback;
  };
  PathRoots r;
  CLM_ASSIGN_OR_RETURN(r.program_data, xdg("XDG_STATE_HOME", ".local/state"));
  CLM_ASSIGN_OR_RETURN(r.local_app_data, xdg("XDG_DATA_HOME", ".local/share"));
  const std::string rt = env("XDG_RUNTIME_DIR");
  r.runtime_dir = (!rt.empty() && fs::path(rt).is_absolute()) ? fs::path(rt) : fs::temp_directory_path();
  return r;
}

Status restrict_to_owner(const fs::path& path) {
  struct stat st {};
  if (::stat(path.c_str(), &st) != 0) return errno_status("stat " + path.string());
  const mode_t mode = S_ISDIR(st.st_mode) ? 0700 : 0600;
  if (::chmod(path.c_str(), mode) != 0) return errno_status("chmod " + path.string());
  return Status::ok();
}

Result<bool> is_owner_only(const fs::path& path) {
  struct stat st {};
  if (::stat(path.c_str(), &st) != 0) return errno_status("stat " + path.string());
  return st.st_uid == ::geteuid() && (st.st_mode & 077) == 0;
}

Status create_owner_only_directory(const fs::path& dir) {
  std::error_code ec;
  fs::create_directories(dir, ec);
  if (ec) return make_error(ErrorCode::kUnavailable, "cannot create " + dir.string() + ": " + ec.message());
  return restrict_to_owner(dir);
}

Status write_owner_only_file(const fs::path& path, ByteSpan data) {
  // Created 0600 from the first instant; fchmod also tightens a pre-existing file before any byte is written.
  const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
  if (fd < 0) return errno_status("open " + path.string());
  (void)::fchmod(fd, 0600);
  std::size_t done = 0;
  while (done < data.size()) {
    const ssize_t w = ::write(fd, data.data() + done, data.size() - done);
    if (w < 0) {
      if (errno == EINTR) continue;
      const Status s = errno_status("write " + path.string());
      ::close(fd);
      return s;
    }
    done += static_cast<std::size_t>(w);
  }
  const int sync_rc = ::fsync(fd);
  const int close_rc = ::close(fd);
  if (sync_rc != 0 || close_rc != 0) return errno_status("flush " + path.string());
  return Status::ok();
}

#endif  // !_WIN32

}  // namespace clusterlm::platform
