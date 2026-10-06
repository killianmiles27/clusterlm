#pragma once
// Default on-disk locations and owner-only file permissions.
//
// Windows layout (see docs/windows-architecture.md):
//   Node (service account)   %ProgramData%\ClusterLM\Node\{staging,identity,logs}
//   Father (per user)        %LOCALAPPDATA%\ClusterLM\{identity,models,logs}
// POSIX layout follows XDG: $XDG_DATA_HOME/clusterlm/{node,father}/..., sockets under $XDG_RUNTIME_DIR.
#include <cstddef>
#include <filesystem>
#include <string>

#include "clusterlm/common/bytes.hpp"
#include "clusterlm/common/status.hpp"

namespace clusterlm::platform {

enum class PathStyle { kWindows, kPosix };

// Raw roots as reported by the OS; separated from layout so both layouts are unit-testable on any host.
struct PathRoots {
  std::filesystem::path program_data;    // Windows: FOLDERID_ProgramData      POSIX: XDG_STATE_HOME or ~/.local/state
  std::filesystem::path local_app_data;  // Windows: FOLDERID_LocalAppData     POSIX: XDG_DATA_HOME  or ~/.local/share
  std::filesystem::path runtime_dir;     // Windows: unused                    POSIX: XDG_RUNTIME_DIR or temp dir
};

struct DefaultPaths {
  // Node: owned by the service account, owner+SYSTEM-only ACL on Windows.
  std::filesystem::path node_root;
  std::filesystem::path node_staging;   // ephemeral LeaseStore root; everything under it is deleted on release
  std::filesystem::path node_identity;  // device key + certificate
  std::filesystem::path node_logs;
  // Father: per-user.
  std::filesystem::path father_root;
  std::filesystem::path father_identity;
  std::filesystem::path father_models;  // the only permanent model store
  std::filesystem::path father_logs;
  // IPC (POSIX socket directory; Windows named pipes need no directory).
  std::filesystem::path ipc_dir;
};

// Pure layout function.
DefaultPaths default_paths_from(const PathRoots& roots, PathStyle style);
// Reads the real roots (Windows: SHGetKnownFolderPath; POSIX: XDG environment) and applies the host's layout.
Result<PathRoots> system_path_roots();
Result<DefaultPaths> default_paths();

// ---- Owner-only access ----------------------------------------------------------------------------------
// "Owner" is the account of the calling process (the service account for the Node service, the logged-on user
// for Father). Windows: a protected (non-inheriting) DACL granting full control to that account and SYSTEM only;
// for directories the entries are inherited by children. POSIX: mode 0600 (files) / 0700 (directories).
Status restrict_to_owner(const std::filesystem::path& path);
// Inspects the actual permissions (Windows: DACL contents and protection; POSIX: owner uid and mode bits).
Result<bool> is_owner_only(const std::filesystem::path& path);
// Creates the directory (and missing parents) with owner-only permissions applied to the leaf. An existing
// directory is tightened.
Status create_owner_only_directory(const std::filesystem::path& dir);
// Creates/truncates `path` with owner-only permissions from the first instant (no window in which it is
// readable by others), writes `data`, flushes it to disk. An existing file is tightened first.
Status write_owner_only_file(const std::filesystem::path& path, ByteSpan data);

}  // namespace clusterlm::platform
