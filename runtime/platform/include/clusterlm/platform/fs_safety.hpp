#pragma once
// Filesystem safety for the app-owned staging root (spec §10-§11).
//
// Deletion never follows symlinks or reparse points and never acts on a path derived from a remote/manifest
// string; every path it touches is validated to lie strictly under the root.
#include <cstdint>
#include <filesystem>
#include <string_view>

#include "clusterlm/common/status.hpp"

namespace clusterlm::platform {

// Joins `relative` under `root`. Rejects (kInvalidArgument / kPermissionDenied): empty or absolute paths,
// root names/directories, "." / ".." / empty components, components containing separator lookalikes
// (backslash, colon), control characters, trailing dot/space, and Windows device names (CON, NUL, COM1, ...,
// with or without extension). The device-name and character rules apply on every OS so behaviour is identical
// on Linux CI and Windows. Also rejects if any EXISTING ancestor below root is a link/reparse point.
Result<std::filesystem::path> resolve_under_root(const std::filesystem::path& root,
                                                 const std::filesystem::path& relative);

// True for a symlink (any OS) or, on Windows, anything with FILE_ATTRIBUTE_REPARSE_POINT (junctions, etc.).
// False if the path does not exist.
bool is_link_or_reparse(const std::filesystem::path& path);

// Removes `path` (file, link, or directory tree). `path` must lie strictly beneath `root`, contain no ".."
// component and have no link ancestors below root. A link met during the walk is removed AS A LINK and never
// traversed. A missing path is success. Best effort: continues after an error, returns the first few.
Status remove_tree_no_follow(const std::filesystem::path& root, const std::filesystem::path& path);

// Census of what is physically left under a path. Does not follow links. Links are counted, not sized.
struct TreeCensus {
  std::uint64_t bytes = 0;  // sum of regular-file sizes (partial files included)
  std::uint64_t files = 0;
  std::uint64_t links = 0;
  std::uint64_t dirs = 0;  // directories below the starting path (excluding it)
  bool empty() const noexcept { return files == 0 && links == 0 && dirs == 0; }
};
// `path` may be a directory (walked), a file or a link (counted as itself). Missing path = empty census.
Result<TreeCensus> census_under(const std::filesystem::path& path);
Result<std::uint64_t> allocated_bytes_under(const std::filesystem::path& root);

}  // namespace clusterlm::platform
