#include "clusterlm/platform/fs_safety.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <string>
#include <vector>

#include "internal_errors.hpp"

namespace clusterlm::platform {

namespace fs = std::filesystem;

namespace {

constexpr int kMaxDepth = 128;  // far beyond anything the store creates; bounds recursion on hostile trees

bool is_device_name(const std::string& component) {
  // Reserved Windows device names, matched on the stem (before the first dot), case-insensitively.
  std::string stem = component.substr(0, component.find('.'));
  while (!stem.empty() && stem.back() == ' ') stem.pop_back();
  std::string up;
  for (char c : stem) up.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
  static constexpr std::array<const char*, 6> kNames = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"};
  for (const char* n : kNames)
    if (up == n) return true;
  if (up.size() == 4 && (up.rfind("COM", 0) == 0 || up.rfind("LPT", 0) == 0) && up[3] >= '0' && up[3] <= '9')
    return true;
  return false;
}

Status validate_component(const std::string& c) {
  if (c.empty()) return make_error(ErrorCode::kInvalidArgument, "empty path component");
  if (c == "." || c == "..") return make_error(ErrorCode::kPermissionDenied, "path traversal component");
  for (char ch : c) {
    const auto u = static_cast<unsigned char>(ch);
    if (u < 0x20 || u == 0x7f || ch == '\\' || ch == ':' || ch == '*' || ch == '?' || ch == '"' || ch == '<' ||
        ch == '>' || ch == '|')
      return make_error(ErrorCode::kInvalidArgument, "illegal character in path component");
  }
  if (c.back() == '.' || c.back() == ' ')
    return make_error(ErrorCode::kInvalidArgument, "path component ends with dot or space");
  if (is_device_name(c)) return make_error(ErrorCode::kInvalidArgument, "reserved device name in path");
  return Status::ok();
}

// True if `p` has no ".." component (lexical normalisation would hide a physical-vs-lexical mismatch).
bool has_dotdot(const fs::path& p) {
  return std::any_of(p.begin(), p.end(), [](const fs::path& c) { return c == ".."; });
}

// relative path of `path` beneath `root`, empty if `path` is not strictly inside it.
fs::path strict_relative(const fs::path& root, const fs::path& path) {
  fs::path rel = path.lexically_normal().lexically_relative(root.lexically_normal());
  if (rel.empty() || rel == ".") return {};
  if (*rel.begin() == "..") return {};
  return rel;
}

// Any existing ancestor strictly between root and path (and not path itself) that is a link?
bool has_link_ancestor(const fs::path& root, const fs::path& path) {
  const fs::path rel = strict_relative(root, path);
  fs::path cur = root.lexically_normal();
  auto it = rel.begin();
  for (auto next = it; it != rel.end(); it = next) {
    ++next;
    if (next == rel.end()) break;  // `it` is the final component = path itself
    cur /= *it;
    if (is_link_or_reparse(cur)) return true;
  }
  return false;
}

void note_error(Status& first, std::size_t& count, const std::string& what) {
  ++count;
  if (count <= 3) {
    if (first.is_ok()) first = make_error(ErrorCode::kInternal, what);
    else first = make_error(ErrorCode::kInternal, first.message() + "; " + what);
  }
}

void remove_rec(const fs::path& p, int depth, Status& first, std::size_t& errors) {
  std::error_code ec;
  const fs::file_status st = fs::symlink_status(p, ec);
  if (ec) {
    if (ec == std::errc::no_such_file_or_directory) return;
    note_error(first, errors, "stat " + p.string() + ": " + ec.message());
    return;
  }
  if (st.type() == fs::file_type::not_found) return;
  const bool link = is_link_or_reparse(p);
  if (!link && st.type() == fs::file_type::directory) {
    if (depth >= kMaxDepth) {
      note_error(first, errors, "directory tree too deep: " + p.string());
      return;
    }
    std::vector<fs::path> kids;
    for (fs::directory_iterator it(p, ec), end; !ec && it != end; it.increment(ec)) kids.push_back(it->path());
    if (ec) note_error(first, errors, "list " + p.string() + ": " + ec.message());
    for (const auto& k : kids) remove_rec(k, depth + 1, first, errors);
    ec.clear();
    fs::remove(p, ec);
    if (ec && ec != std::errc::no_such_file_or_directory)
      note_error(first, errors, "rmdir " + p.string() + ": " + ec.message());
    return;
  }
  // File, symlink, reparse point, or special file: remove the entry itself. fs::remove uses unlink/
  // RemoveDirectory on the link, never the target.
  ec.clear();
  fs::remove(p, ec);
  if (ec && ec != std::errc::no_such_file_or_directory)
    note_error(first, errors, "remove " + p.string() + ": " + ec.message());
}

Status census_rec(const fs::path& p, int depth, TreeCensus& c, bool top) {
  std::error_code ec;
  const fs::file_status st = fs::symlink_status(p, ec);
  if (ec) {
    if (ec == std::errc::no_such_file_or_directory) return Status::ok();
    return make_error(ErrorCode::kInternal, "stat " + p.string() + ": " + ec.message());
  }
  if (st.type() == fs::file_type::not_found) return Status::ok();
  if (is_link_or_reparse(p)) {
    ++c.links;
    return Status::ok();
  }
  if (st.type() == fs::file_type::directory) {
    if (!top) ++c.dirs;
    if (depth >= kMaxDepth) return make_error(ErrorCode::kInternal, "directory tree too deep: " + p.string());
    Status first;
    for (fs::directory_iterator it(p, ec), end; !ec && it != end; it.increment(ec)) {
      Status s = census_rec(it->path(), depth + 1, c, false);
      if (!s.is_ok() && first.is_ok()) first = s;
    }
    if (ec) return make_error(ErrorCode::kInternal, "list " + p.string() + ": " + ec.message());
    return first;
  }
  ++c.files;
  if (st.type() == fs::file_type::regular) {
    const auto sz = fs::file_size(p, ec);
    if (ec) return make_error(ErrorCode::kInternal, "size " + p.string() + ": " + ec.message());
    c.bytes += static_cast<std::uint64_t>(sz);
  }
  return Status::ok();
}

}  // namespace

bool is_link_or_reparse(const fs::path& path) {
  std::error_code ec;
  const fs::file_status st = fs::symlink_status(path, ec);
  if (ec || st.type() == fs::file_type::not_found) return false;
  if (st.type() == fs::file_type::symlink) return true;
#ifdef _WIN32
  const DWORD attrs = ::GetFileAttributesW(path.c_str());
  if (attrs != INVALID_FILE_ATTRIBUTES && (attrs & FILE_ATTRIBUTE_REPARSE_POINT) != 0) return true;
#endif
  return false;
}

Result<fs::path> resolve_under_root(const fs::path& root, const fs::path& relative) {
  if (relative.empty()) return make_error(ErrorCode::kInvalidArgument, "empty relative path");
  if (relative.is_absolute() || relative.has_root_name() || relative.has_root_directory())
    return make_error(ErrorCode::kPermissionDenied, "absolute path rejected");
  // Inspect the raw spelling: path iteration collapses "a//b" and drops a trailing slash, and both indicate
  // a malformed name.
  const std::string raw = relative.string();
  if (raw.find("//") != std::string::npos || raw.back() == '/')
    return make_error(ErrorCode::kInvalidArgument, "empty path component");
  // Store paths are always spelled with '/'. A backslash is a separator on Windows and a filename character on
  // POSIX; rejecting it everywhere keeps one meaning on every platform.
  if (raw.find('\\') != std::string::npos)
    return make_error(ErrorCode::kInvalidArgument, "backslash in a store path");
  fs::path out = root;
  for (const auto& comp : relative) {
    CLM_RETURN_IF_ERROR(validate_component(comp.generic_string()));
    out /= comp;
  }
  if (has_link_ancestor(root, out)) return make_error(ErrorCode::kPermissionDenied, "link in path ancestry");
  return out;
}

Status remove_tree_no_follow(const fs::path& root, const fs::path& path) {
  if (path.empty() || root.empty()) return make_error(ErrorCode::kInvalidArgument, "empty path");
  if (has_dotdot(path)) return make_error(ErrorCode::kPermissionDenied, "path traversal in removal target");
  if (strict_relative(root, path).empty())
    return make_error(ErrorCode::kPermissionDenied, "removal target is not strictly inside the staging root");
  if (has_link_ancestor(root, path))
    return make_error(ErrorCode::kPermissionDenied, "link in ancestry of removal target");
  Status first;
  std::size_t errors = 0;
  remove_rec(path, 0, first, errors);
  return first;
}

Result<TreeCensus> census_under(const fs::path& path) {
  TreeCensus c;
  CLM_RETURN_IF_ERROR(census_rec(path, 0, c, true));
  return c;
}

Result<std::uint64_t> allocated_bytes_under(const fs::path& root) {
  CLM_ASSIGN_OR_RETURN(TreeCensus c, census_under(root));
  return c.bytes;
}

}  // namespace clusterlm::platform
