#include "clusterlm/platform/durable_file.hpp"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>

#include "internal_errors.hpp"

#ifndef _WIN32
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace clusterlm::platform {

using detail::errno_status;

DurableAppendFile::~DurableAppendFile() { (void)close(); }

DurableAppendFile::DurableAppendFile(DurableAppendFile&& o) noexcept
    : is_open_(std::exchange(o.is_open_, false)),
      size_(std::exchange(o.size_, 0)),
      fd_(std::exchange(o.fd_, -1)),
      handle_(std::exchange(o.handle_, nullptr)) {}

DurableAppendFile& DurableAppendFile::operator=(DurableAppendFile&& o) noexcept {
  if (this != &o) {
    (void)close();
    is_open_ = std::exchange(o.is_open_, false);
    size_ = std::exchange(o.size_, 0);
    fd_ = std::exchange(o.fd_, -1);
    handle_ = std::exchange(o.handle_, nullptr);
  }
  return *this;
}

#ifdef _WIN32

using detail::win_status;

Result<DurableAppendFile> DurableAppendFile::open(const std::filesystem::path& path) {
  const bool existed = std::filesystem::exists(path);
  // FILE_APPEND_DATA without FILE_WRITE_DATA makes every WriteFile an atomic append at end-of-file.
  HANDLE h = ::CreateFileW(path.c_str(), FILE_APPEND_DATA | FILE_READ_ATTRIBUTES, FILE_SHARE_READ, nullptr,
                           OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
  if (h == INVALID_HANDLE_VALUE) return win_status("CreateFileW(journal)");
  DurableAppendFile f;
  f.handle_ = h;
  f.is_open_ = true;
  LARGE_INTEGER sz{};
  if (::GetFileSizeEx(h, &sz)) f.size_ = static_cast<std::uint64_t>(sz.QuadPart);
  if (!existed) {
    Status st = f.sync();
    if (!st.is_ok()) return st;
  }
  return f;
}

Status DurableAppendFile::append(ByteSpan data, bool do_sync) {
  if (!is_open_) return make_error(ErrorCode::kFailedPrecondition, "append on closed file");
  std::size_t done = 0;
  while (done < data.size()) {
    const DWORD chunk = static_cast<DWORD>(std::min<std::size_t>(data.size() - done, 1u << 30));
    DWORD written = 0;
    if (!::WriteFile(static_cast<HANDLE>(handle_), data.data() + done, chunk, &written, nullptr))
      return win_status("WriteFile");
    done += written;
    size_ += written;
  }
  return do_sync ? sync() : Status::ok();
}

Status DurableAppendFile::sync() {
  if (!is_open_) return make_error(ErrorCode::kFailedPrecondition, "sync on closed file");
  if (!::FlushFileBuffers(static_cast<HANDLE>(handle_))) return win_status("FlushFileBuffers");
  return Status::ok();
}

Status DurableAppendFile::close() {
  if (!is_open_) return Status::ok();
  Status st;
  if (!::CloseHandle(static_cast<HANDLE>(handle_))) st = win_status("CloseHandle");
  handle_ = nullptr;
  is_open_ = false;
  return st;
}

Status sync_directory(const std::filesystem::path&) { return Status::ok(); }

#else  // POSIX

Result<DurableAppendFile> DurableAppendFile::open(const std::filesystem::path& path) {
  std::error_code ec;
  const bool existed = std::filesystem::exists(path, ec);
  const int fd = ::open(path.c_str(), O_WRONLY | O_APPEND | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
  if (fd < 0) return errno_status("open(journal)");
  DurableAppendFile f;
  f.fd_ = fd;
  f.is_open_ = true;
  struct stat sb {};
  if (::fstat(fd, &sb) == 0) f.size_ = static_cast<std::uint64_t>(sb.st_size);
  if (!existed) {
    Status st = f.sync();
    if (st.is_ok()) st = sync_directory(path.parent_path());
    if (!st.is_ok()) return st;
  }
  return f;
}

Status DurableAppendFile::append(ByteSpan data, bool do_sync) {
  if (!is_open_) return make_error(ErrorCode::kFailedPrecondition, "append on closed file");
  std::size_t done = 0;
  while (done < data.size()) {
    const ssize_t n = ::write(fd_, data.data() + done, data.size() - done);
    if (n < 0) {
      if (errno == EINTR) continue;
      return errno_status("write");
    }
    done += static_cast<std::size_t>(n);
    size_ += static_cast<std::uint64_t>(n);
  }
  return do_sync ? sync() : Status::ok();
}

Status DurableAppendFile::sync() {
  if (!is_open_) return make_error(ErrorCode::kFailedPrecondition, "sync on closed file");
  if (::fsync(fd_) != 0) return errno_status("fsync");
  return Status::ok();
}

Status DurableAppendFile::close() {
  if (!is_open_) return Status::ok();
  Status st;
  if (::close(fd_) != 0) st = errno_status("close");
  fd_ = -1;
  is_open_ = false;
  return st;
}

Status sync_directory(const std::filesystem::path& dir) {
  const int fd = ::open(dir.empty() ? "." : dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (fd < 0) return errno_status("open(dir)");
  Status st;
  if (::fsync(fd) != 0) st = errno_status("fsync(dir)");
  ::close(fd);
  return st;
}

#endif

Result<Bytes> read_file_bytes(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) return make_error(ErrorCode::kNotFound, "no such file: " + path.string());
    return make_error(ErrorCode::kInternal, "cannot open for read: " + path.string());
  }
  Bytes out((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  if (in.bad()) return make_error(ErrorCode::kInternal, "read failed: " + path.string());
  return out;
}

Status write_file_atomic(const std::filesystem::path& target, ByteSpan content) {
  std::filesystem::path tmp = target;
  tmp += ".tmp";
  std::error_code ec;
  std::filesystem::remove(tmp, ec);  // stale temp from a previous crash
  {
    auto file = DurableAppendFile::open(tmp);
    if (!file.is_ok()) return file.status();
    Status st = file->append(content, true);
    Status closed = file->close();
    if (!st.is_ok()) return st;
    if (!closed.is_ok()) return closed;
  }
  std::filesystem::rename(tmp, target, ec);  // replaces target atomically (rename(2) / MoveFileEx)
  if (ec) return make_error(ErrorCode::kInternal, "rename failed: " + ec.message());
  return sync_directory(target.parent_path());
}

}  // namespace clusterlm::platform
