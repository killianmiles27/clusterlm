#include "clusterlm/platform/mapped_file.hpp"

#include <utility>

#include "internal_errors.hpp"

#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace clusterlm::platform {

using detail::errno_status;

MappedFile::~MappedFile() { (void)close(); }

MappedFile::MappedFile(MappedFile&& o) noexcept
    : path_(std::move(o.path_)),
      size_(o.size_),
      mode_(o.mode_),
      is_open_(std::exchange(o.is_open_, false)),
      base_(std::exchange(o.base_, nullptr)),
      fd_(std::exchange(o.fd_, -1)),
      file_handle_(std::exchange(o.file_handle_, nullptr)),
      mapping_handle_(std::exchange(o.mapping_handle_, nullptr)) {
  o.size_ = 0;
}

MappedFile& MappedFile::operator=(MappedFile&& o) noexcept {
  if (this != &o) {
    (void)close();
    path_ = std::move(o.path_);
    size_ = std::exchange(o.size_, 0);
    mode_ = o.mode_;
    is_open_ = std::exchange(o.is_open_, false);
    base_ = std::exchange(o.base_, nullptr);
    fd_ = std::exchange(o.fd_, -1);
    file_handle_ = std::exchange(o.file_handle_, nullptr);
    mapping_handle_ = std::exchange(o.mapping_handle_, nullptr);
  }
  return *this;
}

std::span<const std::uint8_t> MappedFile::span() const noexcept {
  return {static_cast<const std::uint8_t*>(base_), base_ != nullptr ? static_cast<std::size_t>(size_) : 0};
}

std::span<std::uint8_t> MappedFile::writable_span() noexcept {
  if (mode_ != MapMode::kReadWrite || base_ == nullptr) return {};
  return {static_cast<std::uint8_t*>(base_), static_cast<std::size_t>(size_)};
}

#ifdef _WIN32

namespace {

using detail::win_status;

Status map_view(void* file, std::uint64_t size, MapMode mode, void*& mapping_out, void*& base_out) {
  if (size == 0) return Status::ok();  // CreateFileMapping rejects zero-length files
  const DWORD protect = mode == MapMode::kReadWrite ? PAGE_READWRITE : PAGE_READONLY;
  HANDLE mapping = ::CreateFileMappingW(static_cast<HANDLE>(file), nullptr, protect,
                                        static_cast<DWORD>(size >> 32), static_cast<DWORD>(size & 0xFFFFFFFFu),
                                        nullptr);
  if (mapping == nullptr) return win_status("CreateFileMappingW");
  const DWORD access = mode == MapMode::kReadWrite ? FILE_MAP_WRITE : FILE_MAP_READ;
  void* view = ::MapViewOfFile(mapping, access, 0, 0, static_cast<SIZE_T>(size));
  if (view == nullptr) {
    Status st = win_status("MapViewOfFile");
    ::CloseHandle(mapping);
    return st;
  }
  mapping_out = mapping;
  base_out = view;
  return Status::ok();
}

}  // namespace

Result<MappedFile> MappedFile::create(const std::filesystem::path& path, std::uint64_t size) {
  HANDLE h = ::CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, 0 /* exclusive */, nullptr, CREATE_NEW,
                           FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
  if (h == INVALID_HANDLE_VALUE) return win_status("CreateFileW(create)");
  MappedFile f;
  f.path_ = path;
  f.size_ = size;
  f.mode_ = MapMode::kReadWrite;
  f.file_handle_ = h;
  f.is_open_ = true;
  auto fail = [&](Status st) -> Result<MappedFile> {
    (void)f.close();
    std::error_code ec;
    std::filesystem::remove(path, ec);  // we created it moments ago; leave nothing behind
    return st;
  };
  if (size != 0) {
    FILE_END_OF_FILE_INFO eof{};
    eof.EndOfFile.QuadPart = static_cast<LONGLONG>(size);
    // Reserve allocation first so a full disk fails here rather than on first write through the view.
    FILE_ALLOCATION_INFO alloc{};
    alloc.AllocationSize.QuadPart = static_cast<LONGLONG>(size);
    if (!::SetFileInformationByHandle(h, FileAllocationInfo, &alloc, sizeof alloc))
      return fail(win_status("SetFileInformationByHandle(allocation)"));
    if (!::SetFileInformationByHandle(h, FileEndOfFileInfo, &eof, sizeof eof))
      return fail(win_status("SetFileInformationByHandle(eof)"));
  }
  Status st = map_view(f.file_handle_, size, f.mode_, f.mapping_handle_, f.base_);
  if (!st.is_ok()) return fail(st);
  return f;
}

Result<MappedFile> MappedFile::open(const std::filesystem::path& path, MapMode mode) {
  const DWORD access = mode == MapMode::kReadWrite ? (GENERIC_READ | GENERIC_WRITE) : GENERIC_READ;
  HANDLE h = ::CreateFileW(path.c_str(), access, mode == MapMode::kReadWrite ? 0 : FILE_SHARE_READ, nullptr,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
  if (h == INVALID_HANDLE_VALUE) return win_status("CreateFileW(open)");
  MappedFile f;
  f.path_ = path;
  f.mode_ = mode;
  f.file_handle_ = h;
  f.is_open_ = true;
  LARGE_INTEGER sz{};
  if (!::GetFileSizeEx(h, &sz)) {
    Status st = win_status("GetFileSizeEx");
    (void)f.close();
    return st;
  }
  f.size_ = static_cast<std::uint64_t>(sz.QuadPart);
  Status st = map_view(f.file_handle_, f.size_, mode, f.mapping_handle_, f.base_);
  if (!st.is_ok()) {
    (void)f.close();
    return st;
  }
  return f;
}

Status MappedFile::flush() {
  if (!is_open_ || mode_ != MapMode::kReadWrite) return Status::ok();
  if (base_ != nullptr && !::FlushViewOfFile(base_, 0)) return win_status("FlushViewOfFile");
  if (!::FlushFileBuffers(static_cast<HANDLE>(file_handle_))) return win_status("FlushFileBuffers");
  return Status::ok();
}

Status MappedFile::unmap() {
  Status first;
  if (base_ != nullptr) {
    if (!::UnmapViewOfFile(base_)) first = win_status("UnmapViewOfFile");
    else base_ = nullptr;
  }
  // The mapping object may be closed once the view is gone; both must be closed before the file is deletable.
  if (mapping_handle_ != nullptr && base_ == nullptr) {
    if (!::CloseHandle(static_cast<HANDLE>(mapping_handle_))) {
      if (first.is_ok()) first = win_status("CloseHandle(mapping)");
    } else {
      mapping_handle_ = nullptr;
    }
  }
  return first;
}

Status MappedFile::close() {
  Status first = unmap();
  if (file_handle_ != nullptr) {
    if (!::CloseHandle(static_cast<HANDLE>(file_handle_))) {
      if (first.is_ok()) first = win_status("CloseHandle(file)");
    } else {
      file_handle_ = nullptr;
    }
  }
  is_open_ = file_handle_ != nullptr || base_ != nullptr || mapping_handle_ != nullptr;
  return first;
}

#else  // POSIX

namespace {

Status map_view(int fd, std::uint64_t size, MapMode mode, void*& base_out) {
  if (size == 0) return Status::ok();  // mmap rejects zero length
  const int prot = mode == MapMode::kReadWrite ? (PROT_READ | PROT_WRITE) : PROT_READ;
  void* p = ::mmap(nullptr, static_cast<std::size_t>(size), prot, MAP_SHARED, fd, 0);
  if (p == MAP_FAILED) return errno_status("mmap");
  base_out = p;
  return Status::ok();
}

}  // namespace

Result<MappedFile> MappedFile::create(const std::filesystem::path& path, std::uint64_t size) {
  const int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
  if (fd < 0) return errno_status("open(create)");
  MappedFile f;
  f.path_ = path;
  f.size_ = size;
  f.mode_ = MapMode::kReadWrite;
  f.fd_ = fd;
  f.is_open_ = true;
  auto fail = [&](Status st) -> Result<MappedFile> {
    (void)f.close();
    ::unlink(path.c_str());  // we created it moments ago; leave nothing behind
    return st;
  };
  if (size != 0) {
    if (::ftruncate(fd, static_cast<off_t>(size)) != 0) return fail(errno_status("ftruncate"));
    // Reserve blocks so ENOSPC surfaces now, not as SIGBUS on first write through the mapping. Filesystems
    // without fallocate support (EOPNOTSUPP/EINVAL) keep the sparse file.
    const int rc = ::posix_fallocate(fd, 0, static_cast<off_t>(size));
    if (rc != 0 && rc != EOPNOTSUPP && rc != EINVAL) return fail(errno_status("posix_fallocate", rc));
  }
  Status st = map_view(fd, size, MapMode::kReadWrite, f.base_);
  if (!st.is_ok()) return fail(st);
  return f;
}

Result<MappedFile> MappedFile::open(const std::filesystem::path& path, MapMode mode) {
  const int flags = (mode == MapMode::kReadWrite ? O_RDWR : O_RDONLY) | O_CLOEXEC | O_NOFOLLOW;
  const int fd = ::open(path.c_str(), flags);
  if (fd < 0) return errno_status("open");
  MappedFile f;
  f.path_ = path;
  f.mode_ = mode;
  f.fd_ = fd;
  f.is_open_ = true;
  struct stat sb {};
  if (::fstat(fd, &sb) != 0) {
    Status st = errno_status("fstat");
    (void)f.close();
    return st;
  }
  if (!S_ISREG(sb.st_mode)) {
    (void)f.close();
    return make_error(ErrorCode::kInvalidArgument, "not a regular file: " + path.string());
  }
  f.size_ = static_cast<std::uint64_t>(sb.st_size);
  Status st = map_view(fd, f.size_, mode, f.base_);
  if (!st.is_ok()) {
    (void)f.close();
    return st;
  }
  return f;
}

Status MappedFile::flush() {
  if (!is_open_ || mode_ != MapMode::kReadWrite) return Status::ok();
  if (base_ != nullptr && ::msync(base_, static_cast<std::size_t>(size_), MS_SYNC) != 0) return errno_status("msync");
  if (fd_ >= 0 && ::fsync(fd_) != 0) return errno_status("fsync");
  return Status::ok();
}

Status MappedFile::unmap() {
  if (base_ == nullptr) return Status::ok();
  if (::munmap(base_, static_cast<std::size_t>(size_)) != 0) return errno_status("munmap");
  base_ = nullptr;
  return Status::ok();
}

Status MappedFile::close() {
  Status first = unmap();
  if (fd_ >= 0) {
    if (::close(fd_) != 0) {
      if (first.is_ok()) first = errno_status("close");
    }
    fd_ = -1;  // after close() the descriptor is invalid even on EIO; never retry
  }
  is_open_ = base_ != nullptr;
  return first;
}

#endif

}  // namespace clusterlm::platform
