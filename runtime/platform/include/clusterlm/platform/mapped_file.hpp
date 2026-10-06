#pragma once
// MappedFile: a file of a fixed size mapped into the address space.
//
// Lifetime contract (spec §10): a file's disk space is reclaimed only after the view is unmapped AND the
// mapping/file handles are closed. close() therefore does all three and reports failure, so the lease store
// can refuse to claim cleanup while a mapping is still alive. POSIX: open/ftruncate/mmap. Windows:
// CreateFileW/SetFileInformationByHandle/CreateFileMappingW/MapViewOfFile.
#include <cstdint>
#include <filesystem>
#include <span>

#include "clusterlm/common/status.hpp"

namespace clusterlm::platform {

enum class MapMode : std::uint8_t { kReadOnly, kReadWrite };

class MappedFile {
 public:
  MappedFile() = default;
  ~MappedFile();
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;
  MappedFile(MappedFile&& other) noexcept;
  MappedFile& operator=(MappedFile&& other) noexcept;

  // Exclusively creates a new file of `size` bytes (fails with kAlreadyExists if present, never follows a
  // final-component symlink) and maps it read-write. Disk space is reserved up front so a full disk surfaces
  // as kResourceExhausted here instead of a fault on first write through the mapping.
  static Result<MappedFile> create(const std::filesystem::path& path, std::uint64_t size);
  // Opens an existing file and maps it in full. A zero-length file is open but has an empty span.
  static Result<MappedFile> open(const std::filesystem::path& path, MapMode mode);

  bool is_open() const noexcept { return is_open_; }
  bool is_mapped() const noexcept { return base_ != nullptr; }
  std::uint64_t size() const noexcept { return size_; }
  MapMode mode() const noexcept { return mode_; }
  const std::filesystem::path& path() const noexcept { return path_; }

  std::span<const std::uint8_t> span() const noexcept;
  // Empty unless the file is mapped read-write.
  std::span<std::uint8_t> writable_span() noexcept;

  // Pushes dirty pages to the file (msync / FlushViewOfFile + FlushFileBuffers).
  Status flush();
  // Releases the view only; the file stays open (and size() stays valid) but span() becomes empty.
  Status unmap();
  // Unmaps and closes every handle. Idempotent. Always attempts every step; returns the first failure.
  Status close();

 private:
  std::filesystem::path path_;
  std::uint64_t size_ = 0;
  MapMode mode_ = MapMode::kReadOnly;
  bool is_open_ = false;
  void* base_ = nullptr;
  // POSIX file descriptor, or -1.
  int fd_ = -1;
  // Windows HANDLEs (void* keeps <windows.h> out of this header); null elsewhere.
  void* file_handle_ = nullptr;
  void* mapping_handle_ = nullptr;
};

}  // namespace clusterlm::platform
