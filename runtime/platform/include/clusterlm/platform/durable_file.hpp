#pragma once
// Durable file primitives for the content-free lease journal.
#include <cstdint>
#include <filesystem>
#include <string_view>

#include "clusterlm/common/bytes.hpp"
#include "clusterlm/common/status.hpp"

namespace clusterlm::platform {

// Append-only file whose appends reach stable storage (fsync / FlushFileBuffers) before append() returns.
class DurableAppendFile {
 public:
  DurableAppendFile() = default;
  ~DurableAppendFile();
  DurableAppendFile(const DurableAppendFile&) = delete;
  DurableAppendFile& operator=(const DurableAppendFile&) = delete;
  DurableAppendFile(DurableAppendFile&& other) noexcept;
  DurableAppendFile& operator=(DurableAppendFile&& other) noexcept;

  // Opens (creating if absent) for append. A new file's directory entry is also made durable.
  static Result<DurableAppendFile> open(const std::filesystem::path& path);

  bool is_open() const noexcept { return is_open_; }
  std::uint64_t size() const noexcept { return size_; }

  // Appends all bytes then, if `sync`, flushes them to stable storage.
  Status append(ByteSpan data, bool sync = true);
  Status append(std::string_view text, bool sync = true) {
    return append(ByteSpan(reinterpret_cast<const std::uint8_t*>(text.data()), text.size()), sync);
  }
  Status sync();
  Status close();

 private:
  bool is_open_ = false;
  std::uint64_t size_ = 0;
  int fd_ = -1;
  void* handle_ = nullptr;  // Windows HANDLE
};

// Makes directory-entry changes (create/rename) durable. No-op on Windows, where NTFS journals metadata.
Status sync_directory(const std::filesystem::path& dir);

// Reads a whole file. kNotFound if absent.
Result<Bytes> read_file_bytes(const std::filesystem::path& path);

// Replaces `target` with `content` crash-atomically: write `target`.tmp, flush, rename over target.
// Readers see either the old or the new content, never a mixture.
Status write_file_atomic(const std::filesystem::path& target, ByteSpan content);

}  // namespace clusterlm::platform
