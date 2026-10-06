#pragma once
// Shared helpers for lease-store and platform tests.
#include <atomic>
#include <chrono>
#include <filesystem>
#include <string>

#include "clusterlm/common/bytes.hpp"
#include "clusterlm/common/digest.hpp"

namespace clusterlm::testing {

// A unique scratch directory removed on destruction (tests only; product code never uses
// std::filesystem::remove_all on staging).
class TempDir {
 public:
  TempDir() {
    static std::atomic<unsigned> counter{0};
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = std::filesystem::temp_directory_path() /
            ("clusterlm-test-" + std::to_string(stamp) + "-" + std::to_string(counter++));
    std::filesystem::create_directories(path_);
  }
  ~TempDir() {
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
  }
  TempDir(const TempDir&) = delete;
  TempDir& operator=(const TempDir&) = delete;
  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

inline Bytes pattern_bytes(std::size_t n, std::uint8_t seed = 1) {
  Bytes b(n);
  std::uint32_t x = seed;
  for (auto& v : b) {
    x = x * 1664525u + 1013904223u;
    v = static_cast<std::uint8_t>(x >> 24);
  }
  return b;
}

}  // namespace clusterlm::testing
