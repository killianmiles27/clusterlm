#pragma once
// Shared helpers of the fuzz targets.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string_view>

#include "clusterlm/common/bytes.hpp"

// An invariant violation is a finding: print where, then abort so libFuzzer saves the reproducer.
#define CLM_FUZZ_CHECK(cond)                                                    \
  do {                                                                          \
    if (!(cond)) {                                                              \
      std::fprintf(stderr, "fuzz invariant violated: %s (%s:%d)\n", #cond, __FILE__, __LINE__); \
      std::abort();                                                             \
    }                                                                           \
  } while (0)

namespace clusterlm::fuzz {

inline ByteSpan span_of(const std::uint8_t* data, std::size_t size) { return ByteSpan(data, size); }
inline std::string_view text_of(const std::uint8_t* data, std::size_t size) {
  return std::string_view(reinterpret_cast<const char*>(data), size);
}

}  // namespace clusterlm::fuzz
