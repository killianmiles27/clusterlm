#include <algorithm>

#include "unicode_tables.hpp"

namespace clusterlm::father::unicode {

Class class_of(std::uint32_t cp) noexcept {
  if (cp >= 0x110000u) return Class::kOther;
  const std::uint32_t* first = kClassRanges;
  const std::uint32_t* last = kClassRanges + kClassRangeCount;
  // Last entry whose first code point is <= cp.
  const std::uint32_t* it = std::upper_bound(first, last, (cp << 2) | 3u);
  return static_cast<Class>(*(it - 1) & 3u);
}

bool is_white_space(std::uint32_t cp) noexcept {
  if (cp <= 0x20) return cp == 0x20 || (cp >= 0x09 && cp <= 0x0D);
  switch (cp) {
    case 0x85: case 0xA0: case 0x1680: case 0x2028: case 0x2029: case 0x202F: case 0x205F: case 0x3000: return true;
    default: return cp >= 0x2000 && cp <= 0x200A;
  }
}

}  // namespace clusterlm::father::unicode
