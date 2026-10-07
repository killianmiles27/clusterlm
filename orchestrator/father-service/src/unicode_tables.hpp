#pragma once
// Compact Unicode class table for the tokenizer's pre-tokenizer (generated data in unicode_tables.cpp, produced by
// scripts/gen_unicode_tables.py) plus the fixed Unicode White_Space set.
#include <cstddef>
#include <cstdint>

namespace clusterlm::father::unicode {

enum class Class : std::uint8_t { kOther = 0, kLetter = 1, kMark = 2, kNumber = 3 };

extern const char* const kUnicodeVersion;
extern const std::uint32_t kClassRanges[];  // (first code point << 2) | class, ascending
extern const std::size_t kClassRangeCount;

// \p{L} / \p{M} / \p{N} / everything else. Unassigned and out-of-range code points are kOther.
Class class_of(std::uint32_t cp) noexcept;
// Unicode White_Space property (what \s means in the pre-tokenizer patterns): U+0009..U+000D, U+0020, U+0085,
// U+00A0, U+1680, U+2000..U+200A, U+2028, U+2029, U+202F, U+205F, U+3000.
bool is_white_space(std::uint32_t cp) noexcept;

}  // namespace clusterlm::father::unicode
