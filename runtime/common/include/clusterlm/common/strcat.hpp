#pragma once
// str_cat: build one string from several pieces with a single allocation. Use it instead of chained operator+ where
// the pieces are assembled in a loop (clang-tidy performance-inefficient-string-concatenation).
#include <string>
#include <string_view>

namespace clusterlm {

// Each part must convert to std::string_view (std::string, std::string_view, const char*).
template <typename... Parts>
std::string str_cat(const Parts&... parts) {
  std::string out;
  out.reserve((std::string_view(parts).size() + ... + 0));
  (out.append(std::string_view(parts)), ...);
  return out;
}

}  // namespace clusterlm
