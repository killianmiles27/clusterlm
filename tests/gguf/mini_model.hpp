#pragma once
// Test-side helpers around the doctest-free mini model builder.
#include <doctest/doctest.h>

#include "../objects/test_util.hpp"
#include "mini_model_core.hpp"

namespace clusterlm::testutil {

// Writes the model into `dir` and returns the file paths (shard order = split.no order).
inline MiniFiles write_mini(const MiniSpec& s, const std::filesystem::path& dir, const MiniHooks& hooks = {}) {
  GgufWriter main, lookup;
  const Status st = build_mini(s, main, lookup, hooks);
  REQUIRE_MESSAGE(st.is_ok(), st.to_string());
  MiniFiles out;
  const auto p0 = dir / (s.split ? "mini-00001-of-00002.gguf" : "mini.gguf");
  REQUIRE(main.write(p0).is_ok());
  out.paths.push_back(p0);
  if (s.split) {
    const auto p1 = dir / "mini-00002-of-00002.gguf";
    REQUIRE(lookup.write(p1).is_ok());
    out.paths.push_back(p1);
  }
  return out;
}

}  // namespace clusterlm::testutil
