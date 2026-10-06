#pragma once
// Shared by tests/objects and tests/domain: scratch directories and cached fixture models.
#include <atomic>
#include <filesystem>
#include <memory>
#include <string>

#include <doctest/doctest.h>

#include "clusterlm/objects/canonical_store.hpp"
#include "clusterlm/objects/fixture_model.hpp"

namespace clusterlm::testutil {

class TempDir {
 public:
  explicit TempDir(const std::string& tag) {
    static std::atomic<unsigned> counter{0};
    path_ = std::filesystem::temp_directory_path() /
            ("clusterlm-test-" + tag + "-" + std::to_string(::std::hash<std::string>{}(tag + std::to_string(counter++))) +
             "-" + std::to_string(reinterpret_cast<std::uintptr_t>(this)));
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

// A generated fixture model on disk (written once per process per configuration).
struct Fixture {
  explicit Fixture(const objects::FixtureSpec& s, const std::string& tag) : spec(s), dir(tag) {
    auto m = objects::write_fixture_model(spec, dir.path());
    REQUIRE_MESSAGE(m.is_ok(), m.status().to_string());
    manifest = std::move(m).value();
  }
  std::unique_ptr<objects::CanonicalModelStore> open_store() const {
    auto s = objects::CanonicalModelStore::open(dir.path());
    REQUIRE_MESSAGE(s.is_ok(), s.status().to_string());
    return std::move(s).value();
  }
  objects::FixtureSpec spec;
  TempDir dir;
  objects::ModelManifest manifest;
};

inline const Fixture& tiny_fixture() {
  static const Fixture f(objects::FixtureSpec::tiny(), "tiny");
  return f;
}
inline const Fixture& full_fixture() {
  static const Fixture f(objects::FixtureSpec{}, "full");
  return f;
}

}  // namespace clusterlm::testutil
