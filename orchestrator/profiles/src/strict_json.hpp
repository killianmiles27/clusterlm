#pragma once
// Internal: strict typed access to JSON objects. Every reader names the JSON path in its error, and Obj::finish()
// rejects any key that no reader asked for (unknown fields are errors at every level, frozen interface v1.1 §1a).
#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "clusterlm/common/status.hpp"

namespace clusterlm::profiles::detail {

Status bad(std::string what);
// Nesting depth of a JSON text, ignoring string contents.
bool depth_within(std::string_view text, int limit);
// Bounded parse of a whole document into an object.
Result<nlohmann::json> parse_object_document(std::string_view text);

class Obj {
 public:
  static Result<Obj> of(const nlohmann::json& j, std::string path);

  const std::string& path() const { return path_; }
  bool has(const char* key) const { return j_->contains(key); }

  Status req_str(const char* key, std::string& out, std::size_t max_len, std::size_t min_len = 1);
  Status opt_str(const char* key, std::string& out, std::size_t max_len);
  Status opt_nullable_str(const char* key, std::optional<std::string>& out, std::size_t max_len);
  Status opt_bool(const char* key, bool& out);
  Status opt_u32(const char* key, std::uint32_t& out, std::uint32_t lo, std::uint32_t hi);
  Status opt_opt_u32(const char* key, std::optional<std::uint32_t>& out, std::uint32_t lo, std::uint32_t hi);
  Status req_u64(const char* key, std::uint64_t& out, std::uint64_t lo, std::uint64_t hi);
  Status opt_nullable_u64(const char* key, std::optional<std::uint64_t>& out, std::uint64_t hi);
  Status req_number(const char* key, double& out);
  // Index into `names`; `idx` keeps its value when the key is absent.
  Status opt_enum(const char* key, const std::vector<std::string_view>& names, std::size_t& idx);
  Result<std::optional<Obj>> opt_obj(const char* key);
  Result<const nlohmann::json*> opt_array(const char* key, std::size_t max_items);
  // An object whose keys are data (not field names): returned raw, the caller checks keys and values.
  Result<const nlohmann::json*> opt_raw_object(const char* key, std::size_t max_props);
  Result<const nlohmann::json*> req_array(const char* key, std::size_t min_items, std::size_t max_items);
  Status opt_string_array(const char* key, std::vector<std::string>& out, std::size_t max_items, std::size_t max_len);
  // Fails on any key no reader touched.
  Status finish() const;

 private:
  Obj(const nlohmann::json* j, std::string path) : j_(j), path_(std::move(path)) {}
  const nlohmann::json* find(const char* key);
  std::string at(const char* key) const { return path_ + "." + key; }

  const nlohmann::json* j_;
  std::string path_;
  std::set<std::string> seen_;
};

}  // namespace clusterlm::profiles::detail
