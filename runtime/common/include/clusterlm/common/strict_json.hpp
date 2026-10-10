#pragma once
// Strict typed access to JSON objects, shared by every module that reads a persisted or imported document. Every reader
// names the JSON path in its error, and Obj::finish() rejects any key that no reader asked for (unknown fields are errors
// at every level: typo-safe, and a security-relevant field can never be silently ignored). Header-only; includers link
// clusterlm_third_party for nlohmann/json.
#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "clusterlm/common/status.hpp"

namespace clusterlm::strict {
using nlohmann::json;


inline Status bad(std::string what);
// Nesting depth of a JSON text, ignoring string contents.
inline bool depth_within(std::string_view text, int limit);

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
  Status opt_nullable_i64(const char* key, std::optional<std::int64_t>& out);
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



using nlohmann::json;

inline Status bad(std::string what) { return make_error(ErrorCode::kInvalidArgument, std::move(what)); }

inline bool depth_within(std::string_view text, int limit) {
  int depth = 0;
  bool in_str = false, esc = false;
  for (char c : text) {
    if (in_str) {
      if (esc) esc = false;
      else if (c == '\\') esc = true;
      else if (c == '"') in_str = false;
      continue;
    }
    if (c == '"') in_str = true;
    else if (c == '{' || c == '[') {
      if (++depth > limit) return false;
    } else if (c == '}' || c == ']') {
      if (depth > 0) --depth;
    }
  }
  return true;
}

inline Result<Obj> Obj::of(const json& j, std::string path) {
  if (!j.is_object()) return bad(path + " must be an object");
  return Obj(&j, std::move(path));
}

inline const json* Obj::find(const char* key) {
  seen_.insert(key);
  auto it = j_->find(key);
  return it == j_->end() ? nullptr : &*it;
}

inline Status Obj::req_str(const char* key, std::string& out, std::size_t max_len, std::size_t min_len) {
  const json* v = find(key);
  if (v == nullptr) return bad(at(key) + " is required");
  if (!v->is_string()) return bad(at(key) + " must be a string");
  auto s = v->get<std::string>();
  if (s.size() < min_len || s.size() > max_len) return bad(at(key) + " has an invalid length");
  out = std::move(s);
  return Status::ok();
}

inline Status Obj::opt_str(const char* key, std::string& out, std::size_t max_len) {
  const json* v = find(key);
  if (v == nullptr) return Status::ok();
  if (!v->is_string()) return bad(at(key) + " must be a string");
  auto s = v->get<std::string>();
  if (s.size() > max_len) return bad(at(key) + " is too long");
  out = std::move(s);
  return Status::ok();
}

inline Status Obj::opt_nullable_str(const char* key, std::optional<std::string>& out, std::size_t max_len) {
  const json* v = find(key);
  if (v == nullptr) return Status::ok();
  if (v->is_null()) {
    out.reset();
    return Status::ok();
  }
  if (!v->is_string()) return bad(at(key) + " must be a string or null");
  auto s = v->get<std::string>();
  if (s.size() > max_len) return bad(at(key) + " is too long");
  out = std::move(s);
  return Status::ok();
}

inline Status Obj::opt_bool(const char* key, bool& out) {
  const json* v = find(key);
  if (v == nullptr) return Status::ok();
  if (!v->is_boolean()) return bad(at(key) + " must be a boolean");
  out = v->get<bool>();
  return Status::ok();
}

inline Status Obj::req_u64(const char* key, std::uint64_t& out, std::uint64_t lo, std::uint64_t hi) {
  const json* v = find(key);
  if (v == nullptr) return bad(at(key) + " is required");
  if (!v->is_number_unsigned()) return bad(at(key) + " must be a non-negative integer");
  const auto n = v->get<std::uint64_t>();
  if (n < lo || n > hi) return bad(at(key) + " is out of range");
  out = n;
  return Status::ok();
}

inline Status Obj::opt_u32(const char* key, std::uint32_t& out, std::uint32_t lo, std::uint32_t hi) {
  if (!j_->contains(key)) {
    seen_.insert(key);
    return Status::ok();
  }
  std::uint64_t n = 0;
  CLM_RETURN_IF_ERROR(req_u64(key, n, lo, hi));
  out = static_cast<std::uint32_t>(n);
  return Status::ok();
}

inline Status Obj::opt_opt_u32(const char* key, std::optional<std::uint32_t>& out, std::uint32_t lo, std::uint32_t hi) {
  if (!j_->contains(key)) {
    seen_.insert(key);
    return Status::ok();
  }
  std::uint64_t n = 0;
  CLM_RETURN_IF_ERROR(req_u64(key, n, lo, hi));
  out = static_cast<std::uint32_t>(n);
  return Status::ok();
}

inline Status Obj::opt_nullable_i64(const char* key, std::optional<std::int64_t>& out) {
  const json* v = find(key);
  if (v == nullptr) return Status::ok();
  if (v->is_null()) {
    out.reset();
    return Status::ok();
  }
  if (!v->is_number_integer()) return bad(at(key) + " must be an integer or null");
  out = v->get<std::int64_t>();
  return Status::ok();
}

inline Status Obj::opt_nullable_u64(const char* key, std::optional<std::uint64_t>& out, std::uint64_t hi) {
  const json* v = find(key);
  if (v == nullptr) return Status::ok();
  if (v->is_null()) {
    out.reset();
    return Status::ok();
  }
  if (!v->is_number_unsigned() || v->get<std::uint64_t>() > hi) return bad(at(key) + " must be a non-negative integer in range or null");
  out = v->get<std::uint64_t>();
  return Status::ok();
}

inline Status Obj::req_number(const char* key, double& out) {
  const json* v = find(key);
  if (v == nullptr) return bad(at(key) + " is required");
  if (!v->is_number()) return bad(at(key) + " must be a number");
  out = v->get<double>();
  return Status::ok();
}

inline Status Obj::opt_enum(const char* key, const std::vector<std::string_view>& names, std::size_t& idx) {
  const json* v = find(key);
  if (v == nullptr) return Status::ok();
  if (!v->is_string()) return bad(at(key) + " must be a string");
  const auto s = v->get<std::string>();
  for (std::size_t i = 0; i < names.size(); ++i)
    if (names[i] == s) {
      idx = i;
      return Status::ok();
    }
  return bad(at(key) + " has an unknown value");
}

inline Result<std::optional<Obj>> Obj::opt_obj(const char* key) {
  const json* v = find(key);
  if (v == nullptr) return std::optional<Obj>{};
  CLM_ASSIGN_OR_RETURN(Obj o, Obj::of(*v, at(key)));
  return std::optional<Obj>(std::move(o));
}

inline Result<const json*> Obj::opt_array(const char* key, std::size_t max_items) {
  const json* v = find(key);
  if (v == nullptr) return static_cast<const json*>(nullptr);
  if (!v->is_array()) return bad(at(key) + " must be an array");
  if (v->size() > max_items) return bad(at(key) + " has too many items");
  return v;
}

inline Result<const json*> Obj::opt_raw_object(const char* key, std::size_t max_props) {
  const json* v = find(key);
  if (v == nullptr) return static_cast<const json*>(nullptr);
  if (!v->is_object()) return bad(at(key) + " must be an object");
  if (v->size() > max_props) return bad(at(key) + " has too many properties");
  return v;
}

inline Result<const json*> Obj::req_array(const char* key, std::size_t min_items, std::size_t max_items) {
  CLM_ASSIGN_OR_RETURN(const json* v, opt_array(key, max_items));
  if (v == nullptr) return bad(at(key) + " is required");
  if (v->size() < min_items) return bad(at(key) + " has too few items");
  return v;
}

inline Status Obj::opt_string_array(const char* key, std::vector<std::string>& out, std::size_t max_items, std::size_t max_len) {
  CLM_ASSIGN_OR_RETURN(const json* a, opt_array(key, max_items));
  if (a == nullptr) return Status::ok();
  std::vector<std::string> v;
  for (const auto& e : *a) {
    if (!e.is_string() || e.get<std::string>().size() > max_len) return bad(at(key) + " items must be short strings");
    v.push_back(e.get<std::string>());
  }
  out = std::move(v);
  return Status::ok();
}

inline Status Obj::finish() const {
  for (auto it = j_->begin(); it != j_->end(); ++it)
    if (seen_.count(it.key()) == 0) return bad(path_ + ": unknown field '" + it.key().substr(0, 48) + "'");
  return Status::ok();
}

}  // namespace clusterlm::strict
