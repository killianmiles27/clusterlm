#include "clusterlm/library/model_record.hpp"
#include "record_json.hpp"

#include <algorithm>

#include <nlohmann/json.hpp>

namespace clusterlm::library {

using nlohmann::json;

namespace {

constexpr std::size_t kMaxLabel = 256;
constexpr std::size_t kMaxPath = 1024;
constexpr std::size_t kMaxDocument = 1u << 20;

Status bad(std::string what) { return make_error(ErrorCode::kInvalidArgument, std::move(what)); }

bool is_hex(std::string_view s, std::size_t n) {
  return s.size() == n && std::all_of(s.begin(), s.end(), [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}

bool printable(std::string_view s, std::size_t max, bool allow_empty = false) {
  if ((s.empty() && !allow_empty) || s.size() > max) return false;
  return std::all_of(s.begin(), s.end(), [](unsigned char c) { return c >= 0x20 && c != 0x7F; });
}

template <typename T>
Status get(const json& o, const char* key, T& out, bool required = false) {
  auto it = o.find(key);
  if (it == o.end() || it->is_null()) return required ? bad(std::string("missing field '") + key + "'") : Status::ok();
  if constexpr (std::is_same_v<T, std::string>) {
    if (!it->is_string()) return bad(std::string("field '") + key + "' must be a string");
    out = it->get<std::string>();
  } else if constexpr (std::is_same_v<T, bool>) {
    if (!it->is_boolean()) return bad(std::string("field '") + key + "' must be a boolean");
    out = it->get<bool>();
  } else if constexpr (std::is_same_v<T, std::int64_t>) {
    if (!it->is_number_integer()) return bad(std::string("field '") + key + "' must be an integer");
    out = it->get<std::int64_t>();
  } else {
    static_assert(std::is_same_v<T, std::uint64_t>);
    if (!it->is_number_unsigned()) return bad(std::string("field '") + key + "' must be a non-negative integer");
    out = it->get<std::uint64_t>();
  }
  return Status::ok();
}

Status get_u32(const json& o, const char* key, std::optional<std::uint32_t>& out) {
  auto it = o.find(key);
  if (it == o.end() || it->is_null()) return Status::ok();
  if (!it->is_number_unsigned() || it->get<std::uint64_t>() > 0xFFFFFFFFull)
    return bad(std::string("field '") + key + "' must be a 32-bit non-negative integer");
  out = it->get<std::uint32_t>();
  return Status::ok();
}

Status get_hex(const json& o, const char* key, std::optional<std::string>& out) {
  auto it = o.find(key);
  if (it == o.end() || it->is_null()) return Status::ok();
  if (!it->is_string()) return bad(std::string("field '") + key + "' must be a string");
  out = it->get<std::string>();
  return Status::ok();
}

}  // namespace
namespace detail {
json record_to_value(const ModelRecord& r) {
  json files = json::array();
  for (const auto& f : r.files) files.push_back({{"name", f.name}, {"bytes", f.bytes}});
  json types = json::array();
  for (const auto& t : r.tensor_types) types.push_back({{"type", t.type}, {"tensors", t.tensors}, {"bytes", t.bytes}});
  json j = {{"id", r.id},
            {"name", r.name},
            {"family", r.family},
            {"architecture", r.architecture},
            {"container", r.container},
            {"quant", r.quant},
            {"dir", r.dir},
            {"files", std::move(files)},
            {"split", r.split},
            {"total_bytes", r.total_bytes},
            {"tensor_count", r.tensor_count},
            {"tensor_types", std::move(types)},
            {"structure_digest", r.structure_digest},
            {"discovered_at_unix", r.discovered_at_unix}};
  auto opt = [&](const char* k, const std::optional<std::uint32_t>& v) { j[k] = v ? json(*v) : json(nullptr); };
  opt("block_count", r.block_count);
  opt("context_length", r.context_length);
  opt("expert_count", r.expert_count);
  j["root_hash"] = r.root_hash ? json(*r.root_hash) : json(nullptr);
  j["pinned_root"] = r.pinned_root ? json(*r.pinned_root) : json(nullptr);
  return j;
}
}  // namespace detail

bool is_model_id(std::string_view s) {
  return s.size() == kModelIdPrefix.size() + 24 && s.substr(0, kModelIdPrefix.size()) == kModelIdPrefix &&
         is_hex(s.substr(kModelIdPrefix.size()), 24);
}

std::string model_id_from_digest(std::string_view d) {
  return std::string(kModelIdPrefix) + std::string(d.substr(0, 24));
}

Status validate(const ModelRecord& r) {
  if (!is_model_id(r.id)) return bad("model id must be mdl_ + 24 lowercase hex characters");
  if (!is_hex(r.structure_digest, 64)) return bad("structure_digest must be 64 lowercase hex characters");
  if (r.id != model_id_from_digest(r.structure_digest)) return bad("model id does not match its structure digest");
  if (!printable(r.name, kMaxLabel)) return bad("model name must be 1.." + std::to_string(kMaxLabel) + " printable characters");
  if (!printable(r.family, kMaxLabel)) return bad("model family must be 1.." + std::to_string(kMaxLabel) + " printable characters");
  if (!printable(r.architecture, kMaxLabel, true)) return bad("architecture is malformed");
  if (r.container != "gguf") return bad("container must be 'gguf'");
  if (!printable(r.quant, 32)) return bad("quant is malformed");
  if (!printable(r.dir, kMaxPath, true)) return bad("dir is malformed");
  if (r.files.empty() || r.files.size() > kMaxRecordFiles) return bad("a model needs 1.." + std::to_string(kMaxRecordFiles) + " files");
  if (r.split != (r.files.size() > 1)) return bad("split must be true exactly when there are several files");
  std::uint64_t sum = 0;
  for (const auto& f : r.files) {
    if (!printable(f.name, kMaxPath) || f.name.find('/') != std::string::npos || f.name.find('\\') != std::string::npos || f.name == "." ||
        f.name == "..")
      return bad("file names must be plain names without directories");
    if (f.bytes > (1ull << 50) || sum > (1ull << 50) - f.bytes) return bad("file sizes out of range");
    sum += f.bytes;
  }
  if (sum != r.total_bytes) return bad("total_bytes must equal the sum of the file sizes");
  if (r.tensor_types.size() > kMaxTensorTypes) return bad("too many tensor types");
  std::uint64_t tensors = 0;
  for (const auto& t : r.tensor_types) {
    if (!printable(t.type, 32)) return bad("tensor type name is malformed");
    tensors += t.tensors;
  }
  if (tensors != r.tensor_count) return bad("tensor_count must equal the sum of the per-type counts");
  if (r.root_hash && !is_hex(*r.root_hash, 64)) return bad("root_hash must be 64 lowercase hex characters");
  if (r.pinned_root && !is_hex(*r.pinned_root, 64)) return bad("pinned_root must be 64 lowercase hex characters");
  if (r.discovered_at_unix < 0) return bad("discovered_at_unix must be >= 0");
  return Status::ok();
}

std::string to_json(const ModelRecord& r) { return detail::record_to_value(r).dump(2) + "\n"; }

namespace detail {
Result<ModelRecord> record_from_value(const json& j) {
  if (!j.is_object()) return bad("model record must be an object");
  ModelRecord r;
  CLM_RETURN_IF_ERROR(get(j, "id", r.id, true));
  CLM_RETURN_IF_ERROR(get(j, "name", r.name, true));
  CLM_RETURN_IF_ERROR(get(j, "family", r.family, true));
  CLM_RETURN_IF_ERROR(get(j, "architecture", r.architecture));
  CLM_RETURN_IF_ERROR(get(j, "container", r.container));
  CLM_RETURN_IF_ERROR(get(j, "quant", r.quant, true));
  CLM_RETURN_IF_ERROR(get(j, "dir", r.dir));
  CLM_RETURN_IF_ERROR(get(j, "split", r.split));
  CLM_RETURN_IF_ERROR(get(j, "total_bytes", r.total_bytes, true));
  CLM_RETURN_IF_ERROR(get(j, "tensor_count", r.tensor_count));
  CLM_RETURN_IF_ERROR(get(j, "structure_digest", r.structure_digest, true));
  CLM_RETURN_IF_ERROR(get(j, "discovered_at_unix", r.discovered_at_unix));
  CLM_RETURN_IF_ERROR(get_u32(j, "block_count", r.block_count));
  CLM_RETURN_IF_ERROR(get_u32(j, "context_length", r.context_length));
  CLM_RETURN_IF_ERROR(get_u32(j, "expert_count", r.expert_count));
  CLM_RETURN_IF_ERROR(get_hex(j, "root_hash", r.root_hash));
  CLM_RETURN_IF_ERROR(get_hex(j, "pinned_root", r.pinned_root));
  auto files = j.find("files");
  if (files == j.end() || !files->is_array()) return bad("field 'files' must be an array");
  if (files->size() > kMaxRecordFiles) return bad("too many files");
  for (const auto& f : *files) {
    if (!f.is_object()) return bad("file entry must be an object");
    ModelFile mf;
    CLM_RETURN_IF_ERROR(get(f, "name", mf.name, true));
    CLM_RETURN_IF_ERROR(get(f, "bytes", mf.bytes, true));
    r.files.push_back(std::move(mf));
  }
  if (auto types = j.find("tensor_types"); types != j.end()) {
    if (!types->is_array()) return bad("field 'tensor_types' must be an array");
    if (types->size() > kMaxTensorTypes) return bad("too many tensor types");
    for (const auto& t : *types) {
      if (!t.is_object()) return bad("tensor type entry must be an object");
      TensorTypeStat ts;
      CLM_RETURN_IF_ERROR(get(t, "type", ts.type, true));
      CLM_RETURN_IF_ERROR(get(t, "tensors", ts.tensors, true));
      CLM_RETURN_IF_ERROR(get(t, "bytes", ts.bytes, true));
      r.tensor_types.push_back(std::move(ts));
    }
  }
  CLM_RETURN_IF_ERROR(validate(r));
  return r;
}
}  // namespace detail

Result<ModelRecord> model_record_from_json(std::string_view text) {
  if (text.size() > kMaxDocument) return bad("model record document too large");
  json j = json::parse(text.begin(), text.end(), nullptr, /*allow_exceptions=*/false);
  if (j.is_discarded()) return bad("not valid JSON");
  return detail::record_from_value(j);
}

}  // namespace clusterlm::library
