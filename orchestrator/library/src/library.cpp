#include "clusterlm/library/library.hpp"

#include <algorithm>
#include <cctype>

#include "record_json.hpp"

namespace clusterlm::library {

using nlohmann::json;

namespace {

Status bad(std::string what) { return make_error(ErrorCode::kInvalidArgument, std::move(what)); }

bool is_hex64(std::string_view s) {
  return s.size() == 64 && std::all_of(s.begin(), s.end(), [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}

bool iequals(std::string_view a, std::string_view b) {
  return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(), [](char x, char y) {
           return std::tolower(static_cast<unsigned char>(x)) == std::tolower(static_cast<unsigned char>(y));
         });
}

}  // namespace

const ModelRecord* ModelLibrary::find(std::string_view id) const {
  for (const auto& r : records_)
    if (r.id == id) return &r;
  return nullptr;
}

Status ModelLibrary::upsert(ModelRecord record) {
  CLM_RETURN_IF_ERROR(library::validate(record));
  for (auto& r : records_) {
    if (r.id != record.id) continue;
    // Same structure: refresh where the files live; keep the user's labels and every hash/pin the user or verification set.
    record.name = r.name;
    record.family = r.family;
    record.root_hash = r.root_hash ? r.root_hash : record.root_hash;
    record.pinned_root = r.pinned_root;
    record.discovered_at_unix = r.discovered_at_unix;
    r = std::move(record);
    return Status::ok();
  }
  if (records_.size() >= kMaxLibraryRecords) return make_error(ErrorCode::kResourceExhausted, "model library is full");
  records_.push_back(std::move(record));
  return Status::ok();
}

bool ModelLibrary::remove(std::string_view id) {
  const auto before = records_.size();
  records_.erase(std::remove_if(records_.begin(), records_.end(), [&](const ModelRecord& r) { return r.id == id; }), records_.end());
  return records_.size() != before;
}

Status ModelLibrary::add_scan_root(std::string dir) {
  if (dir.empty() || dir.size() > 1024) return bad("scan root is malformed");
  if (std::find(scan_roots_.begin(), scan_roots_.end(), dir) != scan_roots_.end()) return Status::ok();
  if (scan_roots_.size() >= kMaxScanRoots) return make_error(ErrorCode::kResourceExhausted, "too many scan roots");
  scan_roots_.push_back(std::move(dir));
  return Status::ok();
}

bool ModelLibrary::remove_scan_root(std::string_view dir) {
  const auto before = scan_roots_.size();
  scan_roots_.erase(std::remove(scan_roots_.begin(), scan_roots_.end(), dir), scan_roots_.end());
  return scan_roots_.size() != before;
}

Result<ScanResult> ModelLibrary::rescan(const ScanOptions& opt) {
  ScanResult all;
  for (const auto& root : scan_roots_) {
    auto s = scan_directory(root, opt);
    if (!s.is_ok()) {
      all.issues.push_back({root.substr(root.find_last_of("/\\") == std::string::npos ? 0 : root.find_last_of("/\\") + 1), s.status().message()});
      continue;
    }
    for (auto& r : s->records) {
      CLM_RETURN_IF_ERROR(upsert(r));
      all.records.push_back(std::move(r));
    }
    for (auto& i : s->issues) all.issues.push_back(std::move(i));
  }
  return all;
}

Status ModelLibrary::pin_root(std::string_view id, std::string root_hash) {
  if (!is_hex64(root_hash)) return bad("root hash must be 64 lowercase hex characters");
  for (auto& r : records_)
    if (r.id == id) {
      if (r.root_hash && *r.root_hash != root_hash)
        return make_error(ErrorCode::kFailedPrecondition, "the confirmed root does not match the verified root of this model");
      r.pinned_root = std::move(root_hash);
      return Status::ok();
    }
  return make_error(ErrorCode::kNotFound, "unknown model id");
}

Status ModelLibrary::set_verified_root(std::string_view id, std::string root_hash) {
  if (!is_hex64(root_hash)) return bad("root hash must be 64 lowercase hex characters");
  for (auto& r : records_)
    if (r.id == id) {
      if (r.root_hash && *r.root_hash != root_hash)
        return make_error(ErrorCode::kDataLoss, "verified root differs from the previously verified root: the files changed");
      r.root_hash = std::move(root_hash);
      return Status::ok();
    }
  return make_error(ErrorCode::kNotFound, "unknown model id");
}

Resolution ModelLibrary::resolve(const ModelIdentityQuery& q) const {
  Resolution out;
  std::vector<const ModelRecord*> hits;
  if (q.expected_root_hash) {
    for (const auto& r : records_)
      if ((r.root_hash && *r.root_hash == *q.expected_root_hash) || (r.pinned_root && *r.pinned_root == *q.expected_root_hash)) hits.push_back(&r);
    if (hits.size() == 1) {
      out.kind = Resolution::Kind::kVerified;
      out.record = hits.front();
      out.detail = "matched by pinned root hash";
    } else if (hits.empty()) {
      out.detail = "no model with the pinned root hash is in the library";
    }
  } else {
    for (const auto& r : records_)
      if (iequals(r.family, q.family) && iequals(r.quant, q.quant)) hits.push_back(&r);
    if (hits.size() == 1) {
      out.kind = Resolution::Kind::kUnpinnedMatch;
      out.record = hits.front();
      out.detail = "matched by family and quantization only; the user must confirm the inspected model before it is trusted";
    } else if (hits.empty()) {
      out.detail = "no model with this family and quantization is in the library";
    }
  }
  if (hits.size() > 1) {
    out.kind = Resolution::Kind::kAmbiguous;
    for (const auto* h : hits) out.candidates.push_back(h->id);
    out.detail = "several library models match; pin one by its root hash";
  }
  return out;
}

Status ModelLibrary::validate() const {
  if (records_.size() > kMaxLibraryRecords) return bad("too many models");
  if (scan_roots_.size() > kMaxScanRoots) return bad("too many scan roots");
  for (std::size_t i = 0; i < records_.size(); ++i) {
    CLM_RETURN_IF_ERROR(library::validate(records_[i]));
    for (std::size_t k = 0; k < i; ++k)
      if (records_[k].id == records_[i].id) return bad("duplicate model id");
  }
  for (const auto& r : scan_roots_)
    if (r.empty() || r.size() > 1024) return bad("scan root is malformed");
  return Status::ok();
}

std::string ModelLibrary::to_json() const {
  json recs = json::array();
  for (const auto& r : records_) recs.push_back(detail::record_to_value(r));
  json doc = {{"schema", "clusterlm-model-library"}, {"schema_version", 1}, {"scan_roots", scan_roots_}, {"models", std::move(recs)}};
  return doc.dump(2) + "\n";
}

Result<ModelLibrary> ModelLibrary::from_json(std::string_view text) {
  if (text.size() > (8u << 20)) return bad("library document too large");
  json doc = json::parse(text.begin(), text.end(), nullptr, /*allow_exceptions=*/false);
  if (doc.is_discarded() || !doc.is_object()) return bad("not a JSON object");
  if (auto v = doc.find("schema_version"); v == doc.end() || !v->is_number_integer() || v->get<int>() != 1)
    return make_error(ErrorCode::kVersionMismatch, "unsupported model library schema_version");
  ModelLibrary lib;
  if (auto roots = doc.find("scan_roots"); roots != doc.end()) {
    if (!roots->is_array() || roots->size() > kMaxScanRoots) return bad("scan_roots must be a short array");
    for (const auto& r : *roots) {
      if (!r.is_string()) return bad("scan root must be a string");
      lib.scan_roots_.push_back(r.get<std::string>());
    }
  }
  if (auto models = doc.find("models"); models != doc.end()) {
    if (!models->is_array() || models->size() > kMaxLibraryRecords) return bad("models must be an array");
    for (const auto& m : *models) {
      CLM_ASSIGN_OR_RETURN(auto rec, detail::record_from_value(m));
      lib.records_.push_back(std::move(rec));
    }
  }
  CLM_RETURN_IF_ERROR(lib.validate());
  return lib;
}

}  // namespace clusterlm::library
