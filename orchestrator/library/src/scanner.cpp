#include "clusterlm/library/scanner.hpp"

#include <algorithm>
#include <cctype>
#include <map>
#include <regex>
#include <set>

#include "clusterlm/common/digest.hpp"
#include "clusterlm/objects/ggml_types.hpp"

namespace clusterlm::library {

namespace fs = std::filesystem;
using objects::GgufFile;
using objects::GgufModelFiles;

namespace {

std::string upper(std::string s) {
  for (auto& c : s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  return s;
}

std::string meta_string(const GgufFile& f, const std::string& key) {
  const auto* v = f.find(key);
  return (v != nullptr && v->is_string()) ? v->s : std::string();
}

std::optional<std::uint32_t> meta_u32(const GgufFile& f, const std::string& key) {
  const auto* v = f.find(key);
  if (v == nullptr) return std::nullopt;
  auto n = v->as_u64();
  if (!n || *n > 0xFFFFFFFFull) return std::nullopt;
  return static_cast<std::uint32_t>(*n);
}

// Control characters would fail record validation; replace instead of dropping a perfectly good model.
std::string clean_label(std::string s, std::size_t max) {
  for (auto& c : s)
    if (static_cast<unsigned char>(c) < 0x20 || c == 0x7F) c = ' ';
  if (s.size() > max) s.resize(max);
  return s;
}

}  // namespace

Result<ModelRecord> identify_model(const std::vector<fs::path>& paths_in, const ScanOptions& opt) {
  if (paths_in.empty()) return make_error(ErrorCode::kInvalidArgument, "no model file given");
  std::vector<fs::path> paths = paths_in;
  if (paths.size() == 1) {
    CLM_ASSIGN_OR_RETURN(paths, objects::expand_split_paths(paths.front()));
  }
  CLM_ASSIGN_OR_RETURN(GgufModelFiles model, objects::open_gguf_model(paths, opt.limits));

  ModelRecord r;
  const GgufFile& meta = model.meta();
  r.architecture = clean_label(meta_string(meta, "general.architecture"), 256);
  const std::string stem = model.paths.front().stem().string();
  std::string name = meta_string(meta, "general.name");
  std::string basename = meta_string(meta, "general.basename");
  r.name = clean_label(!name.empty() ? name : stem, 256);
  r.family = clean_label(!basename.empty() ? basename : (!name.empty() ? name : stem), 256);
  r.dir = model.paths.front().parent_path().string();
  r.split = model.is_split || model.paths.size() > 1;
  if (!r.architecture.empty()) {
    r.block_count = meta_u32(meta, r.architecture + ".block_count");
    r.context_length = meta_u32(meta, r.architecture + ".context_length");
    r.expert_count = meta_u32(meta, r.architecture + ".expert_count");
  }

  Sha256 h;
  h.update(std::string_view("clusterlm-model-structure-v1\n"));
  std::map<std::string, TensorTypeStat> types;
  for (std::size_t i = 0; i < model.paths.size(); ++i) {
    const GgufFile& g = model.shards[i];
    ModelFile mf{model.paths[i].filename().string(), g.file_size};
    h.update(mf.name + '\0' + std::to_string(mf.bytes) + '\n');
    r.total_bytes += mf.bytes;
    r.files.push_back(std::move(mf));
    for (const auto& t : g.tensors) {
      std::string line = t.name + '\0' + std::to_string(static_cast<std::uint32_t>(t.type)) + ':';
      for (std::uint32_t d = 0; d < t.n_dims; ++d) line += std::to_string(t.dims[d]) + ',';
      h.update(line + '\n');
      const auto* info = objects::ggml_type_info(static_cast<std::uint32_t>(t.type));
      auto& s = types[upper(info != nullptr ? std::string(info->name) : "type" + std::to_string(static_cast<std::uint32_t>(t.type)))];
      ++s.tensors;
      s.bytes += t.n_bytes;
      ++r.tensor_count;
    }
  }
  r.structure_digest = h.finish().hex();
  r.id = model_id_from_digest(r.structure_digest);

  for (auto& [type, stat] : types) {
    stat.type = type;
    r.tensor_types.push_back(stat);
  }
  std::sort(r.tensor_types.begin(), r.tensor_types.end(), [](const TensorTypeStat& a, const TensorTypeStat& b) {
    return a.bytes != b.bytes ? a.bytes > b.bytes : a.type < b.type;
  });
  r.quant = r.tensor_types.empty() ? "UNKNOWN" : r.tensor_types.front().type;
  r.discovered_at_unix = opt.now_unix;
  CLM_RETURN_IF_ERROR(validate(r));
  return r;
}

Result<ScanResult> scan_directory(const fs::path& dir, const ScanOptions& opt) {
  std::error_code ec;
  if (!fs::is_directory(dir, ec)) return make_error(ErrorCode::kNotFound, "model directory does not exist");

  std::vector<fs::path> files;
  auto consider = [&](const fs::directory_entry& e) {
    std::error_code e2;
    if (!e.is_regular_file(e2)) return;
    std::string ext = upper(e.path().extension().string());
    if (ext == ".GGUF") files.push_back(e.path());
  };
  if (opt.recursive) {
    for (fs::recursive_directory_iterator it(dir, fs::directory_options::skip_permission_denied, ec), end; !ec && it != end; it.increment(ec))
      consider(*it);
  } else {
    for (fs::directory_iterator it(dir, fs::directory_options::skip_permission_denied, ec), end; !ec && it != end; it.increment(ec))
      consider(*it);
  }
  if (ec) return make_error(ErrorCode::kUnavailable, "could not list the model directory");
  std::sort(files.begin(), files.end());
  if (files.size() > opt.max_files) files.resize(opt.max_files);

  // Group split sets so a 3-shard model is one record.
  static const std::regex split_re(R"(^(.*)-(\d{5})-of-(\d{5})\.gguf$)", std::regex::icase);
  struct Group {
    std::vector<fs::path> members;
    std::string label;
  };
  std::map<std::string, Group> groups;  // key: parent|stem|count
  std::vector<std::string> order;
  for (const auto& p : files) {
    const std::string fname = p.filename().string();
    std::smatch m;
    std::string key = p.string();
    if (std::regex_match(fname, m, split_re)) key = p.parent_path().string() + "|" + m[1].str() + "|" + m[3].str();
    auto [it, inserted] = groups.try_emplace(key);
    if (inserted) order.push_back(key);
    it->second.members.push_back(p);
    if (it->second.label.empty()) it->second.label = fname;
  }

  ScanResult out;
  std::set<std::string> seen_ids;
  for (const auto& key : order) {
    const Group& g = groups.at(key);
    auto rec = identify_model({g.members.front()}, opt);
    if (!rec.is_ok()) {
      out.issues.push_back({g.label, rec.status().message()});
      continue;
    }
    if (!seen_ids.insert(rec->id).second) continue;  // an identical copy in the same scan
    out.records.push_back(std::move(rec).value());
  }
  return out;
}

}  // namespace clusterlm::library
