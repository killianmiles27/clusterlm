#include "storage_census.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>

namespace clusterlm::bench {

namespace fs = std::filesystem;

namespace {

std::int64_t to_ns(fs::file_time_type t) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count();
}

bool is_under(const fs::path& p, const fs::path& dir) {
  const std::string a = p.lexically_normal().generic_string(), b = dir.lexically_normal().generic_string();
  if (b.empty()) return false;
  if (a.size() < b.size() || a.compare(0, b.size(), b) != 0) return false;
  return a.size() == b.size() || a[b.size()] == '/' || b.back() == '/';
}

}  // namespace

CensusEnvironment CensusEnvironment::from_process() {
  CensusEnvironment e;
#if defined(_WIN32)
  e.windows = true;
#endif
  for (const char* k : {"APPDATA", "LOCALAPPDATA", "TEMP", "TMP", "TMPDIR", "HOME", "USERPROFILE", "CUDA_CACHE_PATH", "XDG_CACHE_HOME",
                        "XDG_DATA_HOME"})
    if (const char* v = std::getenv(k); v != nullptr && *v != '\0') e.vars[k] = v;
  return e;
}

std::vector<CensusRoot> census_roots_for(const CensusEnvironment& env) {
  auto var = [&](const char* k) -> std::string {
    auto it = env.vars.find(k);
    return it == env.vars.end() ? std::string() : it->second;
  };
  std::vector<CensusRoot> roots;
  auto add = [&](std::string label, fs::path p, int depth = 8) {
    CensusRoot r;
    r.label = std::move(label);
    r.path = std::move(p);
    r.max_depth = depth;
    roots.push_back(std::move(r));
  };
  if (env.windows) {
    const std::string cuda = var("CUDA_CACHE_PATH");
    if (!cuda.empty()) add("cuda_compute_cache", cuda);
    else if (!var("APPDATA").empty()) add("cuda_compute_cache", fs::path(var("APPDATA")) / "NVIDIA" / "ComputeCache");
    if (!var("LOCALAPPDATA").empty()) {
      const fs::path local = var("LOCALAPPDATA");
      add("nvidia_dx_cache", local / "NVIDIA" / "DXCache");
      add("nvidia_gl_cache", local / "NVIDIA" / "GLCache");
      add("clusterlm_local_appdata", local / "ClusterLM");
    }
    const std::string temp = !var("TEMP").empty() ? var("TEMP") : var("TMP");
    if (!temp.empty()) add("os_temp", temp, 2);
  } else {
    const std::string home = var("HOME");
    const std::string cuda = var("CUDA_CACHE_PATH");
    if (!cuda.empty()) add("cuda_compute_cache", cuda);
    else if (!home.empty()) add("cuda_compute_cache", fs::path(home) / ".nv" / "ComputeCache");
    const fs::path cache = !var("XDG_CACHE_HOME").empty() ? fs::path(var("XDG_CACHE_HOME")) : (home.empty() ? fs::path() : fs::path(home) / ".cache");
    if (!cache.empty()) add("nvidia_cache", cache / "nvidia");
    const fs::path data = !var("XDG_DATA_HOME").empty() ? fs::path(var("XDG_DATA_HOME")) : (home.empty() ? fs::path() : fs::path(home) / ".local" / "share");
    if (!data.empty()) add("clusterlm_user_data", data / "clusterlm");
    add("os_temp", !var("TMPDIR").empty() ? var("TMPDIR") : std::string("/tmp"), 2);
  }
  return roots;
}

std::vector<CensusRoot> default_census_roots(const std::vector<CensusRoot>& extra, const std::vector<fs::path>& exclude) {
  auto roots = census_roots_for(CensusEnvironment::from_process());
  roots.insert(roots.end(), extra.begin(), extra.end());
  for (auto& r : roots) r.exclude.insert(r.exclude.end(), exclude.begin(), exclude.end());
  return roots;
}

CensusSnapshot take_census(const std::vector<CensusRoot>& roots) {
  CensusSnapshot snap;
  snap.taken_ns = to_ns(fs::file_time_type::clock::now());
  for (const auto& root : roots) {
    RootSnapshot rs;
    rs.label = root.label;
    rs.path = root.path.generic_string();
    std::error_code ec;
    rs.exists = fs::is_directory(root.path, ec);
    if (!rs.exists) {
      snap.roots.push_back(std::move(rs));
      continue;
    }
    fs::recursive_directory_iterator it(root.path, fs::directory_options::skip_permission_denied, ec);
    if (ec) rs.error = ec.message();
    for (; !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
      const fs::path p = it->path();
      if (std::any_of(root.exclude.begin(), root.exclude.end(), [&](const fs::path& x) { return is_under(p, x); })) {
        it.disable_recursion_pending();
        continue;
      }
      std::error_code fec;
      const bool is_dir = it->is_directory(fec);
      if (is_dir) {
        // depth() is 0 for the root's children: descend while depth + 1 < max_depth.
        if (it.depth() + 1 >= root.max_depth) it.disable_recursion_pending();
        continue;
      }
      if (!it->is_regular_file(fec) || fec) continue;
      CensusFile f;
      f.size = it->file_size(fec);
      if (fec) continue;
      f.mtime_ns = to_ns(it->last_write_time(fec));
      if (rs.files.size() >= root.max_entries) {
        rs.truncated = true;
        break;
      }
      rs.files[p.lexically_relative(root.path).generic_string()] = f;
    }
    if (ec && rs.error.empty()) rs.error = ec.message();
    snap.roots.push_back(std::move(rs));
  }
  return snap;
}

CensusDiff diff_census(const CensusSnapshot& before, const CensusSnapshot& after, std::int64_t window_start_ns,
                       std::int64_t window_end_ns) {
  CensusDiff d;
  auto in_window = [&](std::int64_t mtime) {
    return (window_start_ns == 0 || mtime >= window_start_ns) && (window_end_ns == 0 || mtime <= window_end_ns);
  };
  for (const auto& a : after.roots) {
    const RootSnapshot* b = nullptr;
    for (const auto& cand : before.roots)
      if (cand.label == a.label && cand.path == a.path) b = &cand;
    if (!a.exists && (b == nullptr || !b->exists)) {
      d.absent_roots.push_back(a.label);
      continue;
    }
    if (a.truncated || (b != nullptr && b->truncated)) d.truncated_roots.push_back(a.label);
    static const std::map<std::string, CensusFile> kNone;
    const auto& bf = b != nullptr ? b->files : kNone;
    for (const auto& [name, f] : a.files) {
      auto it = bf.find(name);
      if (it == bf.end()) {
        d.added.push_back({a.label, name, 0, f.size, true});
        d.added_bytes += f.size;
        d.bytes_by_root[a.label] += f.size;
      } else if (it->second.size != f.size || it->second.mtime_ns != f.mtime_ns) {
        d.changed.push_back({a.label, name, it->second.size, f.size, in_window(f.mtime_ns)});
        d.changed_bytes_delta_abs += f.size > it->second.size ? f.size - it->second.size : it->second.size - f.size;
        d.bytes_by_root[a.label] += f.size;
      }
    }
    for (const auto& [name, f] : bf)
      if (a.files.find(name) == a.files.end()) {
        d.removed.push_back({a.label, name, f.size, 0, true});
        d.removed_bytes += f.size;
      }
  }
  return d;
}

nlohmann::json census_to_json(const CensusSnapshot& s) {
  nlohmann::json j;
  j["taken_ns"] = s.taken_ns;
  j["roots"] = nlohmann::json::array();
  for (const auto& r : s.roots) {
    nlohmann::json rj = {{"label", r.label}, {"path", r.path}, {"exists", r.exists}, {"truncated", r.truncated}};
    if (!r.error.empty()) rj["error"] = r.error;
    nlohmann::json files = nlohmann::json::object();
    for (const auto& [name, f] : r.files) files[name] = {f.size, f.mtime_ns};
    rj["files"] = std::move(files);
    j["roots"].push_back(std::move(rj));
  }
  return j;
}

Result<CensusSnapshot> census_from_json(const nlohmann::json& j) {
  CensusSnapshot s;
  if (!j.is_object() || !j.contains("roots") || !j["roots"].is_array())
    return make_error(ErrorCode::kInvalidArgument, "not a storage census snapshot");
  s.taken_ns = j.value("taken_ns", std::int64_t{0});
  for (const auto& rj : j["roots"]) {
    RootSnapshot r;
    r.label = rj.value("label", std::string());
    r.path = rj.value("path", std::string());
    r.exists = rj.value("exists", false);
    r.truncated = rj.value("truncated", false);
    r.error = rj.value("error", std::string());
    if (rj.contains("files") && rj["files"].is_object())
      for (const auto& [name, v] : rj["files"].items()) {
        if (!v.is_array() || v.size() != 2) return make_error(ErrorCode::kInvalidArgument, "bad census file entry");
        r.files[name] = CensusFile{v[0].get<std::uint64_t>(), v[1].get<std::int64_t>()};
      }
    s.roots.push_back(std::move(r));
  }
  return s;
}

nlohmann::json census_diff_to_json(const CensusDiff& d, std::size_t max_listed) {
  auto list = [&](const std::vector<CensusChange>& v) {
    nlohmann::json a = nlohmann::json::array();
    for (std::size_t i = 0; i < v.size() && i < max_listed; ++i)
      a.push_back({{"root", v[i].root}, {"name", v[i].name}, {"size_before", v[i].size_before}, {"size_after", v[i].size_after},
                   {"in_window", v[i].in_window}});
    return a;
  };
  nlohmann::json j;
  j["added_files"] = d.added.size();
  j["changed_files"] = d.changed.size();
  j["removed_files"] = d.removed.size();
  j["added_bytes"] = d.added_bytes;
  j["removed_bytes"] = d.removed_bytes;
  j["listed_cap"] = max_listed;
  j["added"] = list(d.added);
  j["changed"] = list(d.changed);
  j["removed"] = list(d.removed);
  j["truncated_roots"] = d.truncated_roots;
  j["absent_roots"] = d.absent_roots;
  j["bytes_by_root"] = d.bytes_by_root;
  return j;
}

void emit_census_diff(BenchmarkResult& r, const CensusDiff& d, const std::string& prefix) {
  const std::string k = prefix + "census.";
  r.metric(k + "added_files", d.added.size());
  r.metric(k + "added_bytes", d.added_bytes);
  r.metric(k + "changed_files", d.changed.size());
  r.metric(k + "removed_files", d.removed.size());
  r.metric(k + "truncated_roots", d.truncated_roots);
  r.metric(k + "absent_roots", d.absent_roots);
  r.metric(k + "diff", census_diff_to_json(d));
}

}  // namespace clusterlm::bench
