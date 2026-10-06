#pragma once
// Minimal command-line option parsing shared by the ClusterLM executables.
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace clusterlm::cli {

class Args {
 public:
  // argv[first..] as "--key value", "--key=value" or "--flag". Repeated keys accumulate.
  Args(int argc, char** argv, int first = 1) {
    for (int i = first; i < argc; ++i) {
      std::string a = argv[i];
      if (a.rfind("--", 0) != 0) {
        positional_.push_back(a);
        continue;
      }
      a = a.substr(2);
      if (auto eq = a.find('='); eq != std::string::npos) {
        values_[a.substr(0, eq)].push_back(a.substr(eq + 1));
      } else if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
        values_[a].push_back(argv[++i]);
      } else {
        values_[a].push_back("");
      }
    }
  }
  bool has(const std::string& k) const { return values_.count(k) != 0; }
  std::string get(const std::string& k, const std::string& def = "") const {
    auto it = values_.find(k);
    return it == values_.end() ? def : it->second.back();
  }
  std::vector<std::string> all(const std::string& k) const {
    auto it = values_.find(k);
    return it == values_.end() ? std::vector<std::string>{} : it->second;
  }
  double number(const std::string& k, double def) const { return has(k) ? std::stod(get(k)) : def; }
  std::uint64_t integer(const std::string& k, std::uint64_t def) const { return has(k) ? std::stoull(get(k)) : def; }
  const std::vector<std::string>& positional() const { return positional_; }

 private:
  std::map<std::string, std::vector<std::string>> values_;
  std::vector<std::string> positional_;
};

inline std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : s) {
    if (c == sep) {
      out.push_back(cur);
      cur.clear();
    } else {
      cur.push_back(c);
    }
  }
  if (!cur.empty()) out.push_back(cur);
  return out;
}

constexpr std::uint64_t kGiB = 1ull << 30;

}  // namespace clusterlm::cli
