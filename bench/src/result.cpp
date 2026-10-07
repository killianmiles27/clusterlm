#include "result.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>

namespace clusterlm::bench {

namespace {
double percentile_of(std::vector<double> v, double p) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  const double idx = p * static_cast<double>(v.size() - 1);
  const auto lo = static_cast<std::size_t>(std::floor(idx));
  const auto hi = static_cast<std::size_t>(std::ceil(idx));
  return v[lo] + (v[hi] - v[lo]) * (idx - static_cast<double>(lo));
}
}  // namespace

double Distribution::mean() const {
  if (samples.empty()) return 0;
  double sum = 0;
  for (double s : samples) sum += s;
  return sum / static_cast<double>(samples.size());
}

double Distribution::stddev() const {
  if (samples.size() < 2) return 0;
  const double m = mean();
  double acc = 0;
  for (double s : samples) acc += (s - m) * (s - m);
  return std::sqrt(acc / static_cast<double>(samples.size() - 1));
}

double Distribution::percentile(double p) const { return percentile_of(samples, p); }

nlohmann::json Distribution::to_json(const std::string& unit) const {
  nlohmann::json j;
  j["n"] = samples.size();
  j["unit"] = unit;
  if (samples.empty()) return j;
  double sum = 0;
  for (double s : samples) sum += s;
  j["mean"] = sum / static_cast<double>(samples.size());
  j["min"] = *std::min_element(samples.begin(), samples.end());
  j["p10"] = percentile_of(samples, 0.10);
  j["p50"] = percentile_of(samples, 0.50);
  j["p90"] = percentile_of(samples, 0.90);
  j["p95"] = percentile_of(samples, 0.95);
  j["p99"] = percentile_of(samples, 0.99);
  j["max"] = *std::max_element(samples.begin(), samples.end());
  j["stddev"] = stddev();
  return j;
}

std::string iso8601_now() {
  const std::time_t t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::tm tm{};
#ifdef _WIN32
  gmtime_s(&tm, &t);
#else
  gmtime_r(&t, &tm);
#endif
  char buf[32];
  std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", &tm);
  return buf;
}

BenchmarkResult::BenchmarkResult(std::string experiment, const HostInfo& host) {
  doc_["schema_version"] = 1;
  doc_["tool"] = {{"name", "clusterlm-bench"},
                  {"version", "0.1.0"},
#ifdef NDEBUG
                  {"build", "release"}
#else
                  {"build", "debug"}
#endif
  };
  doc_["experiment"] = std::move(experiment);
  doc_["environment"] = {{"host_role", "development-host"},
                         {"os", host.os},
                         {"cpu", host.cpu_brand},
                         {"cpu_features", host.cpu_features},
                         {"logical_cpus", host.logical_cpus},
                         {"ram_bytes", host.ram_bytes}};
  doc_["started_at"] = iso8601_now();
  doc_["metrics"] = nlohmann::json::object();
  doc_["checks"] = nlohmann::json::array();
  doc_["pending_qualification"] = nlohmann::json::array();
}

void BenchmarkResult::set_host_role(const std::string& role, const std::string& machine_id) {
  doc_["environment"]["host_role"] = role;
  if (!machine_id.empty()) doc_["environment"]["machine_id"] = machine_id;
}

void BenchmarkResult::set_gpu(const std::string& name, const std::string& driver) {
  doc_["environment"]["gpu"] = name;
  if (!driver.empty()) doc_["environment"]["driver"] = driver;
}

void BenchmarkResult::mark_simulated(const std::string& key, nlohmann::json value) {
  doc_["simulated"][key] = std::move(value);
  provenance_ = "Synthetic";
}

void BenchmarkResult::check(const std::string& name, bool passed, const std::string& detail) {
  doc_["checks"].push_back({{"name", name}, {"passed", passed}, {"detail", detail}});
}

void BenchmarkResult::pending(const std::string& id) { doc_["pending_qualification"].push_back(id); }

void BenchmarkResult::backend(const std::string& name, const std::string& build) {
  doc_["tool"]["backend"] = name;
  doc_["tool"]["backend_build"] = build;
}

bool BenchmarkResult::all_checks_passed() const {
  for (const auto& c : doc_["checks"])
    if (!c["passed"].get<bool>()) return false;
  return true;
}

nlohmann::json BenchmarkResult::finish(double duration_s) {
  // Simulation markers override any attempt to label the run Measured.
  doc_["provenance"] = doc_.contains("simulated") ? "Synthetic" : provenance_;
  doc_["duration_s"] = duration_s;
  return doc_;
}

}  // namespace clusterlm::bench
