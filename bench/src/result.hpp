#pragma once
// BenchmarkResult: builder for bench/schema/benchmark-result.schema.json documents.
//
// Provenance is set from what actually happened: anything run with simulated network conditions, synthetic
// profiles, fixture models or a localhost cluster is Synthetic. The tool never emits "Qualified" — that
// status is assigned by review of Measured results produced on the declared target machines.
#include <nlohmann/json.hpp>

#include <string>
#include <vector>

#include "host_probe.hpp"

namespace clusterlm::bench {

struct Distribution {
  std::vector<double> samples;
  void add(double v) { samples.push_back(v); }
  nlohmann::json to_json(const std::string& unit) const;
};

class BenchmarkResult {
 public:
  BenchmarkResult(std::string experiment, const HostInfo& host);

  void set_measured() { provenance_ = "Measured"; }
  void mark_simulated(const std::string& key, nlohmann::json value);
  void config(const std::string& key, nlohmann::json value) { doc_["configuration"][key] = std::move(value); }
  void metric(const std::string& key, nlohmann::json value) { doc_["metrics"][key] = std::move(value); }
  void metric(const std::string& key, const Distribution& d, const std::string& unit) {
    doc_["metrics"][key] = d.to_json(unit);
  }
  void check(const std::string& name, bool passed, const std::string& detail = "");
  void pending(const std::string& qualification_id);
  void model(nlohmann::json m) { doc_["model"] = std::move(m); }
  void trace(nlohmann::json entry) { doc_["trace"].push_back(std::move(entry)); }
  void backend(const std::string& name, const std::string& build);
  bool all_checks_passed() const;

  nlohmann::json finish(double duration_s);

 private:
  nlohmann::json doc_;
  std::string provenance_ = "Synthetic";
};

std::string iso8601_now();

}  // namespace clusterlm::bench
