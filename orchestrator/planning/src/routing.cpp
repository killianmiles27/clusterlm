// Aggregate routing statistics loader. Privacy: the schema is a closed allowlist of aggregate fields.
#include <cmath>
#include <fstream>
#include <set>
#include <sstream>

#include <nlohmann/json.hpp>

#include "clusterlm/planning/planning.hpp"

namespace clusterlm::planning {

using nlohmann::json;

namespace {

Status bad(const std::string& what) {
  return make_error(ErrorCode::kInvalidArgument, "routing aggregates: " + what);
}

Result<std::vector<std::vector<double>>> read_matrix(const json& j, const char* name, std::uint32_t layers, std::uint32_t experts) {
  if (!j.is_array() || j.size() != layers) return bad(std::string(name) + " must have n_layers rows");
  std::vector<std::vector<double>> m;
  m.reserve(layers);
  for (const auto& row : j) {
    if (!row.is_array() || row.size() != experts) return bad(std::string(name) + " rows must have n_experts entries");
    std::vector<double> r;
    r.reserve(experts);
    for (const auto& v : row) {
      if (!v.is_number()) return bad(std::string(name) + " entries must be numbers");
      const double d = v.get<double>();
      if (!std::isfinite(d) || d < 0) return bad(std::string(name) + " entries must be finite and >= 0");
      r.push_back(d);
    }
    m.push_back(std::move(r));
  }
  return m;
}

}  // namespace

Result<RoutingAggregates> routing_aggregates_from_json(std::string_view text) {
  json j = json::parse(text, nullptr, /*allow_exceptions=*/false);
  if (j.is_discarded() || !j.is_object()) return bad("not a JSON object");

  static const std::set<std::string> kAllowed = {"schema",    "provenance", "source",      "n_layers", "n_experts",
                                                 "n_active",  "positions",  "frequencies", "counts",   "note"};
  for (auto it = j.begin(); it != j.end(); ++it)
    if (!kAllowed.count(it.key()))
      return bad("unknown field '" + it.key() + "': only aggregate routing frequencies are accepted (no tokens, sequences, prompts or per-request routing)");
  if (j.value("schema", "") != "clusterlm.routing_aggregates.v1") return bad("schema must be clusterlm.routing_aggregates.v1");

  RoutingAggregates a;
  if (!j.contains("provenance") || !j["provenance"].is_string() ||
      !placement::provenance_from_string(j["provenance"].get<std::string>(), a.provenance))
    return bad("provenance must be stated as \"synthetic\" or \"measured\"");
  if (a.provenance == placement::Provenance::kQualified)
    return bad("a file cannot claim \"qualified\" provenance; qualification is a separate explicit step");
  if (!j.contains("source") || !j["source"].is_string() || j["source"].get<std::string>().empty()) return bad("source must be stated");
  a.source = j["source"].get<std::string>();
  for (const char* k : {"n_layers", "n_experts", "n_active"})
    if (!j.contains(k) || !j[k].is_number_unsigned() || j[k].get<std::uint64_t>() == 0) return bad(std::string(k) + " must be a positive integer");
  a.n_layers = j["n_layers"].get<std::uint32_t>();
  a.n_experts = j["n_experts"].get<std::uint32_t>();
  a.n_active = j["n_active"].get<std::uint32_t>();
  if (a.n_active > a.n_experts) return bad("n_active exceeds n_experts");
  const bool has_f = j.contains("frequencies"), has_c = j.contains("counts");
  if (has_f == has_c) return bad("exactly one of frequencies / counts must be given");
  if (j.contains("positions")) {
    if (!j["positions"].is_number_unsigned()) return bad("positions must be a non-negative integer");
    a.positions = j["positions"].get<std::uint64_t>();
  }

  const double k = static_cast<double>(a.n_active);
  if (has_c) {
    if (a.positions == 0) return bad("counts require positions > 0");
    auto m = read_matrix(j["counts"], "counts", a.n_layers, a.n_experts);
    if (!m.is_ok()) return m.status();
    const double pos = static_cast<double>(a.positions);
    for (std::size_t l = 0; l < m->size(); ++l) {
      double sum = 0;
      for (double& c : (*m)[l]) {
        c /= pos;
        if (c > 1.0 + 1e-9) return bad("count exceeds positions in layer " + std::to_string(l));
        sum += c;
      }
      if (std::abs(sum - k) > 1e-6 * k) return bad("counts of layer " + std::to_string(l) + " do not sum to positions * n_active");
    }
    a.frequencies = std::move(*m);
  } else {
    auto m = read_matrix(j["frequencies"], "frequencies", a.n_layers, a.n_experts);
    if (!m.is_ok()) return m.status();
    for (std::size_t l = 0; l < m->size(); ++l) {
      double sum = 0;
      for (double v : (*m)[l]) {
        if (v > 1.0 + 1e-9) return bad("frequency above 1 in layer " + std::to_string(l));
        sum += v;
      }
      if (std::abs(sum - k) > 1e-3 * k) return bad("frequencies of layer " + std::to_string(l) + " do not sum to n_active");
      for (double& v : (*m)[l]) v = std::min(1.0, v * (k / sum));  // exact renormalisation of print rounding
    }
    a.frequencies = std::move(*m);
  }
  return a;
}

Result<RoutingAggregates> load_routing_aggregates(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return make_error(ErrorCode::kNotFound, "cannot open routing aggregates " + path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return routing_aggregates_from_json(ss.str());
}

std::string to_json(const RoutingAggregates& a, int indent) {
  json j;
  j["schema"] = "clusterlm.routing_aggregates.v1";
  j["provenance"] = std::string(placement::to_string(a.provenance));
  j["source"] = a.source;
  j["n_layers"] = a.n_layers;
  j["n_experts"] = a.n_experts;
  j["n_active"] = a.n_active;
  if (a.positions > 0) j["positions"] = a.positions;
  j["frequencies"] = a.frequencies;
  return j.dump(indent);
}

void apply_routing_aggregates(CostInputOptions& options, const RoutingAggregates& a) {
  options.routing_freq = a.frequencies;
  options.routing_provenance = a.provenance;
  options.routing_source = a.source;
}

}  // namespace clusterlm::planning
