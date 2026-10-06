#include "schema_check.hpp"

#include <cmath>
#include <fstream>

namespace clusterlm::bench {

using nlohmann::json;

namespace {

bool type_matches(const json& v, const std::string& t) {
  if (t == "object") return v.is_object();
  if (t == "array") return v.is_array();
  if (t == "string") return v.is_string();
  if (t == "boolean") return v.is_boolean();
  if (t == "number") return v.is_number();
  if (t == "integer") return v.is_number_integer() || (v.is_number_float() && std::floor(v.get<double>()) == v.get<double>());
  if (t == "null") return v.is_null();
  return true;
}

void check(const json& v, const json& schema, const std::string& path, std::vector<std::string>& errors) {
  if (!schema.is_object()) return;
  if (schema.contains("const") && v != schema["const"]) errors.push_back(path + ": expected constant " + schema["const"].dump());
  if (schema.contains("enum")) {
    bool found = false;
    for (const auto& e : schema["enum"]) found |= (e == v);
    if (!found) errors.push_back(path + ": value " + v.dump() + " not in " + schema["enum"].dump());
  }
  if (schema.contains("type")) {
    bool ok = false;
    if (schema["type"].is_array()) {
      for (const auto& t : schema["type"]) ok |= type_matches(v, t.get<std::string>());
    } else {
      ok = type_matches(v, schema["type"].get<std::string>());
    }
    if (!ok) {
      errors.push_back(path + ": expected type " + schema["type"].dump() + ", got " + v.type_name());
      return;
    }
  }
  if (schema.contains("oneOf")) {
    int matches = 0;
    for (const auto& alt : schema["oneOf"]) {
      std::vector<std::string> sub;
      check(v, alt, path, sub);
      matches += sub.empty();
    }
    if (matches == 0) errors.push_back(path + ": matches none of the allowed forms");
    else if (matches > 1) {
      // JSON Schema's exactly-one rule: integers are also numbers, so boolean/number/string stay exclusive, but an
      // object that fits more than one alternative is ambiguous.
      errors.push_back(path + ": matches more than one of the allowed forms");
    }
  }
  if (v.is_object()) {
    if (schema.contains("required"))
      for (const auto& r : schema["required"])
        if (!v.contains(r.get<std::string>())) errors.push_back(path + ": missing required key '" + r.get<std::string>() + "'");
    const json props = schema.contains("properties") ? schema["properties"] : json::object();
    for (auto it = v.begin(); it != v.end(); ++it) {
      const std::string sub = path + "/" + it.key();
      if (props.contains(it.key())) {
        check(it.value(), props[it.key()], sub, errors);
      } else if (schema.contains("additionalProperties")) {
        const json& ap = schema["additionalProperties"];
        if (ap.is_boolean()) {
          if (!ap.get<bool>()) errors.push_back(sub + ": additional property not allowed");
        } else {
          check(it.value(), ap, sub, errors);
        }
      }
    }
  }
  if (v.is_array() && schema.contains("items"))
    for (std::size_t i = 0; i < v.size(); ++i) check(v[i], schema["items"], path + "/" + std::to_string(i), errors);
}

}  // namespace

std::vector<std::string> validate_against_schema(const json& document, const json& schema) {
  std::vector<std::string> errors;
  check(document, schema, "", errors);
  return errors;
}

std::vector<std::string> validate_result_semantics(const json& r) {
  std::vector<std::string> errors;
  const std::string prov = r.value("provenance", std::string());
  if (prov == "Qualified") errors.push_back("provenance: the tool never emits Qualified");
  if (r.contains("simulated") && prov != "Synthetic") errors.push_back("provenance: simulated conditions require Synthetic");
  if (prov == "Measured") {
    if (!r.contains("environment") || !r["environment"].contains("host_role"))
      errors.push_back("environment.host_role: required for a Measured result");
  }
  if (r.contains("checks"))
    for (const auto& c : r["checks"])
      if (!c.contains("name") || !c.contains("passed")) errors.push_back("checks: every entry needs name and passed");
  return errors;
}

json load_result_schema(const std::string& source_dir) {
  std::ifstream f(source_dir + "/bench/schema/benchmark-result.schema.json");
  if (!f) return json();
  return json::parse(f, nullptr, false);
}

}  // namespace clusterlm::bench
