#pragma once
// A validator for the JSON Schema subset used by bench/schema/benchmark-result.schema.json (type, const, enum,
// required, properties, additionalProperties-as-schema, items, oneOf, format ignored), plus the semantic rules the
// schema cannot express. No third-party dependency.
#include <nlohmann/json.hpp>

#include <string>
#include <vector>

namespace clusterlm::bench {

// Returns one human-readable message per violation ("/metrics/x: expected number"); empty means valid.
std::vector<std::string> validate_against_schema(const nlohmann::json& document, const nlohmann::json& schema);

// Rules beyond the schema: the tool never emits Qualified; any simulated condition forces Synthetic; a Measured
// result names its machine; check entries are well formed.
std::vector<std::string> validate_result_semantics(const nlohmann::json& result);

// Loads <source_dir>/bench/schema/benchmark-result.schema.json.
nlohmann::json load_result_schema(const std::string& source_dir);

}  // namespace clusterlm::bench
