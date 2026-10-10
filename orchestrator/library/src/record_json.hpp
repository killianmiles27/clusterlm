#pragma once
// Internal: JSON value <-> ModelRecord, shared by the record and the library documents.
#include <nlohmann/json.hpp>

#include "clusterlm/library/model_record.hpp"

namespace clusterlm::library::detail {
nlohmann::json record_to_value(const ModelRecord& r);
Result<ModelRecord> record_from_value(const nlohmann::json& j);
}  // namespace clusterlm::library::detail
