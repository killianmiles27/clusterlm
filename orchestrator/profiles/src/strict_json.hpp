#pragma once
// Internal: profile-document helpers on top of the shared strict reader.
#include "clusterlm/common/strict_json.hpp"

namespace clusterlm::profiles::detail {
using strict::bad;
using strict::depth_within;
using strict::Obj;
// Bounded parse of a whole document into an object.
Result<nlohmann::json> parse_object_document(std::string_view text);
}  // namespace clusterlm::profiles::detail
