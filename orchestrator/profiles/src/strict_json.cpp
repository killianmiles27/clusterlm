#include "strict_json.hpp"

#include "clusterlm/profiles/profile.hpp"

namespace clusterlm::profiles::detail {

Result<nlohmann::json> parse_object_document(std::string_view text) {
  if (text.size() > kMaxDocumentBytes) return bad("document larger than " + std::to_string(kMaxDocumentBytes) + " bytes");
  if (!depth_within(text, kMaxJsonDepth)) return bad("JSON nesting deeper than " + std::to_string(kMaxJsonDepth));
  nlohmann::json doc = nlohmann::json::parse(text.begin(), text.end(), nullptr, /*allow_exceptions=*/false);
  if (doc.is_discarded() || !doc.is_object()) return bad("not a JSON object");
  return doc;
}

}  // namespace clusterlm::profiles::detail
