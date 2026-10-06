// Tier catalog JSON (Catalog::parse): bounded size/depth, strict validation, no exceptions escaping.
#include "clusterlm/catalog/catalog.hpp"
#include "fuzz_util.hpp"

using namespace clusterlm;

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  auto c = catalog::Catalog::parse(fuzz::text_of(data, size));
  if (!c.is_ok()) return 0;
  CLM_FUZZ_CHECK(c->tiers().size() == 3);
  for (const auto& t : c->tiers()) CLM_FUZZ_CHECK(catalog::validate_tier(t).is_ok());
  (void)c->fallbacks_after("ultra");
  return 0;
}
