// ModelManifest::from_json (manifest.json on Father's disk; also the human/tooling form).
#include "clusterlm/objects/manifest.hpp"
#include "fuzz_util.hpp"

using namespace clusterlm;

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  auto m = objects::ModelManifest::from_json(fuzz::text_of(data, size));
  if (!m.is_ok()) return 0;
  CLM_FUZZ_CHECK(m->validate().is_ok());
  auto again = objects::ModelManifest::from_json(m->to_json());
  CLM_FUZZ_CHECK(again.is_ok());
  CLM_FUZZ_CHECK(again->root_hash() == m->root_hash());
  return 0;
}
