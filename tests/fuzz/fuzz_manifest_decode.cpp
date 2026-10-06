// ModelManifest::decode (binary form carried in PreparePlan). A manifest that decodes passed validate().
#include "clusterlm/objects/manifest.hpp"
#include "fuzz_util.hpp"

using namespace clusterlm;

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  ByteReader r(fuzz::span_of(data, size));
  auto m = objects::ModelManifest::decode(r);
  if (!m.is_ok()) return 0;
  CLM_FUZZ_CHECK(m->validate().is_ok());
  (void)m->root_hash();
  (void)m->total_bytes(objects::LayerRange{0, m->geometry.n_layers});
  ByteWriter w;
  m->encode(w);
  ByteReader r2(w.bytes());
  auto m2 = objects::ModelManifest::decode(r2);
  CLM_FUZZ_CHECK(m2.is_ok());
  CLM_FUZZ_CHECK(m2->root_hash() == m->root_hash());
  return 0;
}
