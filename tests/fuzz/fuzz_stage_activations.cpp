// StageActivations::decode (the boundary-ABI payload). Input: byte 0 selects max_positions, rest is the encoding.
#include "clusterlm/domain/boundary.hpp"
#include "fuzz_util.hpp"

using namespace clusterlm;

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  if (size == 0) return 0;
  static constexpr std::uint32_t kMax[] = {1, 4, 64, 4096};
  const std::uint32_t max_positions = kMax[data[0] & 3];
  ByteReader r(fuzz::span_of(data + 1, size - 1));
  auto a = domain::StageActivations::decode(r, max_positions);
  if (!a.is_ok()) return 0;
  CLM_FUZZ_CHECK(a->positions <= max_positions);
  CLM_FUZZ_CHECK(a->validate().is_ok());
  CLM_FUZZ_CHECK(a->data.size() == std::uint64_t{a->positions} * a->layout.floats_per_position());
  ByteWriter w;
  a->encode(w);
  ByteReader r2(w.bytes());
  auto b = domain::StageActivations::decode(r2, max_positions);
  CLM_FUZZ_CHECK(b.is_ok());
  ByteWriter w2;
  b->encode(w2);
  CLM_FUZZ_CHECK(w.bytes() == w2.bytes());
  return 0;
}
