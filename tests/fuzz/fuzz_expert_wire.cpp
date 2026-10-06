// EXPERIMENTAL expert-domain wire messages (experimental/expert-domains). Byte 0 selects the message.
#include "clusterlm/expert_domains/wire.hpp"
#include "fuzz_util.hpp"

using namespace clusterlm;

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  if (size == 0) return 0;
  const expert_domains::ExpertDecodeLimits limits;
  const ByteSpan payload = fuzz::span_of(data + 1, size - 1);
  switch (data[0] % 3) {
    case 0: {
      auto m = expert_domains::decode_batch(payload, limits);
      if (m.is_ok()) {
        const Bytes e = expert_domains::encode(m.value());
        auto again = expert_domains::decode_batch(e, limits);
        CLM_FUZZ_CHECK(again.is_ok() && expert_domains::encode(again.value()) == e);
      }
      break;
    }
    case 1: {
      auto m = expert_domains::decode_result(payload, limits);
      if (m.is_ok()) {
        const Bytes e = expert_domains::encode(m.value());
        auto again = expert_domains::decode_result(e, limits);
        CLM_FUZZ_CHECK(again.is_ok() && expert_domains::encode(again.value()) == e);
      }
      break;
    }
    default: (void)expert_domains::decode_error(payload, limits); break;
  }
  return 0;
}
