// protocol::decode for every MessageType. Input: u16 LE message type, then the payload. A successfully decoded
// message must re-encode to a payload that decodes to the same encoding (canonical form is a fixed point).
#include "clusterlm/protocol/messages.hpp"
#include "fuzz_util.hpp"

using namespace clusterlm;

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  if (size < 2) return 0;
  const auto type = static_cast<protocol::MessageType>(static_cast<std::uint16_t>(data[0] | (data[1] << 8)));
  auto m = protocol::decode(type, fuzz::span_of(data + 2, size - 2));
  (void)protocol::to_string(type);
  if (!m.is_ok()) return 0;
  CLM_FUZZ_CHECK(protocol::type_of(m.value()) == type);
  (void)protocol::channel_of(type);
  const Bytes e1 = protocol::encode(m.value());
  auto m2 = protocol::decode(type, e1);
  CLM_FUZZ_CHECK(m2.is_ok());
  CLM_FUZZ_CHECK(protocol::encode(m2.value()) == e1);
  return 0;
}
