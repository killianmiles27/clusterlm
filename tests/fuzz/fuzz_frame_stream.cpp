// Transport frame reader + MessageStream over an arbitrary byte stream. Input: byte 0 selects the payload
// limit and channel; the rest is the wire stream (24-byte headers + payloads, possibly truncated or hostile).
#include <chrono>
#include <memory>

#include "clusterlm/protocol/wire.hpp"
#include "clusterlm/transport/testing.hpp"
#include "fuzz_util.hpp"

using namespace clusterlm;

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  if (size == 0) return 0;
  static constexpr std::uint32_t kLimits[] = {64, 4096, 1u << 20, 64u << 20};
  const std::uint32_t max_payload = kLimits[data[0] & 3];
  const auto channel = static_cast<protocol::Channel>((data[0] >> 2) % 3);
  const Bytes wire(data + 1, data + size);

  // Raw framing: every received frame obeys the payload bound; the stream ends in an error, never a hang.
  {
    auto conn = transport::testing::make_byte_source_connection(wire, max_payload);
    for (int i = 0; i < 4096; ++i) {
      auto f = conn->receive(std::chrono::milliseconds(0));
      if (!f.is_ok()) break;
      CLM_FUZZ_CHECK(f->payload.size() <= max_payload);
    }
  }
  // Typed stream: frames are decoded with the protocol limits and channel-checked.
  {
    protocol::MessageStream stream(transport::testing::make_byte_source_connection(wire, max_payload), channel);
    for (int i = 0; i < 4096; ++i) {
      auto m = stream.receive(std::chrono::milliseconds(0));
      if (!m.is_ok()) break;
      CLM_FUZZ_CHECK(protocol::channel_of(protocol::type_of(m->message)) == channel);
    }
  }
  return 0;
}
