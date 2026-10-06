#pragma once
// Test and fuzzing seams for the framing layer. Not used by product code.
#include <cstdint>
#include <memory>

#include "clusterlm/common/bytes.hpp"
#include "clusterlm/transport/transport.hpp"

namespace clusterlm::transport::testing {

// A Connection whose receive() runs the real framed reader (header validation, payload bound, resumable
// partial frames) over `input`, then reports kUnavailable once the bytes are exhausted. send() discards.
std::unique_ptr<Connection> make_byte_source_connection(Bytes input, std::uint32_t max_payload = 64u * 1024u * 1024u);

}  // namespace clusterlm::transport::testing
