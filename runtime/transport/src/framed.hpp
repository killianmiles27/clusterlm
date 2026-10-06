#pragma once
// Internal: Connection implementation that frames messages over a Stream.
#include <memory>

#include "clusterlm/transport/transport.hpp"
#include "stream.hpp"

namespace clusterlm::transport {

std::unique_ptr<Connection> make_framed_connection(std::unique_ptr<Stream> stream, PeerIdentity peer,
                                                   std::uint32_t max_payload);

}  // namespace clusterlm::transport
