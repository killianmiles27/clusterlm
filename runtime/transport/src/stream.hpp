#pragma once
// Internal byte-stream abstraction under the framing layer: plain TCP or TLS over a non-blocking socket.
#include <cstddef>
#include <memory>
#include <string_view>

#include "clusterlm/common/bytes.hpp"
#include "clusterlm/common/status.hpp"
#include "socket.hpp"

namespace clusterlm::transport {

class Stream {
 public:
  virtual ~Stream() = default;
  // Reads at least one byte (up to n). kDeadlineExceeded on timeout (stream stays usable), kUnavailable when
  // closed or the peer is gone, kUnauthenticated if the peer's rejection alert is received.
  virtual Status read_some(void* buf, std::size_t n, std::size_t& got, net::Deadline deadline) = 0;
  // Writes everything or fails; blocks until the peer drains or shutdown() is called.
  virtual Status write_all(const void* buf, std::size_t n) = 0;
  // Idempotent, any thread: wakes blocked I/O, which then fails with kUnavailable. The handle itself is only
  // closed on destruction.
  virtual void shutdown() = 0;
  // TLS exporter (see transport::Connection::export_keying_material).
  virtual Result<Bytes> export_keying_material(std::string_view /*label*/, std::size_t /*length*/) const {
    return make_error(ErrorCode::kUnimplemented, "not a TLS stream");
  }
};

std::unique_ptr<Stream> make_plain_stream(net::Socket socket);

}  // namespace clusterlm::transport
