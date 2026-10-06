#include "stream.hpp"

#include <atomic>

namespace clusterlm::transport {
namespace {

class PlainStream final : public Stream {
 public:
  explicit PlainStream(net::Socket s) : sock_(std::move(s)) {}

  Status read_some(void* buf, std::size_t n, std::size_t& got, net::Deadline deadline) override {
    for (;;) {
      if (closed_.load(std::memory_order_acquire)) return make_error(ErrorCode::kUnavailable, "connection closed");
      const net::IoResult r = net::raw_recv(sock_.handle(), buf, n);
      if (r.state == net::IoState::kOk) {
        got = r.n;
        return Status::ok();
      }
      if (r.state != net::IoState::kWouldBlock) return make_error(ErrorCode::kUnavailable, "peer closed connection");
      switch (net::wait_ready(sock_.handle(), false, deadline, &closed_)) {
        case net::WaitResult::kReady: break;
        case net::WaitResult::kTimeout: return make_error(ErrorCode::kDeadlineExceeded, "receive timed out");
        default: return make_error(ErrorCode::kUnavailable, "connection closed");
      }
    }
  }

  Status write_all(const void* buf, std::size_t n) override {
    const auto* p = static_cast<const std::uint8_t*>(buf);
    while (n > 0) {
      if (closed_.load(std::memory_order_acquire)) return make_error(ErrorCode::kUnavailable, "connection closed");
      const net::IoResult r = net::raw_send(sock_.handle(), p, n);
      if (r.state == net::IoState::kOk) {
        p += r.n;
        n -= r.n;
        continue;
      }
      if (r.state != net::IoState::kWouldBlock) return make_error(ErrorCode::kUnavailable, "send failed: peer gone");
      if (net::wait_ready(sock_.handle(), true, net::no_deadline(), &closed_) != net::WaitResult::kReady)
        return make_error(ErrorCode::kUnavailable, "connection closed");
    }
    return Status::ok();
  }

  void shutdown() override {
    closed_.store(true, std::memory_order_release);
    sock_.shutdown_both();
  }

 private:
  net::Socket sock_;
  std::atomic<bool> closed_{false};
};

}  // namespace

std::unique_ptr<Stream> make_plain_stream(net::Socket socket) { return std::make_unique<PlainStream>(std::move(socket)); }

}  // namespace clusterlm::transport
