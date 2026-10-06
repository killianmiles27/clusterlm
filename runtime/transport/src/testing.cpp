#include "clusterlm/transport/testing.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>

#include "framed.hpp"

namespace clusterlm::transport::testing {
namespace {

class ByteSourceStream final : public Stream {
 public:
  explicit ByteSourceStream(Bytes data) : data_(std::move(data)) {}

  Status read_some(void* buf, std::size_t n, std::size_t& got, net::Deadline) override {
    if (closed_.load(std::memory_order_acquire) || pos_ >= data_.size())
      return make_error(ErrorCode::kUnavailable, "peer closed connection");
    got = std::min(n, data_.size() - pos_);
    std::memcpy(buf, data_.data() + pos_, got);
    pos_ += got;
    return Status::ok();
  }
  Status write_all(const void*, std::size_t) override { return Status::ok(); }
  void shutdown() override { closed_.store(true, std::memory_order_release); }

 private:
  Bytes data_;
  std::size_t pos_ = 0;
  std::atomic<bool> closed_{false};
};

}  // namespace

std::unique_ptr<Connection> make_byte_source_connection(Bytes input, std::uint32_t max_payload) {
  return make_framed_connection(std::make_unique<ByteSourceStream>(std::move(input)),
                                PeerIdentity{"byte-source", false, "memory"}, max_payload);
}

}  // namespace clusterlm::transport::testing
