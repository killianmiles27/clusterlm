#include "framed.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <limits>
#include <mutex>

namespace clusterlm::transport {
namespace {

// Frames up to this size are coalesced with their header into a single write (one TLS record / TCP segment
// run) so small control messages cost one syscall.
constexpr std::size_t kCoalesceLimit = 64 * 1024;

// A received payload buffer grows in steps of this size as bytes actually arrive. Sizing it from the header's
// payload_len up front would let a peer that sends 24 header bytes pin max_payload (64 MiB by default) of
// zero-filled memory per connection; this keeps memory proportional to what the peer really delivered.
constexpr std::size_t kReceiveGrowth = 1u << 20;

class FramedConnection final : public Connection {
 public:
  FramedConnection(std::unique_ptr<Stream> stream, PeerIdentity peer, std::uint32_t max_payload)
      : stream_(std::move(stream)), peer_(std::move(peer)), max_payload_(max_payload) {}

  Status send(const Frame& f) override {
    if (f.payload.size() > max_payload_.load(std::memory_order_relaxed))
      return make_error(ErrorCode::kInvalidArgument, "frame payload exceeds max_payload");
    ByteWriter w(kFrameHeaderSize + (f.payload.size() <= kCoalesceLimit ? f.payload.size() : 0));
    w.u32(kFrameMagic);
    w.u16(kFrameVersion);
    w.u16(f.type);
    w.u8(f.channel);
    w.u8(f.flags);
    w.u16(0);
    w.u64(f.correlation);
    w.u32(static_cast<std::uint32_t>(f.payload.size()));
    const bool coalesce = f.payload.size() <= kCoalesceLimit;
    if (coalesce) w.raw(f.payload);
    const Bytes& wire = w.bytes();

    // One frame at a time: a partially written frame must never be interleaved with another sender's bytes.
    std::lock_guard<std::mutex> lk(send_mu_);
    if (closed_.load(std::memory_order_acquire)) return make_error(ErrorCode::kUnavailable, "connection closed");
    Status s = stream_->write_all(wire.data(), wire.size());
    if (s.is_ok() && !coalesce) s = stream_->write_all(f.payload.data(), f.payload.size());
    if (!s.is_ok()) {
      close();  // a half-written frame leaves the stream unparseable
      return make_error(ErrorCode::kUnavailable, "send failed: " + s.message());
    }
    bytes_sent_ += kFrameHeaderSize + f.payload.size();
    ++frames_sent_;
    return Status::ok();
  }

  Result<Frame> receive(std::chrono::milliseconds timeout) override {
    std::lock_guard<std::mutex> lk(recv_mu_);
    if (closed_.load(std::memory_order_acquire)) return make_error(ErrorCode::kUnavailable, "connection closed");
    const net::Deadline deadline = net::deadline_after(timeout);

    // State survives a timeout, so a frame split across the deadline is resumed, never lost.
    while (!have_header_) {
      std::size_t got = 0;
      Status s = stream_->read_some(hdr_.data() + hdr_got_, kFrameHeaderSize - hdr_got_, got, deadline);
      if (!s.is_ok()) return fail(std::move(s));
      hdr_got_ += got;
      if (hdr_got_ == kFrameHeaderSize) {
        Status v = parse_header();
        if (!v.is_ok()) return fail(std::move(v));
        have_header_ = true;
      }
    }
    while (payload_got_ < payload_len_) {
      if (payload_got_ == frame_.payload.size())
        frame_.payload.resize(std::min<std::size_t>(payload_len_, payload_got_ + kReceiveGrowth));
      std::size_t got = 0;
      Status s = stream_->read_some(frame_.payload.data() + payload_got_, frame_.payload.size() - payload_got_,
                                    got, deadline);
      if (!s.is_ok()) return fail(std::move(s));
      payload_got_ += got;
    }
    Frame out = std::move(frame_);
    frame_ = Frame{};
    have_header_ = false;
    hdr_got_ = 0;
    payload_got_ = 0;
    payload_len_ = 0;
    bytes_received_ += kFrameHeaderSize + out.payload.size();
    ++frames_received_;
    return out;
  }

  void close() override {
    closed_.store(true, std::memory_order_release);
    stream_->shutdown();
  }

  PeerIdentity peer() const override { return peer_; }

  ConnectionStats stats() const override {
    return {bytes_sent_.load(), bytes_received_.load(), frames_sent_.load(), frames_received_.load()};
  }

  void set_max_payload(std::uint32_t m) override { max_payload_.store(m, std::memory_order_relaxed); }
  Result<Bytes> export_keying_material(std::string_view label, std::size_t length) const override {
    return stream_->export_keying_material(label, length);
  }

 private:
  Status parse_header() {
    ByteReader r(ByteSpan(hdr_.data(), hdr_.size()));
    std::uint32_t magic = 0, len = 0;
    std::uint16_t version = 0, type = 0, reserved = 0;
    std::uint8_t channel = 0, flags = 0;
    std::uint64_t correlation = 0;
    r.u32(magic);
    r.u16(version);
    r.u16(type);
    r.u8(channel);
    r.u8(flags);
    r.u16(reserved);
    r.u64(correlation);
    r.u32(len);
    if (!r.ok()) return make_error(ErrorCode::kInternal, "frame header decode");
    if (magic != kFrameMagic) return make_error(ErrorCode::kProtocolError, "bad frame magic");
    if (version != kFrameVersion) return make_error(ErrorCode::kProtocolError, "unsupported frame version");
    if (reserved != 0) return make_error(ErrorCode::kProtocolError, "reserved header field is nonzero");
    // Validated before any allocation: a hostile length must not drive memory use.
    if (len > max_payload_.load(std::memory_order_relaxed))
      return make_error(ErrorCode::kProtocolError, "frame payload_len exceeds max_payload");
    frame_.type = type;
    frame_.channel = channel;
    frame_.flags = flags;
    frame_.correlation = correlation;
    payload_len_ = len;
    frame_.payload.clear();
    payload_got_ = 0;
    return Status::ok();
  }

  // Timeouts leave the connection usable; every other failure (including protocol violations) closes it.
  Result<Frame> fail(Status s) {
    if (s.code() != ErrorCode::kDeadlineExceeded) close();
    return s;
  }

  std::unique_ptr<Stream> stream_;
  const PeerIdentity peer_;
  std::atomic<std::uint32_t> max_payload_;
  std::atomic<bool> closed_{false};

  std::mutex send_mu_;
  std::mutex recv_mu_;
  std::array<std::uint8_t, kFrameHeaderSize> hdr_{};
  std::size_t hdr_got_ = 0;
  bool have_header_ = false;
  Frame frame_;
  std::size_t payload_got_ = 0;
  std::size_t payload_len_ = 0;

  std::atomic<std::uint64_t> bytes_sent_{0}, bytes_received_{0}, frames_sent_{0}, frames_received_{0};
};

}  // namespace

std::unique_ptr<Connection> make_framed_connection(std::unique_ptr<Stream> stream, PeerIdentity peer,
                                                   std::uint32_t max_payload) {
  return std::make_unique<FramedConnection>(std::move(stream), std::move(peer), max_payload);
}

}  // namespace clusterlm::transport
