#include <atomic>
#include <cstring>
#include <thread>
#include <vector>

#include "test_helpers.hpp"

#ifndef _WIN32
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

using namespace clm_test;

namespace {

#ifndef _WIN32
// Plain POSIX client used to feed the listener hand-crafted bytes (test code only; Linux CI).
int raw_connect(std::uint16_t port) {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in sa{};
  sa.sin_family = AF_INET;
  sa.sin_port = htons(port);
  inet_pton(AF_INET, "127.0.0.1", &sa.sin_addr);
  REQUIRE(::connect(fd, reinterpret_cast<sockaddr*>(&sa), sizeof sa) == 0);
  return fd;
}

void raw_send_all(int fd, const std::vector<std::uint8_t>& bytes) {
  std::size_t off = 0;
  while (off < bytes.size()) {
    const auto n = ::send(fd, bytes.data() + off, bytes.size() - off, MSG_NOSIGNAL);
    REQUIRE(n > 0);
    off += static_cast<std::size_t>(n);
  }
}

std::vector<std::uint8_t> header_bytes(std::uint32_t magic, std::uint16_t version, std::uint16_t type,
                                       std::uint16_t reserved, std::uint64_t corr, std::uint32_t len) {
  ByteWriter w;
  w.u32(magic);
  w.u16(version);
  w.u16(type);
  w.u8(1);
  w.u8(0);
  w.u16(reserved);
  w.u64(corr);
  w.u32(len);
  return w.bytes();
}

struct RawPeer {
  std::unique_ptr<Listener> listener;
  std::unique_ptr<Connection> server;
  int fd = -1;
  RawPeer() = default;
  RawPeer(RawPeer&& o) noexcept
      : listener(std::move(o.listener)), server(std::move(o.server)), fd(o.fd) {
    o.fd = -1;
  }
  ~RawPeer() {
    if (fd >= 0) ::close(fd);
  }
};

RawPeer raw_peer() {
  RawPeer r;
  auto l = listen(Endpoint{"127.0.0.1", 0}, insecure());
  REQUIRE(l.is_ok());
  r.listener = std::move(l).value();
  r.fd = raw_connect(r.listener->local_endpoint().port);
  auto a = r.listener->accept(5s);
  REQUIRE(a.is_ok());
  r.server = std::move(a).value();
  return r;
}
#endif

}  // namespace

TEST_CASE("endpoint parsing") {
  auto e = Endpoint::parse("127.0.0.1:8080");
  REQUIRE(e.is_ok());
  CHECK(e->host == "127.0.0.1");
  CHECK(e->port == 8080);
  CHECK(e->str() == "127.0.0.1:8080");
  CHECK(e->is_loopback());
  auto v6 = Endpoint::parse("[::1]:9");
  REQUIRE(v6.is_ok());
  CHECK(v6->host == "::1");
  CHECK(v6->str() == "[::1]:9");
  CHECK(v6->is_loopback());
  CHECK_FALSE(Endpoint::parse("192.168.1.5:80")->is_loopback());
  CHECK(Endpoint::parse("localhost:1")->is_loopback());
  CHECK_FALSE(Endpoint::parse("nocolon").is_ok());
  CHECK_FALSE(Endpoint::parse("host:99999").is_ok());
  CHECK_FALSE(Endpoint::parse("host:").is_ok());
  CHECK_FALSE(Endpoint::parse(":80").is_ok());
}

TEST_CASE("frames round trip over loopback") {
  Pair p = insecure_pair();
  CHECK_FALSE(p.client->peer().authenticated);
  CHECK(p.client->peer().device_id == "insecure-loopback");

  SUBCASE("several frames and an empty payload preserve every header field") {
    Frame a = make_frame(7, 100, 0x11, 0x0102030405060708ull);
    a.channel = 9;
    a.flags = 3;
    Frame b = make_frame(8, 0, 0, 42);
    REQUIRE(p.client->send(a).is_ok());
    REQUIRE(p.client->send(b).is_ok());
    REQUIRE(p.client->send(make_frame(9, 1, 0xFF)).is_ok());
    auto ra = p.server->receive(2s);
    REQUIRE(ra.is_ok());
    CHECK(ra->type == 7);
    CHECK(ra->channel == 9);
    CHECK(ra->flags == 3);
    CHECK(ra->correlation == 0x0102030405060708ull);
    CHECK(ra->payload == a.payload);
    auto rb = p.server->receive(2s);
    REQUIRE(rb.is_ok());
    CHECK(rb->type == 8);
    CHECK(rb->payload.empty());
    CHECK(rb->correlation == 42);
    auto rc = p.server->receive(2s);
    REQUIRE(rc.is_ok());
    CHECK(rc->payload.size() == 1);
    CHECK(p.client->stats().frames_sent == 3);
    CHECK(p.server->stats().frames_received == 3);
    CHECK(p.client->stats().bytes_sent == 3 * kFrameHeaderSize + 101);
    CHECK(p.server->stats().bytes_received == p.client->stats().bytes_sent);
  }

  SUBCASE("4 MiB payload, both directions") {
    Frame big = make_frame(1, 4u << 20);
    for (std::size_t i = 0; i < big.payload.size(); ++i) big.payload[i] = static_cast<std::uint8_t>(i * 31 + 7);
    auto sender = std::async(std::launch::async, [&] { return p.client->send(big); });
    auto r = p.server->receive(5s);
    REQUIRE(r.is_ok());
    CHECK(sender.get().is_ok());
    CHECK(r->payload == big.payload);
    REQUIRE(p.server->send(make_frame(2, 10)).is_ok());
    auto back = p.client->receive(2s);
    REQUIRE(back.is_ok());
    CHECK(back->type == 2);
  }
}

TEST_CASE("concurrent senders never interleave frames") {
  Pair p = insecure_pair();
  constexpr int kThreads = 4;
  constexpr int kPerThread = 40;
  constexpr std::size_t kSize = 200 * 1024;  // larger than socket buffers: forces partial writes
  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t] {
      for (int i = 0; i < kPerThread; ++i) {
        Frame f = make_frame(static_cast<std::uint16_t>(t), kSize, static_cast<std::uint8_t>(t + 1),
                             static_cast<std::uint64_t>(i));
        CHECK(p.client->send(f).is_ok());
      }
    });
  }
  std::vector<int> next(kThreads, 0);
  for (int n = 0; n < kThreads * kPerThread; ++n) {
    auto r = p.server->receive(10s);
    REQUIRE(r.is_ok());
    REQUIRE(r->type < kThreads);
    REQUIRE(r->payload.size() == kSize);
    const auto expect = static_cast<std::uint8_t>(r->type + 1);
    bool uniform = true;
    for (std::size_t i = 0; i < kSize; i += 4093) uniform = uniform && r->payload[i] == expect;
    uniform = uniform && r->payload.back() == expect;
    CHECK(uniform);
    CHECK(r->correlation == static_cast<std::uint64_t>(next[r->type]++));  // per-sender order preserved
  }
  for (auto& th : threads) th.join();
}

TEST_CASE("receive timeout leaves the connection usable") {
  Pair p = insecure_pair();
  const auto t0 = std::chrono::steady_clock::now();
  auto r = p.server->receive(60ms);
  CHECK(r.status().code() == ErrorCode::kDeadlineExceeded);
  CHECK(std::chrono::steady_clock::now() - t0 >= 55ms);
  REQUIRE(p.client->send(make_frame(5, 16)).is_ok());
  auto r2 = p.server->receive(2s);
  REQUIRE(r2.is_ok());
  CHECK(r2->type == 5);
}

TEST_CASE("close() from another thread unblocks receive promptly") {
  Pair p = insecure_pair();
  std::atomic<bool> returned{false};
  Status result;
  std::thread reader([&] {
    result = p.server->receive(30s).status();
    returned = true;
  });
  std::this_thread::sleep_for(50ms);
  CHECK_FALSE(returned.load());
  const auto t0 = std::chrono::steady_clock::now();
  p.server->close();
  reader.join();
  CHECK(std::chrono::steady_clock::now() - t0 < 200ms);
  CHECK(result.code() == ErrorCode::kUnavailable);
  CHECK(p.server->send(make_frame(1, 1)).code() == ErrorCode::kUnavailable);
  // The peer observes the closure.
  CHECK(p.client->receive(2s).status().code() == ErrorCode::kUnavailable);
}

TEST_CASE("listener close() unblocks accept") {
  auto l = listen(Endpoint{"127.0.0.1", 0}, insecure());
  REQUIRE(l.is_ok());
  auto lst = std::move(l).value();
  auto fut = std::async(std::launch::async, [&] { return lst->accept(30s); });
  std::this_thread::sleep_for(50ms);
  const auto t0 = std::chrono::steady_clock::now();
  lst->close();
  auto r = fut.get();
  CHECK(std::chrono::steady_clock::now() - t0 < 200ms);
  CHECK(r.status().code() == ErrorCode::kUnavailable);
}

TEST_CASE("accept times out") {
  auto l = listen(Endpoint{"127.0.0.1", 0}, insecure());
  REQUIRE(l.is_ok());
  CHECK((*l)->accept(40ms).status().code() == ErrorCode::kDeadlineExceeded);
  CHECK((*l)->local_endpoint().port != 0);
}

TEST_CASE("insecure mode refuses non-loopback addresses") {
  auto l = listen(Endpoint{"0.0.0.0", 0}, insecure());
  CHECK(l.status().code() == ErrorCode::kPermissionDenied);
  auto l2 = listen(Endpoint{"192.168.1.10", 0}, insecure());
  CHECK(l2.status().code() == ErrorCode::kPermissionDenied);
  auto c = connect(Endpoint{"192.0.2.1", 80}, insecure(), std::nullopt, 100ms);
  CHECK(c.status().code() == ErrorCode::kPermissionDenied);
}

TEST_CASE("connect to a closed port fails") {
  auto l = listen(Endpoint{"127.0.0.1", 0}, insecure());
  REQUIRE(l.is_ok());
  const Endpoint ep = (*l)->local_endpoint();
  (*l)->close();
  auto c = connect(ep, insecure(), std::nullopt, 1s);
  CHECK_FALSE(c.is_ok());
}

TEST_CASE("sender rejects payloads over max_payload") {
  Pair p = insecure_pair();
  p.client->set_max_payload(100);
  CHECK(p.client->send(make_frame(1, 101)).code() == ErrorCode::kInvalidArgument);
  CHECK(p.client->send(make_frame(1, 100)).is_ok());
}

#ifndef _WIN32
TEST_CASE("wire header layout is explicit little-endian") {
  RawPeer r = raw_peer();
  Frame f;
  f.type = 0x1234;
  f.channel = 7;
  f.flags = 3;
  f.correlation = 0x0102030405060708ull;
  f.payload = {0xAA, 0xBB, 0xCC};
  REQUIRE(r.server->send(f).is_ok());
  std::uint8_t buf[27];
  std::size_t got = 0;
  while (got < sizeof buf) {
    const auto n = ::recv(r.fd, buf + got, sizeof buf - got, 0);
    REQUIRE(n > 0);
    got += static_cast<std::size_t>(n);
  }
  const std::uint8_t expect[27] = {0x43, 0x4C, 0x4D, 0x46,  // "CLMF" as LE u32 0x464D4C43
                                   0x01, 0x00,              // version
                                   0x34, 0x12,              // type
                                   0x07, 0x03,              // channel, flags
                                   0x00, 0x00,              // reserved
                                   0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01,
                                   0x03, 0x00, 0x00, 0x00,  // payload_len
                                   0xAA, 0xBB, 0xCC};
  CHECK(std::memcmp(buf, expect, sizeof buf) == 0);
}

TEST_CASE("malformed headers are rejected and close the connection") {
  RawPeer r = raw_peer();
  SUBCASE("bad magic") {
    raw_send_all(r.fd, header_bytes(0xDEADBEEF, 1, 1, 0, 0, 0));
    CHECK(r.server->receive(2s).status().code() == ErrorCode::kProtocolError);
    CHECK(r.server->receive(100ms).status().code() == ErrorCode::kUnavailable);
  }
  SUBCASE("bad version") {
    raw_send_all(r.fd, header_bytes(kFrameMagic, 2, 1, 0, 0, 0));
    CHECK(r.server->receive(2s).status().code() == ErrorCode::kProtocolError);
  }
  SUBCASE("nonzero reserved") {
    raw_send_all(r.fd, header_bytes(kFrameMagic, 1, 1, 5, 0, 0));
    CHECK(r.server->receive(2s).status().code() == ErrorCode::kProtocolError);
  }
  SUBCASE("hostile payload_len is rejected before any payload arrives or is allocated") {
    raw_send_all(r.fd, header_bytes(kFrameMagic, 1, 1, 0, 0, 0xFFFFFFFFu));
    CHECK(r.server->receive(2s).status().code() == ErrorCode::kProtocolError);
  }
  SUBCASE("a lowered limit applies to later frames") {
    r.server->set_max_payload(10);
    auto h = header_bytes(kFrameMagic, 1, 1, 0, 0, 11);
    h.resize(h.size() + 11, 0);
    raw_send_all(r.fd, h);
    CHECK(r.server->receive(2s).status().code() == ErrorCode::kProtocolError);
  }
}

TEST_CASE("a frame split across a receive timeout is resumed, not lost") {
  RawPeer r = raw_peer();
  auto bytes = header_bytes(kFrameMagic, 1, 77, 0, 5, 8);
  bytes.insert(bytes.end(), {1, 2, 3, 4, 5, 6, 7, 8});
  raw_send_all(r.fd, std::vector<std::uint8_t>(bytes.begin(), bytes.begin() + 10));
  CHECK(r.server->receive(50ms).status().code() == ErrorCode::kDeadlineExceeded);
  raw_send_all(r.fd, std::vector<std::uint8_t>(bytes.begin() + 10, bytes.begin() + 28));
  CHECK(r.server->receive(50ms).status().code() == ErrorCode::kDeadlineExceeded);
  raw_send_all(r.fd, std::vector<std::uint8_t>(bytes.begin() + 28, bytes.end()));
  auto f = r.server->receive(2s);
  REQUIRE(f.is_ok());
  CHECK(f->type == 77);
  CHECK(f->correlation == 5);
  CHECK(f->payload == Bytes{1, 2, 3, 4, 5, 6, 7, 8});
}
#endif

TEST_CASE("oversize payload is rejected by the receiver") {
  Pair p = insecure_pair();
  p.server->set_max_payload(1000);
  REQUIRE(p.client->send(make_frame(3, 4096)).is_ok());
  CHECK(p.server->receive(2s).status().code() == ErrorCode::kProtocolError);
  CHECK(p.server->receive(100ms).status().code() == ErrorCode::kUnavailable);
}
