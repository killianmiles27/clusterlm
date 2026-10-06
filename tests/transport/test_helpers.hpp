#pragma once
#include <doctest/doctest.h>

#include <chrono>
#include <future>
#include <memory>
#include <optional>
#include <string>

#include "clusterlm/transport/transport.hpp"

namespace clm_test {

using namespace clusterlm;
using namespace clusterlm::transport;
using namespace std::chrono_literals;

inline Frame make_frame(std::uint16_t type, std::size_t size, std::uint8_t fill = 0xAB, std::uint64_t corr = 0) {
  Frame f;
  f.type = type;
  f.channel = 1;
  f.flags = 0;
  f.correlation = corr;
  f.payload.assign(size, fill);
  return f;
}

inline SecurityConfig insecure() {
  SecurityConfig c;
  c.mode = SecurityConfig::Mode::kInsecureLoopbackOnly;
  return c;
}

struct Pair {
  std::unique_ptr<Listener> listener;
  std::unique_ptr<Connection> client;
  std::unique_ptr<Connection> server;
  Status client_status;
  Status server_status;
};

// Listens on an ephemeral loopback port, then connects and accepts concurrently.
inline Pair connect_pair(const SecurityConfig& server_cfg, const SecurityConfig& client_cfg,
                         std::optional<std::string> expected = std::nullopt) {
  Pair p;
  auto l = listen(Endpoint{"127.0.0.1", 0}, server_cfg);
  REQUIRE(l.is_ok());
  p.listener = std::move(l).value();
  auto accepted = std::async(std::launch::async, [&] { return p.listener->accept(5s); });
  auto c = connect(p.listener->local_endpoint(), client_cfg, std::move(expected), 5s);
  auto a = accepted.get();
  if (c.is_ok()) p.client = std::move(c).value();
  else p.client_status = c.status();
  if (a.is_ok()) p.server = std::move(a).value();
  else p.server_status = a.status();
  return p;
}

inline Pair insecure_pair() {
  Pair p = connect_pair(insecure(), insecure());
  REQUIRE(p.client != nullptr);
  REQUIRE(p.server != nullptr);
  return p;
}

}  // namespace clm_test
