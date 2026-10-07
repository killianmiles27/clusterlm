#pragma once
// Raw protocol client for tests that talk to a NodeWorker the way a (possibly hostile) peer would: no Coordinator,
// just Hello + hand-built messages over insecure loopback or mutual TLS.
#include <doctest/doctest.h>

#include <chrono>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>

#include "clusterlm/node/node_worker.hpp"
#include "clusterlm/protocol/wire.hpp"
#include "clusterlm/transport/transport.hpp"

namespace clusterlm::testing {

using namespace std::chrono_literals;

// A fresh scratch directory (created) under the system temp directory.
inline std::filesystem::path privacy_temp_dir(const char* name) {
  auto p = std::filesystem::temp_directory_path() /
           (std::string("clm-privacy-") + name + "-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  { std::error_code ec_rm; std::filesystem::remove_all(p, ec_rm); }
  std::filesystem::create_directories(p);
  return p;
}

inline transport::SecurityConfig insecure_security() {
  transport::SecurityConfig c;
  c.mode = transport::SecurityConfig::Mode::kInsecureLoopbackOnly;
  return c;
}

// Opens `channel` to `endpoint`, sends Hello and returns the stream positioned after the Hello. The node's first
// reply (HelloAck or an Error) is NOT consumed.
inline Result<std::unique_ptr<protocol::MessageStream>> open_stream(const transport::Endpoint& endpoint,
                                                                   const transport::SecurityConfig& security,
                                                                   protocol::Channel channel, protocol::NodeRole role,
                                                                   const std::string& device_id,
                                                                   LeaseGeneration lease,
                                                                   std::optional<std::string> expected_peer = std::nullopt,
                                                                   const std::string& backend_build = "reference-cpu-fp32-v1") {
  auto conn = transport::connect(endpoint, security, std::move(expected_peer), 5s);
  if (!conn.is_ok()) return conn.status();
  auto stream = std::make_unique<protocol::MessageStream>(std::move(conn).value(), channel);
  protocol::Hello hello;
  hello.role = role;
  hello.channel = channel;
  hello.device_id = device_id;
  hello.backend_build = backend_build;
  hello.lease = lease;
  CLM_RETURN_IF_ERROR(stream->send(hello));
  return stream;
}

// Receives the next message and returns it as an Error status if it is one.
inline Result<protocol::Message> next_message(protocol::MessageStream& s, std::chrono::milliseconds timeout = 3000ms) {
  auto r = s.receive(timeout);
  if (!r.is_ok()) return r.status();
  return std::move(r->message);
}

}  // namespace clusterlm::testing
