#pragma once
// MessageStream: typed protocol messages over one transport connection (one logical channel).
//
// Encodes a Message into a transport Frame tagged with its MessageType and channel, and decodes received
// frames with bounded limits. A message arriving on the wrong channel is a protocol error: bulk provisioning
// must never share a stream with control traffic.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>

#include "clusterlm/protocol/messages.hpp"
#include "clusterlm/transport/transport.hpp"

namespace clusterlm::protocol {

struct ReceivedMessage {
  Message message;
  std::uint64_t correlation = 0;
  std::size_t wire_bytes = 0;  // payload bytes (excluding the frame header)
};

class MessageStream {
 public:
  MessageStream(std::unique_ptr<transport::Connection> connection, Channel channel, DecodeLimits limits = {});

  Status send(const Message& message, std::uint64_t correlation = 0);
  Result<ReceivedMessage> receive(std::chrono::milliseconds timeout);
  // Receive and require a specific message type; an ErrorMessage reply is converted into its Status.
  template <typename T>
  Result<T> expect(std::chrono::milliseconds timeout);

  void close();
  Channel channel() const { return channel_; }
  transport::Connection& connection() { return *connection_; }
  transport::PeerIdentity peer() const { return connection_->peer(); }
  std::uint64_t next_correlation() { return ++correlation_; }

 private:
  std::unique_ptr<transport::Connection> connection_;
  Channel channel_;
  DecodeLimits limits_;
  std::atomic<std::uint64_t> correlation_{0};
};

template <typename T>
Result<T> MessageStream::expect(std::chrono::milliseconds timeout) {
  CLM_ASSIGN_OR_RETURN(ReceivedMessage received, receive(timeout));
  if (auto* value = std::get_if<T>(&received.message)) return std::move(*value);
  if (auto* err = std::get_if<ErrorMessage>(&received.message)) return make_error(err->code, err->message);
  return make_error(ErrorCode::kProtocolError,
                    std::string("unexpected message ") + std::string(to_string(type_of(received.message))));
}

}  // namespace clusterlm::protocol
