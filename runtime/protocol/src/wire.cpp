#include "clusterlm/protocol/wire.hpp"

namespace clusterlm::protocol {

MessageStream::MessageStream(std::unique_ptr<transport::Connection> connection, Channel channel, DecodeLimits limits)
    : connection_(std::move(connection)), channel_(channel), limits_(limits) {}

Status MessageStream::send(const Message& message, std::uint64_t correlation) {
  const MessageType type = type_of(message);
  const Channel expected = channel_of(type);
  // Hello/Error/Ping may appear on any channel (channel setup and failure reporting).
  const bool any_channel = type == MessageType::kHello || type == MessageType::kHelloAck ||
                           type == MessageType::kError || type == MessageType::kPing || type == MessageType::kPong;
  if (!any_channel && expected != channel_) {
    return make_error(ErrorCode::kInternal, std::string(to_string(type)) + " must not be sent on the " +
                                                std::string(to_string(channel_)) + " channel");
  }
  transport::Frame frame;
  frame.type = static_cast<std::uint16_t>(type);
  frame.channel = static_cast<std::uint8_t>(channel_);
  frame.correlation = correlation;
  frame.payload = encode(message);
  return connection_->send(frame);
}

Result<ReceivedMessage> MessageStream::receive(std::chrono::milliseconds timeout) {
  CLM_ASSIGN_OR_RETURN(transport::Frame frame, connection_->receive(timeout));
  if (frame.channel != static_cast<std::uint8_t>(channel_)) {
    connection_->close();
    return make_error(ErrorCode::kProtocolError, "frame for channel " + std::to_string(frame.channel) +
                                                     " received on the " + std::string(to_string(channel_)) +
                                                     " channel");
  }
  const auto type = static_cast<MessageType>(frame.type);
  auto decoded = decode(type, frame.payload, limits_);
  if (!decoded.is_ok()) {
    connection_->close();
    return decoded.status();
  }
  return ReceivedMessage{std::move(decoded).value(), frame.correlation, frame.payload.size()};
}

void MessageStream::close() { connection_->close(); }

}  // namespace clusterlm::protocol
