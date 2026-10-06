#include "clusterlm/father/node_notify.hpp"

#include <thread>

#include "clusterlm/common/clock.hpp"
#include "clusterlm/common/log.hpp"
#include "clusterlm/protocol/wire.hpp"

namespace clusterlm::father {

using namespace std::chrono_literals;

std::string_view to_string(UnpairNotifyOutcome o) noexcept {
  switch (o) {
    case UnpairNotifyOutcome::kDelivered: return "delivered";
    case UnpairNotifyOutcome::kUnreachable: return "unreachable";
    case UnpairNotifyOutcome::kRefused: return "refused";
  }
  return "?";
}

namespace {

UnpairNotifyResult result(UnpairNotifyOutcome o, std::string detail) { return {o, std::move(detail)}; }

// One attempt. kFailedPrecondition from the Node means another Father connection still holds the control channel.
Result<UnpairNotifyResult> attempt(const std::shared_ptr<const transport::DeviceIdentity>& identity,
                                   const transport::Endpoint& ep, const config::PairedDevice& node,
                                   std::chrono::milliseconds step) {
  transport::SecurityConfig sec;
  sec.mode = transport::SecurityConfig::Mode::kMutualTls;
  sec.identity = identity;
  sec.trusted_peers = {node.fingerprint};
  auto conn = transport::connect(ep, sec, node.fingerprint, step);
  if (!conn.is_ok()) return result(UnpairNotifyOutcome::kUnreachable, "could not connect to the Node");
  protocol::MessageStream stream(std::move(conn).value(), protocol::Channel::kControl);
  protocol::Hello hello;
  hello.role = protocol::NodeRole::kFather;
  hello.channel = protocol::Channel::kControl;
  hello.device_id = identity->fingerprint();
  if (!stream.send(hello).is_ok()) return result(UnpairNotifyOutcome::kUnreachable, "the Node did not accept the connection");
  auto ack = stream.expect<protocol::HelloAck>(step);
  if (!ack.is_ok()) {
    stream.close();
    if (ack.status().code() == ErrorCode::kFailedPrecondition) return ack.status();  // retry
    return result(UnpairNotifyOutcome::kUnreachable, "the Node did not answer the handshake");
  }
  const std::uint64_t nonce = static_cast<std::uint64_t>(SteadyClock::now().time_since_epoch().count());
  if (!stream.send(protocol::UnpairNotice{nonce}).is_ok()) {
    stream.close();
    return result(UnpairNotifyOutcome::kUnreachable, "the notice could not be sent");
  }
  // An Available Node offers its resources right after the handshake; skip that and wait for the acknowledgement.
  const auto deadline = SteadyClock::now() + step;
  while (SteadyClock::now() < deadline) {
    auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - SteadyClock::now());
    auto msg = stream.receive(std::max(left, 1ms));
    if (!msg.is_ok()) break;
    if (std::holds_alternative<protocol::OfferResources>(msg->message)) continue;
    if (auto* pong = std::get_if<protocol::Pong>(&msg->message)) {
      stream.close();
      if (pong->nonce == nonce) return result(UnpairNotifyOutcome::kDelivered, "");
      return result(UnpairNotifyOutcome::kRefused, "unexpected reply");
    }
    if (auto* err = std::get_if<protocol::ErrorMessage>(&msg->message)) {
      stream.close();
      return result(UnpairNotifyOutcome::kRefused, std::string(to_string(err->code)));
    }
    stream.close();
    return result(UnpairNotifyOutcome::kRefused, "unexpected reply");
  }
  stream.close();
  return result(UnpairNotifyOutcome::kUnreachable, "no acknowledgement from the Node");
}

}  // namespace

UnpairNotifyResult notify_node_unpaired(std::shared_ptr<const transport::DeviceIdentity> identity,
                                        const config::PairedDevice& node, std::chrono::milliseconds budget) {
  if (!identity) return result(UnpairNotifyOutcome::kRefused, "no device identity");
  auto ep = transport::Endpoint::parse(node.address);
  if (!ep.is_ok()) return result(UnpairNotifyOutcome::kUnreachable, "the Node has no usable address");
  const auto deadline = SteadyClock::now() + budget;
  UnpairNotifyResult last = result(UnpairNotifyOutcome::kUnreachable, "no attempt made");
  while (true) {
    const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - SteadyClock::now());
    if (left.count() <= 0) break;
    auto r = attempt(identity, ep.value(), node, std::min<std::chrono::milliseconds>(left, 2000ms));
    if (r.is_ok()) {
      last = std::move(r).value();
      break;
    }
    last = result(UnpairNotifyOutcome::kUnreachable, "the Node is still serving another session");
    std::this_thread::sleep_for(150ms);  // the previous session is being torn down: try again shortly
  }
  log::info("node_unpair_notice", {{"outcome", std::string(to_string(last.outcome))}});
  return last;
}

}  // namespace clusterlm::father
