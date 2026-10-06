#pragma once
// Father -> Node unpair notification (docs/pairing.md).
//
// When Father unpairs a Node that is reachable it opens an ordinary control channel (mutual TLS, pinned to the Node's
// fingerprint, Father role) and sends one UnpairNotice. The Node accepts it only from a pinned (paired) Father: it
// releases its lease, stops trusting that Father and the Node service clears its paired-Father setting. The notice
// carries no data besides a nonce. Delivery is best effort and never blocks the unpair itself: an unreachable or
// refusing Node keeps the documented behaviour (it trusts this Father until unpaired locally).
#include <chrono>
#include <memory>
#include <string>

#include "clusterlm/config/settings.hpp"
#include "clusterlm/transport/security.hpp"

namespace clusterlm::father {

enum class UnpairNotifyOutcome : std::uint8_t {
  kDelivered,    // the Node acknowledged: it no longer trusts this Father
  kUnreachable,  // no TLS connection, handshake or reply within the budget
  kRefused,      // the Node answered but refused (not the paired Father, or an unexpected reply)
};
std::string_view to_string(UnpairNotifyOutcome o) noexcept;

struct UnpairNotifyResult {
  UnpairNotifyOutcome outcome = UnpairNotifyOutcome::kUnreachable;
  std::string detail;  // short, free of user data
  bool delivered() const { return outcome == UnpairNotifyOutcome::kDelivered; }
};

// Sends the notice to `node` (its `address` is the data endpoint, `fingerprint` the pinned identity). While a
// previous Father session is still being torn down the Node may answer "a Father is already connected"; that is
// retried briefly within `budget`.
UnpairNotifyResult notify_node_unpaired(std::shared_ptr<const transport::DeviceIdentity> identity,
                                        const config::PairedDevice& node,
                                        std::chrono::milliseconds budget = std::chrono::milliseconds(3000));

}  // namespace clusterlm::father
