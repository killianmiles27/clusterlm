#include "clusterlm/pairing/pairing.hpp"

#include <openssl/rand.h>

#include <algorithm>
#include <cctype>

#include "clusterlm/common/clock.hpp"
#include "clusterlm/common/log.hpp"
#include "clusterlm/pairing/spake2.hpp"

namespace clusterlm::pairing {

using namespace std::chrono_literals;

namespace {

constexpr std::string_view kAlphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";
constexpr std::uint32_t kMaxPairingPayload = 4096;  // the whole exchange is a few hundred bytes

transport::Frame make_frame(std::uint16_t type, Bytes payload) {
  transport::Frame f;
  f.type = type;
  f.payload = std::move(payload);
  return f;
}

Bytes error_payload(ErrorCode code) {
  ByteWriter w;
  w.u16(static_cast<std::uint16_t>(code));
  return std::move(w).take();
}

void send_error(transport::Connection& c, ErrorCode code) { (void)c.send(make_frame(wire::kError, error_payload(code))); }

// Receives the next frame, which must have type `expected`. A kError frame becomes its Status; anything else is
// a protocol error.
Result<transport::Frame> expect(transport::Connection& c, std::uint16_t expected, std::chrono::milliseconds timeout) {
  CLM_ASSIGN_OR_RETURN(transport::Frame f, c.receive(timeout));
  if (f.type == wire::kError) {
    ByteReader r(f.payload);
    std::uint16_t code = 0;
    if (!r.u16(code) || !r.at_end()) return make_error(ErrorCode::kProtocolError, "malformed pairing error");
    // The peer rejected us. Never reveal more than that: wrong code, lock and relay detection look alike.
    return make_error(ErrorCode::kUnauthenticated, "pairing refused by the peer");
  }
  if (f.type != expected) return make_error(ErrorCode::kProtocolError, "unexpected pairing message");
  return f;
}

std::string strip_port(const std::string& address) {
  const auto colon = address.rfind(':');
  std::string host = colon == std::string::npos ? address : address.substr(0, colon);
  if (host.size() >= 2 && host.front() == '[' && host.back() == ']') host = host.substr(1, host.size() - 2);
  return host;
}

Result<Bytes> exporter_of(const transport::Connection& c) {
  return c.export_keying_material(wire::kExporterLabel, wire::kExporterBytes);
}

Status check_info(const DeviceInfo& info, std::string_view expected_role) {
  if (info.role != expected_role) return make_error(ErrorCode::kPermissionDenied, "peer has the wrong role for this pairing");
  return Status::ok();
}

}  // namespace

// ---------------------------------------------------------------------------------------------- code

std::string generate_code() {
  unsigned char raw[5];
  std::string out;
  if (RAND_bytes(raw, sizeof raw) != 1) return {};
  std::uint64_t bits = 0;
  for (unsigned char b : raw) bits = (bits << 8) | b;
  for (std::size_t i = 0; i < kCodeChars; ++i) {
    out.push_back(kAlphabet[(bits >> (5 * (kCodeChars - 1 - i))) & 31u]);
    if (i == 3) out.push_back('-');
  }
  return out;
}

Result<std::string> normalize_code(std::string_view text) {
  std::string out;
  for (char ch : text) {
    if (ch == '-' || ch == ' ') continue;
    const char up = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    if (kAlphabet.find(up) == std::string_view::npos)
      return make_error(ErrorCode::kInvalidArgument, "pairing code has an invalid character");
    out.push_back(up);
  }
  if (out.size() != kCodeChars) return make_error(ErrorCode::kInvalidArgument, "pairing code must have 8 characters");
  return out;
}

std::string short_fingerprint(std::string_view fp) {
  std::string out;
  for (std::size_t i = 0; i < 12 && i < fp.size(); ++i) {
    if (i != 0 && i % 4 == 0) out.push_back('-');
    out.push_back(fp[i]);
  }
  return out;
}

std::string_view to_string(PairingState s) noexcept {
  switch (s) {
    case PairingState::kListening: return "listening";
    case PairingState::kPaired: return "paired";
    case PairingState::kLocked: return "locked";
    case PairingState::kExpired: return "expired";
    case PairingState::kCancelled: return "cancelled";
  }
  return "?";
}

// ---------------------------------------------------------------------------------------------- wire

namespace wire {

Bytes channel_binding(std::string_view initiator_fp, std::string_view responder_fp, ByteSpan exporter) {
  ByteWriter w;
  w.str(initiator_fp);
  w.str(responder_fp);
  w.blob(exporter);
  return std::move(w).take();
}

Bytes encode_info(const DeviceInfo& info) {
  ByteWriter w;
  w.str(info.name);
  w.str(info.role);
  w.u16(info.data_port);
  return std::move(w).take();
}

Result<DeviceInfo> decode_info(ByteSpan payload) {
  ByteReader r(payload);
  DeviceInfo info;
  if (!r.str(info.name, 64) || !r.str(info.role, 16) || !r.u16(info.data_port) || !r.at_end())
    return make_error(ErrorCode::kProtocolError, "malformed pairing device info");
  const bool printable = !info.name.empty() && std::all_of(info.name.begin(), info.name.end(), [](unsigned char c) {
                           return c >= 0x20 && c != 0x7F;
                         });
  if (!printable || (info.role != "father" && info.role != "node"))
    return make_error(ErrorCode::kProtocolError, "invalid pairing device info");
  return info;
}

}  // namespace wire

// ---------------------------------------------------------------------------------------------- responder

Result<std::unique_ptr<PairingResponder>> PairingResponder::start(ResponderOptions options) {
  if (!options.identity) return make_error(ErrorCode::kInvalidArgument, "pairing needs a device identity");
  if (options.max_failures == 0 || options.window.count() <= 0)
    return make_error(ErrorCode::kInvalidArgument, "bad pairing limits");
  std::unique_ptr<PairingResponder> p(new PairingResponder());
  p->opts_ = std::move(options);
  if (p->opts_.code.empty()) {
    p->code_display_ = generate_code();
    if (p->code_display_.empty()) return make_error(ErrorCode::kInternal, "no randomness for the pairing code");
    CLM_ASSIGN_OR_RETURN(p->code_, normalize_code(p->code_display_));
  } else {
    CLM_ASSIGN_OR_RETURN(p->code_, normalize_code(p->opts_.code));
    p->code_display_ = p->code_.substr(0, 4) + "-" + p->code_.substr(4);
  }
  p->fingerprint_ = p->opts_.identity->fingerprint();

  transport::SecurityConfig sec;
  sec.mode = transport::SecurityConfig::Mode::kMutualTls;
  sec.identity = p->opts_.identity;
  sec.pairing_channel = true;
  sec.max_payload = kMaxPairingPayload;
  CLM_ASSIGN_OR_RETURN(p->listener_, transport::listen(p->opts_.listen, sec));
  p->endpoint_ = p->listener_->local_endpoint();
  PairingResponder* raw = p.get();
  p->thread_ = std::thread([raw] { raw->run(); });
  return p;
}

PairingResponder::~PairingResponder() {
  cancel();
  if (thread_.joinable()) thread_.join();
}

PairingState PairingResponder::state() const {
  std::lock_guard lk(mu_);
  return state_;
}

void PairingResponder::finish(PairingState s) {
  if (listener_) listener_->close();
  std::lock_guard lk(mu_);
  if (state_ == PairingState::kListening) state_ = s;
  cv_.notify_all();
}

void PairingResponder::cancel() {
  cancel_.store(true);
  std::lock_guard lk(mu_);
  cv_.notify_all();
}

Result<PairedPeer> PairingResponder::wait(std::chrono::milliseconds timeout) {
  std::unique_lock lk(mu_);
  if (!cv_.wait_for(lk, timeout, [this] { return state_ != PairingState::kListening; }))
    return make_error(ErrorCode::kDeadlineExceeded, "pairing mode is still active");
  switch (state_) {
    case PairingState::kPaired: return result_;
    case PairingState::kLocked: return make_error(ErrorCode::kUnauthenticated, "pairing mode locked after repeated failures");
    case PairingState::kExpired: return make_error(ErrorCode::kDeadlineExceeded, "pairing window expired");
    default: return make_error(ErrorCode::kCancelled, "pairing mode cancelled");
  }
}

void PairingResponder::run() {
  const auto deadline = SteadyClock::now() + opts_.window;
  for (;;) {
    if (cancel_.load()) return finish(PairingState::kCancelled);
    const auto now = SteadyClock::now();
    if (now >= deadline) return finish(PairingState::kExpired);
    const auto slice = std::min<std::chrono::milliseconds>(
        200ms, std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now) + 1ms);
    auto accepted = listener_->accept(slice);
    if (!accepted.is_ok()) {
      if (accepted.status().code() == ErrorCode::kUnavailable) return finish(PairingState::kCancelled);
      continue;  // timeout, or a peer that failed the TLS handshake
    }
    auto conn = std::move(accepted).value();
    auto r = serve_one(*conn);
    conn->close();
    if (r.is_ok()) {
      {
        std::lock_guard lk(mu_);
        result_ = std::move(r).value();
      }
      return finish(PairingState::kPaired);
    }
    if (failures_.load() >= opts_.max_failures) {
      log::warn("pairing_locked", {{"failures", std::to_string(failures_.load())}});
      return finish(PairingState::kLocked);
    }
  }
}

Result<PairedPeer> PairingResponder::serve_one(transport::Connection& conn) {
  const auto step = opts_.step_timeout;
  auto fail = [&](Status s) -> Result<PairedPeer> {
    send_error(conn, s.code());
    return s;
  };

  auto hello = expect(conn, wire::kHello, step);
  if (!hello.is_ok()) return hello.status();
  ByteReader hr(hello->payload);
  std::uint16_t version = 0;
  ByteSpan x_share;
  if (!hr.u16(version) || !hr.raw(kSpake2ShareBytes, x_share) || !hr.at_end())
    return fail(make_error(ErrorCode::kProtocolError, "malformed pairing hello"));
  if (version != kPairingProtocolVersion) return fail(make_error(ErrorCode::kVersionMismatch, "pairing version"));

  CLM_ASSIGN_OR_RETURN(Spake2 spake, Spake2::create(Spake2::Role::kResponder, code_));
  CLM_ASSIGN_OR_RETURN(Bytes exporter, exporter_of(conn));
  const std::string initiator_fp = conn.peer().device_id;
  const Bytes binding = wire::channel_binding(initiator_fp, fingerprint_, exporter);
  auto keys = spake.finish(x_share, binding);
  if (!keys.is_ok()) return fail(keys.status());  // an invalid point is malformed input, not a code guess
  CLM_RETURN_IF_ERROR(conn.send(make_frame(wire::kShare, spake.share())));

  auto confirm = expect(conn, wire::kConfirmA, step);
  if (!confirm.is_ok()) return confirm.status();  // abandoned before confirming: nothing was tested
  if (!constant_time_equal(confirm->payload, keys->confirm_initiator)) {
    failures_.fetch_add(1);
    send_error(conn, ErrorCode::kUnauthenticated);
    return make_error(ErrorCode::kUnauthenticated, "pairing confirmation failed");
  }
  CLM_RETURN_IF_ERROR(conn.send(make_frame(wire::kConfirmB, keys->confirm_responder)));

  auto info_frame = expect(conn, wire::kInfo, step);
  if (!info_frame.is_ok()) return info_frame.status();
  CLM_ASSIGN_OR_RETURN(DeviceInfo peer_info, wire::decode_info(info_frame->payload));
  CLM_RETURN_IF_ERROR(check_info(peer_info, "father"));
  CLM_RETURN_IF_ERROR(conn.send(make_frame(wire::kInfo, wire::encode_info(opts_.self))));

  PairedPeer peer;
  peer.fingerprint = initiator_fp;
  peer.info = std::move(peer_info);
  peer.remote_host = strip_port(conn.peer().address);
  return peer;
}

// ---------------------------------------------------------------------------------------------- initiator

Result<PairedPeer> pair_with(const transport::Endpoint& node, const std::shared_ptr<const transport::DeviceIdentity>& identity,
                             std::string_view code, const DeviceInfo& self, std::chrono::milliseconds timeout) {
  if (!identity) return make_error(ErrorCode::kInvalidArgument, "pairing needs a device identity");
  CLM_ASSIGN_OR_RETURN(std::string normalized, normalize_code(code));
  transport::SecurityConfig sec;
  sec.mode = transport::SecurityConfig::Mode::kMutualTls;
  sec.identity = identity;
  sec.pairing_channel = true;
  sec.max_payload = kMaxPairingPayload;
  CLM_ASSIGN_OR_RETURN(auto conn, transport::connect(node, sec, std::nullopt, timeout));
  const auto step = std::max<std::chrono::milliseconds>(timeout, 1000ms);

  CLM_ASSIGN_OR_RETURN(Spake2 spake, Spake2::create(Spake2::Role::kInitiator, normalized));
  ByteWriter hello;
  hello.u16(kPairingProtocolVersion);
  hello.raw(spake.share());
  CLM_RETURN_IF_ERROR(conn->send(make_frame(wire::kHello, std::move(hello).take())));

  CLM_ASSIGN_OR_RETURN(auto share, expect(*conn, wire::kShare, step));
  CLM_ASSIGN_OR_RETURN(Bytes exporter, exporter_of(*conn));
  const std::string responder_fp = conn->peer().device_id;
  const Bytes binding = wire::channel_binding(identity->fingerprint(), responder_fp, exporter);
  CLM_ASSIGN_OR_RETURN(auto keys, spake.finish(share.payload, binding));

  CLM_RETURN_IF_ERROR(conn->send(make_frame(wire::kConfirmA, keys.confirm_initiator)));
  auto confirm = expect(*conn, wire::kConfirmB, step);
  // The responder sends no MAC when ours was wrong. Whether that is a wrong code, a relay or a locked mode is
  // deliberately indistinguishable here.
  if (!confirm.is_ok()) return confirm.status();
  if (!constant_time_equal(confirm->payload, keys.confirm_responder))
    return make_error(ErrorCode::kUnauthenticated, "the Node failed key confirmation");

  CLM_RETURN_IF_ERROR(conn->send(make_frame(wire::kInfo, wire::encode_info(self))));
  CLM_ASSIGN_OR_RETURN(auto info_frame, expect(*conn, wire::kInfo, step));
  CLM_ASSIGN_OR_RETURN(DeviceInfo peer_info, wire::decode_info(info_frame.payload));
  CLM_RETURN_IF_ERROR(check_info(peer_info, "node"));

  PairedPeer peer;
  peer.fingerprint = responder_fp;
  peer.info = std::move(peer_info);
  peer.remote_host = node.host;
  conn->close();
  return peer;
}

}  // namespace clusterlm::pairing
