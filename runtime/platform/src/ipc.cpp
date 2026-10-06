#include "clusterlm/platform/ipc.hpp"

#include <algorithm>

#include "clusterlm/common/clock.hpp"
#include "ipc_internal.hpp"

namespace clusterlm::ipc {

using namespace std::chrono_literals;

// ---- Frame codec ---------------------------------------------------------------------------------------
Status check_frame_length(std::uint32_t body_length, std::uint32_t max_frame) {
  if (body_length < kEnvelopeHeaderBytes)
    return make_error(ErrorCode::kProtocolError, "ipc frame shorter than the envelope header");
  if (body_length > max_frame)
    return make_error(ErrorCode::kProtocolError, "ipc frame exceeds the maximum frame size");
  return Status::ok();
}

Result<Bytes> encode_frame(const Envelope& env, std::uint32_t max_frame) {
  const std::uint64_t body = kEnvelopeHeaderBytes + static_cast<std::uint64_t>(env.payload.size());
  if (max_frame > kHardMaxFrameBytes) max_frame = kHardMaxFrameBytes;
  if (body > max_frame) return make_error(ErrorCode::kResourceExhausted, "ipc message exceeds the maximum frame size");
  ByteWriter w(static_cast<std::size_t>(body) + 4);
  w.u32(static_cast<std::uint32_t>(body));
  w.u16(env.kind);
  w.u16(env.version);
  w.raw(env.payload);
  return std::move(w).take();
}

Result<Envelope> decode_body(ByteSpan body) {
  ByteReader r(body);
  Envelope e;
  ByteSpan rest;
  if (!r.u16(e.kind) || !r.u16(e.version) || !r.raw(r.remaining(), rest))
    return make_error(ErrorCode::kProtocolError, "truncated ipc envelope");
  e.payload.assign(rest.begin(), rest.end());
  return e;
}

// ---- Authorization -------------------------------------------------------------------------------------
Status AuthPolicy::authorize(const PeerCredentials& peer) const {
  if (require_interactive_session && !(peer.session_known && peer.session_id != 0))
    return make_error(ErrorCode::kPermissionDenied, "peer is not in an interactive user session");
  if (!allowed_user_ids.empty()) {
    if (peer.user_id.empty()) return make_error(ErrorCode::kUnauthenticated, "peer user could not be determined");
    if (std::find(allowed_user_ids.begin(), allowed_user_ids.end(), peer.user_id) == allowed_user_ids.end())
      return make_error(ErrorCode::kPermissionDenied, "peer user is not authorized");
  }
  return Status::ok();
}

// ---- Endpoint ------------------------------------------------------------------------------------------
Result<std::string> Endpoint::native_path() const {
  if (name.empty() || name.size() > 64)
    return make_error(ErrorCode::kInvalidArgument, "ipc endpoint name must be 1..64 characters");
  for (char c : name) {
    const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '.' ||
                    c == '-' || c == '_';
    if (!ok) return make_error(ErrorCode::kInvalidArgument, "ipc endpoint name has a character outside [A-Za-z0-9._-]");
  }
#ifdef _WIN32
  return std::string("\\\\.\\pipe\\") + name;
#else
  if (socket_dir.empty()) return make_error(ErrorCode::kInvalidArgument, "ipc socket_dir is required on POSIX");
  return (socket_dir / (name + ".sock")).string();
#endif
}

// ---- Connection ----------------------------------------------------------------------------------------
Connection::Connection(std::unique_ptr<detail::RawChannel> ch, std::uint32_t max_frame)
    : ch_(std::move(ch)), max_frame_(std::min(max_frame, kHardMaxFrameBytes)) {}
Connection::~Connection() { close(); }

const PeerCredentials& Connection::peer() const { return ch_->peer(); }
bool Connection::is_open() const { return !poisoned_.load() && ch_->is_open(); }

void Connection::close() {
  poisoned_.store(true);
  ch_->close();
}

Status Connection::send(const Envelope& env, std::chrono::milliseconds timeout) {
  CLM_ASSIGN_OR_RETURN(auto frame, encode_frame(env, max_frame_));
  std::lock_guard lock(send_mu_);
  if (poisoned_.load()) return make_error(ErrorCode::kUnavailable, "ipc connection closed");
  Status st = ch_->write_all(frame.data(), frame.size(), timeout);
  if (!st.is_ok()) close();  // a partially written frame leaves the stream unusable
  return st;
}

namespace {
// Reads exactly n bytes. `first_timeout` bounds the wait for the first byte; once any byte has arrived the rest
// must complete within kFrameCompletionTimeout. `timed_out_clean` is set when nothing at all arrived.
Status read_exact(detail::RawChannel& ch, std::uint8_t* buf, std::size_t n, std::chrono::milliseconds first_timeout,
                  bool& timed_out_clean) {
  timed_out_clean = false;
  std::size_t got = 0;
  SteadyClock::time_point deadline;
  while (got < n) {
    std::chrono::milliseconds wait = first_timeout;
    if (got > 0) {
      wait = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - SteadyClock::now());
      if (wait <= 0ms) return make_error(ErrorCode::kDeadlineExceeded, "ipc frame was not completed in time");
    }
    CLM_ASSIGN_OR_RETURN(const std::size_t r, ch.read_some(buf + got, n - got, wait));
    if (r == 0) {
      if (got == 0) {
        timed_out_clean = true;
        return make_error(ErrorCode::kDeadlineExceeded, "ipc receive timed out");
      }
      continue;
    }
    if (got == 0) deadline = SteadyClock::now() + kFrameCompletionTimeout;
    got += r;
  }
  return Status::ok();
}
}  // namespace

Result<Envelope> Connection::receive(std::chrono::milliseconds timeout) {
  std::lock_guard lock(recv_mu_);
  if (poisoned_.load()) return make_error(ErrorCode::kUnavailable, "ipc connection closed");
  std::uint8_t hdr[4];
  bool clean_timeout = false;
  if (auto st = read_exact(*ch_, hdr, sizeof hdr, timeout, clean_timeout); !st.is_ok()) {
    if (!clean_timeout) close();  // partial header or dead peer: the stream position is unknown
    return st;
  }
  const std::uint32_t len = static_cast<std::uint32_t>(hdr[0]) | (static_cast<std::uint32_t>(hdr[1]) << 8) |
                            (static_cast<std::uint32_t>(hdr[2]) << 16) | (static_cast<std::uint32_t>(hdr[3]) << 24);
  if (auto st = check_frame_length(len, max_frame_); !st.is_ok()) {
    close();
    return st;
  }
  Bytes body(len);
  if (auto st = read_exact(*ch_, body.data(), body.size(), kFrameCompletionTimeout, clean_timeout); !st.is_ok()) {
    close();
    return st;
  }
  return decode_body(body);
}

// ---- Server / connect ----------------------------------------------------------------------------------
Server::Server(ServerOptions opts, std::unique_ptr<detail::RawListener> l)
    : opts_(std::move(opts)), listener_(std::move(l)) {}
Server::~Server() { close(); }
void Server::close() {
  if (listener_) listener_->close();
}

Result<std::unique_ptr<Server>> Server::create(ServerOptions options) {
  if (options.max_frame_bytes < kEnvelopeHeaderBytes || options.max_frame_bytes > kHardMaxFrameBytes)
    return make_error(ErrorCode::kInvalidArgument, "ipc max_frame_bytes out of range");
  CLM_ASSIGN_OR_RETURN(auto listener, detail::make_listener(options));
  return std::unique_ptr<Server>(new Server(std::move(options), std::move(listener)));
}

Result<std::unique_ptr<Connection>> Server::accept(std::chrono::milliseconds timeout) {
  CLM_ASSIGN_OR_RETURN(auto ch, listener_->accept(timeout));
  if (opts_.authorizer) {
    if (auto st = opts_.authorizer(ch->peer()); !st.is_ok()) {
      ch->close();
      return st;
    }
  }
  return std::make_unique<Connection>(std::move(ch), opts_.max_frame_bytes);
}

Result<std::unique_ptr<Connection>> connect(const Endpoint& endpoint, const ClientOptions& options,
                                            std::chrono::milliseconds timeout) {
  if (options.max_frame_bytes < kEnvelopeHeaderBytes || options.max_frame_bytes > kHardMaxFrameBytes)
    return make_error(ErrorCode::kInvalidArgument, "ipc max_frame_bytes out of range");
  CLM_ASSIGN_OR_RETURN(auto ch, detail::make_client(endpoint, options, timeout));
  return std::make_unique<Connection>(std::move(ch), options.max_frame_bytes);
}

}  // namespace clusterlm::ipc
