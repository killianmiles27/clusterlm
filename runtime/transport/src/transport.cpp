#include "clusterlm/transport/transport.hpp"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <mutex>

#include "clusterlm/common/log.hpp"
#include "framed.hpp"
#include "socket.hpp"
#include "tls.hpp"

namespace clusterlm::transport {

// ---------------------------------------------------------------------------------------------- Endpoint

Result<Endpoint> Endpoint::parse(std::string_view text) {
  const auto colon = text.rfind(':');
  if (colon == std::string_view::npos || colon == 0)
    return make_error(ErrorCode::kInvalidArgument, "endpoint must be host:port");
  std::string_view host = text.substr(0, colon);
  const std::string_view port_text = text.substr(colon + 1);
  if (host.size() >= 2 && host.front() == '[' && host.back() == ']') host = host.substr(1, host.size() - 2);
  else if (host.find(':') != std::string_view::npos)
    return make_error(ErrorCode::kInvalidArgument, "IPv6 endpoints must be written [addr]:port");
  if (host.empty()) return make_error(ErrorCode::kInvalidArgument, "endpoint host is empty");
  unsigned port = 0;
  const auto [ptr, ec] = std::from_chars(port_text.data(), port_text.data() + port_text.size(), port);
  if (port_text.empty() || ec != std::errc() || ptr != port_text.data() + port_text.size() || port > 65535)
    return make_error(ErrorCode::kInvalidArgument, "endpoint port must be 0..65535");
  return Endpoint{std::string(host), static_cast<std::uint16_t>(port)};
}

std::string Endpoint::str() const {
  if (host.find(':') != std::string::npos) return "[" + host + "]:" + std::to_string(port);
  return host + ":" + std::to_string(port);
}

bool Endpoint::is_loopback() const { return net::is_loopback_host(host); }

// ---------------------------------------------------------------------------------------------- Listener

namespace {

// Bound on TCP-accepted-but-not-yet-authenticated peers: a stalled client must not hold accept() forever.
constexpr std::chrono::milliseconds kHandshakeTimeout{10'000};

class TcpListener final : public Listener {
 public:
  TcpListener(net::ListenSocket ls, Endpoint local, SecurityConfig security, std::shared_ptr<TlsContext> tls)
      : sock_(std::move(ls.socket)), local_(std::move(local)), security_(std::move(security)), tls_(std::move(tls)) {}

  Result<std::unique_ptr<Connection>> accept(std::chrono::milliseconds timeout) override {
    const net::Deadline deadline = net::deadline_after(timeout);
    net::Accepted acc;
    for (;;) {
      {
        // fd_mu_ keeps close() from closing the handle while this thread polls/accepts on it; each hold is at
        // most one poll slice, so close() stays prompt.
        std::lock_guard<std::mutex> lk(fd_mu_);
        if (closed_.load(std::memory_order_acquire) || !sock_.valid())
          return make_error(ErrorCode::kUnavailable, "listener closed");
        const net::Deadline slice_end = std::min(deadline, SteadyClock::now() + net::kPollSlice);
        const net::WaitResult w = net::wait_ready(sock_.handle(), false, slice_end, &closed_);
        if (w == net::WaitResult::kReady) {
          auto r = net::tcp_accept(sock_.handle());
          if (r.is_ok()) {
            acc = std::move(r).value();
            break;
          }
          if (r.status().message() != "again") return r.status();
        } else if (w == net::WaitResult::kCancelled) {
          return make_error(ErrorCode::kUnavailable, "listener closed");
        } else if (w == net::WaitResult::kError) {
          return make_error(ErrorCode::kUnavailable, "listener poll failed");
        }
      }
      if (SteadyClock::now() >= deadline) return make_error(ErrorCode::kDeadlineExceeded, "accept timed out");
    }

    PeerIdentity peer;
    peer.address = acc.peer_address;
    if (security_.mode == SecurityConfig::Mode::kInsecureLoopbackOnly) {
      if (!acc.peer_is_loopback)
        return make_error(ErrorCode::kPermissionDenied, "insecure mode refuses non-loopback peer");
      peer.device_id = std::string(kInsecureLoopbackDeviceId);
      peer.authenticated = false;
      return make_framed_connection(make_plain_stream(std::move(acc.socket)), std::move(peer),
                                    security_.max_payload);
    }
    auto est = tls_->establish(std::move(acc.socket), acc.peer_address, std::nullopt,
                               net::deadline_after(kHandshakeTimeout), &closed_);
    if (!est.is_ok()) {
      log::debug("transport.accept_rejected", {{"code", std::string(to_string(est.status().code()))}});
      return est.status();
    }
    auto e = std::move(est).value();
    return make_framed_connection(std::move(e.stream), std::move(e.peer), security_.max_payload);
  }

  Endpoint local_endpoint() const override { return local_; }

  void close() override {
    closed_.store(true, std::memory_order_release);
    std::lock_guard<std::mutex> lk(fd_mu_);  // waits out an in-flight poll slice, then releases the port
    sock_.reset();
  }

 private:
  std::mutex fd_mu_;
  net::Socket sock_;
  std::atomic<bool> closed_{false};
  Endpoint local_;
  SecurityConfig security_;
  std::shared_ptr<TlsContext> tls_;
};

}  // namespace

Result<std::unique_ptr<Listener>> listen(const Endpoint& endpoint, const SecurityConfig& security) {
  const bool insecure = security.mode == SecurityConfig::Mode::kInsecureLoopbackOnly;
  if (insecure && !endpoint.is_loopback())
    return make_error(ErrorCode::kPermissionDenied, "insecure mode only listens on loopback addresses");
  std::shared_ptr<TlsContext> tls;
  if (!insecure) {
    CLM_ASSIGN_OR_RETURN(tls, TlsContext::create(security, true));
  }
  CLM_ASSIGN_OR_RETURN(net::ListenSocket ls, net::tcp_listen(endpoint.host, endpoint.port, insecure));
  Endpoint local{endpoint.host, ls.port};
  return std::unique_ptr<Listener>(new TcpListener(std::move(ls), std::move(local), security, std::move(tls)));
}

Result<std::unique_ptr<Connection>> connect(const Endpoint& endpoint, const SecurityConfig& security,
                                            std::optional<std::string> expected_peer_device_id,
                                            std::chrono::milliseconds timeout) {
  const bool insecure = security.mode == SecurityConfig::Mode::kInsecureLoopbackOnly;
  if (insecure && !endpoint.is_loopback())
    return make_error(ErrorCode::kPermissionDenied, "insecure mode only connects to loopback addresses");
  std::shared_ptr<TlsContext> tls;
  if (!insecure) {
    CLM_ASSIGN_OR_RETURN(tls, TlsContext::create(security, false));
  }
  const net::Deadline deadline = net::deadline_after(timeout);
  CLM_ASSIGN_OR_RETURN(net::Socket sock, net::tcp_connect(endpoint.host, endpoint.port, insecure, deadline));

  PeerIdentity peer;
  peer.address = endpoint.str();
  if (insecure) {
    peer.device_id = std::string(kInsecureLoopbackDeviceId);
    return make_framed_connection(make_plain_stream(std::move(sock)), std::move(peer), security.max_payload);
  }
  CLM_ASSIGN_OR_RETURN(TlsEstablished est,
                       tls->establish(std::move(sock), endpoint.str(), std::move(expected_peer_device_id), deadline,
                                      nullptr));
  return make_framed_connection(std::move(est.stream), std::move(est.peer), security.max_payload);
}

}  // namespace clusterlm::transport
