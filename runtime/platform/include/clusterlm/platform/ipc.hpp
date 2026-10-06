#pragma once
// Local message channel: length-prefixed frames over a Windows named pipe or a POSIX Unix-domain socket.
//
// Used for the Node session helper -> service pipe and the Father UI <-> agent pipe. It is strictly local
// (PIPE_REJECT_REMOTE_CLIENTS on Windows; a 0700 directory, 0600 socket and SO_PEERCRED check on POSIX).
// Wire format, little-endian via ByteWriter/ByteReader:
//     frame    = u32 body_length, body
//     body     = u16 kind, u16 version, payload (opaque to this layer)
// Frame sizes are bounded on both ends: a longer announced frame is a protocol error and the connection is
// poisoned (it must be closed), never skipped. Nothing here logs payload bytes.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "clusterlm/common/bytes.hpp"
#include "clusterlm/common/status.hpp"

namespace clusterlm::ipc {

constexpr std::uint32_t kDefaultMaxFrameBytes = 1u << 20;
constexpr std::uint32_t kHardMaxFrameBytes = 16u << 20;
constexpr std::size_t kEnvelopeHeaderBytes = 4;  // kind + version
// Once the first byte of a frame has arrived the rest must follow within this time, or the connection is dead.
constexpr std::chrono::milliseconds kFrameCompletionTimeout{5000};

// Versioned envelope. `kind` selects the message type, `version` its schema version.
struct Envelope {
  std::uint16_t kind = 0;
  std::uint16_t version = 0;
  Bytes payload;
};

// Frame codec, exposed for tests. encode_frame fails with kResourceExhausted beyond max_frame.
Result<Bytes> encode_frame(const Envelope& env, std::uint32_t max_frame = kDefaultMaxFrameBytes);
// Validates an announced body length (>= envelope header, <= max_frame) -> kProtocolError otherwise.
Status check_frame_length(std::uint32_t body_length, std::uint32_t max_frame);
Result<Envelope> decode_body(ByteSpan body);

// Who is on the other end, as established by the operating system (never by anything the peer sent).
struct PeerCredentials {
  std::uint32_t pid = 0;           // 0 = unknown
  bool session_known = false;      // Windows: GetNamedPipeClientSessionId succeeded
  std::uint32_t session_id = 0;    // Windows terminal-services session; 0 is the services session
  std::string user_id;             // Windows: SID string ("S-1-5-..."); POSIX: "uid:<n>"; empty = unresolved
};

// Declarative authorization of a peer: used by the Node service for the helper pipe.
struct AuthPolicy {
  bool require_interactive_session = false;   // peer must be in a user session (session known and != 0)
  std::vector<std::string> allowed_user_ids;  // empty = any user the pipe ACL admitted
  Status authorize(const PeerCredentials& peer) const;
};
using Authorizer = std::function<Status(const PeerCredentials&)>;

// Who may open the pipe at the OS level (Windows DACL). POSIX always uses owner-only socket permissions.
enum class PipeAccess : std::uint8_t {
  kOwnerAndSystem,                 // the creating account and SYSTEM: Father UI <-> agent
  kOwnerSystemAndInteractiveUsers  // plus interactively logged-on users (read/write only): helper -> service
};

struct Endpoint {
  std::string name;                   // logical name: [A-Za-z0-9._-]{1,64}
  std::filesystem::path socket_dir;   // POSIX only: directory holding "<name>.sock"; ignored on Windows
  // Windows pipe path "\\.\pipe\<name>" / POSIX socket path.
  Result<std::string> native_path() const;
};

struct ServerOptions {
  Endpoint endpoint;
  PipeAccess access = PipeAccess::kOwnerAndSystem;
  std::uint32_t max_frame_bytes = kDefaultMaxFrameBytes;
  Authorizer authorizer;  // applied after the OS-level checks; null = accept whoever passed them
};

struct ClientOptions {
  std::uint32_t max_frame_bytes = kDefaultMaxFrameBytes;
  // If non-empty the server process must run as one of these users (Windows SIDs / POSIX "uid:<n>"): stops a
  // squatter on the pipe name from impersonating the service.
  std::vector<std::string> expected_server_user_ids;
};

namespace detail {
class RawChannel;
class RawListener;
}  // namespace detail

class Connection {
 public:
  Connection(std::unique_ptr<detail::RawChannel> ch, std::uint32_t max_frame);
  ~Connection();
  Connection(const Connection&) = delete;
  Connection& operator=(const Connection&) = delete;

  // Thread-safe against receive() and close(); concurrent senders are serialised.
  Status send(const Envelope& env, std::chrono::milliseconds timeout = std::chrono::milliseconds(5000));
  // kDeadlineExceeded if nothing arrived within `timeout` (the connection remains usable); kUnavailable when the
  // peer closed; kProtocolError for an oversize/short frame (connection is then closed).
  Result<Envelope> receive(std::chrono::milliseconds timeout);
  const PeerCredentials& peer() const;
  // Wakes blocked send/receive calls (they fail with kCancelled/kUnavailable) and releases the OS handle.
  void close();
  bool is_open() const;

 private:
  std::unique_ptr<detail::RawChannel> ch_;
  std::uint32_t max_frame_;
  std::mutex send_mu_, recv_mu_;
  std::atomic<bool> poisoned_{false};
};

class Server {
 public:
  static Result<std::unique_ptr<Server>> create(ServerOptions options);
  ~Server();
  Server(const Server&) = delete;
  Server& operator=(const Server&) = delete;

  // Next authorized client. kDeadlineExceeded on timeout; kPermissionDenied/kUnauthenticated when a client was
  // rejected by the OS-level or Authorizer check (the client has been disconnected; call accept again).
  Result<std::unique_ptr<Connection>> accept(std::chrono::milliseconds timeout);
  void close();
  const ServerOptions& options() const { return opts_; }

 private:
  Server(ServerOptions opts, std::unique_ptr<detail::RawListener> l);
  ServerOptions opts_;
  std::unique_ptr<detail::RawListener> listener_;
};

Result<std::unique_ptr<Connection>> connect(const Endpoint& endpoint, const ClientOptions& options = {},
                                            std::chrono::milliseconds timeout = std::chrono::milliseconds(2000));

// The user identifier of the calling process in the same format as PeerCredentials::user_id.
Result<std::string> current_user_id();

}  // namespace clusterlm::ipc
