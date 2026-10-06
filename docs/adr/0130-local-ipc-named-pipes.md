# 0130 — Local IPC: named pipes (Windows) and Unix sockets (dev), authenticated by the OS

## Context

The Node service runs in session 0 and cannot see user input. A per-user helper must report activity and lock state
to it, and Father's UI must talk to the per-user Father agent. Both are strictly local, carry no model data, and
must not become a way for another local user or a remote machine to steer a Node or read Father state.

## Decision

- One abstraction (`ipc::Server` / `ipc::Connection`, length-prefixed frames, `ipc::Envelope{kind u16, version u16,
  payload}`) with two byte-stream implementations. Framing, bounds and authorization live above the stream and are
  identical on both OSes. Frames are bounded (default 1 MiB, hard cap 16 MiB); an out-of-bound or short length
  poisons the connection (closed, never skipped).
- Windows: named pipe, byte mode, `PIPE_REJECT_REMOTE_CLIENTS`, `FILE_FLAG_FIRST_PIPE_INSTANCE` (a pre-created
  squatter fails creation), overlapped I/O with a cancel event so every wait has a timeout and `close()` works from
  any thread. The DACL is explicit and protected: deny network logons, allow SYSTEM and the creating account;
  the helper pipe additionally allows interactive users read/write only (mask `0x12019b`, i.e. no
  `FILE_CREATE_PIPE_INSTANCE`). Clients open with `SECURITY_IDENTIFICATION`: the service can identify them, never
  impersonate them.
- Peer credentials come from the OS, never from message content: `GetNamedPipeClientProcessId`,
  `GetNamedPipeClientSessionId` and the client token's user SID (read through `ImpersonateNamedPipeClient`, which
  Windows allows only after the server has read once, hence a one-byte hello the client sends on connect). An
  `AuthPolicy` (interactive session required, allowed SIDs) is applied; an `ActivityReport` must name the peer's own
  session.
- POSIX: Unix socket in a directory tightened to 0700, socket 0600, `SO_PEERCRED` uid must equal the server's
  euid (or root). `Authorizer` logic is tested with injected credentials because a second uid is not available in
  tests.
- The Father agent pipe is per user (`ClusterLM.Father.UI.<tag>`), owner+SYSTEM only, and also restricted to the
  agent's own SID.

## Consequences

- A client that can open the pipe but fails `AuthPolicy` is disconnected and counted; nothing about its identity is
  logged.
- Whether `OpenProcess` on the pipe server (used by clients that pin the server user) is permitted for a
  standard user against a LocalService process is unverified; the shipped helper does not pin the server user until
  HQ-WIN-02 shows it works. The first-instance flag and DACL already prevent squatting by unprivileged users.
- The hello byte makes the Windows pipe protocol one byte longer than the POSIX one; it is internal to
  `ipc_win.cpp`.
