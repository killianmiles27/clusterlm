// clusterlm-bench baseline llama-rpc: the P0-A comparison arm (HQ-P0A-01).
//
// Runs stock llama.cpp generation, from the pinned build, with its RPC devices pointing at Nodes that run the
// pinned rpc-server WITHOUT -c (persistent weight cache off), and records:
//   * prefill and decode throughput (prompt tokens and greedy single-token steps, `--repeat` times);
//   * bytes over the wire, per phase and per RPC command, by relaying every Node connection through a counting
//     TCP proxy owned by this process (the proxy parses only the request framing `cmd | size | payload` to count
//     calls; it never inspects tensor contents);
//   * filesystem effects on the Nodes: whether the rpc-server's cache directory stays empty (for Nodes this tool
//     starts itself it watches the directory directly; for remote Nodes it reads the JSON reports written by
//     scripts/run_rpc_server.ps1 -Report);
//   * the allocation each Node reports (free device memory before/after the load) and the client's own resident
//     memory, to show who owns the weights.
// This arm moves raw ggml tensors, including token inputs, over an unauthenticated protocol: it is a throughput
// baseline for an isolated benchmark network, never a product path (docs/backends/llama-rpc.md). The harness feeds
// it synthetic token IDs only.
//
// Provenance: Synthetic unless --on-target names real remote Nodes and the model is not the tiny fixture.
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>

#include "bench_common.hpp"
#include "commands.hpp"

#ifdef CLUSTERLM_BENCH_HAS_LLAMA

#include <atomic>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

#include "clusterlm/common/clock.hpp"
#include "clusterlm/platform/process.hpp"
#include "ggml-backend.h"
#include "ggml-rpc.h"
#include "llama.h"

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#ifndef CLUSTERLM_LLAMA_PIN
#define CLUSTERLM_LLAMA_PIN "unknown"
#endif

namespace clusterlm::bench {

namespace fs = std::filesystem;
using nlohmann::json;

namespace {

// ---- minimal portable TCP ----------------------------------------------------------------------------------

#ifdef _WIN32
using sock_t = SOCKET;
constexpr sock_t kBadSock = INVALID_SOCKET;
void close_sock(sock_t s) { closesocket(s); }
void shutdown_sock(sock_t s, int how) { shutdown(s, how == 0 ? SD_RECEIVE : how == 1 ? SD_SEND : SD_BOTH); }
void net_init() {
  static std::once_flag once;
  std::call_once(once, [] {
    WSADATA d;
    WSAStartup(MAKEWORD(2, 2), &d);
  });
}
#else
using sock_t = int;
constexpr sock_t kBadSock = -1;
void close_sock(sock_t s) { ::close(s); }
void shutdown_sock(sock_t s, int how) { ::shutdown(s, how == 0 ? SHUT_RD : how == 1 ? SHUT_WR : SHUT_RDWR); }
void net_init() {}
#endif

// Connects to host:port (host may be a name; no TLS, plain TCP as the RPC protocol is).
sock_t tcp_connect(const std::string& host, std::uint16_t port) {
  net_init();
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* res = nullptr;
  if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) != 0 || res == nullptr) return kBadSock;
  sock_t s = kBadSock;
  for (addrinfo* a = res; a != nullptr; a = a->ai_next) {
    s = ::socket(a->ai_family, a->ai_socktype, a->ai_protocol);
    if (s == kBadSock) continue;
    if (::connect(s, a->ai_addr, static_cast<int>(a->ai_addrlen)) == 0) break;
    close_sock(s);
    s = kBadSock;
  }
  freeaddrinfo(res);
  return s;
}

// Listens on 127.0.0.1 with an ephemeral port.
sock_t tcp_listen_loopback(std::uint16_t& port_out) {
  net_init();
  sock_t s = ::socket(AF_INET, SOCK_STREAM, 0);
  if (s == kBadSock) return kBadSock;
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  a.sin_port = 0;
  socklen_t len = sizeof a;
  if (::bind(s, reinterpret_cast<sockaddr*>(&a), sizeof a) != 0 || ::listen(s, 8) != 0 ||
      ::getsockname(s, reinterpret_cast<sockaddr*>(&a), &len) != 0) {
    close_sock(s);
    return kBadSock;
  }
  port_out = ntohs(a.sin_port);
  return s;
}

bool send_all(sock_t s, const char* data, std::size_t n) {
  while (n > 0) {
    const int w = static_cast<int>(::send(s, data, static_cast<int>(std::min<std::size_t>(n, 1u << 20)), 0));
    if (w <= 0) return false;
    data += w;
    n -= static_cast<std::size_t>(w);
  }
  return true;
}

bool readable(sock_t s, int timeout_ms) {
  fd_set set;
  FD_ZERO(&set);
  FD_SET(s, &set);
  timeval tv{timeout_ms / 1000, (timeout_ms % 1000) * 1000};
  return ::select(static_cast<int>(s) + 1, &set, nullptr, nullptr, &tv) > 0;
}

std::uint16_t free_loopback_port() {
  std::uint16_t port = 0;
  sock_t s = tcp_listen_loopback(port);
  if (s == kBadSock) return 0;
  close_sock(s);
  return port;
}

// ---- counting proxy ---------------------------------------------------------------------------------------

// Command names of the pinned RPC protocol (ggml-rpc.cpp enum rpc_cmd, protocol 7.0.0).
constexpr std::array<const char*, 18> kRpcCmdNames = {
    "alloc_buffer", "get_alignment", "get_max_size", "buffer_get_base", "free_buffer", "buffer_clear",
    "set_tensor", "set_tensor_hash", "get_tensor", "copy_tensor", "graph_compute", "get_device_memory",
    "init_tensor", "get_alloc_size", "hello", "device_count", "graph_recompute", "memset_tensor"};
constexpr std::size_t kRpcGraphCompute = 10, kRpcGraphRecompute = 16;

struct WireCounters {
  std::atomic<std::uint64_t> to_node{0}, from_node{0};  // raw bytes, client -> server and server -> client
  std::array<std::atomic<std::uint64_t>, kRpcCmdNames.size()> calls{}, payload{};
  std::atomic<bool> parse_ok{true};
};

struct WireSnapshot {
  std::uint64_t to_node = 0, from_node = 0;
  std::array<std::uint64_t, kRpcCmdNames.size()> calls{}, payload{};
  WireSnapshot operator-(const WireSnapshot& o) const {
    WireSnapshot d;
    d.to_node = to_node - o.to_node;
    d.from_node = from_node - o.from_node;
    for (std::size_t i = 0; i < calls.size(); ++i) {
      d.calls[i] = calls[i] - o.calls[i];
      d.payload[i] = payload[i] - o.payload[i];
    }
    return d;
  }
  WireSnapshot& operator+=(const WireSnapshot& o) {
    to_node += o.to_node;
    from_node += o.from_node;
    for (std::size_t i = 0; i < calls.size(); ++i) {
      calls[i] += o.calls[i];
      payload[i] += o.payload[i];
    }
    return *this;
  }
  std::uint64_t total_calls() const {
    std::uint64_t n = 0;
    for (auto c : calls) n += c;
    return n;
  }
  std::uint64_t graph_calls() const { return calls[kRpcGraphCompute] + calls[kRpcGraphRecompute]; }
};

// Relays one Node's RPC connections through this process and counts bytes and requests. The request stream is
// parsed only for its framing: | cmd (1 byte) | size (8 bytes, little-endian) | payload (size bytes) |.
class CountingProxy {
 public:
  static std::unique_ptr<CountingProxy> start(std::string host, std::uint16_t port) {
    auto p = std::unique_ptr<CountingProxy>(new CountingProxy(std::move(host), port));
    p->listener_ = tcp_listen_loopback(p->local_port_);
    if (p->listener_ == kBadSock) return nullptr;
    p->acceptor_ = std::thread([raw = p.get()] { raw->accept_loop(); });
    return p;
  }
  ~CountingProxy() { stop(); }
  std::uint16_t port() const { return local_port_; }
  const WireCounters& counters() const { return counters_; }

  WireSnapshot snapshot() const {
    WireSnapshot s;
    s.to_node = counters_.to_node.load();
    s.from_node = counters_.from_node.load();
    for (std::size_t i = 0; i < s.calls.size(); ++i) {
      s.calls[i] = counters_.calls[i].load();
      s.payload[i] = counters_.payload[i].load();
    }
    return s;
  }

  void stop() {
    if (stopping_.exchange(true)) return;
    if (listener_ != kBadSock) close_sock(listener_);
    {
      std::lock_guard<std::mutex> lk(mu_);
      for (sock_t s : live_) shutdown_sock(s, 2);
    }
    if (acceptor_.joinable()) acceptor_.join();
    for (auto& t : relays_)
      if (t.joinable()) t.join();
  }

 private:
  CountingProxy(std::string host, std::uint16_t port) : host_(std::move(host)), port_(port) {}

  void accept_loop() {
    while (!stopping_) {
      if (!readable(listener_, 100)) continue;
      sock_t c = ::accept(listener_, nullptr, nullptr);
      if (c == kBadSock) continue;
      sock_t u = tcp_connect(host_, port_);
      if (u == kBadSock) {
        close_sock(c);
        continue;
      }
      {
        std::lock_guard<std::mutex> lk(mu_);
        live_.push_back(c);
        live_.push_back(u);
      }
      relays_.emplace_back([this, c, u] { relay(c, u, true); });
      relays_.emplace_back([this, c, u] { relay(u, c, false); });
    }
  }

  void relay(sock_t from, sock_t to, bool request_direction) {
    std::array<char, 1 << 16> buf;
    // Request framing state.
    std::array<unsigned char, 9> head{};
    std::size_t head_have = 0;
    std::uint64_t skip = 0;
    std::size_t cmd = 0;
    bool parsing = true;
    for (;;) {
      const int n = static_cast<int>(::recv(from, buf.data(), static_cast<int>(buf.size()), 0));
      if (n <= 0) break;
      if (request_direction) {
        counters_.to_node += static_cast<std::uint64_t>(n);
        std::size_t i = 0;
        while (parsing && i < static_cast<std::size_t>(n)) {
          if (skip > 0) {
            const std::uint64_t take = std::min<std::uint64_t>(skip, static_cast<std::uint64_t>(n) - i);
            skip -= take;
            i += static_cast<std::size_t>(take);
            continue;
          }
          head[head_have++] = static_cast<unsigned char>(buf[i++]);
          if (head_have < head.size()) continue;
          head_have = 0;
          cmd = head[0];
          std::uint64_t size = 0;
          for (int b = 7; b >= 0; --b) size = (size << 8) | head[1 + static_cast<std::size_t>(b)];
          if (cmd >= kRpcCmdNames.size() || size > (1ull << 40)) {
            parsing = false;  // not the framing we know (a transport upgrade or another protocol version)
            counters_.parse_ok = false;
            break;
          }
          counters_.calls[cmd]++;
          counters_.payload[cmd] += size;
          skip = size;
        }
      } else {
        counters_.from_node += static_cast<std::uint64_t>(n);
      }
      if (!send_all(to, buf.data(), static_cast<std::size_t>(n))) break;
    }
    shutdown_sock(to, 1);
    shutdown_sock(from, 0);
  }

  std::string host_;
  std::uint16_t port_;
  std::uint16_t local_port_ = 0;
  sock_t listener_ = kBadSock;
  std::thread acceptor_;
  std::vector<std::thread> relays_;
  std::mutex mu_;
  std::vector<sock_t> live_;
  std::atomic<bool> stopping_{false};
  WireCounters counters_;
};

// ---- nodes -------------------------------------------------------------------------------------------------

struct HostPort {
  std::string host;
  std::uint16_t port = 0;
  std::string str() const { return host + ":" + std::to_string(port); }
};

Result<HostPort> parse_host_port(const std::string& text) {
  const auto colon = text.rfind(':');
  if (colon == std::string::npos || colon == 0 || colon + 1 >= text.size())
    return make_error(ErrorCode::kInvalidArgument, "node '" + text + "' is not host:port");
  try {
    const unsigned long port = std::stoul(text.substr(colon + 1));
    if (port == 0 || port > 65535) throw std::out_of_range("port");
    return HostPort{text.substr(0, colon), static_cast<std::uint16_t>(port)};
  } catch (const std::exception&) {
    return make_error(ErrorCode::kInvalidArgument, "node '" + text + "' has a bad port");
  }
}

bool is_loopback_host(const std::string& h) { return h == "127.0.0.1" || h == "localhost" || h == "::1" || h.rfind("127.", 0) == 0; }

void set_env(const char* name, const std::string& value) {
#ifdef _WIN32
  _putenv_s(name, value.c_str());
#else
  ::setenv(name, value.c_str(), 1);
#endif
}

bool probe_connect(const HostPort& hp, std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  do {
    sock_t s = tcp_connect(hp.host, hp.port);
    if (s != kBadSock) {
      close_sock(s);
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  } while (std::chrono::steady_clock::now() < deadline);
  return false;
}

struct DirUsage {
  std::uint64_t files = 0, bytes = 0;
  bool exists = false;
};

DirUsage dir_usage(const fs::path& dir) {
  DirUsage u;
  std::error_code ec;
  u.exists = fs::exists(dir, ec);
  if (!u.exists) return u;
  for (fs::recursive_directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
    ++u.files;  // directories count too: even an empty "rpc" cache directory means -c ran
    if (it->is_regular_file(ec)) u.bytes += it->file_size(ec);
  }
  return u;
}

struct LocalNode {
  HostPort hp;                                       // where the rpc-server listens
  std::unique_ptr<platform::ChildProcess> process;   // null for pre-existing Nodes
  fs::path cache_dir;                                // watched for filesystem effects (may be empty)
  std::unique_ptr<CountingProxy> proxy;
  DirUsage cache_before;
  bool with_cache_flag = false;
  std::size_t free_before = 0, total_before = 0, free_after = 0, total_after = 0;
};

std::uint64_t resident_bytes() {
#ifdef _WIN32
  return 0;  // the harness reports it as unavailable on Windows rather than guessing
#else
  std::ifstream f("/proc/self/status");
  std::string line;
  while (std::getline(f, line))
    if (line.rfind("VmRSS:", 0) == 0) return std::stoull(line.substr(6)) * 1024;
  return 0;
#endif
}

json snapshot_json(const WireSnapshot& s) {
  json j = {{"bytes_to_node", s.to_node}, {"bytes_from_node", s.from_node}, {"calls", s.total_calls()},
            {"graph_calls", s.graph_calls()}};
  json by_cmd = json::object();
  for (std::size_t i = 0; i < s.calls.size(); ++i)
    if (s.calls[i] != 0) by_cmd[kRpcCmdNames[i]] = {{"calls", s.calls[i]}, {"payload_bytes", s.payload[i]}};
  j["by_command"] = by_cmd;
  return j;
}

std::int32_t argmax_row(const float* row, std::int32_t n) {
  std::int32_t best = 0;
  for (std::int32_t i = 1; i < n; ++i)
    if (row[i] > row[best]) best = i;
  return best;
}

struct GenerationOutcome {
  double prefill_s = 0, decode_s = 0;
  std::vector<std::int32_t> tokens;
  std::uint64_t prompt_tokens = 0;
  WireSnapshot prefill_wire, decode_wire;
};

// Prompt prefill (chunks of n_batch) then n_predict greedy single-token decode steps. `wire` (may be empty)
// samples the summed wire counters at the phase boundaries.
GenerationOutcome generate(llama_context* ctx, std::uint32_t vocab, std::uint32_t prompt_tokens, std::uint32_t n_predict,
                           std::uint32_t n_batch, const std::function<WireSnapshot()>& wire) {
  GenerationOutcome out;
  out.prompt_tokens = prompt_tokens;
  llama_memory_clear(llama_get_memory(ctx), true);
  std::vector<llama_token> prompt(prompt_tokens);
  for (std::uint32_t i = 0; i < prompt_tokens; ++i) prompt[i] = static_cast<llama_token>((i * 7919u + 13u) % vocab);
  const WireSnapshot w0 = wire ? wire() : WireSnapshot{};
  Stopwatch sw;
  for (std::uint32_t at = 0; at < prompt_tokens; at += n_batch) {
    const std::uint32_t n = std::min(n_batch, prompt_tokens - at);
    if (llama_decode(ctx, llama_batch_get_one(prompt.data() + at, static_cast<std::int32_t>(n))) != 0) return GenerationOutcome{};
  }
  out.prefill_s = sw.elapsed_ms() / 1000.0;
  const WireSnapshot w1 = wire ? wire() : WireSnapshot{};
  out.prefill_wire = w1 - w0;
  llama_token next = argmax_row(llama_get_logits_ith(ctx, -1), static_cast<std::int32_t>(vocab));
  out.tokens.push_back(next);
  Stopwatch dsw;
  for (std::uint32_t i = 0; i < n_predict; ++i) {
    if (llama_decode(ctx, llama_batch_get_one(&next, 1)) != 0) return GenerationOutcome{};
    next = argmax_row(llama_get_logits_ith(ctx, -1), static_cast<std::int32_t>(vocab));
    out.tokens.push_back(next);
  }
  out.decode_s = dsw.elapsed_ms() / 1000.0;
  out.decode_wire = (wire ? wire() : WireSnapshot{}) - w1;
  return out;
}

void quiet_llama_log(ggml_log_level level, const char* text, void*) {
  if (level == GGML_LOG_LEVEL_ERROR && text != nullptr) std::fputs(text, stderr);
}

}  // namespace

int cmd_baseline_llama(const cli::Args& args) {
  Stopwatch total;
  const RunContext ctx = make_run_context(args);
  BenchmarkResult r(experiment_id(args, ctx, "HQ-P0A-01", "dev-baseline-llama-rpc"), probe_host());
  apply_run_context(r, ctx);
  r.backend("llama.cpp-rpc", std::string("llama.cpp@") + CLUSTERLM_LLAMA_PIN);
  r.pending("HQ-P0A-01");
  auto usage_error = [&](const std::string& m) {
    std::fprintf(stderr, "clusterlm-bench baseline llama-rpc: %s\n", m.c_str());
    return 2;
  };

  if (!args.has("model")) return usage_error("--model <file.gguf> is required");
  const fs::path model_path = args.get("model");
  if (!fs::is_regular_file(model_path)) return usage_error("model file not found: " + model_path.string());
  const auto prompt_tokens = static_cast<std::uint32_t>(args.integer("prompt-tokens", 32));
  const auto n_predict = static_cast<std::uint32_t>(args.integer("n-predict", 16));
  const auto repeats = static_cast<std::uint32_t>(args.integer("repeat", 3));
  if (prompt_tokens == 0 || n_predict == 0 || repeats == 0)
    return usage_error("--prompt-tokens, --n-predict and --repeat must be positive");
  const auto spawn_n = static_cast<std::size_t>(args.integer("spawn-local", 0));
  if (spawn_n == 0 && !args.has("nodes"))
    return usage_error("--nodes host:port,... (running the pinned rpc-server without -c) or --spawn-local N is required");
  if (spawn_n > 0 && args.has("nodes")) return usage_error("--nodes and --spawn-local are mutually exclusive");

  // The pin this build is tied to.
  {
    fs::path resolved = args.has("pin") ? fs::path(args.get("pin")) : fs::path("third_party") / "upstream.json";
    if (resolved.is_relative() && !fs::exists(resolved)) resolved = fs::path(source_dir()) / resolved;
    std::string pinned = "unreadable";
    std::ifstream f(resolved);
    if (f) {
      const auto j = json::parse(f, nullptr, false);
      if (j.is_object() && j.contains("pins") && j["pins"].contains("llama.cpp"))
        pinned = j["pins"]["llama.cpp"].value("commit", "unreadable");
    }
    r.check("pin_matches_build", pinned == CLUSTERLM_LLAMA_PIN,
            "upstream.json pins llama.cpp " + pinned + "; this build links " + CLUSTERLM_LLAMA_PIN);
    r.config("llama_cpp_pin", pinned);
  }

  const fs::path work = fs::temp_directory_path() / ("clusterlm-baseline-llama-" + std::to_string(monotonic_ns()));
  std::error_code ec;
  fs::create_directories(work, ec);
  set_env("GGML_RPC_NO_RDMA", "1");  // plain TCP: the byte counters assume the request framing stays visible

  // ---- Nodes ------------------------------------------------------------------------------------------------
  std::vector<LocalNode> nodes;
  const bool control = args.has("control-with-cache");
  if (spawn_n > 0) {
    fs::path server = args.has("rpc-server") ? fs::path(args.get("rpc-server")) : platform::executable_dir() / "ggml-rpc-server";
#ifdef _WIN32
    if (!args.has("rpc-server")) server += ".exe";
#endif
    if (!fs::exists(server)) return usage_error("rpc-server binary not found: " + server.string() + " (pass --rpc-server)");
    for (std::size_t i = 0; i < spawn_n; ++i) {
      LocalNode n;
      n.cache_dir = work / ("node" + std::to_string(i) + "-cache");
      fs::create_directories(n.cache_dir, ec);
      set_env("LLAMA_CACHE", n.cache_dir.string());  // where the server would put its tensor cache if -c were given
      const std::uint16_t port = free_loopback_port();
      if (port == 0) return usage_error("no free loopback port");
      n.hp = {"127.0.0.1", port};
      std::vector<std::string> argv = {"-H", "127.0.0.1", "-p", std::to_string(port), "-t",
                                       std::to_string(args.integer("server-threads", 2))};
      if (control) argv.push_back("-c");  // negative control: proves the filesystem check can see a cache
      n.with_cache_flag = control;
      auto proc = platform::ChildProcess::spawn(server, argv);
      if (!proc.is_ok()) return usage_error("cannot start rpc-server: " + proc.status().to_string());
      n.process = std::move(proc).value();
      if (!probe_connect(n.hp, std::chrono::seconds(15))) return usage_error("rpc-server did not start listening on " + n.hp.str());
      nodes.push_back(std::move(n));
    }
  } else {
    const auto dirs = args.has("node-cache-dirs") ? cli::split(args.get("node-cache-dirs"), ',') : std::vector<std::string>{};
    std::size_t i = 0;
    for (const auto& text : cli::split(args.get("nodes"), ',')) {
      auto hp = parse_host_port(text);
      if (!hp.is_ok()) return usage_error(hp.status().message());
      LocalNode n;
      n.hp = hp.value();
      if (i < dirs.size()) n.cache_dir = dirs[i];
      if (!probe_connect(n.hp, std::chrono::seconds(5))) return usage_error("cannot connect to node " + n.hp.str());
      nodes.push_back(std::move(n));
      ++i;
    }
  }
  if (nodes.empty()) return usage_error("no nodes");
  for (auto& n : nodes) {
    n.proxy = CountingProxy::start(n.hp.host, n.hp.port);
    if (!n.proxy) return usage_error("cannot start the counting proxy");
    if (!n.cache_dir.empty()) n.cache_before = dir_usage(n.cache_dir);
  }
  auto wire_total = [&nodes] {
    WireSnapshot s;
    for (const auto& n : nodes) s += n.proxy->snapshot();
    return s;
  };

  // ---- llama.cpp with RPC devices ------------------------------------------------------------------------------
  llama_log_set(quiet_llama_log, nullptr);
  llama_backend_init();
  std::vector<ggml_backend_dev_t> devices;
  std::vector<ggml_backend_dev_t> first_device;
  for (auto& n : nodes) {
    const std::string endpoint = "127.0.0.1:" + std::to_string(n.proxy->port());
    ggml_backend_reg_t reg = ggml_backend_rpc_add_server(endpoint.c_str());
    if (reg == nullptr) return usage_error("node " + n.hp.str() + " reports no RPC devices");
    ggml_backend_register(reg);
    for (std::size_t d = 0; d < ggml_backend_reg_dev_count(reg); ++d) devices.push_back(ggml_backend_reg_dev_get(reg, d));
    first_device.push_back(ggml_backend_reg_dev_get(reg, 0));
    ggml_backend_dev_memory(first_device.back(), &n.free_before, &n.total_before);
  }
  devices.push_back(nullptr);

  const std::int32_t ngl = args.has("ngl") ? static_cast<std::int32_t>(args.integer("ngl", 999)) : -1;
  llama_model_params mp = llama_model_default_params();
  mp.devices = devices.data();
  mp.n_gpu_layers = ngl;
  const WireSnapshot load0 = wire_total();
  const std::uint64_t rss0 = resident_bytes();
  Stopwatch load_sw;
  llama_model* model = llama_model_load_from_file(model_path.string().c_str(), mp);
  if (model == nullptr) {
    r.check("model_loaded_over_rpc", false, "llama.cpp could not load the model with RPC devices");
    for (auto& n : nodes) n.proxy->stop();
    return emit(args, r, total.elapsed_ms() / 1000.0);
  }
  const double load_s = load_sw.elapsed_ms() / 1000.0;
  const WireSnapshot load_wire = wire_total() - load0;
  const std::uint64_t rss_after_load = resident_bytes();
  r.check("model_loaded_over_rpc", true);
  const auto vocab = static_cast<std::uint32_t>(llama_vocab_n_tokens(llama_model_get_vocab(model)));
  char name_buf[256] = {};
  llama_model_meta_val_str(model, "general.name", name_buf, sizeof name_buf);
  const std::string general_name = name_buf;
  const bool fixture = general_name == "clusterlm-tiny-llama";

  const std::uint32_t n_batch = std::max<std::uint32_t>(1, std::min<std::uint32_t>(prompt_tokens, 512));
  llama_context_params cp = llama_context_default_params();
  cp.n_ctx = prompt_tokens + n_predict + 16;
  cp.n_batch = n_batch;
  cp.n_ubatch = n_batch;
  cp.n_threads = args.has("threads") ? static_cast<std::int32_t>(args.integer("threads", 2))
                                      : static_cast<std::int32_t>(std::max(1u, std::thread::hardware_concurrency() / 2));
  cp.n_threads_batch = cp.n_threads;
  cp.no_perf = true;
  llama_context* lctx = llama_init_from_model(model, cp);
  if (lctx == nullptr) {
    r.check("context_created", false, "llama_init_from_model failed");
    llama_model_free(model);
    for (auto& n : nodes) n.proxy->stop();
    return emit(args, r, total.elapsed_ms() / 1000.0);
  }
  // What each Node reports as free device memory after the load (the delta is the Node's allocation).
  for (std::size_t i = 0; i < nodes.size(); ++i) ggml_backend_dev_memory(first_device[i], &nodes[i].free_after, &nodes[i].total_after);

  Distribution prefill_tok_s, decode_tok_s, decode_bytes_per_token, prefill_bytes_per_token, graph_calls_per_token, calls_per_token;
  std::vector<std::int32_t> rpc_tokens;
  WireSnapshot decode_wire_total;
  for (std::uint32_t rep = 0; rep < repeats; ++rep) {
    GenerationOutcome g = generate(lctx, vocab, prompt_tokens, n_predict, n_batch, wire_total);
    if (g.tokens.empty()) {
      r.check("generation_ran", false, "llama_decode failed over RPC in repeat " + std::to_string(rep));
      break;
    }
    if (rep == 0) rpc_tokens = g.tokens;
    if (g.prefill_s > 0) prefill_tok_s.add(static_cast<double>(prompt_tokens) / g.prefill_s);
    if (g.decode_s > 0) decode_tok_s.add(static_cast<double>(n_predict) / g.decode_s);
    decode_bytes_per_token.add(static_cast<double>(g.decode_wire.to_node + g.decode_wire.from_node) / n_predict);
    prefill_bytes_per_token.add(static_cast<double>(g.prefill_wire.to_node + g.prefill_wire.from_node) / prompt_tokens);
    graph_calls_per_token.add(static_cast<double>(g.decode_wire.graph_calls()) / n_predict);
    calls_per_token.add(static_cast<double>(g.decode_wire.total_calls()) / n_predict);
    decode_wire_total += g.decode_wire;
  }
  const std::uint64_t rss_end = resident_bytes();

  llama_free(lctx);
  llama_model_free(model);

  // Optional: the same generation on this machine without RPC, to show the split run computes the same tokens.
  if (args.has("verify-local") && !rpc_tokens.empty()) {
    std::vector<ggml_backend_dev_t> none = {nullptr};
    llama_model_params lp = llama_model_default_params();
    lp.devices = none.data();
    lp.n_gpu_layers = 0;
    llama_model* local = llama_model_load_from_file(model_path.string().c_str(), lp);
    llama_context* lc = local != nullptr ? llama_init_from_model(local, cp) : nullptr;
    bool same = false;
    if (lc != nullptr) same = generate(lc, vocab, prompt_tokens, n_predict, n_batch, nullptr).tokens == rpc_tokens;
    if (lc != nullptr) llama_free(lc);
    if (local != nullptr) llama_model_free(local);
    r.check("tokens_match_local_run", same, "greedy tokens of the RPC-split run equal a local CPU run of the same model");
  }

  // ---- results -------------------------------------------------------------------------------------------------
  const bool any_loopback =
      std::any_of(nodes.begin(), nodes.end(), [](const LocalNode& n) { return is_loopback_host(n.hp.host); });
  if (ctx.on_target && !any_loopback && !fixture) r.set_measured();
  if (any_loopback) r.mark_simulated("localhost_cluster", true);
  if (fixture) r.mark_simulated("fixture_model", true);
  r.model({{"artifact_id", general_name.empty() ? model_path.filename().string() : general_name}, {"fixture", fixture}});
  r.config("model_file", model_path.filename().string());
  r.config("nodes", static_cast<std::uint64_t>(nodes.size()));
  r.config("prompt_tokens", prompt_tokens);
  r.config("n_predict", n_predict);
  r.config("repeat", repeats);
  r.config("n_gpu_layers", ngl);
  r.config("client_threads", cp.n_threads);
  r.config("rpc_server_cache_flag", control ? "-c (negative control)" : "off");
  r.config("rpc_transport", "TCP (GGML_RPC_NO_RDMA=1)");
  r.config("rpc_protocol", std::to_string(RPC_PROTO_MAJOR_VERSION) + "." + std::to_string(RPC_PROTO_MINOR_VERSION) + "." +
                               std::to_string(RPC_PROTO_PATCH_VERSION));
  r.config("synthetic_token_ids_on_rpc", true);  // the RPC arm sends graph inputs, tokens included; harness tokens are synthetic

  r.metric("prefill_tok_s", prefill_tok_s, "tokens/s");
  r.metric("decode_tok_s", decode_tok_s, "tokens/s");
  r.metric("rpc.bytes_per_token", decode_bytes_per_token, "bytes");
  r.metric("rpc.prefill_bytes_per_token", prefill_bytes_per_token, "bytes");
  r.metric("rpc.graph_calls_per_token", graph_calls_per_token, "calls");
  r.metric("rpc.calls_per_token", calls_per_token, "calls");
  r.metric("rpc.load_seconds", load_s);
  r.metric("rpc.load_bytes_to_nodes", load_wire.to_node);
  r.metric("rpc.set_tensor_payload_bytes_at_load", load_wire.payload[6]);
  r.metric("rpc.load", snapshot_json(load_wire));
  r.metric("rpc.decode_total", snapshot_json(decode_wire_total));
  r.metric("client.resident_bytes_before_load", rss0);
  r.metric("client.resident_bytes_after_load", rss_after_load);
  r.metric("client.resident_bytes_end", rss_end);
  bool framing_ok = true;
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    const std::string k = "node" + std::to_string(i) + ".";
    framing_ok = framing_ok && nodes[i].proxy->counters().parse_ok.load();
    r.metric(k + "endpoint_loopback", is_loopback_host(nodes[i].hp.host));
    r.metric(k + "device_memory_total_bytes", static_cast<std::uint64_t>(nodes[i].total_after));
    r.metric(k + "device_memory_used_by_load_bytes",
             static_cast<std::int64_t>(nodes[i].free_before) - static_cast<std::int64_t>(nodes[i].free_after));
    r.metric(k + "wire", snapshot_json(nodes[i].proxy->snapshot()));
  }
  r.check("rpc_request_framing_parsed", framing_ok, "every request matched the | cmd | size | payload | framing");

  for (auto& n : nodes) n.proxy->stop();
  for (auto& n : nodes)
    if (n.process) {
      n.process->kill();
      (void)n.process->wait(std::chrono::seconds(5));
    }

  // ---- filesystem effects on the Nodes ----------------------------------------------------------------------------
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    const auto& n = nodes[i];
    if (n.cache_dir.empty()) continue;
    const DirUsage after = dir_usage(n.cache_dir);
    const std::string base = "node" + std::to_string(i) + ".fs.";
    r.metric(base + "cache_dir_entries_before", n.cache_before.files);
    r.metric(base + "cache_dir_entries_after", after.files);
    r.metric(base + "cache_dir_bytes_after", after.bytes);
    if (n.with_cache_flag)
      r.check("control_cache_visible_node" + std::to_string(i), after.files > 0,
              "with -c the cache directory fills; the check sees it (" + std::to_string(after.files) + " entries)");
    else
      r.check("rpc_cache_empty_node" + std::to_string(i), after.files == n.cache_before.files && after.bytes == 0,
              "the rpc-server cache directory stays empty without -c (" + std::to_string(after.files) + " entries)");
  }
  if (args.has("node-fs-report")) {
    for (const auto& file : cli::split(args.get("node-fs-report"), ',')) {
      std::ifstream f(file);
      const auto j = f ? json::parse(f, nullptr, false) : json();
      if (!j.is_object()) {
        r.check("node_fs_report:" + file, false, "cannot read the report written by scripts/run_rpc_server.ps1 -Report");
        continue;
      }
      const auto entries = j.value("cache_entries", std::uint64_t{1});
      const bool flag = j.value("cache_flag_used", true);
      r.check("rpc_cache_empty:" + j.value("node", file), entries == 0 && !flag,
              "node report: " + std::to_string(entries) + " cache entries, -c " + (flag ? "used" : "not used"));
    }
  } else if (spawn_n == 0 && !args.has("node-cache-dirs")) {
    r.check("remote_node_filesystem_inspected", false,
            "remote Node filesystems were not inspected: pass --node-fs-report (scripts/run_rpc_server.ps1 -Report) or --node-cache-dirs");
  }
  std::error_code rm_ec;
  fs::remove_all(work, rm_ec);
  return emit(args, r, total.elapsed_ms() / 1000.0);
}

#else  // !CLUSTERLM_BENCH_HAS_LLAMA

int cmd_baseline_llama(const cli::Args& args) {
  BenchmarkResult r("dev-baseline-llama-rpc", probe_host());
  r.check("llama_built", false,
          "this clusterlm-bench was built without the pinned llama.cpp; configure with -DCLUSTERLM_ENABLE_LLAMA=ON "
          "(python3 scripts/fetch_upstream.py llama.cpp first). See docs/backends/llama-rpc.md.");
  r.pending("HQ-P0A-01");
  (void)emit(args, r, 0);
  return 3;
}

#endif

}  // namespace clusterlm::bench
