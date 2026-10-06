// Security hardening regressions: allocation bounds before allocation, Node plan admission, generated-names-only
// staging, process launch quoting, and the "no shell anywhere" source audit. Each case names the finding it pins
// (see docs/security/security-review.md).
#include <doctest/doctest.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <regex>
#include <set>
#include <sstream>

#include "../objects/test_util.hpp"
#include "../privacy/message_samples.hpp"
#include "../privacy/raw_client.hpp"
#include "clusterlm/catalog/catalog.hpp"
#include "clusterlm/domain/boundary.hpp"
#include "clusterlm/platform/command_line.hpp"
#include "clusterlm/transport/testing.hpp"

using namespace clusterlm;
using namespace clusterlm::testing;
using namespace clusterlm::protocol;
namespace fs = std::filesystem;
using namespace std::chrono_literals;

// ---------------------------------------------------------------------------------------------------------------
// F-01 StageActivations: hc*H + H + hc is a 32-bit expression; dimensions that pass the per-field bound could wrap it
// ---------------------------------------------------------------------------------------------------------------

TEST_CASE("activation dimensions whose record size wraps 32 bits are rejected before any allocation") {
  ByteWriter w;
  w.u32(static_cast<std::uint32_t>(domain::BoundaryAbi::kResidualHandoffF32V1));
  w.u32(1u << 20);  // hc: passes the old per-field bound
  w.u32(1u << 20);  // H:  passes the old per-field bound; hc*H = 2^40 wraps a u32 to 0
  w.u64(0);
  w.u32(4096);
  ByteReader r(w.bytes());
  auto a = domain::StageActivations::decode(r, 4096);
  REQUIRE_FALSE(a.is_ok());
  CHECK(a.status().code() == ErrorCode::kProtocolError);

  domain::StageActivations bad;
  bad.layout.residual_streams = 1u << 20;
  bad.layout.hidden_size = 1u << 20;
  CHECK_FALSE(bad.validate().is_ok());
}

// F-02 Geometry: the dimensions of a manifest size every scratch/KV/boundary buffer of a domain.
TEST_CASE("manifest geometry has sanity ceilings so a hostile manifest cannot size buffers from a raw u32") {
  objects::ModelGeometry g = sample_manifest().geometry;
  REQUIRE(g.validate().is_ok());
  auto with = [&](auto&& mutate) {
    objects::ModelGeometry m = g;
    mutate(m);
    return m.validate();
  };
  CHECK_FALSE(with([](auto& m) { m.hidden_size = 1u << 30; }).is_ok());
  CHECK_FALSE(with([](auto& m) { m.residual_streams = 1u << 20; }).is_ok());
  CHECK_FALSE(with([](auto& m) { m.n_experts = 1u << 30; }).is_ok());
  CHECK_FALSE(with([](auto& m) { m.expert_ff = 1u << 30; }).is_ok());
  CHECK_FALSE(with([](auto& m) { m.n_heads = 1u << 30; m.n_kv_heads = 1u << 29; }).is_ok());
  CHECK_FALSE(with([](auto& m) { m.head_dim = 1u << 30; }).is_ok());
  CHECK_FALSE(with([](auto& m) { m.vocab_size = 0xFFFFFFFFu; }).is_ok());
  CHECK_FALSE(with([](auto& m) { m.ple_rows = 0xFFFFFFFFu; }).is_ok());
  // Real Flash-Next geometry stays far inside the ceilings.
  CHECK(with([](auto& m) {
          m.hidden_size = 2560;
          m.residual_streams = 4;
          m.n_experts = 512;
          m.n_active_experts = 10;
          m.expert_ff = 640;
          m.shared_expert_ff = 640;
          m.n_heads = 24;
          m.n_kv_heads = 2;
          m.head_dim = 256;
          m.vocab_size = 262144;
        }).is_ok());
}

// F-03 Manifest validation: the pairwise overlap scan was quadratic in a wire-controlled range count, and lengths
// were summed without an overflow check.
TEST_CASE("manifest range validation is not quadratic and detects overflow") {
  objects::ModelManifest m = sample_manifest();
  REQUIRE(m.validate().is_ok());
  objects::ManifestObject o = m.objects.front();
  const std::uint32_t n = 60000;
  m.shards[0].byte_size = std::max<std::uint64_t>(m.shards[0].byte_size, std::uint64_t{n} * 8 + 64);
  o.name = "many-ranges";
  o.source_ranges.clear();
  for (std::uint32_t i = 0; i < n; ++i) o.source_ranges.push_back(objects::SourceRange{0, std::uint64_t{i} * 8, 4});
  o.byte_size = std::uint64_t{n} * 4;
  o.representation.conversion_version = 0;
  m.objects.push_back(o);
  const auto t0 = std::chrono::steady_clock::now();
  CHECK(m.validate().is_ok());
  // One overlapping pair at the very end must still be found.
  m.objects.back().source_ranges.push_back(objects::SourceRange{0, std::uint64_t{n - 1} * 8 + 2, 4});
  m.objects.back().byte_size += 4;
  CHECK_FALSE(m.validate().is_ok());
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
  CHECK(ms < 2000);  // the pairwise scan needed ~1.8e9 comparisons here

  // Lengths whose sum wraps 2^64 cannot make byte_size == sum hold.
  objects::ModelManifest w = sample_manifest();
  w.shards[0].byte_size = ~std::uint64_t{0} - 1;
  w.shards.push_back(w.shards[0]);
  objects::ManifestObject big = w.objects.front();
  big.name = "overflow";
  big.source_ranges = {objects::SourceRange{0, 0, 1ull << 63}, objects::SourceRange{1, 0, (1ull << 63) + 5}};
  big.byte_size = 5;  // == (2^63 + 2^63 + 5) mod 2^64
  w.objects.push_back(big);
  CHECK_FALSE(w.validate().is_ok());
}

// F-04 Protocol: counts must be backed by remaining input before the vector is sized.
TEST_CASE("protocol counts that the payload cannot back are rejected") {
  {  // ProvisionStatus claiming 2^20 sealed objects with no data
    ByteWriter w;
    w.u64(7);
    w.u32(1u << 20);
    CHECK_FALSE(decode(MessageType::kProvisionStatus, w.bytes()).is_ok());
  }
  {  // PreparePlan: 64 stages claimed, none present
    ByteWriter w;
    w.u64(7);
    w.raw(Bytes(32, 0));
    w.str("b");
    w.raw(Bytes(32, 0));
    w.u32(64);
    CHECK_FALSE(decode(MessageType::kPreparePlan, w.bytes()).is_ok());
  }
  {  // StageResult: 64 timings claimed, none present
    ByteWriter w;
    w.u64(1);
    w.u64(1);
    w.u64(1);
    w.u32(1);
    w.u16(static_cast<std::uint16_t>(ErrorCode::kInternal));
    w.str("e");
    w.boolean(false);
    w.u32(64);
    CHECK_FALSE(decode(MessageType::kStageResult, w.bytes()).is_ok());
  }
}

// F-05 Framing: a 24-byte header must not pin max_payload bytes of memory.
TEST_CASE("frame reader accepts large frames incrementally and rejects hostile lengths") {
  // A 3.5 MiB frame (several growth steps) arrives intact.
  Bytes payload(3'500'000);
  for (std::size_t i = 0; i < payload.size(); ++i) payload[i] = static_cast<std::uint8_t>(i * 31);
  ByteWriter w;
  w.u32(transport::kFrameMagic);
  w.u16(transport::kFrameVersion);
  w.u16(7);
  w.u8(1);
  w.u8(0);
  w.u16(0);
  w.u64(99);
  w.u32(static_cast<std::uint32_t>(payload.size()));
  w.raw(payload);
  {
    auto c = transport::testing::make_byte_source_connection(w.bytes());
    auto f = c->receive(0ms);
    REQUIRE(f.is_ok());
    CHECK(f->payload == payload);
    CHECK(f->correlation == 99);
  }
  // Header claims 4 GiB - 1: refused on the header alone.
  Bytes hostile = w.bytes();
  hostile.resize(transport::kFrameHeaderSize);
  for (int i = 20; i < 24; ++i) hostile[static_cast<std::size_t>(i)] = 0xFF;
  auto c = transport::testing::make_byte_source_connection(hostile);
  auto f = c->receive(0ms);
  REQUIRE_FALSE(f.is_ok());
  CHECK(f.status().code() == ErrorCode::kProtocolError);
  // Header claims 3 MiB but only 10 bytes follow: the receive fails without having allocated 3 MiB up front
  // (it can only fail, the stream ends), and a custom limit below the claim is a protocol error.
  Bytes truncated = w.bytes();
  truncated.resize(transport::kFrameHeaderSize + 10);
  auto t = transport::testing::make_byte_source_connection(truncated);
  CHECK_FALSE(t->receive(0ms).is_ok());
  auto small_limit = transport::testing::make_byte_source_connection(w.bytes(), 1u << 20);
  auto sl = small_limit->receive(0ms);
  REQUIRE_FALSE(sl.is_ok());
  CHECK(sl.status().code() == ErrorCode::kProtocolError);
}

// F-06 Catalog JSON depth guard could be "banked" with stray closing brackets.
TEST_CASE("catalog nesting guard counts depth, not balance") {
  const std::string deep = std::string(40, '}') + std::string(40, '[') + std::string(40, ']');
  auto c = catalog::Catalog::parse(deep);
  REQUIRE_FALSE(c.is_ok());
  CHECK(c.status().message().find("nesting") != std::string::npos);
}

// F-07 Canonical store allocated byte_size before refusing a converted object.
TEST_CASE("a converted object with an absurd byte_size is refused without allocating it") {
  const auto& fx = clusterlm::testutil::tiny_fixture();
  objects::ModelManifest m = fx.manifest;
  auto& victim = m.objects[1];
  victim.representation.conversion_version = 1;
  victim.byte_size = 1ull << 44;  // 16 TiB; not bounded by the shard sizes for a converted object
  auto store = objects::CanonicalModelStore::open(fx.dir.path(), m);
  REQUIRE(store.is_ok());
  // Pre-converted form (source digest == object digest): byte_size must equal the stored ranges.
  auto r = store.value()->read_object_bytes(victim.name);
  REQUIRE_FALSE(r.is_ok());
  CHECK(r.status().code() == ErrorCode::kDataLoss);
  // A conversion the store would have to perform itself is refused outright.
  victim.source_digest.bytes[0] ^= 1;
  auto store2 = objects::CanonicalModelStore::open(fx.dir.path(), m);
  REQUIRE(store2.is_ok());
  auto r2 = store2.value()->read_object_bytes(victim.name);
  REQUIRE_FALSE(r2.is_ok());
  CHECK(r2.status().code() == ErrorCode::kUnimplemented);
}

// ---------------------------------------------------------------------------------------------------------------
// Node plan admission (F-08, F-09): hostile PreparePlans are refused before anything is allocated
// ---------------------------------------------------------------------------------------------------------------

namespace {

struct NodeUnderTest {
  fs::path dir = privacy_temp_dir("hardening");
  std::unique_ptr<node::NodeWorker> worker;
  std::unique_ptr<MessageStream> control;
  LeaseGeneration lease;
  objects::ModelManifest manifest = sample_manifest();

  explicit NodeUnderTest(std::uint64_t disk = 0) {
    node::NodeConfig nc;
    nc.name = "hardening";
    nc.security = insecure_security();
    nc.staging_root = dir / "staging";
    nc.ram_allowance = 1ull << 30;
    nc.vram_allowance = 1ull << 30;
    nc.disk_allowance = disk;
    auto started = node::NodeWorker::start(nc);
    REQUIRE(started.is_ok());
    worker = std::move(started).value();
    lease = LeaseGeneration{worker->status().lease_generation};
    auto s = open_stream(worker->endpoint(), insecure_security(), Channel::kControl, NodeRole::kFather, "father", lease);
    REQUIRE(s.is_ok());
    control = std::move(s).value();
    REQUIRE(std::holds_alternative<HelloAck>(next_message(*control).value()));
    REQUIRE(std::holds_alternative<OfferResources>(next_message(*control).value()));
  }
  ~NodeUnderTest() {
    control->close();
    worker->stop();
    std::error_code ec;
    fs::remove_all(dir, ec);
  }

  PreparePlan plan(const objects::ModelManifest& m, std::uint32_t max_context = 64, std::uint32_t max_window = 8) const {
    PreparePlan p;
    p.lease = lease;
    p.model_root = m.root_hash();
    p.backend_build = "reference-cpu-fp32-v1";
    p.plan_hash = sample_digest(4);
    p.stages.push_back(StageAssignment{StageId{1}, domain::StageRole::kMiddle, objects::LayerRange{3, 4}, max_context, max_window, 4});
    p.manifest = m;
    for (std::uint32_t i = 0; i < m.objects.size(); ++i)
      if (!m.objects[i].father_only() && m.objects[i].layer && *m.objects[i].layer == 3)
        p.assignments.push_back(ObjectAssignment{i, objects::AllocationTarget::kCpuResident});
    return p;
  }

  Message ask(const Message& m) {
    REQUIRE(control->send(m).is_ok());
    auto r = next_message(*control);
    REQUIRE(r.is_ok());
    return std::move(r).value();
  }
};

ErrorCode code_of(const Message& m) {
  REQUIRE(std::holds_alternative<ErrorMessage>(m));
  return std::get<ErrorMessage>(m).code;
}

}  // namespace

TEST_CASE("Node admits stage working memory (max_context/max_window) against its RAM allowance") {
  NodeUnderTest n;
  CHECK(code_of(n.ask(n.plan(n.manifest, 0xFFFFFFFFu, 0xFFFFFFFFu))) == ErrorCode::kResourceExhausted);
  CHECK(n.worker->status().state == node::NodeState::kAvailable);  // nothing was created
  // The same plan with sane limits is accepted.
  CHECK(std::holds_alternative<PlanAccepted>(n.ask(n.plan(n.manifest))));
}

TEST_CASE("Node refuses plans that assign an object twice, assign Father-only objects, or host a non-middle stage") {
  NodeUnderTest n;
  {
    PreparePlan p = n.plan(n.manifest);
    p.assignments.push_back(p.assignments.front());
    CHECK(code_of(n.ask(p)) == ErrorCode::kInvalidArgument);
  }
  {
    PreparePlan p = n.plan(n.manifest);
    p.stages[0].role = domain::StageRole::kTail;
    CHECK(code_of(n.ask(p)) == ErrorCode::kPermissionDenied);
  }
  {
    PreparePlan p = n.plan(n.manifest);
    for (std::uint32_t i = 0; i < n.manifest.objects.size(); ++i)
      if (n.manifest.objects[i].father_only()) {
        p.assignments.push_back(ObjectAssignment{i, objects::AllocationTarget::kCpuResident});
        break;
      }
    CHECK(code_of(n.ask(p)) == ErrorCode::kPermissionDenied);
  }
  CHECK(n.worker->status().state == node::NodeState::kAvailable);
}

TEST_CASE("assigned object sizes are summed saturating: a wrapped total cannot slip under the RAM budget") {
  NodeUnderTest n;
  PreparePlan p = n.plan(n.manifest);
  // Two objects of 2^63 bytes each sum to 0 modulo 2^64. (Unconverted objects must equal their ranges, so they
  // are marked converted; the Node must still refuse them.)
  REQUIRE(p.assignments.size() >= 2);
  for (std::size_t k = 0; k < 2; ++k) {
    auto& o = p.manifest.objects[p.assignments[k].object_index];
    o.representation.conversion_version = 1;
    o.byte_size = 1ull << 63;
  }
  CHECK(code_of(n.ask(p)) == ErrorCode::kResourceExhausted);
  CHECK(n.worker->status().state == node::NodeState::kAvailable);
}

// F-10 Lease store file names are store-generated; manifest object names (attacker text) never become paths.
TEST_CASE("manifest object names never become staging paths") {
  NodeUnderTest n(/*disk=*/1ull << 30);
  PreparePlan p = n.plan(n.manifest);
  // Extra, unreferenced objects with path-traversal and device names; staged on disk so file names matter.
  const std::vector<std::string> evil = {"../../escape", "..\\..\\escape", "/etc/cron.d/x", "C:\\Windows\\evil", "CON", "a/b/c",
                                         std::string("nul\0byte", 8)};
  std::set<std::string> expected = {"journal.log"};
  for (std::size_t k = 0; k < evil.size(); ++k) {
    objects::ManifestObject o = n.manifest.objects[n.manifest.objects.size() - 1];  // any valid object as a template
    o.kind = objects::ObjectKind::kLayerDense;
    o.layer = 3;
    o.expert.reset();
    o.name = evil[k];
    o.dependencies.clear();
    p.manifest.objects.push_back(o);
    const auto index = static_cast<std::uint32_t>(p.manifest.objects.size() - 1);
    p.assignments.push_back(ObjectAssignment{index, objects::AllocationTarget::kTemporaryBacking});
    expected.insert("obj-" + std::to_string(index) + ".part");
  }
  REQUIRE(std::holds_alternative<PlanAccepted>(n.ask(p)));

  // Everything the Node created lives under <staging>/ and is journal.log or leases/<gen>/obj-<index>.part.
  std::set<std::string> found;
  std::size_t files_outside = 0;
  const fs::path staging = n.dir / "staging";
  for (const auto& e : fs::recursive_directory_iterator(staging)) {
    if (e.is_regular_file()) found.insert(e.path().filename().string());
  }
  for (const auto& e : fs::recursive_directory_iterator(n.dir))
    if (e.is_regular_file() && e.path().string().rfind(staging.string(), 0) != 0) ++files_outside;
  CHECK(files_outside == 0);
  CHECK(found == expected);
  CHECK_FALSE(fs::exists(n.dir.parent_path() / "escape"));
}

// ---------------------------------------------------------------------------------------------------------------
// Process launch: one quoted command line, no shell
// ---------------------------------------------------------------------------------------------------------------

namespace {

// Reference implementation of the MSVC C runtime / CommandLineToArgvW argument parsing (the consumer of what
// ChildProcess::spawn builds on Windows). Deliberately independent of the quoting code under test.
std::vector<std::string> parse_command_line(const std::string& cmd) {
  std::vector<std::string> out;
  std::size_t i = 0;
  const std::size_t n = cmd.size();
  while (i < n) {
    while (i < n && (cmd[i] == ' ' || cmd[i] == '\t')) ++i;
    if (i >= n) break;
    std::string arg;
    bool in_quote = false;
    while (i < n) {
      std::size_t backslashes = 0;
      while (i < n && cmd[i] == '\\') {
        ++backslashes;
        ++i;
      }
      if (i < n && cmd[i] == '"') {
        arg.append(backslashes / 2, '\\');
        if (backslashes % 2 == 1) {
          arg.push_back('"');  // escaped quote
        } else if (in_quote && i + 1 < n && cmd[i + 1] == '"') {
          arg.push_back('"');  // "" inside quotes is a literal quote (post-2008 CRT rule)
          ++i;
        } else {
          in_quote = !in_quote;
        }
        ++i;
      } else {
        arg.append(backslashes, '\\');
        if (i >= n) break;
        if (!in_quote && (cmd[i] == ' ' || cmd[i] == '\t')) break;
        arg.push_back(cmd[i]);
        ++i;
      }
    }
    out.push_back(arg);
  }
  return out;
}

}  // namespace

TEST_CASE("Windows command-line quoting round-trips every awkward argument through the CRT parsing rules") {
  const std::vector<std::string> args = {"",
                                         "plain",
                                         "with space",
                                         "tab\there",
                                         "quote\"inside",
                                         "\"",
                                         "\"\"",
                                         "\\",
                                         "\\\\",
                                         "trailing\\",
                                         "trailing\\\\",
                                         "trailing space \\",
                                         "C:\\Program Files\\ClusterLM\\",
                                         "a\\\"b",
                                         "a\\\\\"b",
                                         "a\\\\\\\"b c",
                                         "&|<>^%!()",
                                         "--flag=\"x y\"",
                                         "multiple   spaces",
                                         "caf\xc3\xa9 \xe2\x82\xac",
                                         "line\nbreak",
                                         "--key=value with \\\"quoted\\\" parts"};
  const std::string exe = "C:\\Program Files\\ClusterLM\\clusterlm-node.exe";
  const std::string cmd = platform::build_windows_command_line<char>(exe, args);
  auto parsed = parse_command_line(cmd);
  REQUIRE(parsed.size() == args.size() + 1);
  CHECK(parsed[0] == exe);
  for (std::size_t i = 0; i < args.size(); ++i) {
    INFO("argument #" << i << ": " << args[i]);
    CHECK(parsed[i + 1] == args[i]);
  }
  // Arguments that need no quoting are passed through untouched.
  CHECK(platform::quote_windows_argument<char>("plain-arg_1.2") == "plain-arg_1.2");
  CHECK(platform::quote_windows_argument<char>("") == "\"\"");
  // The wide instantiation used by CreateProcessW compiles and agrees.
  CHECK(platform::quote_windows_argument<wchar_t>(L"a b") == L"\"a b\"");
}

TEST_CASE("source audit: no shell, no system(), no popen, no exec; process launch lives only in ChildProcess") {
  const fs::path root = CLUSTERLM_SOURCE_DIR;
  const std::vector<std::string> dirs = {"runtime", "node", "orchestrator", "apps", "bench", "experimental"};
  const std::regex banned_call(
      R"((^|[^A-Za-z0-9_.:>])(system|popen|_popen|_wpopen|fork|vfork|execl|execle|execlp|execv|execve|execvp|execvpe|WinExec|ShellExecute[AWEx]*)\s*\()");
  const std::regex banned_text(R"(cmd\.exe|/bin/sh|/bin/bash|powershell|comspec|\bsh -c|\bbash -c)", std::regex::icase);
  const std::regex launch_api(R"(CreateProcess[AW]?|posix_spawn[a-z_]*\s*\()");
  // Reviewed launchers: ChildProcess (argv quoted by build_windows_command_line) and the session-helper launcher,
  // which starts only the installed clusterlm-node-helper.exe into the user's session with a fixed argument list.
  const std::vector<std::string> launchers = {
      (fs::path("runtime") / "platform" / "src" / "process.cpp").generic_string(),
      (fs::path("runtime") / "platform" / "src" / "windows" / "helper_launch_win.cpp").generic_string()};

  // Strips a trailing // comment (outside string literals).
  auto code_of_line = [](const std::string& line) {
    bool in_str = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
      if (line[i] == '\\' && in_str) {
        ++i;
      } else if (line[i] == '"') {
        in_str = !in_str;
      } else if (!in_str && line[i] == '/' && i + 1 < line.size() && line[i + 1] == '/') {
        return line.substr(0, i);
      }
    }
    return line;
  };

  std::size_t files = 0;
  std::vector<std::string> violations;
  for (const auto& d : dirs) {
    for (const auto& e : fs::recursive_directory_iterator(root / d)) {
      if (!e.is_regular_file()) continue;
      const auto ext = e.path().extension().string();
      if (ext != ".cpp" && ext != ".hpp") continue;
      ++files;
      const std::string rel = fs::relative(e.path(), root).generic_string();
      std::ifstream in(e.path());
      std::string line;
      std::size_t lineno = 0;
      while (std::getline(in, line)) {
        ++lineno;
        const std::string code = code_of_line(line);
        const std::string where = rel + ":" + std::to_string(lineno);
        if (std::regex_search(code, banned_call)) violations.push_back(where + " banned call: " + code);
        if (std::regex_search(code, banned_text)) violations.push_back(where + " shell reference: " + code);
        if (std::find(launchers.begin(), launchers.end(), rel) == launchers.end() && std::regex_search(code, launch_api)) violations.push_back(where + " process launch outside ChildProcess: " + code);
      }
    }
  }
  CHECK(files > 100);  // the audit really walked the tree
  for (const auto& v : violations) MESSAGE(v);
  CHECK(violations.empty());
}
