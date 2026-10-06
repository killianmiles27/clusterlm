// Seed-corpus generator for the fuzz targets: writes valid encodings (every protocol message, manifests,
// journals, catalog and profile JSON, expert-domain messages, text-parser inputs) so the fuzzers start from
// structurally valid inputs.
//
//   clusterlm_fuzz_gen_corpus <output dir>        e.g. tests/fuzz/corpus
//
// The files under tests/fuzz/corpus are committed; regenerate after changing a wire format. Reproducers of fixed
// findings live in tests/fuzz/corpus/<target>/crash-* and are NOT touched by this program.
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "../privacy/message_samples.hpp"
#include "clusterlm/placement/profile.hpp"
#include "clusterlm/transport/transport.hpp"
#ifdef CLUSTERLM_FUZZ_HAVE_EXPERT_DOMAINS
#include "clusterlm/expert_domains/wire.hpp"
#endif

using namespace clusterlm;
namespace fs = std::filesystem;

namespace {

fs::path g_out;
int g_count = 0;

void put_seed(const std::string& target, const std::string& name, const Bytes& bytes) {
  const fs::path dir = g_out / target;
  fs::create_directories(dir);
  std::ofstream f(dir / ("seed-" + name), std::ios::binary | std::ios::trunc);
  f.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  ++g_count;
}

Bytes with_prefix(std::initializer_list<std::uint8_t> prefix, const Bytes& body) {
  Bytes out(prefix);
  out.insert(out.end(), body.begin(), body.end());
  return out;
}

Bytes bytes_of(std::string_view s) { return Bytes(s.begin(), s.end()); }

// One transport frame exactly as framed.cpp writes it.
Bytes frame_bytes(std::uint16_t type, std::uint8_t channel, std::uint64_t corr, const Bytes& payload) {
  ByteWriter w;
  w.u32(transport::kFrameMagic);
  w.u16(transport::kFrameVersion);
  w.u16(type);
  w.u8(channel);
  w.u8(0);
  w.u16(0);
  w.u64(corr);
  w.u32(static_cast<std::uint32_t>(payload.size()));
  w.raw(payload);
  return std::move(w).take();
}

std::string read_text(const fs::path& p) {
  std::ifstream in(p, std::ios::binary);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: clusterlm_fuzz_gen_corpus <output dir>\n");
    return 2;
  }
  g_out = argv[1];
  const auto messages = testing::sample_messages(3);

  // ---- protocol messages: raw decode seeds and framed streams ------------------------------------------
  Bytes all_frames;
  for (const auto& m : messages) {
    const auto type = protocol::type_of(m);
    const Bytes payload = protocol::encode(m);
    const std::string name(protocol::to_string(type));
    const auto t = static_cast<std::uint16_t>(type);
    put_seed("protocol_decode", name, with_prefix({static_cast<std::uint8_t>(t & 0xFF), static_cast<std::uint8_t>(t >> 8)}, payload));
    const auto channel = static_cast<std::uint8_t>(protocol::channel_of(type));
    const Bytes fr = frame_bytes(t, channel, 17, payload);
    put_seed("frame_stream", name, with_prefix({static_cast<std::uint8_t>(3 | (channel << 2))}, fr));
    all_frames.insert(all_frames.end(), fr.begin(), fr.end());
  }
  // Two frames back to back, a frame with a hostile length, a truncated header.
  {
    const Bytes ping = protocol::encode(protocol::Ping{1});
    Bytes two = frame_bytes(static_cast<std::uint16_t>(protocol::MessageType::kPing), 0, 1, ping);
    const Bytes second = frame_bytes(static_cast<std::uint16_t>(protocol::MessageType::kPong), 0, 2,
                                     protocol::encode(protocol::Pong{1}));
    two.insert(two.end(), second.begin(), second.end());
    put_seed("frame_stream", "two-control-frames", with_prefix({0x03}, two));
    Bytes huge = frame_bytes(1, 0, 0, {});
    for (int i = 20; i < 24; ++i) huge[static_cast<std::size_t>(i)] = 0xFF;  // payload_len = 4 GiB - 1
    put_seed("frame_stream", "hostile-length", with_prefix({0x03}, huge));
    put_seed("frame_stream", "truncated-header", with_prefix({0x00}, Bytes(huge.begin(), huge.begin() + 11)));
  }

  // ---- activations --------------------------------------------------------------------------------------
  {
    const auto layout = testing::sample_layout();
    for (std::uint32_t q : {1u, 3u}) {
      ByteWriter w;
      testing::sample_activations(q, layout, 5).encode(w);
      put_seed("stage_activations", "q" + std::to_string(q), with_prefix({0x03}, w.bytes()));
    }
  }

  // ---- manifest -----------------------------------------------------------------------------------------
  {
    const objects::ModelManifest m = testing::sample_manifest();
    ByteWriter w;
    m.encode(w);
    put_seed("manifest_decode", "fixture", w.bytes());
    put_seed("manifest_json", "fixture", bytes_of(m.to_json()));
  }

  // ---- lease journal -------------------------------------------------------------------------------------
  put_seed("lease_journal", "released-lease", bytes_of("B 1\nF 1 obj-0.part\nF 1 obj-1.part\nR 1\n"));
  put_seed("lease_journal", "orphan-and-torn-tail", bytes_of("G 4\nB 5\nF 5 obj-12.part\nB 6\nF 6 obj-3.pa"));
  put_seed("lease_journal", "corrupt-lines", bytes_of("B 9\nF 9 ../../etc/passwd\nF 9 obj-4.part\nX\nR abc\nB 9\nR 9\n\n"));

  // ---- catalog / profiles -------------------------------------------------------------------------------
  {
    const fs::path root = CLUSTERLM_SOURCE_DIR;
    put_seed("catalog_json", "shipped", bytes_of(read_text(root / "fixtures/catalog/clusterlm-catalog.json")));
    for (const auto& e : fs::directory_iterator(root / "fixtures/profiles")) {
      const std::string name = e.path().stem().string();
      const bool network = name.rfind("network", 0) == 0;
      put_seed("profile_json", name, with_prefix({static_cast<std::uint8_t>(network ? 1 : 0)}, bytes_of(read_text(e.path()))));
    }
  }

  // ---- text parsers (byte 0 selects the parser) ------------------------------------------------------------
  put_seed("text_parsers", "endpoint-v4", with_prefix({0}, bytes_of("127.0.0.1:7001")));
  put_seed("text_parsers", "endpoint-v6", with_prefix({0}, bytes_of("[::1]:7002")));
  put_seed("text_parsers", "plan", with_prefix({1 + 6 * 15}, bytes_of("0-4@father,4-10@0,10-13@1,13-16@father")));
  put_seed("text_parsers", "fault-rule", with_prefix({2}, bytes_of("delay:type=22:nth=3:dir=send:ms=50")));
  put_seed("text_parsers", "digest-hex", with_prefix({3}, bytes_of(testing::sample_digest(1).hex())));
  put_seed("text_parsers", "hex", with_prefix({4}, bytes_of("00ff10Ab")));
  put_seed("text_parsers", "preset", with_prefix({5}, bytes_of("gige-simulated")));

#ifdef CLUSTERLM_FUZZ_HAVE_EXPERT_DOMAINS
  {
    expert_domains::ExpertBatch b;
    b.epoch = Epoch{3};
    b.window = WindowId{4};
    b.layer = 2;
    b.positions = 2;
    b.hidden = 4;
    b.activations.assign(8, 0.5f);
    b.routes = {{{0, 0.6f}, {3, 0.4f}}, {{1, 1.0f}}};
    put_seed("expert_wire", "batch", with_prefix({0}, expert_domains::encode(b)));
    expert_domains::ExpertResult r;
    r.epoch = Epoch{3};
    r.window = WindowId{4};
    r.layer = 2;
    r.positions = 2;
    r.hidden = 4;
    r.partial.assign(8, 0.25f);
    r.compute_ns = 99;
    r.experts_executed = 3;
    put_seed("expert_wire", "result", with_prefix({1}, expert_domains::encode(r)));
    expert_domains::ExpertError e;
    e.code = ErrorCode::kResourceExhausted;
    e.message = "no capacity";
    put_seed("expert_wire", "error", with_prefix({2}, expert_domains::encode(e)));
  }
#endif

  std::printf("wrote %d seed file(s) under %s\n", g_count, g_out.string().c_str());
  return 0;
}
