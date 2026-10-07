#pragma once
// Test support for the Strata backend: synthetic manifests (no weights) and a fake StrataEngine whose arithmetic
// makes every state transition observable (commit, abort, provisional sub-batch commits) without a device.
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "clusterlm/backends/strata/engine.hpp"
#include "clusterlm/common/digest.hpp"
#include "clusterlm/backends/strata/object_map.hpp"
#include "clusterlm/objects/manifest.hpp"

namespace clusterlm::strata_test {

namespace bs = backends::strata;

// Flash-Next's geometry (the Strata kernels' contract), PLE at layer id 2.
inline objects::ModelGeometry flash_next_geometry() {
  objects::ModelGeometry g;
  g.family = "qwen3.8-flash-next";
  g.n_layers = 48;
  g.hidden_size = 2560;
  g.residual_streams = 4;
  g.n_experts = 512;
  g.n_active_experts = 10;
  g.expert_ff = 640;
  g.shared_expert_ff = 640;
  g.n_heads = 24;
  g.n_kv_heads = 2;
  g.head_dim = 256;
  g.vocab_size = 248320;
  g.ple_layer = 2;
  g.ple_ngram = 3;
  g.ple_rows = 1u << 20;
  g.mtp_layers = 1;
  for (std::uint32_t l = 0; l < g.n_layers; ++l)
    g.layer_kinds.push_back(l % 4 == 3 ? objects::LayerKind::kFullAttention : objects::LayerKind::kRecurrent);
  return g;
}

// A small geometry for state-machine tests (hidden 8, hc 4: 44 floats per position).
inline objects::ModelGeometry tiny_geometry() {
  objects::ModelGeometry g = flash_next_geometry();
  g.family = "strata-test-tiny";
  g.n_layers = 8;
  g.hidden_size = 8;
  g.n_experts = 4;
  g.n_active_experts = 2;
  g.expert_ff = 32;
  g.shared_expert_ff = 32;
  g.n_heads = 2;
  g.n_kv_heads = 1;
  g.head_dim = 4;
  g.vocab_size = 16;
  g.ple_rows = 64;
  g.layer_kinds.resize(g.n_layers);
  return g;
}

// Every object of the model with plausible representations and byte sizes; one shard; no bytes anywhere.
inline objects::ModelManifest synthetic_manifest(const objects::ModelGeometry& g, std::string expert_quant) {
  objects::ModelManifest m;
  m.artifact_id = "strata-test:" + g.family;
  m.license = "test";
  m.geometry = g;
  std::uint64_t at = 0;
  auto add = [&](std::string name, objects::ObjectKind kind, std::optional<std::uint32_t> layer,
                 std::optional<std::uint32_t> expert, objects::Representation rep, std::uint64_t bytes) {
    objects::ManifestObject o;
    o.name = std::move(name);
    o.kind = kind;
    o.layer = layer;
    o.expert = expert;
    o.representation = std::move(rep);
    o.source_ranges.push_back({0, at, bytes});
    o.byte_size = bytes;
    at += (bytes + 63) / 64 * 64;
    m.objects.push_back(std::move(o));
  };
  objects::Representation dense{std::string(bs::kDenseQuantType), 0, bs::kDenseConversionVersion, true};
  objects::Representation exp{expert_quant, 0, bs::kExpertConversionVersion, true};
  const auto fmt = bs::expert_format(exp, g);
  const std::uint64_t expert_bytes = fmt.is_ok() ? fmt->total() : 1;
  const std::uint64_t H = g.hidden_size;
  add(std::string(objects::kEmbeddingObjectName), objects::ObjectKind::kEmbedding, {}, {}, dense, H * g.vocab_size / 2 + 4096);
  add(std::string(objects::kPleObjectName), objects::ObjectKind::kPleLookup, {}, {}, {"iq4_nl", 32, 0, true}, 4096);
  for (std::uint32_t L = 0; L < g.n_layers; ++L) {
    add(objects::dense_object_name(L), objects::ObjectKind::kLayerDense, L, {}, dense, 64 * H + 4096);
    add(objects::shared_expert_object_name(L), objects::ObjectKind::kSharedExpert, L, {}, dense, 3 * H * g.shared_expert_ff / 2 + 4096);
    for (std::uint32_t e = 0; e < g.n_experts; ++e)
      add(objects::expert_object_name(L, e), objects::ObjectKind::kRoutedExpert, L, e, exp, expert_bytes);
  }
  add(std::string(objects::kHeadObjectName), objects::ObjectKind::kOutputHead, {}, {}, dense, H * g.vocab_size / 2 + 4096);
  add(std::string(objects::kMtpObjectName), objects::ObjectKind::kMtp, {}, {}, {"q8_0", 32, 0, true}, 4096);
  m.shards.push_back({"strata-test.bin", at, {}});
  return m;
}

// A fake engine with observable, deterministic arithmetic:
//   * committed state per session = number of committed positions (n);
//   * a window's row t (absolute position p) of a token-free domain writes field-major outputs
//       R'  = 2*R + p + n,   bo' = bo - 1,   inj' = inj + 0.5
//   * the prefix writes R = token + p + n, bo = p, inj = n;
//   * the tail writes logits[t][v] = sum(R row t) + v + n.
// A tentative window becomes visible only through commit(keep) (n += keep); abort() discards it.
class FakeEngine final : public bs::StrataEngine {
 public:
  struct Call {
    std::string op;  // run | commit | abort | open | close
    SessionId session;
    std::uint64_t pos0 = 0;
    std::uint32_t n = 0;
    std::size_t tokens = 0;
    std::vector<float> handoff_in;
  };

  FakeEngine(domain::StageRole role, const objects::ModelGeometry& g, std::uint32_t max_window = 8)
      : g_(g), layout_(domain::BoundaryLayout::for_geometry(g)) {
    caps_.max_window = max_window;
    caps_.needs_tokens = role == domain::StageRole::kPrefix;
    caps_.has_head = role == domain::StageRole::kTail;
    caps_.vocab = caps_.has_head ? g.vocab_size : 0;
  }

  bs::EngineCaps caps() const override { return caps_; }
  Result<domain::DomainRequirements> requirements(const std::vector<std::string>& names) const override {
    domain::DomainRequirements r;
    r.gpu_weight_bytes = names.size();
    r.state_bytes = 1000;
    return r;
  }
  Status prepare(const objects::ObjectResolver&) override {
    prepared = true;
    return Status::ok();
  }
  Status open_session(SessionId s) override {
    calls.push_back(op("open", s));
    committed[s] = 0;
    tentative.erase(s);
    return Status::ok();
  }
  Status close_session(SessionId s) override {
    calls.push_back(op("close", s));
    committed.erase(s);
    tentative.erase(s);
    return Status::ok();
  }
  Status run(const bs::EngineWindow& w, domain::StageTiming& timing) override {
    Call c = op("run", w.session, w.positions);
    c.pos0 = w.pos0;
    c.tokens = w.tokens.size();
    c.handoff_in.assign(w.handoff_in.begin(), w.handoff_in.end());
    calls.push_back(std::move(c));
    timing.compute_ns = 1;
    if (!caps_.needs_tokens && !w.tokens.empty()) tokens_on_token_free_domain = true;
    if (fail_run_on == ++runs) return make_error(ErrorCode::kInternal, "injected run failure");
    auto it = committed.find(w.session);
    if (it == committed.end()) return make_error(ErrorCode::kNotFound, "fake: session not open");
    if (tentative.contains(w.session)) return make_error(ErrorCode::kFailedPrecondition, "fake: window already outstanding");
    const float n = static_cast<float>(it->second);
    const std::size_t T = w.positions, hc = layout_.residual_streams, H = layout_.hidden_size, r = hc * H;
    if (!w.handoff_out.empty()) {
      float* R = w.handoff_out.data();
      float* bo = R + T * r;
      float* inj = bo + T * H;
      for (std::size_t t = 0; t < T; ++t) {
        const float p = static_cast<float>(w.pos0 + t);
        for (std::size_t i = 0; i < r; ++i)
          R[t * r + i] = caps_.needs_tokens ? static_cast<float>(w.tokens[t]) + p + n : 2.0f * w.handoff_in[t * r + i] + p + n;
        for (std::size_t i = 0; i < H; ++i) bo[t * H + i] = caps_.needs_tokens ? p : w.handoff_in[T * r + t * H + i] - 1.0f;
        for (std::size_t i = 0; i < hc; ++i) inj[t * hc + i] = caps_.needs_tokens ? n : w.handoff_in[T * (r + H) + t * hc + i] + 0.5f;
      }
    }
    if (caps_.has_head) {
      for (std::size_t t = 0; t < T; ++t) {
        float s = 0;
        for (std::size_t i = 0; i < r; ++i) s += w.handoff_in[t * r + i];
        for (std::size_t v = 0; v < caps_.vocab; ++v) w.logits_out[t * caps_.vocab + v] = s + static_cast<float>(v) + n;
      }
    }
    tentative[w.session] = w.positions;
    return Status::ok();
  }
  Status commit(SessionId s, std::uint32_t keep) override {
    calls.push_back(op("commit", s, keep));
    if (fail_commit) return make_error(ErrorCode::kInternal, "injected commit failure");
    auto it = tentative.find(s);
    if (it == tentative.end() || keep == 0 || keep > it->second) return make_error(ErrorCode::kFailedPrecondition, "fake: bad commit");
    committed[s] += keep;
    tentative.erase(it);
    return Status::ok();
  }
  Status abort(SessionId s) override {
    calls.push_back(op("abort", s));
    tentative.erase(s);
    return Status::ok();
  }
  Status release() override {
    prepared = false;
    committed.clear();
    tentative.clear();
    return Status::ok();
  }
  bs::EngineCounters counters() const override { return {123, 77, 55}; }

  static Call op(std::string name, SessionId s, std::uint32_t n = 0) {
    Call c;
    c.op = std::move(name);
    c.session = s;
    c.n = n;
    return c;
  }

  std::size_t count(const std::string& name) const {
    return static_cast<std::size_t>(std::count_if(calls.begin(), calls.end(), [&](const Call& c) { return c.op == name; }));
  }

  objects::ModelGeometry g_;
  domain::BoundaryLayout layout_;
  bs::EngineCaps caps_;
  bool prepared = false;
  std::map<SessionId, std::uint64_t> committed;
  std::map<SessionId, std::uint32_t> tentative;
  std::vector<Call> calls;
  bool tokens_on_token_free_domain = false;
  std::uint64_t runs = 0, fail_run_on = 0;
  bool fail_commit = false;
};

// A resolver that resolves nothing (the fake engine binds no bytes).
class NullResolver final : public objects::ObjectResolver {
 public:
  Result<objects::ProvisionedObject> resolve(std::string_view name) const override {
    return make_error(ErrorCode::kNotFound, "null resolver: " + std::string(name));
  }
};

// A model directory with Strata's object names (the tiny geometry; pseudo-random bytes, real digests): what a Coordinator over
// the fake engine needs as `model_dir`. Returns its manifest.
inline objects::ModelManifest write_synthetic_model(const std::filesystem::path& dir) {
  objects::ModelManifest m = synthetic_manifest(tiny_geometry(), "q8_0");
  Bytes shard(m.shards.at(0).byte_size, 0);
  std::uint64_t x = 0x9E3779B97F4A7C15ull;
  for (auto& b : shard) {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    b = static_cast<std::uint8_t>(x);
  }
  for (auto& o : m.objects) {
    Sha256 h;
    for (const auto& r : o.source_ranges) h.update(ByteSpan(shard).subspan(r.offset, r.length));
    o.source_digest = h.finish();
    o.object_digest = o.source_digest;
  }
  m.shards[0].digest = Sha256::of(shard);
  std::filesystem::create_directories(dir);
  std::ofstream(dir / m.shards[0].file_name, std::ios::binary)
      .write(reinterpret_cast<const char*>(shard.data()), static_cast<std::streamsize>(shard.size()));
  const std::string json = m.to_json();
  std::ofstream(dir / "manifest.json", std::ios::binary).write(json.data(), static_cast<std::streamsize>(json.size()));
  return m;
}

}  // namespace clusterlm::strata_test
