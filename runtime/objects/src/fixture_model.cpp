#include "clusterlm/objects/fixture_model.hpp"

#include <cmath>
#include <fstream>

#include "clusterlm/objects/tensor_codec.hpp"

namespace clusterlm::objects {

namespace {

constexpr std::size_t kTensorAlign = 64;

std::uint64_t splitmix64(std::uint64_t& s) {
  s += 0x9E3779B97F4A7C15ull;
  std::uint64_t z = s;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

std::uint64_t fnv1a(std::string_view s) {
  std::uint64_t h = 0xCBF29CE484222325ull;
  for (char c : s) h = (h ^ static_cast<std::uint8_t>(c)) * 0x100000001B3ull;
  return h;
}

// Uniform in [-1, 1) from the top 24 bits; exactly representable in float.
float uniform(std::uint64_t& s) {
  return static_cast<float>(static_cast<std::int64_t>(splitmix64(s) >> 40) - (1 << 23)) / static_cast<float>(1 << 23);
}

// Each tensor draws from its own stream so tensors are independent of generation order.
class TensorRng {
 public:
  TensorRng(std::uint64_t seed, std::string_view name) : state_(seed ^ fnv1a(name)) { splitmix64(state_); }
  std::vector<float> uniform_scaled(std::size_t n, float scale) {
    std::vector<float> v(n);
    for (float& x : v) x = uniform(state_) * scale;
    return v;
  }
  std::vector<float> affine(std::size_t n, float base, float amp) {
    std::vector<float> v(n);
    for (float& x : v) x = base + amp * uniform(state_);
    return v;
  }

 private:
  std::uint64_t state_;
};

float fan_in_scale(std::size_t fan_in) { return 1.0f / std::sqrt(static_cast<float>(fan_in)); }

struct Shard {
  Bytes buf;
  // Appends a tensor at the next 64-byte boundary and returns its offset.
  std::uint64_t add(const Bytes& data) {
    buf.resize((buf.size() + kTensorAlign - 1) / kTensorAlign * kTensorAlign, 0);
    const std::uint64_t off = buf.size();
    buf.insert(buf.end(), data.begin(), data.end());
    return off;
  }
};

Bytes f32_bytes(const std::vector<float>& v) {
  Bytes b;
  encode_f32(v, b);
  return b;
}

class Builder {
 public:
  Builder(const FixtureSpec& spec) : spec_(spec), g_(spec.geometry()) { manifest_.geometry = g_; }

  Status build() {
    CLM_RETURN_IF_ERROR(spec_.validate());
    const std::uint32_t H = g_.hidden_size;
    shards_.resize(2);
    manifest_.artifact_id = "clusterlm-fixture/" + std::to_string(spec_.seed) + "/L" + std::to_string(g_.n_layers);
    manifest_.license = "CC0-1.0 (synthetic generated data, not derived from any real model)";

    {  // embedding, PLE table, head, MTP
      TensorRng r(spec_.seed, "token_embd");
      add_f32_object(std::string(kEmbeddingObjectName), ObjectKind::kEmbedding, std::nullopt, std::nullopt, 0,
                     {{0, r.uniform_scaled(std::size_t{g_.vocab_size} * H, 1.0f)}});
      TensorRng p(spec_.seed, "ple_lookup");
      add_f32_object(std::string(kPleObjectName), ObjectKind::kPleLookup, std::nullopt, std::nullopt, 1,
                     {{1, p.uniform_scaled(std::size_t{g_.ple_rows} * H, 1.0f)}});
      for (auto [name, kind] : {std::pair{kHeadObjectName, ObjectKind::kOutputHead}, std::pair{kMtpObjectName, ObjectKind::kMtp}}) {
        TensorRng hr(spec_.seed, name);
        add_f32_object(std::string(name), kind, std::nullopt, std::nullopt, 0,
                       {{0, hr.affine(H, 1.0f, 0.1f)},
                        {0, hr.uniform_scaled(std::size_t{g_.vocab_size} * H, fan_in_scale(H))}});
      }
    }

    for (std::uint32_t L = 0; L < g_.n_layers; ++L) build_layer(L);

    for (std::size_t i = 0; i < shards_.size(); ++i) {
      ShardInfo s;
      s.file_name = std::string(i == 0 ? kFixtureTransformerShard : kFixtureLookupShard);
      s.byte_size = shards_[i].buf.size();
      s.digest = Sha256::of(shards_[i].buf);
      manifest_.shards.push_back(std::move(s));
    }
    return manifest_.validate();
  }

  ModelManifest& manifest() { return manifest_; }
  const std::vector<Shard>& shards() const { return shards_; }

 private:
  struct Piece {
    std::uint32_t shard;
    std::vector<float> values;
  };

  void finish_object(ManifestObject& obj, const std::vector<std::pair<std::uint32_t, Bytes>>& pieces) {
    Sha256 h;
    for (const auto& [shard, bytes] : pieces) {
      SourceRange r;
      r.shard = shard;
      r.offset = shards_[shard].add(bytes);
      r.length = bytes.size();
      obj.source_ranges.push_back(r);
      obj.byte_size += r.length;
      h.update(bytes);
    }
    obj.source_digest = h.finish();
    obj.object_digest = obj.source_digest;  // conversion_version 0: provisioned bytes == source bytes
    manifest_.objects.push_back(std::move(obj));
  }

  void add_f32_object(std::string name, ObjectKind kind, std::optional<std::uint32_t> layer,
                      std::optional<std::uint32_t> expert, std::uint32_t /*shard_hint*/, std::vector<Piece> pieces) {
    ManifestObject obj;
    obj.name = std::move(name);
    obj.kind = kind;
    obj.layer = layer;
    obj.expert = expert;
    obj.representation = {std::string(kQuantF32), 0, 0, true};
    std::vector<std::pair<std::uint32_t, Bytes>> raw;
    for (Piece& p : pieces) raw.emplace_back(p.shard, f32_bytes(p.values));
    finish_object(obj, raw);
  }

  void build_layer(std::uint32_t L) {
    const std::uint32_t H = g_.hidden_size, E = g_.n_experts, ff = g_.expert_ff, sff = g_.shared_expert_ff;
    const std::string lp = "blk." + std::to_string(L) + ".";
    auto rng = [&](const char* t) { return TensorRng(spec_.seed, lp + t); };
    const std::size_t qd = std::size_t{g_.n_heads} * g_.head_dim, kvd = std::size_t{g_.n_kv_heads} * g_.head_dim;

    std::vector<Piece> dense;
    dense.push_back({0, rng("norm").affine(H, 1.0f, 0.1f)});
    if (g_.layer_kinds[L] == LayerKind::kRecurrent) {
      dense.push_back({0, rng("w_in").uniform_scaled(std::size_t{H} * H, fan_in_scale(H))});
      dense.push_back({0, rng("decay").affine(H, 0.725f, 0.2f)});  // (0.525, 0.925) subset of (0.5, 0.95)
      dense.push_back({0, rng("w_out").uniform_scaled(std::size_t{H} * H, fan_in_scale(H))});
    } else {
      dense.push_back({0, rng("wq").uniform_scaled(qd * H, fan_in_scale(H))});
      dense.push_back({0, rng("wk").uniform_scaled(kvd * H, fan_in_scale(H))});
      dense.push_back({0, rng("wv").uniform_scaled(kvd * H, fan_in_scale(H))});
      dense.push_back({0, rng("wo").uniform_scaled(std::size_t{H} * qd, fan_in_scale(qd))});
    }
    // Router weights are scaled 4x so routing varies visibly between tokens in tests.
    dense.push_back({0, rng("router").uniform_scaled(std::size_t{E} * H, 4.0f * fan_in_scale(H))});
    dense.push_back({0, rng("inj").uniform_scaled(std::size_t{g_.residual_streams} * H, fan_in_scale(H))});
    add_f32_object(dense_object_name(L), ObjectKind::kLayerDense, L, std::nullopt, 0, std::move(dense));

    if (sff > 0) {
      std::vector<Piece> sh;
      sh.push_back({0, rng("shexp_gate").uniform_scaled(std::size_t{sff} * H, fan_in_scale(H))});
      sh.push_back({0, rng("shexp_up").uniform_scaled(std::size_t{sff} * H, fan_in_scale(H))});
      sh.push_back({0, rng("shexp_down").uniform_scaled(std::size_t{H} * sff, fan_in_scale(sff))});
      add_f32_object(shared_expert_object_name(L), ObjectKind::kSharedExpert, L, std::nullopt, 0, std::move(sh));
    }

    // Stacked expert tensors: generated whole, stored whole, then described per expert as three slices.
    const bool q8 = (L % 2) == 1;
    const std::string quant(q8 ? kQuantQ8Fixture : kQuantF32);
    struct Stacked {
      std::uint64_t offset = 0;
      std::uint64_t per_expert = 0;
    };
    auto store_stacked = [&](const char* t, std::size_t elems_per_expert, std::size_t fan_in) {
      const std::vector<float> w = rng(t).uniform_scaled(elems_per_expert * E, fan_in_scale(fan_in));
      Bytes bytes;
      if (q8) {
        // Quantize expert by expert so every slice is a self-contained run of whole blocks.
        for (std::uint32_t e = 0; e < E; ++e)
          (void)encode_q8_fixture(std::span<const float>(w).subspan(e * elems_per_expert, elems_per_expert), bytes);
      } else {
        encode_f32(w, bytes);
      }
      Stacked s;
      s.offset = shards_[0].add(bytes);
      s.per_expert = bytes.size() / E;
      return s;
    };
    const Stacked gate = store_stacked("exps_gate", std::size_t{ff} * H, H);
    const Stacked up = store_stacked("exps_up", std::size_t{ff} * H, H);
    const Stacked down = store_stacked("exps_down", std::size_t{H} * ff, ff);

    for (std::uint32_t e = 0; e < E; ++e) {
      ManifestObject obj;
      obj.name = expert_object_name(L, e);
      obj.kind = ObjectKind::kRoutedExpert;
      obj.layer = L;
      obj.expert = e;
      obj.representation = {quant, q8 ? kQ8BlockElems : 0u, 0, true};
      Sha256 h;
      for (const Stacked* s : {&gate, &up, &down}) {
        SourceRange r{0, s->offset + e * s->per_expert, s->per_expert};
        h.update(ByteSpan(shards_[0].buf).subspan(static_cast<std::size_t>(r.offset), static_cast<std::size_t>(r.length)));
        obj.source_ranges.push_back(r);
        obj.byte_size += r.length;
      }
      obj.source_digest = h.finish();
      obj.object_digest = obj.source_digest;
      manifest_.objects.push_back(std::move(obj));
    }
  }

  FixtureSpec spec_;
  ModelGeometry g_;
  ModelManifest manifest_;
  std::vector<Shard> shards_;
};

Status write_file(const std::filesystem::path& p, ByteSpan data) {
  std::ofstream f(p, std::ios::binary | std::ios::trunc);
  if (!f) return make_error(ErrorCode::kUnavailable, "cannot create " + p.string());
  f.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
  f.flush();
  if (!f) return make_error(ErrorCode::kUnavailable, "write failed: " + p.string());
  return Status::ok();
}

}  // namespace

FixtureSpec FixtureSpec::tiny() {
  FixtureSpec s;
  s.n_layers = 8;
  s.hidden = 32;
  s.n_experts = 8;
  s.n_active = 2;
  s.expert_ff = 32;
  s.shared_expert_ff = 32;
  s.n_heads = 2;
  s.n_kv_heads = 1;
  s.head_dim = 16;
  s.vocab = 64;
  s.ple_rows = 128;
  return s;
}

ModelGeometry FixtureSpec::geometry() const {
  ModelGeometry g;
  g.family = family;
  g.n_layers = n_layers;
  g.hidden_size = hidden;
  g.residual_streams = residual_streams;
  g.n_experts = n_experts;
  g.n_active_experts = n_active;
  g.expert_ff = expert_ff;
  g.shared_expert_ff = shared_expert_ff;
  g.n_heads = n_heads;
  g.n_kv_heads = n_kv_heads;
  g.head_dim = head_dim;
  g.vocab_size = vocab;
  g.ple_layer = ple_layer;
  g.ple_ngram = ple_ngram;
  g.ple_rows = ple_rows;
  g.mtp_layers = mtp_layers;
  for (std::uint32_t i = 0; i < n_layers; ++i)
    g.layer_kinds.push_back(i % 4 == 3 ? LayerKind::kFullAttention : LayerKind::kRecurrent);
  return g;
}

Status FixtureSpec::validate() const {
  CLM_RETURN_IF_ERROR(geometry().validate());
  // q8 slices are whole blocks only if every expert tensor has a multiple of 32 elements.
  if (n_layers > 1 && (std::uint64_t{expert_ff} * hidden) % kQ8BlockElems != 0)
    return make_error(ErrorCode::kInvalidArgument, "fixture: expert_ff*hidden must be a multiple of 32");
  return Status::ok();
}

DenseLayout dense_layout(const ModelGeometry& g, LayerKind kind) {
  DenseLayout l;
  const std::size_t H = g.hidden_size, qd = std::size_t{g.n_heads} * g.head_dim, kvd = std::size_t{g.n_kv_heads} * g.head_dim;
  std::size_t off = 0;
  auto take = [&](std::size_t n) {
    const std::size_t at = off;
    off += n;
    return at;
  };
  l.norm = take(H);
  if (kind == LayerKind::kRecurrent) {
    l.w_in = take(H * H);
    l.decay = take(H);
    l.w_out = take(H * H);
  } else {
    l.wq = take(qd * H);
    l.wk = take(kvd * H);
    l.wv = take(kvd * H);
    l.wo = take(H * qd);
  }
  l.router = take(std::size_t{g.n_experts} * H);
  l.inj = take(std::size_t{g.residual_streams} * H);
  l.total = off;
  return l;
}

Result<ModelManifest> write_fixture_model(const FixtureSpec& spec, const std::filesystem::path& dir) {
  Builder b(spec);
  CLM_RETURN_IF_ERROR(b.build());
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  if (ec) return make_error(ErrorCode::kUnavailable, "cannot create " + dir.string() + ": " + ec.message());
  for (std::size_t i = 0; i < b.shards().size(); ++i)
    CLM_RETURN_IF_ERROR(write_file(dir / b.manifest().shards[i].file_name, b.shards()[i].buf));
  const std::string json = b.manifest().to_json();
  CLM_RETURN_IF_ERROR(write_file(dir / "manifest.json",
                                 ByteSpan(reinterpret_cast<const std::uint8_t*>(json.data()), json.size())));
  return std::move(b.manifest());
}

}  // namespace clusterlm::objects
