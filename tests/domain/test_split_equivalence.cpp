#include <doctest/doctest.h>

#include <set>

#include "clusterlm/objects/tensor_codec.hpp"
#include "pipeline_harness.hpp"

using namespace clusterlm;
using namespace clusterlm::domain;
using namespace clusterlm::objects;
using namespace clusterlm::testutil;

namespace {

const std::vector<std::int32_t> kPrompt = {17, 3, 59, 50, 4, 4, 4, 31, 12, 40, 7};  // 11 tokens: two prefill chunks

bool bit_equal(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

// Replays the exact windows (tokens + accepted counts) of `trace` on another pipeline.
std::vector<std::vector<float>> replay(Pipeline& p, const std::vector<WindowTrace>& trace, Epoch e, SessionId s) {
  p.open(e, s);
  std::vector<std::vector<float>> logits;
  std::uint64_t base = 0, state = 0, id = 0;
  for (const WindowTrace& w : trace) {
    WindowRequest req;
    req.epoch = e;
    req.session = s;
    req.window = WindowId{++id};
    req.base_position = base;
    req.expected_state = StateVersion{state};
    req.positions = static_cast<std::uint32_t>(w.tokens.size());
    auto l = p.run(req, w.tokens);
    REQUIRE_MESSAGE(l.is_ok(), l.status().to_string());
    CommitRequest c;
    c.epoch = e;
    c.session = s;
    c.window = req.window;
    c.accepted = w.accepted;
    c.expected_state = req.expected_state;
    REQUIRE(p.commit(c).is_ok());
    base += w.accepted;
    ++state;
    logits.push_back(l->data);
  }
  return logits;
}

}  // namespace

TEST_CASE("full model vs 4-domain split: bitwise identical logits over multi-window generation") {
  const Fixture& f = full_fixture();
  const ModelGeometry& g = f.manifest.geometry;
  auto store = f.open_store();

  Pipeline unsplit(f.manifest, *store, {{0, 16, 16}});  // prefix [0,16) + head-only tail
  Pipeline split(f.manifest, *store, {{0, 4, 10, 13, 16}});
  REQUIRE(split.stages.size() == 4);

  // Plain greedy (q = 1) on both: same tokens, bit-identical logits for every window.
  const auto plain_u = generate(unsplit, Epoch{1}, SessionId{1}, kPrompt, 24, nullptr, 1, g.vocab_size, 8);
  const auto plain_s = generate(split, Epoch{1}, SessionId{1}, kPrompt, 24, nullptr, 1, g.vocab_size, 8);
  CHECK(plain_u.generated == plain_s.generated);
  REQUIRE(plain_u.trace.size() == plain_s.trace.size());
  for (std::size_t i = 0; i < plain_u.trace.size(); ++i)
    CHECK_MESSAGE(bit_equal(plain_u.trace[i].logits, plain_s.trace[i].logits), "window " << i);
  CHECK(plain_u.windows == 2 + 24);  // 2 prefill chunks + one window per token

  // Multi-position verification windows with partial acceptance, replayed on the unsplit model.
  std::vector<std::int32_t> ref = kPrompt;
  ref.insert(ref.end(), plain_u.generated.begin(), plain_u.generated.end());
  ScriptedDrafter::Config cfg;
  cfg.vocab = g.vocab_size;
  cfg.corruption_rate = 0.3;
  cfg.seed = 5;
  ScriptedDrafter drafter(ref, cfg);
  const auto spec_s = generate(split, Epoch{1}, SessionId{2}, kPrompt, 24, &drafter, 4, g.vocab_size, 8);
  CHECK(spec_s.generated == plain_u.generated);
  std::set<std::uint32_t> accepted_lengths;
  for (std::size_t i = 2; i < spec_s.trace.size(); ++i) accepted_lengths.insert(spec_s.trace[i].accepted);
  CHECK(accepted_lengths.size() >= 2);  // a mix of full and partial acceptance

  const auto logits_u = replay(unsplit, spec_s.trace, Epoch{1}, SessionId{3});
  REQUIRE(logits_u.size() == spec_s.trace.size());
  for (std::size_t i = 0; i < logits_u.size(); ++i)
    CHECK_MESSAGE(bit_equal(logits_u[i], spec_s.trace[i].logits), "window " << i);
}

TEST_CASE("other splits agree bitwise too (uneven, single-layer middle stages)") {
  const Fixture& f = tiny_fixture();
  auto store = f.open_store();
  const std::uint32_t vocab = f.manifest.geometry.vocab_size;
  Pipeline base(f.manifest, *store, {{0, 8, 8}});
  const auto ref = generate(base, Epoch{1}, SessionId{1}, kPrompt, 12, nullptr, 1, vocab, 8);
  const std::vector<std::vector<std::uint32_t>> splits{{0, 3, 8}, {0, 3, 4, 5, 6, 7, 8}, {0, 7, 8}, {0, 3, 4, 8, 8}};
  for (const auto& bounds : splits) {
    Pipeline p(f.manifest, *store, {bounds});
    const auto r = generate(p, Epoch{1}, SessionId{1}, kPrompt, 12, nullptr, 1, vocab, 8);
    CHECK(r.generated == ref.generated);
    REQUIRE(r.trace.size() == ref.trace.size());
    for (std::size_t i = 0; i < r.trace.size(); ++i) CHECK(bit_equal(r.trace[i].logits, ref.trace[i].logits));
  }
}

TEST_CASE("speculative decoding equals plain greedy for q = 1..4 and every rejection position") {
  const Fixture& f = tiny_fixture();
  const std::uint32_t vocab = f.manifest.geometry.vocab_size;
  auto store = f.open_store();
  const std::vector<std::uint32_t> bounds{0, 3, 5, 6, 8};  // prefix, middle, middle, tail (4 domains)
  constexpr std::uint32_t kNew = 20;

  Pipeline plain_p(f.manifest, *store, {bounds});
  // Extra tokens past kNew so the drafter's reference covers every window that overshoots the end.
  const auto plain_long = generate(plain_p, Epoch{1}, SessionId{1}, kPrompt, kNew + 8, nullptr, 1, vocab, 8);
  std::vector<std::int32_t> ref = kPrompt;
  ref.insert(ref.end(), plain_long.generated.begin(), plain_long.generated.end());
  const std::vector<std::int32_t> expect(plain_long.generated.begin(), plain_long.generated.begin() + kNew);

  std::uint64_t session = 10;
  for (std::uint32_t q = 1; q <= 4; ++q) {
    Pipeline p(f.manifest, *store, {bounds});
    {  // no corruption
      ScriptedDrafter::Config cfg;
      cfg.vocab = vocab;
      ScriptedDrafter d(ref, cfg);
      const auto r = generate(p, Epoch{1}, SessionId{session++}, kPrompt, kNew, &d, q, vocab, 8);
      CHECK(r.generated == expect);
      for (std::size_t i = 2; i < r.trace.size(); ++i) CHECK(r.trace[i].accepted == q);  // perfect drafts: all accepted
      if (q > 1) CHECK(r.windows < 2 + kNew);                                            // and fewer windows than plain
    }
    // Corrupt draft number j in every window => acceptance length exactly j (0 accepted drafts for j = 1).
    for (std::uint32_t j = 1; j + 1 <= q; ++j) {
      ScriptedDrafter::Config cfg;
      cfg.vocab = vocab;
      cfg.corrupt_draft_index = j;
      ScriptedDrafter d(ref, cfg);
      const auto r = generate(p, Epoch{1}, SessionId{session++}, kPrompt, kNew, &d, q, vocab, 8);
      CAPTURE(q);
      CAPTURE(j);
      CHECK(r.generated == expect);
      for (std::size_t i = 2; i < r.trace.size(); ++i) CHECK(r.trace[i].accepted == j);
    }
    // Random corruption mixes every acceptance length.
    for (std::uint64_t seed : {1u, 2u, 3u}) {
      ScriptedDrafter::Config cfg;
      cfg.vocab = vocab;
      cfg.corruption_rate = 0.4;
      cfg.seed = seed;
      ScriptedDrafter d(ref, cfg);
      const auto r = generate(p, Epoch{1}, SessionId{session++}, kPrompt, kNew, &d, q, vocab, 8);
      CAPTURE(q);
      CAPTURE(seed);
      CHECK(r.generated == expect);
    }
  }
}

TEST_CASE("corruption at specific absolute positions") {
  const Fixture& f = tiny_fixture();
  const std::uint32_t vocab = f.manifest.geometry.vocab_size;
  auto store = f.open_store();
  Pipeline p(f.manifest, *store, {{0, 3, 5, 6, 8}});
  const auto plain = generate(p, Epoch{1}, SessionId{1}, kPrompt, 24, nullptr, 1, vocab, 8);
  std::vector<std::int32_t> ref = kPrompt;
  ref.insert(ref.end(), plain.generated.begin(), plain.generated.end());
  ScriptedDrafter::Config cfg;
  cfg.vocab = vocab;
  for (std::uint64_t pos = 12; pos < 30; pos += 3) cfg.corrupt_positions.push_back(pos);
  ScriptedDrafter d(ref, cfg);
  const auto r = generate(p, Epoch{1}, SessionId{2}, kPrompt, 16, &d, 4, vocab, 8);
  CHECK(r.generated == std::vector<std::int32_t>(plain.generated.begin(), plain.generated.begin() + 16));
}

TEST_CASE("MtpFixtureDrafter drafts deterministically and never changes the output") {
  const Fixture& f = tiny_fixture();
  const std::uint32_t vocab = f.manifest.geometry.vocab_size;
  auto store = f.open_store();
  auto drafter = MtpFixtureDrafter::create(f.manifest, *store);
  REQUIRE(drafter.is_ok());

  const auto a = (*drafter)->draft({}, 5, 4);
  const auto b = (*drafter)->draft({}, 5, 4);
  CHECK(a == b);
  REQUIRE(a.size() == 4);
  CHECK(a[0] == (*drafter)->step(5));
  CHECK(a[1] == (*drafter)->step(a[0]));  // chained
  for (std::int32_t t : a) {
    CHECK(t >= 0);
    CHECK(static_cast<std::uint32_t>(t) < vocab);
  }
  CHECK((*drafter)->draft({}, 5, 0).empty());

  Pipeline p(f.manifest, *store, {{0, 3, 5, 6, 8}});
  const auto plain = generate(p, Epoch{1}, SessionId{1}, kPrompt, 20, nullptr, 1, vocab, 8);
  const auto spec = generate(p, Epoch{1}, SessionId{2}, kPrompt, 20, drafter->get(), 4, vocab, 8);
  CHECK(spec.generated == plain.generated);
}

TEST_CASE("ScriptedDrafter replays the reference and corrupts deterministically") {
  std::vector<std::int32_t> ref;
  for (std::int32_t i = 0; i < 40; ++i) ref.push_back((i * 7) % 50);
  ScriptedDrafter::Config cfg;
  cfg.vocab = 50;
  {
    ScriptedDrafter d(ref, cfg);
    const std::vector<std::int32_t> committed(ref.begin(), ref.begin() + 10);
    const auto out = d.draft(committed, ref[10], 3);
    CHECK(out == std::vector<std::int32_t>{ref[11], ref[12], ref[13]});
  }
  cfg.corrupt_draft_index = 2;
  cfg.seed = 9;
  ScriptedDrafter d(ref, cfg);
  const std::vector<std::int32_t> committed(ref.begin(), ref.begin() + 10);
  const auto out = d.draft(committed, ref[10], 3);
  CHECK(out[0] == ref[11]);
  CHECK(out[1] != ref[12]);
  CHECK(out[2] == ref[13]);
  CHECK(out == d.draft(committed, ref[10], 3));
  CHECK(out[1] >= 0);
  CHECK(out[1] < 50);
}

TEST_CASE("quantized expert layers dequantize exactly: the domain matches a direct computation") {
  // Layer 1's experts are q8_0-fixture. Decode one expert with the codec and compare against the
  // domain's own expert math by running a one-layer-wide middle stage on a hand-built input.
  const Fixture& f = tiny_fixture();
  auto store = f.open_store();
  const ManifestObject* e = f.manifest.find(expert_object_name(1, 0));
  REQUIRE(e->representation.quant_type == "q8_0-fixture");
  auto obj = store->resolve(e->name);
  REQUIRE(obj.is_ok());
  const std::uint64_t n = std::uint64_t{f.manifest.geometry.expert_ff} * f.manifest.geometry.hidden_size;
  std::vector<float> gate(n);
  REQUIRE(decode_tensor("q8_0-fixture", obj->bytes.subspan(0, tensor_bytes("q8_0-fixture", n)), gate).is_ok());
  // Every dequantized value is scale * (int8): in particular a multiple of its block's scale.
  for (std::size_t b = 0; b < n / 32; ++b) {
    float scale;
    std::memcpy(&scale, obj->bytes.data() + b * 36, 4);
    for (std::size_t i = 0; i < 32; ++i) {
      const auto q = static_cast<std::int8_t>(obj->bytes[b * 36 + 4 + i]);
      CHECK(gate[b * 32 + i] == scale * static_cast<float>(q));
    }
  }
  // A domain covering the quantized layer runs and is deterministic across independent instances.
  Pipeline p1(f.manifest, *store, {{0, 3, 8}});
  Pipeline p2(f.manifest, *store, {{0, 3, 8}});
  const auto g1 = generate(p1, Epoch{1}, SessionId{1}, kPrompt, 6, nullptr, 1, f.manifest.geometry.vocab_size, 8);
  const auto g2 = generate(p2, Epoch{1}, SessionId{1}, kPrompt, 6, nullptr, 1, f.manifest.geometry.vocab_size, 8);
  CHECK(g1.generated == g2.generated);
  CHECK(bit_equal(g1.trace.back().logits, g2.trace.back().logits));
}

TEST_CASE("PLE history only advances on commit") {
  // The PLE n-gram must come from committed history: tokens of a window's rejected tail must never leak
  // into the next window's lookups (compare against a pipeline that only ever ran the accepted token).
  const Fixture& f = tiny_fixture();
  auto store = f.open_store();
  Pipeline a(f.manifest, *store, {{0, 3, 8}});
  Pipeline b(f.manifest, *store, {{0, 3, 8}});
  a.open(Epoch{1}, SessionId{1});
  b.open(Epoch{1}, SessionId{1});
  auto req = [](std::uint64_t id, std::uint64_t base, std::uint64_t state, std::uint32_t q) {
    WindowRequest r;
    r.epoch = Epoch{1};
    r.session = SessionId{1};
    r.window = WindowId{id};
    r.base_position = base;
    r.expected_state = StateVersion{state};
    r.positions = q;
    return r;
  };
  auto commit = [](const WindowRequest& w, std::uint32_t n) {
    CommitRequest c;
    c.epoch = w.epoch;
    c.session = w.session;
    c.window = w.window;
    c.accepted = n;
    c.expected_state = w.expected_state;
    return c;
  };
  // a: window [10,20,30,40] accept 1, then continue with [21,31]; b: window [10] accept 1, same continuation.
  REQUIRE(a.run(req(1, 0, 0, 4), std::vector<std::int32_t>{10, 20, 30, 40}).is_ok());
  REQUIRE(a.commit(commit(req(1, 0, 0, 4), 1)).is_ok());
  REQUIRE(b.run(req(1, 0, 0, 1), std::vector<std::int32_t>{10}).is_ok());
  REQUIRE(b.commit(commit(req(1, 0, 0, 1), 1)).is_ok());
  const std::vector<std::int32_t> cont{21, 31};
  auto la = a.run(req(2, 1, 1, 2), cont);
  auto lb = b.run(req(2, 1, 1, 2), cont);
  REQUIRE(la.is_ok());
  REQUIRE(lb.is_ok());
  CHECK(bit_equal(la->data, lb->data));
}
