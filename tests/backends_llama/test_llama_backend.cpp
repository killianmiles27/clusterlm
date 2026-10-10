// The llama.cpp Fast-tier backend against the pinned llama.cpp, on a tiny random-weight llama-architecture GGUF:
// manifest, Coordinator generation through a Father-only plan, speculative window commit/abort, KV reuse across
// turns, logits against llama.cpp's own decode, and release.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>

#include "clusterlm/domain/drafter.hpp"
#include "clusterlm/objects/canonical_store.hpp"
#include "llama_test_support.hpp"

using namespace clusterlm;
using namespace clusterlm::llamatest;

TEST_CASE("llama manifest: geometry, objects and stored tensor types of the tiny model") {
  TinyModel m("manifest");
  const auto& g = m.manifest.geometry;
  CHECK(g.family == "llama.cpp:llama");
  CHECK(g.n_layers == 2);
  CHECK(g.hidden_size == 64);
  CHECK(g.residual_streams == 1);
  CHECK(g.vocab_size == 256);
  CHECK(g.n_heads == 4);
  CHECK(g.n_kv_heads == 2);
  CHECK(g.head_dim == 16);
  for (auto k : g.layer_kinds) CHECK(k == objects::LayerKind::kFullAttention);
  CHECK(m.manifest.find("token_embd") != nullptr);
  CHECK(m.manifest.find("output_head") != nullptr);
  REQUIRE(m.manifest.find(objects::dense_object_name(1)) != nullptr);
  // Every tensor byte is covered exactly once.
  std::uint64_t objects_total = 0;
  for (const auto& o : m.manifest.objects) objects_total += o.byte_size;
  backends::LlamaModelReport rep;
  auto again = backends::build_llama_manifest({m.gguf}, {}, &rep);
  REQUIRE(again.is_ok());
  CHECK(objects_total == rep.tensor_bytes);
  CHECK(rep.architecture == "llama");
  REQUIRE(rep.tensor_types.count("f32") == 1);
  CHECK(rep.tensor_types.size() == 1);
  CHECK(rep.tensor_count == 3 + 9 * 2);
  // The Coordinator can open the directory.
  CHECK(objects::CanonicalModelStore::open(m.dir).is_ok());

  TinyModel f16("manifest-f16", [] {
    backends::TinyLlamaSpec s;
    s.f16_matrices = true;
    return s;
  }());
  backends::LlamaModelReport rep16;
  REQUIRE(backends::build_llama_manifest({f16.gguf}, {}, &rep16).is_ok());
  CHECK(rep16.tensor_types.count("f16") == 1);
  CHECK(rep16.tensor_types.count("f32") == 1);
  CHECK(rep16.tensor_types.at("f16").tensors == 7 * 2);
}

TEST_CASE("llama backend: contract limits and honest refusals") {
  TinyModel m("refusals");
  auto backend = backends::make_llama_backend(options_for(m));
  const auto info = backend->info();
  CHECK(info.name == "llama");
  CHECK(info.build_hash.rfind("llama.cpp@", 0) == 0);
  const std::uint32_t L = m.layers();
  domain::DomainSpec mid{StageId{1}, domain::StageRole::kMiddle, {1, 2}, 128, 8, 1};
  CHECK(backend->create_domain(m.manifest, mid).status().code() == ErrorCode::kUnimplemented);
  domain::DomainSpec partial{StageId{0}, domain::StageRole::kPrefix, {0, 1}, 128, 8, 1};
  CHECK(backend->create_domain(m.manifest, partial).status().code() == ErrorCode::kUnimplemented);
  domain::DomainSpec orphan_tail{StageId{1}, domain::StageRole::kTail, {L, L}, 128, 8, 1};
  CHECK_FALSE(backend->create_domain(m.manifest, orphan_tail).is_ok());
  domain::DomainSpec bad{StageId{0}, domain::StageRole::kPrefix, {0, L}, 4, 8, 1};
  CHECK(backend->create_domain(m.manifest, bad).status().code() == ErrorCode::kInvalidArgument);

  Pair p(m, 128, 8);
  domain::WindowRequest req = p.next_request(2);
  domain::StageActivations dummy;
  CHECK(p.prefix->run_window(req, dummy).status().code() == ErrorCode::kUnimplemented);
  // Requirements: weights are the GGUF bytes, KV is sized from the geometry.
  auto rq = p.prefix->describe_requirements();
  REQUIRE(rq.is_ok());
  CHECK(rq->cpu_weight_bytes == m.manifest.total_bytes({0, L}) + m.manifest.find("token_embd")->byte_size +
                                    m.manifest.find("output_head")->byte_size);
  CHECK(rq->gpu_weight_bytes == 0);
  CHECK(rq->state_bytes == 2ull * 2 * 2 * 16 * 128 * 2);  // layers * (K,V) * kv_heads * head_dim * ctx * fp16
  CHECK(rq->required_objects.empty());
}

TEST_CASE("greedy generation through the Coordinator with a Father-only plan matches llama.cpp") {
  TinyModel m("greedy");
  RefLlama ref(m.gguf, 256);
  for (std::size_t prompt_len : {1u, 5u, 40u, 100u}) {
    CAPTURE(prompt_len);
    const auto prompt = prompt_of(prompt_len, m.vocab());
    const auto expect = ref.greedy(prompt, 24);
    auto father = make_father(m);
    for (std::uint32_t chunk : {1u, 7u, 128u}) {
      CAPTURE(chunk);
      coordinator::GenerationRequest r;
      r.prompt = prompt;
      r.max_new_tokens = 24;
      r.prefill_chunk = chunk;
      auto g = father->generate(r);
      REQUIRE_MESSAGE(g.is_ok(), g.status().to_string());
      CHECK(g->tokens == expect);
      CHECK(g->prefill_tokens == prompt_len);
    }
    REQUIRE(father->release().is_ok());
  }
}

TEST_CASE("every speculative acceptance length gives the q=1 greedy tokens") {
  TinyModel m("accept");
  const auto prompt = prompt_of(12, m.vocab());
  const std::uint32_t max_new = 20;
  auto father = make_father(m);
  coordinator::GenerationRequest base;
  base.prompt = prompt;
  base.max_new_tokens = max_new + 8;
  auto ref = father->generate(base);
  REQUIRE(ref.is_ok());
  std::vector<std::int32_t> sequence = prompt;
  sequence.insert(sequence.end(), ref->tokens.begin(), ref->tokens.end());
  const std::vector<std::int32_t> expect(ref->tokens.begin(), ref->tokens.begin() + max_new);
  for (std::uint32_t q = 1; q <= 4; ++q) {
    // corrupt draft index j (1-based) => exactly j-1 drafts accepted; none => all accepted.
    for (std::uint32_t j = 0; j <= q - 1; ++j) {
      CAPTURE(q);
      CAPTURE(j);
      domain::ScriptedDrafter::Config dc;
      dc.vocab = m.vocab();
      if (j > 0) dc.corrupt_draft_index = j;
      coordinator::GenerationRequest r;
      r.prompt = prompt;
      r.max_new_tokens = max_new;
      r.q = q;
      r.drafter = std::make_shared<domain::ScriptedDrafter>(sequence, dc);
      auto out = father->generate(r);
      REQUIRE_MESSAGE(out.is_ok(), out.status().to_string());
      CHECK(out->tokens == expect);
      const std::uint32_t expect_drafts = j == 0 ? q - 1 : j - 1;
      for (const auto& rd : out->rounds) {
        if (rd.prefill) continue;
        if (rd.positions == q) CHECK(rd.accepted_drafts == expect_drafts);
        CHECK(rd.accepted == rd.accepted_drafts + 1);
      }
    }
  }
  REQUIRE(father->release().is_ok());
}

TEST_CASE("window commit keeps exactly the accepted prefix and abort restores the committed state") {
  TinyModel m("window");
  Pair p(m);
  RefLlama ref(m.gguf, 256);
  const auto toks = prompt_of(24, m.vocab());
  const std::size_t V = m.vocab();
  const auto all = ref.logits_all(toks);

  // Prefill 8 tokens (committed), then a q=4 window of which only 2 positions are accepted.
  {
    const std::vector<std::int32_t> w(toks.begin(), toks.begin() + 8);
    auto l = p.run(w);
    REQUIRE(l.is_ok());
    // More than 32 positions would return only the last row; 8 returns every row.
    REQUIRE(l->positions == 8);
    CHECK(max_abs_diff(l->data.data(), all.data(), 8 * V) <= 1e-4f);
    REQUIRE(p.commit(8).is_ok());
  }
  const std::vector<std::int32_t> w4(toks.begin() + 8, toks.begin() + 12);
  auto first = p.run(w4);
  REQUIRE(first.is_ok());
  CHECK(max_abs_diff(first->data.data(), all.data() + 8 * V, 4 * V) <= 1e-4f);
  REQUIRE(p.commit(2).is_ok());
  CHECK(p.pos == 10);

  // The next window starts at position 10 and must see KV for exactly positions 0..9: its logits equal
  // llama.cpp's for the same continuation.
  std::vector<std::int32_t> hist(toks.begin(), toks.begin() + 10);
  const std::vector<std::int32_t> cont = {toks[10], toks[11], toks[12]};
  auto second = p.run(cont);
  REQUIRE(second.is_ok());
  std::vector<std::int32_t> full = hist;
  full.insert(full.end(), cont.begin(), cont.end());
  const auto ref_full = ref.logits_all(full);
  CHECK(max_abs_diff(second->data.data(), ref_full.data() + 10 * V, 3 * V) <= 1e-4f);

  // Abort: discard the window, then run a different one at the same base. Equal to llama.cpp for that sequence.
  domain::WindowRequest aborted = p.last;
  REQUIRE(p.prefix->abort_window(aborted.epoch, aborted.session, aborted.window).is_ok());
  REQUIRE(p.tail->abort_window(aborted.epoch, aborted.session, aborted.window).is_ok());
  // Repeating the abort is harmless.
  CHECK(p.prefix->abort_window(aborted.epoch, aborted.session, aborted.window).is_ok());
  const std::vector<std::int32_t> other = {toks[20], toks[21]};
  auto third = p.run(other);
  REQUIRE(third.is_ok());
  std::vector<std::int32_t> alt = hist;
  alt.insert(alt.end(), other.begin(), other.end());
  const auto ref_alt = ref.logits_all(alt);
  CHECK(max_abs_diff(third->data.data(), ref_alt.data() + 10 * V, 2 * V) <= 1e-4f);
  REQUIRE(p.commit(2).is_ok());
  CHECK(p.pos == 12);

  // Idempotent commit replay returns the same ack.
  domain::CommitRequest again{p.last.epoch, p.last.session, p.last.window, 2, p.last.expected_state};
  auto replay = p.prefix->commit_window(again);
  REQUIRE(replay.is_ok());
  CHECK(replay->committed_position == 12);
  // A commit for a window that was never run is refused.
  domain::CommitRequest stray{p.epoch, p.session, WindowId{999}, 1, p.state};
  CHECK_FALSE(p.prefix->commit_window(stray).is_ok());
  // Stale epoch is rejected.
  domain::WindowRequest stale = p.next_request(1);
  stale.epoch = Epoch{9};
  const std::vector<std::int32_t> one = {1};
  CHECK(p.prefix->run_prefix(stale, one).status().code() == ErrorCode::kStaleEpoch);
}

TEST_CASE("window logits are bitwise llama.cpp's for the same batch and close for other batchings") {
  TinyModel m("logits");
  const auto toks = prompt_of(16, m.vocab(), 3);
  const std::size_t V = m.vocab();
  RefLlama ref(m.gguf, 256);
  const auto all = ref.logits_all(toks);

  // One window of all 16 tokens decodes the identical batch: bitwise equal.
  {
    Pair p(m);
    auto l = p.run(toks);
    REQUIRE(l.is_ok());
    REQUIRE(l->positions == 16);
    CHECK(std::equal(l->data.begin(), l->data.end(), all.begin()));
  }
  // q = 1..4 windows decode different batch shapes: the same math, compared at a stated tolerance (float
  // reassociation across matmul batch shapes), and identical argmax at every position.
  for (std::uint32_t q = 1; q <= 4; ++q) {
    CAPTURE(q);
    Pair p(m);
    std::size_t at = 0;
    float worst = 0;
    while (at < toks.size()) {
      const std::size_t n = std::min<std::size_t>(q, toks.size() - at);
      const std::vector<std::int32_t> w(toks.begin() + static_cast<std::ptrdiff_t>(at), toks.begin() + static_cast<std::ptrdiff_t>(at + n));
      auto l = p.run(w);
      REQUIRE(l.is_ok());
      worst = std::max(worst, max_abs_diff(l->data.data(), all.data() + at * V, n * V));
      for (std::size_t i = 0; i < n; ++i)
        CHECK(coordinator::argmax(std::span<const float>(l->data).subspan(i * V, V)) ==
              coordinator::argmax(std::span<const float>(all).subspan((at + i) * V, V)));
      REQUIRE(p.commit(static_cast<std::uint32_t>(n)).is_ok());
      at += n;
    }
    CHECK(worst <= 1e-4f);
    MESSAGE("q=" << q << " max |logit difference| vs one batch = " << worst);
  }
}

TEST_CASE("long windows are prefill chunks: only the final position's logits are returned") {
  TinyModel m("prefill-logits");
  const auto toks = prompt_of(50, m.vocab());
  const std::size_t V = m.vocab();
  RefLlama ref(m.gguf, 256);
  const auto all = ref.logits_all(toks);
  Pair p(m);
  auto l = p.run(toks);
  REQUIRE(l.is_ok());
  CHECK(l->positions == 1);
  REQUIRE(l->data.size() == V);
  // The same batch with one output row: bitwise llama.cpp's. Against the all-rows batch it is the same math at
  // another output-head shape: same argmax and a tiny float difference.
  const auto last_only = ref.logits_last(toks);
  CHECK(l->data == last_only);
  CHECK(max_abs_diff(l->data.data(), all.data() + 49 * V, V) <= 1e-4f);
  CHECK(coordinator::argmax(l->data) == coordinator::argmax(std::span<const float>(all).subspan(49 * V, V)));
  REQUIRE(p.commit(50).is_ok());

  // full_logits_max_positions raised: every position comes back.
  auto opts = options_for(m);
  opts.full_logits_max_positions = 64;
  Pair full(m, 256, 64, 1, opts);
  auto lf = full.run(toks);
  REQUIRE(lf.is_ok());
  CHECK(lf->positions == 50);
  CHECK(std::equal(lf->data.begin(), lf->data.end(), all.begin()));
}

TEST_CASE("a conversation reuses the KV cache across turns") {
  TinyModel m("conversation");
  auto father = make_father(m);
  const auto a = prompt_of(9, m.vocab(), 1), b = prompt_of(7, m.vocab(), 2);
  auto conv = father->open_conversation();
  REQUIRE(conv.is_ok());
  coordinator::GenerationRequest r;
  r.conversation = conv.value();
  r.prompt = a;
  r.max_new_tokens = 8;
  auto t1 = father->generate(r);
  REQUIRE(t1.is_ok());
  r.prompt = b;
  auto t2 = father->generate(r);
  REQUIRE(t2.is_ok());
  // Only the new tokens (and the one pending prediction) were prefilled in turn 2.
  CHECK(t2->prefill_tokens == b.size() + 1);
  CHECK(conv.value()->valid());
  CHECK(conv.value()->committed_positions() >= a.size() + 8 + b.size());
  REQUIRE(father->close_conversation(*conv.value()).is_ok());
  REQUIRE(father->release().is_ok());

  std::vector<std::int32_t> history = a;
  history.insert(history.end(), t1->tokens.begin(), t1->tokens.end());
  history.insert(history.end(), b.begin(), b.end());
  RefLlama ref(m.gguf, 256);
  CHECK(ref.greedy(history, 8) == t2->tokens);
}

TEST_CASE("two sessions in one engine keep independent state") {
  TinyModel m("sessions");
  Pair p(m, 128, 16, 2);
  RefLlama ref(m.gguf, 128);
  const std::size_t V = m.vocab();
  const auto s1 = prompt_of(6, m.vocab(), 1), s2 = prompt_of(6, m.vocab(), 2);
  REQUIRE(p.prefix->open_session(p.epoch, SessionId{2}).is_ok());
  REQUIRE(p.tail->open_session(p.epoch, SessionId{2}).is_ok());
  // A third session exceeds max_sessions.
  CHECK(p.prefix->open_session(p.epoch, SessionId{3}).code() == ErrorCode::kResourceExhausted);

  auto run_on = [&](SessionId sid, std::uint64_t base, StateVersion sv, std::uint64_t win, const std::vector<std::int32_t>& t) {
    domain::WindowRequest req{p.epoch, sid, WindowId{win}, base, sv, static_cast<std::uint32_t>(t.size())};
    auto a = p.prefix->run_prefix(req, t);
    REQUIRE(a.is_ok());
    auto l = p.tail->run_tail(req, a.value());
    REQUIRE(l.is_ok());
    domain::CommitRequest c{p.epoch, sid, WindowId{win}, static_cast<std::uint32_t>(t.size()), sv};
    auto ack = p.prefix->commit_window(c);
    REQUIRE(ack.is_ok());
    REQUIRE(p.tail->commit_window(c).is_ok());
    return std::make_pair(std::move(l).value(), ack.value());
  };
  auto [l1, ack1] = run_on(SessionId{1}, 0, StateVersion{0}, 1, s1);
  auto [l2, ack2] = run_on(SessionId{2}, 0, StateVersion{0}, 1, s2);
  // Continue session 1 after session 2 ran: unaffected.
  const std::vector<std::int32_t> more = {5, 6, 7};
  auto [l1b, ack1b] = run_on(SessionId{1}, ack1.committed_position, ack1.state, 2, more);
  std::vector<std::int32_t> full = s1;
  full.insert(full.end(), more.begin(), more.end());
  const auto ref_full = ref.logits_all(full);
  CHECK(max_abs_diff(l1b.data.data(), ref_full.data() + 6 * V, 3 * V) <= 1e-4f);
  const auto ref2 = ref.logits_all(s2);
  CHECK(max_abs_diff(l2.data.data(), ref2.data(), 6 * V) <= 1e-4f);

  // Freeing session 2 lets a new session take its slot, starting empty.
  REQUIRE(p.prefix->abort_session(p.epoch, SessionId{2}).is_ok());
  REQUIRE(p.tail->abort_session(p.epoch, SessionId{2}).is_ok());
  REQUIRE(p.prefix->open_session(p.epoch, SessionId{4}).is_ok());
}

TEST_CASE("release frees every llama.cpp object and the domain can be prepared again") {
  TinyModel m("release");
  CHECK(backends::llama_live_counts().models == 0);
  CHECK(backends::llama_live_counts().contexts == 0);
  {
    Pair p(m);
    CHECK(backends::llama_live_counts().models == 1);
    CHECK(backends::llama_live_counts().contexts == 1);
    auto* d = dynamic_cast<backends::LlamaLocalDomain*>(p.prefix.get());
    REQUIRE(d != nullptr);
    CHECK(d->llama_pin().size() == 40);
    const auto before = p.prefix->read_metrics();
    CHECK(before.resident_weight_bytes > 0);
    auto l = p.run(prompt_of(4, m.vocab()));
    REQUIRE(l.is_ok());
    REQUIRE(p.commit(4).is_ok());
    CHECK(p.prefix->read_metrics().positions_committed == 4);
    CHECK(p.prefix->read_metrics().state_bytes > 0);

    REQUIRE(p.tail->release().is_ok());
    REQUIRE(p.prefix->release().is_ok());
    CHECK(backends::llama_live_counts().models == 0);
    CHECK(backends::llama_live_counts().contexts == 0);
    CHECK(p.prefix->read_metrics().resident_weight_bytes == 0);
    // Released domains refuse work until prepared again.
    CHECK(p.prefix->open_session(Epoch{2}, SessionId{5}).code() == ErrorCode::kFailedPrecondition);
    REQUIRE(p.prefix->prepare(*p.store).is_ok());
    REQUIRE(p.tail->prepare(*p.store).is_ok());
    CHECK(backends::llama_live_counts().models == 1);
    REQUIRE(p.prefix->open_session(Epoch{2}, SessionId{5}).is_ok());
  }
  CHECK(backends::llama_live_counts().models == 0);
  CHECK(backends::llama_live_counts().contexts == 0);

  // Through the Coordinator: release() frees the model; destroying it does too.
  auto father = make_father(m);
  CHECK(backends::llama_live_counts().models == 1);
  coordinator::GenerationRequest r;
  r.prompt = prompt_of(5, m.vocab());
  r.max_new_tokens = 3;
  REQUIRE(father->generate(r).is_ok());
  REQUIRE(father->release().is_ok());
  CHECK(backends::llama_live_counts().models == 0);
  CHECK(backends::llama_live_counts().contexts == 0);
}

TEST_CASE("a corrupted or mismatched model is refused before it is used") {
  TinyModel m("mismatch");
  // The manifest names a different size than the file: refused at create_domain, never mapped.
  auto bad = m.manifest;
  bad.shards[0].byte_size += 1;
  auto backend = backends::make_llama_backend(options_for(m));
  domain::DomainSpec ps{StageId{0}, domain::StageRole::kPrefix, {0, m.layers()}, 128, 8, 1};
  CHECK(backend->create_domain(bad, ps).status().code() == ErrorCode::kDataLoss);
  // A vocabulary disagreement between manifest and model is caught at prepare().
  auto wrong_vocab = m.manifest;
  wrong_vocab.geometry.vocab_size = 100;
  auto d = backend->create_domain(wrong_vocab, ps);
  REQUIRE(d.is_ok());
  auto store = objects::CanonicalModelStore::open(m.dir);
  REQUIRE(store.is_ok());
  CHECK_FALSE(d.value()->prepare(*store.value()).is_ok());
  CHECK(backends::llama_live_counts().models == 0);
}
