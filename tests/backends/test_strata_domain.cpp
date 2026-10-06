// StrataDomain's state machine against a fake engine: role rules, the boundary transpose inside the domain path,
// token-freedom of middle/tail domains, WindowLedger behaviour (stale epochs, idempotent commits, abort_window),
// local sub-batching, failure handling, and the MTP drafter adapter. No device needed.
#include <doctest/doctest.h>

#include <cstring>
#include <type_traits>

#include "clusterlm/backends/strata/mtp_drafter.hpp"
#include "clusterlm/backends/strata/strata_domain.hpp"
#include "clusterlm/backends/strata_handoff.hpp"
#include "strata_test_support.hpp"

using namespace clusterlm;
using namespace clusterlm::strata_test;
using backends::strata::StrataDomain;
using domain::StageRole;

namespace {

domain::DomainSpec spec_of(StageRole role, std::uint32_t begin, std::uint32_t end, std::uint32_t max_window = 8,
                           std::uint32_t local_batch = 0) {
  domain::DomainSpec s;
  s.role = role;
  s.layers = {begin, end};
  s.max_context = 256;
  s.max_window = max_window;
  s.max_sessions = 2;
  s.max_local_batch = local_batch;
  return s;
}

struct Fixture {
  objects::ModelGeometry g = tiny_geometry();
  objects::ModelManifest m = synthetic_manifest(g, "q8_0");
  FakeEngine* engine = nullptr;
  std::unique_ptr<StrataDomain> d;
  NullResolver resolver;

  explicit Fixture(domain::DomainSpec spec, std::uint32_t engine_window = 8) {
    auto e = std::make_unique<FakeEngine>(spec.role, g, engine_window);
    engine = e.get();
    auto r = StrataDomain::create(m, spec, std::move(e));
    REQUIRE_MESSAGE(r.is_ok(), r.status().to_string());
    d = std::move(r.value());
    REQUIRE(d->prepare(resolver).is_ok());
  }

  domain::StageActivations input(std::uint64_t first, std::uint32_t q) const {
    domain::StageActivations a;
    a.layout = domain::BoundaryLayout::for_geometry(g);
    a.first_position = first;
    a.positions = q;
    a.data.resize(std::size_t{q} * a.layout.floats_per_position());
    for (std::size_t i = 0; i < a.data.size(); ++i) a.data[i] = static_cast<float>(i % 97) * 0.25f - 3.0f;
    return a;
  }
};

domain::WindowRequest req(Epoch e, SessionId s, std::uint64_t w, std::uint64_t base, std::uint64_t state, std::uint32_t q) {
  return domain::WindowRequest{e, s, WindowId{w}, base, StateVersion{state}, q};
}

}  // namespace

TEST_CASE("role rules: the prefix holds every token-dependent layer, other roles are token-free") {
  const objects::ModelGeometry g = tiny_geometry();  // ple_layer 2 -> layers 0..2 belong to the prefix
  const objects::ModelManifest m = synthetic_manifest(g, "q8_0");
  CHECK(backends::strata::token_free_first_layer(g) == 3);
  auto make = [&](domain::DomainSpec s) {
    return StrataDomain::create(m, s, std::make_unique<FakeEngine>(s.role, g)).status();
  };
  CHECK(make(spec_of(StageRole::kPrefix, 0, 3)).is_ok());
  CHECK(make(spec_of(StageRole::kPrefix, 0, 2)).code() == ErrorCode::kInvalidArgument);  // PLE layer outside
  CHECK(make(spec_of(StageRole::kPrefix, 1, 4)).code() == ErrorCode::kInvalidArgument);
  CHECK(make(spec_of(StageRole::kPrefix, 0, 8)).code() == ErrorCode::kInvalidArgument);   // the head is the tail's
  CHECK(make(spec_of(StageRole::kMiddle, 3, 6)).is_ok());
  CHECK(make(spec_of(StageRole::kMiddle, 2, 6)).code() == ErrorCode::kInvalidArgument);   // would need tokens
  CHECK(make(spec_of(StageRole::kMiddle, 1, 6)).code() == ErrorCode::kInvalidArgument);   // Strata's PLE block
  CHECK(make(spec_of(StageRole::kMiddle, 5, 8)).code() == ErrorCode::kInvalidArgument);   // head belongs to the tail
  CHECK(make(spec_of(StageRole::kTail, 6, 8)).is_ok());
  CHECK(make(spec_of(StageRole::kTail, 6, 7)).code() == ErrorCode::kInvalidArgument);
  // an engine whose capabilities disagree with the role is an internal error, not a silent token leak
  auto bad = std::make_unique<FakeEngine>(StageRole::kPrefix, g);
  CHECK(StrataDomain::create(m, spec_of(StageRole::kMiddle, 3, 6), std::move(bad)).status().code() == ErrorCode::kInternal);
}

TEST_CASE("middle domain: the wire is transposed to Strata's field-major hand-off and back, exactly") {
  Fixture f(spec_of(StageRole::kMiddle, 3, 6));
  const Epoch e{1};
  const SessionId s{7};
  REQUIRE(f.d->open_session(e, s).is_ok());
  const auto in = f.input(0, 5);
  auto out = f.d->run_window(req(e, s, 1, 0, 0, 5), in);
  REQUIRE_MESSAGE(out.is_ok(), out.status().to_string());

  // what the engine received is wire_to_strata(input), bit for bit
  const auto& call = f.engine->calls.back();
  REQUIRE(call.op == "run");
  std::vector<float> expect(in.data.size());
  REQUIRE(backends::strata::wire_to_strata(in.layout, 5, in.data, expect).is_ok());
  REQUIRE(call.handoff_in.size() == expect.size());
  CHECK(std::memcmp(call.handoff_in.data(), expect.data(), expect.size() * sizeof(float)) == 0);

  // and what came back is the engine's field-major output transposed to position-major
  const std::size_t r = 4 * 8, H = 8, per = in.layout.floats_per_position();
  for (std::uint32_t t = 0; t < 5; ++t) {
    const auto src = in.position(t);
    const auto dst = out->position(t);
    CHECK(dst[in.layout.streams_offset() + 3] == 2.0f * src[3] + static_cast<float>(t));
    CHECK(dst[in.layout.pending_offset() + 1] == src[r + 1] - 1.0f);
    CHECK(dst[in.layout.injection_offset() + 2] == src[r + H + 2] + 0.5f);
  }
  CHECK(out->data.size() == 5 * per);
  CHECK(out->first_position == 0);
}

TEST_CASE("token-free: middle and tail APIs take no tokens and the engine never sees any") {
  // static: the middle/tail entry points have no token parameter
  static_assert(std::is_same_v<decltype(&domain::ExecutionDomain::run_window),
                               Result<domain::StageActivations> (domain::ExecutionDomain::*)(
                                   const domain::WindowRequest&, const domain::StageActivations&)>);
  static_assert(std::is_same_v<decltype(&domain::ExecutionDomain::run_tail),
                               Result<domain::Logits> (domain::ExecutionDomain::*)(const domain::WindowRequest&,
                                                                                   const domain::StageActivations&)>);
  // runtime: the pointer handed to Strata's Verifier::run is null on a token-free domain whatever is passed
  const std::int32_t toks[3] = {1, 2, 3};
  backends::strata::EngineCaps mid;
  mid.needs_tokens = false;
  CHECK(backends::strata::verifier_tokens(mid, toks) == nullptr);
  backends::strata::EngineCaps pre;
  pre.needs_tokens = true;
  CHECK(backends::strata::verifier_tokens(pre, toks) == toks);

  for (StageRole role : {StageRole::kMiddle, StageRole::kTail}) {
    Fixture f(role == StageRole::kMiddle ? spec_of(role, 3, 6) : spec_of(role, 6, 8));
    REQUIRE(f.d->open_session(Epoch{1}, SessionId{1}).is_ok());
    for (std::uint64_t w = 1; w <= 3; ++w) {
      const std::uint64_t base = w - 1;
      const auto in = f.input(base, 1);
      if (role == StageRole::kMiddle) {
        REQUIRE(f.d->run_window(req(Epoch{1}, SessionId{1}, w, base, base, 1), in).is_ok());
      } else {
        REQUIRE(f.d->run_tail(req(Epoch{1}, SessionId{1}, w, base, base, 1), in).is_ok());
        CHECK(f.d->run_window(req(Epoch{1}, SessionId{1}, w + 100, base, base, 1), in).status().code() ==
              ErrorCode::kFailedPrecondition);
      }
      REQUIRE(f.d->commit_window({Epoch{1}, SessionId{1}, WindowId{w}, 1, StateVersion{base}}).is_ok());
    }
    CHECK_FALSE(f.engine->tokens_on_token_free_domain);
    for (const auto& c : f.engine->calls) CHECK(c.tokens == 0);
  }

  // the prefix does take tokens, validated against the vocabulary
  Fixture p(spec_of(StageRole::kPrefix, 0, 3));
  REQUIRE(p.d->open_session(Epoch{1}, SessionId{1}).is_ok());
  const std::int32_t bad[2] = {1, 99};
  CHECK(p.d->run_prefix(req(Epoch{1}, SessionId{1}, 1, 0, 0, 2), bad).status().code() == ErrorCode::kInvalidArgument);
  const std::int32_t good[2] = {1, 5};
  auto a = p.d->run_prefix(req(Epoch{1}, SessionId{1}, 2, 0, 0, 2), good);
  REQUIRE(a.is_ok());
  CHECK(a->position(1)[0] == 5.0f + 1.0f);  // token + position + committed
  CHECK(p.engine->calls.back().tokens == 2);
}

TEST_CASE("ledger at the adapter: stale epochs, idempotent commit, abort_window restores the committed state") {
  Fixture f(spec_of(StageRole::kMiddle, 3, 6));
  const Epoch e{3};
  const SessionId s{9};
  REQUIRE(f.d->open_session(e, s).is_ok());
  const auto in = f.input(0, 4);

  // stale epoch: nothing reaches the engine
  const std::size_t calls0 = f.engine->calls.size();
  CHECK(f.d->run_window(req(Epoch{2}, s, 1, 0, 0, 4), in).status().code() == ErrorCode::kStaleEpoch);
  CHECK(f.engine->calls.size() == calls0);

  auto first = f.d->run_window(req(e, s, 1, 0, 0, 4), in);
  REQUIRE(first.is_ok());
  // a second outstanding window is refused
  CHECK(f.d->run_window(req(e, s, 2, 0, 0, 4), in).status().code() == ErrorCode::kFailedPrecondition);
  CHECK(f.d->commit_window({Epoch{2}, s, WindowId{1}, 2, StateVersion{0}}).status().code() == ErrorCode::kStaleEpoch);

  // abort_window drops it (the engine restores) and keeps the session; repeating it is harmless
  auto ack = f.d->abort_window(e, s, WindowId{1});
  REQUIRE(ack.is_ok());
  CHECK(ack->committed_position == 0);
  CHECK(f.engine->count("abort") == 1);
  CHECK(f.d->abort_window(e, s, WindowId{1}).is_ok());
  CHECK(f.engine->count("abort") == 1);
  CHECK(f.d->abort_window(e, s, WindowId{50}).status().code() == ErrorCode::kFailedPrecondition);  // never admitted

  // the same window content again (new id) gives the identical result: the abort restored the committed state
  auto again = f.d->run_window(req(e, s, 2, 0, 0, 4), in);
  REQUIRE(again.is_ok());
  CHECK(again->data == first->data);

  // commit 3 of 4; the replay returns the identical ack and does not commit twice
  auto c1 = f.d->commit_window({e, s, WindowId{2}, 3, StateVersion{0}});
  REQUIRE(c1.is_ok());
  CHECK(c1->committed_position == 3);
  CHECK(c1->state == StateVersion{1});
  const std::size_t commits = f.engine->count("commit");
  auto c2 = f.d->commit_window({e, s, WindowId{2}, 3, StateVersion{0}});
  REQUIRE(c2.is_ok());
  CHECK(*c2 == *c1);
  CHECK(f.engine->count("commit") == commits);
  CHECK(f.engine->committed[s] == 3);
  // a different replay is refused
  CHECK(f.d->commit_window({e, s, WindowId{2}, 2, StateVersion{0}}).status().code() == ErrorCode::kFailedPrecondition);
  // the next window must start at the committed position
  CHECK(f.d->run_window(req(e, s, 3, 0, 1, 1), f.input(0, 1)).status().code() == ErrorCode::kFailedPrecondition);
  CHECK(f.d->run_window(req(e, s, 4, 3, 1, 1), f.input(3, 1)).is_ok());

  // abort_session forgets the session; the engine frees it
  CHECK(f.d->abort_session(Epoch{2}, s).code() == ErrorCode::kStaleEpoch);
  REQUIRE(f.d->abort_session(e, s).is_ok());
  CHECK(f.engine->count("close") == 1);
  CHECK(f.d->run_window(req(e, s, 5, 4, 2, 1), f.input(4, 1)).status().code() == ErrorCode::kNotFound);
  const auto m = f.d->read_metrics();
  CHECK(m.windows_committed == 1);
  CHECK(m.windows_aborted == 2);
  CHECK(m.stale_rejections == 3);
  CHECK(m.resident_weight_bytes == 123);
}

TEST_CASE("local sub-batches: a large window runs in engine windows of max_local_batch, committed whole") {
  Fixture f(spec_of(StageRole::kMiddle, 3, 6, /*max_window=*/16, /*local_batch=*/3));
  CHECK(f.d->local_batch() == 3);
  const Epoch e{1};
  const SessionId s{1};
  REQUIRE(f.d->open_session(e, s).is_ok());
  const auto in = f.input(0, 8);
  auto out = f.d->run_window(req(e, s, 1, 0, 0, 8), in);
  REQUIRE(out.is_ok());
  std::vector<std::uint32_t> runs, commits;
  for (const auto& c : f.engine->calls) {
    if (c.op == "run") runs.push_back(c.n);
    if (c.op == "commit") commits.push_back(c.n);
  }
  CHECK(runs == std::vector<std::uint32_t>{3, 3, 2});
  CHECK(commits == std::vector<std::uint32_t>{3, 3});  // provisional
  // rows of later sub-batches see the earlier ones committed (n = 3, then 6), exactly as one engine window would
  // see nothing - so the assembled result differs from an unsplit run only through that committed count
  CHECK(out->position(4)[0] == 2.0f * in.position(4)[0] + 4.0f + 3.0f);
  CHECK(out->position(7)[0] == 2.0f * in.position(7)[0] + 7.0f + 6.0f);
  auto ack = f.d->commit_window({e, s, WindowId{1}, 8, StateVersion{0}});
  REQUIRE(ack.is_ok());
  CHECK(ack->committed_position == 8);
  CHECK(f.engine->calls.back().op == "commit");
  CHECK(f.engine->calls.back().n == 2);
  CHECK(f.engine->committed[s] == 8);

  // a partial commit of a split window cannot be honoured: the session is invalidated, never silently wrong
  REQUIRE(f.d->run_window(req(e, s, 2, 8, 1, 7), f.input(8, 7)).is_ok());
  CHECK(f.d->commit_window({e, s, WindowId{2}, 4, StateVersion{1}}).status().code() == ErrorCode::kAborted);
  CHECK(f.d->run_window(req(e, s, 3, 8, 1, 1), f.input(8, 1)).status().code() == ErrorCode::kNotFound);
  // nor can an abort
  REQUIRE(f.d->open_session(Epoch{2}, s).is_ok());
  REQUIRE(f.d->run_window(req(Epoch{2}, s, 1, 0, 0, 5), f.input(0, 5)).is_ok());
  CHECK(f.d->abort_window(Epoch{2}, s, WindowId{1}).status().code() == ErrorCode::kAborted);
  // a window within one sub-batch still commits any accepted prefix
  REQUIRE(f.d->open_session(Epoch{3}, s).is_ok());
  REQUIRE(f.d->run_window(req(Epoch{3}, s, 1, 0, 0, 3), f.input(0, 3)).is_ok());
  CHECK(f.d->commit_window({Epoch{3}, s, WindowId{1}, 1, StateVersion{0}}).is_ok());
}

TEST_CASE("tail: logits of all sub-batches land at their positions") {
  Fixture f(spec_of(StageRole::kTail, 6, 8, 8, 2));
  REQUIRE(f.d->open_session(Epoch{1}, SessionId{1}).is_ok());
  const auto in = f.input(0, 5);
  auto lg = f.d->run_tail(req(Epoch{1}, SessionId{1}, 1, 0, 0, 5), in);
  REQUIRE(lg.is_ok());
  CHECK(lg->positions == 5);
  CHECK(lg->vocab == 16);
  for (std::uint32_t t = 0; t < 5; ++t) {
    float sum = 0;
    for (std::size_t i = 0; i < 32; ++i) sum += in.position(t)[i];
    const float committed_before = static_cast<float>(t / 2 * 2);
    CHECK(lg->data[t * 16 + 3] == doctest::Approx(sum + 3.0f + committed_before));
  }
}

TEST_CASE("failures: a failed first sub-batch is rolled back, a failed later one invalidates the session") {
  {
    Fixture f(spec_of(StageRole::kMiddle, 3, 6));
    REQUIRE(f.d->open_session(Epoch{1}, SessionId{1}).is_ok());
    f.engine->fail_run_on = 1;
    CHECK(f.d->run_window(req(Epoch{1}, SessionId{1}, 1, 0, 0, 2), f.input(0, 2)).status().code() == ErrorCode::kInternal);
    CHECK(f.engine->count("abort") == 1);
    // the session stays usable at its committed state (the window id is consumed)
    CHECK(f.d->run_window(req(Epoch{1}, SessionId{1}, 2, 0, 0, 2), f.input(0, 2)).is_ok());
  }
  {
    Fixture f(spec_of(StageRole::kMiddle, 3, 6, 8, 2));
    REQUIRE(f.d->open_session(Epoch{1}, SessionId{1}).is_ok());
    f.engine->fail_run_on = 2;
    CHECK(f.d->run_window(req(Epoch{1}, SessionId{1}, 1, 0, 0, 4), f.input(0, 4)).status().code() == ErrorCode::kAborted);
    CHECK(f.d->run_window(req(Epoch{1}, SessionId{1}, 2, 0, 0, 1), f.input(0, 1)).status().code() == ErrorCode::kNotFound);
  }
  {
    Fixture f(spec_of(StageRole::kMiddle, 3, 6));
    REQUIRE(f.d->open_session(Epoch{1}, SessionId{1}).is_ok());
    REQUIRE(f.d->run_window(req(Epoch{1}, SessionId{1}, 1, 0, 0, 2), f.input(0, 2)).is_ok());
    f.engine->fail_commit = true;
    CHECK(f.d->commit_window({Epoch{1}, SessionId{1}, WindowId{1}, 2, StateVersion{0}}).status().code() ==
          ErrorCode::kAborted);
  }
}

TEST_CASE("describe_requirements names only the domain's own objects") {
  Fixture f(spec_of(StageRole::kMiddle, 3, 5));
  auto r = f.d->describe_requirements();
  REQUIRE(r.is_ok());
  const auto& names = r->required_objects;
  CHECK(names.size() == 2 * (2 + f.g.n_experts));
  for (const std::string& n : names) {
    CHECK(n.rfind("blk.", 0) == 0);
    const auto* o = f.m.find(n);
    REQUIRE(o != nullptr);
    CHECK((*o->layer == 3 || *o->layer == 4));
    CHECK_FALSE(o->father_only());
  }
  Fixture p(spec_of(StageRole::kPrefix, 0, 3));
  auto rp = p.d->describe_requirements();
  REQUIRE(rp.is_ok());
  CHECK(std::find(rp->required_objects.begin(), rp->required_objects.end(), "token_embd") != rp->required_objects.end());
  CHECK(std::find(rp->required_objects.begin(), rp->required_objects.end(), "ple_lookup") == rp->required_objects.end());
  Fixture t(spec_of(StageRole::kTail, 6, 8));
  auto rt = t.d->describe_requirements();
  REQUIRE(rt.is_ok());
  CHECK(rt->required_objects.back() == "output_head");
}

namespace {
class FakeMtp final : public backends::strata::MtpEngine {
 public:
  std::uint32_t mtp_vocab() const override { return 4; }
  Status mtp_draft(std::span<const std::int32_t> committed, std::int32_t next_token, std::uint32_t count,
                   const domain::SamplingParams* sampling, std::vector<std::int32_t>& drafts,
                   std::vector<std::vector<float>>* probs) override {
    last_committed = committed.size();
    if (fail) return make_error(ErrorCode::kInternal, "injected");
    for (std::uint32_t j = 0; j < produce; ++j) drafts.push_back((next_token + 1 + static_cast<std::int32_t>(j)) % 4);
    if (sampling != nullptr && probs != nullptr)
      for (std::uint32_t j = 0; j < produce; ++j) {
        std::vector<float> p(4, 0.1f);
        p[static_cast<std::size_t>(drafts[j])] = 0.7f;
        probs->push_back(p);
      }
    (void)count;
    return Status::ok();
  }
  std::uint32_t produce = 3;
  bool fail = false;
  std::size_t last_committed = 0;
};
}  // namespace

TEST_CASE("StrataMtpDrafter: exactly `count` drafts, distributions for sampling, deterministic fallback") {
  FakeMtp mtp;
  backends::strata::StrataMtpDrafter d(mtp);
  const std::int32_t hist[5] = {0, 1, 2, 3, 0};
  auto g = d.draft(hist, 1, 3);
  CHECK(g == std::vector<std::int32_t>{2, 3, 0});
  CHECK(mtp.last_committed == 5);
  mtp.produce = 1;  // a short engine round is padded
  CHECK(d.draft(hist, 1, 3).size() == 3);
  mtp.produce = 5;  // a long one truncated
  CHECK(d.draft(hist, 1, 2).size() == 2);

  domain::SamplingParams sp;
  sp.temperature = 0.8f;
  domain::Rng rng(1);
  mtp.produce = 3;
  auto p = d.propose(hist, 2, 3, sp, rng);
  REQUIRE(p.tokens.size() == 3);
  REQUIRE(p.probs.size() == 3);
  CHECK(p.probs[0][static_cast<std::size_t>(p.tokens[0])] == doctest::Approx(0.7f));

  mtp.fail = true;
  auto f = d.propose(hist, 2, 3, sp, rng);
  CHECK(f.tokens.size() == 3);
  CHECK(f.probs.empty());  // one-hot: exactness never depends on the drafter
  CHECK_FALSE(d.last_status().is_ok());
  CHECK(d.fallbacks() == 1);
}
