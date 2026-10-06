// The grouped expert-domain executor against the unsplit reference domain.
#include <doctest/doctest.h>

#include <cmath>
#include <cstring>

#include "../domain/pipeline_harness.hpp"
#include "clusterlm/expert_domains/rig.hpp"

using namespace clusterlm;
using namespace clusterlm::expert_domains;
using namespace clusterlm::testutil;

namespace {

// Window schedule: prefill, then windows of every q = 1..4 with partial and full acceptance, including a
// rejection at every position of the widest window.
struct Step {
  std::uint32_t q;
  std::uint32_t accepted;
};
const std::vector<Step> kSchedule = {{5, 5}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {4, 1}, {4, 2}, {4, 3}, {3, 1}, {2, 1}, {4, 4}, {1, 1}};

std::int32_t tok(std::uint64_t i, std::uint32_t vocab) { return static_cast<std::int32_t>((i * 31u + 7u) % vocab); }

bool bit_equal(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

double rel_diff(const std::vector<float>& a, const std::vector<float>& b) {
  double mx = 0, ref = 1e-30;
  for (std::size_t i = 0; i < a.size(); ++i) {
    mx = std::max(mx, std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
    ref = std::max(ref, std::fabs(static_cast<double>(b[i])));
  }
  return mx / ref;
}

struct Run {
  std::vector<std::vector<float>> logits;  // per window, q*vocab
  std::vector<std::vector<LayerExchange>> layers;
};

// Runs the schedule on a grouped rig; every window is compared with `reference` by the caller afterwards.
Run run_grouped(GroupedRig& rig, std::uint32_t vocab) {
  Run out;
  std::uint64_t t = 0;
  for (const Step& s : kSchedule) {
    std::vector<std::int32_t> tokens;
    for (std::uint32_t i = 0; i < s.q; ++i) tokens.push_back(tok(t++, vocab));
    t -= s.q - s.accepted;  // rejected positions are re-proposed by the next window
    auto lg = rig.executor().run_window(tokens);
    REQUIRE_MESSAGE(lg.is_ok(), lg.status().to_string());
    out.logits.push_back(lg->data);
    out.layers.push_back(rig.executor().last_window_metrics().layers);
    REQUIRE(rig.executor().commit(s.accepted).is_ok());
  }
  return out;
}

Run run_reference(const Fixture& f, const objects::ObjectResolver& store) {
  const std::uint32_t n = f.manifest.geometry.n_layers, vocab = f.manifest.geometry.vocab_size;
  Pipeline p(f.manifest, store, {{0, n, n}, 64, 8});
  p.open(Epoch{1}, SessionId{1});
  Run out;
  std::uint64_t t = 0, base = 0, id = 0, state = 0;
  for (const Step& s : kSchedule) {
    std::vector<std::int32_t> tokens;
    for (std::uint32_t i = 0; i < s.q; ++i) tokens.push_back(tok(t++, vocab));
    t -= s.q - s.accepted;
    WindowRequest req;
    req.epoch = Epoch{1};
    req.session = SessionId{1};
    req.window = WindowId{++id};
    req.base_position = base;
    req.expected_state = StateVersion{state};
    req.positions = s.q;
    auto lg = p.run(req, tokens);
    REQUIRE_MESSAGE(lg.is_ok(), lg.status().to_string());
    out.logits.push_back(lg->data);
    CommitRequest c;
    c.epoch = Epoch{1};
    c.session = SessionId{1};
    c.window = req.window;
    c.accepted = s.accepted;
    c.expected_state = req.expected_state;
    REQUIRE(p.commit(c).is_ok());
    base += s.accepted;
    ++state;
  }
  return out;
}

RigOptions base_options(const objects::CanonicalModelStore& store) {
  RigOptions o;
  o.store = &store;
  o.max_context = 64;
  o.max_window = 8;
  o.layer_timeout = std::chrono::milliseconds(10'000);
  return o;
}

}  // namespace

TEST_CASE("one owner executing every selected expert is BITWISE identical to the reference") {
  // Father owns nothing and one remote domain owns all experts: each position's routed sum is then one
  // chain 0 + w1 d1 + w2 d2 + ..., exactly the reference's accumulation, and the shared expert is added after it.
  const Fixture& f = full_fixture();
  auto store = f.open_store();
  RigOptions o = base_options(*store);
  o.remote_domains = 1;
  o.explicit_owner_of.assign(f.manifest.geometry.n_experts, 1);
  auto rig = GroupedRig::create(o);
  REQUIRE_MESSAGE(rig.is_ok(), rig.status().to_string());
  const Run grouped = run_grouped(**rig, f.manifest.geometry.vocab_size);
  const Run ref = run_reference(f, *store);
  REQUIRE(grouped.logits.size() == ref.logits.size());
  for (std::size_t w = 0; w < ref.logits.size(); ++w) CHECK_MESSAGE(bit_equal(grouped.logits[w], ref.logits[w]), "window " << w);
}

TEST_CASE("three owners (Father + 2 domains): within tolerance, same argmax, q = 1..4 over all layers") {
  const Fixture& f = full_fixture();
  auto store = f.open_store();
  const std::uint32_t vocab = f.manifest.geometry.vocab_size;
  const Run ref = run_reference(f, *store);
  for (bool strided : {false, true}) {
    CAPTURE(strided);
    RigOptions o = base_options(*store);
    o.strided = strided;
    auto rig = GroupedRig::create(o);
    REQUIRE_MESSAGE(rig.is_ok(), rig.status().to_string());
    const Run grouped = run_grouped(**rig, vocab);
    double worst = 0;
    for (std::size_t w = 0; w < ref.logits.size(); ++w) {
      const double d = rel_diff(grouped.logits[w], ref.logits[w]);
      worst = std::max(worst, d);
      const std::uint32_t q = kSchedule[w].q;
      for (std::uint32_t p = 0; p < q; ++p)
        CHECK(argmax(grouped.logits[w].data() + std::size_t{p} * vocab, vocab) ==
              argmax(ref.logits[w].data() + std::size_t{p} * vocab, vocab));
    }
    // The only difference is float association of the per-owner partial sums; observed ~5e-7 relative; bound is 20x that.
    CHECK_MESSAGE(worst <= 1e-5, "worst relative logit difference " << worst);
    MESSAGE("strided=" << strided << " worst relative logit difference " << worst);
  }
}

TEST_CASE("the result does not depend on timing: jitter, overlap and arrival order give identical bits") {
  const Fixture& f = full_fixture();
  auto store = f.open_store();
  const std::uint32_t vocab = f.manifest.geometry.vocab_size;

  RigOptions plain = base_options(*store);
  auto r1 = GroupedRig::create(plain);
  REQUIRE(r1.is_ok());
  const Run a = run_grouped(**r1, vocab);

  RigOptions jittery = base_options(*store);
  jittery.network = transport::NetworkConditions{"test-jitter", 0, 0.2, 2.0, 99};  // jitter >> latency: order varies
  jittery.overlap_local = false;
  auto r2 = GroupedRig::create(jittery);
  REQUIRE(r2.is_ok());
  const Run b = run_grouped(**r2, vocab);

  for (std::size_t w = 0; w < a.logits.size(); ++w) CHECK_MESSAGE(bit_equal(a.logits[w], b.logits[w]), "window " << w);
}

TEST_CASE("exactly one message per participating domain per layer, never one per expert") {
  const Fixture& f = full_fixture();
  auto store = f.open_store();
  const std::uint32_t n_layers = f.manifest.geometry.n_layers;
  RigOptions o = base_options(*store);
  auto rig = GroupedRig::create(o);
  REQUIRE(rig.is_ok());
  const Run run = run_grouped(**rig, f.manifest.geometry.vocab_size);

  std::uint64_t participating = 0, selections = 0;
  for (std::size_t w = 0; w < run.layers.size(); ++w) {
    REQUIRE(run.layers[w].size() == n_layers);  // one exchange record per layer: ~L sequential barriers per pass
    for (const LayerExchange& l : run.layers[w]) {
      CHECK(l.domains_participating <= (*rig)->remote_count());
      participating += l.domains_participating;
      selections += l.remote_selections;
      CHECK(l.selections == l.positions * f.manifest.geometry.n_active_experts);
    }
  }
  std::uint64_t batches = 0, executions = 0;
  for (std::size_t i = 0; i < (*rig)->remote_count(); ++i) {
    const ExpertDomainMetrics m = (*rig)->server(i).metrics();
    batches += m.batches;
    executions += m.executions;
    CHECK(m.batches <= run.layers.size() * n_layers);  // at most one per layer per window
    CHECK(m.rejected == 0);
  }
  CHECK(batches == participating);  // one message per participating domain per layer, nothing else
  CHECK(executions == selections);  // every remote selection executed exactly once
  CHECK(executions > batches);      // many experts per message
}

TEST_CASE("the expert union grows with q but each domain still gets one message per layer") {
  const Fixture& f = full_fixture();
  auto store = f.open_store();
  const std::uint32_t vocab = f.manifest.geometry.vocab_size;
  RigOptions o = base_options(*store);
  auto rig = GroupedRig::create(o);
  REQUIRE(rig.is_ok());
  FatherExecutor& ex = (*rig)->executor();
  double distinct1 = 0, distinct4 = 0;
  std::uint64_t t = 0;
  for (std::uint32_t q : {1u, 4u}) {
    double sum = 0;
    for (int rep = 0; rep < 3; ++rep) {
      std::vector<std::int32_t> tokens;
      for (std::uint32_t i = 0; i < q; ++i) tokens.push_back(tok(t++, vocab));
      REQUIRE(ex.run_window(tokens).is_ok());
      for (const LayerExchange& l : ex.last_window_metrics().layers) {
        sum += l.distinct_experts;
        CHECK(l.domains_participating <= 2);
      }
      REQUIRE(ex.commit(q).is_ok());
    }
    (q == 1 ? distinct1 : distinct4) = sum / (3.0 * f.manifest.geometry.n_layers);
  }
  CHECK(distinct4 > distinct1);
  CHECK(distinct4 <= 4.0 * distinct1 + 1e-9);
}

TEST_CASE("ownership by ranges: owners combine in ascending expert-id order") {
  // With contiguous ranges, owner order equals ascending expert-id order, so the combination order is the
  // ascending-expert order the spec asks for. This is what makes arrival order irrelevant.
  auto a = ExpertAssignment::ranges(32, {1, 1, 1});
  REQUIRE(a.is_ok());
  for (std::uint32_t o = 1; o < 3; ++o) CHECK(a->owned[o].front() > a->owned[o - 1].back());
}

TEST_CASE("windows obey the commit contract: accepted prefix only, single outstanding window") {
  const Fixture& f = tiny_fixture();
  auto store = f.open_store();
  RigOptions o = base_options(*store);
  o.remote_domains = 1;
  auto rig = GroupedRig::create(o);
  REQUIRE_MESSAGE(rig.is_ok(), rig.status().to_string());
  FatherExecutor& ex = (*rig)->executor();
  const std::vector<std::int32_t> t = {1, 2, 3};
  CHECK_FALSE(ex.commit(1).is_ok());  // nothing outstanding
  REQUIRE(ex.run_window(t).is_ok());
  CHECK_FALSE(ex.run_window(t).is_ok());  // already outstanding
  CHECK_FALSE(ex.commit(0).is_ok());
  CHECK_FALSE(ex.commit(4).is_ok());
  REQUIRE(ex.commit(2).is_ok());
  CHECK(ex.committed_position() == 2);
  CHECK_FALSE(ex.run_window({}).is_ok());
  const std::vector<std::int32_t> bad = {1000};
  CHECK_FALSE(ex.run_window(bad).is_ok());  // outside the vocabulary (tiny vocab = 64)
}
