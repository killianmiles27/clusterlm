// Stochastic speculative sampling must reproduce the target distribution exactly, whatever the drafter does.
// These tests check distributions statistically (chi-square), not greedy equality.
#include <doctest/doctest.h>

#include <cmath>
#include <map>

#include "cluster_fixture.hpp"
#include "clusterlm/domain/drafter.hpp"
#include "clusterlm/domain/sampling.hpp"
#include "clusterlm/objects/canonical_store.hpp"

using namespace clusterlm;
using namespace clusterlm::testing;

namespace {

// Upper-tail test of a chi-square statistic with k degrees of freedom via the Wilson–Hilferty approximation.
// Returns the standard-normal z; z > 3.29 corresponds to p < 0.0005.
double chi_square_z(double x, double k) {
  const double c = 2.0 / (9.0 * k);
  return (std::cbrt(x / k) - (1.0 - c)) / std::sqrt(c);
}

double goodness_of_fit(const std::vector<double>& counts, const std::vector<double>& expected_p, double n,
                       double& dof) {
  double x = 0;
  dof = -1;
  for (std::size_t i = 0; i < counts.size(); ++i) {
    const double e = expected_p[i] * n;
    if (e <= 0) {
      CHECK(counts[i] == 0);  // impossible outcome observed => hard failure
      continue;
    }
    x += (counts[i] - e) * (counts[i] - e) / e;
    dof += 1;
  }
  return x;
}

std::vector<float> random_distribution(domain::Rng& rng, std::size_t n, bool sparse) {
  std::vector<float> p(n);
  double s = 0;
  for (auto& v : p) {
    v = static_cast<float>(rng.uniform());
    if (sparse && rng.uniform() < 0.3) v = 0;
    s += v;
  }
  for (auto& v : p) v = static_cast<float>(v / s);
  return p;
}

}  // namespace

TEST_CASE("speculative verification emits the target distribution for random target and drafter distributions") {
  domain::Rng setup(42);
  const std::size_t V = 7;
  for (int trial = 0; trial < 4; ++trial) {
    const bool one_hot = trial == 3;
    std::vector<std::vector<float>> target, draft;
    for (int i = 0; i < 3; ++i) target.push_back(random_distribution(setup, V, trial == 1));
    for (int i = 0; i < 2; ++i) draft.push_back(random_distribution(setup, V, trial == 2));
    domain::Rng rng(1000 + static_cast<std::uint64_t>(trial));
    const int N = 200000;
    std::vector<double> first(V, 0), second_given_first0(V, 0);
    double n_first0 = 0;
    for (int s = 0; s < N; ++s) {
      // Sample drafts from the drafter distribution (or a fixed deterministic draft).
      std::vector<std::int32_t> d;
      if (one_hot) {
        d = {2, 5};
      } else {
        d = {domain::sample_from(draft[0], rng), domain::sample_from(draft[1], rng)};
      }
      auto o = domain::verify_speculative(target, d, one_hot ? std::vector<std::vector<float>>{} : draft, rng);
      const std::int32_t t1 = o.accepted_drafts >= 1 ? d[0] : o.next_token;
      first[static_cast<std::size_t>(t1)] += 1;
      // Conditional second token: only meaningful when the target's second distribution is the one used, i.e. it
      // does not depend on the first token in this synthetic setup (target[1] is fixed).
      if (o.accepted_drafts >= 1) {
        const std::int32_t t2 = o.accepted_drafts >= 2 ? d[1] : o.next_token;
        second_given_first0[static_cast<std::size_t>(t2)] += 1;
        n_first0 += 1;
      }
    }
    double dof;
    const double x1 = goodness_of_fit(first, std::vector<double>(target[0].begin(), target[0].end()), N, dof);
    CAPTURE(trial);
    CHECK(chi_square_z(x1, dof) < 3.29);
    if (n_first0 > 1000) {
      const double x2 =
          goodness_of_fit(second_given_first0, std::vector<double>(target[1].begin(), target[1].end()), n_first0, dof);
      CHECK(chi_square_z(x2, dof) < 3.29);
    }
  }
}

TEST_CASE("sampling distributions honour temperature, top-k and top-p") {
  const std::vector<float> logits = {1.0f, 3.0f, 2.0f, 0.0f};
  auto greedy = domain::distribution(logits, {});
  CHECK(greedy == std::vector<float>{0, 1, 0, 0});
  domain::SamplingParams p;
  p.temperature = 1.0f;
  auto full = domain::distribution(logits, p);
  double s = 0;
  for (float v : full) s += v;
  CHECK(s == doctest::Approx(1.0));
  CHECK(full[1] > full[2]);
  CHECK(full[2] > full[0]);
  p.top_k = 2;
  auto k2 = domain::distribution(logits, p);
  CHECK(k2[0] == 0.0f);
  CHECK(k2[3] == 0.0f);
  CHECK(k2[1] + k2[2] == doctest::Approx(1.0));
  p.top_k = 0;
  p.top_p = 0.5f;
  auto p5 = domain::distribution(logits, p);
  CHECK(p5[1] == doctest::Approx(1.0));  // the top token alone exceeds 0.5
  domain::Rng a(7), b(7);
  for (int i = 0; i < 10; ++i) CHECK(domain::sample_from(full, a) == domain::sample_from(full, b));
}

TEST_CASE("end to end: stochastic speculative decoding has the same output distribution as plain sampling") {
  // Father-only plan keeps this fast; the speculative path (drafts, verification, partial commits) is the same
  // code that runs across Nodes.
  objects::FixtureSpec spec = objects::FixtureSpec::tiny();
  TestCluster cl("stochastic", 0, {}, spec);
  const auto& g = cl.manifest().geometry;
  auto cfg = cl.config();
  auto f = cl.father(cfg, "0-" + std::to_string(g.n_layers) + "@father," + std::to_string(g.n_layers) + "-" +
                              std::to_string(g.n_layers) + "@father");
  auto store = objects::CanonicalModelStore::open(cl.dir() / "model");
  REQUIRE(store.is_ok());
  auto drafter = domain::MtpFixtureDrafter::create(cl.manifest(), *store.value());
  REQUIRE(drafter.is_ok());
  std::shared_ptr<domain::Drafter> d = std::move(drafter).value();
  const auto prompt = test_prompt(6, g.vocab_size, 9);

  auto histogram = [&](std::uint32_t q, std::uint64_t seed_base, int n, std::size_t index) {
    std::map<std::int32_t, double> h;
    for (int i = 0; i < n; ++i) {
      coordinator::GenerationRequest r;
      r.prompt = prompt;
      r.max_new_tokens = 4;
      r.q = q;
      r.drafter = q > 1 ? d : nullptr;
      r.sampling.temperature = 1.3f;
      r.sampling.seed = seed_base + static_cast<std::uint64_t>(i);
      auto out = f->generate(r);
      REQUIRE(out.is_ok());
      REQUIRE(out->tokens.size() == 4);
      h[out->tokens[index]] += 1;
    }
    return h;
  };
  // Reproducibility: same seed, same tokens.
  {
    coordinator::GenerationRequest r;
    r.prompt = prompt;
    r.max_new_tokens = 6;
    r.q = 3;
    r.drafter = d;
    r.sampling.temperature = 1.3f;
    r.sampling.seed = 77;
    auto a = f->generate(r), b = f->generate(r);
    REQUIRE(a.is_ok());
    REQUIRE(b.is_ok());
    CHECK(a->tokens.size() == 6);
    CHECK(a->tokens == b->tokens);  // a non-zero seed makes the turn reproducible
  }
  const int N = 1500;
  for (std::size_t index : {1u, 2u}) {
    CAPTURE(index);
    auto plain = histogram(1, 100000, N, index);
    auto spec3 = histogram(3, 900000, N, index);
    // Two-sample chi-square over categories with enough mass; rare categories pooled.
    std::map<std::int32_t, bool> keys;
    for (auto& [k, v] : plain) keys[k] = true;
    for (auto& [k, v] : spec3) keys[k] = true;
    double x = 0, dof = -1, pool_a = 0, pool_b = 0;
    for (auto& [k, unused] : keys) {
      const double a = plain.count(k) ? plain[k] : 0, b = spec3.count(k) ? spec3[k] : 0;
      if (a + b < 20) {
        pool_a += a;
        pool_b += b;
        continue;
      }
      x += (a - b) * (a - b) / (a + b);
      dof += 1;
    }
    if (pool_a + pool_b > 0) {
      x += (pool_a - pool_b) * (pool_a - pool_b) / (pool_a + pool_b);
      dof += 1;
    }
    REQUIRE(dof >= 1);
    CHECK(chi_square_z(x, dof) < 3.29);
  }
  REQUIRE(f->release().is_ok());
}
