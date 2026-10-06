#pragma once
// In-process pipeline driver for tests: Father prefix -> middle stages -> Father tail, with activations
// pushed through the wire encoding between stages exactly as a transport would.
#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

#include <doctest/doctest.h>

#include "clusterlm/domain/drafter.hpp"
#include "clusterlm/domain/reference_domain.hpp"
#include "../objects/test_util.hpp"

namespace clusterlm::testutil {

using namespace clusterlm::domain;

struct PipelineConfig {
  std::vector<std::uint32_t> bounds;  // layer boundaries: stage i owns [bounds[i], bounds[i+1])
  std::uint32_t max_context = 256;
  std::uint32_t max_window = 8;
};

struct WindowTrace {
  std::vector<std::int32_t> tokens;  // window input tokens (positions base..base+q-1)
  std::uint32_t accepted = 0;
  std::vector<float> logits;         // q * vocab
};

inline std::int32_t argmax(const float* row, std::uint32_t n) {
  std::uint32_t best = 0;
  for (std::uint32_t i = 1; i < n; ++i)
    if (row[i] > row[best]) best = i;
  return static_cast<std::int32_t>(best);
}

class Pipeline {
 public:
  Pipeline(const objects::ModelManifest& manifest, const objects::ObjectResolver& resolver, const PipelineConfig& cfg)
      : max_window_(cfg.max_window) {
    REQUIRE(cfg.bounds.size() >= 3);  // at least prefix + tail
    const std::size_t n = cfg.bounds.size() - 1;
    for (std::size_t i = 0; i < n; ++i) {
      DomainSpec spec;
      spec.stage = StageId{static_cast<std::uint32_t>(i)};
      spec.role = i == 0 ? StageRole::kPrefix : (i + 1 == n ? StageRole::kTail : StageRole::kMiddle);
      spec.layers = {cfg.bounds[i], cfg.bounds[i + 1]};
      spec.max_context = cfg.max_context;
      spec.max_window = cfg.max_window;
      spec.max_sessions = 4;
      auto d = ReferenceDomain::create(manifest, spec);
      REQUIRE_MESSAGE(d.is_ok(), d.status().to_string());
      auto st = (*d)->prepare(resolver);
      REQUIRE_MESSAGE(st.is_ok(), st.to_string());
      stages.push_back(std::move(d).value());
    }
  }

  void open(Epoch epoch, SessionId session) {
    for (auto& s : stages) REQUIRE(s->open_session(epoch, session).is_ok());
  }

  static StageActivations over_the_wire(const StageActivations& a, std::uint32_t max_positions) {
    ByteWriter w;
    a.encode(w);
    ByteReader r(w.bytes());
    auto back = StageActivations::decode(r, max_positions);
    REQUIRE_MESSAGE(back.is_ok(), back.status().to_string());
    CHECK(r.at_end());
    return std::move(back).value();
  }

  Result<Logits> run(const WindowRequest& req, std::span<const std::int32_t> tokens) {
    auto act = stages.front()->run_prefix(req, tokens);
    if (!act.is_ok()) return act.status();
    StageActivations cur = over_the_wire(*act, max_window_);
    for (std::size_t i = 1; i + 1 < stages.size(); ++i) {
      auto next = stages[i]->run_window(req, cur);
      if (!next.is_ok()) return next.status();
      cur = over_the_wire(*next, max_window_);
    }
    return stages.back()->run_tail(req, cur);
  }

  Status commit(const CommitRequest& req, CommitAck* ack_out = nullptr) {
    for (auto& s : stages) {
      auto ack = s->commit_window(req);
      if (!ack.is_ok()) return ack.status();
      if (ack_out != nullptr) *ack_out = *ack;
    }
    return Status::ok();
  }

  std::vector<std::unique_ptr<ReferenceDomain>> stages;

 private:
  std::uint32_t max_window_;
};

struct GenerateResult {
  std::vector<std::int32_t> generated;  // exactly n_new tokens
  std::vector<WindowTrace> trace;       // every window, including prefill chunks
  std::uint64_t windows = 0;
  std::uint64_t accepted_drafts = 0;
};

// Greedy generation with verification windows of width q (drafter == nullptr or q == 1 means plain greedy).
inline GenerateResult generate(Pipeline& p, Epoch epoch, SessionId session, const std::vector<std::int32_t>& prompt,
                               std::uint32_t n_new, Drafter* drafter, std::uint32_t q, std::uint32_t vocab,
                               std::uint32_t max_window) {
  GenerateResult out;
  p.open(epoch, session);
  std::uint64_t window_id = 0, base = 0, state = 0;
  std::vector<std::int32_t> committed;
  std::int32_t next = 0;

  auto step = [&](const std::vector<std::int32_t>& toks, std::uint32_t accept_hint, bool prefill) -> std::pair<Logits, std::uint32_t> {
    WindowRequest req;
    req.epoch = epoch;
    req.session = session;
    req.window = WindowId{++window_id};
    req.base_position = base;
    req.expected_state = StateVersion{state};
    req.positions = static_cast<std::uint32_t>(toks.size());
    auto logits = p.run(req, toks);
    REQUIRE_MESSAGE(logits.is_ok(), logits.status().to_string());
    std::uint32_t accepted = static_cast<std::uint32_t>(toks.size());
    if (!prefill) {  // verify drafts: logits[i] predicts the token at window position i+1
      accepted = 1;
      for (std::uint32_t i = 0; i + 1 < toks.size(); ++i) {
        if (argmax(logits->data.data() + std::size_t{i} * vocab, vocab) != toks[i + 1]) break;
        ++accepted;
      }
      out.accepted_drafts += accepted - 1;
    }
    (void)accept_hint;
    CommitRequest c;
    c.epoch = epoch;
    c.session = session;
    c.window = req.window;
    c.accepted = accepted;
    c.expected_state = req.expected_state;
    CommitAck ack;
    REQUIRE(p.commit(c, &ack).is_ok());
    CHECK(ack.committed_position == base + accepted);
    base += accepted;
    ++state;
    ++out.windows;
    out.trace.push_back({toks, accepted, logits->data});
    return {std::move(logits).value(), accepted};
  };

  for (std::size_t off = 0; off < prompt.size(); off += max_window) {
    const std::size_t n = std::min<std::size_t>(max_window, prompt.size() - off);
    std::vector<std::int32_t> chunk(prompt.begin() + static_cast<std::ptrdiff_t>(off),
                                    prompt.begin() + static_cast<std::ptrdiff_t>(off + n));
    auto [logits, accepted] = step(chunk, 0, true);
    committed.insert(committed.end(), chunk.begin(), chunk.end());
    next = argmax(logits.data.data() + (n - 1) * vocab, vocab);
  }

  while (out.generated.size() < n_new) {
    std::vector<std::int32_t> toks{next};
    if (drafter != nullptr && q > 1) {
      auto drafts = drafter->draft(committed, next, q - 1);
      toks.insert(toks.end(), drafts.begin(), drafts.end());
    }
    auto [logits, accepted] = step(toks, 0, false);
    for (std::uint32_t i = 0; i < accepted; ++i) {
      committed.push_back(toks[i]);
      out.generated.push_back(toks[i]);
    }
    next = argmax(logits.data.data() + std::size_t{accepted - 1} * vocab, vocab);
  }
  out.generated.resize(n_new);
  for (auto& stage : p.stages) REQUIRE(stage->abort_session(epoch, session).is_ok());  // free the session slot
  return out;
}

}  // namespace clusterlm::testutil
