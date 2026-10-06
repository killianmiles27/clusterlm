#include <doctest/doctest.h>

#include <type_traits>

#include "clusterlm/domain/backend_adapter.hpp"
#include "pipeline_harness.hpp"

using namespace clusterlm;
using namespace clusterlm::domain;
using namespace clusterlm::objects;
using namespace clusterlm::testutil;

namespace {

DomainSpec spec_for(StageRole role, std::uint32_t b, std::uint32_t e, std::uint32_t ctx = 64, std::uint32_t win = 8) {
  DomainSpec s;
  s.stage = StageId{1};
  s.role = role;
  s.layers = {b, e};
  s.max_context = ctx;
  s.max_window = win;
  s.max_sessions = 2;
  return s;
}

WindowRequest wreq(Epoch e, SessionId s, std::uint64_t id, std::uint64_t base, std::uint64_t state, std::uint32_t q) {
  WindowRequest r;
  r.epoch = e;
  r.session = s;
  r.window = WindowId{id};
  r.base_position = base;
  r.expected_state = StateVersion{state};
  r.positions = q;
  return r;
}

CommitRequest creq(const WindowRequest& w, std::uint32_t accepted) {
  CommitRequest c;
  c.epoch = w.epoch;
  c.session = w.session;
  c.window = w.window;
  c.accepted = accepted;
  c.expected_state = w.expected_state;
  return c;
}

std::uint64_t bytes_of(const ModelManifest& m, std::string_view name) { return m.find(name)->byte_size; }

}  // namespace

// The middle path takes only activations: this is a compile-time property of the interface.
static_assert(std::is_same_v<decltype(&ExecutionDomain::run_window),
                             Result<StageActivations> (ExecutionDomain::*)(const WindowRequest&, const StageActivations&)>);

TEST_CASE("reference backend identifies itself and creates domains") {
  auto backend = make_reference_backend();
  const BackendInfo info = backend->info();
  CHECK(info.name == "reference");
  CHECK(info.hardware_available);
  CHECK_FALSE(info.supports_gpu);
  auto d = backend->create_domain(tiny_fixture().manifest, spec_for(StageRole::kMiddle, 3, 5));
  REQUIRE(d.is_ok());
  CHECK((*d)->spec().layers == LayerRange{3, 5});
  CHECK((*d)->boundary() == BoundaryLayout::for_geometry(tiny_fixture().manifest.geometry));
}

TEST_CASE("domain role and range validation") {
  const ModelManifest& m = full_fixture().manifest;  // ple_layer = 2, 16 layers
  auto rejects = [&](StageRole r, std::uint32_t b, std::uint32_t e) {
    return !ReferenceDomain::create(m, spec_for(r, b, e)).is_ok();
  };
  CHECK_FALSE(rejects(StageRole::kPrefix, 0, 4));
  CHECK_FALSE(rejects(StageRole::kPrefix, 0, 16));
  CHECK(rejects(StageRole::kPrefix, 1, 4));    // must start at layer 0
  CHECK(rejects(StageRole::kPrefix, 0, 2));    // must contain the PLE layer
  CHECK_FALSE(rejects(StageRole::kMiddle, 4, 10));
  CHECK(rejects(StageRole::kMiddle, 1, 5));    // contains the PLE layer
  CHECK(rejects(StageRole::kMiddle, 2, 3));
  CHECK(rejects(StageRole::kMiddle, 5, 5));    // empty
  CHECK(rejects(StageRole::kMiddle, 10, 17));  // beyond the model
  CHECK_FALSE(rejects(StageRole::kTail, 13, 16));
  CHECK_FALSE(rejects(StageRole::kTail, 16, 16));  // head-only tail
  CHECK(rejects(StageRole::kTail, 10, 12));    // must end at the last layer
  CHECK(rejects(StageRole::kTail, 0, 16));     // contains the PLE layer
  DomainSpec s = spec_for(StageRole::kMiddle, 4, 6);
  s.max_window = s.max_context + 1;
  CHECK_FALSE(ReferenceDomain::create(m, s).is_ok());
  s = spec_for(StageRole::kMiddle, 4, 6);
  s.max_sessions = 0;
  CHECK_FALSE(ReferenceDomain::create(m, s).is_ok());
}

TEST_CASE("a partial domain loads only its own objects") {
  const Fixture& f = full_fixture();
  const ModelManifest& m = f.manifest;

  SUBCASE("middle [4,8)") {
    auto store = f.open_store();
    auto d = ReferenceDomain::create(m, spec_for(StageRole::kMiddle, 4, 8));
    REQUIRE(d.is_ok());
    REQUIRE((*d)->prepare(*store).is_ok());
    const LayerRange r{4, 8};
    CHECK(store->loaded_bytes() == m.total_bytes(r));
    CHECK(store->loaded_object_count() == m.layer_objects(r).size());
    CHECK((*d)->read_metrics().resident_weight_bytes == m.total_bytes(r));
    CHECK(store->loaded_bytes() < m.total_bytes({0, 16}) / 3);
  }
  SUBCASE("prefix [0,4) adds embedding and PLE only") {
    auto store = f.open_store();
    auto d = ReferenceDomain::create(m, spec_for(StageRole::kPrefix, 0, 4));
    REQUIRE(d.is_ok());
    REQUIRE((*d)->prepare(*store).is_ok());
    CHECK(store->loaded_bytes() ==
          m.total_bytes({0, 4}) + bytes_of(m, kEmbeddingObjectName) + bytes_of(m, kPleObjectName));
    CHECK(store->loaded_object_count() == m.layer_objects({0, 4}).size() + 2);
  }
  SUBCASE("tail [13,16) adds the head only; head-only tail loads nothing else") {
    auto store = f.open_store();
    auto d = ReferenceDomain::create(m, spec_for(StageRole::kTail, 13, 16));
    REQUIRE(d.is_ok());
    REQUIRE((*d)->prepare(*store).is_ok());
    CHECK(store->loaded_bytes() == m.total_bytes({13, 16}) + bytes_of(m, kHeadObjectName));

    auto store2 = f.open_store();
    auto head_only = ReferenceDomain::create(m, spec_for(StageRole::kTail, 16, 16));
    REQUIRE(head_only.is_ok());
    REQUIRE((*head_only)->prepare(*store2).is_ok());
    CHECK(store2->loaded_bytes() == bytes_of(m, kHeadObjectName));
    CHECK(store2->loaded_object_count() == 1);
  }
}

TEST_CASE("prepare needs every required object and fails closed when one is missing") {
  const Fixture& f = tiny_fixture();
  auto store = f.open_store();
  auto d = ReferenceDomain::create(f.manifest, spec_for(StageRole::kMiddle, 3, 5));
  REQUIRE(d.is_ok());
  auto req = (*d)->describe_requirements();
  REQUIRE(req.is_ok());

  InMemoryResolver mem(f.manifest);
  for (const std::string& n : req->required_objects) {
    auto b = store->read_object_bytes(n);
    REQUIRE(b.is_ok());
    REQUIRE(mem.add(n, std::move(b).value()).is_ok());
  }
  // A domain must not fetch anything beyond its own objects: provisioning other layers is irrelevant to it.
  CHECK(mem.size() == req->required_objects.size());
  REQUIRE(mem.remove(expert_object_name(4, 1)));
  CHECK((*d)->prepare(mem).code() == ErrorCode::kNotFound);
  CHECK((*d)->open_session(Epoch{1}, SessionId{1}).code() == ErrorCode::kFailedPrecondition);  // still unprepared

  auto b = store->read_object_bytes(expert_object_name(4, 1));
  REQUIRE(mem.add(expert_object_name(4, 1), std::move(b).value()).is_ok());
  CHECK((*d)->prepare(mem).is_ok());
  CHECK((*d)->prepare(mem).code() == ErrorCode::kFailedPrecondition);  // already prepared
  CHECK((*d)->release().is_ok());
  CHECK((*d)->read_metrics().resident_weight_bytes == 0);
  CHECK((*d)->prepare(mem).is_ok());
}

TEST_CASE("describe_requirements accounts weights, state and window memory") {
  const Fixture& f = full_fixture();
  const ModelManifest& m = f.manifest;
  const ModelGeometry& g = m.geometry;
  auto d = ReferenceDomain::create(m, spec_for(StageRole::kMiddle, 4, 8, /*ctx=*/64, /*win=*/8));
  REQUIRE(d.is_ok());
  auto req = (*d)->describe_requirements();
  REQUIRE(req.is_ok());
  CHECK(req->cpu_weight_bytes == m.total_bytes({4, 8}));
  CHECK(req->gpu_weight_bytes == 0);
  CHECK(req->required_objects.size() == 4 * (2 + g.n_experts));
  // layers 4,5,6 recurrent, layer 7 attention.
  const std::uint64_t kvd = std::uint64_t{g.n_kv_heads} * g.head_dim;
  const std::uint64_t per_session = 3 * g.hidden_size * 4 + 2 * kvd * 64 * 4;
  CHECK(req->state_bytes == per_session * 2);
  CHECK(req->window_bytes == 3 * 8 * g.hidden_size * 4);
  CHECK(req->scratch_bytes > 0);

  // Marking objects as GPU-resident moves their bytes between the two buckets.
  std::unordered_map<std::string, AllocationTarget> targets;
  std::uint64_t moved = 0;
  for (std::uint32_t e = 0; e < g.n_experts; ++e) {
    targets[expert_object_name(5, e)] = AllocationTarget::kGpuResident;
    moved += m.find(expert_object_name(5, e))->byte_size;
  }
  (*d)->set_allocation_targets(targets);
  auto req2 = (*d)->describe_requirements();
  REQUIRE(req2.is_ok());
  CHECK(req2->gpu_weight_bytes == moved);
  CHECK(req2->cpu_weight_bytes == m.total_bytes({4, 8}) - moved);

  auto prefix = ReferenceDomain::create(m, spec_for(StageRole::kPrefix, 0, 4));
  auto preq = (*prefix)->describe_requirements();
  REQUIRE(preq.is_ok());
  CHECK(preq->cpu_weight_bytes ==
        m.total_bytes({0, 4}) + bytes_of(m, kEmbeddingObjectName) + bytes_of(m, kPleObjectName));
}

TEST_CASE("expert selections are counted per CPU/GPU allocation target") {
  const Fixture& f = tiny_fixture();
  const ModelGeometry& g = f.manifest.geometry;
  auto store = f.open_store();
  const std::vector<std::int32_t> toks{1, 2, 3};

  auto run = [&](bool all_gpu) {
    auto d = ReferenceDomain::create(f.manifest, spec_for(StageRole::kPrefix, 0, 3));
    REQUIRE(d.is_ok());
    if (all_gpu) {
      std::unordered_map<std::string, AllocationTarget> t;
      for (std::uint32_t L = 0; L < 3; ++L)
        for (std::uint32_t e = 0; e < g.n_experts; ++e) t[expert_object_name(L, e)] = AllocationTarget::kGpuResident;
      (*d)->set_allocation_targets(t);
    }
    REQUIRE((*d)->prepare(*store).is_ok());
    REQUIRE((*d)->open_session(Epoch{1}, SessionId{1}).is_ok());
    REQUIRE((*d)->run_prefix(wreq(Epoch{1}, SessionId{1}, 1, 0, 0, 3), toks).is_ok());
    return (*d)->last_timing();
  };
  const std::uint32_t expected = 3 /*positions*/ * 3 /*layers*/ * g.n_active_experts;
  const StageTiming cpu = run(false);
  CHECK(cpu.experts_selected == expected);
  CHECK(cpu.experts_cpu == expected);
  CHECK(cpu.experts_gpu == 0);
  CHECK(cpu.compute_ns > 0);
  const StageTiming gpu = run(true);
  CHECK(gpu.experts_selected == expected);
  CHECK(gpu.experts_gpu == expected);
  CHECK(gpu.experts_cpu == 0);
  CHECK(gpu.gpu_ns == 0);  // the reference backend executes everything on the CPU
}

TEST_CASE("role gating: each entry point only works on its own role") {
  const Fixture& f = tiny_fixture();
  auto store = f.open_store();
  Pipeline p(f.manifest, *store, {{0, 3, 5, 8}});
  p.open(Epoch{1}, SessionId{1});
  const std::vector<std::int32_t> toks{4, 5};
  const WindowRequest req = wreq(Epoch{1}, SessionId{1}, 1, 0, 0, 2);
  StageActivations dummy;
  dummy.layout = p.stages[1]->boundary();
  dummy.positions = 2;
  dummy.data.assign(2 * dummy.layout.floats_per_position(), 0.0f);

  CHECK(p.stages[1]->run_prefix(req, toks).status().code() == ErrorCode::kFailedPrecondition);
  CHECK(p.stages[2]->run_prefix(req, toks).status().code() == ErrorCode::kFailedPrecondition);
  CHECK(p.stages[0]->run_window(req, dummy).status().code() == ErrorCode::kFailedPrecondition);
  CHECK(p.stages[0]->run_tail(req, dummy).status().code() == ErrorCode::kFailedPrecondition);
  CHECK(p.stages[1]->run_tail(req, dummy).status().code() == ErrorCode::kFailedPrecondition);
  // None of the above consumed a window id: the real window still runs.
  CHECK(p.run(req, toks).is_ok());
}

TEST_CASE("request validation: stale epoch, state, positions, tokens and payloads") {
  const Fixture& f = tiny_fixture();
  auto store = f.open_store();
  Pipeline p(f.manifest, *store, {{0, 3, 5, 8}});
  const Epoch e1{1};
  const SessionId s1{1};
  p.open(e1, s1);
  const std::vector<std::int32_t> toks{4, 5, 6};

  CHECK(p.stages[0]->run_prefix(wreq(Epoch{2}, s1, 1, 0, 0, 3), toks).status().code() == ErrorCode::kStaleEpoch);
  CHECK(p.stages[0]->read_metrics().stale_rejections == 1);
  CHECK(p.stages[0]->run_prefix(wreq(e1, SessionId{9}, 1, 0, 0, 3), toks).status().code() == ErrorCode::kNotFound);
  CHECK(p.stages[0]->run_prefix(wreq(e1, s1, 1, 1, 0, 3), toks).status().code() == ErrorCode::kFailedPrecondition);
  CHECK(p.stages[0]->run_prefix(wreq(e1, s1, 1, 0, 1, 3), toks).status().code() == ErrorCode::kFailedPrecondition);
  CHECK(p.stages[0]->run_prefix(wreq(e1, s1, 1, 0, 0, 0), {}).status().code() == ErrorCode::kInvalidArgument);

  // Bad payload shape: rejected, window id consumed, session unchanged.
  CHECK(p.stages[0]->run_prefix(wreq(e1, s1, 1, 0, 0, 2), toks).status().code() == ErrorCode::kInvalidArgument);
  const std::vector<std::int32_t> oob{4, 9999, 6};
  CHECK(p.stages[0]->run_prefix(wreq(e1, s1, 2, 0, 0, 3), oob).status().code() == ErrorCode::kInvalidArgument);
  CHECK(p.stages[0]->run_prefix(wreq(e1, s1, 3, 0, 0, 9), std::vector<std::int32_t>(9, 1)).status().code() ==
        ErrorCode::kResourceExhausted);  // > max_window
  CHECK(p.stages[0]->run_prefix(wreq(e1, s1, 4, 0, 0, 3), toks).is_ok());

  // Middle-stage input checks, against activations produced for window 4 of the prefix.
  CHECK(p.stages[0]->run_prefix(wreq(e1, s1, 5, 0, 0, 3), toks).status().code() == ErrorCode::kFailedPrecondition);  // window 4 outstanding
  auto act0 = p.stages[0]->run_prefix(wreq(e1, SessionId{2}, 1, 0, 0, 3), toks);
  CHECK(act0.status().code() == ErrorCode::kNotFound);  // session 2 was never opened
  REQUIRE(p.stages[0]->commit_window(creq(wreq(e1, s1, 4, 0, 0, 3), 3)).is_ok());
  REQUIRE(p.stages[0]->open_session(e1, SessionId{2}).is_ok());
  act0 = p.stages[0]->run_prefix(wreq(e1, SessionId{2}, 1, 0, 0, 3), toks);
  REQUIRE(act0.is_ok());

  StageActivations wrong_pos = *act0;
  wrong_pos.first_position += 1;
  CHECK(p.stages[1]->run_window(wreq(e1, s1, 1, 0, 0, 3), wrong_pos).status().code() == ErrorCode::kInvalidArgument);
  StageActivations wrong_layout = *act0;
  wrong_layout.layout.hidden_size += 1;
  wrong_layout.data.assign(3 * wrong_layout.layout.floats_per_position(), 0.0f);
  CHECK(p.stages[1]->run_window(wreq(e1, s1, 2, 0, 0, 3), wrong_layout).status().code() == ErrorCode::kVersionMismatch);
  StageActivations short_payload = *act0;
  short_payload.data.pop_back();
  CHECK(p.stages[1]->run_window(wreq(e1, s1, 3, 0, 0, 3), short_payload).status().code() == ErrorCode::kInvalidArgument);
  CHECK(p.stages[1]->run_window(wreq(e1, s1, 4, 0, 0, 3), *act0).is_ok());
}

TEST_CASE("max_context and max_sessions are enforced") {
  const Fixture& f = tiny_fixture();
  auto store = f.open_store();
  DomainSpec s = spec_for(StageRole::kPrefix, 0, 3, /*ctx=*/6, /*win=*/4);
  auto d = ReferenceDomain::create(f.manifest, s);
  REQUIRE(d.is_ok());
  REQUIRE((*d)->prepare(*store).is_ok());
  REQUIRE((*d)->open_session(Epoch{1}, SessionId{1}).is_ok());
  REQUIRE((*d)->open_session(Epoch{1}, SessionId{2}).is_ok());
  CHECK((*d)->open_session(Epoch{1}, SessionId{3}).code() == ErrorCode::kResourceExhausted);
  const std::vector<std::int32_t> t4{1, 2, 3, 4};
  REQUIRE((*d)->run_prefix(wreq(Epoch{1}, SessionId{1}, 1, 0, 0, 4), t4).is_ok());
  REQUIRE((*d)->commit_window(creq(wreq(Epoch{1}, SessionId{1}, 1, 0, 0, 4), 4)).is_ok());
  CHECK((*d)->run_prefix(wreq(Epoch{1}, SessionId{1}, 2, 4, 1, 4), t4).status().code() == ErrorCode::kResourceExhausted);
  CHECK((*d)->run_prefix(wreq(Epoch{1}, SessionId{1}, 3, 4, 1, 2), std::vector<std::int32_t>{1, 2}).is_ok());
}

TEST_CASE("commit semantics: idempotent replay, wrong window, bad accepted, abort") {
  const Fixture& f = tiny_fixture();
  auto store = f.open_store();
  Pipeline p(f.manifest, *store, {{0, 3, 5, 8}});
  const Epoch e{1};
  const SessionId s{1};
  p.open(e, s);
  const std::vector<std::int32_t> toks{7, 8, 9, 10};
  const WindowRequest w1 = wreq(e, s, 1, 0, 0, 4);
  REQUIRE(p.run(w1, toks).is_ok());

  CHECK(p.commit(creq(w1, 0)).code() == ErrorCode::kInvalidArgument);
  CHECK(p.commit(creq(w1, 5)).code() == ErrorCode::kInvalidArgument);
  WindowRequest other = w1;
  other.window = WindowId{2};
  CHECK(p.commit(creq(other, 1)).code() == ErrorCode::kFailedPrecondition);

  CommitAck first, second;
  REQUIRE(p.commit(creq(w1, 3), &first).is_ok());
  REQUIRE(p.commit(creq(w1, 3), &second).is_ok());  // replay
  CHECK(first == second);
  CHECK(first.committed_position == 3);
  CHECK(first.state == StateVersion{1});
  for (auto& st : p.stages) CHECK(st->read_metrics().windows_committed == 1);
  for (auto& st : p.stages) CHECK(st->read_metrics().positions_committed == 3);
  CHECK(p.commit(creq(w1, 2)).code() == ErrorCode::kFailedPrecondition);  // not the same commit

  // Next window starts at the committed position, not at the end of the previous window.
  CHECK(p.run(wreq(e, s, 2, 4, 1, 1), std::vector<std::int32_t>{1}).status().code() == ErrorCode::kFailedPrecondition);
  REQUIRE(p.run(wreq(e, s, 2, 3, 1, 1), std::vector<std::int32_t>{1}).is_ok());

  // Abort with a window outstanding discards everything; the session must be re-opened.
  for (auto& st : p.stages) REQUIRE(st->abort_session(e, s).is_ok());
  for (auto& st : p.stages) CHECK(st->read_metrics().windows_aborted == 1);
  CHECK(p.run(wreq(e, s, 3, 3, 1, 1), std::vector<std::int32_t>{1}).status().code() == ErrorCode::kNotFound);
  CHECK(p.commit(creq(wreq(e, s, 2, 3, 1, 1), 1)).code() == ErrorCode::kNotFound);
  p.open(e, s);
  CHECK(p.run(wreq(e, s, 1, 0, 0, 1), std::vector<std::int32_t>{1}).is_ok());
}

TEST_CASE("committing fewer positions equals having run only those positions") {
  const Fixture& f = tiny_fixture();
  auto store = f.open_store();
  const Epoch e{1};
  const SessionId s{1};
  const std::vector<std::int32_t> window{3, 9, 14, 20};  // q = 4, keep 2
  const std::vector<std::int32_t> prefix_only{3, 9};
  const std::vector<std::int32_t> next{9, 1, 33};  // deliberately not the rejected tokens

  Pipeline a(f.manifest, *store, {{0, 3, 5, 8}});
  Pipeline b(f.manifest, *store, {{0, 3, 5, 8}});
  a.open(e, s);
  b.open(e, s);

  const WindowRequest wa = wreq(e, s, 1, 0, 0, 4);
  REQUIRE(a.run(wa, window).is_ok());
  REQUIRE(a.commit(creq(wa, 2)).is_ok());
  const WindowRequest wb = wreq(e, s, 1, 0, 0, 2);
  REQUIRE(b.run(wb, prefix_only).is_ok());
  REQUIRE(b.commit(creq(wb, 2)).is_ok());

  auto la = a.run(wreq(e, s, 2, 2, 1, 3), next);
  auto lb = b.run(wreq(e, s, 2, 2, 1, 3), next);
  REQUIRE(la.is_ok());
  REQUIRE(lb.is_ok());
  REQUIRE(la->data.size() == lb->data.size());
  CHECK(std::memcmp(la->data.data(), lb->data.data(), la->data.size() * 4) == 0);
}

TEST_CASE("restarting a session under a new epoch reproduces the same logits") {
  const Fixture& f = tiny_fixture();
  auto store = f.open_store();
  Pipeline p(f.manifest, *store, {{0, 3, 5, 8}});
  const std::vector<std::int32_t> toks{5, 6, 7, 8};

  p.open(Epoch{1}, SessionId{1});
  auto first = p.run(wreq(Epoch{1}, SessionId{1}, 1, 0, 0, 4), toks);
  REQUIRE(first.is_ok());
  REQUIRE(p.commit(creq(wreq(Epoch{1}, SessionId{1}, 1, 0, 0, 4), 4)).is_ok());
  for (auto& st : p.stages) CHECK(st->open_session(Epoch{1}, SessionId{1}).code() == ErrorCode::kAlreadyExists);
  p.open(Epoch{2}, SessionId{1});  // newer epoch re-establishes the session from scratch
  auto again = p.run(wreq(Epoch{2}, SessionId{1}, 1, 0, 0, 4), toks);
  REQUIRE(again.is_ok());
  CHECK(std::memcmp(first->data.data(), again->data.data(), first->data.size() * 4) == 0);
  // The old epoch is now a straggler.
  CHECK(p.run(wreq(Epoch{1}, SessionId{1}, 2, 4, 1, 1), std::vector<std::int32_t>{1}).status().code() ==
        ErrorCode::kStaleEpoch);

  // release() + prepare() gives a domain with identical behaviour.
  for (auto& st : p.stages) {
    REQUIRE(st->release().is_ok());
    REQUIRE(st->prepare(*store).is_ok());
  }
  p.open(Epoch{3}, SessionId{1});
  auto third = p.run(wreq(Epoch{3}, SessionId{1}, 1, 0, 0, 4), toks);
  REQUIRE(third.is_ok());
  CHECK(std::memcmp(first->data.data(), third->data.data(), first->data.size() * 4) == 0);
}

TEST_CASE("sessions in one domain do not interfere") {
  const Fixture& f = tiny_fixture();
  auto store = f.open_store();
  Pipeline p(f.manifest, *store, {{0, 3, 5, 8}});
  const std::vector<std::int32_t> a{1, 2, 3}, b{40, 41, 42};
  p.open(Epoch{1}, SessionId{1});
  p.open(Epoch{1}, SessionId{2});
  auto la = p.run(wreq(Epoch{1}, SessionId{1}, 1, 0, 0, 3), a);
  auto lb = p.run(wreq(Epoch{1}, SessionId{2}, 1, 0, 0, 3), b);
  REQUIRE(la.is_ok());
  REQUIRE(lb.is_ok());
  REQUIRE(p.commit(creq(wreq(Epoch{1}, SessionId{1}, 1, 0, 0, 3), 3)).is_ok());
  REQUIRE(p.commit(creq(wreq(Epoch{1}, SessionId{2}, 1, 0, 0, 3), 3)).is_ok());

  // The same continuation in session 1 must match a pipeline that never saw session 2.
  Pipeline solo(f.manifest, *store, {{0, 3, 5, 8}});
  solo.open(Epoch{1}, SessionId{1});
  REQUIRE(solo.run(wreq(Epoch{1}, SessionId{1}, 1, 0, 0, 3), a).is_ok());
  REQUIRE(solo.commit(creq(wreq(Epoch{1}, SessionId{1}, 1, 0, 0, 3), 3)).is_ok());
  const std::vector<std::int32_t> cont{11, 12};
  auto with_other = p.run(wreq(Epoch{1}, SessionId{1}, 2, 3, 1, 2), cont);
  auto alone = solo.run(wreq(Epoch{1}, SessionId{1}, 2, 3, 1, 2), cont);
  REQUIRE(with_other.is_ok());
  REQUIRE(alone.is_ok());
  CHECK(std::memcmp(with_other->data.data(), alone->data.data(), alone->data.size() * 4) == 0);
}
