// Admission: the eight checks of scheduler-admission-v1 §3 in order, alias selection (ADR 0402), context classes.
#include <doctest/doctest.h>

#include "clusterlm/scheduler/admission.hpp"

using namespace clusterlm::scheduler;

namespace {

ProfileSnapshot profile(const std::string& id = "prof_a", const std::string& api = "model-a") {
  ProfileSnapshot p;
  p.id = id;
  p.revision = 3;
  p.api_model_id = api;
  p.exposed_api = true;
  p.context_max_tokens = 16384;
  p.context_default_tokens = 4096;
  p.offered_contexts = {4096, 8192, 16384};
  return p;
}
ClientInfo client() {
  ClientInfo c;
  c.id = "key1";
  c.allowed_models = {"*"};
  return c;
}
ReadinessView rv(ReadyState s) {
  ReadinessView r;
  r.state = s;
  return r;
}
struct Env {
  ProfileSnapshot p = profile();
  ClientInfo c = client();
  RequestShape req;
  ReadinessView ready = rv(ReadyState::kReady);
  ResourceView res;
  QueueView q;
  AdmissionPolicy pol;
  Env() { q.slot_free = true; req.prompt_tokens = 100; req.max_tokens = 200; }
  Decision run(bool alias_prepare = false) {
    ResolvedTarget t;
    t.profile = &p;
    t.api_model_id = p.api_model_id;
    return admit(c, t, p, req, ready, res, q, pol, alias_prepare);
  }
};
}  // namespace

TEST_CASE("a ready profile with a free slot is admitted immediately") {
  Env e;
  auto d = e.run();
  CHECK(d.kind == Decision::Kind::kAdmit);
  CHECK(d.profile_id == "prof_a");
  CHECK(d.profile_revision == 3);
  CHECK(d.completion_budget == 200);
  CHECK(d.context_class == 4096);
}

TEST_CASE("busy slot or a non-empty queue queues, FIFO is never jumped") {
  Env e;
  e.q.slot_free = false;
  CHECK(e.run().kind == Decision::Kind::kQueue);
  e.q.slot_free = true;
  e.q.depth = 1;
  CHECK(e.run().kind == Decision::Kind::kQueue);
}

TEST_CASE("check 1: auth, scope, visibility") {
  Env e;
  SUBCASE("revoked") { e.c.revoked = true; CHECK(e.run().code == Code::kInvalidApiKey); }
  SUBCASE("no inference scope") { e.c.scope_inference = false; CHECK(e.run().code == Code::kInsufficientScope); }
  SUBCASE("model not in allowed_models is 404, not 403") {
    e.c.allowed_models = {"other"};
    auto d = e.run();
    CHECK(d.code == Code::kModelNotFound);
    CHECK(http_status(d.code) == 404);
  }
  SUBCASE("api exposure off") { e.p.exposed_api = false; CHECK(e.run().code == Code::kModelNotFound); }
  SUBCASE("LAN key needs allow_lan") {
    e.c.limits.lan = true;
    CHECK(e.run().code == Code::kModelNotFound);
    e.p.exposed_lan = true;
    CHECK(e.run().kind == Decision::Kind::kAdmit);
  }
}

TEST_CASE("check 2: body size and unsupported parameters") {
  Env e;
  e.req.body_bytes = e.pol.max_body_bytes + 1;
  CHECK(e.run().code == Code::kRequestTooLarge);
  e.req.body_bytes = 10;
  e.req.unsupported = {"logprobs"};
  auto d = e.run();
  CHECK(d.code == Code::kUnsupportedParameter);
  CHECK(d.message.find("logprobs") != std::string::npos);
}

TEST_CASE("check 3: context is checked against the profile, with classes") {
  Env e;
  SUBCASE("class grows with the need") {
    e.req.prompt_tokens = 4000;
    e.req.max_tokens = 200;
    CHECK(e.run().context_class == 8192);
    e.req.prompt_tokens = 16000;
    e.req.max_tokens = 384;
    CHECK(e.run().context_class == 16384);
  }
  SUBCASE("exceeding the profile maximum is rejected, never truncated") {
    e.req.prompt_tokens = 16000;
    e.req.max_tokens = 385;
    auto d = e.run();
    CHECK(d.code == Code::kContextLengthExceeded);
    CHECK(d.message.find("16384") != std::string::npos);
  }
  SUBCASE("prompt that fills the context leaves no room") {
    e.req.prompt_tokens = 16384;
    e.req.max_tokens.reset();
    CHECK(e.run().code == Code::kContextLengthExceeded);
  }
  SUBCASE("default budget is the rest of the context, capped by profile and policy") {
    e.req.prompt_tokens = 1000;
    e.req.max_tokens.reset();
    CHECK(e.run().completion_budget == 15384);
    e.p.max_completion_tokens_cap = 2048;
    CHECK(e.run().completion_budget == 2048);
    e.pol.max_completion_tokens_cap = 512;
    CHECK(e.run().completion_budget == 512);
  }
  SUBCASE("a request larger than the prepared plan but within the profile is not a context error") {
    e.req.prompt_tokens = 9000;
    e.req.max_tokens = 100;
    CHECK(e.run().context_class == 16384);
  }
}

TEST_CASE("check 4: client limits") {
  Env e;
  SUBCASE("requests per minute") {
    e.c.limits.requests_per_minute = 10;
    e.q.client_requests_last_minute = 10;
    auto d = e.run();
    CHECK(d.code == Code::kRateLimitExceeded);
    CHECK(d.retry_after_s.has_value());
  }
  SUBCASE("per-client queue share") {
    e.q.client_queued = 8;
    CHECK(e.run().code == Code::kRateLimitExceeded);
    e.q.client_queued = 7;
    e.q.slot_free = false;
    CHECK(e.run().kind == Decision::Kind::kQueue);
  }
  SUBCASE("client concurrency") {
    e.q.client_in_flight = e.c.limits.max_concurrent;
    CHECK(e.run().code == Code::kRateLimitExceeded);
  }
}

TEST_CASE("check 5: readiness states map to structured outcomes") {
  Env e;
  SUBCASE("busy queues") { e.ready = rv(ReadyState::kBusy); e.q.slot_free = false; CHECK(e.run().kind == Decision::Kind::kQueue); }
  SUBCASE("preparing fails fast with the job id when the client does not wait") {
    e.ready = rv(ReadyState::kPreparing);
    e.ready.job_id = "job_7";
    auto d = e.run();
    CHECK(d.code == Code::kModelNotReady);
    CHECK(d.job_id == "job_7");
    CHECK(d.retry_after_s.has_value());
    CHECK_FALSE(d.start_prepare);
  }
  SUBCASE("preparing queues behind the job when the client waits") {
    e.ready = rv(ReadyState::kPreparing);
    e.c.limits.prepare_wait_ms = 30'000;
    auto d = e.run();
    CHECK(d.kind == Decision::Kind::kQueue);
    CHECK(d.wait_for_prepare);
  }
  SUBCASE("loadable + on-demand starts a prepare job but the API client still fails fast") {
    e.ready = rv(ReadyState::kLoadable);
    auto d = e.run();
    CHECK(d.code == Code::kModelNotReady);
    CHECK(d.start_prepare);
  }
  SUBCASE("loadable + manual never starts a prepare job") {
    e.ready = rv(ReadyState::kLoadable);
    e.p.preparation = Preparation::kManual;
    auto d = e.run();
    CHECK(d.code == Code::kModelNotReady);
    CHECK_FALSE(d.start_prepare);
  }
  SUBCASE("loadable + waiting client queues and starts the job (interactive chat)") {
    e.ready = rv(ReadyState::kLoadable);
    e.c.limits.prepare_wait_ms = 600'000;
    auto d = e.run();
    CHECK(d.kind == Decision::Kind::kQueue);
    CHECK(d.start_prepare);
    CHECK(d.wait_for_prepare);
  }
  SUBCASE("compatible (transient) is workers_unavailable; unavailable is model_not_ready") {
    e.ready = rv(ReadyState::kCompatible);
    e.ready.reasons = {"G14 is in use"};
    e.c.limits.can_see_machine_names = true;
    auto d = e.run();
    CHECK(d.code == Code::kWorkersUnavailable);
    CHECK(d.reasons.front() == "G14 is in use");
    e.ready = rv(ReadyState::kUnavailable);
    CHECK(e.run().code == Code::kModelNotReady);
  }
  SUBCASE("keys without status:read get generic reasons with no machine names") {
    e.ready = rv(ReadyState::kCompatible);
    e.ready.reasons = {"G14 is in use"};
    auto d = e.run();
    REQUIRE(d.reasons.size() == 1);
    CHECK(d.reasons.front() == "a required Worker is not available");
    CHECK(d.message.find("G14") == std::string::npos);
  }
  SUBCASE("memory blocker") {
    e.ready = rv(ReadyState::kCompatible);
    e.ready.blockers = {"insufficient_memory"};
    CHECK(e.run().code == Code::kInsufficientMemory);
  }
}

TEST_CASE("check 6: resource policy") {
  Env e;
  e.res.workers_allowed = false;
  e.res.reason = "sharing is paused on G14";
  auto d = e.run();
  CHECK(d.code == Code::kWorkersUnavailable);
  CHECK(d.message == "sharing is paused on G14");
  CHECK_FALSE(d.start_prepare);
}

TEST_CASE("check 7 and 0: global queue capacity and shutdown") {
  Env e;
  e.q.slot_free = false;
  e.q.depth = e.q.max_queue_depth;
  CHECK(e.run().code == Code::kQueueFull);
  e.q.depth = e.q.max_queue_depth - 1;
  CHECK(e.run().kind == Decision::Kind::kQueue);
  e.q.shutting_down = true;
  CHECK(e.run().code == Code::kShuttingDown);
}

TEST_CASE("first failing check wins") {
  Env e;
  e.c.revoked = true;
  e.req.body_bytes = ~0ULL;
  e.q.depth = 999;
  CHECK(e.run().code == Code::kInvalidApiKey);
}

TEST_CASE("alias selection: deterministic, never widens access, prepare only with allow_prepare") {
  const auto a = profile("prof_a", "model-a");
  const auto b = profile("prof_b", "model-b");
  AliasSnapshot alias;
  alias.id = "alias_x";
  alias.api_model_id = "auto";
  alias.candidates = {"prof_a", "prof_b"};
  auto mk = [&](const ProfileSnapshot& p, ReadyState s, bool visible = true) {
    CandidateView c;
    c.profile = &p;
    c.readiness = rv(s);
    c.visible = visible;
    return c;
  };
  SUBCASE("first-viable picks list order among prepared") {
    std::vector<CandidateView> v = {mk(a, ReadyState::kReady), mk(b, ReadyState::kReady)};
    CHECK(select_candidate(alias, v, nullptr) == 0);
  }
  SUBCASE("a loadable candidate is not viable without allow_prepare") {
    std::vector<CandidateView> v = {mk(a, ReadyState::kLoadable), mk(b, ReadyState::kReady)};
    CHECK(select_candidate(alias, v, nullptr) == 1);
    alias.allow_prepare = true;
    CHECK(select_candidate(alias, v, nullptr) == 0);
  }
  SUBCASE("best-ready prefers a prepared candidate over an earlier loadable one") {
    alias.allow_prepare = true;
    alias.strategy = AliasStrategy::kBestReady;
    std::vector<CandidateView> v = {mk(a, ReadyState::kLoadable), mk(b, ReadyState::kBusy)};
    CHECK(select_candidate(alias, v, nullptr) == 1);
    v = {mk(a, ReadyState::kLoadable), mk(b, ReadyState::kLoadable)};
    CHECK(select_candidate(alias, v, nullptr) == 0);
  }
  SUBCASE("invisible candidates are never considered") {
    std::vector<CandidateView> v = {mk(a, ReadyState::kReady, false), mk(b, ReadyState::kReady)};
    CHECK(select_candidate(alias, v, nullptr) == 1);
  }
  SUBCASE("no viable candidate explains each visible one") {
    std::vector<CandidateView> v = {mk(a, ReadyState::kCompatible), mk(b, ReadyState::kUnavailable, false)};
    std::vector<std::string> why;
    CHECK_FALSE(select_candidate(alias, v, &why).has_value());
    REQUIRE(why.size() == 1);
    CHECK(why.front().find("model-a") != std::string::npos);
  }
}

TEST_CASE("error vocabulary matches the frozen table") {
  CHECK(to_string(Code::kModelNotReady) == "model_not_ready");
  CHECK(http_status(Code::kQueueTimeout) == 504);
  CHECK(http_status(Code::kQueueFull) == 503);
  CHECK(http_status(Code::kRateLimitExceeded) == 429);
  CHECK(error_type(Code::kInsufficientScope) == "permission_error");
  CHECK(error_type(Code::kWorkerLost) == "server_error");
  CHECK(error_type(Code::kContextLengthExceeded) == "invalid_request_error");
}
