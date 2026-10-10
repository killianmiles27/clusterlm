// Dynamic topology resolution: pure, deterministic, honest about why a slot is empty.
#include <doctest/doctest.h>

#include <algorithm>

#include "clusterlm/profiles/topology.hpp"

using namespace clusterlm;
using namespace clusterlm::profiles;

namespace {
std::string fp(char c) { return std::string(64, c); }

WorkerCapability worker(char id, const char* name, std::uint32_t vram, const char* vendor = "nvidia", const char* cc = "8.6") {
  WorkerCapability w;
  w.fingerprint = fp(id);
  w.name = name;
  w.ram_mib = 32768;
  w.link_mbit = 1000;
  if (vram > 0) w.gpus.push_back({vendor, "gpu", vram, cc});
  w.backends = {"strata-hybrid"};
  return w;
}

Slot binding_slot(const char* slot, const char* binding, bool optional = false) {
  Slot s;
  s.slot = slot;
  s.optional = optional;
  s.select = Selector{Selector::Mode::kBinding, binding, "", std::nullopt};
  return s;
}
Slot req_slot(const char* slot, Requirements r, bool optional = false) {
  Slot s;
  s.slot = slot;
  s.optional = optional;
  s.select = Selector{Selector::Mode::kRequirements, "", "", std::move(r)};
  return s;
}
Topology topo(std::vector<Slot> workers) {
  Topology t;
  Slot host;
  host.kind = Slot::Kind::kHost;
  host.slot = "host";
  t.slots.push_back(host);
  for (auto& w : workers) t.slots.push_back(std::move(w));
  return t;
}
}  // namespace

TEST_CASE("binding slots resolve through the Host's bindings; unbound is never auto-bound") {
  const auto t = topo({binding_slot("w1", "node:laptop-class"), binding_slot("w2", "node:designated-3060")});
  std::vector<WorkerCapability> ws = {worker('a', "G14", 8192), worker('b', "Desk", 12288)};
  BindingMap b = {{"node:laptop-class", fp('a')}, {"node:designated-3060", fp('b')}};
  auto r = resolve_topology(t, b, ws);
  CHECK(r.satisfiable);
  CHECK(r.filled == 2);
  CHECK(r.fingerprints() == std::vector<std::string>{fp('a'), fp('b')});

  b.erase("node:designated-3060");
  r = resolve_topology(t, b, ws);
  CHECK_FALSE(r.satisfiable);
  CHECK(r.slots[1].state == SlotResolution::State::kUnsatisfied);
  CHECK(r.headline.find("is not assigned") != std::string::npos);
}

TEST_CASE("a busy, offline, paused, releasing or battery-powered Worker is not available compute") {
  const auto t = topo({binding_slot("w1", "g")});
  BindingMap b = {{"g", fp('a')}};
  for (auto st : {WorkerAvailability::kBusy, WorkerAvailability::kOffline, WorkerAvailability::kPaused, WorkerAvailability::kReleasing}) {
    auto w = worker('a', "G14", 8192);
    w.availability = st;
    auto r = resolve_topology(t, b, {w});
    CHECK_FALSE(r.satisfiable);
    CHECK_FALSE(r.headline.empty());
  }
  auto w = worker('a', "G14", 8192);
  w.on_battery = true;
  CHECK_FALSE(resolve_topology(t, b, {w}).satisfiable);
  ResolveOptions allow;
  allow.allow_on_battery = true;
  CHECK(resolve_topology(t, b, {w}, allow).satisfiable);
}

TEST_CASE("one machine fills at most one slot") {
  const auto t = topo({binding_slot("w1", "x"), binding_slot("w2", "y")});
  BindingMap b = {{"x", fp('a')}, {"y", fp('a')}};
  auto r = resolve_topology(t, b, {worker('a', "G14", 8192)});
  CHECK_FALSE(r.satisfiable);
  CHECK(r.filled == 1);
  CHECK(r.slots[1].reason.find("already fills") != std::string::npos);
}

TEST_CASE("requirements slots pick the most free VRAM, then the lowest fingerprint, over unchosen machines") {
  Requirements nv;
  nv.gpu_vendor = {"nvidia"};
  nv.min_vram_mib = 8192;
  const auto t = topo({binding_slot("w1", "x"), req_slot("w2", nv)});
  BindingMap b = {{"x", fp('c')}};
  std::vector<WorkerCapability> ws = {worker('c', "Bound", 24576), worker('a', "Small", 8192), worker('b', "Big", 24576), worker('d', "Amd", 32768, "amd")};
  auto r = resolve_topology(t, b, ws);
  REQUIRE(r.satisfiable);
  CHECK(r.slots[0].fingerprint == fp('c'));
  CHECK(r.slots[1].fingerprint == fp('b'));  // c is taken; b and a tie on... b has more VRAM than a
  // Tie on VRAM: lowest fingerprint.
  ws = {worker('c', "Bound", 24576), worker('e', "E", 16384), worker('b', "B", 16384)};
  r = resolve_topology(t, b, ws);
  CHECK(r.slots[1].fingerprint == fp('b'));
}

TEST_CASE("requirements are checked against what the Worker advertises; unknown never satisfies") {
  Requirements r;
  r.min_compute_capability = "8.6";
  CHECK(meets_requirements(worker('a', "ok", 8192, "nvidia", "8.9"), r));
  std::string why;
  CHECK_FALSE(meets_requirements(worker('a', "old", 8192, "nvidia", "7.5"), r, &why));
  CHECK(why.find("compute capability") != std::string::npos);
  CHECK_FALSE(meets_requirements(worker('a', "nocc", 8192, "nvidia", ""), r));
  Requirements link;
  link.min_link_mbit = 1000;
  auto w = worker('a', "w", 0);
  w.link_mbit = 0;
  link.cpu_only_ok = true;
  CHECK_FALSE(meets_requirements(w, link, &why));
  CHECK(why.find("unknown") != std::string::npos);
  w.link_mbit = 2500;
  CHECK(meets_requirements(w, link));  // CPU-only Worker allowed
  link.cpu_only_ok = false;
  CHECK_FALSE(meets_requirements(w, link));
  CHECK(compute_capability_at_least("10.0", "9.0"));
  CHECK_FALSE(compute_capability_at_least("8.6", "8.10"));
  CHECK_FALSE(compute_capability_at_least("x", "8.0"));
}

TEST_CASE("optional slots may stay empty within [min,max]; extras beyond max are shed") {
  Requirements nv;
  nv.gpu_vendor = {"nvidia"};
  auto t = topo({binding_slot("w1", "x"), req_slot("w2", nv, true)});
  BindingMap b = {{"x", fp('a')}};
  auto r = resolve_topology(t, b, {worker('a', "A", 8192)});
  CHECK(r.satisfiable);
  CHECK(r.filled == 1);
  CHECK(r.slots[1].state == SlotResolution::State::kEmptyOptional);

  r = resolve_topology(t, b, {worker('a', "A", 8192), worker('b', "B", 8192)});
  CHECK(r.satisfiable);
  CHECK(r.filled == 2);

  t.max_workers = 1;
  r = resolve_topology(t, b, {worker('a', "A", 8192), worker('b', "B", 8192)});
  CHECK(r.satisfiable);
  CHECK(r.filled == 1);
  CHECK(r.slots[1].state == SlotResolution::State::kEmptyOptional);

  // min_workers above what is available is unsatisfiable even though no slot is required.
  auto t2 = topo({req_slot("w1", nv, true), req_slot("w2", nv, true)});
  t2.min_workers = 2;
  CHECK_FALSE(resolve_topology(t2, {}, {worker('a', "A", 8192)}).satisfiable);
  CHECK(resolve_topology(t2, {}, {worker('a', "A", 8192), worker('b', "B", 8192)}).satisfiable);
}

TEST_CASE("a Worker without the profile's backend does not satisfy a selector") {
  const auto t = topo({binding_slot("w1", "x")});
  BindingMap b = {{"x", fp('a')}};
  ResolveOptions o;
  o.backend_id = "llama-local";
  auto r = resolve_topology(t, b, {worker('a', "A", 8192)}, o);
  CHECK_FALSE(r.satisfiable);
  CHECK(r.headline.find("llama-local") != std::string::npos);
}

TEST_CASE("resolution is deterministic regardless of worker list order") {
  Requirements any;
  any.cpu_only_ok = true;
  const auto t = topo({req_slot("w1", any), req_slot("w2", any)});
  std::vector<WorkerCapability> ws = {worker('c', "C", 8192), worker('a', "A", 8192), worker('b', "B", 8192)};
  const auto r1 = resolve_topology(t, {}, ws);
  std::reverse(ws.begin(), ws.end());
  const auto r2 = resolve_topology(t, {}, ws);
  CHECK(r1.fingerprints() == r2.fingerprints());
  CHECK(r1.fingerprints() == std::vector<std::string>{fp('a'), fp('b')});
}

TEST_CASE("worker capability documents round-trip and reject garbage") {
  auto w = worker('a', "G14", 8192);
  w.source = "worker advertisement";
  auto back = worker_capability_from_json(to_json(w));
  REQUIRE_MESSAGE(back.is_ok(), back.status().to_string());
  CHECK(*back == w);
  CHECK_FALSE(worker_capability_from_json("{\"fingerprint\":\"zz\",\"name\":\"x\"}").is_ok());
  CHECK_FALSE(worker_capability_from_json("{\"fingerprint\":\"" + fp('a') + "\",\"name\":\"x\",\"bogus\":1}").is_ok());
}
