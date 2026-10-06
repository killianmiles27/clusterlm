// The CUDA Strata backend (CLUSTERLM_ENABLE_STRATA). What needs no device runs everywhere: hardware detection,
// domain construction rules, sizing from Strata's own arithmetic (session_bytes, Verifier::init_bytes), and the
// clean kHardwareUnavailable at prepare(). Cases that need a GPU and a real model report a skip and pass; they are
// the hardware qualification entries HQ-NUM-01, HQ-GPU-02, HQ-GPU-05 and HQ-MTP-02.
#include <doctest/doctest.h>

#include <cuda_runtime.h>

#include <cstdlib>

#include "clusterlm/backends/strata/strata_domain.hpp"
#include "clusterlm/backends/strata_backend.hpp"
#include "strata/core/layout.hpp"
#include "strata/core/session.hpp"
#include "strata_test_support.hpp"

using namespace clusterlm;
using domain::StageRole;

namespace {

bool have_device() {
  int n = 0;
  const bool ok = cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
  (void)cudaGetLastError();
  return ok;
}

domain::DomainSpec spec_of(StageRole role, std::uint32_t begin, std::uint32_t end, std::uint32_t ctx = 32768) {
  domain::DomainSpec s;
  s.role = role;
  s.layers = {begin, end};
  s.max_context = ctx;
  s.max_window = 8;
  s.max_sessions = 1;
  return s;
}

const objects::ModelManifest& flash_next() {
  static const objects::ModelManifest m =
      strata_test::synthetic_manifest(strata_test::flash_next_geometry(), "iq3_s+iq4_nl");
  return m;
}

}  // namespace

TEST_CASE("strata backend reports its build and whether a CUDA device exists") {
  auto b = backends::make_strata_backend({});
  const auto info = b->info();
  CHECK(info.name == "strata");
  CHECK(info.build_hash.find("1735d6471df29b42c26170efaac1f1446a58640f") != std::string::npos);
  CHECK(info.supports_gpu);
  CHECK(info.hardware_available == have_device());
  if (!have_device()) MESSAGE("no CUDA device on this host: device cases below report SKIP");
}

TEST_CASE("domain construction follows the Strata kernels' contract") {
  auto b = backends::make_strata_backend({});
  CHECK(b->create_domain(flash_next(), spec_of(StageRole::kMiddle, 12, 24)).is_ok());
  CHECK(b->create_domain(flash_next(), spec_of(StageRole::kPrefix, 0, 12)).is_ok());
  CHECK(b->create_domain(flash_next(), spec_of(StageRole::kTail, 36, 48)).is_ok());
  // token-dependent layers stay on the prefix
  CHECK(b->create_domain(flash_next(), spec_of(StageRole::kMiddle, 2, 24)).status().code() == ErrorCode::kInvalidArgument);
  // a geometry the kernels were not built for is refused, never run
  const auto tiny = strata_test::synthetic_manifest(strata_test::tiny_geometry(), "q8_0");
  CHECK(b->create_domain(tiny, spec_of(StageRole::kMiddle, 3, 6, 256)).status().code() == ErrorCode::kVersionMismatch);
  auto broken = flash_next();
  broken.geometry.layer_kinds[5] = objects::LayerKind::kFullAttention;
  CHECK(b->create_domain(broken, spec_of(StageRole::kMiddle, 4, 8)).status().code() == ErrorCode::kVersionMismatch);
  backends::StrataBackendOptions wide;
  wide.max_window_cap = 9;
  CHECK(backends::make_strata_backend(wide)->create_domain(flash_next(), spec_of(StageRole::kMiddle, 12, 24)).status().code() ==
        ErrorCode::kInvalidArgument);
}

TEST_CASE("describe_requirements: Strata's own sizing, and a partial domain only pays for its layers") {
  auto b = backends::make_strata_backend({});
  const auto& m = flash_next();
  auto mid12 = b->create_domain(m, spec_of(StageRole::kMiddle, 12, 24));
  auto mid44 = b->create_domain(m, spec_of(StageRole::kMiddle, 3, 47));
  REQUIRE(mid12.is_ok());
  REQUIRE(mid44.is_ok());
  auto r12 = (*mid12)->describe_requirements();
  auto r44 = (*mid44)->describe_requirements();
  REQUIRE_MESSAGE(r12.is_ok(), r12.status().to_string());
  REQUIRE(r44.is_ok());

  // state: exactly Strata's session carve for the domain's range - not the whole model's
  strata::core::ModelGeometry sg;
  const std::uint64_t s12 = strata::core::session_bytes(sg, 32768, 10, 12, 24);
  const std::uint64_t s48 = strata::core::session_bytes(sg, 32768, 10, 0, 48);
  CHECK(r12->state_bytes == s12);
  CHECK(r44->state_bytes == strata::core::session_bytes(sg, 32768, 10, 3, 47));
  CHECK(r12->state_bytes < r44->state_bytes);
  CHECK(r12->state_bytes < s48);
  // the context-proportional part (QSA K/V + indexer) grows with the QSA layers owned: 3 in [12,24), 11 in [3,47)
  const std::uint64_t s12_short = strata::core::session_bytes(sg, 1024, 10, 12, 24);
  const std::uint64_t s44_short = strata::core::session_bytes(sg, 1024, 10, 3, 47);
  CHECK(r44->state_bytes - s44_short > s12 - s12_short);
  // and the recurrent part with the GDN layers owned (9 vs 33): same context, fewer layers, less state
  CHECK(s12_short < s44_short);
  MESSAGE("12-layer middle domain: state " << (s12 >> 20) << " MiB vs the whole model's " << (s48 >> 20)
                                          << " MiB at 32K context; window " << (r12->window_bytes >> 20) << " MiB");

  // weights: only the domain's own objects (12 layers x (dense + shared + 512 experts))
  CHECK(r12->required_objects.size() == 12 * 514);
  std::uint64_t own = 0;
  for (const auto& n : r12->required_objects) own += m.find(n)->byte_size;
  CHECK(r12->gpu_weight_bytes + r12->cpu_weight_bytes == own);
  CHECK(r12->cpu_weight_bytes == 12ull * 512 * (2 * 704000 + 921600));  // routed experts default to the CPU tier
  CHECK(r12->window_bytes > 0);
  CHECK(r12->staging_bytes > 0);
  CHECK(r12->scratch_bytes > 0);
  // the window state does not depend on how many layers the domain owns (one Verifier per session)
  CHECK(r12->window_bytes == r44->window_bytes);

  auto tail = b->create_domain(m, spec_of(StageRole::kTail, 36, 48));
  REQUIRE(tail.is_ok());
  auto rt = (*tail)->describe_requirements();
  REQUIRE(rt.is_ok());
  CHECK(rt->required_objects.back() == "output_head");
}

TEST_CASE("prepare without a CUDA device reports kHardwareUnavailable and leaves nothing behind") {
  if (have_device()) {
    MESSAGE("SKIP: a CUDA device is present; this case covers the device-less path");
    return;
  }
  auto b = backends::make_strata_backend({});
  auto d = b->create_domain(flash_next(), spec_of(StageRole::kMiddle, 12, 24));
  REQUIRE(d.is_ok());
  strata_test::NullResolver none;
  CHECK((*d)->prepare(none).code() == ErrorCode::kHardwareUnavailable);
  CHECK((*d)->open_session(Epoch{1}, SessionId{1}).code() == ErrorCode::kFailedPrecondition);
  CHECK((*d)->release().is_ok());
  CHECK((*d)->read_metrics().resident_weight_bytes == 0);
}

TEST_CASE("the MTP drafter binds only to a Strata tail created with an MTP directory") {
  auto b = backends::make_strata_backend({});
  auto mid = b->create_domain(flash_next(), spec_of(StageRole::kMiddle, 12, 24));
  REQUIRE(mid.is_ok());
  CHECK(backends::make_strata_mtp_drafter(**mid).status().code() == ErrorCode::kInvalidArgument);
  auto tail = b->create_domain(flash_next(), spec_of(StageRole::kTail, 36, 48));
  REQUIRE(tail.is_ok());
  CHECK(backends::make_strata_mtp_drafter(**tail).status().code() == ErrorCode::kFailedPrecondition);
  backends::StrataBackendOptions o;
  o.mtp_dir = "/nonexistent/mtp-rt";
  auto tail2 = backends::make_strata_backend(o)->create_domain(flash_next(), spec_of(StageRole::kTail, 36, 48));
  REQUIRE(tail2.is_ok());
  auto drafter = backends::make_strata_mtp_drafter(**tail2);
  REQUIRE(drafter.is_ok());
  // the tail with MTP also binds the embedding (Father-local)
  auto rt = (*tail2)->describe_requirements();
  REQUIRE(rt.is_ok());
  CHECK(rt->required_objects.front() == "token_embd");
  // without a prepared session the drafter falls back to a valid deterministic proposal
  const std::int32_t hist[3] = {1, 2, 3};
  CHECK((*drafter)->draft(hist, 4, 3).size() == 3);
}

TEST_CASE("GPU: verify windows on a real Strata model (qualification HQ-NUM-01 / HQ-GPU-05)") {
  const char* model = std::getenv("CLUSTERLM_STRATA_TEST_MODEL");
  if (!have_device() || model == nullptr) {
    MESSAGE("SKIP: needs a CUDA device and CLUSTERLM_STRATA_TEST_MODEL (a Father model directory with strata-dense "
            "objects); run the HQ-NUM-01 / HQ-GPU-05 commands on the target machines");
    return;
  }
  MESSAGE("CLUSTERLM_STRATA_TEST_MODEL is set but the end-to-end numerics run is the clusterlm-bench strata-numerics "
          "command (HQ-NUM-01), not a unit test");
}
