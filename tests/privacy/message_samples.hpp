#pragma once
// One maximal, distinctive instance of EVERY protocol::Message alternative. Shared by the privacy schema test
// and by the fuzz seed-corpus generator. Adding a message type to the variant fails the static_assert below
// until a sample (and a privacy allowlist entry) is added deliberately.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "clusterlm/objects/fixture_model.hpp"
#include "clusterlm/protocol/messages.hpp"

namespace clusterlm::testing {

inline constexpr std::size_t kMessageAlternatives = std::variant_size_v<protocol::Message>;
static_assert(kMessageAlternatives == 26,
              "a protocol::Message alternative was added or removed: update sample_messages(), the privacy "
              "allowlist in tests/privacy/test_message_schema.cpp and docs/security/threat-model.md deliberately");

// Small deterministic fixture geometry: 4 layers (R,R,R,A), hidden 32, 4 residual streams.
inline objects::FixtureSpec sample_spec() {
  objects::FixtureSpec s;
  s.n_layers = 4;
  s.hidden = 32;
  s.residual_streams = 4;
  s.n_experts = 4;
  s.n_active = 2;
  s.expert_ff = 32;
  s.shared_expert_ff = 32;
  s.n_heads = 2;
  s.n_kv_heads = 1;
  s.head_dim = 16;
  s.vocab = 64;
  s.ple_layer = 2;
  s.ple_rows = 16;
  return s;
}

// Manifest of the sample fixture (written to a scratch directory that is removed again).
inline objects::ModelManifest sample_manifest() {
  static std::atomic<unsigned> counter{0};
  const auto dir = std::filesystem::temp_directory_path() /
                   ("clusterlm-samples-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
                    "-" + std::to_string(counter++));
  auto m = objects::write_fixture_model(sample_spec(), dir);
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
  return m.is_ok() ? m.value() : objects::ModelManifest{};
}

inline Digest256 sample_digest(std::uint8_t seed) {
  Digest256 d;
  for (std::size_t i = 0; i < d.bytes.size(); ++i) d.bytes[i] = static_cast<std::uint8_t>(seed + i);
  return d;
}

inline domain::StageActivations sample_activations(std::uint32_t positions, const domain::BoundaryLayout& layout,
                                                   std::uint64_t first_position = 0) {
  domain::StageActivations a;
  a.layout = layout;
  a.first_position = first_position;
  a.positions = positions;
  a.data.resize(std::size_t{positions} * layout.floats_per_position());
  for (std::size_t i = 0; i < a.data.size(); ++i) a.data[i] = 0.25f * static_cast<float>(i % 17) - 1.0f;
  return a;
}

inline domain::BoundaryLayout sample_layout() { return domain::BoundaryLayout::for_geometry(sample_manifest().geometry); }

inline domain::StageTiming sample_timing(std::uint32_t k) {
  return domain::StageTiming{1000u + k, 200u + k, 300u + k, 10u + k, 6u + k, 4u + k};
}

// Order follows the std::variant alternative order.
inline std::vector<protocol::Message> sample_messages(std::uint32_t positions = 3) {
  using namespace protocol;
  const objects::ModelManifest manifest = sample_manifest();
  const domain::BoundaryLayout layout = domain::BoundaryLayout::for_geometry(manifest.geometry);
  std::vector<Message> out;

  Hello hello;
  hello.role = NodeRole::kNode;
  hello.channel = Channel::kActivation;
  hello.device_id = "device-id-aaaaaaaa";
  hello.backend_build = "reference-cpu-fp32-v1";
  hello.lease = LeaseGeneration{7};
  out.emplace_back(hello);

  out.emplace_back(HelloAck{kProtocolVersion, "device-id-bbbbbbbb", LeaseGeneration{7}});

  OfferResources offer;
  offer.lease = LeaseGeneration{7};
  offer.safe_ram_bytes = 1ull << 30;
  offer.safe_vram_bytes = 1ull << 29;
  offer.staging_disk_bytes = 1ull << 20;
  offer.cpu_summary = "x86-64 avx2 16t";
  offer.gpu_summary = "gpu";
  offer.power = PowerSource::kAc;
  out.emplace_back(offer);

  PreparePlan plan;
  plan.lease = LeaseGeneration{7};
  plan.model_root = manifest.root_hash();
  plan.backend_build = "reference-cpu-fp32-v1";
  plan.plan_hash = sample_digest(9);
  plan.stages.push_back(StageAssignment{StageId{1}, domain::StageRole::kMiddle, objects::LayerRange{3, 4}, 64, 8, 4});
  plan.ram_cap_bytes = 1ull << 30;
  plan.vram_cap_bytes = 1ull << 29;
  plan.staging_cap_bytes = 1ull << 20;
  plan.manifest = manifest;
  for (std::uint32_t i = 0; i < manifest.objects.size(); ++i)
    if (!manifest.objects[i].father_only() && manifest.objects[i].layer && *manifest.objects[i].layer == 3)
      plan.assignments.push_back(ObjectAssignment{i, objects::AllocationTarget::kCpuResident});
  out.emplace_back(plan);

  out.emplace_back(PlanAccepted{LeaseGeneration{7}, sample_digest(9), 1000, 2000, 3000});

  ProvisionChunk chunk;
  chunk.lease = LeaseGeneration{7};
  chunk.object_index = 3;
  chunk.offset = 4096;
  chunk.data.assign(256, 0x5A);
  chunk.chunk_digest = Sha256::of(chunk.data);
  out.emplace_back(chunk);

  out.emplace_back(SealObject{LeaseGeneration{7}, 3, 4096 + 256, sample_digest(11)});
  out.emplace_back(ObjectSealed{LeaseGeneration{7}, 3});
  out.emplace_back(PlanReady{LeaseGeneration{7}, sample_digest(9), 1ull << 20, 123456});
  out.emplace_back(AuthorizePeer{LeaseGeneration{7}, sample_digest(9), StageId{1}, StageId{2}, "peer-device-id-cccc",
                                 "127.0.0.1:7002", LeaseGeneration{8}});
  out.emplace_back(OpenSession{Epoch{2}, SessionId{5}});
  out.emplace_back(SessionOpened{Epoch{2}, SessionId{5}});

  RunWindow run;
  run.lease = LeaseGeneration{7};
  run.request = domain::WindowRequest{Epoch{2}, SessionId{5}, WindowId{9}, 12, StateVersion{4}, positions};
  run.stage = StageId{1};
  run.activations = sample_activations(positions, layout, 12);
  run.forward_to_peer = true;
  run.auto_commit = true;
  run.upstream_timings = {sample_timing(1), sample_timing(2)};
  out.emplace_back(run);

  StageResult result;
  result.epoch = Epoch{2};
  result.session = SessionId{5};
  result.window = WindowId{9};
  result.stage = StageId{1};
  result.status = ErrorCode::kOk;
  result.activations = sample_activations(positions, layout, 12);
  result.timings = {sample_timing(3)};
  out.emplace_back(result);

  out.emplace_back(CommitWindow{domain::CommitRequest{Epoch{2}, SessionId{5}, WindowId{9}, 2, StateVersion{4}}, StageId{1}});
  out.emplace_back(CommitAckMessage{StageId{1}, domain::CommitAck{SessionId{5}, WindowId{9}, 14, StateVersion{5}}});
  out.emplace_back(AbortSession{Epoch{2}, SessionId{5}, "operator requested"});
  out.emplace_back(ReleaseLease{LeaseGeneration{7}, ReleaseReason::kLocalActivity});
  out.emplace_back(ReleaseComplete{LeaseGeneration{7}, true, true, 0, 555, "none"});
  out.emplace_back(ErrorMessage{ErrorCode::kStaleEpoch, "stale lease", MessageType::kRunWindow});
  out.emplace_back(Ping{42});
  out.emplace_back(Pong{42});
  out.emplace_back(UnpairNotice{43});
  out.emplace_back(AbortWindow{Epoch{2}, SessionId{5}, WindowId{9}, StageId{1}});
  out.emplace_back(WindowAborted{StageId{1}, domain::WindowAbortAck{SessionId{5}, WindowId{9}, 12, StateVersion{4}}});
  out.emplace_back(ProvisionStatus{LeaseGeneration{7}, {1, 2, 3}});
  return out;
}

}  // namespace clusterlm::testing
