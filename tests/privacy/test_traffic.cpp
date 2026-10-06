// Traffic privacy: run a real Father against two real NodeWorkers (loopback TCP, real protocol), capture every
// frame that crosses any connection (Father -> Node, Node -> Father, Node -> Node) and prove that neither a
// distinctive token sequence nor the chat text ever appears in a Node-facing payload.
#include <doctest/doctest.h>

#include <algorithm>

#include "../coordinator/cluster_fixture.hpp"
#include "chat_fixture.hpp"
#include "clusterlm/protocol/messages.hpp"

using namespace clusterlm;
using namespace clusterlm::testing;

namespace {

// 12 distinctive token IDs, all inside the fixture vocabulary (256).
const std::vector<std::int32_t> kSecretPrompt = {201, 17, 99, 250, 3, 142, 77, 188, 9, 230, 61, 124};

// True if any captured payload contains the int32-LE image of the sequence or of any `k`-token window of it.
bool leaks_tokens(const std::vector<CapturedFrame>& frames, const std::vector<std::int32_t>& seq, std::size_t k) {
  const auto windows = token_windows(seq, k);
  for (const auto& f : frames)
    for (const auto& w : windows)
      if (contains(f.payload, w)) return true;
  return false;
}

std::size_t count_frames(const std::vector<CapturedFrame>& frames, const std::string& party, bool sent,
                         protocol::MessageType type) {
  return static_cast<std::size_t>(std::count_if(frames.begin(), frames.end(), [&](const CapturedFrame& f) {
    return f.party == party && f.sent == sent && f.type == static_cast<std::uint16_t>(type);
  }));
}

}  // namespace

TEST_CASE("positive control: the leak detector finds a token window when one is present") {
  CapturedFrame f;
  f.payload = Bytes(100, 0x11);
  const auto img = int32_le({kSecretPrompt[4], kSecretPrompt[5], kSecretPrompt[6]});
  f.payload.insert(f.payload.begin() + 37, img.begin(), img.end());
  CHECK(leaks_tokens({f}, kSecretPrompt, 3));
  CHECK_FALSE(leaks_tokens({CapturedFrame{"x", true, 0, 0, Bytes(100, 0x11)}}, kSecretPrompt, 3));
}

TEST_CASE("Coordinator generation: the prompt tokens never appear in any Father or Node traffic") {
  TrafficRecorder rec;
  std::vector<NodeOptions> nodes = {NodeOptions::faulty(rec.tap("node0")), NodeOptions::faulty(rec.tap("node1"))};
  TestCluster cluster("privacy-traffic", 2, nodes);
  auto cfg = cluster.config(/*direct=*/true);
  cfg.faults = rec.tap("father");
  auto father = cluster.father(cfg, kSplitPlan, 32);

  coordinator::GenerationRequest req;
  req.prompt = kSecretPrompt;
  req.max_new_tokens = 12;
  auto gen = father->generate(req);
  REQUIRE_MESSAGE(gen.is_ok(), gen.status().to_string());
  REQUIRE(gen->tokens.size() == 12);
  REQUIRE(father->release().is_ok());

  const auto frames = rec.frames();
  // The capture really contains the pipeline: provisioning, activations Father -> node0, node0 -> node1 (direct
  // peer path) and the final StageResult node1 -> Father.
  CHECK(count_frames(frames, "father", true, protocol::MessageType::kProvisionChunk) > 0);
  CHECK(count_frames(frames, "father", true, protocol::MessageType::kRunWindow) > 0);
  CHECK(count_frames(frames, "node0", true, protocol::MessageType::kRunWindow) > 0);   // node -> node
  CHECK(count_frames(frames, "node1", true, protocol::MessageType::kStageResult) > 0);  // node -> Father
  CHECK(count_frames(frames, "node1", false, protocol::MessageType::kRunWindow) > 0);   // received from the peer

  // Neither the whole prompt, nor any 3-token window of it, nor the generated continuation.
  CHECK_FALSE(leaks_tokens(frames, kSecretPrompt, kSecretPrompt.size()));
  CHECK_FALSE(leaks_tokens(frames, kSecretPrompt, 3));
  CHECK_FALSE(leaks_tokens(frames, gen->tokens, 3));
  // The only variable-sized, content-bearing payloads Nodes receive are digest-checked model objects and
  // fixed-ABI activations; every other frame is small and fixed-shape.
  for (const auto& f : frames) {
    const auto type = static_cast<protocol::MessageType>(f.type);
    if (type == protocol::MessageType::kProvisionChunk || type == protocol::MessageType::kRunWindow ||
        type == protocol::MessageType::kStageResult || type == protocol::MessageType::kPreparePlan)
      continue;
    CHECK_MESSAGE(f.payload.size() < 4096, std::string(protocol::to_string(type)));  // ProvisionStatus lists indices
  }
}

TEST_CASE("Father service chat: neither the chat text nor its tokens reach any Node") {
  ChatStack stack;
  const std::string user = "Distinctive-Payload-Zebra-42 tell me something";
  const std::string system = "You are PRIVATE-SYSTEM-PROMPT-QUOKKA";
  stack.run_chat(user, system, 24);

  const auto prompt_tokens = stack.tokenizer->encode_chat(std::vector<father::ChatMessage>{
      {father::ChatRole::kSystem, system}, {father::ChatRole::kUser, user}});
  const auto generated = stack.events->all_tokens();
  const std::string response = stack.events->all_text();
  REQUIRE(generated.size() == 24);

  const auto frames = stack.recorder.frames();
  REQUIRE(count_frames(frames, "node0", false, protocol::MessageType::kRunWindow) > 0);
  REQUIRE(count_frames(frames, "node1", true, protocol::MessageType::kStageResult) > 0);

  // Text: the raw UTF-8 of the system prompt, the user message, the chat-template opener and the response.
  for (const auto& f : frames) {
    CHECK_FALSE(contains(f.payload, user));
    CHECK_FALSE(contains(f.payload, system));
    CHECK_FALSE(contains(f.payload, std::string("PRIVATE-SYSTEM")));
    CHECK_FALSE(contains(f.payload, std::string("Zebra-42")));
    CHECK_FALSE(contains(f.payload, std::string("assistant: ")));
    if (response.size() >= 8) CHECK_FALSE(contains(f.payload, response.substr(0, 8)));
  }
  // Tokens: int32-LE windows of the chat prompt and of the generated tokens.
  CHECK_FALSE(leaks_tokens(frames, prompt_tokens, 3));
  CHECK_FALSE(leaks_tokens(frames, generated, 3));
  // The byte tokenizer makes token id == byte, so text bytes are tokens: also check 8-token windows packed as
  // single bytes (the most compact way anybody could smuggle them).
  Bytes prompt_bytes;
  for (auto t : prompt_tokens) prompt_bytes.push_back(static_cast<std::uint8_t>(t));
  for (const auto& f : frames)
    for (std::size_t i = 0; i + 8 <= prompt_bytes.size(); i += 4)
      CHECK_FALSE(contains(f.payload, Bytes(prompt_bytes.begin() + static_cast<std::ptrdiff_t>(i),
                                            prompt_bytes.begin() + static_cast<std::ptrdiff_t>(i + 8))));
  REQUIRE(stack.svc->release().is_ok());
}
