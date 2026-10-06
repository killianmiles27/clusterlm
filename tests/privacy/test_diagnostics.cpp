// Diagnostics bundle: build info, recent logs, node statuses, plan, metrics, bench paths - and no conversation.
#include <doctest/doctest.h>

#include <fstream>
#include <sstream>

#include <nlohmann/json.hpp>

#include "chat_fixture.hpp"
#include "clusterlm/common/log.hpp"
#include "clusterlm/diagnostics/diagnostics.hpp"

using namespace clusterlm;
using namespace clusterlm::testing;
using nlohmann::json;

namespace {

const std::string kUser = "Distinctive-Payload-Zebra-42 tell me something";
const std::string kSystem = "You are PRIVATE-SYSTEM-PROMPT-QUOKKA";

}  // namespace

TEST_CASE("redact_line replaces content-bearing keys, token arrays, float dumps and blobs") {
  using diagnostics::redact_line;
  // Content keys lose their value; neighbouring structured fields survive.
  CHECK(redact_line("ts=1 level=info component=father event=x prompt=hello secret world lease=3") ==
        "ts=1 level=info component=father event=x prompt=[redacted:content] lease=3");
  CHECK(redact_line("event=chat text=a b c response=d e f") == "event=chat text=[redacted:content] response=[redacted:content]");
  CHECK(redact_line("event=chat Content=whatever") == "event=chat Content=[redacted:content]");
  // Token arrays in any common rendering.
  CHECK(redact_line("event=x ids=1,2,3,4,5") == "event=x ids=[redacted:int-list]");
  CHECK(redact_line("event=x ids=[201, 17, 99, 250]") == "event=x ids=[[redacted:int-list]]");
  CHECK(redact_line("event=x ids=201 17 99 250 3") == "event=x ids=[redacted:int-list]");
  // Three integers are not a token array; plans, ids and timings keep their digits.
  CHECK(redact_line("event=x a=1,2,3") == "event=x a=1,2,3");
  CHECK(redact_line("event=prepare_plan plan=0-4@father,4-10@0,10-13@1,13-16@father lease=7") ==
        "event=prepare_plan plan=0-4@father,4-10@0,10-13@1,13-16@father lease=7");
  CHECK(redact_line("event=x node=node2 lease=12 window=9") == "event=x node=node2 lease=12 window=9");
  // Activation dumps.
  CHECK(redact_line("event=x v=0.123456,-1.5e-3,2.25,3.5") == "event=x v=[redacted:float-dump]");
  CHECK(redact_line("event=x ms=12.5 other=3.25") == "event=x ms=12.5 other=3.25");
  // Long hex / base64 runs; 64-hex device ids and digests are identifiers and stay.
  const std::string dev(64, 'a');
  CHECK(redact_line("event=x device=" + dev) == "event=x device=" + dev);
  CHECK(redact_line("event=x blob=" + std::string(128, 'b')) == "event=x blob=[redacted:hex-blob]");
  CHECK(redact_line("event=x blob=" + std::string(200, 'Q')) == "event=x blob=[redacted:blob]");
  // Over-long values and control characters.
  CHECK(redact_line("event=x v=" + std::string(300, '#')) == "event=x [redacted:long-value]");
  CHECK(redact_line(std::string("event=x\x01y")).find('\x01') == std::string::npos);
}

TEST_CASE("a bundle built after a chat contains no conversation content") {
  diagnostics::LogRing ring(1024);
  const auto previous_level = log::level();
  log::set_level(log::Level::kDebug);
  ring.install();

  diagnostics::BundleInputs in;
  in.component = "father";
  std::string response;
  std::vector<std::int32_t> generated;
  {
    ChatStack stack;
    stack.run_chat(kUser, kSystem, 24);
    response = stack.events->all_text();
    generated = stack.events->all_tokens();
    REQUIRE(generated.size() == 24);

    for (auto* n : {stack.node0.get(), stack.node1.get()}) {
      const auto st = n->status();
      diagnostics::NodeStatusEntry e;
      e.name = n->endpoint().str();
      e.state = std::string(node::to_string(st.state));
      e.lease_generation = st.lease_generation;
      e.counters = {{"windows_executed", st.windows_executed},
                    {"windows_forwarded", st.windows_forwarded},
                    {"stale_rejections", st.stale_rejections},
                    {"provisioned_bytes", st.provisioned_bytes}};
      in.nodes.push_back(std::move(e));
    }
    in.plan_description = "0-4@father,4-10@0,10-13@1,13-16@father";
    const auto d = stack.svc->diagnostics(/*include_text=*/false);
    in.last_error = d.last_error;
    for (const auto& m : stack.svc->conversation()) in.conversation.push_back({std::string(father::to_string(m.role)), m.content});
    REQUIRE(in.conversation.size() == 3);
    REQUIRE(stack.svc->release().is_ok());
  }
  ring.uninstall();
  log::set_level(previous_level);
  in.log_lines = ring.snapshot();
  in.log_lines_dropped = ring.dropped();
  in.metrics = {{"decode_tok_s", 12.5, "tok/s", "synthetic"}, {"windows", 25, "count", ""}};
  in.bench_result_paths = {"results/cluster.json", "results/faults.json"};
  REQUIRE(in.log_lines.size() > 20);

  // ---- default bundle: no conversation ---------------------------------------------------------------------
  const std::string bundle = diagnostics::build_bundle(in);
  const json j = json::parse(bundle);
  CHECK(j["schema"] == "clusterlm.diagnostics.v1");
  CHECK(j["component"] == "father");
  CHECK(j["build"]["version"].is_string());
  CHECK(j["build"]["platform"].is_string());
  CHECK(j["logs"].size() == in.log_lines.size());
  CHECK(j["nodes"].size() == 2);
  CHECK(j["nodes"][0]["counters"]["windows_executed"].get<std::uint64_t>() > 0);
  CHECK(j["plan"] == "0-4@father,4-10@0,10-13@1,13-16@father");
  CHECK(j["metrics"][0]["provenance"] == "synthetic");
  CHECK(j["bench_result_paths"][1] == "results/faults.json");
  CHECK(j["redaction"]["conversation_included"] == false);
  CHECK_FALSE(j.contains("user_content"));

  CHECK(bundle.find(kUser) == std::string::npos);
  CHECK(bundle.find(kSystem) == std::string::npos);
  CHECK(bundle.find("Zebra") == std::string::npos);
  CHECK(bundle.find("QUOKKA") == std::string::npos);
  CHECK(bundle.find("assistant: ") == std::string::npos);
  if (response.size() >= 8) CHECK(bundle.find(response.substr(0, 8)) == std::string::npos);
  // Token arrays in decimal (4 consecutive generated tokens with any common separator).
  for (std::size_t i = 0; i + 4 <= generated.size(); ++i)
    for (const char* sep : {",", ", ", " "}) {
      std::string s;
      for (std::size_t k = 0; k < 4; ++k) s += (k ? std::string(sep) : std::string()) + std::to_string(generated[i + k]);
      CHECK(bundle.find(s) == std::string::npos);
    }

  // ---- explicit opt-in: the conversation appears, clearly marked ---------------------------------------------
  diagnostics::BundleOptions opt;
  opt.include_conversation = true;
  const json k = json::parse(diagnostics::build_bundle(in, opt));
  CHECK(k["redaction"]["conversation_included"] == true);
  REQUIRE(k.contains("user_content"));
  CHECK(k["user_content"]["warning"].get<std::string>().find("OPT-IN") != std::string::npos);
  CHECK(k["user_content"]["conversation"].size() == 3);
  CHECK(k["user_content"]["conversation"][1]["content"] == kUser);
  // Even with the opt-in, the conversation stays out of the logs section.
  CHECK(k["logs"].dump().find("Zebra") == std::string::npos);
}

TEST_CASE("bundle logs are bounded and the ring evicts oldest lines first") {
  diagnostics::LogRing ring(3);
  for (int i = 0; i < 10; ++i) ring.push("line " + std::to_string(i));
  const auto lines = ring.snapshot();
  REQUIRE(lines.size() == 3);
  CHECK(lines.front() == "line 7");
  CHECK(ring.dropped() == 7);

  diagnostics::BundleInputs in;
  for (int i = 0; i < 100; ++i) in.log_lines.push_back("event=x n=" + std::to_string(i));
  diagnostics::BundleOptions opt;
  opt.max_log_lines = 10;
  const json j = json::parse(diagnostics::build_bundle(in, opt));
  CHECK(j["logs"].size() == 10);
  CHECK(j["logs"][0] == "event=x n=90");
  CHECK(j["logs_dropped"] == 90);
}

TEST_CASE("write_bundle writes valid JSON atomically, and invalid UTF-8 in a field cannot break it") {
  const auto dir = privacy_temp_dir("bundle");
  diagnostics::BundleInputs in;
  in.component = "father";
  in.log_lines = {std::string("event=x bad=\xff\xfe\xfd end")};
  in.last_error = "status text \xc3\x28";
  const auto path = (dir / "bundle.json").string();
  REQUIRE(diagnostics::write_bundle(path, in).is_ok());
  std::ifstream f(path, std::ios::binary);
  std::stringstream ss;
  ss << f.rdbuf();
  const json j = json::parse(ss.str());
  CHECK(j["schema"] == "clusterlm.diagnostics.v1");
  CHECK(j["logs"].size() == 1);
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}
