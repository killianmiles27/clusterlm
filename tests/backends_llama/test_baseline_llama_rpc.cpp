// `clusterlm-bench baseline llama-rpc` proven locally: the pinned rpc-server on loopback, the tiny GGUF, Synthetic.
#include <doctest/doctest.h>

#include <fstream>

#include <nlohmann/json.hpp>

#include "clusterlm/platform/process.hpp"
#include "llama_test_support.hpp"

using namespace clusterlm;
using namespace clusterlm::llamatest;

namespace {

nlohmann::json run_bench(const std::vector<std::string>& extra, const std::filesystem::path& out, int expect_exit) {
  std::vector<std::string> args = {"baseline", "llama-rpc", "--out", out.string(), "--quiet"};
  args.insert(args.end(), extra.begin(), extra.end());
  auto p = platform::ChildProcess::spawn(platform::executable_dir() / "clusterlm-bench", args);
  REQUIRE_MESSAGE(p.is_ok(), p.status().to_string());
  auto code = p.value()->wait(std::chrono::seconds(240));
  REQUIRE(code.is_ok());
  CHECK(code.value() == expect_exit);
  std::ifstream f(out);
  REQUIRE(f);
  return nlohmann::json::parse(f);
}

bool check_passed(const nlohmann::json& r, const std::string& name) {
  for (const auto& c : r["checks"])
    if (c["name"] == name) return c["passed"].get<bool>();
  return false;
}

}  // namespace

TEST_CASE("baseline llama-rpc against local rpc-servers: throughput, wire bytes, empty cache") {
  TinyModel m("baseline");
  auto r = run_bench({"--model", m.gguf.string(), "--spawn-local", "2", "--prompt-tokens", "16", "--n-predict", "8",
                      "--repeat", "2", "--verify-local"},
                     m.dir / "r.json", 0);
  CHECK(r["provenance"] == "Synthetic");
  CHECK(r["experiment"] == "dev-baseline-llama-rpc");
  CHECK(r["simulated"]["fixture_model"] == true);
  CHECK(check_passed(r, "pin_matches_build"));
  CHECK(check_passed(r, "model_loaded_over_rpc"));
  CHECK(check_passed(r, "rpc_request_framing_parsed"));
  CHECK(check_passed(r, "tokens_match_local_run"));
  CHECK(check_passed(r, "rpc_cache_empty_node0"));
  CHECK(check_passed(r, "rpc_cache_empty_node1"));
  CHECK(r["metrics"]["decode_tok_s"]["n"] == 2);
  CHECK(r["metrics"]["decode_tok_s"]["mean"].get<double>() > 0);
  CHECK(r["metrics"]["prefill_tok_s"]["mean"].get<double>() > 0);
  CHECK(r["metrics"]["rpc.bytes_per_token"]["mean"].get<double>() > 0);
  CHECK(r["metrics"]["rpc.graph_calls_per_token"]["mean"].get<double>() >= 1);
  CHECK(r["metrics"]["rpc.load_bytes_to_nodes"].get<std::uint64_t>() > 0);
  CHECK(r["metrics"]["node0.fs.cache_dir_entries_after"] == 0);
}

TEST_CASE("baseline llama-rpc: the filesystem check sees a cache when the server is started with -c") {
  // Weights above the RPC hash threshold (10 MiB) are what the server persists, so this model needs a big tensor.
  backends::TinyLlamaSpec spec;
  spec.hidden = 512;
  spec.vocab = 8192;  // token_embd and output: 16 MiB each in F32
  spec.ff = 128;
  spec.heads = 8;
  spec.kv_heads = 2;
  TinyModel m("baseline-control", spec);
  auto r = run_bench({"--model", m.gguf.string(), "--spawn-local", "1", "--control-with-cache", "--prompt-tokens", "4",
                      "--n-predict", "2", "--repeat", "1"},
                     m.dir / "r.json", 0);
  CHECK(check_passed(r, "control_cache_visible_node0"));
  CHECK(r["metrics"]["node0.fs.cache_dir_entries_after"].get<std::uint64_t>() > 0);
}

TEST_CASE("baseline llama-rpc usage errors") {
  TinyModel m("baseline-usage");
  auto p = platform::ChildProcess::spawn(platform::executable_dir() / "clusterlm-bench",
                                         {"baseline", "llama-rpc", "--model", m.gguf.string()});
  REQUIRE(p.is_ok());
  auto code = p.value()->wait(std::chrono::seconds(30));
  REQUIRE(code.is_ok());
  CHECK(code.value() == 2);  // neither --nodes nor --spawn-local
}
