// WP21: the qualification harness driving real backends and real models. Everything here runs without a GPU and without a
// real model: the pieces are checked against the reference backend, the Strata fake engine (through the same
// run_cluster_pass the bench uses), synthetic and llama.cpp vocabularies, and the result files of the smoke runs.
#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "backend_cli.hpp"
#include "bench_backend.hpp"
#include "cluster_pass.hpp"
#include "clusterlm/backends/backend_factory.hpp"
#include "clusterlm/backends/strata/strata_domain.hpp"
#include "clusterlm/common/clock.hpp"
#include "clusterlm/platform/process.hpp"
#include "clusterlm/objects/fixture_model.hpp"
#include "commands.hpp"
#include "corpus_prompts.hpp"
#include "rank_agreement.hpp"
#include "../backends/strata_test_support.hpp"
#include "../father_service/synthetic_vocab.hpp"

using namespace clusterlm;
using namespace clusterlm::bench;
using nlohmann::json;
namespace fs = std::filesystem;

namespace {

cli::Args make_args(std::vector<std::string> words) {
  std::vector<std::string> storage = {"clusterlm-bench"};
  storage.insert(storage.end(), words.begin(), words.end());
  std::vector<char*> argv;
  for (auto& w : storage) argv.push_back(w.data());
  return cli::Args(static_cast<int>(argv.size()), argv.data(), 1);
}

fs::path temp_dir(const std::string& name) {
  const auto dir = fs::temp_directory_path() / ("clm-wp21-" + name + "-" + std::to_string(monotonic_ns()));
  fs::create_directories(dir);
  return dir;
}

json load_json_file(const fs::path& path) {
  std::ifstream in(path);
  REQUIRE_MESSAGE(in, "missing " << path);
  return json::parse(in);
}

void write_file(const fs::path& path, const std::string& text) {
  fs::create_directories(path.parent_path());
  std::ofstream(path, std::ios::binary) << text;
}

}  // namespace

// ---- predicted vs measured order --------------------------------------------------------------------------------

TEST_CASE("rank agreement: identical, reversed, tied and degenerate orderings") {
  const std::vector<double> predicted = {10, 20, 30, 40};
  auto same = compare_rankings(predicted, {1, 2, 3, 4}, true);
  REQUIRE(same.valid);
  CHECK(same.spearman == doctest::Approx(1.0));
  CHECK(same.concordant_fraction == doctest::Approx(1.0));
  CHECK(same.best_agrees);

  auto reversed = compare_rankings(predicted, {4, 3, 2, 1}, true);
  REQUIRE(reversed.valid);
  CHECK(reversed.spearman == doctest::Approx(-1.0));
  CHECK(reversed.concordant_fraction == doctest::Approx(0.0));
  CHECK_FALSE(reversed.best_agrees);
  // For times (lower is better) the same reversal agrees on which candidate is best.
  CHECK(compare_rankings(predicted, {4, 3, 2, 1}, false).best_agrees == false);
  CHECK(compare_rankings({1, 2, 3}, {1, 2, 3}, false).best_agrees);

  // One swapped neighbour pair: Spearman 0.8, 5 of 6 pairs concordant.
  auto swap = compare_rankings(predicted, {1, 3, 2, 4}, true);
  REQUIRE(swap.valid);
  CHECK(swap.spearman == doctest::Approx(0.8));
  CHECK(swap.concordant_fraction == doctest::Approx(5.0 / 6.0));

  // Ties share their average rank; pairs tied in either series do not count.
  auto ties = compare_rankings({1, 2, 3, 4}, {5, 5, 7, 9}, true);
  REQUIRE(ties.valid);
  CHECK(ties.spearman > 0.9);
  CHECK(ties.concordant_fraction == doctest::Approx(1.0));

  CHECK_FALSE(compare_rankings({1}, {1}, true).valid);             // nothing to order
  CHECK_FALSE(compare_rankings({1, 2, 3}, {5, 5, 5}, true).valid);  // a constant series orders nothing
  CHECK_FALSE(compare_rankings({1, 2}, {1, 2, 3}, true).valid);     // mismatched sizes
}

// ---- backend flags ----------------------------------------------------------------------------------------------

TEST_CASE("backend flags are shared with clusterlm-node: Nodes get the engine options, never the Father-only ones") {
  const auto args = make_args({"--backend", "strata", "--cuda-device", "1", "--vram-reserve-mib", "512", "--strata-cpu-threads", "6",
                               "--strata-ple-gguf", "/models/first-shard.gguf", "--strata-mtp-dir", "/models/mtp"});
  const auto node = cli::node_backend_args(args);
  const std::vector<std::string> expected = {"--backend", "strata", "--cuda-device", "1", "--vram-reserve-mib", "512",
                                             "--strata-cpu-threads", "6"};
  CHECK(node == expected);
  for (const auto& w : node) {
    CHECK(w.find("ple") == std::string::npos);  // the PLE table and the MTP head are Father-local
    CHECK(w.find("mtp") == std::string::npos);
  }
  const auto father = cli::backend_options_from_args(args, "model-dir");
  CHECK(father.name == "strata");
  CHECK(father.strata.cuda_device == 1);
  CHECK(father.strata.vram_reserve_mib == 512);
  CHECK(father.strata.cpu_threads == 6);
  CHECK(father.strata.ple_table_gguf == fs::path("/models/first-shard.gguf"));
  CHECK(father.strata.mtp_dir == fs::path("/models/mtp"));
  CHECK(cli::node_backend_args(make_args({})).empty());
  CHECK(backend_flags() == cli::backend_flag_names());
}

TEST_CASE("resolve_backend: reference always; unknown or unbuilt backends fail before anything is spawned") {
  auto ref = resolve_backend(make_args({}), "model", 2);
  REQUIRE(ref.is_ok());
  CHECK(ref->name == "reference");
  CHECK_FALSE(ref->real());
  CHECK(ref->father == nullptr);
  CHECK(ref->node_args.empty());

  auto unknown = resolve_backend(make_args({"--backend", "vulkan-magic"}), "model", 0);
  CHECK(unknown.status().code() == ErrorCode::kInvalidArgument);

  auto strata = resolve_backend(make_args({"--backend", "strata", "--cuda-device", "0"}), "model", 2);
  if (backends::backend_built("strata")) {
    // A CUDA build: the adapter exists; without a device its domains refuse to prepare (kHardwareUnavailable).
    REQUIRE(strata.is_ok());
    CHECK(strata->real());
    CHECK(strata->node_args.size() == 4);
  } else {
    CHECK(strata.status().code() == ErrorCode::kUnimplemented);
    CHECK(strata.status().message().find("CLUSTERLM_ENABLE_STRATA") != std::string::npos);
  }

  auto llama = resolve_backend(make_args({"--backend", "llama"}), "model", 1);
  CHECK_FALSE(llama.is_ok());  // unbuilt (kUnimplemented) or built but Father-only with a Node stage (kInvalidArgument)
  CHECK((llama.status().code() == ErrorCode::kUnimplemented || llama.status().code() == ErrorCode::kInvalidArgument));
}

// ---- the command surface ----------------------------------------------------------------------------------------

TEST_CASE("the new flags are in the command specs and the registry still parses") {
  for (const char* cmd : {"cluster", "faults", "placement-inputs", "placement-validate"}) {
    const auto* spec = find_command_spec(cmd);
    REQUIRE_MESSAGE(spec != nullptr, cmd);
    for (const auto& flag : backend_flags())
      CHECK_MESSAGE(check_command_tokens(*spec, {"--" + flag, "x"}).empty(), cmd << " --" << flag);
  }
  CHECK(check_command_tokens(*find_command_spec("cluster"), {"--corpus", "c.txt", "--tokenizer-gguf", "v.gguf", "--no-reference"}).empty());
  CHECK(check_command_tokens(*find_command_spec("faults"), {"--supervised", "--deadline-ms", "1500"}).empty());
  CHECK(check_command_tokens(*find_command_spec("placement-validate"), {"--top", "4", "--q", "1", "--repeat", "2"}).empty());
  CHECK(validate_registry_command("clusterlm-bench cluster --backend strata --model <dir> --corpus c.txt --tokenizer-gguf v.gguf").empty());
  CHECK_FALSE(validate_registry_command("clusterlm-bench placement-validate --no-such-flag").empty());
}

TEST_CASE("fault scenario selection: the supervised scenarios are opt-in") {
  const auto base = fault_scenario_names();
  const auto sup = supervised_fault_scenario_names();
  CHECK(sup == std::vector<std::string>{"supervised_forced_termination", "supervised_crash_recovery"});
  CHECK(select_fault_scenarios("").value() == base);  // default selection is unchanged
  for (const auto& n : sup) CHECK(std::find(base.begin(), base.end(), n) == base.end());
  auto with = select_fault_scenarios("", true).value();
  CHECK(with.size() == base.size() + sup.size());
  CHECK(select_fault_scenarios("supervised").value() == sup);
  CHECK(select_fault_scenarios("supervised_crash_recovery,stall").value() ==
        std::vector<std::string>{"stall", "supervised_crash_recovery"});
  CHECK(select_fault_scenarios("crash,supervised").value().size() == 9 + sup.size());
  CHECK_FALSE(select_fault_scenarios("supervised_nonsense").is_ok());
}

// ---- real text: corpus tokenized by the Father tokenizer ----------------------------------------------------------

TEST_CASE("corpus prompts: a synthetic GGUF vocabulary tokenizes the corpus; ids fold only on the fixture model") {
  const auto dir = temp_dir("corpus");
  const Bytes gguf = father::testing::synthetic_vocab_gguf();
  REQUIRE_FALSE(gguf.empty());
  write_file(dir / "vocab.gguf", std::string(gguf.begin(), gguf.end()));
  auto tokenizer = load_corpus_tokenizer(dir / "vocab.gguf");
  REQUIRE_MESSAGE(tokenizer.is_ok(), tokenizer.status().to_string());
  const std::uint32_t tokenizer_vocab = tokenizer.value()->vocab_size();
  REQUIRE(tokenizer_vocab > 256);

  std::string text;
  for (int i = 0; i < 40; ++i) text += "the quick brown fox hello world ";
  write_file(dir / "corpus.txt", text);
  const auto direct = tokenizer.value()->encode(text);
  REQUIRE(direct.size() >= 64);

  SUBCASE("a real-sized model vocabulary: the tokenizer's own ids, cut into consecutive windows") {
    auto cp = make_corpus_prompts(dir / "corpus.txt", 16, 4096, tokenizer.value().get(), /*fixture_model=*/false);
    REQUIRE_MESSAGE(cp.is_ok(), cp.status().to_string());
    CHECK(cp->tokenizer == "gguf-bpe");
    CHECK_FALSE(cp->ids_folded);
    CHECK(cp->source_tokens == direct.size());
    REQUIRE(cp->prompts.size() == std::min<std::size_t>(kMaxCorpusPrompts, direct.size() / 16));
    for (std::size_t w = 0; w < cp->prompts.size(); ++w) {
      REQUIRE(cp->prompts[w].size() == 16);
      CHECK(std::equal(cp->prompts[w].begin(), cp->prompts[w].end(), direct.begin() + static_cast<std::ptrdiff_t>(w * 16)));
    }
  }
  SUBCASE("the fixture model's 256-token vocabulary: ids fold into range and the run says so") {
    auto cp = make_corpus_prompts(dir / "corpus.txt", 16, 256, tokenizer.value().get(), /*fixture_model=*/true);
    REQUIRE(cp.is_ok());
    CHECK(cp->ids_folded);
    for (const auto& p : cp->prompts)
      for (auto id : p) CHECK((id >= 0 && id < 256));
  }
  SUBCASE("a real model whose vocabulary is smaller than the tokenizer's is the wrong tokenizer") {
    auto cp = make_corpus_prompts(dir / "corpus.txt", 16, 256, tokenizer.value().get(), /*fixture_model=*/false);
    CHECK(cp.status().code() == ErrorCode::kFailedPrecondition);
  }
  SUBCASE("a short text is repeated to the requested length; a directory gives one prompt per file") {
    write_file(dir / "docs" / "a.txt", "hello world");
    write_file(dir / "docs" / "b.txt", "the quick brown fox");
    auto cp = make_corpus_prompts(dir / "docs", 40, 4096, tokenizer.value().get(), false);
    REQUIRE(cp.is_ok());
    REQUIRE(cp->prompts.size() == 2);
    for (const auto& p : cp->prompts) CHECK(p.size() == 40);
    const auto a = tokenizer.value()->encode("hello world");
    for (std::size_t i = 0; i < 40; ++i) CHECK(cp->prompts[0][i] == a[i % a.size()]);
  }
  SUBCASE("without a tokenizer the bytes are the ids (the fixture path), and a missing corpus is kNotFound") {
    auto cp = make_corpus_prompts(dir / "docs-missing", 8, 256, nullptr, true);
    CHECK(cp.status().code() == ErrorCode::kNotFound);
    write_file(dir / "bytes.txt", "ABCDEFGH");
    auto bytes = make_corpus_prompts(dir / "bytes.txt", 8, 256, nullptr, true);
    REQUIRE(bytes.is_ok());
    CHECK(bytes->tokenizer == "fixture-bytes");
    CHECK(bytes->prompts.at(0) == std::vector<std::int32_t>{65, 66, 67, 68, 69, 70, 71, 72});
  }
  std::error_code ec;
  fs::remove_all(dir, ec);
}

TEST_CASE("corpus prompts: the llama.cpp qwen2 vocabulary (present with the pinned upstream checkout)") {
#ifdef CLUSTERLM_SOURCE_DIR
  const fs::path vocab = fs::path(CLUSTERLM_SOURCE_DIR) / "third_party/upstream/llama.cpp/models/ggml-vocab-qwen2.gguf";
#else
  const fs::path vocab;
#endif
  if (vocab.empty() || !fs::exists(vocab)) {
    MESSAGE("skipped: " << vocab << " not present (python3 scripts/fetch_upstream.py llama.cpp)");
    return;
  }
  auto tokenizer = load_corpus_tokenizer(vocab);
  REQUIRE_MESSAGE(tokenizer.is_ok(), tokenizer.status().to_string());
  CHECK(tokenizer.value()->vocab_size() > 150000);
  const std::string text = "Distributed inference across idle machines: the quick brown fox jumps over the lazy dog.";
  const auto ids = tokenizer.value()->encode(text);
  CHECK(ids.size() < text.size());  // real merges, not bytes
  CHECK(tokenizer.value()->decode(ids) == text);

  const auto dir = temp_dir("qwen2");
  std::string corpus;
  for (int i = 0; i < 30; ++i) corpus += text + "\n";
  write_file(dir / "corpus.txt", corpus);
  auto cp = make_corpus_prompts(dir / "corpus.txt", 64, 151936, tokenizer.value().get(), false);
  REQUIRE_MESSAGE(cp.is_ok(), cp.status().to_string());
  CHECK_FALSE(cp->ids_folded);
  REQUIRE(cp->prompts.size() >= 2);
  for (const auto& p : cp->prompts) CHECK(p.size() == 64);
  CHECK(cp->prompts[0] != cp->prompts[1]);  // distinct windows of the text
  auto folded = make_corpus_prompts(dir / "corpus.txt", 64, 256, tokenizer.value().get(), true);
  REQUIRE(folded.is_ok());
  CHECK(folded->ids_folded);
  std::error_code ec;
  fs::remove_all(dir, ec);
}

// ---- the bench's own pass over a Strata backend (fake engine) -----------------------------------------------------

namespace {

using clusterlm::strata_test::FakeEngine;

// The Strata backend adapter over the strata-core fake engine: what `--backend strata` is in the bench, minus CUDA.
class FakeStrataBackend final : public domain::BackendAdapter {
 public:
  domain::BackendInfo info() const override {
    domain::BackendInfo i;
    i.name = "strata";
    i.build_hash = "strata-fake-engine";
    i.supports_gpu = true;
    i.hardware_available = true;
    return i;
  }
  Result<std::unique_ptr<domain::ExecutionDomain>> create_domain(const objects::ModelManifest& m, const domain::DomainSpec& spec) override {
    CLM_ASSIGN_OR_RETURN(auto d, backends::strata::StrataDomain::create(m, spec, std::make_unique<FakeEngine>(spec.role, m.geometry)));
    return std::unique_ptr<domain::ExecutionDomain>(std::move(d));
  }
};

}  // namespace

TEST_CASE("the bench cluster pass runs through StrataDomain (fake engine) and reports per-domain state bytes") {
  const auto dir = temp_dir("strata-pass");
  const auto manifest = strata_test::write_synthetic_model(dir / "model");
  const std::uint32_t vocab = manifest.geometry.vocab_size;

  BackendChoice backend;
  backend.name = "strata";
  backend.father = std::make_shared<FakeStrataBackend>();
  backend.build_hash = "strata-fake-engine";

  // Father-only plan: the Node-free part of the Strata path. (The spawned clusterlm-node processes of a real run build their
  // backend through the factory only; the fake engine is an in-process seam, so Node stages cannot go through it from here -
  // see docs/adr/0350-bench-drives-real-backends.md.)
  const auto args = make_args({"--insecure", "--drafter", "scripted:0.3", "--no-resources"});
  BenchmarkResult result("test-strata-pass", probe_host());
  auto sets = build_prompt_sets(args, dir / "model", &backend, {20}, 6, /*with_reference=*/true, result);
  REQUIRE_MESSAGE(sets.is_ok(), sets.status().to_string());
  REQUIRE(sets->size() == 1);
  REQUIRE(sets->at(0).references.size() == 1);
  CHECK(sets->at(0).references[0].size() == 6);
  // The fake tail's logits grow with the token index: greedy always picks the last vocabulary entry.
  for (auto t : sets->at(0).references[0]) CHECK(t == static_cast<std::int32_t>(vocab) - 1);

  PassSummary summary;
  PassContext pc(args, result);
  pc.work = dir / "work";
  pc.model_dir = dir / "model";
  pc.plan_text = "0-3@father,3-8@father";
  pc.qs = {1, 3};
  pc.contexts = {20};
  pc.prompt_sets = std::move(sets).value();
  pc.max_new = 6;
  pc.prefill_chunk = 16;  // the fake engine's window is 8: local sub-batches
  pc.repeat = 1;
  pc.fixture = false;
  pc.resources = false;
  pc.backend = &backend;
  pc.summary = &summary;
  const Status st = run_cluster_pass(pc);
  REQUIRE_MESSAGE(st.is_ok(), st.to_string());
  CHECK(summary.completed);
  CHECK(summary.remote_stages == 0);

  const json doc = result.finish(0.0);
  for (const auto& c : doc["checks"]) CHECK_MESSAGE(c["passed"] == true, c["name"].get<std::string>() << ": " << c["detail"].get<std::string>());
  CHECK(doc["metrics"].contains("q3.mtp.acceptance_rate"));
  // The fake engine allocates 77 bytes of sequence state per open session; both Father domains report it at release.
  CHECK(doc["metrics"]["state.father.stage0.bytes"] == 77);
  CHECK(doc["metrics"]["state.father.stage1.bytes"] == 77);
  CHECK(doc["metrics"]["state.total_bytes"] == 154);
  CHECK(doc["metrics"]["state.father.stage0.window_bytes"] == 55);  // the engine's per-session window scratch
  CHECK(doc["metrics"]["state.total_window_bytes"] == 110);
  std::error_code ec;
  fs::remove_all(dir, ec);
}

// ---- Nodes that are already running (--node): real links, attached rather than spawned ------------------------------

TEST_CASE("--node: the bench attaches to Nodes somebody else runs, pins their identities and measures through them") {
  CHECK(is_loopback_host("127.0.0.1"));
  CHECK(is_loopback_host("localhost"));
  CHECK_FALSE(is_loopback_host("192.168.1.20"));
  CHECK_FALSE(is_loopback_host("g14.lan"));
  {
    auto bad = parse_external_nodes(make_args({"--node", "no-equals-sign"}));
    CHECK(bad.status().code() == ErrorCode::kInvalidArgument);
    auto bad_endpoint = parse_external_nodes(make_args({"--node", "a=not-an-endpoint"}));
    CHECK_FALSE(bad_endpoint.is_ok());
    auto none = parse_external_nodes(make_args({}));
    REQUIRE(none.is_ok());
    CHECK(none->empty());
    auto two = parse_external_nodes(make_args({"--node", "g14=10.0.0.2:47600@ab12", "--node", "n3060=10.0.0.3:47600@cd34"}));
    REQUIRE(two.is_ok());
    REQUIRE(two->size() == 2);
    CHECK(two->at(0).name == "g14");
    CHECK(two->at(0).endpoint.host == "10.0.0.2");
    CHECK(two->at(0).endpoint.port == 47600);
    CHECK(two->at(1).device_id == "cd34");
  }

  const auto dir = temp_dir("external");
  const auto fixture = objects::write_fixture_model(objects::FixtureSpec{}, dir / "model");
  REQUIRE(fixture.is_ok());

  // "The other machines": two clusterlm-node processes this test starts and then only reaches by endpoint.
  LocalClusterOptions host_opts;
  host_opts.work_dir = dir / "hosts";
  host_opts.nodes.resize(2);
  host_opts.nodes[0].name = "node0";
  host_opts.nodes[1].name = "node1";
  auto hosts = LocalCluster::start(host_opts);
  REQUIRE_MESSAGE(hosts.is_ok(), hosts.status().to_string());
  std::vector<std::string> words = {"--identity", (dir / "hosts" / "father-id").string(), "--no-resources"};
  for (const auto& ep : hosts.value()->endpoints()) {
    words.push_back("--node");
    words.push_back(ep.name + "=" + ep.endpoint.str() + "@" + ep.device_id);
  }
  const auto args = make_args(words);

  SUBCASE("a pass over attached Nodes: provisioning, generation, release reports - no local process or staging evidence") {
    BenchmarkResult result("test-external-pass", probe_host());
    BackendChoice backend;  // the reference backend
    auto sets = build_prompt_sets(args, dir / "model", nullptr, {24}, 6, true, result);
    REQUIRE_MESSAGE(sets.is_ok(), sets.status().to_string());
    PassSummary summary;
    PassContext pc(args, result);
    pc.work = dir / "work";
    pc.model_dir = dir / "model";
    pc.plan_text = "0-4@father,4-10@0,10-13@1,13-16@father";
    pc.qs = {1};
    pc.contexts = {24};
    pc.prompt_sets = std::move(sets).value();
    pc.max_new = 6;
    pc.repeat = 1;
    pc.resources = true;
    pc.backend = &backend;
    pc.summary = &summary;
    const Status st = run_cluster_pass(pc);
    REQUIRE_MESSAGE(st.is_ok(), st.to_string());
    CHECK(summary.completed);
    CHECK(summary.remote_stages == 2);
    const json doc = result.finish(0.0);
    for (const auto& c : doc["checks"]) CHECK_MESSAGE(c["passed"] == true, c["name"].get<std::string>() << ": " << c["detail"].get<std::string>());
    const auto& m = doc["metrics"];
    CHECK(m["prepare.node.node0.bytes"].get<double>() > 0);
    CHECK(m["prepare.node.node1.bytes"].get<double>() > 0);
    CHECK(m.contains("state.node0.stage1.bytes"));  // the Node's own report, carried in its ReleaseComplete
    CHECK(m.contains("post_release_census_zero.unavailable"));
    for (const auto& [key, value] : m.items()) {
      (void)value;
      CHECK_MESSAGE(key.rfind("resources.node", 0) != 0, key);  // no process memory of a machine this one cannot read
    }
  }
  SUBCASE("an endpoint without its device id cannot be attached over TLS; the plan cannot need more Nodes than are named") {
    LocalClusterOptions opts;
    opts.work_dir = dir / "attach";
    opts.external = {coordinator::NodeEndpoint{"x", hosts.value()->endpoints().front().endpoint, ""}};
    CHECK(LocalCluster::start(opts).status().code() == ErrorCode::kInvalidArgument);
    opts.tls = false;
    opts.external = {coordinator::NodeEndpoint{"x", transport::Endpoint{"192.168.1.9", 47600}, ""}};
    CHECK(LocalCluster::start(opts).status().code() == ErrorCode::kInvalidArgument);  // insecure only on loopback
    const auto one_node = make_args({"--node", "a=127.0.0.1:1@x"});
    auto s = setup(one_node, dir / "w2", dir / "model", 2, nullptr);
    CHECK(s.status().code() == ErrorCode::kInvalidArgument);
    // An attached cluster controls no process.
    auto attached = LocalCluster::start([&] {
      LocalClusterOptions o;
      o.work_dir = dir / "attach2";
      o.external = hosts.value()->endpoints();
      return o;
    }());
    REQUIRE_MESSAGE(attached.is_ok(), attached.status().to_string());
    CHECK(attached.value()->external());
    CHECK(attached.value()->size() == 2);
    CHECK(attached.value()->local_activity(0).code() == ErrorCode::kFailedPrecondition);
    CHECK(attached.value()->status(0).status().code() == ErrorCode::kFailedPrecondition);
    CHECK(attached.value()->pid(0) == 0);
  }
  std::error_code ec;
  fs::remove_all(dir, ec);
}

TEST_CASE("placement-validate --endpoint: each candidate runs on the already-running Nodes of the profiles it places") {
  const auto dir = temp_dir("endpoints");
  // The Nodes carry the profile ids (as `clusterlm-node --name` would on the real machines) and offer 8 GiB of VRAM.
  LocalClusterOptions host_opts;
  host_opts.work_dir = dir / "hosts";
  host_opts.nodes.resize(2);
  host_opts.nodes[0].name = "Node-G14-4070-8945HS";
  host_opts.nodes[1].name = "Node-3060-5600";
  for (auto& n : host_opts.nodes) n.vram_gib = 8.0;
  auto hosts = LocalCluster::start(host_opts);
  REQUIRE_MESSAGE(hosts.is_ok(), hosts.status().to_string());
  std::vector<std::string> argv = {"placement-validate", "--top", "4", "--repeat", "1", "--max-new", "6", "--prompt-len", "16",
                                   "--no-resources", "--identity", (dir / "hosts" / "father-id").string(), "--work", (dir / "work").string(),
                                   "--out", (dir / "result.json").string()};
  for (const auto& ep : hosts.value()->endpoints()) {
    argv.push_back("--endpoint");
    argv.push_back(ep.name + "=" + ep.endpoint.str() + "@" + ep.device_id);
  }
  auto bench = platform::ChildProcess::spawn(platform::executable_dir() / "clusterlm-bench", argv);
  REQUIRE_MESSAGE(bench.is_ok(), bench.status().to_string());
  auto code = bench.value()->wait(std::chrono::seconds(120));
  REQUIRE_MESSAGE(code.is_ok(), code.status().to_string());
  CHECK(code.value() == 0);
  const json doc = load_json_file(dir / "result.json");
  CHECK(doc["provenance"] == "Synthetic");
  CHECK(doc["configuration"]["external_nodes"] == 2);
  CHECK(doc["simulated"]["localhost_cluster"] == true);  // the endpoints are loopback here, so the links are not real
  std::size_t traced = 0, with_nodes = 0;
  for (const auto& t : doc["trace"])
    if (t.value("kind", "") == "placement_candidate") {
      ++traced;
      with_nodes += t["nodes"].get<std::size_t>() > 0;
      CHECK(t["completed"] == true);
    }
  CHECK(traced >= 2);
  CHECK(with_nodes >= 1);  // at least one candidate placed a stage on an attached Node
  for (const auto& c : doc["checks"]) CHECK_MESSAGE(c["passed"] == true, c["name"].get<std::string>() << ": " << c["detail"].get<std::string>());
  std::error_code ec;
  fs::remove_all(dir, ec);
}

// ---- result files of the smoke runs (tests/bench/CMakeLists.txt) --------------------------------------------------

#ifdef CLUSTERLM_BENCH_RESULTS_DIR
namespace {
std::string results(const char* name) { return std::string(CLUSTERLM_BENCH_RESULTS_DIR) + "/" + name; }
json load_json(const std::string& path) {
  std::ifstream in(path);
  REQUIRE_MESSAGE(in, "missing " << path);
  return json::parse(in);
}
void require_all_checks_pass(const json& doc) {
  for (const auto& c : doc["checks"]) CHECK_MESSAGE(c["passed"] == true, c["name"].get<std::string>() << ": " << c["detail"].get<std::string>());
}
}  // namespace

TEST_CASE("cluster --backend reference: Synthetic, labelled with its backend, and per-domain state bytes in the result") {
  const auto doc = load_json(results("result-cluster-backend.json"));
  CHECK(doc["provenance"] == "Synthetic");
  CHECK(doc["tool"]["backend"] == "reference");
  CHECK(doc["simulated"]["reference_backend"] == true);
  CHECK(doc["simulated"]["fixture_model"] == true);
  CHECK(doc["configuration"]["backend"] == "reference");
  const auto& m = doc["metrics"];
  for (const char* k : {"state.father.stage0.bytes", "state.node0.stage1.bytes", "state.total_bytes", "state.max_context",
                        "state.node0.stage1.window_bytes", "state.total_window_bytes"})
    CHECK_MESSAGE(m.contains(k), k);
  CHECK(m["state.node0.stage1.bytes"].get<double>() > 0);
  CHECK(m["state.node0.stage1.window_bytes"].get<double>() > 0);  // the recurrent snapshots of a window
  CHECK(m["state.total_bytes"].get<double>() >=
        m["state.father.stage0.bytes"].get<double>() + m["state.node0.stage1.bytes"].get<double>());
  require_all_checks_pass(doc);
}

TEST_CASE("cluster --no-reference makes no correctness check it cannot make") {
  const auto doc = load_json(results("result-cluster-noref.json"));
  CHECK(doc["configuration"]["reference_check"] == false);
  for (const auto& c : doc["checks"]) CHECK(c["name"].get<std::string>().find("father_only_reference") == std::string::npos);
  require_all_checks_pass(doc);
}

TEST_CASE("placement-validate: predicted next to measured for each candidate, and how the orders agree") {
  const auto doc = load_json(results("result-placement-validate.json"));
  CHECK(doc["provenance"] == "Synthetic");
  CHECK(doc["simulated"].contains("synthetic_profiles"));
  CHECK(doc["simulated"]["fixture_model"] == true);
  const auto n = doc["configuration"]["candidates_run"].get<std::size_t>();
  CHECK(n >= 2);
  std::size_t traced = 0;
  for (const auto& t : doc["trace"]) {
    if (t.value("kind", "") != "placement_candidate") continue;
    ++traced;
    CHECK(t["completed"] == true);
    CHECK(t["predicted_decode_tok_s"].get<double>() > 0);
    CHECK(t["measured_decode_tok_s"].get<double>() > 0);
    CHECK(t["measured_prefill_s"].get<double>() > 0);
    CHECK(t["predicted_prepare_s"].get<double>() >= 0);
  }
  CHECK(traced == n);
  const auto& m = doc["metrics"];
  for (const char* k : {"rank.decode_tok_s.candidates", "rank.decode_tok_s.valid", "rank.prefill_s.valid", "rank.prepare_s.valid",
                        "cand0.predicted.decode_tok_s", "cand0.measured.decode_tok_s", "cand0.plan"})
    CHECK_MESSAGE(m.contains(k), k);
  CHECK(m["rank.decode_tok_s.candidates"] == n);
  if (m["rank.decode_tok_s.valid"].get<bool>()) {
    CHECK(m["rank.decode_tok_s.spearman"].get<double>() >= -1.0);
    CHECK(m["rank.decode_tok_s.spearman"].get<double>() <= 1.0);
  }
  bool pending = false;
  for (const auto& id : doc["pending_qualification"]) pending |= id == "HQ-PLACE-01";
  CHECK(pending);
  require_all_checks_pass(doc);
}

#ifdef CLUSTERLM_QWEN2_VOCAB_PRESENT
TEST_CASE("cluster --corpus --tokenizer-gguf: the real tokenizer's ids drive the run; text and ids never reach the result") {
  const auto doc = load_json(results("result-cluster-corpus.json"));
  CHECK(doc["provenance"] == "Synthetic");
  CHECK(doc["simulated"]["tokenizer_ids_folded"] == true);  // a 150k vocabulary on the 256-token fixture model
  CHECK(doc["configuration"]["corpus_tokenizer"] == "gguf-bpe");
  REQUIRE(doc["configuration"]["corpus"].is_array());
  CHECK(doc["configuration"]["corpus"][0]["prompts"].get<std::size_t>() >= 1);
  CHECK(doc["configuration"]["corpus"][0]["source_tokens"].get<std::size_t>() > 16);
  require_all_checks_pass(doc);
  const std::string text = doc.dump();
  for (const char* phrase : {"quick brown fox", "mixture of experts", "std::puts", "sky is blue"})
    CHECK_MESSAGE(text.find(phrase) == std::string::npos, phrase);
  CHECK(doc["metrics"].contains("q2.mtp.acceptance_rate"));

  const auto inputs = load_json(results("result-placement-inputs-tokenizer.json"));
  CHECK(inputs["provenance"] == "Synthetic");
  CHECK(inputs["simulated"]["tokenizer_ids_folded"] == true);
  CHECK(inputs["metrics"].contains("q2.accepted_per_round"));
  require_all_checks_pass(inputs);
}
#endif

TEST_CASE("faults --supervised: forced terminations and relaunches under clusterlm-node-service are counted") {
  const auto doc = load_json(results("result-faults-supervised.json"));
  CHECK(doc["provenance"] == "Synthetic");
  const auto& m = doc["metrics"];
  CHECK(m["supervised_forced_termination.forced_terminations"] == 2);
  CHECK(m["supervised_forced_termination.worker_relaunches"] == 0);
  CHECK(m["supervised_forced_termination.residual_bytes_at_relaunch"] == 0);
  CHECK(m["supervised_forced_termination.activity_to_forced_termination_ms"]["n"] == 2);
  // The forced termination happens at the cooperative deadline, not before.
  CHECK(m["supervised_forced_termination.activity_to_forced_termination_ms"]["min"].get<double>() >=
        0.8 * m["supervised_forced_termination.cooperative_deadline_ms"].get<double>());
  CHECK(m["supervised_forced_termination.staged_bytes_before_fault"].get<double>() > 0);  // orphans existed
  CHECK(m["supervised_crash_recovery.forced_terminations"] == 0);
  CHECK(m["supervised_crash_recovery.worker_relaunches"] == 2);
  CHECK(m["supervised_crash_recovery.staged_bytes_before_fault"].get<double>() > 0);
  CHECK(m["supervised_crash_recovery.release_to_relaunch_ms"]["n"] == 2);
  require_all_checks_pass(doc);
}
#endif
