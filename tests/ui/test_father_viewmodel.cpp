// FatherViewModel against a scripted FatherClient: tier rendering, prepare progress, streaming accumulation,
// fallback and attribution, cancel, errors, diagnostics redaction and the "never claim Ready" rule.
#include <doctest/doctest.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>

#include "clusterlm/ui/clients.hpp"
#include "clusterlm/ui/father_viewmodel.hpp"

using namespace clusterlm;
using namespace clusterlm::ui;
using catalog::TierState;

namespace {

catalog::TierReadiness tier(const char* id, const char* model, TierState st, const char* headline) {
  catalog::TierReadiness t;
  t.tier_id = id;
  t.model_name = model;
  t.state = st;
  t.reasons.push_back(headline);
  return t;
}

struct Env {
  ScriptedFatherClient client{"Measured"};
  std::unique_ptr<FatherViewModel> vm;
  Env() {
    client.set_tiers({tier("fast", "Fast Model", TierState::kReady, "Fast is ready"),
                      tier("strong", "Strong Model", TierState::kAvailable, "Strong can be prepared"),
                      tier("ultra", "Ultra Model", TierState::kUnavailable, "The 3060 PC is offline")});
    client.set_participants("fast", {{"father", "This PC"}});
    client.set_participants("strong", {{"father", "This PC"}, {"node:laptop-class", "G14"}});
    REQUIRE(client.select_tier("fast").is_ok());
    vm = std::make_unique<FatherViewModel>(client);
    vm->tick(true);
  }
  FatherViewState st() const { return vm->snapshot(); }
  void emit(const father::Event& e) { client.emit(e); }
  static father::TokensEvent tokens(father::RequestId r, const char* tier, const char* model, const char* text, std::size_t n = 1) {
    father::TokensEvent t;
    t.request = r;
    t.tier_id = tier;
    t.model_name = model;
    t.text = text;
    t.tokens.assign(n, 1);
    return t;
  }
};

}  // namespace

TEST_CASE("tier list shows each state with the service's own headline") {
  Env e;
  auto s = e.st();
  REQUIRE(s.tiers.size() == 3);
  CHECK(s.tiers[0].name == "Fast");
  CHECK(s.tiers[0].state_label == "Ready");
  CHECK(s.tiers[0].selected);
  CHECK(s.tiers[0].participants == std::vector<std::string>{"This PC"});
  CHECK(s.tiers[1].state_label == "Available");
  CHECK(s.tiers[1].can_prepare);
  CHECK(s.tiers[1].participants == std::vector<std::string>{"This PC", "G14"});
  CHECK(s.tiers[2].state_label == "Unavailable");
  CHECK(s.tiers[2].headline == "The 3060 PC is offline");
  CHECK_FALSE(s.tiers[2].can_prepare);
  CHECK(s.active_model == "Fast Model");
  CHECK(s.can_send);
  CHECK(s.send_hint.empty());
}

TEST_CASE("Ready is never claimed from an event: only the service's tier list says Ready") {
  Env e;
  e.emit(father::PrepareProgressEvent{"strong", "Strong Model", 40.0, 90.0, "copying"});
  auto s = e.st();
  CHECK(s.tiers[1].state_label == "Preparing 40%");
  CHECK(s.tiers[1].eta == "about 2 min (estimate)");
  CHECK(s.prepare.active);
  CHECK(s.prepare.eta.find("estimate") != std::string::npos);
  CHECK_FALSE(s.tiers[1].can_prepare);

  // TierReadyEvent arrives, but the service still lists Strong as Available (a node went busy again).
  e.emit(father::TierReadyEvent{"strong", "Strong Model"});
  e.vm->tick(true);
  s = e.st();
  CHECK_FALSE(s.prepare.active);
  CHECK(s.tiers[1].state_label == "Available");  // never "Ready" without the service saying so

  e.client.patch_tier("strong", [](catalog::TierReadiness& t) { t.state = TierState::kReady; });
  e.vm->tick(true);
  CHECK(e.st().tiers[1].state_label == "Ready");
}

TEST_CASE("prepare progress shows per-machine bytes and parts, and keeps them between percentage-only events") {
  Env e;
  father::PrepareDetail d;
  d.phase = "provisioning";
  d.nodes = {{"G14", "provisioning", 1'200'000'000ull, 4'500'000'000ull, 3, 12},
             {"3060", "node-preparing", 2'000'000'000ull, 2'000'000'000ull, 5, 5}};
  father::PrepareProgressEvent ev{"strong", "Strong Model", 35.0, std::nullopt, "Preparing", d};
  e.emit(ev);
  auto s = e.st();
  REQUIRE(s.tiers[1].progress_lines.size() == 2);
  CHECK(s.tiers[1].progress_lines[0] == "G14: 1.2 GB of 4.5 GB (3 of 12 parts)");
  CHECK(s.tiers[1].progress_lines[1] == "3060: received everything, loading it");
  // The service's smooth percentage events carry no detail: the lines stay until new ones arrive.
  e.emit(father::PrepareProgressEvent{"strong", "Strong Model", 40.0, std::nullopt, "Preparing"});
  s = e.st();
  CHECK(s.tiers[1].progress_lines.size() == 2);
  CHECK(s.tiers[1].state_label == "Preparing 40%");
  d.nodes[1].phase = "node-ready";
  e.emit(father::PrepareProgressEvent{"strong", "Strong Model", 100.0, std::nullopt, "Preparing", d});
  CHECK(e.st().tiers[1].progress_lines[1] == "3060: ready");
  e.emit(father::TierReadyEvent{"strong", "Strong Model"});
  CHECK(e.st().tiers[1].progress_lines.empty());
}

TEST_CASE("prepare progress without a rate shows no ETA rather than a guess") {
  Env e;
  e.emit(father::PrepareProgressEvent{"strong", "Strong Model", std::nullopt, std::nullopt, "starting"});
  auto s = e.st();
  CHECK(s.prepare.active);
  CHECK_FALSE(s.prepare.percent.has_value());
  CHECK(s.prepare.eta.empty());
  CHECK(s.tiers[1].state_label == "Preparing");
}

TEST_CASE("prepare_tier calls the client and blocks other jobs while running") {
  Env e;
  REQUIRE(e.vm->prepare_tier("strong").is_ok());
  CHECK(e.st().prepare.active);
  CHECK(e.vm->prepare_tier("ultra").code() == ErrorCode::kFailedPrecondition);
  CHECK_FALSE(e.st().can_send);
  e.emit(father::ErrorEvent{0, ErrorCode::kFailedPrecondition, "G14 is busy"});
  auto s = e.st();
  CHECK_FALSE(s.prepare.active);
  CHECK(s.banner.kind == Banner::Kind::kError);
  CHECK(s.banner.text.find("G14 is busy") != std::string::npos);
}

TEST_CASE("prepare failure from the client is shown and leaves no stuck state") {
  Env e;
  e.client.fail_next("prepare_tier", make_error(ErrorCode::kUnavailable, "Father agent is not running"));
  CHECK_FALSE(e.vm->prepare_tier("strong").is_ok());
  auto s = e.st();
  CHECK_FALSE(s.prepare.active);
  CHECK(s.banner.kind == Banner::Kind::kError);
}

TEST_CASE("chat streams token text live and shows the user message first") {
  Env e;
  REQUIRE(e.vm->send("  hello there  ").is_ok());
  auto s = e.st();
  REQUIRE(s.entries.size() == 1);
  CHECK(s.entries[0].kind == EntryKind::kUser);
  CHECK(s.entries[0].text == "hello there");
  CHECK(s.chat_busy);
  CHECK_FALSE(s.can_send);
  REQUIRE(e.client.chats().size() == 1);
  CHECK(e.client.chats()[0].user_message == "hello there");

  e.emit(Env::tokens(1, "fast", "Fast Model", "Hel", 2));
  e.emit(Env::tokens(1, "fast", "Fast Model", "lo!", 1));
  s = e.st();
  REQUIRE(s.entries.size() == 2);
  CHECK(s.entries[1].kind == EntryKind::kAssistant);
  CHECK(s.entries[1].full_text() == "Hello!");
  CHECK(s.entries[1].streaming);
  CHECK(s.entries[1].attribution == "Answered by Fast (Fast Model)");

  father::GenerationStats stats;
  stats.tokens = 3;
  stats.rounds = 3;
  stats.ttft_ms = 420;
  stats.decode_tok_s = 12.34;
  stats.accepted_per_round = 1.0;
  stats.final_tier = "fast";
  stats.final_model = "Fast Model";
  stats.answered_by = {{"fast", "Fast Model", 3}};
  e.emit(father::FinishedEvent{1, father::FinishReason::kCompleted, stats});
  s = e.st();
  CHECK_FALSE(s.chat_busy);
  CHECK(s.can_send);
  CHECK_FALSE(s.entries[1].streaming);
  REQUIRE(s.last_answer.valid);
  CHECK(s.last_answer.tok_s == "12.3 tok/s");
  CHECK(s.last_answer.ttft == "420 ms");
  CHECK(s.last_answer.accepted_per_round == "1.00");
  CHECK(s.last_answer.provenance == "Measured");
}

TEST_CASE("empty and whitespace messages are rejected and nothing is sent") {
  Env e;
  CHECK(e.vm->send("   \n ").code() == ErrorCode::kInvalidArgument);
  CHECK(e.client.chats().empty());
}

TEST_CASE("sending is blocked for an unavailable tier with the reason in words") {
  Env e;
  REQUIRE(e.vm->select_tier("ultra").is_ok());
  e.vm->tick(true);
  auto s = e.st();
  CHECK_FALSE(s.can_send);
  CHECK(s.send_hint == "Ultra is not available: The 3060 PC is offline");
  CHECK(e.vm->send("hi").code() == ErrorCode::kFailedPrecondition);
  CHECK(e.client.chats().empty());
}

TEST_CASE("an available but unprepared tier may be used and says it will prepare first") {
  Env e;
  REQUIRE(e.vm->select_tier("strong").is_ok());
  e.vm->tick(true);
  auto s = e.st();
  CHECK(s.can_send);
  CHECK(s.send_hint.find("prepared first") != std::string::npos);
}

TEST_CASE("send failure removes the message again and shows the error") {
  Env e;
  e.client.fail_next("send_chat", make_error(ErrorCode::kUnavailable, "Father agent is not running"));
  CHECK_FALSE(e.vm->send("hi").is_ok());
  auto s = e.st();
  CHECK(s.entries.empty());
  CHECK_FALSE(s.chat_busy);
  CHECK(s.banner.kind == Banner::Kind::kError);
  CHECK(s.banner.text.find("not sent") != std::string::npos);
}

TEST_CASE("events that race ahead of send_chat returning do not leave the UI stuck busy") {
  Env e;
  e.client.on_chat = [](ScriptedFatherClient& c, father::RequestId id, const father::ChatRequest&) {
    // The service answers before send_chat() has even returned its request id.
    c.emit(Env::tokens(id, "fast", "Fast Model", "quick"));
    father::GenerationStats s;
    s.tokens = 1;
    s.final_tier = "fast";
    s.final_model = "Fast Model";
    c.emit(father::FinishedEvent{id, father::FinishReason::kCompleted, s});
  };
  REQUIRE(e.vm->send("hi").is_ok());
  auto s = e.st();
  CHECK_FALSE(s.chat_busy);
  CHECK(s.can_send);
  REQUIRE(s.entries.size() == 2);
  CHECK(s.entries[1].full_text() == "quick");
}

TEST_CASE("fallback is written into the chat in plain words and the answer continues under the new model") {
  Env e;
  REQUIRE(e.vm->select_tier("strong").is_ok());
  e.vm->tick(true);
  REQUIRE(e.vm->send("long question").is_ok());
  e.emit(Env::tokens(1, "strong", "Strong Model", "Part one. "));

  father::FallbackEvent fb;
  fb.request = 1;
  fb.kind = father::FallbackEvent::Kind::kDowngrade;
  fb.from_tier = "strong";
  fb.to_tier = "fast";
  fb.from_model = "Strong Model";
  fb.to_model = "Fast Model";
  fb.message = "G14 became busy \xE2\x80\x94 switching from Strong (Strong Model) to Fast (Fast Model)";
  e.emit(fb);
  e.emit(Env::tokens(1, "fast", "Fast Model", "Part two."));

  auto s = e.st();
  REQUIRE(s.entries.size() == 4);  // user, assistant(Strong), notice, assistant(Fast)
  CHECK(s.entries[2].kind == EntryKind::kNotice);
  CHECK(s.entries[2].text ==
        "G14 became busy - switching from Strong (Strong Model) to Fast (Fast Model) Your conversation is kept.");
  CHECK(s.entries[1].attribution == "Answered by Strong (Strong Model)");
  CHECK(s.entries[1].full_text() == "Part one. ");
  CHECK(s.entries[3].continued);
  CHECK(s.entries[3].attribution == "Continued by Fast (Fast Model)");
  CHECK(s.entries[3].full_text() == "Part two.");
  CHECK(s.active_model == "Fast Model");

  father::GenerationStats stats;
  stats.fallbacks = 1;
  stats.final_tier = "fast";
  stats.final_model = "Fast Model";
  stats.answered_by = {{"strong", "Strong Model", 3}, {"fast", "Fast Model", 4}};
  e.emit(father::FinishedEvent{1, father::FinishReason::kCompleted, stats});
  s = e.st();
  CHECK(s.entries[3].attribution.find("whole answer: Strong (Strong Model), then Fast (Fast Model)") != std::string::npos);
  CHECK(s.last_answer.fallbacks == 1);
}

TEST_CASE("a fallback message that already says the conversation is kept is not extended") {
  Env e;
  father::FallbackEvent fb;
  fb.request = 1;
  fb.message = "Retrying Strong; your conversation is kept.";
  e.emit(fb);
  auto s = e.st();
  REQUIRE(s.entries.size() == 1);
  CHECK(s.entries[0].text == "Retrying Strong; your conversation is kept.");
}

TEST_CASE("a model change inside one entry (without a fallback event) is attributed to both models") {
  Env e;
  REQUIRE(e.vm->send("hi").is_ok());
  e.emit(Env::tokens(1, "strong", "A", "one "));
  e.emit(Env::tokens(1, "fast", "B", "two"));
  auto s = e.st();
  REQUIRE(s.entries.size() == 2);
  CHECK(s.entries[1].spans.size() == 2);
  CHECK(s.entries[1].attribution == "Answered by Strong (A), then Fast (B)");
}

TEST_CASE("cancel asks the client, keeps the partial answer and says so") {
  Env e;
  REQUIRE(e.vm->send("hi").is_ok());
  e.emit(Env::tokens(1, "fast", "Fast Model", "partial"));
  REQUIRE(e.vm->cancel().is_ok());
  auto calls = e.client.calls();
  CHECK(calls.back() == "cancel:1");
  father::GenerationStats stats;
  stats.final_tier = "fast";
  stats.final_model = "Fast Model";
  e.emit(father::FinishedEvent{1, father::FinishReason::kCancelled, stats});
  auto s = e.st();
  CHECK_FALSE(s.chat_busy);
  CHECK(s.entries[1].cancelled);
  CHECK(s.entries[1].full_text() == "partial");
  CHECK(s.entries.back().kind == EntryKind::kNotice);
  CHECK(s.entries.back().text.find("stopped") != std::string::npos);
  CHECK(e.vm->cancel().code() == ErrorCode::kFailedPrecondition);  // nothing left to cancel
}

TEST_CASE("an error event is shown inline and in the banner, with the code, and frees the UI") {
  Env e;
  REQUIRE(e.vm->send("hi").is_ok());
  e.emit(father::ErrorEvent{1, ErrorCode::kUnavailable, "The G14 PC stopped answering"});
  auto s = e.st();
  CHECK_FALSE(s.chat_busy);
  REQUIRE(s.entries.size() == 2);
  CHECK(s.entries[1].kind == EntryKind::kError);
  CHECK(s.entries[1].text == "The G14 PC stopped answering (UNAVAILABLE)");
  CHECK(s.banner.kind == Banner::Kind::kError);
  e.vm->dismiss_banner();
  CHECK(e.st().banner.kind == Banner::Kind::kNone);
}

TEST_CASE("a failed finish without an error event still tells the user") {
  Env e;
  REQUIRE(e.vm->send("hi").is_ok());
  e.emit(father::FinishedEvent{1, father::FinishReason::kFailed, {}});
  auto s = e.st();
  CHECK(s.entries.back().kind == EntryKind::kError);
  CHECK(s.entries.back().text.find("could not be completed") != std::string::npos);
}

TEST_CASE("only Tokens events put text in the transcript: progress, ready and finished never do") {
  Env e;
  REQUIRE(e.vm->send("hi").is_ok());
  e.emit(father::PrepareProgressEvent{"fast", "Fast Model", 10.0, std::nullopt, "SPECULATIVE-DRAFT-TEXT"});
  e.emit(father::TierReadyEvent{"fast", "Fast Model"});
  e.emit(father::TierSelectedEvent{"fast", "Fast Model"});
  auto s = e.st();
  REQUIRE(s.entries.size() == 1);  // just the user message
  for (const auto& en : s.entries) CHECK(en.text.find("SPECULATIVE") == std::string::npos);
}

TEST_CASE("new conversation clears the transcript; it is refused (and explained) while a job runs") {
  Env e;
  REQUIRE(e.vm->send("hi").is_ok());
  e.client.fail_next("reset_conversation", make_error(ErrorCode::kFailedPrecondition, "A job is running"));
  CHECK_FALSE(e.vm->new_conversation().is_ok());
  CHECK(e.st().entries.size() == 1);
  CHECK(e.st().banner.kind == Banner::Kind::kError);

  e.emit(father::FinishedEvent{1, father::FinishReason::kCompleted, {}});
  REQUIRE(e.vm->new_conversation().is_ok());
  auto s = e.st();
  CHECK(s.entries.empty());
  CHECK_FALSE(s.last_answer.valid);
}

TEST_CASE("context use shows exact numbers from the client, or an honest lower bound") {
  Env e;
  e.client.set_context_use(ContextUse{812, true});
  e.vm->tick(true);
  CHECK(e.st().context.label == "812 / 4096 tokens");

  e.client.set_context_use(std::nullopt);  // client cannot tell
  REQUIRE(e.vm->send("hi").is_ok());
  e.emit(Env::tokens(1, "fast", "Fast Model", "abc", 3));
  e.emit(father::FinishedEvent{1, father::FinishReason::kCompleted, {}});
  e.vm->tick(true);
  auto s = e.st();
  CHECK_FALSE(s.context.exact);
  CHECK(s.context.label == "at least 3 / 4096 tokens");
}

TEST_CASE("settings are validated and change the context size used for requests") {
  Env e;
  FatherSettings bad;
  bad.context_tokens = 100;
  CHECK_FALSE(e.vm->apply_settings(bad).is_ok());
  CHECK(e.st().banner.kind == Banner::Kind::kError);

  FatherSettings good;
  good.context_tokens = 8192;
  good.max_new_tokens = 64;
  good.system_prompt = "be brief";
  REQUIRE(e.vm->apply_settings(good).is_ok());
  CHECK(e.st().context.context_tokens == 8192);
  REQUIRE(e.vm->send("hi").is_ok());
  auto req = e.client.chats().back();
  CHECK(req.context_tokens == 8192);
  CHECK(req.max_new_tokens == 64);
  CHECK(req.system_prompt == std::optional<std::string>("be brief"));
}

TEST_CASE("an unreachable Father side is reported once, in words, and blocks sending") {
  ScriptedFatherClient c;
  c.fail_next("list_tiers", make_error(ErrorCode::kUnavailable, "The Father agent is not running"));
  FatherViewModel vm(c);
  vm.tick(true);
  auto s = vm.snapshot();
  CHECK(s.tiers.empty());
  CHECK(s.connection_message == "The Father agent is not running");
  CHECK_FALSE(s.can_send);
  CHECK(s.send_hint == "The Father agent is not running");
  vm.tick(true);  // recovers when the client does
  CHECK(vm.snapshot().connection_message.empty());
}

TEST_CASE("pairing: unimplemented is explained, unpair removes the machine") {
  Env e;
  e.client.set_paired({{"G14", "Node", "ab12"}, {"3060", "Node", "cd34"}});
  FatherViewModel vm(e.client);
  CHECK(vm.snapshot().pairing.machines.size() == 2);
  REQUIRE(vm.unpair("G14").is_ok());
  CHECK(vm.snapshot().pairing.machines.size() == 1);
  e.client.fail_next("start_pairing", make_error(ErrorCode::kUnimplemented, "Pairing is done in the Father agent in this version"));
  CHECK(vm.start_pairing({"10.0.0.2:7000", "ABCD-1234", ""}).code() == ErrorCode::kUnimplemented);
  CHECK(e.client.calls().back() == "start_pairing:10.0.0.2:7000");
  CHECK(vm.snapshot().pairing.message == "Pairing is done in the Father agent in this version");
}

TEST_CASE("diagnostics: provenance labels, honest 'not reported', and the redaction flag defaults off") {
  Env e;
  DiagnosticsExport d;
  d.text = "requests_started=1";
  e.client.set_diagnostics(d);  // no plan, no stage timings
  REQUIRE(e.vm->send("hi").is_ok());
  father::GenerationStats stats;
  stats.tokens = 4;
  stats.rounds = 2;
  stats.ttft_ms = 1500;
  stats.decode_tok_s = 150;
  stats.accepted_per_round = 2.0;
  e.emit(father::FinishedEvent{1, father::FinishReason::kCompleted, stats});
  e.vm->set_diagnostics_open(true);
  auto s = e.st();
  CHECK(s.diagnostics.open);
  CHECK_FALSE(s.diagnostics.include_conversation);
  CHECK(e.client.last_export_include_text() == std::optional<bool>(false));  // the preview is always redacted
  CHECK(s.diagnostics.plan_summary.empty());
  CHECK(s.diagnostics.stage_timings.empty());
  REQUIRE_FALSE(s.diagnostics.rows.empty());
  CHECK(s.diagnostics.rows[0].label == "Time to first token");
  CHECK(s.diagnostics.rows[0].value == "1.5 s");
  CHECK(s.diagnostics.rows[1].value == "150 tok/s");
  for (const auto& r : s.diagnostics.rows) {
    CHECK(r.provenance == "Measured");
    CHECK(r.provenance != "Qualified");
  }
}

TEST_CASE("diagnostics from a synthetic client are labelled Synthetic") {
  ScriptedFatherClient c("Synthetic");
  c.set_tiers({tier("fast", "F", TierState::kReady, "ok")});
  REQUIRE(c.select_tier("fast").is_ok());
  FatherViewModel vm(c);
  vm.tick(true);
  REQUIRE(vm.send("hi").is_ok());
  father::GenerationStats stats;
  stats.decode_tok_s = 10;
  c.emit(father::FinishedEvent{1, father::FinishReason::kCompleted, stats});
  vm.set_diagnostics_open(true);
  for (const auto& r : vm.snapshot().diagnostics.rows) CHECK(r.provenance == "Synthetic");
}

TEST_CASE("export honours the include-conversation flag and writes the file; the flag never leaks into the preview") {
  namespace fs = std::filesystem;
  const fs::path dir = fs::temp_directory_path() / "clm-ui-export";
  { std::error_code ec_rm; fs::remove_all(dir, ec_rm); }

  Env e;
  e.client.set_diagnostics(DiagnosticsExport{"snapshot-body", false, "plan: 4 stages", {{"G14 4-9", 5.0}}});
  e.vm->set_exporter(make_file_exporter(dir.string()));

  REQUIRE(e.vm->export_diagnostics().is_ok());
  CHECK(e.client.last_export_include_text() == std::optional<bool>(false));
  auto s = e.st();
  CHECK(s.diagnostics.export_message.find("no prompts or answers") != std::string::npos);
  REQUIRE_FALSE(s.diagnostics.last_export_path.empty());
  {
    std::ifstream f(s.diagnostics.last_export_path);
    std::stringstream ss;
    ss << f.rdbuf();
    CHECK(ss.str() == "snapshot-body");
  }

  e.vm->set_include_conversation(true);
  REQUIRE(e.vm->export_diagnostics().is_ok());
  CHECK(e.client.last_export_include_text() == std::optional<bool>(true));
  s = e.st();
  CHECK(s.diagnostics.export_message.find("includes your conversation") != std::string::npos);
  {
    std::ifstream f(s.diagnostics.last_export_path);
    std::stringstream ss;
    ss << f.rdbuf();
    CHECK(ss.str().rfind("WARNING", 0) == 0);
  }
  e.vm->refresh_diagnostics();
  CHECK(e.client.last_export_include_text() == std::optional<bool>(false));  // preview stays redacted
  { std::error_code ec_rm; fs::remove_all(dir, ec_rm); }
}

TEST_CASE("export without a configured destination fails visibly") {
  Env e;
  CHECK(e.vm->export_diagnostics().code() == ErrorCode::kFailedPrecondition);
  CHECK_FALSE(e.st().diagnostics.export_message.empty());
}

TEST_CASE("release reports success and clears the prepare state") {
  Env e;
  e.emit(father::PrepareProgressEvent{"strong", "Strong Model", 10.0, std::nullopt, ""});
  REQUIRE(e.vm->release().is_ok());
  auto s = e.st();
  CHECK_FALSE(s.prepare.active);
  CHECK(s.banner.kind == Banner::Kind::kInfo);
}

TEST_CASE("the view-model detaches from the client on destruction") {
  ScriptedFatherClient c;
  c.set_tiers({tier("fast", "F", TierState::kReady, "ok")});
  {
    FatherViewModel vm(c);
  }
  c.emit(father::ReleasedEvent{"late"});  // must not touch the destroyed view-model
  CHECK(true);
}

TEST_CASE("typographic punctuation from the service is folded to ASCII for display") {
  Env e;
  father::ErrorEvent err{0, ErrorCode::kInternal, "It\xE2\x80\x99" "s \xE2\x80\x9C" "broken\xE2\x80\x9D\xE2\x80\xA6"};
  e.emit(err);
  CHECK(e.st().entries.back().text == "It's \"broken\"... (INTERNAL)");
}

TEST_CASE("the demo client drives a full prepare and chat through the view-model") {
  auto demo = make_demo_father_client();
  FatherViewModel vm(*demo, true);
  vm.tick(true);
  REQUIRE(vm.snapshot().demo);
  REQUIRE(vm.select_tier("fast").is_ok());
  REQUIRE(vm.prepare_tier("fast").is_ok());
  for (int i = 0; i < 200 && vm.snapshot().prepare.active; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
    vm.tick();
  }
  vm.tick(true);
  REQUIRE(vm.snapshot().tiers[0].state_label == "Ready");
  REQUIRE(vm.send("hello").is_ok());
  for (int i = 0; i < 300 && vm.snapshot().chat_busy; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(25));
  auto s = vm.snapshot();
  CHECK_FALSE(s.chat_busy);
  REQUIRE(s.entries.size() == 2);
  CHECK(s.entries[1].full_text().find("scripted demo answer") != std::string::npos);
  CHECK(s.last_answer.provenance == "Synthetic");
}
