// Log privacy: capture every log line (all levels, including debug) produced by a full cluster + Father-service
// chat run and prove none contains prompt/response text, token arrays or activation dumps.
#include <doctest/doctest.h>

#include <mutex>
#include <string>
#include <vector>

#include "chat_fixture.hpp"
#include "clusterlm/common/log.hpp"

using namespace clusterlm;
using namespace clusterlm::testing;

namespace {

// Installs a capturing sink and the most verbose level; restores both on scope exit.
class LogCapture {
 public:
  LogCapture() : previous_level_(log::level()) {
    log::set_level(log::Level::kDebug);
    log::set_sink([this](std::string_view line) {
      std::lock_guard lock(mu_);
      lines_.emplace_back(line);
    });
  }
  ~LogCapture() {
    log::set_sink({});
    log::set_level(previous_level_);
  }
  std::vector<std::string> lines() const {
    std::lock_guard lock(mu_);
    return lines_;
  }

 private:
  log::Level previous_level_;
  mutable std::mutex mu_;
  std::vector<std::string> lines_;
};

// Decimal renderings of k consecutive tokens with the separators a careless logger would use.
std::vector<std::string> decimal_windows(const std::vector<std::int32_t>& seq, std::size_t k) {
  std::vector<std::string> out;
  for (std::size_t i = 0; i + k <= seq.size(); ++i)
    for (const char* sep : {",", ", ", " ", ";", "|"}) {
      std::string s;
      for (std::size_t j = 0; j < k; ++j) s += (j ? std::string(sep) : std::string()) + std::to_string(seq[i + j]);
      out.push_back(std::move(s));
    }
  return out;
}

// Number of decimal floating-point literals with >= 4 fraction digits in the line (an activation dump signature).
std::size_t long_float_literals(const std::string& line) {
  std::size_t n = 0;
  for (std::size_t i = 0; i < line.size(); ++i) {
    if (line[i] != '.' || i == 0 || !std::isdigit(static_cast<unsigned char>(line[i - 1]))) continue;
    std::size_t d = 0;
    while (i + 1 + d < line.size() && std::isdigit(static_cast<unsigned char>(line[i + 1 + d]))) ++d;
    if (d >= 4) ++n;
  }
  return n;
}

}  // namespace

TEST_CASE("logs of a full cluster chat contain no prompt, response, token array or activation dump") {
  LogCapture capture;
  const std::string user = "Distinctive-Payload-Zebra-42 tell me something";
  const std::string system = "You are PRIVATE-SYSTEM-PROMPT-QUOKKA";
  std::vector<std::int32_t> generated, prompt_tokens;
  std::string response;
  {
    ChatStack stack;
    stack.run_chat(user, system, 24);
    prompt_tokens = stack.tokenizer->encode_chat(
        std::vector<father::ChatMessage>{{father::ChatRole::kSystem, system}, {father::ChatRole::kUser, user}});
    generated = stack.events->all_tokens();
    response = stack.events->all_text();
    REQUIRE(stack.svc->release().is_ok());
  }
  REQUIRE(generated.size() == 24);

  const auto lines = capture.lines();
  // The capture is meaningful: the lifecycle of Father, both Nodes and the service was logged.
  auto any = [&](std::string_view needle) {
    return std::any_of(lines.begin(), lines.end(), [&](const std::string& l) { return l.find(needle) != std::string::npos; });
  };
  CHECK(lines.size() > 20);
  CHECK(any("event=plan_accepted"));
  CHECK(any("event=plan_ready"));
  CHECK(any("event=father.request_finished"));
  CHECK(any("event=lease_released"));
  CHECK(any("event=node_state"));

  const auto prompt_windows = decimal_windows(prompt_tokens, 4);
  const auto generated_windows = decimal_windows(generated, 4);
  for (const std::string& line : lines) {
    CAPTURE(line);
    CHECK(line.size() < 1024);
    CHECK(line.find(user) == std::string::npos);
    CHECK(line.find(system) == std::string::npos);
    CHECK(line.find("Zebra") == std::string::npos);
    CHECK(line.find("QUOKKA") == std::string::npos);
    CHECK(line.find("assistant: ") == std::string::npos);
    if (response.size() >= 8) CHECK(line.find(response.substr(0, 8)) == std::string::npos);
    for (const auto& w : prompt_windows) CHECK(line.find(w) == std::string::npos);
    for (const auto& w : generated_windows) CHECK(line.find(w) == std::string::npos);
    CHECK(long_float_literals(line) < 3);
  }
}

TEST_CASE("the logger API has no way to log a tensor, token array or message object") {
  // Field values are strings built by the caller; the only structured inputs are an event name and key/value
  // strings. This is a compile-time statement about the API surface the privacy contract relies on.
  static_assert(std::is_same_v<log::Field, std::pair<std::string_view, std::string>>);
  static_assert(std::is_invocable_v<decltype(&log::write), log::Level, std::string_view, std::initializer_list<log::Field>>);
  CHECK(true);
}
