#include "clusterlm/diagnostics/diagnostics.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <ctime>

#include <nlohmann/json.hpp>

#include "clusterlm/common/log.hpp"
#include "clusterlm/platform/durable_file.hpp"

#ifndef CLUSTERLM_VERSION_STRING
#define CLUSTERLM_VERSION_STRING "unknown"
#endif
#ifndef CLUSTERLM_BUILD_TYPE_STRING
#define CLUSTERLM_BUILD_TYPE_STRING "unknown"
#endif

namespace clusterlm::diagnostics {

using nlohmann::json;

// ---------------------------------------------------------------------------------------------- LogRing

LogRing::~LogRing() { uninstall(); }

void LogRing::install() {
  {
    std::lock_guard lock(mu_);
    installed_ = true;
  }
  // The sink runs under the logger's lock; push() only takes this ring's own lock, so the order is fixed.
  log::set_sink([this](std::string_view line) { push(line); });
}

void LogRing::uninstall() {
  bool was_installed;
  {
    std::lock_guard lock(mu_);
    was_installed = installed_;
    installed_ = false;
  }
  if (was_installed) log::set_sink({});
}

void LogRing::push(std::string_view line) {
  std::lock_guard lock(mu_);
  lines_.emplace_back(line);
  while (lines_.size() > capacity_) {
    lines_.pop_front();
    ++dropped_;
  }
}

std::vector<std::string> LogRing::snapshot() const {
  std::lock_guard lock(mu_);
  return std::vector<std::string>(lines_.begin(), lines_.end());
}

std::uint64_t LogRing::dropped() const {
  std::lock_guard lock(mu_);
  return dropped_;
}

// ---------------------------------------------------------------------------------------------- redaction

namespace {

bool is_digit(char c) { return c >= '0' && c <= '9'; }
bool is_hex(char c) { return is_digit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'); }
// Base64 payload characters. '=' (padding / key=value separator) deliberately does not extend a run.
bool is_b64(char c) { return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '+' || c == '/'; }
bool is_ident_start(char c) { return std::isalpha(static_cast<unsigned char>(c)) != 0 || c == '_'; }
bool is_ident(char c) { return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_' || c == '.'; }

// Keys whose values may carry conversation content. A key matching one of these never keeps its value.
bool is_content_key(std::string_view key) {
  static constexpr std::string_view kKeys[] = {"prompt",   "text",     "response", "content",    "message",
                                               "messages", "token",    "tokens",   "logits",     "activations",
                                               "system",   "user",     "assistant", "role",      "completion",
                                               "answer",   "input",    "output",   "candidates", "chat",
                                               "history",  "stream",   "delta",    "body",       "payload",
                                               "data"};
  std::string lower(key);
  std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return std::find(std::begin(kKeys), std::end(kKeys), lower) != std::end(kKeys);
}

// Parses one number at `i` (optionally signed, optional fraction/exponent). Returns the end, or `i` if none.
std::size_t scan_number(std::string_view s, std::size_t i, bool& is_float) {
  std::size_t j = i;
  if (j < s.size() && (s[j] == '-' || s[j] == '+')) ++j;
  if (j >= s.size() || !is_digit(s[j])) return i;
  while (j < s.size() && is_digit(s[j])) ++j;
  is_float = false;
  if (j + 1 < s.size() && s[j] == '.' && is_digit(s[j + 1])) {
    is_float = true;
    ++j;
    while (j < s.size() && is_digit(s[j])) ++j;
  }
  if (j < s.size() && (s[j] == 'e' || s[j] == 'E')) {
    std::size_t k = j + 1;
    if (k < s.size() && (s[k] == '-' || s[k] == '+')) ++k;
    if (k < s.size() && is_digit(s[k])) {
      is_float = true;
      while (k < s.size() && is_digit(s[k])) ++k;
      j = k;
    }
  }
  return j;
}

// Skips a list separator (comma/semicolon with optional spaces, or plain spaces). Returns `i` if none.
std::size_t scan_separator(std::string_view s, std::size_t i) {
  std::size_t j = i;
  while (j < s.size() && s[j] == ' ') ++j;
  if (j < s.size() && (s[j] == ',' || s[j] == ';')) {
    ++j;
    while (j < s.size() && s[j] == ' ') ++j;
  }
  return j;
}

std::string redact_patterns(std::string_view s, std::size_t max_value_chars) {
  std::string out;
  out.reserve(s.size());
  std::size_t i = 0;
  while (i < s.size()) {
    const char c = s[i];
    const bool at_boundary = i == 0 || !is_ident(s[i - 1]);  // numbers glued to identifiers (e.g. "node2") are names
    // A run of numbers separated by commas/spaces: token arrays (ints) or activation dumps (floats).
    if (at_boundary && (is_digit(c) || ((c == '-' || c == '+') && i + 1 < s.size() && is_digit(s[i + 1])))) {
      std::size_t pos = i, items = 0, floats = 0;
      for (;;) {
        bool f = false;
        const std::size_t e = scan_number(s, pos, f);
        if (e == pos) break;
        ++items;
        if (f) ++floats;
        pos = e;
        const std::size_t sep = scan_separator(s, pos);
        if (sep == pos) break;
        // The separator only continues the run if another number follows it.
        bool f2 = false;
        if (scan_number(s, sep, f2) == sep) break;
        pos = sep;
      }
      if (floats >= 3) {
        out += "[redacted:float-dump]";
        i = pos;
        continue;
      }
      if (items >= 4) {
        out += "[redacted:int-list]";
        i = pos;
        continue;
      }
    }
    // Long hex / base64 runs (not the 64-hex device ids and digests).
    if (is_b64(c) && (i == 0 || !is_b64(s[i - 1]))) {
      std::size_t j = i;
      bool hex_only = true;
      while (j < s.size() && is_b64(s[j])) {
        hex_only = hex_only && is_hex(s[j]);
        ++j;
      }
      const std::size_t run = j - i;
      if ((hex_only && run >= 96) || run >= 128) {
        out += hex_only ? "[redacted:hex-blob]" : "[redacted:blob]";
        i = j;
        continue;
      }
    }
    out.push_back(c);
    ++i;
  }
  // Over-long whitespace-delimited values.
  std::string capped;
  capped.reserve(out.size());
  std::size_t k = 0;
  while (k < out.size()) {
    if (out[k] == ' ') {
      capped.push_back(' ');
      ++k;
      continue;
    }
    std::size_t e = k;
    while (e < out.size() && out[e] != ' ') ++e;
    if (e - k > max_value_chars) capped += "[redacted:long-value]";
    else capped.append(out, k, e - k);
    k = e;
  }
  // Control characters (other than the tab) have no business in a diagnostic line.
  for (char& ch : capped)
    if (static_cast<unsigned char>(ch) < 0x20 && ch != '\t') ch = '?';
  return capped;
}

// Replaces the value of every content-bearing `key=value` pair. A value runs until the next " ident=" boundary.
std::string redact_content_keys(std::string_view line) {
  std::string out;
  std::size_t i = 0;
  while (i < line.size()) {
    const bool token_start = i == 0 || line[i - 1] == ' ';
    if (token_start && is_ident_start(line[i])) {
      std::size_t j = i;
      while (j < line.size() && is_ident(line[j])) ++j;
      if (j < line.size() && line[j] == '=' && is_content_key(line.substr(i, j - i))) {
        out.append(line.substr(i, j + 1 - i));
        out += "[redacted:content]";
        // Skip the value: up to the next " ident=" (or the end of the line).
        std::size_t v = j + 1;
        while (v < line.size()) {
          if (line[v] == ' ' && v + 1 < line.size() && is_ident_start(line[v + 1])) {
            std::size_t e = v + 1;
            while (e < line.size() && is_ident(line[e])) ++e;
            if (e < line.size() && line[e] == '=') break;
          }
          ++v;
        }
        i = v;
        continue;
      }
    }
    out.push_back(line[i]);
    ++i;
  }
  return out;
}

std::string iso_utc_now() {
  const std::time_t t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::tm tm{};
#ifdef _WIN32
  gmtime_s(&tm, &t);
#else
  gmtime_r(&t, &tm);
#endif
  char buf[32];
  std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", &tm);
  return buf;
}

std::string dump(const json& j) { return j.dump(2, ' ', false, json::error_handler_t::replace); }

}  // namespace

std::string redact_text(std::string_view text, const BundleOptions& options) {
  return redact_patterns(text, options.max_value_chars);
}

std::string redact_line(std::string_view line, const BundleOptions& options) {
  return redact_patterns(redact_content_keys(line), options.max_value_chars);
}

BuildInfo build_info() {
  BuildInfo b;
  b.version = CLUSTERLM_VERSION_STRING;
  b.build_type = CLUSTERLM_BUILD_TYPE_STRING;
#if defined(__clang__)
  b.compiler = std::string("clang ") + __clang_version__;
#elif defined(__GNUC__)
  b.compiler = std::string("gcc ") + __VERSION__;
#elif defined(_MSC_VER)
  b.compiler = "msvc " + std::to_string(_MSC_VER);
#else
  b.compiler = "unknown";
#endif
#if defined(_WIN32)
  b.platform = "windows";
#elif defined(__linux__)
  b.platform = "linux";
#elif defined(__APPLE__)
  b.platform = "macos";
#else
  b.platform = "unknown";
#endif
  b.cxx_standard = std::to_string(__cplusplus);
  return b;
}

std::string build_bundle(const BundleInputs& in, const BundleOptions& options) {
  json j;
  j["schema"] = "clusterlm.diagnostics.v1";
  j["generated_at_utc"] = iso_utc_now();
  j["component"] = redact_text(in.component, options);
  const BuildInfo b = build_info();
  j["build"] = {{"version", b.version},
                {"build_type", b.build_type},
                {"compiler", b.compiler},
                {"platform", b.platform},
                {"cxx_standard", b.cxx_standard}};

  json logs = json::array();
  const std::size_t first = in.log_lines.size() > options.max_log_lines ? in.log_lines.size() - options.max_log_lines : 0;
  for (std::size_t i = first; i < in.log_lines.size(); ++i) logs.push_back(redact_line(in.log_lines[i], options));
  j["logs"] = std::move(logs);
  j["logs_dropped"] = in.log_lines_dropped + first;

  json nodes = json::array();
  for (const auto& n : in.nodes) {
    json counters = json::object();
    for (const auto& [k, v] : n.counters) counters[redact_text(k, options)] = v;
    nodes.push_back({{"name", redact_text(n.name, options)},
                     {"state", redact_text(n.state, options)},
                     {"lease_generation", n.lease_generation},
                     {"counters", std::move(counters)}});
  }
  j["nodes"] = std::move(nodes);
  j["plan"] = redact_text(in.plan_description, options);

  json metrics = json::array();
  for (const auto& m : in.metrics)
    metrics.push_back({{"name", redact_text(m.name, options)},
                       {"value", m.value},
                       {"unit", redact_text(m.unit, options)},
                       {"provenance", redact_text(m.provenance, options)}});
  j["metrics"] = std::move(metrics);

  json paths = json::array();
  for (const auto& p : in.bench_result_paths) paths.push_back(redact_text(p, options));
  j["bench_result_paths"] = std::move(paths);
  j["last_error"] = redact_line(in.last_error, options);

  j["redaction"] = {{"conversation_included", options.include_conversation},
                    {"applied", json::array({"content-keys", "int-lists", "float-dumps", "hex-blobs", "long-values"})},
                    {"note", "Prompts, responses, token IDs and activations are never written to logs or this bundle; "
                             "the rules above are a second line of defence."}};
  if (options.include_conversation) {
    json conv = json::array();
    for (const auto& c : in.conversation) conv.push_back({{"role", c.role}, {"content", c.content}});
    j["user_content"] = {{"warning", "OPT-IN: this section contains the user's conversation verbatim"},
                         {"conversation", std::move(conv)}};
  }
  return dump(j);
}

Status write_bundle(const std::string& path, const BundleInputs& inputs, const BundleOptions& options) {
  const std::string text = build_bundle(inputs, options) + "\n";
  return platform::write_file_atomic(path, ByteSpan(reinterpret_cast<const std::uint8_t*>(text.data()), text.size()));
}

}  // namespace clusterlm::diagnostics
