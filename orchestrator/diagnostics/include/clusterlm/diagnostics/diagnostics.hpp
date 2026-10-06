#pragma once
// Diagnostics export: one JSON bundle a user can attach to a bug report without leaking a conversation.
//
// What goes in: build/version info, recent structured log lines (bounded ring buffer fed by log::set_sink),
// node statuses, the plan description, metrics and the PATHS of bench result files (never their contents).
//
// Redaction contract (enforced here, tested in tests/privacy):
//   * Conversation text (roles and message content) is NEVER included unless BundleOptions::include_conversation
//     is set, and then it appears only under a section explicitly marked "user_content".
//   * Every log line passes through redact_line(): values of content-bearing keys are replaced, lists of integers
//     (token arrays), runs of floating-point numbers (activation dumps), long hex/base64 runs and over-long values
//     are replaced by "[redacted:<kind>]" markers. Redaction is applied to node status text, plan text and metric
//     names as well, because those strings come from callers.
//   * The bundle states what it redacted ("redaction" section) so a reader knows the absence of text is by design.
// Redaction is a safety net, not the privacy mechanism: the primary guarantee is that the logger has no API that
// accepts prompts, tokens or tensors (see runtime/common/log.hpp).
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "clusterlm/common/status.hpp"

namespace clusterlm::diagnostics {

// Bounded ring of the most recent log lines. install() hooks log::set_sink; lines are stored UNREDACTED in memory
// only so that the redaction rules can improve without losing information, and are redacted when a bundle is built.
class LogRing {
 public:
  explicit LogRing(std::size_t capacity = 512) : capacity_(capacity == 0 ? 1 : capacity) {}
  ~LogRing();
  LogRing(const LogRing&) = delete;
  LogRing& operator=(const LogRing&) = delete;

  void install();    // route log::write output into this ring (replaces any previous sink)
  void uninstall();  // detach (a no-op if another ring/sink replaced this one)
  void push(std::string_view line);
  std::vector<std::string> snapshot() const;
  std::uint64_t dropped() const;  // lines evicted by the capacity bound

 private:
  const std::size_t capacity_;
  mutable std::mutex mu_;
  std::deque<std::string> lines_;
  std::uint64_t dropped_ = 0;
  bool installed_ = false;
};

struct NodeStatusEntry {
  std::string name;
  std::string state;                 // "Ready", "Busy", ...
  std::uint64_t lease_generation = 0;
  std::vector<std::pair<std::string, std::uint64_t>> counters;  // e.g. windows_executed, stale_rejections
};

struct MetricEntry {
  std::string name;
  double value = 0;
  std::string unit;
  std::string provenance;  // "synthetic" | "measured" | "qualified" | "" (not a performance claim)
};

struct ConversationEntry {
  std::string role;
  std::string content;
};

struct BundleInputs {
  std::string component = "clusterlm";
  std::vector<std::string> log_lines;              // typically LogRing::snapshot()
  std::uint64_t log_lines_dropped = 0;
  std::vector<NodeStatusEntry> nodes;
  std::string plan_description;                    // ClusterPlan::describe()
  std::vector<MetricEntry> metrics;
  std::vector<std::string> bench_result_paths;     // paths only
  std::vector<ConversationEntry> conversation;     // honoured only with include_conversation
  std::string last_error;                          // status text
};

struct BundleOptions {
  bool include_conversation = false;  // explicit opt-in; the section is marked "user_content"
  std::size_t max_log_lines = 512;
  std::size_t max_value_chars = 256;  // values longer than this are replaced
};

struct BuildInfo {
  std::string version;
  std::string build_type;
  std::string compiler;
  std::string platform;
  std::string cxx_standard;
};
BuildInfo build_info();

// Redacts one structured log line (the `key=value key=value` format of runtime/common/log). Exposed for tests.
std::string redact_line(std::string_view line, const BundleOptions& options = {});
// Redacts free text from callers (status text, plan description): lists/floats/blobs/long runs only.
std::string redact_text(std::string_view text, const BundleOptions& options = {});

// Builds the bundle as pretty-printed JSON ("clusterlm.diagnostics.v1").
std::string build_bundle(const BundleInputs& inputs, const BundleOptions& options = {});
// Atomically writes `build_bundle` to `path`.
Status write_bundle(const std::string& path, const BundleInputs& inputs, const BundleOptions& options = {});

}  // namespace clusterlm::diagnostics
