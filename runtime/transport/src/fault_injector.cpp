#include <algorithm>
#include <charconv>
#include <string>

#include "clusterlm/transport/fault_injector.hpp"

namespace clusterlm::transport {
namespace {

std::size_t dir_index(FaultDirection d) { return d == FaultDirection::kSend ? 0 : 1; }

bool parse_uint(std::string_view s, unsigned long long max, unsigned long long& out) {
  if (s.empty()) return false;
  const auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), out);
  return ec == std::errc() && ptr == s.data() + s.size() && out <= max;
}

}  // namespace

Result<FaultRule> parse_fault_rule(std::string_view text) {
  FaultRule rule;
  bool first = true;
  bool have_ms = false;
  while (!text.empty() || first) {
    const auto colon = text.find(':');
    const std::string_view token = text.substr(0, colon);
    text = colon == std::string_view::npos ? std::string_view() : text.substr(colon + 1);
    if (first) {
      first = false;
      if (token == "close-before") rule.action = FaultAction::kCloseBefore;
      else if (token == "close-after") rule.action = FaultAction::kCloseAfter;
      else if (token == "delay") rule.action = FaultAction::kDelay;
      else if (token == "stall") rule.action = FaultAction::kStall;
      else if (token == "corrupt") rule.action = FaultAction::kCorruptPayload;
      else return make_error(ErrorCode::kInvalidArgument, "unknown fault action '" + std::string(token) + "'");
      if (colon == std::string_view::npos) break;
      continue;
    }
    const auto eq = token.find('=');
    if (eq == std::string_view::npos)
      return make_error(ErrorCode::kInvalidArgument, "fault option '" + std::string(token) + "' is not key=value");
    const std::string_view key = token.substr(0, eq);
    const std::string_view value = token.substr(eq + 1);
    unsigned long long n = 0;
    if (key == "type") {
      if (!parse_uint(value, 0xFFFF, n)) return make_error(ErrorCode::kInvalidArgument, "bad fault type");
      rule.frame_type = static_cast<std::uint16_t>(n);
    } else if (key == "channel") {
      if (!parse_uint(value, 0xFF, n)) return make_error(ErrorCode::kInvalidArgument, "bad fault channel");
      rule.channel = static_cast<std::uint8_t>(n);
    } else if (key == "nth") {
      if (!parse_uint(value, 0xFFFFFFFFull, n) || n == 0)
        return make_error(ErrorCode::kInvalidArgument, "fault nth must be >= 1");
      rule.nth = static_cast<std::uint32_t>(n);
    } else if (key == "dir") {
      if (value == "send") rule.dir = FaultDirection::kSend;
      else if (value == "recv" || value == "receive") rule.dir = FaultDirection::kReceive;
      else return make_error(ErrorCode::kInvalidArgument, "fault dir must be send or recv");
    } else if (key == "ms") {
      if (!parse_uint(value, 3'600'000, n)) return make_error(ErrorCode::kInvalidArgument, "bad fault ms");
      rule.delay_ms = static_cast<double>(n);
      have_ms = true;
    } else {
      return make_error(ErrorCode::kInvalidArgument, "unknown fault option '" + std::string(key) + "'");
    }
    if (colon == std::string_view::npos) break;
  }
  if (rule.action == FaultAction::kDelay && !have_ms)
    return make_error(ErrorCode::kInvalidArgument, "delay fault needs ms=<milliseconds>");
  return rule;
}

void FaultInjector::add_rule(const FaultRule& rule) {
  std::lock_guard<std::mutex> lk(mu_);
  rules_.push_back(Slot{rule, 0, false});
}

void FaultInjector::clear() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    rules_.clear();
    stalled_[0] = stalled_[1] = false;
  }
  cv_.notify_all();
}

std::uint64_t FaultInjector::fired_count() const {
  std::lock_guard<std::mutex> lk(mu_);
  return fired_;
}

void FaultInjector::clear_stall() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    stalled_[0] = stalled_[1] = false;
  }
  cv_.notify_all();
}

FaultInjector::Decision FaultInjector::on_frame(FaultDirection dir, const Frame& frame) {
  Decision d;
  std::lock_guard<std::mutex> lk(mu_);
  for (Slot& s : rules_) {
    if (s.fired || s.rule.dir != dir) continue;
    if (s.rule.frame_type && *s.rule.frame_type != frame.type) continue;
    if (s.rule.channel && *s.rule.channel != frame.channel) continue;
    if (++s.seen != s.rule.nth) continue;
    s.fired = true;
    ++fired_;
    switch (s.rule.action) {
      case FaultAction::kCloseBefore: d.close_before = true; break;
      case FaultAction::kCloseAfter: d.close_after = true; break;
      case FaultAction::kDelay: d.delay_ms += s.rule.delay_ms; break;
      case FaultAction::kStall:
        d.stall = true;
        stalled_[dir_index(dir)] = true;
        break;
      case FaultAction::kCorruptPayload: d.corrupt = true; break;
    }
  }
  return d;
}

bool FaultInjector::wait_unstalled(FaultDirection dir, std::chrono::steady_clock::time_point deadline,
                                   const std::atomic<bool>& abort) {
  std::unique_lock<std::mutex> lk(mu_);
  for (;;) {
    if (!stalled_[dir_index(dir)]) return true;
    if (abort.load(std::memory_order_acquire)) return false;
    const auto now = std::chrono::steady_clock::now();
    if (now >= deadline) return false;
    // Short slices: `abort` is a plain flag owned by the connection and cannot notify our condvar.
    cv_.wait_until(lk, std::min(deadline, now + std::chrono::milliseconds(10)));
  }
}

}  // namespace clusterlm::transport
