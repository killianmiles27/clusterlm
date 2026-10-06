#include "clusterlm/common/log.hpp"

#include <chrono>
#include <cstdio>
#include <mutex>

namespace clusterlm::log {
namespace {
std::atomic<Level> g_level{Level::kInfo};
std::mutex g_mutex;
std::string g_component = "clusterlm";

const char* level_name(Level l) {
  switch (l) {
    case Level::kDebug: return "debug";
    case Level::kInfo: return "info";
    case Level::kWarn: return "warn";
    case Level::kError: return "error";
  }
  return "?";
}
}  // namespace

void set_level(Level l) { g_level.store(l); }
Level level() { return g_level.load(); }

void set_component(std::string component) {
  std::lock_guard lock(g_mutex);
  g_component = std::move(component);
}

void write(Level l, std::string_view event, std::initializer_list<Field> fields) {
  if (l < g_level.load()) return;
  auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::system_clock::now().time_since_epoch())
                 .count();
  std::string line;
  line.reserve(128);
  line += "ts=" + std::to_string(now);
  line += " level=";
  line += level_name(l);
  std::lock_guard lock(g_mutex);
  line += " component=" + g_component;
  line += " event=";
  line += event;
  for (const auto& [k, v] : fields) {
    line += ' ';
    line += k;
    line += '=';
    line += v;
  }
  line += '\n';
  std::fputs(line.c_str(), stderr);
}

}  // namespace clusterlm::log
