#pragma once
// Minimal structured logger.
//
// Privacy rule (spec §9): logs carry object IDs, timings and sizes — never activation contents, token IDs,
// candidate strings or expert-selection sequences. There is deliberately no API for logging a tensor.
#include <atomic>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <string>
#include <string_view>
#include <utility>

namespace clusterlm::log {

enum class Level : std::uint8_t { kDebug, kInfo, kWarn, kError };

void set_level(Level level);
Level level();
// Component name shown in every line (e.g. "father", "node-a").
void set_component(std::string component);

// Optional observer of every emitted line (the formatted text, without the trailing newline), invoked in
// addition to stderr output. Used by the diagnostics ring buffer and by privacy tests. The sink runs with the
// logger's internal lock held: it must be fast and must not log. An empty function clears it.
using Sink = std::function<void(std::string_view line)>;
void set_sink(Sink sink);

using Field = std::pair<std::string_view, std::string>;
void write(Level level, std::string_view event, std::initializer_list<Field> fields = {});

inline void debug(std::string_view e, std::initializer_list<Field> f = {}) { write(Level::kDebug, e, f); }
inline void info(std::string_view e, std::initializer_list<Field> f = {}) { write(Level::kInfo, e, f); }
inline void warn(std::string_view e, std::initializer_list<Field> f = {}) { write(Level::kWarn, e, f); }
inline void error(std::string_view e, std::initializer_list<Field> f = {}) { write(Level::kError, e, f); }

}  // namespace clusterlm::log
