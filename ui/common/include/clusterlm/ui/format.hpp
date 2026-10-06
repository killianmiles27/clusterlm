#pragma once
// Display helpers shared by the view-models. Pure functions, ASCII output: the bundled ImGui font covers Latin
// only, so typographic punctuation in service messages is folded to ASCII before it reaches a draw call.
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "clusterlm/common/status.hpp"

namespace clusterlm::ui {

// Folds em/en dashes, curly quotes, ellipsis and non-breaking spaces to ASCII. Other multi-byte sequences are
// kept (they are the model's own text). Never throws.
std::string ascii_display(std::string_view text);
// "12.3 GB", "640 MB", "0 B".
std::string format_bytes(std::uint64_t bytes);
// "42 tok/s" with one decimal below 100; "--" when not positive.
std::string format_rate(double tok_s);
// "820 ms", "1.4 s"; "--" when not positive.
std::string format_millis(double ms);
// "about 3 min (estimate)"; empty when no ETA is known: a missing rate yields no ETA rather than a guess.
std::string format_eta(std::optional<double> seconds);
// A user-readable error: "<what happened> (<code>)". Never includes request text.
std::string describe_error(ErrorCode code, std::string_view message);

}  // namespace clusterlm::ui
