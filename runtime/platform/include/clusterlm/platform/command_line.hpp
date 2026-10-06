#pragma once
// Windows command-line construction for CreateProcessW, kept free of any Win32 type so the exact quoting used
// in production is unit-tested on every host.
//
// ClusterLM never starts a process through a shell: no cmd.exe, no system(), no ShellExecute. The only launch
// path is ChildProcess::spawn (posix_spawn / CreateProcessW with an explicit executable and one quoted argument
// per argv entry). Arguments are therefore only ever interpreted by the child's own C runtime, i.e. by the
// rules implemented here (the MSVC / CommandLineToArgvW parsing rules):
//   * an argument needs quotes iff it is empty or contains space, tab, newline, vertical tab or '"';
//   * inside quotes, a run of N backslashes followed by '"' becomes 2N+1 backslashes and the quote;
//   * a run of N backslashes at the end of the argument becomes 2N backslashes (so the closing quote survives);
//   * backslashes anywhere else are literal.
#include <string>
#include <string_view>
#include <vector>

namespace clusterlm::platform {

template <typename CharT>
std::basic_string<CharT> quote_windows_argument(std::basic_string_view<CharT> arg) {
  constexpr CharT kQuote = static_cast<CharT>('"');
  constexpr CharT kSlash = static_cast<CharT>('\\');
  bool needs_quotes = arg.empty();
  for (CharT c : arg)
    if (c == static_cast<CharT>(' ') || c == static_cast<CharT>('\t') || c == static_cast<CharT>('\n') ||
        c == static_cast<CharT>('\v') || c == kQuote)
      needs_quotes = true;
  if (!needs_quotes) return std::basic_string<CharT>(arg);
  std::basic_string<CharT> out(1, kQuote);
  std::size_t backslashes = 0;
  for (CharT c : arg) {
    if (c == kSlash) {
      ++backslashes;
    } else if (c == kQuote) {
      out.append(backslashes * 2 + 1, kSlash);
      out.push_back(c);
      backslashes = 0;
    } else {
      out.append(backslashes, kSlash);
      out.push_back(c);
      backslashes = 0;
    }
  }
  out.append(backslashes * 2, kSlash);
  out.push_back(kQuote);
  return out;
}

// `executable` followed by every argument, each quoted, separated by single spaces.
template <typename CharT>
std::basic_string<CharT> build_windows_command_line(std::basic_string_view<CharT> executable,
                                                    const std::vector<std::basic_string<CharT>>& args) {
  std::basic_string<CharT> cmd = quote_windows_argument<CharT>(executable);
  for (const auto& a : args) {
    cmd.push_back(static_cast<CharT>(' '));
    cmd += quote_windows_argument<CharT>(a);
  }
  return cmd;
}

}  // namespace clusterlm::platform
