#pragma once
// Pieces shared by the Host management view-models: pages, notices, naming.
//
// Naming (R-50): the UI says Host and Worker; the code and the wire keep Father and Node. `display_role` is the one
// place the mapping lives, so a stray "Father" or "Node" in a UI string is a bug a test can find.
#include <array>
#include <cstdint>
#include <string>
#include <string_view>

namespace clusterlm::ui {

enum class Page : std::uint8_t { kChat, kModels, kProfiles, kMachines, kConnections, kPerformance, kSettings };
inline constexpr std::array<Page, 7> kAllPages = {Page::kChat,        Page::kModels,      Page::kProfiles,   Page::kMachines,
                                                  Page::kConnections, Page::kPerformance, Page::kSettings};
std::string_view page_label(Page p);  // "Chat", "Models", ...

// Internal role/term -> what the user sees. "Father" -> "Host", "Node" -> "Worker"; anything else unchanged.
std::string display_role(std::string_view internal);
// Rewrites whole words "Father"/"Node" (and plurals) in service-authored text to "Host"/"Worker".
std::string user_words(std::string_view text);

// A message shown at the top of a page: what happened and what to do next. Errors never carry request text.
struct Notice {
  enum class Kind : std::uint8_t { kNone, kInfo, kSuccess, kError } kind = Kind::kNone;
  std::string text;
  bool empty() const { return kind == Kind::kNone; }
};

}  // namespace clusterlm::ui
