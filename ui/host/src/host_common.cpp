#include "clusterlm/ui/host_common.hpp"

#include <cctype>

namespace clusterlm::ui {

std::string_view page_label(Page p) {
  switch (p) {
    case Page::kChat: return "Chat";
    case Page::kModels: return "Models";
    case Page::kProfiles: return "Profiles";
    case Page::kMachines: return "Machines";
    case Page::kConnections: return "Connections";
    case Page::kPerformance: return "Performance";
    case Page::kSettings: return "Settings";
  }
  return "Chat";
}

std::string display_role(std::string_view internal) {
  if (internal == "Father" || internal == "father") return "Host";
  if (internal == "Node" || internal == "node") return "Worker";
  return std::string(internal);
}

std::string user_words(std::string_view text) {
  struct Map {
    std::string_view from, to;
  };
  static constexpr Map kMap[] = {{"Fathers", "Hosts"}, {"Father", "Host"}, {"Nodes", "Workers"}, {"Node", "Worker"}};
  std::string out;
  out.reserve(text.size());
  auto is_word = [](char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-' || c == ':'; };
  for (std::size_t i = 0; i < text.size();) {
    bool done = false;
    if (i == 0 || !is_word(text[i - 1])) {
      for (const auto& m : kMap) {
        if (text.substr(i, m.from.size()) == m.from && (i + m.from.size() == text.size() || !is_word(text[i + m.from.size()]))) {
          out += m.to;
          i += m.from.size();
          done = true;
          break;
        }
      }
    }
    if (!done) out.push_back(text[i++]);
  }
  return out;
}

}  // namespace clusterlm::ui
