#pragma once
// PerformanceViewModel: what recent answers actually did. Only numbers the Host reported are shown, each with its
// provenance; with nothing reported the page says so (no placeholder figures, no averages across provenances).
#include <string>
#include <vector>

#include "clusterlm/ui/host_admin.hpp"
#include "clusterlm/ui/host_common.hpp"

namespace clusterlm::ui {

struct RunRow {
  std::string when, profile, model;
  std::string speed, first_token, tokens;  // "14.2 tok/s" or "--"
  std::string result;                      // "Completed" | "Stopped by you" | "Failed"
  std::string provenance;                  // "Measured" | "Synthetic" | "Qualified"
  std::string machines;
};

struct PerformanceState {
  bool stale = false;
  bool dev_fixture = false;
  std::string banner;          // "Development fixture model: none of these numbers are measurements of real hardware."
  std::vector<RunRow> rows;
  std::string empty_text;      // shown when rows is empty
  Notice notice;
};

class PerformanceViewModel {
 public:
  explicit PerformanceViewModel(HostAdminClient& client) : client_(client) {}
  const PerformanceState& state() const { return st_; }
  void refresh();
  // Plain text for the clipboard, one line per run, with provenance on every line.
  std::string copy_text() const;

 private:
  HostAdminClient& client_;
  PerformanceState st_;
};

}  // namespace clusterlm::ui
