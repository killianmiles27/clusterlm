#include "clusterlm/ui/performance_viewmodel.hpp"

#include "clusterlm/ui/format.hpp"

namespace clusterlm::ui {

void PerformanceViewModel::refresh() {
  if (auto sum = client_.summary(); sum.is_ok()) {
    st_.dev_fixture = sum->dev_fixture;
    st_.banner = sum->dev_fixture ? "This Host is running the development test model. None of these numbers describes real hardware." : "";
  }
  auto runs = client_.recent_runs();
  if (!runs.is_ok()) {
    st_.stale = true;
    st_.notice = Notice{Notice::Kind::kError, ascii_display(runs.status().message())};
    st_.rows.clear();  // old numbers are not kept when they can no longer be refreshed
    st_.empty_text = "ClusterLM could not read recent answers from the Host.";
    return;
  }
  st_.stale = false;
  st_.rows.clear();
  for (const auto& r : runs.value()) {
    RunRow row;
    row.when = r.when;
    row.profile = ascii_display(r.profile_name);
    row.model = ascii_display(r.model);
    row.speed = format_rate(r.tok_s);
    row.first_token = format_millis(r.ttft_ms);
    row.tokens = std::to_string(r.tokens);
    row.result = r.reason == "completed" ? "Completed" : r.reason == "cancelled" ? "Stopped by you" : r.reason == "failed" ? "Failed" : ascii_display(r.reason);
    row.provenance = std::string(to_string(r.provenance));
    for (std::size_t i = 0; i < r.machines.size(); ++i) row.machines += (i ? ", " : "") + ascii_display(r.machines[i]);
    st_.rows.push_back(std::move(row));
  }
  st_.empty_text = st_.rows.empty() ? "No answers yet. Speeds appear here after you chat, each marked as measured or synthetic." : "";
}

std::string PerformanceViewModel::copy_text() const {
  std::string out;
  for (const auto& r : st_.rows) {
    out += r.when + "\t" + r.profile + "\t" + r.speed + "\t" + r.first_token + "\t" + r.tokens + " tokens\t" + r.result + "\t[" + r.provenance + "]\n";
  }
  return out;
}

}  // namespace clusterlm::ui
