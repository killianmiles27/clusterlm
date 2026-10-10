// Names and wording shared by the Host management views.
#include "clusterlm/ui/host_admin.hpp"

namespace clusterlm::ui {

std::string_view to_string(Provenance p) {
  switch (p) {
    case Provenance::kSynthetic: return "Synthetic";
    case Provenance::kMeasured: return "Measured";
    case Provenance::kQualified: return "Qualified";
  }
  return "Synthetic";
}

std::string_view to_string(CompatLabel l) {
  switch (l) {
    case CompatLabel::kSupportedQualified: return "Supported and qualified";
    case CompatLabel::kSupportedAwaitingQualification: return "Supported, awaiting hardware qualification";
    case CompatLabel::kExperimental: return "Experimental";
    case CompatLabel::kUnsupported: return "Unsupported";
  }
  return "Unsupported";
}

std::string_view to_string(ReadyState s) {
  switch (s) {
    case ReadyState::kUnavailable: return "unavailable";
    case ReadyState::kInstalled: return "installed";
    case ReadyState::kCompatible: return "compatible";
    case ReadyState::kLoadable: return "loadable";
    case ReadyState::kPreparing: return "preparing";
    case ReadyState::kReady: return "ready";
    case ReadyState::kBusy: return "busy";
  }
  return "unavailable";
}

std::string describe_ready_state(ReadyState s) {
  switch (s) {
    case ReadyState::kUnavailable: return "Needs attention";
    case ReadyState::kInstalled: return "Checking";
    case ReadyState::kCompatible: return "Waiting for machines";
    case ReadyState::kLoadable: return "Can be prepared";
    case ReadyState::kPreparing: return "Preparing";
    case ReadyState::kReady: return "Ready";
    case ReadyState::kBusy: return "Ready, answering";
  }
  return "Needs attention";
}

std::string describe_compat(CompatLabel l) { return std::string(to_string(l)); }

}  // namespace clusterlm::ui
