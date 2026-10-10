#include "clusterlm/ui/machines_viewmodel.hpp"

#include <algorithm>
#include <cctype>

#include "clusterlm/ui/format.hpp"

namespace clusterlm::ui {

namespace {
std::string join(const std::vector<std::string>& v, const char* sep) {
  std::string out;
  for (std::size_t i = 0; i < v.size(); ++i) out += (i ? sep : "") + v[i];
  return out;
}
std::string mem_line(const char* what, std::uint64_t total, std::uint64_t free) {
  if (total == 0) return std::string();
  std::string s = format_bytes(total) + " " + what;
  if (free > 0) s += ", " + format_bytes(free) + " free";
  return s;
}
bool port_ok(const std::string& hp) {
  auto c = hp.rfind(':');
  if (c == std::string::npos || c == 0 || c + 1 >= hp.size()) return false;
  const std::string port = hp.substr(c + 1);
  if (port.size() > 5 || !std::all_of(port.begin(), port.end(), [](char ch) { return std::isdigit(static_cast<unsigned char>(ch)); })) return false;
  const int p = std::stoi(port);
  return p >= 1 && p <= 65535;
}
}  // namespace

void MachinesViewModel::fail(const Status& s) { st_.notice = Notice{Notice::Kind::kError, ascii_display(user_words(s.message()))}; }

void MachinesViewModel::refresh() {
  st_.pairing_available = pairing_ != nullptr;
  if (!client_.capabilities().machines) {
    st_.supported = false;
    st_.unsupported_text = "This Host does not report machine details yet.";
    st_.rows.clear();
    return;
  }
  st_.supported = true;
  auto ms = client_.list_machines();
  auto bs = client_.list_bindings();
  if (!ms.is_ok()) {
    st_.stale = true;
    for (auto& r : st_.rows) {
      r.state = "Status unknown";
      r.state_detail = "ClusterLM could not reach the Host.";
      r.usable_now = false;
    }
    fail(ms.status());
    return;
  }
  st_.stale = false;
  st_.rows.clear();
  st_.worker_choices.clear();
  for (const auto& m : ms.value()) {
    MachineRow r;
    r.id = m.id;
    r.name = ascii_display(m.name);
    r.role = m.is_host ? "Host" : "Worker";
    r.state = ascii_display(user_words(m.state));
    r.state_detail = ascii_display(user_words(m.state_detail));
    r.usable_now = m.state == "Available" || m.state == "Ready";
    if (m.on_battery) r.state_detail += (r.state_detail.empty() ? "" : " ") + std::string("Running on battery.");
    if (m.paused) r.state_detail += (r.state_detail.empty() ? "" : " ") + std::string("Paused by its user.");
    if (!m.os.empty()) r.details.push_back(ascii_display(m.os));
    if (!m.cpu.empty()) r.details.push_back(ascii_display(m.cpu));
    if (auto l = mem_line("memory", m.ram_bytes, m.free_ram_bytes); !l.empty()) r.details.push_back(l);
    for (const auto& g : m.gpus) r.details.push_back(ascii_display(g));
    if (auto l = mem_line("graphics memory", m.vram_bytes, m.free_vram_bytes); !l.empty()) r.details.push_back(l);
    r.last_seen = m.last_seen.empty() ? "no recent status" : m.last_seen;
    r.used_in = join(m.used_in, ", ");
    r.slots = m.bindings.empty() ? std::string() : "Fills: " + join(m.bindings, ", ");
    r.fingerprint = m.id.size() > 12 ? m.id.substr(0, 12) : (m.is_host ? std::string() : m.id);
    r.can_unpair = !m.is_host && pairing_ != nullptr;
    if (!m.is_host) st_.worker_choices.emplace_back(m.id, r.name);
    st_.rows.push_back(std::move(r));
  }
  st_.bindings.clear();
  if (bs.is_ok()) {
    for (const auto& b : bs.value()) {
      BindingRow r;
      r.name = b.name;
      r.label = b.name;
      r.machine_id = b.machine;
      r.machine_name = "Not assigned";
      for (const auto& m : st_.rows) if (!b.machine.empty() && m.id == b.machine) r.machine_name = m.name;
      if (!b.used_by.empty()) r.used_by = "Used by: " + join(b.used_by, ", ");
      st_.bindings.push_back(std::move(r));
    }
  }
}

Status MachinesViewModel::assign(const std::string& binding, const std::string& machine_id) {
  auto s = client_.set_binding(binding, machine_id);
  if (!s.is_ok()) { fail(s); return s; }
  st_.notice = Notice{Notice::Kind::kSuccess, machine_id.empty() ? "Unassigned." : "Assigned. Profiles that use this slot will check it now."};
  refresh();
  return s;
}

Status MachinesViewModel::pair() {
  st_.pair.error.clear();
  if (!pairing_) {
    st_.pair.error = "Pairing is not available from this window in this build.";
    return make_error(ErrorCode::kUnimplemented, st_.pair.error);
  }
  if (!port_ok(st_.pair.address)) {
    st_.pair.error = "Enter the Worker's address as name-or-IP:port, for example 192.168.1.20:47600.";
    return make_error(ErrorCode::kInvalidArgument, st_.pair.error);
  }
  if (st_.pair.code.empty()) {
    st_.pair.error = "Enter the code shown on the Worker.";
    return make_error(ErrorCode::kInvalidArgument, st_.pair.error);
  }
  PairingRequest req{st_.pair.address, st_.pair.code, st_.pair.name};
  auto s = pairing_->pair(req);
  st_.pair.code.clear();  // a one-time code is never kept
  if (!s.is_ok()) {
    st_.pair.error = ascii_display(user_words(s.message()));
    return s;
  }
  st_.pair = {};
  st_.notice = Notice{Notice::Kind::kSuccess, "Paired. Compare the short fingerprint with the one on the Worker's screen."};
  refresh();
  return s;
}

Status MachinesViewModel::unpair(const std::string& machine_name, bool i_confirm) {
  if (!pairing_) return make_error(ErrorCode::kUnimplemented, "Pairing is not available from this window in this build.");
  if (!i_confirm) {
    Status s = make_error(ErrorCode::kFailedPrecondition, "Confirm to unpair this Worker.");
    fail(s);
    return s;
  }
  auto s = pairing_->unpair(machine_name);
  if (!s.is_ok()) { fail(s); return s; }
  st_.notice = Notice{Notice::Kind::kSuccess, "Unpaired. It no longer takes part in anything."};
  refresh();
  return s;
}

}  // namespace clusterlm::ui
