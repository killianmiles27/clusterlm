#pragma once
// MachinesViewModel: the Host and its paired Workers by human name, what each is doing, and which Worker fills each
// named slot ("binding") that profiles refer to. Pairing keeps using the unchanged pairing protocol: this view only
// forwards the address and one-time code the user types (PairingPort), it never handles keys or fingerprints beyond
// showing the short fingerprint for the user to compare.
#include <string>
#include <vector>

#include "clusterlm/ui/father_client.hpp"
#include "clusterlm/ui/host_admin.hpp"
#include "clusterlm/ui/host_common.hpp"

namespace clusterlm::ui {

// Pairing is the Father agent's (docs/pairing.md); FatherClient already carries exactly these three calls.
class PairingPort {
 public:
  virtual ~PairingPort() = default;
  virtual Result<std::vector<PairedMachine>> paired() = 0;
  virtual Status pair(const PairingRequest& request) = 0;
  virtual Status unpair(std::string_view machine_name) = 0;
};
class FatherClientPairingPort final : public PairingPort {
 public:
  explicit FatherClientPairingPort(FatherClient& c) : c_(c) {}
  Result<std::vector<PairedMachine>> paired() override { return c_.paired_machines(); }
  Status pair(const PairingRequest& r) override { return c_.start_pairing(r); }
  Status unpair(std::string_view n) override { return c_.unpair(n); }
 private:
  FatherClient& c_;
};

struct MachineRow {
  std::string id, name;
  std::string role;         // "Host" | "Worker"
  std::string state;        // plain words, from the Host
  std::string state_detail;
  bool usable_now = false;  // state says Available/Ready; everything else is not compute you can count on
  std::vector<std::string> details;   // "Windows 11", "16 GB memory, 9.2 GB free", "NVIDIA RTX 3060, 12 GB (10.8 GB free)"
  std::string last_seen;
  std::string used_in;      // "Strong, Ultra" or empty
  std::string slots;        // "Fills: node:laptop-class"
  std::string fingerprint;  // short, for comparing with the Worker's own screen
  bool can_unpair = false;
};

struct BindingRow {
  std::string name;
  std::string label;        // friendlier form of the name
  std::string machine_id;   // current, "" = not assigned
  std::string machine_name; // "Not assigned" when empty
  std::string used_by;      // "Used by: Strong, Ultra"
};

struct PairForm {
  std::string address, code, name;
  std::string error;
};

struct MachinesState {
  bool supported = true;
  std::string unsupported_text;
  bool stale = false;
  std::vector<MachineRow> rows;
  std::vector<BindingRow> bindings;
  std::vector<std::pair<std::string, std::string>> worker_choices;  // machine id, name
  PairForm pair;
  bool pairing_available = true;
  Notice notice;
};

class MachinesViewModel {
 public:
  MachinesViewModel(HostAdminClient& client, PairingPort* pairing) : client_(client), pairing_(pairing) {}
  const MachinesState& state() const { return st_; }
  PairForm& pair_form() { return st_.pair; }

  void refresh();
  Status assign(const std::string& binding, const std::string& machine_id);   // "" unassigns
  Status pair();
  Status unpair(const std::string& machine_name, bool i_confirm);
  void dismiss_notice() { st_.notice = {}; }

 private:
  void fail(const Status& s);
  HostAdminClient& client_;
  PairingPort* pairing_;
  MachinesState st_;
};

}  // namespace clusterlm::ui
