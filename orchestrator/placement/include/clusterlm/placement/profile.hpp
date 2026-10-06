#pragma once
// Measured hardware and network profiles.
//
// Placement never branches on a device name: a domain is fully described by these numbers. `gpu.name` is
// informational and is never read by the cost model or the search. Every numeric field is a Quantity so
// that provenance survives JSON round trips and propagates into plan provenance.
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "clusterlm/common/status.hpp"
#include "clusterlm/placement/provenance.hpp"

namespace clusterlm::placement {

enum class DomainRole : std::uint8_t { kFather, kNode };

struct CpuProfile {
  std::string arch;                   // e.g. "zen4"; informational
  std::vector<std::string> features;  // e.g. "avx2", "avx512f"
  Quantity usable_threads;
  // Effective expert throughput at q=1, INCLUDING dequant + GEMV, keyed by quant type ("q4_k", ...).
  std::map<std::string, Quantity> expert_bytes_per_s;
  // Extra CPU compute cost per additional verified position, as a multiplier: time *= 1 + q_scaling*(q-1).
  // Weight traffic for the q-position expert union is modelled separately; this covers per-position GEMV work.
  Quantity q_scaling;
  // Thermal / power derate in (0,1] applied to CPU throughput under sustained load.
  Quantity sustained_factor;
};

struct MemoryProfile {
  Quantity ram_total;
  Quantity ram_safe_allowance;  // the most this product may commit without hurting the user's machine
  Quantity ram_bandwidth;
  Quantity pinned_limit;  // max pinned/staging memory
};

struct GpuProfile {
  std::string name;  // informational only
  Quantity vram_total;
  Quantity vram_budget;  // 0 => no usable GPU
  Quantity gpu_expert_bytes_per_s;
  std::map<std::string, Quantity> dense_layer_ms;  // keyed by "recurrent" / "attention", q=1
  Quantity dense_q_scaling;                        // dense time *= 1 + dense_q_scaling*(q-1)
  Quantity pcie_h2d_bytes_per_s;
  // Prefill rate when this domain would run ALL model layers; a domain with a fraction f of the layers
  // prefills at rate/f.
  Quantity prefill_tokens_per_s;
};

struct OverheadProfile {
  Quantity os_reserve_ram;
  Quantity scratch_vram;
  Quantity staging_ram;
};

struct HardwareProfile {
  std::string id;
  DomainRole role = DomainRole::kNode;
  CpuProfile cpu;
  MemoryProfile memory;
  GpuProfile gpu;
  OverheadProfile overheads;
  // Mirrored to JSON ("synthetic": true) for fixtures; per-quantity provenance is the real authority.
  bool synthetic_fixture = false;
  std::string note;
};

struct LinkProfile {
  std::string from, to;
  Quantity bandwidth_bytes_per_s;
  Quantity rtt_ms;
  Quantity jitter_ms;
};

struct NetworkProfile {
  std::vector<LinkProfile> links;
  Quantity father_egress_bytes_per_s;  // shared NIC: provisioning to all nodes shares this
  bool synthetic_fixture = false;
  std::string note;

  // Directed lookup first, then the reverse direction (links are symmetric unless both are listed).
  const LinkProfile* find_link(std::string_view from, std::string_view to) const;
};

// Visits every Quantity with a dotted field path (for validation, qualification and provenance folding).
using QuantityVisitor = std::function<void(const std::string& path, Quantity&)>;
void for_each_quantity(HardwareProfile& p, const QuantityVisitor& fn);
void for_each_quantity(NetworkProfile& n, const QuantityVisitor& fn);
Provenance weakest_provenance(const HardwareProfile& p);
Provenance weakest_provenance(const NetworkProfile& n);
Provenance weakest_link_provenance(const LinkProfile& l);

// Structural/physical sanity checks (non-negative, sustained_factor in (0,1], budgets <= totals, ...).
Status validate(const HardwareProfile& p);
Status validate(const NetworkProfile& n);

// The only way to produce a Qualified profile. Fails if ANY quantity is still Synthetic; on success every
// quantity becomes Qualified and `qualification_source` is appended to its source. Never partially applied.
Status mark_qualified(HardwareProfile& p, std::string_view qualification_source);
Status mark_qualified(NetworkProfile& n, std::string_view qualification_source);

std::string to_json(const HardwareProfile& p, int indent = 2);
std::string to_json(const NetworkProfile& n, int indent = 2);
Result<HardwareProfile> hardware_profile_from_json(std::string_view json);
Result<NetworkProfile> network_profile_from_json(std::string_view json);
Result<HardwareProfile> load_hardware_profile(const std::string& path);
Result<NetworkProfile> load_network_profile(const std::string& path);
Status save_hardware_profile(const HardwareProfile& p, const std::string& path);
Status save_network_profile(const NetworkProfile& n, const std::string& path);

}  // namespace clusterlm::placement
