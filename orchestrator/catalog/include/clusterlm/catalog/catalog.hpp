#pragma once
// Tier catalog: the three product tiers (Fast, Strong, Ultra) as data.
//
// The catalog is policy and identity, never a measurement:
//   * model identity is "unpinned" until Father has downloaded and inspected the artifact (expected root hash
//     null); readiness refuses an unpinned model unless the user confirmed the inspected manifest;
//   * topology is expressed as ROLES ("father", "node:laptop-class", "node:designated-3060"), never machine
//     names. A TierAssignment (pairing/user level) binds roles to concrete paired machines;
//   * every context profile starts "qualified: false" and every performance target "pending_qualification" —
//     a target is a goal, never a promise.
// Loading is bounded: size-capped input, nesting-depth cap, strict field validation, no exceptions escape.
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "clusterlm/common/status.hpp"

namespace clusterlm::catalog {

inline constexpr std::string_view kRoleFather = "father";
inline constexpr std::string_view kRoleLaptop = "node:laptop-class";
inline constexpr std::string_view kRole3060 = "node:designated-3060";

inline constexpr std::size_t kMaxCatalogBytes = 1u << 20;  // 1 MiB
inline constexpr int kMaxJsonDepth = 16;

enum class BackendKind : std::uint8_t {
  kLlamaLocal,    // Father-local llama.cpp (Fast tier may remain a plain local model)
  kStrataHybrid,  // Strata-derived hybrid backend driven through ClusterLM execution domains
};
std::string_view to_string(BackendKind b) noexcept;

enum class PinStatus : std::uint8_t { kUnpinned, kPinned };

struct ExpectedFile {
  std::string role;                         // "model", "transformer-shard", "lookup-shard"
  std::optional<std::string> name;          // null until first inspection
  std::optional<std::uint64_t> approx_bytes;  // published size, approximate
};

struct ModelIdentity {
  std::string family;
  std::string display_name;                 // what every user-facing event names ("which model answers")
  std::optional<std::string> artifact_id;
  std::string quant;
  std::vector<ExpectedFile> expected_files;
  std::optional<std::string> expected_root_hash;  // hex SHA-256 of the manifest root; null = unpinned
  PinStatus pin_status() const { return expected_root_hash ? PinStatus::kPinned : PinStatus::kUnpinned; }
};

// Requirement keys a context profile may list.
inline constexpr std::string_view kReqFeasiblePlacement = "feasible_placement";      // enforced by readiness
inline constexpr std::string_view kReqQualificationRequired = "qualification_required";  // informational note

struct ContextProfile {
  std::uint32_t tokens = 0;
  bool offered = true;      // false: not offered for this tier at all
  bool qualified = false;   // never true in the shipped catalog until the qualification experiments pass
  std::vector<std::string> requirements;
  bool has_requirement(std::string_view key) const;
};

enum class TargetStatus : std::uint8_t { kPendingQualification, kMeasured, kQualified };
std::string_view to_string(TargetStatus s) noexcept;

struct PerformanceTarget {
  std::string metric;      // e.g. "decode_tok_s_median"
  double value = 0;
  TargetStatus status = TargetStatus::kPendingQualification;
  std::string experiment;  // qualification experiment id that decides it
};

enum class LossAction : std::uint8_t { kStop, kDowngrade };

// Explicit policy when a distributed session of this tier is invalidated (node lost, local activity, fault).
struct SessionLossPolicy {
  std::uint32_t max_retries = 0;  // attempts to re-prepare the SAME tier before acting on `then`
  LossAction then = LossAction::kStop;
};

struct TierEntry {
  std::string id;            // "fast" | "strong" | "ultra"
  std::string display_name;  // "Fast"
  ModelIdentity model;
  BackendKind backend = BackendKind::kStrataHybrid;
  std::vector<std::string> roles;  // pipeline order: father first, then nodes
  std::vector<ContextProfile> contexts;
  std::vector<PerformanceTarget> targets;
  SessionLossPolicy on_session_loss;
  std::vector<std::string> qualification_experiments;

  const ContextProfile* find_context(std::uint32_t tokens) const;
  std::size_t node_role_count() const;
};

class Catalog {
 public:
  // Bounded parse + full validation.
  static Result<Catalog> parse(std::string_view json);
  static Result<Catalog> load(const std::string& path);

  const std::string& id() const { return id_; }
  const std::vector<TierEntry>& tiers() const { return tiers_; }
  const TierEntry* find(std::string_view tier_id) const;
  // Ultra -> Strong -> Fast.
  const std::vector<std::string>& fallback_order() const { return fallback_; }
  // Tiers strictly below `tier_id` in the fallback order, in order.
  std::vector<const TierEntry*> fallbacks_after(std::string_view tier_id) const;

 private:
  std::string id_;
  std::vector<TierEntry> tiers_;
  std::vector<std::string> fallback_;
};

// Validates the structural rules the parser enforces (also usable on hand-built catalogs in tests).
// Exactly three tiers fast/strong/ultra; known roles only; father first; node counts 0/1/2; Node 3 (any third
// node role) never in Ultra; llama-local only for father-only tiers; fallback order a permutation of the tiers
// from highest to lowest.
Status validate_tier(const TierEntry& tier);

// Binds roles to concrete paired machines (user/pairing level). A machine appears at most once per tier.
class TierAssignment {
 public:
  void bind(std::string role, std::string machine_id) { map_[std::move(role)] = std::move(machine_id); }
  std::optional<std::string> machine_for(std::string_view role) const;
  // Every role of the tier bound; no machine bound to two roles of this tier.
  Status validate_for(const TierEntry& tier) const;

 private:
  std::map<std::string, std::string, std::less<>> map_;
};

}  // namespace clusterlm::catalog
