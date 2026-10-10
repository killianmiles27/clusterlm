#include "clusterlm/profiles/compat.hpp"

#include <algorithm>

namespace clusterlm::profiles {

using domain::CompatLabel;
using domain::Finding;
using Sev = Finding::Severity;

domain::ModelFacts model_facts(const library::ModelRecord& r, bool experimental_opt_in) {
  domain::ModelFacts f;
  f.container = r.files.size() > 1 ? "gguf-split" : "gguf";
  f.family = r.family;
  f.architecture = r.architecture;
  for (const auto& t : r.tensor_types) f.tensor_types.push_back(t.type);
  f.experimental_opt_in = experimental_opt_in;
  return f;
}

bool ProfileCompat::has_blocker() const {
  const auto blocker = [](const Finding& f) { return f.severity == Sev::kBlocker; };
  return std::any_of(findings.begin(), findings.end(), blocker) || (report && report->has_blocker()) ||
         (model.kind != library::Resolution::Kind::kVerified && model.kind != library::Resolution::Kind::kUnpinnedMatch);
}

CompatLabel ProfileCompat::label() const { return report ? report->label : CompatLabel::kUnsupported; }

Status validate_backend_options(const BackendRef& ref, const domain::BackendDescriptor& d) {
  for (const auto& o : ref.options) {
    auto it = std::find_if(d.options.begin(), d.options.end(), [&](const auto& kv) { return kv.first == o.key; });
    if (it == d.options.end())
      return make_error(ErrorCode::kInvalidArgument, "backend option '" + o.key + "' is not offered by " + d.id);
    const auto& spec = it->second;
    switch (spec.type) {
      case domain::OptionSpec::Type::kInteger: {
        const auto* v = std::get_if<std::int64_t>(&o.value);
        if (v == nullptr) return make_error(ErrorCode::kInvalidArgument, "backend option '" + o.key + "' must be an integer");
        if ((spec.minimum && *v < *spec.minimum) || (spec.maximum && *v > *spec.maximum))
          return make_error(ErrorCode::kInvalidArgument, "backend option '" + o.key + "' is out of range");
        break;
      }
      case domain::OptionSpec::Type::kBoolean:
        if (!std::holds_alternative<bool>(o.value)) return make_error(ErrorCode::kInvalidArgument, "backend option '" + o.key + "' must be true or false");
        break;
      case domain::OptionSpec::Type::kString: {
        const auto* v = std::get_if<std::string>(&o.value);
        if (v == nullptr) return make_error(ErrorCode::kInvalidArgument, "backend option '" + o.key + "' must be a string");
        if (!spec.allowed.empty() && std::find(spec.allowed.begin(), spec.allowed.end(), *v) == spec.allowed.end())
          return make_error(ErrorCode::kInvalidArgument, "backend option '" + o.key + "' has a value this backend does not offer");
        break;
      }
    }
  }
  return Status::ok();
}

ProfileCompat check_profile(const Profile& p, const CompatContext& ctx) {
  ProfileCompat out;
  auto add = [&](Sev s, std::string code, std::string msg) { out.findings.push_back({s, std::move(code), std::move(msg)}); };

  const domain::BackendDescriptor* desc = ctx.backends != nullptr ? ctx.backends->find(p.backend.id) : nullptr;
  if (desc == nullptr) {
    add(Sev::kBlocker, "backend_unknown", "This ClusterLM has no backend named '" + p.backend.id + "'.");
  } else {
    if (desc->development_only) add(Sev::kBlocker, "backend_development_only", "The " + desc->display_name + " backend is for testing ClusterLM and cannot serve a product profile.");
    if (auto st = validate_backend_options(p.backend, *desc); !st.is_ok()) add(Sev::kBlocker, "backend_options_invalid", st.message());
  }

  // Model identity: never guess.
  if (ctx.library != nullptr) {
    library::ModelIdentityQuery q{p.model.identity.family, p.model.identity.quant, p.model.identity.expected_root_hash};
    out.model = ctx.library->resolve(q);
    // library_id selects among candidates but never bypasses the identity or pin check.
    if (p.model.library_id) {
      const auto* by_id = ctx.library->find(*p.model.library_id);
      if (by_id == nullptr) {
        add(Sev::kBlocker, "library_id_stale", "The model this profile was set up with is no longer in the library.");
      } else if (out.model.kind == library::Resolution::Kind::kAmbiguous) {
        const auto& c = out.model.candidates;
        if (std::find(c.begin(), c.end(), by_id->id) != c.end()) {
          out.model.kind = q.expected_root_hash ? library::Resolution::Kind::kVerified : library::Resolution::Kind::kUnpinnedMatch;
          out.model.record = by_id;
          out.model.candidates.clear();
          out.model.detail = "chosen among several matches by the profile's library_id";
        } else {
          add(Sev::kBlocker, "library_id_mismatch", "The profile's library model is not one of the models that match its identity.");
        }
      } else if (out.model.record != nullptr && out.model.record->id != by_id->id) {
        add(Sev::kBlocker, "library_id_mismatch", "The profile's library model does not match its model identity.");
      }
    }
  }
  switch (out.model.kind) {
    case library::Resolution::Kind::kMissing:
      add(Sev::kBlocker, "model_missing", out.model.detail.empty() ? "The model is not in the library." : out.model.detail);
      break;
    case library::Resolution::Kind::kAmbiguous:
      add(Sev::kBlocker, "model_ambiguous", out.model.detail);
      break;
    case library::Resolution::Kind::kUnpinnedMatch:
      add(Sev::kWarning, "model_unpinned", "The model is matched by name and quantization only; confirm the inspected model before it is trusted.");
      break;
    case library::Resolution::Kind::kVerified:
      break;
  }

  if (desc != nullptr && out.model.record != nullptr) {
    domain::BackendRuntimeStatus rt;
    if (ctx.runtime) rt = ctx.runtime(*desc);
    const bool opt_in = ctx.experimental_opt_in && ctx.experimental_opt_in(out.model.record->id);
    out.report = domain::check_model(*desc, model_facts(*out.model.record, opt_in), rt);

    const auto workers = worker_count_bounds(p.topology);
    if (workers.max > out.report->max_workers) {
      add(Sev::kBlocker, "too_many_workers",
          "This profile may use " + std::to_string(workers.max) + " Workers but this model on this backend is validated for " +
              std::to_string(out.report->max_workers) + ".");
    }
    if (worker_slot_count(p.topology) > 0 && !desc->execution.cross_machine)
      add(Sev::kBlocker, "backend_not_distributed", "This backend runs on one machine; the profile asks for Workers.");
    if (workers.min > 0 && !out.report->distributable)
      add(Sev::kBlocker, "model_not_distributable", "This model is not supported across machines, so Worker slots cannot be required.");
    const std::uint32_t q = p.speculation.enabled ? p.speculation.max_q : 1;
    if (q > desc->speculation.max_q)
      add(Sev::kBlocker, "speculation_too_wide", "Speculation width " + std::to_string(q) + " exceeds what this backend validates (" + std::to_string(desc->speculation.max_q) + ").");
    if (q > 1 && !desc->speculation.window_commit_abort)
      add(Sev::kBlocker, "speculation_unsupported", "This backend cannot commit or abort speculative windows.");
    if (p.placement.mode == Placement::Mode::kManual && !desc->execution.manual_layer_ranges)
      add(Sev::kBlocker, "manual_placement_unsupported", "This backend does not accept manual layer ranges.");
    if (out.model.record->context_length && p.context.max_tokens > *out.model.record->context_length)
      add(Sev::kWarning, "context_above_model_limit", "The profile offers contexts longer than the model's trained context.");
  }
  return out;
}

}  // namespace clusterlm::profiles
