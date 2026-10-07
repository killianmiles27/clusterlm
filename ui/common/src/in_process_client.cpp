// InProcessFatherClient: a thin, honest adapter over father::FatherService.
#include "clusterlm/ui/clients.hpp"

namespace clusterlm::ui {

InProcessFatherClient::InProcessFatherClient(father::FatherService& service, Options options)
    : svc_(service), opts_(std::move(options)) {}

Result<std::vector<catalog::TierReadiness>> InProcessFatherClient::list_tiers(std::uint32_t context_tokens) {
  return svc_.list_tiers(context_tokens);
}

Result<std::vector<TierParticipant>> InProcessFatherClient::participants(std::string_view tier_id) {
  const auto* tier = opts_.catalog.find(tier_id);
  if (!tier) return make_error(ErrorCode::kNotFound, "unknown tier");
  std::vector<TierParticipant> out;
  out.reserve(tier->roles.size());
  for (const auto& role : tier->roles) out.push_back({role, opts_.assignment.machine_for(role).value_or(role)});
  return out;
}

std::string InProcessFatherClient::selected_tier() { return svc_.selected_tier(); }
Status InProcessFatherClient::select_tier(std::string_view id) { return svc_.select_tier(id); }
Status InProcessFatherClient::prepare_tier(std::string_view id, std::uint32_t ctx) { return svc_.prepare_tier(id, ctx); }
Result<father::RequestId> InProcessFatherClient::send_chat(father::ChatRequest r) { return svc_.chat(std::move(r)); }
Status InProcessFatherClient::cancel(father::RequestId r) { return svc_.cancel(r); }
Status InProcessFatherClient::release() { return svc_.release(); }
Status InProcessFatherClient::reset_conversation() { return svc_.reset_conversation(); }

Result<ContextUse> InProcessFatherClient::context_use() {
  if (!opts_.tokenizer) return make_error(ErrorCode::kUnavailable, "no tokenizer attached: context use is unknown");
  const auto conv = svc_.conversation();
  ContextUse u;
  u.tokens_used = static_cast<std::uint32_t>(opts_.tokenizer->encode_chat(conv).size());
  u.exact = true;
  return u;
}

Result<DiagnosticsExport> InProcessFatherClient::export_diagnostics(bool include_text) {
  DiagnosticsExport e;
  e.text = svc_.diagnostics(include_text).to_string();
  e.includes_conversation = include_text;
  // FatherService does not expose its plan or per-stage timings; leave them empty so the UI says "not reported".
  return e;
}

Result<FatherSettings> InProcessFatherClient::get_settings() {
  std::lock_guard lk(mu_);
  return settings_;
}

Status InProcessFatherClient::set_settings(const FatherSettings& s) {
  if (auto st = validate(s); !st.is_ok()) return st;
  std::lock_guard lk(mu_);
  settings_ = s;
  return Status::ok();
}

Result<std::vector<PairedMachine>> InProcessFatherClient::paired_machines() { return opts_.paired; }

Status InProcessFatherClient::start_pairing(const PairingRequest&) {
  return make_error(ErrorCode::kUnimplemented, "Pairing is not available in the in-process client.");
}
Status InProcessFatherClient::unpair(std::string_view) {
  return make_error(ErrorCode::kUnimplemented, "Unpairing is not available in the in-process client.");
}

}  // namespace clusterlm::ui
