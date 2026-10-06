#include "clusterlm/expert_domains/wire.hpp"

#include <cmath>

namespace clusterlm::expert_domains {

namespace {

Status bad(std::string m) { return make_error(ErrorCode::kProtocolError, std::move(m)); }

void put_floats(ByteWriter& w, const std::vector<float>& v) {
  for (float f : v) w.f32(f);
}

// Reads exactly n floats after checking that the input really holds them (so a lying header can't make us
// allocate more than the payload size) and that none is NaN/Inf.
bool get_floats(ByteReader& r, std::size_t n, std::vector<float>& out) {
  if (r.remaining() / 4 < n) return false;
  out.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    if (!r.f32(out[i]) || !std::isfinite(out[i])) return false;
  }
  return true;
}

Status check_dims(std::uint32_t layer, std::uint32_t q, std::uint32_t h, const ExpertDecodeLimits& lim) {
  if (layer >= lim.max_layer) return bad("layer out of bounds");
  if (q == 0 || q > lim.max_positions) return bad("positions out of bounds");
  if (h == 0 || h > lim.max_hidden) return bad("hidden size out of bounds");
  return Status::ok();
}

transport::Frame frame(std::uint16_t type, Bytes payload, std::uint64_t correlation) {
  transport::Frame f;
  f.type = type;
  f.channel = kExpertChannel;
  f.correlation = correlation;
  f.payload = std::move(payload);
  return f;
}

}  // namespace

Bytes encode(const ExpertBatch& m) {
  ByteWriter w(64 + m.activations.size() * 4);
  w.u64(m.epoch.value);
  w.u64(m.window.value);
  w.u32(m.layer);
  w.u32(m.positions);
  w.u32(m.hidden);
  put_floats(w, m.activations);
  for (const auto& per : m.routes) {
    w.u16(static_cast<std::uint16_t>(per.size()));
    for (const ExpertRoute& r : per) {
      w.u32(r.local_expert);
      w.f32(r.weight);
    }
  }
  return std::move(w).take();
}

Bytes encode(const ExpertResult& m) {
  ByteWriter w(64 + m.partial.size() * 4);
  w.u64(m.epoch.value);
  w.u64(m.window.value);
  w.u32(m.layer);
  w.u32(m.positions);
  w.u32(m.hidden);
  put_floats(w, m.partial);
  w.u64(m.compute_ns);
  w.u32(m.experts_executed);
  return std::move(w).take();
}

Bytes encode(const ExpertError& m) {
  ByteWriter w;
  w.u16(static_cast<std::uint16_t>(m.code));
  w.str(m.message);
  return std::move(w).take();
}

Result<ExpertBatch> decode_batch(ByteSpan payload, const ExpertDecodeLimits& lim) {
  ByteReader r(payload);
  ExpertBatch m;
  std::uint64_t epoch = 0, window = 0;
  if (!r.u64(epoch) || !r.u64(window) || !r.u32(m.layer) || !r.u32(m.positions) || !r.u32(m.hidden))
    return bad("truncated ExpertBatch header");
  m.epoch = Epoch{epoch};
  m.window = WindowId{window};
  CLM_RETURN_IF_ERROR(check_dims(m.layer, m.positions, m.hidden, lim));
  if (!get_floats(r, std::size_t{m.positions} * m.hidden, m.activations)) return bad("bad ExpertBatch activations");
  m.routes.resize(m.positions);
  for (auto& per : m.routes) {
    std::uint16_t n = 0;
    if (!r.u16(n)) return bad("truncated ExpertBatch routes");
    if (n > lim.max_routes_per_position) return bad("too many routes for one position");
    if (r.remaining() / 8 < n) return bad("truncated ExpertBatch routes");
    per.resize(n);
    for (std::uint16_t i = 0; i < n; ++i) {
      if (!r.u32(per[i].local_expert) || !r.f32(per[i].weight)) return bad("truncated ExpertBatch routes");
      if (per[i].local_expert >= lim.max_local_experts) return bad("local expert id out of bounds");
      if (!std::isfinite(per[i].weight)) return bad("non-finite route weight");
      // Strictly ascending: no duplicates, and the domain's summation order is fixed by the wire form.
      if (i > 0 && per[i].local_expert <= per[i - 1].local_expert) return bad("routes must be strictly ascending");
    }
  }
  CLM_RETURN_IF_ERROR(r.finish("ExpertBatch"));
  return m;
}

Result<ExpertResult> decode_result(ByteSpan payload, const ExpertDecodeLimits& lim) {
  ByteReader r(payload);
  ExpertResult m;
  std::uint64_t epoch = 0, window = 0;
  if (!r.u64(epoch) || !r.u64(window) || !r.u32(m.layer) || !r.u32(m.positions) || !r.u32(m.hidden))
    return bad("truncated ExpertResult header");
  m.epoch = Epoch{epoch};
  m.window = WindowId{window};
  CLM_RETURN_IF_ERROR(check_dims(m.layer, m.positions, m.hidden, lim));
  if (!get_floats(r, std::size_t{m.positions} * m.hidden, m.partial)) return bad("bad ExpertResult partial sums");
  if (!r.u64(m.compute_ns) || !r.u32(m.experts_executed)) return bad("truncated ExpertResult trailer");
  CLM_RETURN_IF_ERROR(r.finish("ExpertResult"));
  return m;
}

Result<ExpertError> decode_error(ByteSpan payload, const ExpertDecodeLimits& lim) {
  ByteReader r(payload);
  ExpertError m;
  std::uint16_t code = 0;
  if (!r.u16(code) || !r.str(m.message, lim.max_error_message)) return bad("bad ExpertError");
  CLM_RETURN_IF_ERROR(r.finish("ExpertError"));
  m.code = code <= static_cast<std::uint16_t>(ErrorCode::kHardwareUnavailable) ? static_cast<ErrorCode>(code)
                                                                              : ErrorCode::kInternal;
  return m;
}

transport::Frame to_frame(const ExpertBatch& m, std::uint64_t c) { return frame(kMsgExpertBatch, encode(m), c); }
transport::Frame to_frame(const ExpertResult& m, std::uint64_t c) { return frame(kMsgExpertResult, encode(m), c); }
transport::Frame to_frame(const ExpertError& m, std::uint64_t c) { return frame(kMsgExpertError, encode(m), c); }

}  // namespace clusterlm::expert_domains
