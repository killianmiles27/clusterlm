#pragma once
// The routed-expert kernel shared by the expert-domain server and Father's executor.
//
// It is a line-for-line copy of the reference backend's SwiGLU expert (runtime/domain/src/reference_domain.cpp,
// `swiglu` + the `y += w * d` accumulation), built on the same refmath primitives and compiled with
// -ffp-contract=off. tests/expert_domains proves bitwise equality with the reference domain when one owner
// executes every selected expert of a position.
#include <cstddef>
#include <vector>

#include "clusterlm/common/status.hpp"
#include "clusterlm/objects/provisioned.hpp"
#include "clusterlm/objects/tensor_codec.hpp"
// Private header of runtime/domain (header-only kernels). Included by relative path so no extra include directory is
// needed anywhere (CMake, the Windows syntax check). It is deliberately the SAME code the reference backend runs.
#include "../../../runtime/domain/src/reference_math.hpp"

namespace clusterlm::expert_domains {

struct ExpertWeights {
  std::vector<float> gate, up, down;  // [ff][H], [ff][H], [H][ff] after exact dequantization
};

struct ExpertScratch {
  std::vector<float> a, b, z, d;
  void init(std::size_t hidden, std::size_t ff) {
    a.assign(ff, 0.0f);
    b.assign(ff, 0.0f);
    z.assign(ff, 0.0f);
    d.assign(hidden, 0.0f);
  }
};

// Decodes one routed-expert object: three stacked source ranges (gate, up, down) in the object's quantization.
inline Status load_expert_weights(const objects::ProvisionedObject& ex, std::size_t hidden, std::size_t ff,
                                  ExpertWeights& out) {
  const std::string& qt = ex.entry->representation.quant_type;
  const std::uint64_t gb = objects::tensor_bytes(qt, ff * hidden), db = objects::tensor_bytes(qt, hidden * ff);
  if (gb == 0 || db == 0 || ex.bytes.size() != 2 * gb + db)
    return make_error(ErrorCode::kDataLoss, "expert object '" + ex.entry->name + "' has the wrong size");
  out.gate.resize(ff * hidden);
  out.up.resize(ff * hidden);
  out.down.resize(hidden * ff);
  CLM_RETURN_IF_ERROR(objects::decode_tensor(qt, ex.bytes.subspan(0, gb), out.gate));
  CLM_RETURN_IF_ERROR(objects::decode_tensor(qt, ex.bytes.subspan(gb, gb), out.up));
  return objects::decode_tensor(qt, ex.bytes.subspan(2 * gb, db), out.down);
}

// sc.d = down * (silu(gate*h) (.) (up*h))
inline void swiglu(const float* gate, const float* up, const float* down, std::size_t hidden, std::size_t ff,
                   const float* h, ExpertScratch& sc) {
  domain::refmath::matvec(gate, ff, hidden, h, sc.a.data());
  domain::refmath::matvec(up, ff, hidden, h, sc.b.data());
  for (std::size_t c = 0; c < ff; ++c) sc.z[c] = domain::refmath::silu(sc.a[c]) * sc.b[c];
  domain::refmath::matvec(down, hidden, ff, sc.z.data(), sc.d.data());
}

// y += weight * expert(h)
inline void accumulate_expert(const ExpertWeights& w, std::size_t hidden, std::size_t ff, const float* h,
                              float weight, ExpertScratch& sc, float* y) {
  swiglu(w.gate.data(), w.up.data(), w.down.data(), hidden, ff, h, sc);
  for (std::size_t c = 0; c < hidden; ++c) y[c] += weight * sc.d[c];
}

}  // namespace clusterlm::expert_domains
