// Placement profile loaders (hardware and network JSON). Byte 0 selects which loader runs.
#include "clusterlm/placement/profile.hpp"
#include "fuzz_util.hpp"

using namespace clusterlm;

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  if (size == 0) return 0;
  const std::string_view text = fuzz::text_of(data + 1, size - 1);
  if ((data[0] & 1) == 0) {
    auto p = placement::hardware_profile_from_json(text);
    if (p.is_ok()) {
      CLM_FUZZ_CHECK(placement::validate(p.value()).is_ok());
      (void)placement::weakest_provenance(p.value());
      auto again = placement::hardware_profile_from_json(placement::to_json(p.value()));
      CLM_FUZZ_CHECK(again.is_ok());
    }
  } else {
    auto n = placement::network_profile_from_json(text);
    if (n.is_ok()) {
      CLM_FUZZ_CHECK(placement::validate(n.value()).is_ok());
      auto again = placement::network_profile_from_json(placement::to_json(n.value()));
      CLM_FUZZ_CHECK(again.is_ok());
    }
  }
  return 0;
}
