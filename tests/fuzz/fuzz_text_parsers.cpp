// Small text parsers reachable from the command line, config files and the network: endpoint strings, the
// cluster plan language, fault-rule specs, hex digests and network presets. Byte 0 selects the parser.
#include "clusterlm/common/digest.hpp"
#include "clusterlm/coordinator/coordinator.hpp"
#include "clusterlm/transport/fault_injector.hpp"
#include "clusterlm/transport/impairment.hpp"
#include "clusterlm/transport/transport.hpp"
#include "fuzz_util.hpp"

using namespace clusterlm;

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  if (size == 0) return 0;
  const std::string_view text = fuzz::text_of(data + 1, size - 1);
  switch (data[0] % 6) {
    case 0: {
      auto e = transport::Endpoint::parse(text);
      if (e.is_ok()) {
        (void)e->is_loopback();
        auto again = transport::Endpoint::parse(e->str());
        CLM_FUZZ_CHECK(again.is_ok() && again->port == e->port);
      }
      break;
    }
    case 1: {
      auto p = coordinator::ClusterPlan::parse(text, 1 + data[0] / 6);
      if (p.is_ok()) (void)p->describe();
      break;
    }
    case 2: (void)transport::parse_fault_rule(text); break;
    case 3: (void)Digest256::from_hex(text); break;
    case 4: (void)from_hex(text); break;
    default: (void)transport::network_preset(text); break;
  }
  return 0;
}
