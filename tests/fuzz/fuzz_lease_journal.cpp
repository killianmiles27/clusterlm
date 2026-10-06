// Lease journal replay (orphan recovery reads this file before any lease is accepted). The journal is
// attacker-influenced only by a local user who can write the staging root, but it drives file deletion, so every
// name it yields must be a store-generated one.
#include <set>

#include "clusterlm/lease/journal.hpp"
#include "fuzz_util.hpp"

using namespace clusterlm;

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  const lease::JournalReplay r = lease::Journal::parse(fuzz::text_of(data, size));
  std::set<std::uint64_t> seen;
  for (const auto& l : r.leases) {
    CLM_FUZZ_CHECK(seen.insert(l.generation).second);  // generations are unique
    CLM_FUZZ_CHECK(l.generation <= r.max_generation);
    for (const std::string& f : l.files) {
      CLM_FUZZ_CHECK(lease::is_valid_lease_file_name(f));
      CLM_FUZZ_CHECK(f.find('/') == std::string::npos && f.find('\\') == std::string::npos &&
                     f.find("..") == std::string::npos);
    }
  }
  return 0;
}
