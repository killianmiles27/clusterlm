// Crash helper: opens a store at argv[1], begins a lease, stages RAM + disk objects and kills the process
// with std::_Exit(3) (no destructors, no cleanup) when the named phase (argv[2]) is reached.
//
// Exit codes: 3 = crashed at the requested phase, 0 = ran to completion without reaching it, 2 = setup error.
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string_view>

#include "clusterlm/lease/lease_store.hpp"

using namespace clusterlm;
using namespace clusterlm::lease;

namespace {

int fail(const char* what, const Status& st) {
  std::cerr << "helper setup error: " << what << ": " << st.to_string() << "\n";
  return 2;
}

Bytes pattern(std::size_t n, std::uint8_t seed) {
  Bytes b(n);
  std::uint32_t x = seed;
  for (auto& v : b) {
    x = x * 1664525u + 1013904223u;
    v = static_cast<std::uint8_t>(x >> 24);
  }
  return b;
}

Status stage(LeaseStore& store, std::uint32_t index, const Bytes& data, Placement placement) {
  auto w = store.create_object(index, data.size(), Sha256::of(ByteSpan(data)), placement);
  if (!w.is_ok()) return w.status();
  constexpr std::size_t kChunk = 8192;
  for (std::size_t off = 0; off < data.size(); off += kChunk) {
    ByteSpan span(data.data() + off, std::min(kChunk, data.size() - off));
    CLM_RETURN_IF_ERROR((*w)->write_chunk(off, span, Sha256::of(span)));
  }
  return (*w)->seal();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: lease_store_crash_helper <root> <phase>\n";
    return 2;
  }
  const std::string phase = argv[2];
  LeaseStoreOptions opts;
  opts.crash_hook = [phase](std::string_view p) {
    if (p == phase) std::_Exit(3);
  };
  auto store = LeaseStore::open(argv[1], opts);
  if (!store.is_ok()) return fail("open", store.status());
  LeaseStore& s = **store;
  // Generation must exceed anything already journaled so repeated runs on one root work.
  if (auto st = s.begin_lease(LeaseGeneration(s.last_generation() + 1), LeaseBudget{1 << 20, 1 << 22}); !st.is_ok())
    return fail("begin_lease", st);
  if (auto st = stage(s, 0, pattern(50000, 1), Placement::kRam); !st.is_ok()) return fail("stage ram", st);
  if (auto st = stage(s, 1, pattern(300000, 2), Placement::kDisk); !st.is_ok()) return fail("stage disk 1", st);
  if (auto st = stage(s, 2, pattern(70000, 3), Placement::kDisk); !st.is_ok()) return fail("stage disk 2", st);
  s.notify_phase("allocation");
  if (auto st = s.mark_ready(); !st.is_ok()) return fail("mark_ready", st);
  s.notify_phase("inference");
  (void)s.release();
  return 0;
}
