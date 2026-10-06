#include "host_probe.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <memory>
#include <numeric>
#include <thread>

#if defined(_WIN32)
#include <windows.h>
#include <immintrin.h>  // _xgetbv
#include <intrin.h>
#else
#include <sys/utsname.h>
#include <unistd.h>
#endif
#if (defined(__x86_64__) || defined(__i386__)) && !defined(_MSC_VER)
#include <cpuid.h>
#define CLM_X86 1
#elif defined(_M_X64)
#define CLM_X86 1
#endif

namespace clusterlm::bench {
namespace {

#ifdef CLM_X86
void cpuid(unsigned leaf, unsigned sub, unsigned r[4]) {
#ifdef _MSC_VER
  int regs[4];
  __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(sub));
  for (int i = 0; i < 4; ++i) r[i] = static_cast<unsigned>(regs[i]);
#else
  __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
}

std::uint64_t xgetbv0() {
#ifdef _MSC_VER
  return _xgetbv(0);
#else
  unsigned lo, hi;
  __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
  return (static_cast<std::uint64_t>(hi) << 32) | lo;
#endif
}

void detect_x86(HostInfo& h) {
  h.arch = "x86-64";
  unsigned r[4];
  cpuid(0x80000000u, 0, r);
  if (r[0] >= 0x80000004u) {
    char brand[49] = {};
    for (unsigned i = 0; i < 3; ++i) {
      cpuid(0x80000002u + i, 0, r);
      std::memcpy(brand + 16 * i, r, 16);
    }
    h.cpu_brand = brand;
    h.cpu_brand.erase(0, h.cpu_brand.find_first_not_of(' '));
  }
  cpuid(1, 0, r);
  const bool osxsave = (r[2] >> 27) & 1;
  const bool fma = (r[2] >> 12) & 1, f16c = (r[2] >> 29) & 1, avx = (r[2] >> 28) & 1;
  // A vector extension is usable only if the OS saves its register state (XCR0), not merely if CPUID lists it.
  const std::uint64_t xcr0 = osxsave ? xgetbv0() : 0;
  const bool os_avx = (xcr0 & 0x6) == 0x6;
  const bool os_avx512 = (xcr0 & 0xE6) == 0xE6;
  cpuid(7, 0, r);
  const unsigned ebx = r[1], ecx = r[2];
  auto add = [&](bool cond, const char* name) {
    if (cond) h.cpu_features.emplace_back(name);
  };
  add(avx && os_avx, "avx");
  add(((ebx >> 5) & 1) && os_avx, "avx2");
  add(fma && os_avx, "fma");
  add(f16c && os_avx, "f16c");
  add(((ebx >> 16) & 1) && os_avx512, "avx512f");
  add(((ebx >> 30) & 1) && os_avx512, "avx512bw");
  add(((ebx >> 31) & 1) && os_avx512, "avx512vl");
  add(((ebx >> 17) & 1) && os_avx512, "avx512dq");
  add(((ecx >> 11) & 1) && os_avx512, "avx512vnni");
  cpuid(7, 1, r);
  add(((r[0] >> 5) & 1) && os_avx512, "avx512bf16");
}
#endif

}  // namespace

HostInfo probe_host() {
  HostInfo h;
  h.logical_cpus = std::max(1u, std::thread::hardware_concurrency());
#ifdef CLM_X86
  detect_x86(h);
#else
  h.arch = "unknown";
#endif
#if defined(_WIN32)
  h.os = "Windows";
  MEMORYSTATUSEX ms{};
  ms.dwLength = sizeof ms;
  if (GlobalMemoryStatusEx(&ms)) {
    h.ram_bytes = ms.ullTotalPhys;
    h.ram_available_bytes = ms.ullAvailPhys;
  }
#else
  utsname u{};
  if (uname(&u) == 0) h.os = std::string(u.sysname) + " " + u.release;
  const long pages = sysconf(_SC_PHYS_PAGES), avail = sysconf(_SC_AVPHYS_PAGES), page = sysconf(_SC_PAGESIZE);
  if (pages > 0 && page > 0) h.ram_bytes = static_cast<std::uint64_t>(pages) * static_cast<std::uint64_t>(page);
  if (avail > 0 && page > 0)
    h.ram_available_bytes = static_cast<std::uint64_t>(avail) * static_cast<std::uint64_t>(page);
#endif
  return h;
}

bool has_feature(const HostInfo& h, const std::string& f) {
  return std::find(h.cpu_features.begin(), h.cpu_features.end(), f) != h.cpu_features.end();
}

double measure_read_bandwidth(std::uint64_t bytes, int repeats) {
  const std::size_t n = static_cast<std::size_t>(bytes / sizeof(std::uint64_t));
  std::unique_ptr<std::uint64_t[]> buf(new std::uint64_t[n]);
  std::iota(buf.get(), buf.get() + n, std::uint64_t{1});
  double best = 0;
  volatile std::uint64_t sink = 0;
  for (int r = 0; r < repeats; ++r) {
    const auto t0 = std::chrono::steady_clock::now();
    std::uint64_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
    for (std::size_t i = 0; i + 3 < n; i += 4) {
      acc0 += buf[i];
      acc1 += buf[i + 1];
      acc2 += buf[i + 2];
      acc3 += buf[i + 3];
    }
    sink = sink + acc0 + acc1 + acc2 + acc3;
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    best = std::max(best, static_cast<double>(n * sizeof(std::uint64_t)) / s);
  }
  return best;
}

}  // namespace clusterlm::bench
