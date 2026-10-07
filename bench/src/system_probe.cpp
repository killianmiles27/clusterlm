#include "system_probe.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>

#include "clusterlm/common/clock.hpp"
#include "nvml_probe.hpp"

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <iphlpapi.h>
#include <netioapi.h>
#include <powrprof.h>
#include <psapi.h>
#else
#include <unistd.h>
#endif

namespace clusterlm::bench {

namespace fs = std::filesystem;

namespace {

#if !defined(_WIN32)
std::string read_text_file(const std::string& path) {
  std::ifstream f(path);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}
#endif

bool parse_kb_line(const std::string& line, const char* key, std::uint64_t& out) {
  const std::string k = std::string(key) + ":";
  if (line.rfind(k, 0) != 0) return false;
  std::istringstream ls(line.substr(k.size()));
  std::uint64_t kb = 0;
  if (!(ls >> kb)) return false;
  out = kb * 1024;
  return true;
}

bool parse_ipv4(const std::string& text, std::uint32_t& out) {
  unsigned a = 0, b = 0, c = 0, d = 0;
  char tail = 0;
  if (std::sscanf(text.c_str(), "%u.%u.%u.%u%c", &a, &b, &c, &d, &tail) != 4) return false;
  if (a > 255 || b > 255 || c > 255 || d > 255) return false;
  out = a | (b << 8) | (c << 16) | (d << 24);  // the byte order /proc/net/route prints on little-endian hosts
  return true;
}

}  // namespace

// ---- process memory ---------------------------------------------------------------------------------------

Result<ProcessMemory> parse_proc_status_memory(const std::string& text) {
  ProcessMemory m;
  bool have_rss = false;
  std::istringstream in(text);
  std::string line;
  while (std::getline(in, line)) {
    if (parse_kb_line(line, "VmRSS", m.rss_bytes)) have_rss = true;
    else parse_kb_line(line, "VmData", m.commit_bytes);
  }
  if (!have_rss) return make_error(ErrorCode::kDataLoss, "no VmRSS in process status (kernel thread or exited process)");
  return m;
}

Result<ProcessMemory> sample_process_memory(std::int64_t pid) {
#if defined(_WIN32)
  HANDLE h = pid <= 0 ? GetCurrentProcess()
                      : OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, static_cast<DWORD>(pid));
  if (h == nullptr) return make_error(ErrorCode::kNotFound, "OpenProcess(" + std::to_string(pid) + ") failed: " + std::to_string(GetLastError()));
  PROCESS_MEMORY_COUNTERS_EX pmc{};
  pmc.cb = sizeof pmc;
  const BOOL ok = GetProcessMemoryInfo(h, reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), sizeof pmc);
  const DWORD err = ok ? 0 : GetLastError();
  if (pid > 0) CloseHandle(h);
  if (!ok) return make_error(ErrorCode::kUnavailable, "GetProcessMemoryInfo failed: " + std::to_string(err));
  ProcessMemory m;
  m.rss_bytes = pmc.WorkingSetSize;
  m.commit_bytes = pmc.PrivateUsage;
  return m;
#else
  const std::string path = pid <= 0 ? "/proc/self/status" : "/proc/" + std::to_string(pid) + "/status";
  std::ifstream f(path);
  if (!f) return make_error(ErrorCode::kNotFound, "cannot read " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  return parse_proc_status_memory(ss.str());
#endif
}

// ---- NIC counters -----------------------------------------------------------------------------------------

Result<std::vector<NicCounters>> parse_proc_net_dev(const std::string& text) {
  std::vector<NicCounters> out;
  std::istringstream in(text);
  std::string line;
  int header = 0;
  while (std::getline(in, line)) {
    if (header < 2) {
      ++header;
      continue;
    }
    const auto colon = line.find(':');
    if (colon == std::string::npos) continue;
    NicCounters c;
    std::string name = line.substr(0, colon);
    name.erase(0, name.find_first_not_of(' '));
    c.name = name;
    std::istringstream ls(line.substr(colon + 1));
    std::uint64_t v[16] = {};
    for (auto& x : v)
      if (!(ls >> x)) return make_error(ErrorCode::kDataLoss, "malformed /proc/net/dev line for " + name);
    // rx: bytes packets errs drop fifo frame compressed multicast | tx: bytes packets errs drop ...
    c.rx_bytes = v[0], c.rx_packets = v[1], c.rx_errors = v[2], c.rx_dropped = v[3];
    c.tx_bytes = v[8], c.tx_packets = v[9], c.tx_errors = v[10], c.tx_dropped = v[11];
    out.push_back(std::move(c));
  }
  return out;
}

Result<std::string> parse_route_interface(const std::string& text, const std::string& ipv4) {
  std::uint32_t addr = 0;
  if (!parse_ipv4(ipv4, addr)) return make_error(ErrorCode::kInvalidArgument, "'" + ipv4 + "' is not an IPv4 literal");
  if ((addr & 0xFFu) == 127u) return std::string("lo");
  std::istringstream in(text);
  std::string line;
  std::getline(in, line);  // header
  std::string best;
  int best_prefix = -1;
  unsigned best_metric = ~0u;
  while (std::getline(in, line)) {
    std::istringstream ls(line);
    std::string iface, dest_s, gw_s, flags_s, refcnt, use, metric_s, mask_s;
    if (!(ls >> iface >> dest_s >> gw_s >> flags_s >> refcnt >> use >> metric_s >> mask_s)) continue;
    unsigned long dest = 0, mask = 0, flags = 0, metric = 0;
    try {
      dest = std::stoul(dest_s, nullptr, 16);
      mask = std::stoul(mask_s, nullptr, 16);
      flags = std::stoul(flags_s, nullptr, 16);
      metric = std::stoul(metric_s);
    } catch (const std::exception&) {
      continue;
    }
    if ((flags & 0x1u) == 0) continue;  // RTF_UP
    if ((addr & static_cast<std::uint32_t>(mask)) != static_cast<std::uint32_t>(dest)) continue;
    int prefix = 0;
    for (std::uint32_t m = static_cast<std::uint32_t>(mask); m != 0; m &= m - 1) ++prefix;
    if (prefix > best_prefix || (prefix == best_prefix && metric < best_metric)) {
      best = iface;
      best_prefix = prefix;
      best_metric = static_cast<unsigned>(metric);
    }
  }
  if (best.empty()) return make_error(ErrorCode::kNotFound, "no route to " + ipv4);
  return best;
}

#if defined(_WIN32)
namespace {
std::string narrow_ascii(const wchar_t* w) {
  std::string s;
  for (; *w; ++w) s.push_back(*w < 0x80 ? static_cast<char>(*w) : '?');
  return s;
}

NicCounters counters_from_row(const MIB_IF_ROW2& row) {
  NicCounters c;
  c.name = narrow_ascii(row.Alias);
  c.rx_bytes = row.InOctets;
  c.tx_bytes = row.OutOctets;
  c.rx_packets = row.InUcastPkts + row.InNUcastPkts;
  c.tx_packets = row.OutUcastPkts + row.OutNUcastPkts;
  c.rx_errors = row.InErrors;
  c.tx_errors = row.OutErrors;
  c.rx_dropped = row.InDiscards;
  c.tx_dropped = row.OutDiscards;
  return c;
}
}  // namespace
#endif

std::vector<std::string> list_nics() {
  std::vector<std::string> out;
#if defined(_WIN32)
  PMIB_IF_TABLE2 table = nullptr;
  if (GetIfTable2(&table) == NO_ERROR && table != nullptr) {
    for (ULONG i = 0; i < table->NumEntries; ++i) out.push_back(narrow_ascii(table->Table[i].Alias));
    FreeMibTable(table);
  }
#else
  if (auto parsed = parse_proc_net_dev(read_text_file("/proc/net/dev")); parsed.is_ok())
    for (const auto& c : parsed.value()) out.push_back(c.name);
#endif
  return out;
}

Result<NicCounters> read_nic_counters(const std::string& name) {
#if defined(_WIN32)
  PMIB_IF_TABLE2 table = nullptr;
  if (const ULONG rc = GetIfTable2(&table); rc != NO_ERROR || table == nullptr)
    return make_error(ErrorCode::kUnavailable, "GetIfTable2 failed: " + std::to_string(rc));
  Result<NicCounters> result = make_error(ErrorCode::kNotFound, "no network interface named '" + name + "'");
  for (ULONG i = 0; i < table->NumEntries; ++i) {
    const MIB_IF_ROW2& row = table->Table[i];
    // GetIfTable2 returns the rows; refresh the counters of the matching one with GetIfEntry2.
    if (narrow_ascii(row.Alias) != name && narrow_ascii(row.Description) != name &&
        std::to_string(row.InterfaceIndex) != name)
      continue;
    MIB_IF_ROW2 fresh{};
    fresh.InterfaceIndex = row.InterfaceIndex;
    fresh.InterfaceLuid = row.InterfaceLuid;
    result = GetIfEntry2(&fresh) == NO_ERROR ? Result<NicCounters>(counters_from_row(fresh))
                                             : Result<NicCounters>(counters_from_row(row));
    break;
  }
  FreeMibTable(table);
  return result;
#else
  std::ifstream f("/proc/net/dev");
  if (!f) return make_error(ErrorCode::kUnavailable, "cannot read /proc/net/dev");
  std::stringstream ss;
  ss << f.rdbuf();
  CLM_ASSIGN_OR_RETURN(auto all, parse_proc_net_dev(ss.str()));
  for (auto& c : all)
    if (c.name == name) return std::move(c);
  return make_error(ErrorCode::kNotFound, "no network interface named '" + name + "'");
#endif
}

Result<std::string> default_nic_for(const std::string& requested) {
  const std::string host = requested == "localhost" ? std::string("127.0.0.1") : requested;
#if defined(_WIN32)
  sockaddr_in sa{};
  sa.sin_family = AF_INET;
  if (InetPtonA(AF_INET, host.c_str(), &sa.sin_addr) != 1)
    return make_error(ErrorCode::kInvalidArgument, "'" + host + "' is not an IPv4 literal");
  DWORD index = 0;
  if (const DWORD rc = GetBestInterfaceEx(reinterpret_cast<sockaddr*>(&sa), &index); rc != NO_ERROR)
    return make_error(ErrorCode::kNotFound, "GetBestInterfaceEx(" + host + ") failed: " + std::to_string(rc));
  MIB_IF_ROW2 row{};
  row.InterfaceIndex = index;
  if (GetIfEntry2(&row) != NO_ERROR)
    return make_error(ErrorCode::kNotFound, "GetIfEntry2 failed for interface " + std::to_string(index));
  return narrow_ascii(row.Alias);
#else
  std::ifstream f("/proc/net/route");
  if (!f) return make_error(ErrorCode::kUnavailable, "cannot read /proc/net/route");
  std::stringstream ss;
  ss << f.rdbuf();
  return parse_route_interface(ss.str(), host);
#endif
}

NicDelta diff_nic(const NicCounters& before, const NicCounters& after, bool* wrapped) {
  NicDelta d;
  d.name = after.name;
  auto sub = [&](std::uint64_t a, std::uint64_t b) -> std::uint64_t {
    if (a >= b) return a - b;
    if (wrapped) *wrapped = true;
    return 0;
  };
  d.rx_bytes = sub(after.rx_bytes, before.rx_bytes);
  d.tx_bytes = sub(after.tx_bytes, before.tx_bytes);
  d.rx_packets = sub(after.rx_packets, before.rx_packets);
  d.tx_packets = sub(after.tx_packets, before.tx_packets);
  d.rx_errors = sub(after.rx_errors, before.rx_errors);
  d.tx_errors = sub(after.tx_errors, before.tx_errors);
  d.rx_dropped = sub(after.rx_dropped, before.rx_dropped);
  d.tx_dropped = sub(after.tx_dropped, before.tx_dropped);
  return d;
}

NicWindow::NicWindow(std::string nic, const std::string& first_peer_host) : nic_(std::move(nic)) {
  if (nic_.empty()) {
    if (first_peer_host.empty()) {
      unavailable_ = "no Node endpoint to route to and no --nic given";
      return;
    }
    auto d = default_nic_for(first_peer_host);
    if (!d.is_ok()) {
      unavailable_ = "interface of the route to " + first_peer_host + ": " + d.status().message();
      return;
    }
    nic_ = d.value();
  }
  auto probe = read_nic_counters(nic_);
  if (!probe.is_ok()) unavailable_ = "counters of '" + nic_ + "': " + probe.status().message();
}

void NicWindow::begin() {
  before_.reset();
  delta_ = {};
  if (!available()) return;
  auto c = read_nic_counters(nic_);
  if (c.is_ok()) before_ = std::move(c).value();
  else unavailable_ = c.status().message();
}

void NicWindow::end() {
  if (!available() || !before_) return;
  auto c = read_nic_counters(nic_);
  if (!c.is_ok()) {
    unavailable_ = c.status().message();
    return;
  }
  delta_ = diff_nic(*before_, c.value(), &wrapped_);
}

void NicWindow::emit(BenchmarkResult& r, const std::string& prefix) const {
  const std::string k = prefix + "nic.";
  if (!available()) {
    r.metric(k + "unavailable", "unavailable: " + unavailable_);
    return;
  }
  r.metric(k + "name", nic_);
  r.metric(k + "rx_bytes", delta_.rx_bytes);
  r.metric(k + "tx_bytes", delta_.tx_bytes);
  r.metric(k + "rx_packets", delta_.rx_packets);
  r.metric(k + "tx_packets", delta_.tx_packets);
  r.metric(k + "rx_errors", delta_.rx_errors);
  r.metric(k + "tx_errors", delta_.tx_errors);
  r.metric(k + "rx_dropped", delta_.rx_dropped);
  r.metric(k + "tx_dropped", delta_.tx_dropped);
  r.metric(k + "counter_wrapped", wrapped_);
}

// ---- power plan / governor --------------------------------------------------------------------------------

std::string summarize_governors(const std::vector<std::string>& per_cpu) {
  std::vector<std::string> distinct;
  for (const auto& g : per_cpu)
    if (std::find(distinct.begin(), distinct.end(), g) == distinct.end()) distinct.push_back(g);
  std::sort(distinct.begin(), distinct.end());
  if (distinct.empty()) return {};
  if (distinct.size() == 1) return distinct.front();
  std::string out = "mixed:";
  for (std::size_t i = 0; i < distinct.size(); ++i) out += (i ? "," : "") + distinct[i];
  return out;
}

std::string PowerEnvironment::summary() const {
  std::string s;
  if (plan_name || plan_guid) s += "plan '" + plan_name.value_or("?") + "'" + (plan_guid ? " " + *plan_guid : "");
  if (governor) s += (s.empty() ? "" : "; ") + std::string("governor '") + *governor + "'";
  if (s.empty()) return "unavailable: " + unavailable;
  if (!unavailable.empty()) s += " (" + unavailable + ")";
  return s;
}

PowerEnvironment probe_power_environment() {
  PowerEnvironment p;
#if defined(_WIN32)
  GUID* scheme = nullptr;
  if (const DWORD rc = PowerGetActiveScheme(nullptr, &scheme); rc != ERROR_SUCCESS || scheme == nullptr) {
    p.unavailable = "PowerGetActiveScheme failed: " + std::to_string(rc);
    return p;
  }
  char guid[48];
  std::snprintf(guid, sizeof guid, "{%08lx-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x}", static_cast<unsigned long>(scheme->Data1),
                static_cast<unsigned>(scheme->Data2), static_cast<unsigned>(scheme->Data3), scheme->Data4[0], scheme->Data4[1],
                scheme->Data4[2], scheme->Data4[3], scheme->Data4[4], scheme->Data4[5], scheme->Data4[6], scheme->Data4[7]);
  p.plan_guid = guid;
  UCHAR buffer[512] = {};
  DWORD size = sizeof buffer;
  if (PowerReadFriendlyName(nullptr, scheme, nullptr, nullptr, buffer, &size) == ERROR_SUCCESS) {
    std::string name;
    const auto* w = reinterpret_cast<const wchar_t*>(buffer);
    for (std::size_t i = 0; i < size / sizeof(wchar_t) && w[i] != 0; ++i) name.push_back(w[i] < 0x80 ? static_cast<char>(w[i]) : '?');
    p.plan_name = name;
  } else {
    p.unavailable = "PowerReadFriendlyName failed (plan GUID recorded)";
  }
  LocalFree(scheme);
#else
  std::vector<std::string> governors;
  std::error_code ec;
  for (const auto& e : fs::directory_iterator("/sys/devices/system/cpu", ec)) {
    const std::string n = e.path().filename().string();
    if (n.size() < 4 || n.rfind("cpu", 0) != 0 || n.find_first_not_of("0123456789", 3) != std::string::npos) continue;
    std::ifstream f(e.path() / "cpufreq" / "scaling_governor");
    std::string g;
    if (f && std::getline(f, g) && !g.empty()) governors.push_back(g);
  }
  if (governors.empty()) p.unavailable = "cpufreq scaling_governor is not exposed on this host (virtual machine or no cpufreq driver)";
  else p.governor = summarize_governors(governors);
#endif
  return p;
}

void emit_power_environment(BenchmarkResult& r, const PowerEnvironment& p) {
  if (p.plan_guid) r.metric("power.plan_guid", *p.plan_guid);
  if (p.plan_name) r.metric("power.plan_name", *p.plan_name);
  if (p.governor) r.metric("power.governor", *p.governor);
  if (!p.unavailable.empty()) r.metric("power.environment_unavailable", "unavailable: " + p.unavailable);
}

// ---- ResourceSampler --------------------------------------------------------------------------------------

void ResourceSampler::add_process(const std::string& label, std::function<std::int64_t()> pid) {
  std::lock_guard lock(mu_);
  procs_.push_back({label, std::move(pid), {}});
}

void ResourceSampler::record_locked(const std::string& phase) {
  if (start_ns_ == 0) start_ns_ = monotonic_ns();
  Point pt;
  pt.t_s = static_cast<double>(monotonic_ns() - start_ns_) * 1e-9;
  pt.phase = phase;
  for (auto& p : procs_) {
    auto m = sample_process_memory(p.pid ? p.pid() : 0);
    if (m.is_ok()) {
      pt.memory.emplace_back(m.value());
    } else {
      pt.memory.emplace_back(std::nullopt);
      p.last_error = m.status().message();
    }
  }
  if (nvml_ != nullptr) {
    auto g = nvml_->sample();
    if (g.is_ok()) {
      if (gpu_names_.empty()) gpu_names_ = nvml_->device_names();
      for (const auto& s : g.value()) pt.vram_used.push_back(s.mem_used_bytes);
    } else {
      gpu_error_ = g.status().message();
    }
  }
  last_t_ms_ = pt.t_s * 1000.0;
  points_.push_back(std::move(pt));
}

void ResourceSampler::sample(const std::string& phase) {
  std::lock_guard lock(mu_);
  record_locked(phase);
}

void ResourceSampler::sample_throttled(const std::string& phase, double min_interval_ms) {
  std::lock_guard lock(mu_);
  const double now_ms = start_ns_ == 0 ? 1e18 : static_cast<double>(monotonic_ns() - start_ns_) * 1e-6;
  if (start_ns_ != 0 && now_ms - last_t_ms_ < min_interval_ms) return;
  record_locked(phase);
}

std::size_t ResourceSampler::samples() const {
  std::lock_guard lock(mu_);
  return points_.size();
}

std::vector<ResourceSampler::Point> ResourceSampler::points() const {
  std::lock_guard lock(mu_);
  return points_;
}

void ResourceSampler::emit(BenchmarkResult& r, const std::string& prefix) const {
  std::lock_guard lock(mu_);
  r.metric(prefix + "samples", points_.size());
  const std::size_t n = points_.size();
  const std::size_t stride = n > kMaxSeriesPoints ? (n + kMaxSeriesPoints - 1) / kMaxSeriesPoints : 1;
  r.metric(prefix + "series_stride", stride);
  nlohmann::json t = nlohmann::json::array(), phase = nlohmann::json::array();
  for (std::size_t i = 0; i < n; i += stride) {
    t.push_back(points_[i].t_s);
    phase.push_back(points_[i].phase);
  }
  r.metric(prefix + "t_s.series", t);
  r.metric(prefix + "phase.series", phase);
  for (std::size_t pi = 0; pi < procs_.size(); ++pi) {
    const std::string k = prefix + procs_[pi].label + ".";
    Distribution rss, commit;
    nlohmann::json rss_series = nlohmann::json::array(), commit_series = nlohmann::json::array();
    std::optional<std::uint64_t> rss_first, rss_last, commit_first, commit_last;
    for (std::size_t i = 0; i < n; ++i) {
      const auto& m = points_[i].memory[pi];
      if (!m) continue;
      rss.add(static_cast<double>(m->rss_bytes));
      commit.add(static_cast<double>(m->commit_bytes));
      if (!rss_first) rss_first = m->rss_bytes, commit_first = m->commit_bytes;
      rss_last = m->rss_bytes;
      commit_last = m->commit_bytes;
    }
    for (std::size_t i = 0; i < n; i += stride) {
      const auto& m = points_[i].memory[pi];
      // A failed read inside the series is null, not zero.
      rss_series.push_back(m ? nlohmann::json(m->rss_bytes) : nlohmann::json(nullptr));
      commit_series.push_back(m ? nlohmann::json(m->commit_bytes) : nlohmann::json(nullptr));
    }
    if (rss.empty()) {
      r.metric(k + "unavailable", "unavailable: " + (procs_[pi].last_error.empty() ? std::string("no sample") : procs_[pi].last_error));
      continue;
    }
    r.metric(k + "rss_bytes", rss, "bytes");
    r.metric(k + "rss_bytes.series", rss_series);
    r.metric(k + "commit_bytes", commit, "bytes");
    r.metric(k + "commit_bytes.series", commit_series);
    r.metric(k + "rss_growth_bytes", static_cast<std::int64_t>(*rss_last) - static_cast<std::int64_t>(*rss_first));
    r.metric(k + "commit_growth_bytes", static_cast<std::int64_t>(*commit_last) - static_cast<std::int64_t>(*commit_first));
  }
  if (nvml_ != nullptr) {
    if (gpu_names_.empty()) {
      r.metric(prefix + "gpu.unavailable", "unavailable: " + (gpu_error_.empty() ? std::string("no NVML sample") : gpu_error_));
    } else {
      for (std::size_t g = 0; g < gpu_names_.size(); ++g) {
        Distribution used;
        nlohmann::json series = nlohmann::json::array();
        for (std::size_t i = 0; i < n; ++i)
          if (g < points_[i].vram_used.size() && points_[i].vram_used[g]) used.add(static_cast<double>(*points_[i].vram_used[g]));
        for (std::size_t i = 0; i < n; i += stride)
          series.push_back(g < points_[i].vram_used.size() && points_[i].vram_used[g] ? nlohmann::json(*points_[i].vram_used[g])
                                                                                      : nlohmann::json(nullptr));
        const std::string k = prefix + "gpu" + std::to_string(g) + ".vram_used_bytes";
        r.metric(k, used, "bytes");
        r.metric(k + ".series", series);
      }
    }
  }
}

}  // namespace clusterlm::bench
