#include "clusterlm/lease/journal.hpp"

#include <algorithm>
#include <charconv>
#include <map>

namespace clusterlm::lease {

namespace {

bool parse_u64(std::string_view s, std::uint64_t& out) {
  if (s.empty() || s.size() > 20) return false;
  for (char c : s)
    if (c < '0' || c > '9') return false;
  const auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), out);
  return ec == std::errc() && ptr == s.data() + s.size();
}

std::vector<std::string_view> split_spaces(std::string_view line) {
  std::vector<std::string_view> out;
  std::size_t i = 0;
  while (i <= line.size()) {
    const std::size_t j = line.find(' ', i);
    if (j == std::string_view::npos) {
      out.push_back(line.substr(i));
      break;
    }
    out.push_back(line.substr(i, j - i));
    i = j + 1;
  }
  return out;
}

}  // namespace

std::string lease_file_name(std::uint32_t object_index) { return "obj-" + std::to_string(object_index) + ".part"; }

bool is_valid_lease_file_name(std::string_view name) {
  constexpr std::string_view kPrefix = "obj-";
  constexpr std::string_view kSuffix = ".part";
  if (name.size() <= kPrefix.size() + kSuffix.size()) return false;
  if (name.substr(0, kPrefix.size()) != kPrefix) return false;
  if (name.substr(name.size() - kSuffix.size()) != kSuffix) return false;
  const std::string_view digits = name.substr(kPrefix.size(), name.size() - kPrefix.size() - kSuffix.size());
  std::uint64_t v = 0;
  return digits.size() <= 10 && parse_u64(digits, v) && v <= 0xFFFFFFFFull;
}

Result<JournalReplay> Journal::replay(const std::filesystem::path& path) {
  JournalReplay out;
  auto bytes = platform::read_file_bytes(path);
  if (!bytes.is_ok()) {
    if (bytes.status().code() == ErrorCode::kNotFound) return out;
    return bytes.status();
  }
  return parse(std::string_view(reinterpret_cast<const char*>(bytes->data()), bytes->size()));
}

JournalReplay Journal::parse(std::string_view text) {
  JournalReplay out;
  std::map<std::uint64_t, std::size_t> index_of;  // generation -> position in out.leases

  std::size_t pos = 0;
  while (pos < text.size()) {
    const std::size_t nl = text.find('\n', pos);
    if (nl == std::string_view::npos) {
      out.torn_tail = true;  // crash mid-append: the record never became durable, so it never happened
      break;
    }
    const std::string_view line = text.substr(pos, nl - pos);
    pos = nl + 1;
    if (line.empty()) continue;

    const auto tok = split_spaces(line);
    std::uint64_t gen = 0;
    if (tok.size() == 2 && tok[0] == "B" && parse_u64(tok[1], gen)) {
      if (index_of.count(gen) != 0) {
        ++out.corrupt_lines;
        continue;
      }
      index_of[gen] = out.leases.size();
      out.leases.push_back(JournalLease{gen, {}, false});
      out.max_generation = std::max(out.max_generation, gen);
    } else if (tok.size() == 3 && tok[0] == "F" && parse_u64(tok[1], gen) && is_valid_lease_file_name(tok[2])) {
      auto it = index_of.find(gen);
      if (it == index_of.end()) {
        ++out.corrupt_lines;
        continue;
      }
      out.leases[it->second].files.emplace_back(tok[2]);
    } else if (tok.size() == 2 && tok[0] == "R" && parse_u64(tok[1], gen)) {
      auto it = index_of.find(gen);
      if (it == index_of.end()) {
        ++out.corrupt_lines;
        continue;
      }
      out.leases[it->second].released = true;
    } else if (tok.size() == 2 && tok[0] == "G" && parse_u64(tok[1], gen)) {
      out.max_generation = std::max(out.max_generation, gen);
    } else {
      ++out.corrupt_lines;
    }
  }
  return out;
}

Result<Journal> Journal::open(const std::filesystem::path& path) {
  Journal j;
  j.path_ = path;
  // Probe the tail before opening for append: a torn final line must be terminated first.
  bool needs_newline = false;
  auto existing = platform::read_file_bytes(path);
  if (existing.is_ok()) {
    needs_newline = !existing->empty() && existing->back() != '\n';
  } else if (existing.status().code() != ErrorCode::kNotFound) {
    return existing.status();
  }
  CLM_ASSIGN_OR_RETURN(j.file_, platform::DurableAppendFile::open(path));
  if (needs_newline) CLM_RETURN_IF_ERROR(j.file_.append(std::string_view("\n")));
  return j;
}

Status Journal::append_line(const std::string& line) {
  if (!file_.is_open()) return make_error(ErrorCode::kFailedPrecondition, "journal not open");
  return file_.append(std::string_view(line));
}

Status Journal::append_begin(LeaseGeneration gen) { return append_line("B " + gen.str() + "\n"); }

Status Journal::append_file(LeaseGeneration gen, std::string_view name) {
  if (!is_valid_lease_file_name(name))
    return make_error(ErrorCode::kInvalidArgument, "journal accepts only store-generated file names");
  return append_line("F " + gen.str() + " " + std::string(name) + "\n");
}

Status Journal::append_released(LeaseGeneration gen) { return append_line("R " + gen.str() + "\n"); }

Status Journal::compact(std::uint64_t max_generation) {
  CLM_RETURN_IF_ERROR(file_.close());
  const std::string text = "G " + std::to_string(max_generation) + "\n";
  Status st = platform::write_file_atomic(
      path_, ByteSpan(reinterpret_cast<const std::uint8_t*>(text.data()), text.size()));
  // Reopen regardless so the journal stays usable after a failed compaction (old content is then intact).
  auto reopened = platform::DurableAppendFile::open(path_);
  if (!reopened.is_ok()) return st.is_ok() ? reopened.status() : st;
  file_ = std::move(reopened).value();
  return st;
}

}  // namespace clusterlm::lease
