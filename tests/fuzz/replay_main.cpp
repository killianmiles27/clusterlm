// Corpus replay driver for the fuzz targets: runs LLVMFuzzerTestOneInput over every file under the given
// directories/files, then over a bounded, deterministic mutation pass derived from each seed (bit flips, byte
// overwrites, truncation, duplication, splicing two seeds). It is a regression net, not a fuzzer: a failure
// is a crash, a sanitizer report or a trap raised by the target's own invariant checks.
//
//   fuzz_replay_<name> <corpus dir or file>... [--mutations N]
//   N defaults to 48 per seed; the environment variable CLUSTERLM_FUZZ_MUTATIONS overrides the default.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size);

namespace {

using Input = std::vector<std::uint8_t>;

struct Rng {
  std::uint64_t s;
  std::uint64_t next() {  // splitmix64
    s += 0x9E3779B97F4A7C15ull;
    std::uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  std::size_t below(std::size_t n) { return n == 0 ? 0 : static_cast<std::size_t>(next() % n); }
};

Input read_file(const std::filesystem::path& p) {
  std::ifstream in(p, std::ios::binary);
  return Input(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

void run(const Input& in) { (void)LLVMFuzzerTestOneInput(in.data(), in.size()); }

Input mutate(const Input& base, const Input& other, Rng& rng) {
  Input m = base;
  const int rounds = 1 + static_cast<int>(rng.below(3));
  for (int r = 0; r < rounds; ++r) {
    switch (rng.below(7)) {
      case 0:
        if (!m.empty()) m[rng.below(m.size())] ^= static_cast<std::uint8_t>(1u << rng.below(8));
        break;
      case 1:
        if (!m.empty()) m[rng.below(m.size())] = static_cast<std::uint8_t>(rng.next());
        break;
      case 2:
        if (!m.empty()) m.resize(rng.below(m.size()));
        break;
      case 3:  // insert a random byte
        m.insert(m.begin() + static_cast<std::ptrdiff_t>(rng.below(m.size() + 1)), static_cast<std::uint8_t>(rng.next()));
        break;
      case 4: {  // overwrite a 4-byte window with an interesting integer (length/count fields)
        static const std::uint32_t kInteresting[] = {0,          1,          0x7F,       0x80,       0xFF,
                                                     0x100,      0xFFFF,     0x10000,    0x7FFFFFFFu, 0x80000000u,
                                                     0xFFFFFFFFu, 0x00100000u, 0x04000000u};
        if (m.size() >= 4) {
          const std::size_t at = rng.below(m.size() - 3);
          const std::uint32_t v = kInteresting[rng.below(sizeof kInteresting / sizeof kInteresting[0])];
          for (int i = 0; i < 4; ++i) m[at + static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(v >> (8 * i));
        }
        break;
      }
      case 5:  // splice the tail of another seed
        if (!other.empty()) {
          m.resize(rng.below(m.size() + 1));
          const std::size_t from = rng.below(other.size());
          m.insert(m.end(), other.begin() + static_cast<std::ptrdiff_t>(from), other.end());
        }
        break;
      default:  // duplicate a chunk
        if (!m.empty()) {
          const std::size_t at = rng.below(m.size());
          const std::size_t n = 1 + rng.below(std::min<std::size_t>(64, m.size() - at));
          Input chunk(m.begin() + static_cast<std::ptrdiff_t>(at), m.begin() + static_cast<std::ptrdiff_t>(at + n));
          m.insert(m.begin() + static_cast<std::ptrdiff_t>(at), chunk.begin(), chunk.end());
        }
        break;
    }
    if (m.size() > (1u << 20)) m.resize(1u << 20);
  }
  return m;
}

}  // namespace

int main(int argc, char** argv) {
  std::size_t mutations = 48;
#ifdef _MSC_VER
  char* env = nullptr;
  std::size_t env_len = 0;
  if (_dupenv_s(&env, &env_len, "CLUSTERLM_FUZZ_MUTATIONS") == 0 && env != nullptr) {
    mutations = static_cast<std::size_t>(std::strtoull(env, nullptr, 10));
  }
  std::free(env);
#else
  if (const char* env = std::getenv("CLUSTERLM_FUZZ_MUTATIONS"))
    mutations = static_cast<std::size_t>(std::strtoull(env, nullptr, 10));
#endif
  std::vector<std::filesystem::path> files;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--mutations" && i + 1 < argc) {
      mutations = static_cast<std::size_t>(std::strtoull(argv[++i], nullptr, 10));
      continue;
    }
    std::error_code ec;
    if (std::filesystem::is_directory(a, ec)) {
      for (const auto& e : std::filesystem::recursive_directory_iterator(a, ec))
        if (e.is_regular_file()) files.push_back(e.path());
    } else {
      files.emplace_back(a);
    }
  }
  std::sort(files.begin(), files.end());
  if (files.empty()) {
    std::fprintf(stderr, "no corpus files found\n");
    return 2;
  }
  std::vector<Input> seeds;
  for (const auto& f : files) seeds.push_back(read_file(f));
  run(Input{});  // the empty input is always interesting
  for (const Input& s : seeds) run(s);
  Rng rng{0xC1A57E41ull};
  std::size_t executed = seeds.size() + 1;
  for (std::size_t i = 0; i < seeds.size(); ++i) {
    for (std::size_t k = 0; k < mutations; ++k) {
      run(mutate(seeds[i], seeds[rng.below(seeds.size())], rng));
      ++executed;
    }
  }
  std::printf("replayed %zu seed(s), %zu execution(s)\n", seeds.size(), executed);
  return 0;
}
