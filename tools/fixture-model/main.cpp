// clusterlm-fixture-model --out DIR [--layers N] [--seed S] [--tiny] [--format bin|gguf] [--f32-experts]
// Writes the deterministic fixture and its manifest.json. Output is synthetic and redistributable.
//   --format bin   (default) raw fixture shards: fixture-transformer.bin + fixture-lookup.bin
//   --format gguf  a 2-shard split GGUF with Flash-Next style names; the manifest is built through the real
//                  GGUF path (so the reference backend can run from a GGUF-derived manifest). Odd layers use
//                  ggml Q8_0 experts unless --f32-experts is given.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "clusterlm/objects/fixture_gguf.hpp"
#include "clusterlm/objects/fixture_model.hpp"

namespace {
constexpr const char* kUsage =
    "usage: clusterlm-fixture-model --out DIR [--layers N] [--seed S] [--tiny] [--format bin|gguf] [--f32-experts]\n";
}

int main(int argc, char** argv) {
  using namespace clusterlm;
  objects::FixtureSpec spec;
  objects::FixtureGgufOptions gguf_opts;
  std::string out;
  std::string format = "bin";
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto value = [&]() -> const char* { return i + 1 < argc ? argv[++i] : nullptr; };
    if (a == "--out") {
      if (const char* v = value()) out = v;
    } else if (a == "--layers") {
      const char* v = value();
      if (!v) { std::fprintf(stderr, "--layers needs a value\n"); return 2; }
      spec.n_layers = static_cast<std::uint32_t>(std::strtoul(v, nullptr, 10));
    } else if (a == "--seed") {
      const char* v = value();
      if (!v) { std::fprintf(stderr, "--seed needs a value\n"); return 2; }
      spec.seed = std::strtoull(v, nullptr, 0);
    } else if (a == "--tiny") {
      const auto layers = spec.n_layers;
      const auto seed = spec.seed;
      spec = objects::FixtureSpec::tiny();
      if (layers != 16) spec.n_layers = layers;
      spec.seed = seed;
    } else if (a == "--format") {
      const char* v = value();
      if (!v || (std::strcmp(v, "bin") != 0 && std::strcmp(v, "gguf") != 0)) { std::fprintf(stderr, "--format must be bin or gguf\n"); return 2; }
      format = v;
    } else if (a == "--f32-experts") {
      gguf_opts.q8_experts = false;
    } else {
      std::fprintf(stderr, "%s", kUsage);
      return 2;
    }
  }
  if (out.empty()) {
    std::fprintf(stderr, "%s", kUsage);
    return 2;
  }
  auto m = format == "gguf" ? objects::write_fixture_gguf(spec, out, gguf_opts) : objects::write_fixture_model(spec, out);
  if (!m.is_ok()) {
    std::fprintf(stderr, "error: %s\n", m.status().to_string().c_str());
    return 1;
  }
  std::printf("wrote %zu objects, %zu shards to %s\nroot_hash %s\n", m->objects.size(), m->shards.size(), out.c_str(),
              m->root_hash().hex().c_str());
  return 0;
}
