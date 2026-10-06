// clusterlm-fixture-model --out DIR [--layers N] [--seed S] [--tiny]
// Writes the deterministic fixture shards and manifest.json. Output is synthetic and redistributable.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "clusterlm/objects/fixture_model.hpp"

int main(int argc, char** argv) {
  using namespace clusterlm;
  objects::FixtureSpec spec;
  std::string out;
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
    } else {
      std::fprintf(stderr, "usage: clusterlm-fixture-model --out DIR [--layers N] [--seed S] [--tiny]\n");
      return 2;
    }
  }
  if (out.empty()) {
    std::fprintf(stderr, "usage: clusterlm-fixture-model --out DIR [--layers N] [--seed S] [--tiny]\n");
    return 2;
  }
  auto m = objects::write_fixture_model(spec, out);
  if (!m.is_ok()) {
    std::fprintf(stderr, "error: %s\n", m.status().to_string().c_str());
    return 1;
  }
  std::printf("wrote %zu objects, %zu shards to %s\nroot_hash %s\n", m->objects.size(), m->shards.size(), out.c_str(),
              m->root_hash().hex().c_str());
  return 0;
}
