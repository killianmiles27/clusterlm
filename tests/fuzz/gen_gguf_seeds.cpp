// gen_gguf_seeds <out-dir>: writes the libFuzzer seed corpus for fuzz_gguf. Seeds are the HEADER REGION ONLY of
// writer-produced GGUF files (tensor data cut off), because the fuzz harness pads the file with virtual zeros.
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include "../gguf/mini_model_core.hpp"

using namespace clusterlm;
using namespace clusterlm::objects;
using namespace clusterlm::testutil;

namespace {

int g_written = 0;

void emit(const std::filesystem::path& dir, const std::string& name, const GgufWriter& w) {
  auto full = w.serialize();
  if (!full.is_ok()) {
    std::fprintf(stderr, "seed %s: %s\n", name.c_str(), full.status().to_string().c_str());
    std::exit(1);
  }
  auto parsed = parse_gguf_bytes(*full);
  if (!parsed.is_ok()) {
    std::fprintf(stderr, "seed %s does not parse: %s\n", name.c_str(), parsed.status().to_string().c_str());
    std::exit(1);
  }
  std::ofstream f(dir / (name + ".gguf"), std::ios::binary | std::ios::trunc);
  f.write(reinterpret_cast<const char*>(full->data()), static_cast<std::streamsize>(parsed->data_start));
  ++g_written;
}

void emit_mini(const std::filesystem::path& dir, const std::string& name, const MiniSpec& s) {
  GgufWriter main, lookup;
  const Status st = build_mini(s, main, lookup);
  if (!st.is_ok()) {
    std::fprintf(stderr, "seed %s: %s\n", name.c_str(), st.to_string().c_str());
    std::exit(1);
  }
  emit(dir, name, main);
  if (s.split) emit(dir, name + "-shard2", lookup);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: gen_gguf_seeds <out-dir>\n");
    return 2;
  }
  const std::filesystem::path dir = argv[1];
  std::filesystem::create_directories(dir);

  emit_mini(dir, "mini-flashnext", MiniSpec{});
  MiniSpec fused;
  fused.fused_gate_up = true;
  fused.alignment = 64;
  emit_mini(dir, "mini-fused-aligned", fused);
  MiniSpec split;
  split.split = true;
  split.layers = 8;
  split.type_for = [](std::uint32_t L, const char* w) { return (L == 2 && std::string(w) == "down") ? GgmlType::kIQ4_XS : GgmlType::kIQ3_S; };
  emit_mini(dir, "mini-split-mixed", split);
  MiniSpec no_shared;
  no_shared.shared_ff = 0;
  no_shared.with_mtp = false;
  emit_mini(dir, "mini-no-shared-no-mtp", no_shared);

  GgufWriter rich;
  rich.add_string("general.architecture", "test");
  rich.add_u32("general.alignment", 16);
  rich.add_u32_array("arr", {1, 2, 3, 4});
  rich.add_f32_array("farr", {1.f, 2.f});
  rich.add_string_array("tokens", {"a", "bb", "ccc", ""});
  rich.add_bool("flag", true);
  rich.add_f32("scale", 0.5f);
  rich.add_u64("big", 1ull << 40);
  (void)rich.add_random_tensor("a", {4, 4}, GgmlType::kF16, 1);
  (void)rich.add_random_tensor("b", {512, 3}, GgmlType::kIQ2_XXS, 1);
  emit(dir, "rich-metadata", rich);

  GgufWriter v2(2);
  v2.add_string("general.architecture", "qwen4exp");
  (void)v2.add_random_tensor("token_embd.weight", {64, 2}, GgmlType::kQ8_0, 1);
  emit(dir, "v2-minimal", v2);

  GgufWriter empty;
  emit(dir, "empty", empty);
  std::printf("wrote %d seeds to %s\n", g_written, dir.string().c_str());
  return 0;
}
