// libFuzzer target: GGUF header parser and manifest builder.
//
// The input is the head of a GGUF file. The parser is run twice: once against exactly the input bytes (every
// truncation is a parse error) and once against the input followed by a large virtual zero-filled data region,
// so small header-only inputs can describe tensors whose ranges lie "inside the file" and reach the manifest
// builder. Accepted files must satisfy the reader's invariants; the builder must never crash, and every manifest
// it produces must validate and keep its ranges inside the (virtual) file.
//
//   clang++ build:  cmake -B build-fuzz -G Ninja -DCMAKE_CXX_COMPILER=clang++ -DCLUSTERLM_FUZZ=ON
//                   cmake --build build-fuzz --target fuzz_gguf gen_gguf_seeds
//   seeds:          build-fuzz/bin/gen_gguf_seeds corpus/ && build-fuzz/bin/fuzz_gguf corpus/ -max_len=65536 -max_total_time=60
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "clusterlm/objects/gguf.hpp"
#include "clusterlm/objects/gguf_manifest.hpp"

using namespace clusterlm;
using namespace clusterlm::objects;

namespace {

constexpr std::uint64_t kVirtualData = 64ull << 20;

class PaddedSource final : public GgufSource {
 public:
  PaddedSource(const std::uint8_t* d, std::size_t n, std::uint64_t virtual_size) : d_(d), n_(n), size_(virtual_size) {}
  std::uint64_t size() const override { return size_; }
  Status read(std::uint64_t off, void* dst, std::size_t len) override {
    if (off > size_ || len > size_ - off) return make_error(ErrorCode::kOutOfRange, "oob");
    auto* out = static_cast<std::uint8_t*>(dst);
    const std::size_t real = off >= n_ ? 0 : static_cast<std::size_t>(std::min<std::uint64_t>(len, n_ - off));
    if (real != 0) std::memcpy(out, d_ + off, real);
    if (real < len) std::memset(out + real, 0, len - real);
    return Status::ok();
  }

 private:
  const std::uint8_t* d_;
  std::size_t n_;
  std::uint64_t size_;
};

[[noreturn]] void invariant_violated() { std::abort(); }

void check_file(const GgufFile& f) {
  if (f.data_start > f.file_size || f.data_start % f.alignment != 0 || f.header_bytes > f.data_start) invariant_violated();
  for (const GgufTensorInfo& t : f.tensors) {
    if (t.n_dims == 0 || t.n_dims > 4) invariant_violated();
    if (t.offset % f.alignment != 0) invariant_violated();
    if (t.file_offset != f.data_start + t.offset) invariant_violated();
    if (t.file_offset > f.file_size || t.n_bytes > f.file_size - t.file_offset) invariant_violated();
  }
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  GgufLimits limits;
  limits.max_header_bytes = 1u << 20;  // keep individual runs fast
  {
    auto r = parse_gguf_bytes(ByteSpan(data, size), limits);
    if (r.is_ok()) check_file(*r);
  }
  PaddedSource src(data, size, size + kVirtualData);
  auto r = parse_gguf(src, limits, "fuzz.gguf");
  if (!r.is_ok()) return 0;
  check_file(*r);

  GgufModelFiles model;
  model.paths.emplace_back("fuzz.gguf");
  model.shards.push_back(std::move(r).value());
  ManifestBuildOptions opts;
  opts.allow_unclassified = true;
  opts.limits = limits;
  auto built = build_manifest(model, opts);
  if (!built.is_ok()) return 0;
  const ModelManifest& m = built->manifest;
  if (!m.validate().is_ok()) invariant_violated();
  for (const ManifestObject& o : m.objects) {
    std::uint64_t total = 0;
    for (const SourceRange& rg : o.source_ranges) {
      if (rg.shard != 0 || rg.offset > size + kVirtualData || rg.length > size + kVirtualData - rg.offset) invariant_violated();
      total += rg.length;
    }
    if (total != o.byte_size) invariant_violated();
  }
  return 0;
}
