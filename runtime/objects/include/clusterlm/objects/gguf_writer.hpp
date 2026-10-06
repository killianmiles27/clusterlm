#pragma once
// GGUF writer for tests, fixtures and fuzz seed corpora. Writes v3 (or v2), little-endian files with
// arbitrary metadata and tensors of any ggml type. Never used to write real model weights.
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "clusterlm/common/bytes.hpp"
#include "clusterlm/common/status.hpp"
#include "clusterlm/objects/ggml_types.hpp"
#include "clusterlm/objects/gguf.hpp"

namespace clusterlm::objects {

class GgufWriter {
 public:
  explicit GgufWriter(std::uint32_t version = 3) : version_(version) {}

  void set_alignment(std::uint64_t a);  // also writes general.alignment when != 32

  void add_u16(const std::string& key, std::uint16_t v);
  void add_u32(const std::string& key, std::uint32_t v);
  void add_u64(const std::string& key, std::uint64_t v);
  void add_i32(const std::string& key, std::int32_t v);
  void add_f32(const std::string& key, float v);
  void add_bool(const std::string& key, bool v);
  void add_string(const std::string& key, const std::string& v);
  void add_u32_array(const std::string& key, const std::vector<std::uint32_t>& v);
  void add_f32_array(const std::string& key, const std::vector<float>& v);
  void add_string_array(const std::string& key, const std::vector<std::string>& v);
  // Escape hatch: pre-encoded value (type id + payload bytes) for malformed-file tests.
  void add_raw(const std::string& key, std::uint32_t type_id, ByteSpan payload);

  // `data` must be exactly ggml_tensor_bytes(type, dims). dims[0] is the contiguous axis.
  Status add_tensor(const std::string& name, std::vector<std::uint64_t> dims, GgmlType type, Bytes data);
  // Deterministic pseudo-random payload of the right size (any bit pattern is a valid quantized block for
  // layout purposes; floats may be NaN, which nothing here interprets).
  Status add_random_tensor(const std::string& name, std::vector<std::uint64_t> dims, GgmlType type, std::uint64_t seed);

  std::size_t tensor_count() const { return tensors_.size(); }
  Result<Bytes> serialize() const;
  Status write(const std::filesystem::path& path) const;

 private:
  struct Tensor {
    std::string name;
    std::vector<std::uint64_t> dims;
    GgmlType type;
    Bytes data;
  };
  void add_kv(const std::string& key, std::uint32_t type, const Bytes& payload);

  std::uint32_t version_;
  std::uint64_t alignment_ = 32;
  std::vector<std::pair<std::string, Bytes>> kv_;  // key + (u32 type | payload) already encoded
  std::vector<Tensor> tensors_;
};

// Deterministic bytes (splitmix64 stream).
Bytes pseudo_random_bytes(std::size_t n, std::uint64_t seed);

}  // namespace clusterlm::objects
