#include "clusterlm/objects/gguf_writer.hpp"

#include <cstring>
#include <fstream>

namespace clusterlm::objects {

namespace {

template <typename U>
void put_le(Bytes& out, U v) {
  for (std::size_t i = 0; i < sizeof(U); ++i) out.push_back(static_cast<std::uint8_t>(static_cast<std::uint64_t>(v) >> (8 * i)));
}

void put_string(Bytes& out, const std::string& s) {
  put_le<std::uint64_t>(out, s.size());
  out.insert(out.end(), s.begin(), s.end());
}

std::uint64_t splitmix64(std::uint64_t& s) {
  s += 0x9E3779B97F4A7C15ull;
  std::uint64_t z = s;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

constexpr std::uint32_t kTU16 = 2, kTU32 = 4, kTI32 = 5, kTF32 = 6, kTBool = 7, kTString = 8, kTArray = 9, kTU64 = 10;

}  // namespace

Bytes pseudo_random_bytes(std::size_t n, std::uint64_t seed) {
  Bytes b(n);
  std::uint64_t s = seed;
  std::size_t i = 0;
  while (i < n) {
    std::uint64_t v = splitmix64(s);
    for (int k = 0; k < 8 && i < n; ++k, ++i) b[i] = static_cast<std::uint8_t>(v >> (8 * k));
  }
  return b;
}

void GgufWriter::add_kv(const std::string& key, std::uint32_t type, const Bytes& payload) {
  Bytes enc;
  put_string(enc, key);
  put_le<std::uint32_t>(enc, type);
  enc.insert(enc.end(), payload.begin(), payload.end());
  kv_.emplace_back(key, std::move(enc));
}

void GgufWriter::set_alignment(std::uint64_t a) {
  alignment_ = a;
  if (a != 32) add_u32("general.alignment", static_cast<std::uint32_t>(a));
}

void GgufWriter::add_u16(const std::string& key, std::uint16_t v) { Bytes p; put_le(p, v); add_kv(key, kTU16, p); }
void GgufWriter::add_u32(const std::string& key, std::uint32_t v) { Bytes p; put_le(p, v); add_kv(key, kTU32, p); }
void GgufWriter::add_u64(const std::string& key, std::uint64_t v) { Bytes p; put_le(p, v); add_kv(key, kTU64, p); }
void GgufWriter::add_i32(const std::string& key, std::int32_t v) { Bytes p; put_le(p, static_cast<std::uint32_t>(v)); add_kv(key, kTI32, p); }
void GgufWriter::add_f32(const std::string& key, float v) {
  std::uint32_t bits;
  std::memcpy(&bits, &v, sizeof bits);
  Bytes p;
  put_le(p, bits);
  add_kv(key, kTF32, p);
}
void GgufWriter::add_bool(const std::string& key, bool v) { add_kv(key, kTBool, Bytes{static_cast<std::uint8_t>(v ? 1 : 0)}); }
void GgufWriter::add_string(const std::string& key, const std::string& v) { Bytes p; put_string(p, v); add_kv(key, kTString, p); }

void GgufWriter::add_u32_array(const std::string& key, const std::vector<std::uint32_t>& v) {
  Bytes p;
  put_le<std::uint32_t>(p, kTU32);
  put_le<std::uint64_t>(p, v.size());
  for (std::uint32_t x : v) put_le(p, x);
  add_kv(key, kTArray, p);
}

void GgufWriter::add_f32_array(const std::string& key, const std::vector<float>& v) {
  Bytes p;
  put_le<std::uint32_t>(p, kTF32);
  put_le<std::uint64_t>(p, v.size());
  for (float x : v) {
    std::uint32_t bits;
    std::memcpy(&bits, &x, sizeof bits);
    put_le(p, bits);
  }
  add_kv(key, kTArray, p);
}

void GgufWriter::add_string_array(const std::string& key, const std::vector<std::string>& v) {
  Bytes p;
  put_le<std::uint32_t>(p, kTString);
  put_le<std::uint64_t>(p, v.size());
  for (const std::string& s : v) put_string(p, s);
  add_kv(key, kTArray, p);
}

void GgufWriter::add_raw(const std::string& key, std::uint32_t type_id, ByteSpan payload) {
  add_kv(key, type_id, Bytes(payload.begin(), payload.end()));
}

Status GgufWriter::add_tensor(const std::string& name, std::vector<std::uint64_t> dims, GgmlType type, Bytes data) {
  if (dims.empty() || dims.size() > 4) return make_error(ErrorCode::kInvalidArgument, "gguf writer: n_dims must be in [1,4]");
  CLM_ASSIGN_OR_RETURN(std::uint64_t want, ggml_tensor_bytes(type, dims));
  if (data.size() != want)
    return make_error(ErrorCode::kInvalidArgument, "gguf writer: tensor '" + name + "' has " + std::to_string(data.size()) +
                                                       " bytes, expected " + std::to_string(want));
  tensors_.push_back({name, std::move(dims), type, std::move(data)});
  return Status::ok();
}

Status GgufWriter::add_random_tensor(const std::string& name, std::vector<std::uint64_t> dims, GgmlType type, std::uint64_t seed) {
  CLM_ASSIGN_OR_RETURN(std::uint64_t n, ggml_tensor_bytes(type, dims));
  std::uint64_t h = seed;
  for (char c : name) h = (h ^ static_cast<std::uint8_t>(c)) * 0x100000001B3ull;
  return add_tensor(name, std::move(dims), type, pseudo_random_bytes(static_cast<std::size_t>(n), h));
}

Result<Bytes> GgufWriter::serialize() const {
  Bytes out;
  put_le<std::uint32_t>(out, 0x46554747u);
  put_le<std::uint32_t>(out, version_);
  put_le<std::uint64_t>(out, tensors_.size());
  put_le<std::uint64_t>(out, kv_.size());
  for (const auto& [key, enc] : kv_) out.insert(out.end(), enc.begin(), enc.end());

  std::vector<std::uint64_t> offsets;
  std::uint64_t cursor = 0;
  for (const Tensor& t : tensors_) {
    cursor = (cursor + alignment_ - 1) / alignment_ * alignment_;
    offsets.push_back(cursor);
    cursor += t.data.size();
  }
  for (std::size_t i = 0; i < tensors_.size(); ++i) {
    const Tensor& t = tensors_[i];
    put_string(out, t.name);
    put_le<std::uint32_t>(out, static_cast<std::uint32_t>(t.dims.size()));
    for (std::uint64_t d : t.dims) put_le(out, d);
    put_le<std::uint32_t>(out, static_cast<std::uint32_t>(t.type));
    put_le(out, offsets[i]);
  }
  out.resize((out.size() + alignment_ - 1) / alignment_ * alignment_, 0);
  const std::size_t data_start = out.size();
  for (std::size_t i = 0; i < tensors_.size(); ++i) {
    out.resize(data_start + static_cast<std::size_t>(offsets[i]), 0);
    out.insert(out.end(), tensors_[i].data.begin(), tensors_[i].data.end());
  }
  return out;
}

Status GgufWriter::write(const std::filesystem::path& path) const {
  CLM_ASSIGN_OR_RETURN(Bytes b, serialize());
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f) return make_error(ErrorCode::kUnavailable, "cannot create " + path.string());
  f.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
  f.flush();
  if (!f) return make_error(ErrorCode::kUnavailable, "write failed: " + path.string());
  return Status::ok();
}

}  // namespace clusterlm::objects
