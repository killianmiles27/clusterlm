#include "clusterlm/objects/gguf.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <set>
#include <unordered_set>

namespace clusterlm::objects {

namespace {

constexpr std::uint32_t kMagic = 0x46554747u;  // "GGUF" little-endian
constexpr std::size_t kChunkBytes = 1u << 20;

Status err(ErrorCode c, const std::string& m) { return make_error(c, "gguf: " + m); }

template <typename U>
U load_le(const std::uint8_t* p) {
  U v = 0;
  for (std::size_t i = 0; i < sizeof(U); ++i) v = static_cast<U>(v | (static_cast<U>(p[i]) << (8 * i)));
  return v;
}

// Sequential cursor over the header region with chunked, bounded reads from the source.
class Cursor {
 public:
  Cursor(GgufSource& src, const GgufLimits& limits) : src_(src), size_(src.size()), limits_(limits) {}

  std::uint64_t pos() const { return pos_; }

  Status check(std::uint64_t n) const {
    if (n > size_ - pos_) return err(ErrorCode::kOutOfRange, "truncated: need " + std::to_string(n) + " bytes at offset " +
                                                                  std::to_string(pos_) + " of a " + std::to_string(size_) + "-byte file");
    if (pos_ + n > limits_.max_header_bytes)
      return err(ErrorCode::kResourceExhausted, "header exceeds " + std::to_string(limits_.max_header_bytes) + " bytes");
    return Status::ok();
  }

  Status read(void* dst, std::size_t n) {
    CLM_RETURN_IF_ERROR(check(n));
    if (n != 0) {
      if (pos_ < buf_off_ || pos_ + n > buf_off_ + buf_.size()) {
        const std::uint64_t len = std::min<std::uint64_t>(std::max<std::uint64_t>(kChunkBytes, n), size_ - pos_);
        buf_.resize(static_cast<std::size_t>(len));
        buf_off_ = pos_;
        CLM_RETURN_IF_ERROR(src_.read(pos_, buf_.data(), buf_.size()));
      }
      std::memcpy(dst, buf_.data() + (pos_ - buf_off_), n);
    }
    pos_ += n;
    return Status::ok();
  }

  Status skip(std::uint64_t n) {
    CLM_RETURN_IF_ERROR(check(n));
    pos_ += n;
    return Status::ok();
  }

  template <typename U>
  Status u(U& out) {
    std::uint8_t b[sizeof(U)] = {};
    CLM_RETURN_IF_ERROR(read(b, sizeof(U)));
    out = load_le<U>(b);
    return Status::ok();
  }

  Status string(std::string& out) {
    std::uint64_t len = 0;
    CLM_RETURN_IF_ERROR(u(len));
    if (len > limits_.max_string_bytes)
      return err(ErrorCode::kResourceExhausted, "string of " + std::to_string(len) + " bytes exceeds the " +
                                                    std::to_string(limits_.max_string_bytes) + "-byte limit");
    CLM_RETURN_IF_ERROR(check(len));  // before allocating
    out.resize(static_cast<std::size_t>(len));
    return read(out.data(), out.size());
  }

  Status skip_string() {
    std::uint64_t len = 0;
    CLM_RETURN_IF_ERROR(u(len));
    if (len > limits_.max_string_bytes)
      return err(ErrorCode::kResourceExhausted, "string of " + std::to_string(len) + " bytes exceeds the " +
                                                    std::to_string(limits_.max_string_bytes) + "-byte limit");
    return skip(len);
  }

 private:
  GgufSource& src_;
  std::uint64_t size_;
  const GgufLimits& limits_;
  std::uint64_t pos_ = 0;
  std::vector<std::uint8_t> buf_;
  std::uint64_t buf_off_ = 0;
};

std::size_t scalar_size(GgufValueType t) {
  switch (t) {
    case GgufValueType::kU8: case GgufValueType::kI8: case GgufValueType::kBool: return 1;
    case GgufValueType::kU16: case GgufValueType::kI16: return 2;
    case GgufValueType::kU32: case GgufValueType::kI32: case GgufValueType::kF32: return 4;
    case GgufValueType::kU64: case GgufValueType::kI64: case GgufValueType::kF64: return 8;
    default: return 0;
  }
}

Status read_scalar(Cursor& c, GgufValueType t, GgufValue& v) {
  v.type = t;
  switch (t) {
    case GgufValueType::kU8: { std::uint8_t x = 0; CLM_RETURN_IF_ERROR(c.u(x)); v.u = x; return Status::ok(); }
    case GgufValueType::kI8: { std::uint8_t x = 0; CLM_RETURN_IF_ERROR(c.u(x)); v.i = static_cast<std::int8_t>(x); return Status::ok(); }
    case GgufValueType::kU16: { std::uint16_t x = 0; CLM_RETURN_IF_ERROR(c.u(x)); v.u = x; return Status::ok(); }
    case GgufValueType::kI16: { std::uint16_t x = 0; CLM_RETURN_IF_ERROR(c.u(x)); v.i = static_cast<std::int16_t>(x); return Status::ok(); }
    case GgufValueType::kU32: { std::uint32_t x = 0; CLM_RETURN_IF_ERROR(c.u(x)); v.u = x; return Status::ok(); }
    case GgufValueType::kI32: { std::uint32_t x = 0; CLM_RETURN_IF_ERROR(c.u(x)); v.i = static_cast<std::int32_t>(x); return Status::ok(); }
    case GgufValueType::kU64: { std::uint64_t x = 0; CLM_RETURN_IF_ERROR(c.u(x)); v.u = x; return Status::ok(); }
    case GgufValueType::kI64: { std::uint64_t x = 0; CLM_RETURN_IF_ERROR(c.u(x)); v.i = static_cast<std::int64_t>(x); return Status::ok(); }
    case GgufValueType::kF32: {
      std::uint32_t x = 0;
      CLM_RETURN_IF_ERROR(c.u(x));
      float f;
      std::memcpy(&f, &x, sizeof f);
      v.f = static_cast<double>(f);
      return Status::ok();
    }
    case GgufValueType::kF64: {
      std::uint64_t x = 0;
      CLM_RETURN_IF_ERROR(c.u(x));
      std::memcpy(&v.f, &x, sizeof v.f);
      return Status::ok();
    }
    case GgufValueType::kBool: {
      std::uint8_t x = 0;
      CLM_RETURN_IF_ERROR(c.u(x));
      if (x > 1) return err(ErrorCode::kInvalidArgument, "bool value " + std::to_string(x) + " is not 0 or 1");
      v.u = x;
      return Status::ok();
    }
    default: return err(ErrorCode::kInvalidArgument, "not a scalar type");
  }
}

Status read_value(Cursor& c, GgufValueType t, GgufValue& v, const GgufLimits& limits, bool retain_all) {
  if (static_cast<std::uint32_t>(t) > static_cast<std::uint32_t>(GgufValueType::kF64))
    return err(ErrorCode::kInvalidArgument, "unknown metadata value type " + std::to_string(static_cast<std::uint32_t>(t)));
  if (t == GgufValueType::kString) {
    v.type = t;
    return c.string(v.s);
  }
  if (t != GgufValueType::kArray) return read_scalar(c, t, v);

  v.type = GgufValueType::kArray;
  std::uint32_t et = 0;
  CLM_RETURN_IF_ERROR(c.u(et));
  if (et == static_cast<std::uint32_t>(GgufValueType::kArray)) return err(ErrorCode::kInvalidArgument, "nested arrays are not supported");
  if (et > static_cast<std::uint32_t>(GgufValueType::kF64))
    return err(ErrorCode::kInvalidArgument, "unknown array element type " + std::to_string(et));
  v.elem_type = static_cast<GgufValueType>(et);
  CLM_RETURN_IF_ERROR(c.u(v.count));
  if (v.count > limits.max_array_elements)
    return err(ErrorCode::kResourceExhausted, "array of " + std::to_string(v.count) + " elements exceeds the " +
                                                  std::to_string(limits.max_array_elements) + "-element limit");
  if (retain_all && v.count > limits.max_retained_array_elements)
    return err(ErrorCode::kResourceExhausted, "array of " + std::to_string(v.count) + " elements exceeds the " +
                                                  std::to_string(limits.max_retained_array_elements) + "-element retention limit");
  const std::size_t keep = static_cast<std::size_t>(
      retain_all ? v.count : std::min<std::uint64_t>(v.count, GgufValue::kRetainedArrayItems));
  if (v.elem_type == GgufValueType::kString) {
    // Every string carries at least its 8-byte length: a count the file cannot hold is rejected before reserving.
    if (retain_all) CLM_RETURN_IF_ERROR(c.check(v.count * 8));
    v.items.reserve(keep);
    for (std::uint64_t i = 0; i < v.count; ++i) {
      if (i < keep) {
        GgufValue e;
        e.type = GgufValueType::kString;
        CLM_RETURN_IF_ERROR(c.string(e.s));
        v.items.push_back(std::move(e));
      } else {
        CLM_RETURN_IF_ERROR(c.skip_string());
      }
    }
    return Status::ok();
  }
  const std::uint64_t es = scalar_size(v.elem_type);
  if (v.count > std::numeric_limits<std::uint64_t>::max() / es) return err(ErrorCode::kOutOfRange, "array byte size overflows");
  CLM_RETURN_IF_ERROR(c.check(v.count * es));  // the whole array must fit in the file before anything is retained
  v.items.reserve(keep);
  for (std::size_t i = 0; i < keep; ++i) {
    GgufValue e;
    CLM_RETURN_IF_ERROR(read_scalar(c, v.elem_type, e));
    v.items.push_back(std::move(e));
  }
  return c.skip((v.count - keep) * es);
}

bool is_pow2(std::uint64_t v) { return v != 0 && (v & (v - 1)) == 0; }

class FileSource final : public GgufSource {
 public:
  FileSource(const std::filesystem::path& p, std::uint64_t size) : f_(p, std::ios::binary), size_(size) {}
  bool ok() const { return static_cast<bool>(f_); }
  std::uint64_t size() const override { return size_; }
  Status read(std::uint64_t offset, void* dst, std::size_t n) override {
    f_.clear();
    f_.seekg(static_cast<std::streamoff>(offset));
    f_.read(static_cast<char*>(dst), static_cast<std::streamsize>(n));
    if (!f_ || static_cast<std::size_t>(f_.gcount()) != n)
      return err(ErrorCode::kUnavailable, "short read at offset " + std::to_string(offset) + " (file changed or I/O error)");
    return Status::ok();
  }

 private:
  std::ifstream f_;
  std::uint64_t size_;
};

class MemorySource final : public GgufSource {
 public:
  explicit MemorySource(ByteSpan b) : b_(b) {}
  std::uint64_t size() const override { return b_.size(); }
  Status read(std::uint64_t offset, void* dst, std::size_t n) override {
    if (offset > b_.size() || n > b_.size() - offset) return err(ErrorCode::kOutOfRange, "read past end of buffer");
    if (n != 0) std::memcpy(dst, b_.data() + offset, n);
    return Status::ok();
  }

 private:
  ByteSpan b_;
};

std::optional<std::uint64_t> meta_u64(const GgufFile& f, const char* key) {
  const GgufValue* v = f.find(key);
  return v == nullptr ? std::nullopt : v->as_u64();
}

}  // namespace

bool GgufValue::is_integer() const {
  switch (type) {
    case GgufValueType::kU8: case GgufValueType::kI8: case GgufValueType::kU16: case GgufValueType::kI16:
    case GgufValueType::kU32: case GgufValueType::kI32: case GgufValueType::kU64: case GgufValueType::kI64: return true;
    default: return false;
  }
}

std::optional<std::uint64_t> GgufValue::as_u64() const {
  switch (type) {
    case GgufValueType::kU8: case GgufValueType::kU16: case GgufValueType::kU32: case GgufValueType::kU64: return u;
    case GgufValueType::kI8: case GgufValueType::kI16: case GgufValueType::kI32: case GgufValueType::kI64:
      if (i < 0) return std::nullopt;
      return static_cast<std::uint64_t>(i);
    default: return std::nullopt;
  }
}

std::optional<double> GgufValue::as_double() const {
  if (is_float()) return f;
  if (!is_integer()) return std::nullopt;
  switch (type) {
    case GgufValueType::kI8: case GgufValueType::kI16: case GgufValueType::kI32: case GgufValueType::kI64:
      return static_cast<double>(i);
    default: return static_cast<double>(u);
  }
}

std::uint64_t GgufTensorInfo::row_bytes() const {
  auto r = ggml_row_bytes(type, dims[0]);
  return r.is_ok() ? *r : 0;
}

const GgufValue* GgufFile::find(const std::string& key) const {
  auto it = metadata.find(key);
  return it == metadata.end() ? nullptr : &it->second;
}

const GgufTensorInfo* GgufFile::find_tensor(std::string_view name) const {
  auto it = tensor_index.find(std::string(name));
  return it == tensor_index.end() ? nullptr : &tensors[it->second];
}

Result<GgufFile> parse_gguf(GgufSource& src, const GgufLimits& limits, std::string label) {
  Cursor c(src, limits);
  GgufFile f;
  f.path = std::move(label);
  f.file_size = src.size();

  std::uint32_t magic = 0;
  CLM_RETURN_IF_ERROR(c.u(magic));
  if (magic != kMagic) return err(ErrorCode::kInvalidArgument, "bad magic (not a GGUF file)");
  CLM_RETURN_IF_ERROR(c.u(f.version));
  if (f.version == 1) return make_error(ErrorCode::kVersionMismatch, "gguf: version 1 (32-bit counts) is not supported");
  if (f.version > 0xFFFFu) return make_error(ErrorCode::kVersionMismatch, "gguf: big-endian files are not supported");
  if (f.version != 2 && f.version != 3)
    return make_error(ErrorCode::kVersionMismatch, "gguf: unsupported version " + std::to_string(f.version));
  std::uint64_t n_tensors = 0, n_kv = 0;
  CLM_RETURN_IF_ERROR(c.u(n_tensors));
  CLM_RETURN_IF_ERROR(c.u(n_kv));
  if (n_tensors > limits.max_tensors)
    return err(ErrorCode::kResourceExhausted, std::to_string(n_tensors) + " tensors exceeds the limit of " + std::to_string(limits.max_tensors));
  if (n_kv > limits.max_kv)
    return err(ErrorCode::kResourceExhausted, std::to_string(n_kv) + " metadata entries exceeds the limit of " + std::to_string(limits.max_kv));
  // Each tensor info is at least 8 (name len) + 4 + 8 + 4 + 8 bytes and each KV at least 8 + 4: reject counts the
  // file cannot possibly hold before reserving anything.
  if (n_tensors > f.file_size / 32 || n_kv > f.file_size / 12)
    return err(ErrorCode::kOutOfRange, "declared counts exceed what the file can hold");

  for (std::uint64_t k = 0; k < n_kv; ++k) {
    std::string key;
    CLM_RETURN_IF_ERROR(c.string(key));
    if (key.empty()) return err(ErrorCode::kInvalidArgument, "empty metadata key");
    std::uint32_t t = 0;
    CLM_RETURN_IF_ERROR(c.u(t));
    GgufValue v;
    const bool retain_all = std::find(limits.retain_full_arrays.begin(), limits.retain_full_arrays.end(), key) !=
                            limits.retain_full_arrays.end();
    CLM_RETURN_IF_ERROR(read_value(c, static_cast<GgufValueType>(t), v, limits, retain_all));
    if (!f.metadata.emplace(std::move(key), std::move(v)).second)
      return err(ErrorCode::kInvalidArgument, "duplicate metadata key");
  }

  f.tensors.reserve(static_cast<std::size_t>(std::min<std::uint64_t>(n_tensors, 65536)));  // grow with the data actually present
  for (std::uint64_t i = 0; i < n_tensors; ++i) {
    GgufTensorInfo t;
    CLM_RETURN_IF_ERROR(c.string(t.name));
    if (t.name.empty()) return err(ErrorCode::kInvalidArgument, "empty tensor name");
    CLM_RETURN_IF_ERROR(c.u(t.n_dims));
    if (t.n_dims == 0 || t.n_dims > 4)
      return err(ErrorCode::kInvalidArgument, "tensor '" + t.name + "': n_dims " + std::to_string(t.n_dims) + " not in [1,4]");
    for (std::uint32_t d = 0; d < t.n_dims; ++d) CLM_RETURN_IF_ERROR(c.u(t.dims[d]));
    std::uint32_t type_id = 0;
    CLM_RETURN_IF_ERROR(c.u(type_id));
    const GgmlTypeInfo* info = ggml_type_info(type_id);
    if (info == nullptr)
      return err(ErrorCode::kInvalidArgument, "tensor '" + t.name + "': unknown ggml type id " + std::to_string(type_id));
    t.type = info->type;
    CLM_RETURN_IF_ERROR(c.u(t.offset));
    t.n_elements = 1;
    for (std::uint32_t d = 0; d < t.n_dims; ++d) {
      if (t.dims[d] == 0 || t.dims[d] > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
        return err(ErrorCode::kInvalidArgument, "tensor '" + t.name + "': dimension " + std::to_string(d) + " is zero or too large");
      if (t.n_elements > std::numeric_limits<std::uint64_t>::max() / t.dims[d])
        return err(ErrorCode::kOutOfRange, "tensor '" + t.name + "': element count overflows");
      t.n_elements *= t.dims[d];
    }
    auto bytes = ggml_tensor_bytes(t.type, std::span<const std::uint64_t>(t.dims.data(), t.n_dims));
    if (!bytes.is_ok()) return err(bytes.status().code(), "tensor '" + t.name + "': " + bytes.status().message());
    t.n_bytes = *bytes;
    f.tensors.push_back(std::move(t));
  }
  f.header_bytes = c.pos();

  if (const GgufValue* a = f.find("general.alignment")) {
    auto v = a->as_u64();
    if (!v || !is_pow2(*v)) return err(ErrorCode::kInvalidArgument, "general.alignment must be a positive power of two");
    if (*v > limits.max_alignment) return err(ErrorCode::kResourceExhausted, "general.alignment exceeds the limit");
    f.alignment = *v;
  }
  f.data_start = (f.header_bytes + f.alignment - 1) / f.alignment * f.alignment;
  // With allow_vocab_only_tail, a file without tensors may end right after the header without the alignment padding.
  if (limits.allow_vocab_only_tail && f.data_start > f.file_size && f.tensors.empty()) f.data_start = f.file_size;
  if (f.data_start > f.file_size)
    return err(ErrorCode::kOutOfRange, "data section starts at " + std::to_string(f.data_start) + ", past the end of the " +
                                           std::to_string(f.file_size) + "-byte file");
  const std::uint64_t payload = f.file_size - f.data_start;

  f.tensor_index.reserve(f.tensors.size());
  for (std::size_t i = 0; i < f.tensors.size(); ++i) {
    GgufTensorInfo& t = f.tensors[i];
    if (!f.tensor_index.emplace(t.name, i).second) return err(ErrorCode::kInvalidArgument, "duplicate tensor name '" + t.name + "'");
    if (t.offset % f.alignment != 0)
      return err(ErrorCode::kInvalidArgument, "tensor '" + t.name + "': offset " + std::to_string(t.offset) + " is not a multiple of the alignment");
    if (t.offset > payload || t.n_bytes > payload - t.offset)
      return err(ErrorCode::kOutOfRange, "tensor '" + t.name + "': bytes [" + std::to_string(t.offset) + ", +" +
                                             std::to_string(t.n_bytes) + ") lie outside the data section of " + std::to_string(payload) + " bytes");
    t.file_offset = f.data_start + t.offset;
  }
  // Tensors may not share bytes.
  std::vector<std::size_t> order(f.tensors.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return f.tensors[a].offset < f.tensors[b].offset; });
  for (std::size_t k = 1; k < order.size(); ++k) {
    const GgufTensorInfo &p = f.tensors[order[k - 1]], &q = f.tensors[order[k]];
    if (p.offset + p.n_bytes > q.offset)
      return err(ErrorCode::kInvalidArgument, "tensors '" + p.name + "' and '" + q.name + "' overlap");
  }
  return f;
}

Result<GgufFile> read_gguf_file(const std::filesystem::path& path, const GgufLimits& limits) {
  std::error_code ec;
  const auto size = std::filesystem::file_size(path, ec);
  if (ec) return make_error(ErrorCode::kNotFound, "gguf: cannot stat " + path.string() + ": " + ec.message());
  FileSource src(path, size);
  if (!src.ok()) return make_error(ErrorCode::kUnavailable, "gguf: cannot open " + path.string());
  auto r = parse_gguf(src, limits, path.string());
  if (!r.is_ok()) return make_error(r.status().code(), path.filename().string() + ": " + r.status().message());
  return r;
}

Result<GgufFile> parse_gguf_bytes(ByteSpan bytes, const GgufLimits& limits) {
  MemorySource src(bytes);
  return parse_gguf(src, limits, "<memory>");
}

std::optional<GgufModelFiles::TensorRef> GgufModelFiles::find_tensor(std::string_view name) const {
  for (std::size_t s = 0; s < shards.size(); ++s)
    if (const GgufTensorInfo* t = shards[s].find_tensor(name)) return TensorRef{static_cast<std::uint32_t>(s), t};
  return std::nullopt;
}

std::size_t GgufModelFiles::tensor_count() const {
  std::size_t n = 0;
  for (const GgufFile& s : shards) n += s.tensors.size();
  return n;
}

Result<GgufModelFiles> open_gguf_model(const std::vector<std::filesystem::path>& paths, const GgufLimits& limits) {
  if (paths.empty()) return err(ErrorCode::kInvalidArgument, "a model needs at least one file");
  if (paths.size() > limits.max_shards) return err(ErrorCode::kResourceExhausted, "too many shard files");
  GgufModelFiles m;
  std::vector<GgufFile> files;
  for (const auto& p : paths) {
    CLM_ASSIGN_OR_RETURN(GgufFile f, read_gguf_file(p, limits));
    files.push_back(std::move(f));
  }
  const std::size_t n = files.size();
  std::size_t with_split = 0;
  for (const GgufFile& f : files)
    if (f.find("split.count") != nullptr) ++with_split;

  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), std::size_t{0});
  if (n == 1) {
    const auto count = meta_u64(files[0], "split.count");
    if (count && *count > 1)
      return err(ErrorCode::kFailedPrecondition, files[0].path + " is shard " + std::to_string(meta_u64(files[0], "split.no").value_or(0) + 1) +
                                                     " of " + std::to_string(*count) + " but was opened as a whole model");
  } else if (with_split == n) {
    m.is_split = true;
    std::vector<std::optional<std::size_t>> by_no(n);
    std::optional<std::uint64_t> total_decl;
    std::uint64_t total_actual = 0;
    for (std::size_t i = 0; i < n; ++i) {
      const GgufFile& f = files[i];
      const auto count = meta_u64(f, "split.count"), no = meta_u64(f, "split.no"), tc = meta_u64(f, "split.tensors.count");
      if (!count || !no) return err(ErrorCode::kInvalidArgument, f.path + ": split.count / split.no missing or not integers");
      if (*count != n)
        return err(ErrorCode::kFailedPrecondition, f.path + " declares " + std::to_string(*count) + " shards but " + std::to_string(n) + " were given");
      if (*no >= n || by_no[static_cast<std::size_t>(*no)])
        return err(ErrorCode::kInvalidArgument, f.path + ": split.no " + std::to_string(*no) + " out of range or repeated");
      by_no[static_cast<std::size_t>(*no)] = i;
      if (tc) {
        if (total_decl && *total_decl != *tc) return err(ErrorCode::kInvalidArgument, "shards disagree on split.tensors.count");
        total_decl = tc;
      }
      total_actual += f.tensors.size();
    }
    if (total_decl && *total_decl != total_actual)
      return err(ErrorCode::kDataLoss, "shards hold " + std::to_string(total_actual) + " tensors but split.tensors.count is " + std::to_string(*total_decl));
    for (std::size_t k = 0; k < n; ++k) order[k] = *by_no[k];
    if (files[order[0]].find("general.architecture") == nullptr)
      return err(ErrorCode::kInvalidArgument, "the first shard of a split model must carry general.architecture");
  } else if (with_split != 0) {
    return err(ErrorCode::kInvalidArgument, "some files carry split.* keys and some do not");
  }

  std::unordered_set<std::string> names;
  for (std::size_t k = 0; k < n; ++k) {
    GgufFile& f = files[order[k]];
    for (const GgufTensorInfo& t : f.tensors)
      if (!names.insert(t.name).second)
        return err(ErrorCode::kAlreadyExists, "tensor '" + t.name + "' appears in more than one file (GGUF has no way to say which is meant)");
    m.paths.push_back(paths[order[k]]);
    m.shards.push_back(std::move(f));
  }
  return m;
}

Result<std::vector<std::filesystem::path>> expand_split_paths(const std::filesystem::path& any_shard) {
  const std::string name = any_shard.filename().string();
  constexpr std::string_view kExt = ".gguf";
  // "<stem>-NNNNN-of-MMMMM.gguf"
  if (name.size() < kExt.size() + 16 || name.compare(name.size() - kExt.size(), kExt.size(), kExt) != 0) return std::vector{any_shard};
  const std::string base = name.substr(0, name.size() - kExt.size());
  const std::size_t tail = base.size() - 15;  // "-NNNNN-of-MMMMM" is 15 chars
  auto digits = [&](std::size_t at) {
    for (std::size_t i = 0; i < 5; ++i)
      if (base[at + i] < '0' || base[at + i] > '9') return false;
    return true;
  };
  if (base.size() < 15 || base[tail] != '-' || !digits(tail + 1) || base.compare(tail + 6, 4, "-of-") != 0 || !digits(tail + 10))
    return std::vector{any_shard};
  const unsigned total = static_cast<unsigned>(std::stoul(base.substr(tail + 10, 5)));
  if (total == 0 || total > 4096) return err(ErrorCode::kInvalidArgument, "implausible shard count in " + name);
  std::vector<std::filesystem::path> out;
  for (unsigned i = 1; i <= total; ++i) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "-%05u-of-%05u", i, total);
    std::filesystem::path p = any_shard.parent_path() / (base.substr(0, tail) + buf + std::string(kExt));
    std::error_code ec;
    if (!std::filesystem::exists(p, ec)) return make_error(ErrorCode::kNotFound, "gguf: missing shard " + p.string());
    out.push_back(std::move(p));
  }
  return out;
}

}  // namespace clusterlm::objects
