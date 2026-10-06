#include "clusterlm/objects/canonical_store.hpp"

#include <fstream>
#include <sstream>

namespace clusterlm::objects {

// A converted object (conversion_version > 0) is served as stored only when its shard already holds the converted
// bytes - a Father-side pre-converted store (e.g. clusterlm-strata convert), recognisable by source == object digest.
// Converting on the fly is not this store's job.
namespace {
bool served_as_stored(const ManifestObject& obj) {
  return obj.representation.conversion_version == 0 || obj.source_digest == obj.object_digest;
}
}  // namespace

Status verify_object_bytes(const ManifestObject& obj, ByteSpan bytes) {
  if (bytes.size() != obj.byte_size)
    return make_error(ErrorCode::kDataLoss, "object '" + obj.name + "': size mismatch");
  const Digest256 d = Sha256::of(bytes);
  if (d != obj.object_digest) return make_error(ErrorCode::kDataLoss, "object '" + obj.name + "': object digest mismatch");
  if (obj.representation.conversion_version == 0 && d != obj.source_digest)
    return make_error(ErrorCode::kDataLoss, "object '" + obj.name + "': source digest mismatch");
  return Status::ok();
}

Result<std::unique_ptr<CanonicalModelStore>> CanonicalModelStore::open(const std::filesystem::path& dir) {
  std::ifstream in(dir / std::string(kManifestFileName), std::ios::binary);
  if (!in) return make_error(ErrorCode::kNotFound, "cannot open " + (dir / std::string(kManifestFileName)).string());
  std::ostringstream text;
  text << in.rdbuf();
  CLM_ASSIGN_OR_RETURN(ModelManifest m, ModelManifest::from_json(text.str()));
  return open(dir, std::move(m));
}

Result<std::unique_ptr<CanonicalModelStore>> CanonicalModelStore::open(const std::filesystem::path& dir,
                                                                       ModelManifest manifest) {
  CLM_RETURN_IF_ERROR(manifest.validate());
  for (const ShardInfo& s : manifest.shards) {
    std::error_code ec;
    const auto size = std::filesystem::file_size(dir / s.file_name, ec);
    if (ec) return make_error(ErrorCode::kNotFound, "shard missing: " + s.file_name);
    if (size != s.byte_size) return make_error(ErrorCode::kDataLoss, "shard size mismatch: " + s.file_name);
  }
  return std::unique_ptr<CanonicalModelStore>(new CanonicalModelStore(dir, std::move(manifest)));
}

Result<Bytes> CanonicalModelStore::load_unlocked(const ManifestObject& obj) const {
  Bytes out(obj.byte_size);
  std::size_t pos = 0;
  // One open per object per shard, one bounded read per source range.
  std::unordered_map<std::uint32_t, std::ifstream> files;
  for (const SourceRange& r : obj.source_ranges) {
    if (r.length > out.size() - pos)
      return make_error(ErrorCode::kDataLoss, "object '" + obj.name + "': ranges exceed object size");
    auto it = files.find(r.shard);
    if (it == files.end()) {
      it = files.emplace(r.shard, std::ifstream(dir_ / manifest_.shards[r.shard].file_name, std::ios::binary)).first;
      if (!it->second) return make_error(ErrorCode::kNotFound, "cannot open shard " + manifest_.shards[r.shard].file_name);
    }
    std::ifstream& f = it->second;
    f.seekg(static_cast<std::streamoff>(r.offset));
    f.read(reinterpret_cast<char*>(out.data() + pos), static_cast<std::streamsize>(r.length));
    if (!f || static_cast<std::uint64_t>(f.gcount()) != r.length)
      return make_error(ErrorCode::kDataLoss, "object '" + obj.name + "': short read");
    pos += static_cast<std::size_t>(r.length);
  }
  // Conversion (conversion_version > 0) would happen here; only pre-converted objects are served.
  if (!served_as_stored(obj))
    return make_error(ErrorCode::kUnimplemented, "object '" + obj.name + "': conversion not supported by this store");
  CLM_RETURN_IF_ERROR(verify_object_bytes(obj, out));
  return out;
}

Status CanonicalModelStore::stream_object(std::string_view name, std::size_t chunk_bytes,
                                          const std::function<Status(std::uint64_t, ByteSpan)>& sink) const {
  const ManifestObject* obj = manifest_.find(name);
  if (obj == nullptr) return make_error(ErrorCode::kNotFound, "unknown object '" + std::string(name) + "'");
  if (!served_as_stored(*obj))
    return make_error(ErrorCode::kUnimplemented, "object '" + obj->name + "': conversion not supported by this store");
  if (chunk_bytes == 0) return make_error(ErrorCode::kInvalidArgument, "chunk size must be positive");
  Sha256 digest;
  Bytes buf(chunk_bytes);
  std::size_t fill = 0;
  std::uint64_t emitted = 0;
  std::unordered_map<std::uint32_t, std::ifstream> files;
  auto flush = [&]() -> Status {
    if (fill == 0) return Status::ok();
    const ByteSpan chunk(buf.data(), fill);
    digest.update(chunk);
    CLM_RETURN_IF_ERROR(sink(emitted, chunk));
    emitted += fill;
    fill = 0;
    return Status::ok();
  };
  std::uint64_t total = 0;
  for (const SourceRange& r : obj->source_ranges) {
    total += r.length;
    if (total > obj->byte_size)
      return make_error(ErrorCode::kDataLoss, "object '" + obj->name + "': ranges exceed object size");
    auto it = files.find(r.shard);
    if (it == files.end()) {
      it = files.emplace(r.shard, std::ifstream(dir_ / manifest_.shards[r.shard].file_name, std::ios::binary)).first;
      if (!it->second) return make_error(ErrorCode::kNotFound, "cannot open shard " + manifest_.shards[r.shard].file_name);
    }
    std::ifstream& f = it->second;
    f.seekg(static_cast<std::streamoff>(r.offset));
    std::uint64_t left = r.length;
    while (left > 0) {
      const auto n = static_cast<std::size_t>(std::min<std::uint64_t>(left, chunk_bytes - fill));
      f.read(reinterpret_cast<char*>(buf.data() + fill), static_cast<std::streamsize>(n));
      if (!f || static_cast<std::size_t>(f.gcount()) != n)
        return make_error(ErrorCode::kDataLoss, "object '" + obj->name + "': short read");
      fill += n;
      left -= n;
      if (fill == chunk_bytes) CLM_RETURN_IF_ERROR(flush());
    }
  }
  CLM_RETURN_IF_ERROR(flush());
  if (emitted != obj->byte_size) return make_error(ErrorCode::kDataLoss, "object '" + obj->name + "': size mismatch");
  if (digest.finish() != obj->object_digest)
    return make_error(ErrorCode::kDataLoss, "object '" + obj->name + "': digest mismatch in canonical store");
  return Status::ok();
}

Result<ProvisionedObject> CanonicalModelStore::resolve(std::string_view name) const {
  const ManifestObject* obj = manifest_.find(name);
  if (obj == nullptr) return make_error(ErrorCode::kNotFound, "unknown object '" + std::string(name) + "'");
  std::lock_guard lock(mu_);
  auto it = cache_.find(obj->name);
  if (it == cache_.end()) {
    CLM_ASSIGN_OR_RETURN(Bytes bytes, load_unlocked(*obj));
    loaded_bytes_ += bytes.size();
    it = cache_.emplace(obj->name, std::move(bytes)).first;
  }
  return ProvisionedObject{obj, ByteSpan(it->second), AllocationTarget::kCpuResident};
}

Result<Bytes> CanonicalModelStore::read_object_bytes(std::string_view name) const {
  const ManifestObject* obj = manifest_.find(name);
  if (obj == nullptr) return make_error(ErrorCode::kNotFound, "unknown object '" + std::string(name) + "'");
  std::lock_guard lock(mu_);
  return load_unlocked(*obj);
}

std::uint64_t CanonicalModelStore::loaded_bytes() const {
  std::lock_guard lock(mu_);
  return loaded_bytes_;
}

std::size_t CanonicalModelStore::loaded_object_count() const {
  std::lock_guard lock(mu_);
  return cache_.size();
}

void CanonicalModelStore::evict_all() {
  std::lock_guard lock(mu_);
  cache_.clear();
  loaded_bytes_ = 0;
}

Status InMemoryResolver::add(const std::string& name, Bytes bytes) {
  const ManifestObject* obj = manifest_.find(name);
  if (obj == nullptr) return make_error(ErrorCode::kNotFound, "object not in manifest: " + name);
  CLM_RETURN_IF_ERROR(verify_object_bytes(*obj, bytes));
  objects_[name] = std::move(bytes);
  return Status::ok();
}

bool InMemoryResolver::remove(const std::string& name) { return objects_.erase(name) != 0; }

std::uint64_t InMemoryResolver::bytes() const {
  std::uint64_t n = 0;
  for (const auto& [name, b] : objects_) n += b.size();
  return n;
}

Result<ProvisionedObject> InMemoryResolver::resolve(std::string_view name) const {
  auto it = objects_.find(std::string(name));
  if (it == objects_.end()) return make_error(ErrorCode::kNotFound, "object not provisioned: " + std::string(name));
  return ProvisionedObject{manifest_.find(name), ByteSpan(it->second), AllocationTarget::kCpuResident};
}

}  // namespace clusterlm::objects
