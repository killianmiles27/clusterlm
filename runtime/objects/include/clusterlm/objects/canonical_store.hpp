#pragma once
// CanonicalModelStore (Father side) and InMemoryResolver (tests, Nodes).
//
// The canonical store is the only permanent holder of the full model. It opens a model directory +
// manifest, and materializes an object lazily on first resolve(): one bounded read per source range,
// digest verification, then caching. Counters make "this domain loaded only its own objects" testable.
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "clusterlm/objects/provisioned.hpp"

namespace clusterlm::objects {

class CanonicalModelStore final : public ObjectResolver {
 public:
  static inline constexpr std::string_view kManifestFileName = "manifest.json";

  // Reads <dir>/manifest.json, validates it and checks every shard's file size.
  static Result<std::unique_ptr<CanonicalModelStore>> open(const std::filesystem::path& dir);
  static Result<std::unique_ptr<CanonicalModelStore>> open(const std::filesystem::path& dir, ModelManifest manifest);

  const ModelManifest& manifest() const { return manifest_; }

  // Loads (once), verifies and caches. kNotFound for an unknown name; kDataLoss on a digest mismatch or a
  // short read. Returned bytes stay valid until evict_all() or destruction.
  Result<ProvisionedObject> resolve(std::string_view name) const override;
  // Reads and verifies without caching.
  Result<Bytes> read_object_bytes(std::string_view name) const;
  // Streams an object's provisioned bytes to `sink` in chunks of at most `chunk_bytes`, never materializing
  // the whole object (provisioning memory stays bounded by the chunk size). The object digest is computed
  // incrementally; a mismatch is reported as kDataLoss after the final chunk — the receiver's seal rejects the
  // object independently. Thread-safe: each call uses its own file handles.
  Status stream_object(std::string_view name, std::size_t chunk_bytes,
                       const std::function<Status(std::uint64_t offset, ByteSpan chunk)>& sink) const;

  std::uint64_t loaded_bytes() const;
  std::size_t loaded_object_count() const;
  void evict_all();

 private:
  CanonicalModelStore(std::filesystem::path dir, ModelManifest manifest)
      : dir_(std::move(dir)), manifest_(std::move(manifest)) {}
  Result<Bytes> load_unlocked(const ManifestObject& obj) const;

  std::filesystem::path dir_;
  ModelManifest manifest_;
  mutable std::mutex mu_;
  mutable std::unordered_map<std::string, Bytes> cache_;
  mutable std::uint64_t loaded_bytes_ = 0;
};

// A resolver over bytes already in memory (tests; a Node's lease store builds on the same idea). Every
// object added is digest-verified against its manifest entry.
class InMemoryResolver final : public ObjectResolver {
 public:
  explicit InMemoryResolver(ModelManifest manifest) : manifest_(std::move(manifest)) {}

  const ModelManifest& manifest() const { return manifest_; }
  Status add(const std::string& name, Bytes bytes);
  bool remove(const std::string& name);
  void clear() { objects_.clear(); }
  std::size_t size() const { return objects_.size(); }
  std::uint64_t bytes() const;

  Result<ProvisionedObject> resolve(std::string_view name) const override;

 private:
  ModelManifest manifest_;
  std::unordered_map<std::string, Bytes> objects_;
};

// Digest check shared by both resolvers: object_digest always, and source_digest too when the object is
// unconverted (conversion_version 0), since the provisioned bytes are then the source bytes.
Status verify_object_bytes(const ManifestObject& obj, ByteSpan bytes);

}  // namespace clusterlm::objects
