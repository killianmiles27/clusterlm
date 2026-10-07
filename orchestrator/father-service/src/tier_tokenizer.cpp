#include "clusterlm/father/tier_tokenizer.hpp"

#include "clusterlm/objects/canonical_store.hpp"

namespace clusterlm::father {

namespace fs = std::filesystem;

Result<std::shared_ptr<Tokenizer>> TierTokenizerProvider::get(const catalog::TierEntry& tier) {
  const auto settings = settings_->get();
  auto dir = settings.model_dirs.find(tier.id);
  if (dir == settings.model_dirs.end())
    return make_error(ErrorCode::kNotFound, "no model directory is configured for " + tier.display_name);
  auto store = objects::CanonicalModelStore::open(dir->second);
  if (!store.is_ok()) return store.status();
  const auto& shards = store.value()->manifest().shards;
  if (shards.empty()) return make_error(ErrorCode::kInvalidArgument, "the model manifest lists no GGUF shard");
  const fs::path gguf = fs::path(dir->second) / shards.front().file_name;

  std::error_code ec;
  const auto size = fs::file_size(gguf, ec);
  const auto mtime = ec ? fs::file_time_type{} : fs::last_write_time(gguf, ec);
  const std::string key = gguf.string() + "|" + std::to_string(ec ? 0 : size) + "|" +
                          std::to_string(ec ? 0 : mtime.time_since_epoch().count());
  {
    std::lock_guard lk(mu_);
    auto it = cache_.find(tier.id);
    if (it != cache_.end() && it->second.key == key) return it->second.result;
  }
  Result<std::shared_ptr<Tokenizer>> built = make_error(ErrorCode::kInternal, "unset");
  auto t = GgufBpeTokenizer::from_gguf_file(gguf, options_);
  if (t.is_ok()) built = std::shared_ptr<Tokenizer>(std::move(t).value());
  else built = make_error(t.status().code(), t.status().message());
  std::lock_guard lk(mu_);
  cache_[tier.id] = Entry{key, built};
  return built;
}

}  // namespace clusterlm::father
