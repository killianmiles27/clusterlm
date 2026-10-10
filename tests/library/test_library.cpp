// Model library: identification of real (fixture) GGUF files, split sets, identity resolution, persistence.
// Real code paths over real files written by the GGUF fixture writer; no model weights of a shipped model are involved.
#include <doctest/doctest.h>

#include <filesystem>
#include <cctype>
#include <fstream>

#include "clusterlm/library/library.hpp"
#include "clusterlm/objects/fixture_gguf.hpp"
#include "test_util.hpp"

using namespace clusterlm;
using namespace clusterlm::library;
using clusterlm::testutil::TempDir;
namespace fs = std::filesystem;

namespace {
fs::path write_fixture(const fs::path& dir, bool q8 = true) {
  objects::FixtureGgufOptions o;
  o.q8_experts = q8;
  auto m = objects::write_fixture_gguf(objects::FixtureSpec::tiny(), dir, o);
  REQUIRE_MESSAGE(m.is_ok(), m.status().to_string());
  return dir / std::string(objects::kFixtureGgufShard0);
}
}  // namespace

TEST_CASE("a split GGUF set is one record with stable, structure-derived identity") {
  TempDir d("lib-split");
  const auto first = write_fixture(d.path());
  auto rec = identify_model({first}, {});
  REQUIRE_MESSAGE(rec.is_ok(), rec.status().to_string());
  CHECK(rec->split);
  CHECK(rec->files.size() == 2);
  CHECK(is_model_id(rec->id));
  CHECK(rec->id == model_id_from_digest(rec->structure_digest));
  CHECK(rec->total_bytes == rec->files[0].bytes + rec->files[1].bytes);
  CHECK(rec->tensor_count > 0);
  CHECK_FALSE(rec->root_hash);    // never claimed without verification
  CHECK_FALSE(rec->pinned_root);  // never pinned without the user
  // Passing the second shard, or both, identifies the same model.
  auto again = identify_model({d.path() / std::string(objects::kFixtureGgufShard1)}, {});
  REQUIRE(again.is_ok());
  CHECK(again->id == rec->id);
  // The same artifact copied elsewhere has the same id.
  TempDir d2("lib-split-copy");
  fs::copy(d.path(), d2.path(), fs::copy_options::recursive);
  auto copy = identify_model({d2.path() / std::string(objects::kFixtureGgufShard0)}, {});
  REQUIRE(copy.is_ok());
  CHECK(copy->id == rec->id);
}

TEST_CASE("a different quantization is a different model id and a mixed set lists every tensor type") {
  TempDir a("lib-q8"), b("lib-f32");
  auto r8 = identify_model({write_fixture(a.path(), true)}, {});
  auto rf = identify_model({write_fixture(b.path(), false)}, {});
  REQUIRE(r8.is_ok());
  REQUIRE(rf.is_ok());
  CHECK(r8->id != rf->id);
  CHECK(r8->tensor_types.size() > rf->tensor_types.size());
  CHECK(rf->tensor_types.size() == 1);
  CHECK(rf->quant == "F32");
}

TEST_CASE("a record round-trips through JSON and rejects tampered documents") {
  TempDir d("lib-json");
  auto rec = identify_model({write_fixture(d.path())}, {});
  REQUIRE(rec.is_ok());
  auto back = model_record_from_json(to_json(*rec));
  REQUIRE_MESSAGE(back.is_ok(), back.status().to_string());
  CHECK(*back == *rec);
  auto bad_id = rec.value();
  bad_id.id = "mdl_000000000000000000000000";
  CHECK_FALSE(validate(bad_id).is_ok());
  auto bad_sum = rec.value();
  bad_sum.total_bytes += 1;
  CHECK_FALSE(validate(bad_sum).is_ok());
  auto bad_file = rec.value();
  bad_file.files[0].name = "../escape.gguf";
  CHECK_FALSE(validate(bad_file).is_ok());
  CHECK_FALSE(model_record_from_json("{\"id\": 3}").is_ok());
  CHECK_FALSE(model_record_from_json("not json").is_ok());
}

TEST_CASE("scanning a directory finds split sets once, skips non-GGUF files and reports damaged ones") {
  TempDir d("lib-scan");
  write_fixture(d.path());
  { std::ofstream(d.path() / "notes.txt") << "hello"; }
  { std::ofstream(d.path() / "broken.gguf", std::ios::binary) << "GGUF-not-really"; }
  ModelLibrary lib;
  REQUIRE(lib.add_scan_root(d.path().string()).is_ok());
  auto scan = lib.rescan({});
  REQUIRE_MESSAGE(scan.is_ok(), scan.status().to_string());
  CHECK(scan->records.size() == 1);
  CHECK(lib.records().size() == 1);
  REQUIRE(scan->issues.size() == 1);
  CHECK(scan->issues[0].file == "broken.gguf");
  CHECK(scan->issues[0].message.find(d.path().string()) == std::string::npos);  // no host path in an issue
}

TEST_CASE("a split set with a missing shard is reported, never recorded") {
  TempDir d("lib-missing");
  write_fixture(d.path());
  fs::remove(d.path() / std::string(objects::kFixtureGgufShard1));
  auto scan = scan_directory(d.path(), {});
  REQUIRE(scan.is_ok());
  CHECK(scan->records.empty());
  CHECK(scan->issues.size() == 1);
}

TEST_CASE("identity resolution never guesses") {
  TempDir a("lib-res-a"), b("lib-res-b");
  auto r1 = identify_model({write_fixture(a.path(), true)}, {});
  auto r2 = identify_model({write_fixture(b.path(), false)}, {});
  REQUIRE(r1.is_ok());
  REQUIRE(r2.is_ok());
  ModelLibrary lib;
  const std::string family = r2->family;
  auto shout = family;
  for (auto& c : shout) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  CHECK(lib.resolve({family, "F32", std::nullopt}).kind == Resolution::Kind::kMissing);
  REQUIRE(lib.upsert(*r1).is_ok());
  REQUIRE(lib.upsert(*r2).is_ok());

  // Unpinned: family+quant (case-insensitive) is only an "unpinned match", never verified.
  auto m = lib.resolve({shout, "f32", std::nullopt});
  CHECK(m.kind == Resolution::Kind::kUnpinnedMatch);
  REQUIRE(m.record != nullptr);
  CHECK(m.record->id == r2->id);
  // A quantization that is not in the library is missing: no nearest-quant substitution.
  CHECK(lib.resolve({family, "IQ3_S", std::nullopt}).kind == Resolution::Kind::kMissing);

  // Pinning: only an explicit confirmation makes a pinned identity resolvable.
  const std::string root(64, 'a');
  CHECK(lib.resolve({"x", "y", root}).kind == Resolution::Kind::kMissing);
  REQUIRE(lib.pin_root(r2->id, root).is_ok());
  auto p = lib.resolve({"x", "y", root});
  CHECK(p.kind == Resolution::Kind::kVerified);
  REQUIRE(p.record != nullptr);
  CHECK(p.record->id == r2->id);
  CHECK_FALSE(lib.pin_root("mdl_ffffffffffffffffffffffff", root).is_ok());
  CHECK_FALSE(lib.pin_root(r2->id, "short").is_ok());

  // Ambiguity: two models claiming the same family+quant are never "first wins".
  auto dup = *r1;
  dup.quant = "F32";
  dup.family = family;
  dup.id = "mdl_" + std::string(24, 'c');
  dup.structure_digest = std::string(24, 'c') + std::string(40, '0');
  REQUIRE(lib.upsert(dup).is_ok());
  auto amb = lib.resolve({family, "F32", std::nullopt});
  CHECK(amb.kind == Resolution::Kind::kAmbiguous);
  CHECK(amb.record == nullptr);
  CHECK(amb.candidates.size() == 2);
}

TEST_CASE("verification and pins are kept across rescans and a changed verified root is refused") {
  TempDir d("lib-keep");
  write_fixture(d.path());
  ModelLibrary lib;
  REQUIRE(lib.add_scan_root(d.path().string()).is_ok());
  REQUIRE(lib.rescan({}).is_ok());
  const std::string id = lib.records().front().id;
  REQUIRE(lib.set_verified_root(id, std::string(64, '1')).is_ok());
  REQUIRE(lib.pin_root(id, std::string(64, '1')).is_ok());
  CHECK_FALSE(lib.pin_root(id, std::string(64, '2')).is_ok());
  CHECK(lib.set_verified_root(id, std::string(64, '2')).code() == ErrorCode::kDataLoss);
  REQUIRE(lib.rescan({}).is_ok());
  CHECK(lib.records().size() == 1);
  CHECK(lib.records().front().root_hash == std::string(64, '1'));
  CHECK(lib.records().front().pinned_root == std::string(64, '1'));
}

TEST_CASE("the library document round-trips and refuses a newer schema") {
  TempDir d("lib-doc");
  write_fixture(d.path());
  ModelLibrary lib;
  REQUIRE(lib.add_scan_root(d.path().string()).is_ok());
  REQUIRE(lib.rescan({}).is_ok());
  auto back = ModelLibrary::from_json(lib.to_json());
  REQUIRE_MESSAGE(back.is_ok(), back.status().to_string());
  CHECK(back->records() == lib.records());
  CHECK(back->scan_roots() == lib.scan_roots());
  auto text = lib.to_json();
  const auto pos = text.find("\"schema_version\": 1");
  REQUIRE(pos != std::string::npos);
  text.replace(pos, 19, "\"schema_version\": 2");
  CHECK(ModelLibrary::from_json(text).status().code() == ErrorCode::kVersionMismatch);
}
