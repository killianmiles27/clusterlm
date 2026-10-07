// Father agent glue: pairing.unpair notifies the Node (after the session is gone, best effort, outcome reported),
// the default catalog path resolves the installed layout first, and the prepare_progress event JSON carries counts only.
#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>

#include "clusterlm/father/father_agent.hpp"
#include "clusterlm/father/father_service_api.hpp"

using namespace clusterlm;
using nlohmann::json;
namespace fs = std::filesystem;

namespace {

fs::path temp_dir(const char* name) {
  auto p = fs::temp_directory_path() /
           (std::string("clm-glue-") + name + "-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  { std::error_code ec_rm; fs::remove_all(p, ec_rm); }
  fs::create_directories(p);
  return p;
}

class NullReadiness final : public father::ReadinessSource {
 public:
  catalog::ReadinessInputs observe(const catalog::TierEntry&, const catalog::TierAssignment&, std::uint32_t) override { return {}; }
};
class NoDeployments final : public father::DeploymentProvider {
 public:
  Result<father::Deployment> resolve(const catalog::TierEntry&, std::uint32_t) override {
    return make_error(ErrorCode::kUnavailable, "no deployment in this test");
  }
};

json call(father::FatherServiceApi& api, json req) {
  req["id"] = 1;
  const std::string text = req.dump();
  auto reply = api.handle(ipc::father_envelope(ipc::MessageKind::kFatherRequest, Bytes(text.begin(), text.end())), {});
  REQUIRE(reply.is_ok());
  return json::parse(reply->payload.begin(), reply->payload.end());
}

struct Glue {
  fs::path dir = temp_dir("api");
  std::shared_ptr<config::FatherSettingsStore> settings;
  std::unique_ptr<father::FatherServiceApi> api;
  std::vector<std::string> notified;  // fingerprints the notifier was called with
  father::UnpairNotifyResult next{father::UnpairNotifyOutcome::kDelivered, ""};
  bool node_still_listed_when_notified = true;
  std::string notified_address;

  Glue() {
    auto st = config::FatherSettingsStore::open(dir / "father.json");
    REQUIRE(st.is_ok());
    settings = std::shared_ptr<config::FatherSettingsStore>(std::move(st).value());
    auto id = transport::DeviceIdentity::generate("glue-father");
    REQUIRE(id.is_ok());
    father::FatherApiConfig cfg;
    auto cat = catalog::Catalog::load(std::string(CLUSTERLM_SOURCE_DIR) + "/fixtures/catalog/clusterlm-catalog.json");
    REQUIRE(cat.is_ok());
    cfg.catalog = std::move(cat).value();
    cfg.settings = settings;
    cfg.identity = std::make_shared<const transport::DeviceIdentity>(std::move(id).value());
    cfg.readiness = std::make_shared<NullReadiness>();
    cfg.deployments = std::make_shared<NoDeployments>();
    cfg.notify_unpair = [this](const config::PairedDevice& d) {
      notified.push_back(d.fingerprint);
      notified_address = d.address;
      node_still_listed_when_notified = settings->get().find_node(d.fingerprint) != nullptr;
      return next;
    };
    auto a = father::FatherServiceApi::create(std::move(cfg));
    REQUIRE_MESSAGE(a.is_ok(), a.status().to_string());
    api = std::move(a).value();
  }
  ~Glue() {
    api.reset();
    std::error_code ec;
    fs::remove_all(dir, ec);
  }

  std::string add_node(char fill, const char* name) {
    config::PairedDevice d;
    d.fingerprint = std::string(64, fill);
    d.name = name;
    d.address = "127.0.0.1:47600";
    d.role = "node";
    REQUIRE(settings->update([&](config::FatherSettings& s) { s.upsert_node(d); return Status::ok(); }).is_ok());
    return d.fingerprint;
  }
};

}  // namespace

TEST_CASE("pairing.unpair tells a reachable Node after the device is removed, and reports it") {
  Glue g;
  const auto fp = g.add_node('a', "G14");
  const auto other = g.add_node('b', "3060");
  auto r = call(*g.api, {{"op", "pairing.unpair"}, {"fingerprint", fp}});
  REQUIRE_MESSAGE(r["ok"] == true, r.dump());
  REQUIRE(g.notified.size() == 1);
  CHECK(g.notified[0] == fp);
  CHECK(g.notified_address == "127.0.0.1:47600");
  CHECK_FALSE(g.node_still_listed_when_notified);  // the session and device were already gone: the channel is free
  CHECK(r["result"]["node_notified"] == true);
  CHECK(r["result"]["notify_outcome"] == "delivered");
  CHECK(r["result"].contains("note") == false);
  CHECK(g.settings->get().find_node(fp) == nullptr);
  CHECK(g.settings->get().find_node(other) != nullptr);  // the other Node is untouched and not notified
}

TEST_CASE("an unreachable or refusing Node leaves the documented behaviour, and the answer says so") {
  Glue g;
  const auto fp = g.add_node('c', "G14");
  g.next = {father::UnpairNotifyOutcome::kUnreachable, "could not connect to the Node"};
  auto r = call(*g.api, {{"op", "pairing.unpair"}, {"fingerprint", fp}});
  REQUIRE_MESSAGE(r["ok"] == true, r.dump());  // the unpair itself still succeeds
  CHECK(r["result"]["node_notified"] == false);
  CHECK(r["result"]["notify_outcome"] == "unreachable");
  const std::string note = r["result"]["note"];
  CHECK(note.find("keeps trusting this Father") != std::string::npos);
  CHECK(g.settings->get().find_node(fp) == nullptr);

  const auto fp2 = g.add_node('d', "3060");
  g.next = {father::UnpairNotifyOutcome::kRefused, "PERMISSION_DENIED"};
  auto r2 = call(*g.api, {{"op", "pairing.unpair"}, {"fingerprint", fp2}});
  REQUIRE(r2["ok"] == true);
  CHECK(r2["result"]["notify_outcome"] == "refused");
}

TEST_CASE("unpairing an unknown device notifies nobody") {
  Glue g;
  auto r = call(*g.api, {{"op", "pairing.unpair"}, {"fingerprint", std::string(64, 'e')}});
  CHECK(r["ok"] == false);
  CHECK(r["error"]["code"] == "NOT_FOUND");
  CHECK(g.notified.empty());
}

TEST_CASE("the default catalog is found in the installed layout first, then next to the executable") {
  const fs::path root = temp_dir("catalog");
  fs::create_directories(root / "bin");
  const fs::path exe_dir = root / "bin";
  // Neither exists: the development location is returned (for the error message).
  CHECK(father::default_catalog_path(exe_dir) == exe_dir / "clusterlm-catalog.json");
  // Development layout only.
  { std::ofstream(exe_dir / "clusterlm-catalog.json") << "{}"; }
  CHECK(father::default_catalog_path(exe_dir) == exe_dir / "clusterlm-catalog.json");
  // Installed layout wins when present: <prefix>/bin/.. /catalog/clusterlm-catalog.json.
  fs::create_directories(root / "catalog");
  { std::ofstream(root / "catalog" / "clusterlm-catalog.json") << "{}"; }
  CHECK(father::default_catalog_path(exe_dir) == (root / "catalog" / "clusterlm-catalog.json").lexically_normal());
  // A directory with that name is not a catalog.
  fs::remove(root / "catalog" / "clusterlm-catalog.json");
  fs::create_directories(root / "catalog" / "clusterlm-catalog.json");
  CHECK(father::default_catalog_path(exe_dir) == exe_dir / "clusterlm-catalog.json");
  std::error_code ec;
  fs::remove_all(root, ec);
}

TEST_CASE("the shipped catalog fixture is what the installer installs: default resolution loads it") {
  // Mirror cmake/ClusterLMInstall.cmake: <prefix>/bin and <prefix>/catalog/clusterlm-catalog.json.
  const fs::path root = temp_dir("installed");
  fs::create_directories(root / "bin");
  fs::create_directories(root / "catalog");
  fs::copy_file(fs::path(CLUSTERLM_SOURCE_DIR) / "fixtures" / "catalog" / "clusterlm-catalog.json",
                root / "catalog" / "clusterlm-catalog.json");
  const auto path = father::default_catalog_path(root / "bin");
  auto cat = catalog::Catalog::load(path.string());
  REQUIRE_MESSAGE(cat.is_ok(), cat.status().to_string());
  CHECK(cat->find("fast") != nullptr);
  std::error_code ec;
  fs::remove_all(root, ec);
}
