clusterlm_module(clusterlm_objects
  SOURCES src/geometry.cpp src/manifest.cpp src/canonical_store.cpp src/fixture_model.cpp src/tensor_codec.cpp
  DEPS clusterlm_common
  PRIVATE_DEPS clusterlm_third_party)

# Fixture generator: writes the deterministic fixture shards + manifest.json (never real weights).
add_executable(clusterlm-fixture-model ${PROJECT_SOURCE_DIR}/tools/fixture-model/main.cpp)
target_link_libraries(clusterlm-fixture-model PRIVATE clusterlm_objects)
clusterlm_set_warnings(clusterlm-fixture-model)
