add_library(clusterlm_cli INTERFACE)
target_include_directories(clusterlm_cli INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/common)

add_executable(clusterlm-node node/main.cpp)
target_link_libraries(clusterlm-node PRIVATE clusterlm_node_worker clusterlm_cli)
clusterlm_set_warnings(clusterlm-node)

add_executable(clusterlm-father father/main.cpp)
target_link_libraries(clusterlm-father PRIVATE clusterlm_coordinator clusterlm_domain clusterlm_cli)
clusterlm_set_warnings(clusterlm-father)

add_executable(clusterlm-node-service node-service/main.cpp)
target_link_libraries(clusterlm-node-service PRIVATE clusterlm_node_service clusterlm_config clusterlm_pairing clusterlm_cli)
clusterlm_set_warnings(clusterlm-node-service)

# Per-user session helper of the Node (activity + lock reports over a local pipe).
add_executable(clusterlm-node-helper node-helper/main.cpp)
target_link_libraries(clusterlm-node-helper PRIVATE clusterlm_platform clusterlm_cli)
clusterlm_set_warnings(clusterlm-node-helper)

# Father per-user agent: Coordinator host + Father UI pipe. The library is what tests link.
add_library(clusterlm_father_agent STATIC father-agent/father_agent.cpp father-agent/father_service_api.cpp)
target_include_directories(clusterlm_father_agent PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/father-agent/include)
target_link_libraries(clusterlm_father_agent PUBLIC clusterlm_coordinator clusterlm_platform clusterlm_father_service
                      clusterlm_config clusterlm_pairing PRIVATE clusterlm_third_party)
clusterlm_set_warnings(clusterlm_father_agent)
add_executable(clusterlm-father-agent father-agent/main.cpp)
target_link_libraries(clusterlm-father-agent PRIVATE clusterlm_father_agent clusterlm_cli)
clusterlm_set_warnings(clusterlm-father-agent)

# Default resources next to the executables (the agent looks for <exe dir>/clusterlm-catalog.json and
# <exe dir>/profiles/). The profiles are the SYNTHETIC development fixtures; the agent says so wherever a plan
# used them.
configure_file(${PROJECT_SOURCE_DIR}/fixtures/catalog/clusterlm-catalog.json ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/clusterlm-catalog.json COPYONLY)
file(COPY ${PROJECT_SOURCE_DIR}/fixtures/profiles DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
