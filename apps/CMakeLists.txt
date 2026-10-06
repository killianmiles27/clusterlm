add_library(clusterlm_cli INTERFACE)
target_include_directories(clusterlm_cli INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/common)

add_executable(clusterlm-node node/main.cpp)
target_link_libraries(clusterlm-node PRIVATE clusterlm_node_worker clusterlm_cli)
clusterlm_set_warnings(clusterlm-node)

add_executable(clusterlm-father father/main.cpp)
target_link_libraries(clusterlm-father PRIVATE clusterlm_coordinator clusterlm_domain clusterlm_diagnostics clusterlm_cli)
clusterlm_set_warnings(clusterlm-father)

add_executable(clusterlm-node-service node-service/main.cpp)
target_link_libraries(clusterlm-node-service PRIVATE clusterlm_node_service clusterlm_cli)
clusterlm_set_warnings(clusterlm-node-service)
