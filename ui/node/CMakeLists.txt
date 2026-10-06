clusterlm_module(clusterlm_ui_node
  SOURCES src/node_viewmodel.cpp
  DEPS clusterlm_ui_common)

clusterlm_module(clusterlm_ui_node_draw
  SOURCES src/node_draw.cpp
  DEPS clusterlm_ui_node clusterlm_imgui)

if(WIN32)
  add_executable(clusterlm-node-ui WIN32 win32/main.cpp)
  target_link_libraries(clusterlm-node-ui PRIVATE clusterlm_ui_node_draw clusterlm_ui_win32_host shell32 ole32)
  clusterlm_set_warnings(clusterlm-node-ui)
endif()
