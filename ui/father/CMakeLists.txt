clusterlm_module(clusterlm_ui_father
  SOURCES src/father_viewmodel.cpp
  DEPS clusterlm_ui_common)

# Portable ImGui drawing (no backend): linked by the Windows executable and by the headless smoke test.
clusterlm_module(clusterlm_ui_father_draw
  SOURCES src/father_draw.cpp
  DEPS clusterlm_ui_father clusterlm_imgui)

if(WIN32)
  add_executable(clusterlm-father-ui WIN32 win32/main.cpp)
  target_link_libraries(clusterlm-father-ui PRIVATE clusterlm_ui_father_draw clusterlm_ui_win32_host shell32 ole32)
  clusterlm_set_warnings(clusterlm-father-ui)
endif()
