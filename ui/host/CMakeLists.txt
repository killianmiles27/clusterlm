# Host management views (workstream E): view-models only here; drawing is in host_draw (portable) and the Win32 shell.
clusterlm_module(clusterlm_ui_host
  SOURCES src/host_common.cpp src/models_viewmodel.cpp src/profiles_viewmodel.cpp src/machines_viewmodel.cpp
          src/connections_viewmodel.cpp src/performance_viewmodel.cpp src/settings_viewmodel.cpp
          src/wizard_viewmodel.cpp src/app_viewmodel.cpp
  DEPS clusterlm_ui_common)

# Portable ImGui drawing for the Host window (no backend): linked by the Windows executable and the headless smoke test.
clusterlm_module(clusterlm_ui_host_draw
  SOURCES src/host_draw.cpp
  DEPS clusterlm_ui_host clusterlm_ui_father_draw clusterlm_imgui)
