clusterlm_module(clusterlm_ui_common
  SOURCES src/common.cpp src/in_process_client.cpp src/ipc_clients.cpp src/scripted_client.cpp
  DEPS clusterlm_common clusterlm_catalog clusterlm_father_service clusterlm_platform Threads::Threads)

if(WIN32)
  # Win32 + Direct3D 11 host for ImGui, shared by both executables.
  add_library(clusterlm_ui_win32_host STATIC
    win32/imgui_host.cpp
    ${IMGUI_DIR}/backends/imgui_impl_win32.cpp ${IMGUI_DIR}/backends/imgui_impl_dx11.cpp)
  target_link_libraries(clusterlm_ui_win32_host PUBLIC clusterlm_imgui d3d11 dxgi d3dcompiler dwmapi gdi32 user32)
  # Only the host's own file follows the project warning policy; the backends are vendored.
  if(MSVC)
    set_source_files_properties(win32/imgui_host.cpp PROPERTIES COMPILE_OPTIONS "/W4;/WX;/wd4324")
  else()
    set_source_files_properties(win32/imgui_host.cpp PROPERTIES COMPILE_OPTIONS "-Wall;-Wextra;-Werror")
  endif()
endif()
