# ClusterLM UI (WP13): portable view-models + draw code (built and tested on every platform) and the Win32 shells
# (clusterlm-father-ui, clusterlm-node-ui; WIN32 only). Toolkit: Dear ImGui, see docs/adr/0320.

# Dear ImGui core, vendored under third_party/imgui. Built without the project's warning policy: it is third-party
# code. No backend here: the Win32/D3D11 backends join only the Windows host library below.
set(IMGUI_DIR ${PROJECT_SOURCE_DIR}/third_party/imgui)
add_library(clusterlm_imgui STATIC
  ${IMGUI_DIR}/imgui.cpp ${IMGUI_DIR}/imgui_draw.cpp ${IMGUI_DIR}/imgui_tables.cpp ${IMGUI_DIR}/imgui_widgets.cpp
  ${IMGUI_DIR}/misc/cpp/imgui_stdlib.cpp)
# Public as <imgui/imgui.h> (via third_party) and as "imgui.h" for the backends' own includes.
target_include_directories(clusterlm_imgui SYSTEM PUBLIC ${PROJECT_SOURCE_DIR}/third_party ${IMGUI_DIR})
target_compile_definitions(clusterlm_imgui PUBLIC IMGUI_DISABLE_OBSOLETE_FUNCTIONS)

add_subdirectory(common)
add_subdirectory(father)
add_subdirectory(node)
