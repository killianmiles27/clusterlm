set(CLUSTERLM_PLATFORM_SOURCES
    src/mapped_file.cpp src/durable_file.cpp src/fs_safety.cpp src/process.cpp
    src/ipc.cpp src/ipc_messages.cpp src/ipc_posix.cpp src/ipc_server_loop.cpp src/helper_client.cpp src/helper_activity.cpp src/helper_startup.cpp
    src/firewall.cpp src/paths.cpp src/service_host.cpp)
set(CLUSTERLM_PLATFORM_PRIVATE_DEPS)
if(WIN32)
  # Real Windows adapters (GetLastInputInfo, GetSystemPowerStatus, Job Objects, DXGI) and the product platform
  # layer (named pipes, SCM service host, WTS/power notifications, firewall COM, ACLs). Not buildable on Linux CI;
  # scripts/check_windows_compile.sh syntax-checks them with MinGW.
  list(APPEND CLUSTERLM_PLATFORM_SOURCES
       src/windows/adapters_win.cpp src/windows/win_events.cpp src/windows/ipc_win.cpp
       src/windows/service_host_win.cpp src/windows/event_window_win.cpp src/windows/firewall_win.cpp
       src/windows/paths_win.cpp src/windows/helper_launch_win.cpp)
  list(APPEND CLUSTERLM_PLATFORM_PRIVATE_DEPS dxgi user32 advapi32 wtsapi32 ole32 oleaut32 powrprof shell32 userenv)
endif()
clusterlm_module(clusterlm_platform
  SOURCES ${CLUSTERLM_PLATFORM_SOURCES}
  DEPS clusterlm_common
  PRIVATE_DEPS ${CLUSTERLM_PLATFORM_PRIVATE_DEPS})
target_include_directories(clusterlm_platform PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
