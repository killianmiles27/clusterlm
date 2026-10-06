set(CLUSTERLM_PLATFORM_SOURCES src/mapped_file.cpp src/durable_file.cpp src/fs_safety.cpp)
set(CLUSTERLM_PLATFORM_PRIVATE_DEPS)
if(WIN32)
  # Real Windows adapters (GetLastInputInfo, GetSystemPowerStatus, Job Objects, DXGI). Not buildable on Linux CI.
  list(APPEND CLUSTERLM_PLATFORM_SOURCES src/windows/adapters_win.cpp)
  list(APPEND CLUSTERLM_PLATFORM_PRIVATE_DEPS dxgi user32)
endif()
clusterlm_module(clusterlm_platform
  SOURCES ${CLUSTERLM_PLATFORM_SOURCES}
  DEPS clusterlm_common
  PRIVATE_DEPS ${CLUSTERLM_PLATFORM_PRIVATE_DEPS})
