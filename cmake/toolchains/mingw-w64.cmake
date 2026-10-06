# MinGW-w64 cross toolchain (Linux host -> Windows x64). MSVC is the product compiler; this is a proxy used to
# prove the _WIN32 code paths compile and link (see scripts/check_windows_compile.sh).
set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR x86_64)
set(_triple x86_64-w64-mingw32)
set(CMAKE_C_COMPILER   ${_triple}-gcc-posix)
set(CMAKE_CXX_COMPILER ${_triple}-g++-posix)
set(CMAKE_RC_COMPILER  ${_triple}-windres)
set(CMAKE_FIND_ROOT_PATH /usr/${_triple})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
# Same macros the MSVC build sets in ClusterLMWarnings.cmake.
add_compile_definitions(_WIN32_WINNT=0x0A00 WIN32_LEAN_AND_MEAN NOMINMAX)
# No MinGW runtime DLLs beside the executables, so they run under wine/Windows as-is.
set(CMAKE_EXE_LINKER_FLAGS_INIT "-static -static-libgcc -static-libstdc++")
