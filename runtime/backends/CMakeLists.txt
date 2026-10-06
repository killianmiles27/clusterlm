# Optional backend adapters. NOTE: the top-level CMakeLists.txt module list must include runtime/backends for
# this directory to be configured; each adapter is additionally gated by its own option.
foreach(adapter strata llama)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${adapter}/CMakeLists.txt)
    add_subdirectory(${adapter})
  endif()
endforeach()
