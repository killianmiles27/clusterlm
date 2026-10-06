add_library(clusterlm_test_main STATIC test_main.cpp)
target_link_libraries(clusterlm_test_main PUBLIC clusterlm_third_party)

# Each module registers its tests from tests/<area>/CMakeLists.txt.
file(GLOB test_dirs LIST_DIRECTORIES true ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(dir ${test_dirs})
  if(IS_DIRECTORY ${dir} AND EXISTS ${dir}/CMakeLists.txt)
    add_subdirectory(${dir})
  endif()
endforeach()
