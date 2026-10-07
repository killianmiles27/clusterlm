# clusterlm_module(<target> [SOURCES ...] [DEPS ...]) — a static library with the project's warning policy.
function(clusterlm_set_warnings target)
  if(MSVC)
    # C4324: "structure was padded due to alignment specifier" is the intended effect of alignas.
    target_compile_options(${target} PRIVATE /W4 /permissive- /utf-8 /wd4324)
    # C4996 "unsafe" CRT functions (getenv, sscanf, ...): the portable standard functions are used deliberately; the
    # MSVC-only *_s replacements would fork the code per platform.
    target_compile_definitions(${target} PRIVATE _CRT_SECURE_NO_WARNINGS)
    if(CLUSTERLM_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE /WX)
    endif()
    target_compile_definitions(${target} PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX _WIN32_WINNT=0x0A00)
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion)
    if(CLUSTERLM_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE -Werror)
    endif()
  endif()
endfunction()

function(clusterlm_module target)
  cmake_parse_arguments(ARG "" "" "SOURCES;DEPS;PRIVATE_DEPS" ${ARGN})
  add_library(${target} STATIC ${ARG_SOURCES})
  target_include_directories(${target} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(${target} PUBLIC ${ARG_DEPS} PRIVATE ${ARG_PRIVATE_DEPS})
  clusterlm_set_warnings(${target})
endfunction()

# clusterlm_test(<name> [SLOW] SOURCES ... DEPS ...) — a doctest executable registered with CTest.
# SLOW tests carry the ctest label "slow": `ctest -LE slow` skips them, `ctest -L slow` selects them. They are
# registered as DISABLED when CLUSTERLM_RUN_SLOW_TESTS=OFF.
function(clusterlm_test name)
  cmake_parse_arguments(ARG "SLOW" "" "SOURCES;DEPS" ${ARGN})
  add_executable(${name} ${ARG_SOURCES})
  target_link_libraries(${name} PRIVATE ${ARG_DEPS} clusterlm_third_party clusterlm_test_main)
  clusterlm_set_warnings(${name})
  add_test(NAME ${name} COMMAND ${name})
  if(ARG_SLOW)
    set_tests_properties(${name} PROPERTIES LABELS slow)
    if(NOT CLUSTERLM_RUN_SLOW_TESTS)
      set_tests_properties(${name} PROPERTIES DISABLED TRUE)
    endif()
  endif()
endfunction()
