# install() rules for the two Windows packages (docs/packaging.md, packaging/).
#
# Two install COMPONENTs, one per MSI:
#   father  clusterlm-father-agent, clusterlm-father, clusterlm-bench, clusterlm-model-inspect, (clusterlm-father-ui)
#   node    clusterlm-node-service, clusterlm-node, clusterlm-node-helper, (clusterlm-node-ui)
# Layout inside a component prefix:  bin/  catalog/  licenses/
#
# Targets are installed only when they exist at configure time, so a tree without the optional UI workstreams (or
# without the model inspector) still configures and installs. Workstreams that add a UI executable or its runtime
# files install them with `install(... COMPONENT father|node DESTINATION bin)`; the MSI picks up everything under the
# staged prefix. Include this file after every add_subdirectory() so the TARGET checks see all targets.
include_guard(GLOBAL)

# Installs the targets that exist; returns the ones that do not via <var>_MISSING for the configure log.
function(clusterlm_install_executables component)
  set(missing "")
  foreach(t ${ARGN})
    if(TARGET ${t})
      install(TARGETS ${t} RUNTIME DESTINATION bin COMPONENT ${component})
    else()
      list(APPEND missing ${t})
    endif()
  endforeach()
  if(missing)
    message(STATUS "install(${component}): skipping targets not in this build: ${missing}")
  endif()
endfunction()

clusterlm_install_executables(father
  clusterlm-father-agent clusterlm-father clusterlm-bench clusterlm-model-inspect clusterlm-father-ui)
clusterlm_install_executables(node
  clusterlm-node-service clusterlm-node clusterlm-node-helper clusterlm-node-ui)

# The shipped tier catalog (the fixture is the catalog: tiers are data, ADR 0170). Father only.
install(FILES ${PROJECT_SOURCE_DIR}/fixtures/catalog/clusterlm-catalog.json
        DESTINATION catalog COMPONENT father)

# Licenses and notices, both packages. nlohmann/json is compiled into the binaries; OpenSSL is linked statically
# on Windows (its license text is copied in by packaging/build-msi.ps1 from the vcpkg port). The pinned upstream
# licenses (third_party/upstream.json) are installed when the sources were fetched into this tree.
foreach(component father node)
  install(FILES ${PROJECT_SOURCE_DIR}/packaging/licenses/THIRD-PARTY-NOTICES.txt
          DESTINATION licenses COMPONENT ${component})
  install(FILES ${PROJECT_SOURCE_DIR}/third_party/nlohmann/LICENSE.MIT
          DESTINATION licenses RENAME nlohmann-json-LICENSE.MIT COMPONENT ${component})
  file(GLOB _upstream_dirs LIST_DIRECTORIES true ${PROJECT_SOURCE_DIR}/third_party/upstream/*)
  foreach(dir ${_upstream_dirs})
    get_filename_component(_name ${dir} NAME)
    foreach(lic LICENSE LICENSE.md LICENSE.txt)
      if(EXISTS ${dir}/${lic})
        install(FILES ${dir}/${lic} DESTINATION licenses RENAME upstream-${_name}-${lic} COMPONENT ${component})
        break()
      endif()
    endforeach()
  endforeach()
endforeach()
