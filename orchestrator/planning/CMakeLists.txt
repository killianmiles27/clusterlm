clusterlm_module(clusterlm_planning
  SOURCES src/planning.cpp src/routing.cpp src/replan.cpp src/report.cpp
  DEPS clusterlm_placement clusterlm_coordinator clusterlm_objects
  PRIVATE_DEPS clusterlm_third_party)
