clusterlm_module(clusterlm_scheduler
  SOURCES src/types.cpp src/resource_policy.cpp src/admission.cpp src/status.cpp src/fair_queue.cpp src/scheduler.cpp
  DEPS clusterlm_common Threads::Threads)
