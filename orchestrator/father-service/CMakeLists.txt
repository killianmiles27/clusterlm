clusterlm_module(clusterlm_father_service
  SOURCES src/tokenizer.cpp src/gguf_bpe_tokenizer.cpp src/chat_template.cpp src/unicode_class.cpp src/unicode_tables.cpp src/tier_tokenizer.cpp src/service.cpp src/production.cpp src/node_notify.cpp
  DEPS clusterlm_common clusterlm_catalog clusterlm_coordinator clusterlm_domain clusterlm_config
       clusterlm_planning clusterlm_placement clusterlm_protocol clusterlm_transport clusterlm_objects)
