// Embedded Fast/Strong/Ultra example profile set. GENERATED from docs/interfaces/examples/profile-{fast,strong,ultra}.example.json
// and compared by tests/migration against the profiles migration derives from the shipped tier catalog
// (fixtures/catalog/clusterlm-catalog.json), so the three sources cannot drift. Embedded so a fresh install and a migrated
// install seed the identical set without reading any file.
#include "clusterlm/migration/migration.hpp"

namespace clusterlm::migration {

static const char* const kFastJson = R"clusterlm-json({
  "schema": "clusterlm.profile",
  "schema_version": 1,
  "id": "prof_example_fast",
  "name": "Fast",
  "description": "Whole model on the Host (migrated from tier 'fast').",
  "revision": 1,
  "example_of": "tier:fast",
  "qualification_experiments": [
    "HQ-FAST-01"
  ],
  "model": {
    "identity": {
      "family": "Swift-1.5-Qwen3.8-27B",
      "display_name": "Swift-1.5-Qwen3.8-27B GSQ-RCO IQ3_S",
      "quant": "IQ3_S",
      "artifact_id": null,
      "expected_root_hash": null,
      "expected_files": [
        {
          "role": "model",
          "name": null,
          "approx_bytes": null
        }
      ]
    }
  },
  "backend": {
    "id": "llama-local",
    "min_contract_version": 1
  },
  "context": {
    "default_tokens": 4096,
    "max_tokens": 32768,
    "offered_profiles": [
      4096,
      8192,
      16384,
      32768
    ]
  },
  "topology": {
    "slots": [
      {
        "slot": "host",
        "kind": "host",
        "label": "Host"
      }
    ],
    "min_workers": 0,
    "max_workers": 0
  },
  "placement": {
    "mode": "auto",
    "objective": "balanced"
  },
  "speculation": {
    "enabled": false,
    "max_q": 1
  },
  "lifecycle": {
    "preparation": "on-demand",
    "release_after_idle_seconds": 60,
    "prepare_when_available": false
  },
  "on_worker_loss": {
    "max_retries": 0,
    "then": "stop",
    "preserve_conversation": true
  },
  "exposure": {
    "api": false,
    "api_model_id": "example-fast",
    "allow_lan": false
  }
}
)clusterlm-json";

static const char* const kStrongJson = R"clusterlm-json({
  "schema": "clusterlm.profile",
  "schema_version": 1,
  "id": "prof_example_strong",
  "name": "Strong",
  "description": "Host plus one laptop-class Worker (migrated from tier 'strong').",
  "revision": 1,
  "example_of": "tier:strong",
  "qualification_experiments": [
    "HQ-STRONG-01"
  ],
  "model": {
    "identity": {
      "family": "Swift-1.5-Qwen3.8-Flash-Next",
      "display_name": "Swift-1.5-Qwen3.8-Flash-Next GSQ-RCO IQ2_XS",
      "quant": "IQ2_XS",
      "artifact_id": null,
      "expected_root_hash": null,
      "expected_files": [
        {
          "role": "transformer-shard",
          "name": null,
          "approx_bytes": null
        },
        {
          "role": "lookup-shard",
          "name": null,
          "approx_bytes": null
        }
      ]
    }
  },
  "backend": {
    "id": "strata-hybrid",
    "min_contract_version": 1
  },
  "context": {
    "default_tokens": 4096,
    "max_tokens": 131072,
    "offered_profiles": [
      4096,
      8192,
      16384,
      32768,
      65536,
      131072
    ]
  },
  "topology": {
    "slots": [
      {
        "slot": "host",
        "kind": "host",
        "label": "Host"
      },
      {
        "slot": "w1",
        "kind": "worker",
        "label": "Laptop-class Worker",
        "select": {
          "mode": "binding",
          "binding": "node:laptop-class"
        }
      }
    ],
    "min_workers": 1,
    "max_workers": 1
  },
  "placement": {
    "mode": "auto",
    "objective": "balanced"
  },
  "speculation": {
    "enabled": true,
    "max_q": 4
  },
  "lifecycle": {
    "preparation": "on-demand",
    "release_after_idle_seconds": 60,
    "prepare_when_available": false
  },
  "on_worker_loss": {
    "max_retries": 1,
    "then": "fallback-profile",
    "fallback_profile_id": "prof_example_fast",
    "preserve_conversation": true
  },
  "exposure": {
    "api": false,
    "api_model_id": "example-strong",
    "allow_lan": false
  }
}
)clusterlm-json";

static const char* const kUltraJson = R"clusterlm-json({
  "schema": "clusterlm.profile",
  "schema_version": 1,
  "id": "prof_example_ultra",
  "name": "Ultra",
  "description": "Host plus laptop-class and designated-3060 Workers (migrated from tier 'ultra').",
  "revision": 1,
  "example_of": "tier:ultra",
  "qualification_experiments": [
    "HQ-PERF-01",
    "HQ-PERF-02",
    "HQ-PERF-03",
    "HQ-PERF-04"
  ],
  "model": {
    "identity": {
      "family": "Qwen3.8-Flash-Next",
      "display_name": "Qwen3.8-Flash-Next GSQ-RCO IQ3_S",
      "quant": "IQ3_S",
      "artifact_id": "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
      "expected_root_hash": null,
      "expected_files": [
        {
          "role": "transformer-shard",
          "name": null,
          "approx_bytes": 54800000000
        },
        {
          "role": "lookup-shard",
          "name": null,
          "approx_bytes": 28800000000
        }
      ]
    }
  },
  "backend": {
    "id": "strata-hybrid",
    "min_contract_version": 1
  },
  "context": {
    "default_tokens": 4096,
    "max_tokens": 131072,
    "offered_profiles": [
      4096,
      8192,
      16384,
      32768,
      65536,
      131072
    ]
  },
  "topology": {
    "slots": [
      {
        "slot": "host",
        "kind": "host",
        "label": "Host"
      },
      {
        "slot": "w1",
        "kind": "worker",
        "label": "Laptop-class Worker",
        "select": {
          "mode": "binding",
          "binding": "node:laptop-class"
        }
      },
      {
        "slot": "w2",
        "kind": "worker",
        "label": "Designated 3060 Worker",
        "select": {
          "mode": "binding",
          "binding": "node:designated-3060"
        }
      }
    ],
    "min_workers": 2,
    "max_workers": 2
  },
  "placement": {
    "mode": "auto",
    "objective": "balanced"
  },
  "speculation": {
    "enabled": true,
    "max_q": 4
  },
  "lifecycle": {
    "preparation": "on-demand",
    "release_after_idle_seconds": 60,
    "prepare_when_available": false
  },
  "on_worker_loss": {
    "max_retries": 1,
    "then": "fallback-profile",
    "fallback_profile_id": "prof_example_strong",
    "preserve_conversation": true
  },
  "exposure": {
    "api": false,
    "api_model_id": "example-ultra",
    "allow_lan": false
  },
  "goals": [
    {
      "metric": "decode_tok_s_median",
      "value": 20,
      "status": "pending_qualification",
      "experiment": "HQ-PERF-01"
    }
  ]
}
)clusterlm-json";

std::vector<profiles::Profile> example_profiles() {
  std::vector<profiles::Profile> out;
  for (const char* text : {kFastJson, kStrongJson, kUltraJson}) {
    auto p = profiles::profile_from_json(text);
    // Embedded text is verified by tests; a parse failure here would be a build defect, never silently skipped.
    if (p.is_ok()) out.push_back(std::move(p).value());
  }
  return out;
}

}  // namespace clusterlm::migration
