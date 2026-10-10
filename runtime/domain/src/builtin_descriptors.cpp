// Embedded provisional backend descriptors. GENERATED from docs/interfaces/examples/backend-*.example.json by the author
// of workstream A and compared against those files by tests/domain/test_backend_descriptor.cpp so the two cannot drift.
// Workstream F owns the contents (docs/interfaces/backend-capability-v1.md §6): they are upper bounds from reading code
// tables, not verified support. Embedded on purpose: a missing file can never advertise a backend.
#include "builtin_descriptors.hpp"

namespace clusterlm::domain::detail {

const char* const kReferenceDescriptorJson = R"clusterlm-json({
  "schema": "clusterlm.backend-descriptor",
  "contract_version": 1,
  "id": "reference",
  "display_name": "Reference CPU (development)",
  "factory_name": "reference",
  "role_in_product": "development-only",
  "model_support": {
    "containers": [
      "gguf",
      "manifest-native"
    ],
    "tensor_formats": [
      "F32",
      "Q8_0"
    ],
    "families": [
      {
        "family_match": "*",
        "architecture": "clusterlm-fixture",
        "status": "Supported and qualified",
        "evidence": [
          "tests/domain",
          "tests/integration"
        ]
      }
    ],
    "unlisted_family_status": "Unsupported"
  },
  "devices": {
    "cpu": true,
    "gpu_vendors": [],
    "requires_runtime": []
  },
  "execution": {
    "single_host": true,
    "cross_machine": true,
    "validated_max_workers": 2,
    "validated_topologies": [
      "any split of the layer range; bitwise-identical logits"
    ],
    "layer_partitioning": "contiguous-layers",
    "cpu_gpu_hybrid": false,
    "token_free_middle_stages": true,
    "manual_layer_ranges": true,
    "worker_roles": [
      "middle"
    ]
  },
  "state": {
    "kinds": [
      "kv",
      "recurrent"
    ],
    "supports_rollback": "native",
    "max_sessions_per_domain": 1,
    "per_client_isolation": true
  },
  "speculation": {
    "window_commit_abort": true,
    "mtp": "full-distribution",
    "max_q": 4
  },
  "serving": {
    "full_logits_for_sampling": true,
    "constrained_decoding": false,
    "logprobs": false
  },
  "limitations": [
    "Deterministic FP32 fixture math. Never offered as a product profile; never evidence of real-model or CUDA behaviour."
  ],
  "qualification": {
    "label": "Supported and qualified",
    "notes": "Qualified only for what it is: the deterministic test backend. This label must never be shown beside a real model.",
    "hardware_experiments": []
  }
}
)clusterlm-json";

const char* const kLlamaDescriptorJson = R"clusterlm-json({
  "schema": "clusterlm.backend-descriptor",
  "contract_version": 1,
  "id": "llama-local",
  "display_name": "llama.cpp (Host-local)",
  "factory_name": "llama",
  "role_in_product": "production",
  "model_support": {
    "containers": [
      "gguf",
      "gguf-split"
    ],
    "tensor_formats": [
      "F32",
      "F16",
      "BF16",
      "Q4_0",
      "Q4_1",
      "Q5_0",
      "Q5_1",
      "Q8_0",
      "Q2_K",
      "Q3_K",
      "Q4_K",
      "Q5_K",
      "Q6_K",
      "IQ1_S",
      "IQ1_M",
      "IQ2_XXS",
      "IQ2_XS",
      "IQ2_S",
      "IQ3_XXS",
      "IQ3_S",
      "IQ4_NL",
      "IQ4_XS",
      "TQ1_0",
      "TQ2_0",
      "MXFP4"
    ],
    "families": [
      {
        "family_match": "Swift-1.5-Qwen3.8-27B",
        "status": "Supported, awaiting hardware qualification",
        "evidence": [
          "linux-llama-cpu: decode equals llama.cpp on fixture",
          "HQ-FAST-01"
        ]
      }
    ],
    "unlisted_family_status": "Experimental"
  },
  "devices": {
    "cpu": true,
    "gpu_vendors": [
      "nvidia"
    ],
    "min_compute_capability": "8.6",
    "requires_runtime": [
      "cuda>=12.0 (CLUSTERLM_LLAMA_CUDA build only)"
    ]
  },
  "execution": {
    "single_host": true,
    "cross_machine": false,
    "validated_max_workers": 0,
    "validated_topologies": [
      "host-only"
    ],
    "layer_partitioning": "none",
    "cpu_gpu_hybrid": true,
    "token_free_middle_stages": false,
    "manual_layer_ranges": false,
    "worker_roles": [],
    "evidence": [
      "tests/backends_llama"
    ]
  },
  "state": {
    "kinds": [
      "kv",
      "recurrent"
    ],
    "supports_rollback": "recompute",
    "per_client_isolation": true,
    "max_sessions_per_domain": 1
  },
  "speculation": {
    "window_commit_abort": true,
    "mtp": "none",
    "max_q": 32
  },
  "serving": {
    "full_logits_for_sampling": true,
    "constrained_decoding": false,
    "logprobs": false
  },
  "limitations": [
    "Host only; middle stages are refused (docs/backends/llama-local.md).",
    "Prefill windows longer than 32 positions return only the last position's logits (ADR 0300).",
    "tensor_formats lists the ggml types of the pinned llama.cpp (runtime/objects/ggml_types.cpp) minus non-weight types; F confirms which ones the pinned CPU/CUDA builds actually execute.",
    "Descriptor is provisional: workstream F reviews family list, per-client isolation and rollback mode against the pinned llama.cpp.",
    "Distributed llama.cpp (RPC) is a baseline harness only, not an advertised mode.",
    "Family entries match the user-editable family label until F adds the GGUF architecture to each entry (ADR 0407 residual risk)."
  ],
  "options_schema": {
    "n_threads": {
      "type": "integer",
      "minimum": 0,
      "maximum": 1024,
      "description": "0 = automatic; still capped by the Host resource policy."
    },
    "use_mmap": {
      "type": "boolean",
      "description": "Map the GGUF instead of reading it; memory admission is unchanged."
    }
  },
  "qualification": {
    "label": "Supported, awaiting hardware qualification",
    "hardware_experiments": [
      "HQ-FAST-01",
      "HQ-TIER-01"
    ],
    "notes": "CPU mechanics verified in CI on a fixture model; the real model on the target GPU is unmeasured."
  }
}
)clusterlm-json";

const char* const kStrataDescriptorJson = R"clusterlm-json({
  "schema": "clusterlm.backend-descriptor",
  "contract_version": 1,
  "id": "strata-hybrid",
  "display_name": "Flash-Next (Strata-derived)",
  "factory_name": "strata",
  "role_in_product": "production",
  "model_support": {
    "containers": [
      "gguf",
      "gguf-split"
    ],
    "tensor_formats": [
      "F32",
      "F16",
      "BF16",
      "Q4_0",
      "Q4_1",
      "Q5_0",
      "Q5_1",
      "Q8_0",
      "Q2_K",
      "Q3_K",
      "Q4_K",
      "Q5_K",
      "Q6_K",
      "IQ2_XXS",
      "IQ2_XS",
      "IQ2_S",
      "IQ3_XXS",
      "IQ3_S",
      "IQ1_S",
      "IQ1_M",
      "IQ4_NL",
      "IQ4_XS",
      "Q2_0"
    ],
    "families": [
      {
        "family_match": "*Flash-Next*",
        "architecture": "qwen4exp",
        "status": "Supported, awaiting hardware qualification",
        "evidence": [
          "linux-strata-cuda-build (compile only)",
          "fake-engine StrataDomain tests",
          "HQ-GPU-01",
          "HQ-NUM-01"
        ]
      }
    ],
    "unlisted_family_status": "Unsupported"
  },
  "devices": {
    "cpu": true,
    "gpu_vendors": [
      "nvidia"
    ],
    "min_compute_capability": "8.6",
    "requires_runtime": [
      "cuda>=12.0"
    ]
  },
  "execution": {
    "single_host": true,
    "cross_machine": true,
    "validated_max_workers": 2,
    "validated_topologies": [
      "host-prefix \u2192 worker \u2192 worker \u2192 host-tail (reference backend, bitwise equal to host-only)",
      "host-prefix \u2192 worker \u2192 host-tail"
    ],
    "layer_partitioning": "contiguous-layers",
    "cpu_gpu_hybrid": true,
    "token_free_middle_stages": true,
    "manual_layer_ranges": true,
    "worker_roles": [
      "middle"
    ],
    "evidence": [
      "tests/integration (reference backend)",
      "tests/coordinator",
      "HQ-P0B-01"
    ]
  },
  "state": {
    "kinds": [
      "kv",
      "recurrent",
      "indexer",
      "ple-history"
    ],
    "supports_rollback": "native",
    "per_client_isolation": true,
    "max_sessions_per_domain": 1
  },
  "speculation": {
    "window_commit_abort": true,
    "mtp": "one-hot-proposals",
    "max_q": 4
  },
  "serving": {
    "full_logits_for_sampling": true,
    "constrained_decoding": false,
    "logprobs": false
  },
  "limitations": [
    "CUDA engine compiled but never run on a GPU in this repository's CI (DEVELOPMENT-STATUS.md).",
    "MTP drafts verified as one-hot proposals; lower acceptance than optimal (ADR 0203).",
    "Prefill goes through the verifier in sub-batches of at most 8 positions (known issue 2).",
    "Worker count above 2 is not covered by any test; do not raise validated_max_workers without one.",
    "tensor_formats is the type table the Strata pin can map (strata-core object_map.cpp); only the tensor types of the two shipped artifacts have been exercised. F narrows it to what is executed."
  ],
  "options_schema": {
    "local_micro_batch": {
      "type": "integer",
      "minimum": 1,
      "maximum": 8,
      "description": "Upper bound on the local sub-batch; the effective value is still min(this, VRAM-headroom choice, engine window 8) (ADR 0250)."
    }
  },
  "qualification": {
    "label": "Supported, awaiting hardware qualification",
    "hardware_experiments": [
      "HQ-GPU-01",
      "HQ-GPU-02",
      "HQ-GPU-03",
      "HQ-NUM-01",
      "HQ-MTP-01",
      "HQ-P0B-01",
      "HQ-PERF-01"
    ],
    "notes": "All GPU behaviour is pending; fixture and fake-engine tests prove protocol and ownership logic only."
  }
}
)clusterlm-json";

}  // namespace clusterlm::domain::detail
