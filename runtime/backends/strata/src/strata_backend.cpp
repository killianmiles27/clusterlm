// The Strata backend: a StrataEngine over the pinned, patched Strata engine, and the BackendAdapter that builds
// StrataDomains on it. Compiled only with CLUSTERLM_ENABLE_STRATA (CUDA). See strata_backend.hpp for what a domain
// owns, docs/backends/strata-port.md for the seams and ADRs 0200-0203 for the decisions.
//
// Everything a domain allocates hangs off its CudaStrataEngine; no pointer, stream, table or pool is shared with
// another domain (strata patch 0004 makes the Verifier's commit mode and embedding table per object). The one
// process-wide Strata setting a domain writes is the CPU expert layout (`expert_layout_set`): it describes the
// model, not a domain, and every domain of a process derives the identical table from its manifest.
#include "clusterlm/backends/strata_backend.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <map>
#include <optional>
#include <set>
#include <utility>
#include <vector>

#include "clusterlm/backends/strata/mtp_drafter.hpp"
#include "clusterlm/backends/strata/object_map.hpp"
#include "clusterlm/backends/strata/strata_domain.hpp"
#include "clusterlm/common/log.hpp"

#include <cuda_runtime.h>

#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/layout.hpp"
#include "strata/core/mtp.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/core/native_head.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/session.hpp"
#include "strata/core/verify.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/verify_kernels.hpp"

#ifndef CLUSTERLM_STRATA_PIN
#define CLUSTERLM_STRATA_PIN "unknown"  // set by CMake from third_party/upstream.json
#endif

namespace clusterlm::backends {
namespace {

namespace sc = ::strata::core;
namespace sk = ::strata::kernels;
namespace cs = ::clusterlm::backends::strata;
using domain::StageRole;
using objects::AllocationTarget;
using objects::ModelGeometry;

Status unavailable(std::string m) { return make_error(ErrorCode::kHardwareUnavailable, "strata backend: " + m); }
Status internal(std::string m) { return make_error(ErrorCode::kInternal, "strata backend: " + m); }
Status cuda_error(const char* what, cudaError_t e) {
  return make_error(ErrorCode::kInternal, std::string("strata backend: ") + what + ": " + cudaGetErrorString(e));
}

int cuda_device_count() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) {
    (void)cudaGetLastError();
    return 0;
  }
  return count;
}
bool cuda_device_available(int device) { return device >= 0 && device < cuda_device_count(); }

constexpr std::uint64_t kMiB = 1ull << 20;

// ---------------------------------------------------------------- geometry

// Strata's kernels are built for Flash-Next's geometry (strata/core/layout.hpp): the fields ClusterLM's geometry
// carries are mapped and checked, the rest are the artifact constants Strata's ModelGeometry defaults to.
Result<sc::ModelGeometry> strata_geometry(const ModelGeometry& g) {
  sc::ModelGeometry s;
  const sc::ModelGeometry ref;
  if (g.hidden_size != ref.n_embd || g.residual_streams != ref.hc || g.n_heads != ref.n_head ||
      g.n_kv_heads != ref.n_head_kv || g.head_dim != ref.head_dim || g.expert_ff != ref.n_ff ||
      g.n_active_experts != 10 || g.n_experts == 0)
    return make_error(ErrorCode::kVersionMismatch,
                      "strata backend: the pinned Strata kernels are built for the Flash-Next geometry (H 2560, hc 4, "
                      "24/2 heads x 256, expert ff 640, 10 active experts); this model differs");
  s.n_layers = g.n_layers;
  s.n_expert = g.n_experts;
  for (std::uint32_t l = 0; l < g.n_layers; ++l)
    if ((g.layer_kinds[l] == objects::LayerKind::kFullAttention) != sc::is_qsa_layer(s, l))
      return make_error(ErrorCode::kVersionMismatch, "strata backend: layer " + std::to_string(l) +
                                                         " breaks Strata's 3:1 GDN/QSA pattern (layer % 4 == 3 is QSA)");
  return s;
}

// ---------------------------------------------------------------- objects -> Strata structures

struct BoundDense {
  std::string object;
  cs::DenseObject parsed;
};

// The provisioned strata-dense objects as one Strata WeightSource (strata patch 0005): a synthesized index whose
// rows point into a virtual file made of the objects' payloads, so WeightTable::load copies straight from the
// lease-store / canonical-store bytes. Rows the engine serves natively and metadata-only rows are in `skip`.
class ObjectWeightSource final : public sc::WeightSource {
 public:
  void add(const cs::DenseObject& obj) {
    const std::uint64_t base = virtual_size_;
    segments_.push_back({base, obj.payload});
    virtual_size_ += (obj.payload.size() + 4095) / 4096 * 4096;
    align_ = std::max(align_, obj.align);
    for (const cs::DenseRow& r : obj.rows) {
      lines_ += cs::index_line(r, 0, base + r.payload_offset, 0);
      ++rows_;
      if (r.src_bytes == 0 || r.has_native()) skip_.insert(r.name);
    }
  }
  // A row the Strata structures need the shape of (e.g. output.weight on a domain without the head), no bytes.
  void add_metadata(const std::string& name, std::int64_t ne0, std::int64_t ne1) {
    cs::DenseRow r;
    r.name = name;
    r.kind = 0;
    r.ne0 = ne0;
    r.ne1 = ne1;
    lines_ += cs::index_line(r, 0, 0, 0);
    ++rows_;
    skip_.insert(name);
  }
  const std::set<std::string>& skip() const { return skip_; }

  std::string name() const override { return "clusterlm provisioned strata-dense objects"; }
  bool index_text(std::string& out, std::string& err) const override {
    if (rows_ == 0) {
      err = "no strata-dense rows";
      return false;
    }
    // The pool is recomputed by the loader (a skip set compacts the arena); it only has to be non-zero here.
    out = "# align " + std::to_string(align_) + " pool " + std::to_string(std::max<std::uint64_t>(virtual_size_, 1)) +
          " tensors " + std::to_string(rows_) + "\n" + lines_;
    return true;
  }
  bool read(int file_id, std::uint64_t off, void* dst, std::size_t n, std::string& err) const override {
    if (file_id != 0) {
      err = "unexpected pack file id";
      return false;
    }
    for (const Segment& s : segments_)
      if (off >= s.base && off - s.base <= s.bytes.size() && n <= s.bytes.size() - (off - s.base)) {
        std::memcpy(dst, s.bytes.data() + (off - s.base), n);
        return true;
      }
    err = "read outside the provisioned objects";
    return false;
  }

 private:
  struct Segment {
    std::uint64_t base;
    ByteSpan bytes;
  };
  std::vector<Segment> segments_;
  std::uint64_t virtual_size_ = 0;
  std::uint32_t align_ = 256;
  std::string lines_;
  std::size_t rows_ = 0;
  std::set<std::string> skip_;
};

// The CPU expert complement: Strata's pool reads each routed expert's blob in place from the provisioned object
// (no copy, no experts.bin). Experts of layers this domain does not own are absent (null), which Strata reports as
// a dispatch failure rather than computing from foreign memory.
class ResolverExpertSource final : public sc::ExpertSource {
 public:
  ResolverExpertSource(std::int64_t n_layers, std::int64_t n_expert)
      : n_expert_(n_expert), blobs_(static_cast<std::size_t>(n_layers * n_expert), nullptr) {}
  void set(std::int64_t layer, std::int64_t expert, const std::uint8_t* p) {
    blobs_[static_cast<std::size_t>(layer * n_expert_ + expert)] = p;
  }
  const std::uint8_t* blob(std::int64_t layer, std::int64_t expert) override {
    ++reads_;
    if (layer < 0 || expert < 0 || expert >= n_expert_) return nullptr;
    const std::size_t i = static_cast<std::size_t>(layer * n_expert_ + expert);
    return i < blobs_.size() ? blobs_[i] : nullptr;
  }
  std::int64_t reads() const override { return reads_; }

 private:
  std::int64_t n_expert_;
  std::vector<const std::uint8_t*> blobs_;
  std::int64_t reads_ = 0;
};

// Pinned, device-mapped host memory (the Verifier's hand-off buffers must be device-visible).
struct MappedBuffer {
  float* host = nullptr;
  float* dev = nullptr;
  std::size_t floats = 0;
  Status alloc(std::size_t n) {
    void* h = nullptr;
    const cudaError_t e = cudaHostAlloc(&h, n * sizeof(float), cudaHostAllocMapped | cudaHostAllocPortable);
    if (e != cudaSuccess) return cuda_error("hand-off allocation", e);
    void* d = nullptr;
    const cudaError_t e2 = cudaHostGetDevicePointer(&d, h, 0);
    if (e2 != cudaSuccess) {
      cudaFreeHost(h);
      return cuda_error("hand-off device alias", e2);
    }
    std::memset(h, 0, n * sizeof(float));
    host = static_cast<float*>(h);
    dev = static_cast<float*>(d);
    floats = n;
    return Status::ok();
  }
  void free() {
    if (host != nullptr) cudaFreeHost(host);
    host = dev = nullptr;
    floats = 0;
  }
};

// ---------------------------------------------------------------- the engine

class CudaStrataEngine final : public cs::StrataEngine, public cs::MtpEngine {
 public:
  CudaStrataEngine(objects::ModelManifest manifest, domain::DomainSpec spec, StrataBackendOptions options,
                   sc::ModelGeometry sg)
      : manifest_(std::move(manifest)), spec_(spec), options_(std::move(options)), sg_(sg) {
    caps_.max_window = std::min<std::uint32_t>(options_.max_window_cap, static_cast<std::uint32_t>(sk::kVerifyMaxT));
    caps_.needs_tokens = spec_.role == StageRole::kPrefix;
    caps_.has_head = spec_.role == StageRole::kTail;
    caps_.vocab = caps_.has_head ? manifest_.geometry.vocab_size : 0;
  }
  ~CudaStrataEngine() override { (void)release(); }

  cs::EngineCaps caps() const override { return caps_; }
  bool mtp_enabled() const { return spec_.role == StageRole::kTail && !options_.mtp_dir.empty(); }

  // ---- sizing (Strata's own arithmetic; no device needed) ----------------------------------------------------

  Result<domain::DomainRequirements> requirements(const std::vector<std::string>& names) const override {
    domain::DomainRequirements req;
    const ModelGeometry& g = manifest_.geometry;
    std::uint64_t max_blob = 0;
    for (const std::string& n : names) {
      const objects::ManifestObject* o = manifest_.find(n);
      if (o == nullptr) return make_error(ErrorCode::kNotFound, "manifest lacks required object '" + n + "'");
      const AllocationTarget t = target_of(n, o->kind);
      switch (o->kind) {
        case objects::ObjectKind::kRoutedExpert: {
          CLM_ASSIGN_OR_RETURN(cs::ExpertFormat f, cs::expert_format(o->representation, g));
          max_blob = std::max(max_blob, f.total());
          (t == AllocationTarget::kGpuResident ? req.gpu_weight_bytes : req.cpu_weight_bytes) += f.total();
          break;
        }
        case objects::ObjectKind::kEmbedding:
          req.cpu_weight_bytes += o->byte_size;  // NativeEmbed: mapped pinned host memory, read over PCIe
          break;
        default:
          // strata-dense containers: canonical planes + GGUF-form copies, uploaded to the device arena / NativeDense.
          // The engine form of a canonical row can differ from its pack bytes (bf16 re-rounding shrinks, fp16
          // scale widening grows); the exact arena is reported by read_metrics() after prepare.
          req.gpu_weight_bytes += o->byte_size;
          break;
      }
    }
    const std::int64_t k = g.n_active_experts;
    const std::int64_t lb = spec_.layers.begin;
    const std::int64_t le = spec_.layers.end;
    const std::uint64_t per_session = sc::session_bytes(sg_, spec_.max_context, k, lb, le);
    req.state_bytes = per_session * spec_.max_sessions;
    // Verifier::init's device arena and mapped staging (strata patch 0006), with this model's largest expert blob
    // as its PCIe staging slot size (the canonical layout's is what init_bytes sees before the layout is set).
    const int max_t = verifier_window();
    const std::int64_t vocab = g.vocab_size;
    const sc::Verifier::InitBytes vb = sc::Verifier::init_bytes(sg_, k, spec_.max_context, max_t, vocab);
    const std::uint64_t layout_blob = sk::cpu::expert_layout().max_blob;
    std::uint64_t window_dev = vb.device;
    if (max_blob > layout_blob) window_dev += 16 * (max_blob - layout_blob);  // Verifier::kStagingBlobs slots
    req.window_bytes = window_dev * spec_.max_sessions;
    const std::uint64_t hand = static_cast<std::uint64_t>(sk::kVerifyMaxT) *
                               static_cast<std::uint64_t>(sc::Verifier::handoff_floats(sg_)) * sizeof(float);
    req.staging_bytes = (vb.mapped + 2 * hand) * spec_.max_sessions + 24 * kMiB;  // + WeightTable's load staging
    // residency table + NativeDense/NativeHead activation scratch + the PLE run (prefix)
    const std::uint64_t res = static_cast<std::uint64_t>(g.n_layers) * g.n_experts * sizeof(std::int32_t);
    req.scratch_bytes = res + 2 * sk::native_q8_1_bytes(static_cast<int>(sg_.ssm_value_dim), 1);
    if (spec_.role == StageRole::kPrefix)
      req.scratch_bytes += sc::ple_run_scratch_bytes() + static_cast<std::uint64_t>(sk::NG_N_EMBD) * sizeof(float);
    return req;
  }

  // ---- prepare -------------------------------------------------------------------------------------------------

  Status prepare(const objects::ObjectResolver& resolver) override {
    if (prepared_) return make_error(ErrorCode::kFailedPrecondition, "strata backend: already prepared");
    if (!cuda_device_available(options_.cuda_device)) return unavailable("no usable CUDA device");
    if (spec_.role == StageRole::kPrefix && options_.ple_table_gguf.empty())
      return make_error(ErrorCode::kInvalidArgument,
                        "strata backend: the prefix needs the PLE n-gram table (StrataBackendOptions::ple_table_gguf); "
                        "running without the PLE is a different model");
    device_ = options_.cuda_device;
    const sc::OnDevice on(device_);
    if (const cudaError_t e = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking); e != cudaSuccess)
      return cuda_error("stream", e);
    const ModelGeometry& g = manifest_.geometry;

    // 1. resolve everything this domain owns (a kNotFound here is a plan error, never a cue to fetch elsewhere)
    std::vector<std::string> names = cs::required_objects(g, spec_, {mtp_enabled()});
    std::vector<BoundDense> dense;
    ResolverExpertSource* src = nullptr;
    expert_src_ = std::make_unique<ResolverExpertSource>(g.n_layers, g.n_experts);
    src = expert_src_.get();
    std::vector<std::tuple<std::int64_t, std::int64_t, const std::uint8_t*, std::uint64_t>> gpu_experts;
    std::map<std::uint32_t, cs::ExpertFormat> layer_fmt;
    for (const std::string& n : names) {
      CLM_ASSIGN_OR_RETURN(objects::ProvisionedObject p, resolver.resolve(n));
      if (p.entry == nullptr || p.entry->name != n || p.bytes.size() != p.entry->byte_size)
        return make_error(ErrorCode::kDataLoss, "strata backend: object '" + n + "' resolved inconsistently");
      resident_bytes_ += p.bytes.size();
      if (p.entry->kind == objects::ObjectKind::kRoutedExpert) {
        CLM_ASSIGN_OR_RETURN(cs::ExpertFormat f, cs::expert_format(p.entry->representation, g));
        if (f.total() != p.bytes.size())
          return make_error(ErrorCode::kDataLoss, "strata backend: expert '" + n + "' is not its formats' size");
        const std::uint32_t L = *p.entry->layer, E = *p.entry->expert;
        auto [it, fresh] = layer_fmt.emplace(L, f);
        if (!fresh && (it->second.gate_up != f.gate_up || it->second.down != f.down))
          return make_error(ErrorCode::kVersionMismatch, "strata backend: layer " + std::to_string(L) +
                                                             " mixes expert formats (Strata needs one per layer)");
        src->set(L, E, p.bytes.data());
        if (target_of(n, p.entry->kind, p.target) == AllocationTarget::kGpuResident)
          gpu_experts.emplace_back(L, E, p.bytes.data(), p.bytes.size());
        continue;
      }
      if (p.entry->representation.quant_type != cs::kDenseQuantType ||
          p.entry->representation.conversion_version != cs::kDenseConversionVersion)
        return make_error(ErrorCode::kVersionMismatch, "strata backend: object '" + n +
                                                           "' is not in the strata-dense representation");
      CLM_ASSIGN_OR_RETURN(cs::DenseObject d, cs::parse_dense_object(p.bytes));
      dense.push_back({n, std::move(d)});
    }

    // 2. the CPU expert layout (process-wide, model-derived): every layer gets its format; layers this process
    //    does not own take the format of an owned one (their blobs are never asked for).
    CLM_RETURN_IF_ERROR(install_expert_layout(layer_fmt));

    // 3. dense weights: one device arena from the provisioned objects (strata patch 0005)
    ObjectWeightSource ws;
    for (const BoundDense& d : dense) ws.add(d.parsed);
    if (caps_.has_head == false) ws.add_metadata("output.weight", g.hidden_size, g.vocab_size);  // Verifier::init's n_vocab
    std::string err;
    std::uint64_t pool = 0;
    if (!sc::WeightTable::pool_bytes(ws, pool, err, &ws.skip())) return internal("weight index: " + err);
    CLM_RETURN_IF_ERROR(check_vram(pool, "dense weight arena"));
    if (const cudaError_t e = cudaMalloc(&wt_arena_, std::max<std::uint64_t>(pool, 256)); e != cudaSuccess)
      return cuda_error("dense weight arena", e);
    wt_ = std::make_unique<sc::WeightTable>();
    if (!wt_->load(ws, wt_arena_, std::max<std::uint64_t>(pool, 256), err, &ws.skip()))
      return internal("dense weights: " + err);
    for (std::uint32_t L = spec_.layers.begin; L < spec_.layers.end; ++L)
      if (!sc::check_layer(*wt_, sg_, L, err)) return make_error(ErrorCode::kDataLoss, "strata backend: " + err);

    // 4. GGUF-form tensors: the layers' projections (NativeDense), the head (tail), the embedding (prefix / MTP)
    native_ = std::make_unique<sc::NativeDense>();
    head_ = std::make_unique<sc::NativeHead>();
    embed_ = std::make_unique<sc::NativeEmbed>();
    std::vector<sc::NativeTensorView> views;
    for (const BoundDense& d : dense)
      for (const cs::DenseRow& r : d.parsed.rows) {
        if (!r.has_native()) continue;
        const ByteSpan b = d.parsed.native(r);
        if (r.name == "output.weight") {
          if (!head_->load_tensor(r.native_type, b.data(), b.size(), static_cast<std::int64_t>(r.native_ne0),
                                 static_cast<std::int64_t>(r.native_ne1), err))
            return internal(err);
        } else if (r.name == "token_embd.weight") {
          if (!embed_->load_tensor(r.native_type, b.data(), b.size(), static_cast<std::int64_t>(r.native_ne0),
                                  static_cast<std::int64_t>(r.native_ne1), err))
            return internal(err);
        } else {
          views.push_back({r.name, r.native_type, {r.native_ne0, r.native_ne1}, b.data(), b.size()});
        }
      }
    if (!views.empty() && !native_->load_tensors(views, *wt_, err, false, spec_.layers.begin, spec_.layers.end))
      return internal(err);
    if (caps_.has_head && !head_->loaded()) return make_error(ErrorCode::kDataLoss, "strata backend: the head object has no GGUF-form output.weight");

    // 5. routed experts: GPU-resident ones into a VRAM tier, the rest stay where the resolver keeps them (CPU pool)
    CLM_RETURN_IF_ERROR(build_expert_tier(gpu_experts));
    pool_ = std::make_unique<sk::cpu::ExpertPool>(static_cast<int>(options_.cpu_threads));
    dispatch_.pool = pool_.get();
    dispatch_.src = src;
    dispatch_.n_expert = g.n_experts;
    dispatch_.host_res = host_res_.data();
    dispatch_.cache = &cache_;
    dispatch_.cache_base = cache_.device_slot(0);
    dispatch_.cache_blob = static_cast<std::int64_t>(sk::cpu::expert_layout().max_blob);
    dispatch_.cache_slot_off = cache_.slot_offsets();
    dispatch_.pcie_num = 0;  // the provisioned objects are not CUDA-registered: no PCIe share of the misses

    // 6. the PLE table (prefix): Father-local, read in place; the PLE weights are layer 1's rows
    if (spec_.role == StageRole::kPrefix) CLM_RETURN_IF_ERROR(open_ple());
    // 7. session slots: allocated per open session (state for [layers.begin, layers.end) only)
    slots_.resize(spec_.max_sessions);
    prepared_ = true;
    log::info("strata_domain_prepared", {{"layers", std::to_string(spec_.layers.begin) + "-" + std::to_string(spec_.layers.end)},
                                         {"dense_arena_mib", std::to_string(pool / kMiB)},
                                         {"gpu_experts", std::to_string(gpu_experts.size())}});
    return Status::ok();
  }

  // ---- sessions ------------------------------------------------------------------------------------------------

  Status open_session(SessionId session) override {
    if (!prepared_) return make_error(ErrorCode::kFailedPrecondition, "strata backend: not prepared");
    const sc::OnDevice on(device_);
    Slot* s = find(session);
    if (s != nullptr) {  // re-open: drop any window, zero the state
      std::string err;
      if (s->ver->window_outstanding() && !s->ver->abort_window(err)) return internal(err);
      return zero(*s);
    }
    for (Slot& f : slots_)
      if (!f.used) return init_slot(f, session);
    return make_error(ErrorCode::kResourceExhausted, "strata backend: no free session slot");
  }

  Status close_session(SessionId session) override {
    Slot* s = find(session);
    if (s == nullptr) return Status::ok();
    const sc::OnDevice on(device_);
    free_slot(*s);
    return Status::ok();
  }

  // ---- windows -------------------------------------------------------------------------------------------------

  Status run(const cs::EngineWindow& w, domain::StageTiming& timing) override {
    Slot* s = find(w.session);
    if (s == nullptr) return make_error(ErrorCode::kNotFound, "strata backend: session not open in the engine");
    if (w.positions == 0 || w.positions > caps_.max_window) return internal("engine window size out of range");
    const sc::OnDevice on(device_);
    const std::size_t hb = static_cast<std::size_t>(sc::Verifier::handoff_floats(sg_));
    const std::size_t floats = hb * w.positions;
    if (spec_.layers.begin > 0) {
      if (w.handoff_in.size() != floats) return internal("hand-off input size");
      std::memcpy(s->hand_in.host, w.handoff_in.data(), floats * sizeof(float));
    }
    const std::int32_t* tokens = cs::verifier_tokens(caps_, w.tokens);  // nullptr on a token-free domain
    if (caps_.needs_tokens && tokens == nullptr) return internal("prefix window without tokens");
    if (options_.verifier_call_observer) options_.verifier_call_observer(tokens, static_cast<int>(w.positions));
    dispatch_.plan = s->ver->plan_sink();
    dispatch_.failed = false;
    dispatch_.fail = nullptr;
    const double cpu0 = s->ver->ms_pool, wait0 = s->ver->ms_wait;
    const auto t0 = std::chrono::steady_clock::now();
    std::int32_t out[sk::kVerifyMaxT] = {};
    std::string err;
    const bool ok = s->ver->run(static_cast<int>(w.positions), tokens, static_cast<std::int64_t>(w.pos0), &pool_fn, this,
                                out, err);
    timing.compute_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
    timing.cpu_expert_ns = static_cast<std::uint64_t>((s->ver->ms_pool - cpu0) * 1e6);
    timing.gpu_ns = static_cast<std::uint64_t>((s->ver->ms_wait - wait0) * 1e6);
    if (!ok) return internal("verify window: " + err);
    if (dispatch_.failed)
      return internal(std::string("CPU expert pool: ") + (dispatch_.fail != nullptr ? dispatch_.fail : "failed"));
    if (!w.handoff_out.empty()) {
      if (w.handoff_out.size() != floats) return internal("hand-off output size");
      std::memcpy(w.handoff_out.data(), s->hand_out.host, floats * sizeof(float));
    }
    if (caps_.has_head) {
      const std::size_t V = caps_.vocab;
      if (w.logits_out.size() != V * w.positions) return internal("logits buffer size");
      for (std::uint32_t t = 0; t < w.positions; ++t)
        if (!s->ver->copy_logits(static_cast<int>(t), w.logits_out.data() + t * V)) return internal("copy_logits");
    }
    s->last_pos0 = w.pos0;
    s->last_T = w.positions;
    s->window_valid = true;
    return Status::ok();
  }

  Status commit(SessionId session, std::uint32_t keep) override {
    Slot* s = find(session);
    if (s == nullptr) return make_error(ErrorCode::kNotFound, "strata backend: session not open in the engine");
    const sc::OnDevice on(device_);
    std::string err;
    // set_commit_sync(true) (patch 0004): commit() has waited for the commit graph when it returns.
    if (!s->ver->commit(static_cast<int>(keep), err) || !s->ver->wait_commit(err)) return internal("commit: " + err);
    s->last_keep = keep;
    return Status::ok();
  }

  Status abort(SessionId session) override {
    Slot* s = find(session);
    if (s == nullptr) return Status::ok();
    const sc::OnDevice on(device_);
    std::string err;
    if (!s->ver->abort_window(err)) return internal("abort_window: " + err);  // strata patch 0003
    s->window_valid = false;
    return Status::ok();
  }

  Status release() override {
    if (device_ < 0) return Status::ok();
    const sc::OnDevice on(device_);
    mtp_.reset();
    for (Slot& s : slots_)
      if (s.used) free_slot(s);
    slots_.clear();
    pool_.reset();
    cache_.close();
    if (d_res_ != nullptr) cudaFree(d_res_);
    d_res_ = nullptr;
    host_res_.clear();
    ple_table_.reset();
    if (ple_emb_dev_ != nullptr) cudaFree(ple_emb_dev_);
    if (ple_scratch_ != nullptr) cudaFree(ple_scratch_);
    ple_emb_dev_ = ple_scratch_ = nullptr;
    wt_.reset();
    if (wt_arena_ != nullptr) cudaFree(wt_arena_);
    wt_arena_ = nullptr;
    native_.reset();
    head_.reset();
    embed_.reset();
    expert_src_.reset();
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
    stream_ = nullptr;
    resident_bytes_ = 0;
    prepared_ = false;
    (void)cudaDeviceSynchronize();
    device_ = -1;
    return Status::ok();
  }

  cs::EngineCounters counters() const override {
    cs::EngineCounters c;
    c.resident_weight_bytes = resident_bytes_;
    c.session_state_bytes = sc::session_bytes(sg_, spec_.max_context, manifest_.geometry.n_active_experts,
                                              spec_.layers.begin, spec_.layers.end);
    // Verifier::init allocates exactly what init_bytes counts (the same carve): the window scratch of one open session.
    const sc::Verifier::InitBytes vb = sc::Verifier::init_bytes(sg_, manifest_.geometry.n_active_experts, spec_.max_context,
                                                                verifier_window(), manifest_.geometry.vocab_size);
    c.window_bytes_per_session = vb.device + vb.mapped;
    return c;
  }

  // ---- MTP (Father tail) ---------------------------------------------------------------------------------------

  std::uint32_t mtp_vocab() const override { return manifest_.geometry.vocab_size; }

  Status mtp_draft(std::span<const std::int32_t> committed, std::int32_t next_token, std::uint32_t count,
                   const domain::SamplingParams* sampling, std::vector<std::int32_t>& drafts,
                   std::vector<std::vector<float>>* probs) override {
    (void)sampling;
    (void)probs;  // Strata drafts greedily here; ClusterLM then verifies them as one-hot proposals (exact). ADR 0203.
    if (!mtp_enabled()) return make_error(ErrorCode::kFailedPrecondition, "strata backend: this tail has no MTP drafter");
    if (slots_.empty() || !slots_[0].used) return make_error(ErrorCode::kFailedPrecondition, "strata mtp: no open session");
    Slot& s = slots_[0];
    const sc::OnDevice on(device_);
    std::string err;
    if (!mtp_) CLM_RETURN_IF_ERROR(load_mtp(s));
    const std::uint64_t P = committed.size();
    if (!s.window_valid || s.last_keep == 0 || s.last_pos0 + s.last_keep != P)
      return make_error(ErrorCode::kFailedPrecondition,
                        "strata mtp: the tail's last committed window does not end at the committed history");
    const int T = static_cast<int>(s.last_T), a = static_cast<int>(s.last_keep) - 1;
    std::int32_t toks[sk::kVerifyMaxT] = {};
    for (int t = 0; t < T; ++t) {
      const std::uint64_t at = s.last_pos0 + static_cast<std::uint64_t>(t) + 1;
      toks[t] = at < P ? committed[at] : next_token;  // row t pairs R_{p+t} with the token at p+t+1
    }
    std::int32_t out[sk::kVerifyMaxT] = {};
    int n = 0;
    if (!mtp_->draft(T, toks, static_cast<std::int64_t>(s.last_pos0), a, out, err, nullptr, 0.0f, &n))
      return internal("mtp draft: " + err);
    drafts.assign(out, out + std::min<int>(n > 0 ? n : T - 1, static_cast<int>(count)));
    return Status::ok();
  }

 private:
  struct Slot {
    bool used = false;
    SessionId id;
    void* state_mem = nullptr;
    sc::SessionState ss;
    std::unique_ptr<sc::Verifier> ver;
    MappedBuffer hand_in, hand_out;
    std::uint64_t last_pos0 = 0;
    std::uint32_t last_T = 0, last_keep = 0;
    bool window_valid = false;
  };

  static void pool_fn(void* user, const float* x_f, const std::int32_t* ids, std::int64_t n_tok, std::int64_t k, float* out,
                      std::int64_t layer) {
    auto* self = static_cast<CudaStrataEngine*>(user);
    self->dispatch_.layers = layer;
    sc::expert_pool_dispatch_multi(self->dispatch_, x_f, ids, n_tok, k, out);
  }

  int verifier_window() const { return std::max(2, static_cast<int>(caps_.max_window)); }

  AllocationTarget target_of(const std::string& name, objects::ObjectKind kind,
                             std::optional<AllocationTarget> resolved = std::nullopt) const {
    (void)name;
    if (resolved && (*resolved == AllocationTarget::kGpuResident || *resolved == AllocationTarget::kCpuResident))
      return *resolved;
    return kind == objects::ObjectKind::kRoutedExpert ? AllocationTarget::kCpuResident : AllocationTarget::kGpuResident;
  }

  Status check_vram(std::uint64_t bytes, const char* what) const {
    std::size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return Status::ok();  // the allocation reports it then
    if (bytes + std::uint64_t{options_.vram_reserve_mib} * kMiB > free_b)
      return make_error(ErrorCode::kResourceExhausted,
                        std::string("strata backend: ") + what + " needs " + std::to_string(bytes / kMiB) + " MiB + " +
                            std::to_string(options_.vram_reserve_mib) + " MiB reserve; " + std::to_string(free_b / kMiB) +
                            " MiB free");
    return Status::ok();
  }

  Status install_expert_layout(const std::map<std::uint32_t, cs::ExpertFormat>& layer_fmt) {
    const ModelGeometry& g = manifest_.geometry;
    if (layer_fmt.empty()) return make_error(ErrorCode::kNotFound, "strata backend: no routed expert objects");
    sk::cpu::ExpertLayout lay;
    lay.native = true;
    lay.n_layers = g.n_layers;
    lay.n_expert = g.n_experts;
    lay.version = sk::cpu::kExpertLayoutVersion;
    lay.max_blob = 0;
    std::uint64_t at = 0;
    for (std::uint32_t l = 0; l < g.n_layers; ++l) {
      auto it = layer_fmt.lower_bound(l);
      if (it == layer_fmt.end()) it = std::prev(layer_fmt.end());
      const cs::ExpertFormat& f = it->second;
      sk::cpu::NativeFmt nf;
      std::string err;
      if (!sk::cpu::native_fmt(f.gate_up->id, f.down->id, g.hidden_size, g.expert_ff, nf, err))
        return make_error(ErrorCode::kVersionMismatch, "strata backend: " + err);
      if (nf.bytes != f.total()) return internal("expert blob size disagrees with Strata's native format");
      lay.fmt.push_back(nf);
      lay.offset.push_back(at);
      lay.bytes.push_back(nf.bytes);
      lay.gguf_off.assign(static_cast<std::size_t>(3 * g.n_layers), 0);
      at += nf.bytes * g.n_experts;
      lay.max_blob = std::max<std::uint64_t>(lay.max_blob, nf.bytes);
    }
    lay.total = at;
    std::string err;
    if (!sk::cpu::expert_layout_set(lay, err)) return internal(err);  // strata patch 0005
    return Status::ok();
  }

  Status build_expert_tier(
      const std::vector<std::tuple<std::int64_t, std::int64_t, const std::uint8_t*, std::uint64_t>>& gpu_experts) {
    const ModelGeometry& g = manifest_.geometry;
    const sk::cpu::ExpertLayout& lay = sk::cpu::expert_layout();
    // The Verifier needs a VRAM tier even when no expert is GPU-resident; one slot then stands empty.
    std::vector<std::int64_t> slot_bytes;
    std::uint64_t total = 0;
    for (const auto& [L, E, p, n] : gpu_experts) {
      (void)E;
      (void)p;
      (void)n;
      slot_bytes.push_back(static_cast<std::int64_t>(lay.bytes[static_cast<std::size_t>(L)]));
      total += lay.bytes[static_cast<std::size_t>(L)];
    }
    if (slot_bytes.empty()) slot_bytes.push_back(static_cast<std::int64_t>(lay.max_blob));
    CLM_RETURN_IF_ERROR(check_vram(total, "VRAM expert tier"));
    std::string err;
    if (!cache_.open_sized(slot_bytes, g.n_layers, g.n_experts, err)) return internal("expert tier: " + err);
    for (const auto& [L, E, p, n] : gpu_experts) {
      const std::int32_t slot = cache_.admit(L, E);
      if (slot == sc::kNotResident) return internal("expert tier ran out of slots");
      // bounded staging: the driver stages pageable memory through its own pinned buffers, one blob at a time
      if (!cache_.fill_slot_queued(slot, p, err, static_cast<std::int64_t>(n))) return internal("expert upload: " + err);
    }
    if (!cache_.sync_queued(err)) return internal("expert upload: " + err);
    host_res_.assign(static_cast<std::size_t>(g.n_layers) * g.n_experts, sc::kNotResident);
    for (std::uint32_t l = 0; l < g.n_layers; ++l)
      for (std::uint32_t e = 0; e < g.n_experts; ++e)
        host_res_[static_cast<std::size_t>(l) * g.n_experts + e] = cache_.slot_of(l, e);
    if (const cudaError_t e = cudaMalloc(reinterpret_cast<void**>(&d_res_), host_res_.size() * sizeof(std::int32_t));
        e != cudaSuccess)
      return cuda_error("residency table", e);
    if (const cudaError_t e = cudaMemcpy(d_res_, host_res_.data(), host_res_.size() * sizeof(std::int32_t),
                                         cudaMemcpyHostToDevice);
        e != cudaSuccess)
      return cuda_error("residency table", e);
    return Status::ok();
  }

  Status open_ple() {
    std::string err;
    sk::PleIoOptions io;
    ple_table_ = std::make_unique<sk::PleTable>();
    if (!ple_table_->open(options_.ple_table_gguf.string(), err, io)) return internal("PLE table: " + err);
    const char* names[] = {"blk.1.ple_key.weight", "blk.1.ple_value.weight", "blk.1.ple_norm_key.weight",
                           "blk.1.ple_norm_query.weight", "blk.1.ple_norm_conv.weight", "blk.1.ple_conv1d.weight"};
    for (const char* n : names)
      if (wt_->find(n) == nullptr)
        return make_error(ErrorCode::kDataLoss, std::string("strata backend: layer 1 has no ") + n + " (the PLE cannot be wired)");
    if (cudaMalloc(reinterpret_cast<void**>(&ple_emb_dev_), static_cast<std::size_t>(sk::NG_N_EMBD) * 4) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&ple_scratch_), sc::ple_run_scratch_bytes()) != cudaSuccess)
      return internal("PLE buffers");
    ple_emb_host_.assign(static_cast<std::size_t>(sk::NG_N_EMBD), 0.0f);
    return Status::ok();
  }

  void wire_ple(sc::SessionState& ss) {
    const sc::WeightRef* wk = wt_->find("blk.1.ple_key.weight");
    if (!wk->quantized()) {
      ss.ple.w.key_bf16 = static_cast<const std::uint16_t*>(wk->data);
    } else if (wk->native_data != nullptr) {
      ss.ple.w.key_native_data = wk->native_data;
      ss.ple.w.key_native_type = wk->native_type;
      ss.ple.w.key_native_q8_1 = wk->native_q8_1;
    } else {
      ss.ple.w.key_codes = static_cast<const std::uint8_t*>(wk->data);
      ss.ple.w.key_scales = reinterpret_cast<const float*>(static_cast<const std::uint8_t*>(wk->data) + wk->codes_bytes);
    }
    ss.ple.w.value_bf16 = static_cast<const std::uint16_t*>(wt_->find("blk.1.ple_value.weight")->data);
    ss.ple.w.norm_key = static_cast<const float*>(wt_->find("blk.1.ple_norm_key.weight")->data);
    ss.ple.w.norm_query = static_cast<const float*>(wt_->find("blk.1.ple_norm_query.weight")->data);
    ss.ple.w.norm_conv = static_cast<const float*>(wt_->find("blk.1.ple_norm_conv.weight")->data);
    ss.ple.w.conv1d_f16 = static_cast<const std::uint16_t*>(wt_->find("blk.1.ple_conv1d.weight")->data);
    ss.ple.consts = sk::ple_artifact_consts();
    ss.ple.table = ple_table_.get();
    ss.ple.token = &ss.ple_token;
    ss.ple.prev = ss.ple_prev;
    ss.ple.emb_host = ple_emb_host_.data();
    ss.ple.emb_dev = ple_emb_dev_;
    ss.ple.scratch = ple_scratch_;
    ss.ple.hist = ss.ple_hist;
  }

  Slot* find(SessionId id) {
    for (Slot& s : slots_)
      if (s.used && s.id == id) return &s;
    return nullptr;
  }

  Status zero(Slot& s) {
    sc::session_zero(s.ss, sg_, nullptr, stream_);
    s.ss.ple_prev[0] = s.ss.ple_prev[1] = -1;
    s.ss.ple_token = -1;
    s.window_valid = false;
    s.last_keep = 0;
    if (const cudaError_t e = cudaStreamSynchronize(stream_); e != cudaSuccess) return cuda_error("session zero", e);
    return Status::ok();
  }

  Status init_slot(Slot& s, SessionId id) {
    const std::int64_t k = manifest_.geometry.n_active_experts;
    const std::int64_t lb = spec_.layers.begin, le = spec_.layers.end;
    const std::uint64_t bytes = sc::session_bytes(sg_, spec_.max_context, k, lb, le);
    CLM_RETURN_IF_ERROR(check_vram(bytes, "session state"));
    if (const cudaError_t e = cudaMalloc(&s.state_mem, bytes); e != cudaSuccess) return cuda_error("session state", e);
    s.ss = sc::SessionState{};
    if (sc::session_init(sg_, spec_.max_context, k, s.state_mem, s.ss, lb, le) == 0) {
      free_slot(s);
      return internal("session_init failed");
    }
    if (spec_.role == StageRole::kPrefix) wire_ple(s.ss);
    s.used = true;
    s.id = id;
    CLM_RETURN_IF_ERROR(zero(s));
    const std::size_t hb = static_cast<std::size_t>(sk::kVerifyMaxT) * static_cast<std::size_t>(sc::Verifier::handoff_floats(sg_));
    if (lb > 0) CLM_RETURN_IF_ERROR(s.hand_in.alloc(hb));
    if (!caps_.has_head) CLM_RETURN_IF_ERROR(s.hand_out.alloc(hb));
    s.ver = std::make_unique<sc::Verifier>();
    s.ver->set_stage(lb, caps_.has_head ? -1 : le, s.hand_in.dev, s.hand_out.dev);
    s.ver->set_self_commit(false);  // patch 0003: every window can be aborted
    s.ver->set_commit_sync(true);   // patch 0004: commit returns once its graph ran (the ack waits for it)
    if (embed_ && embed_->bytes() > 0) s.ver->set_embed(embed_.get());
    s.ver->set_head_sampling(false);  // Father samples from the logits itself
    sc::VerifyHits hits;
    hits.d_res = d_res_;
    hits.h_res = host_res_.data();
    hits.cache_base = cache_.device_slot(0);
    hits.blob = static_cast<std::int64_t>(sk::cpu::expert_layout().max_blob);
    hits.slot_off = cache_.slot_offsets();
    hits.n_slots = cache_.slots();
    std::string err;
    if (!s.ver->init(*wt_, sg_, s.ss, hits, caps_.has_head ? head_.get() : nullptr, verifier_window(), err)) {
      free_slot(s);
      return internal("verifier: " + err);
    }
    return Status::ok();
  }

  void free_slot(Slot& s) {
    if (s.ver) {
      std::string err;
      (void)s.ver->wait_commit(err);
    }
    if (mtp_ && &s == &slots_[0]) mtp_.reset();
    s.ver.reset();
    s.hand_in.free();
    s.hand_out.free();
    if (s.state_mem != nullptr) {
      sc::session_release(s.ss);
      delete[] s.ss.qsa_states;
      s.ss.qsa_states = nullptr;
      cudaFree(s.state_mem);
    }
    s.state_mem = nullptr;
    s.used = false;
    s.window_valid = false;
  }

  Status load_mtp(Slot& s) {
    std::string err;
    mtp_ = std::make_unique<sc::MtpDrafter>();
    if (embed_ && embed_->bytes() > 0) mtp_->set_embed(embed_.get());  // patch 0004: this domain's own table
    if (!mtp_->load(options_.mtp_dir.string(), sg_, s.ss, verifier_window(), err)) {
      mtp_.reset();
      return internal("mtp load: " + err);
    }
    if (!mtp_->bind(*wt_, head_.get(), s.ver->final_R_all(), err)) {
      mtp_.reset();
      return internal("mtp bind: " + err);
    }
    return Status::ok();
  }

  objects::ModelManifest manifest_;
  domain::DomainSpec spec_;
  StrataBackendOptions options_;
  sc::ModelGeometry sg_;
  cs::EngineCaps caps_;
  int device_ = -1;
  bool prepared_ = false;
  cudaStream_t stream_ = nullptr;
  std::uint64_t resident_bytes_ = 0;

  void* wt_arena_ = nullptr;
  std::unique_ptr<sc::WeightTable> wt_;
  std::unique_ptr<sc::NativeDense> native_;
  std::unique_ptr<sc::NativeHead> head_;
  std::unique_ptr<sc::NativeEmbed> embed_;
  std::unique_ptr<ResolverExpertSource> expert_src_;
  sc::ExpertCache cache_;
  std::vector<std::int32_t> host_res_;
  std::int32_t* d_res_ = nullptr;
  std::unique_ptr<sk::cpu::ExpertPool> pool_;
  sc::ExpertDispatch dispatch_;
  std::unique_ptr<sk::PleTable> ple_table_;
  std::vector<float> ple_emb_host_;
  float* ple_emb_dev_ = nullptr;
  float* ple_scratch_ = nullptr;
  std::vector<Slot> slots_;
  std::unique_ptr<sc::MtpDrafter> mtp_;
};

// ---------------------------------------------------------------- the adapter

class StrataBackend final : public domain::BackendAdapter {
 public:
  explicit StrataBackend(StrataBackendOptions options) : options_(std::move(options)) {}

  domain::BackendInfo info() const override {
    domain::BackendInfo i;
    i.name = "strata";
    i.build_hash = std::string("strata@") + CLUSTERLM_STRATA_PIN + "+clusterlm-patches";
    i.supports_gpu = true;
    i.hardware_available = cuda_device_available(options_.cuda_device);
    return i;
  }

  Result<std::unique_ptr<domain::ExecutionDomain>> create_domain(const objects::ModelManifest& manifest,
                                                                 const domain::DomainSpec& spec) override {
    if (spec.layers.empty() && spec.role != StageRole::kTail)
      return make_error(ErrorCode::kInvalidArgument, "strata backend: empty layer range");
    if (options_.max_window_cap == 0 || options_.max_window_cap > static_cast<std::uint32_t>(sk::kVerifyMaxT))
      return make_error(ErrorCode::kInvalidArgument, "strata backend: max_window_cap must be 1..8 (Strata kVerifyMaxT)");
    CLM_ASSIGN_OR_RETURN(sc::ModelGeometry sg, strata_geometry(manifest.geometry));
    StrataBackendOptions o = options_;
    if (spec.role != StageRole::kPrefix) o.ple_table_gguf.clear();  // Father files never reach other roles
    if (spec.role != StageRole::kTail) o.mtp_dir.clear();
    const bool with_mtp = spec.role == StageRole::kTail && !o.mtp_dir.empty();
    auto engine = std::make_unique<CudaStrataEngine>(manifest, spec, std::move(o), sg);
    cs::StrataDomainOptions dopt;
    dopt.objects.with_mtp = with_mtp;
    CLM_ASSIGN_OR_RETURN(std::unique_ptr<cs::StrataDomain> d, cs::StrataDomain::create(manifest, spec, std::move(engine), dopt));
    return std::unique_ptr<domain::ExecutionDomain>(std::move(d));
  }

 private:
  StrataBackendOptions options_;
};

}  // namespace

std::unique_ptr<domain::BackendAdapter> make_strata_backend(const StrataBackendOptions& options) {
  return std::make_unique<StrataBackend>(options);
}

Result<std::unique_ptr<domain::Drafter>> make_strata_mtp_drafter(domain::ExecutionDomain& tail) {
  auto* d = dynamic_cast<cs::StrataDomain*>(&tail);
  if (d == nullptr || d->spec().role != StageRole::kTail)
    return make_error(ErrorCode::kInvalidArgument, "strata mtp: needs a Strata tail domain");
  auto* engine = dynamic_cast<CudaStrataEngine*>(&d->engine());
  if (engine == nullptr || !engine->mtp_enabled())
    return make_error(ErrorCode::kFailedPrecondition, "strata mtp: the tail was created without StrataBackendOptions::mtp_dir");
  return std::unique_ptr<domain::Drafter>(std::make_unique<cs::StrataMtpDrafter>(*engine));
}

}  // namespace clusterlm::backends
