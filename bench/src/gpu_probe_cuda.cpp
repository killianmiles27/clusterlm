// CUDA implementation of GpuProbe (cudart + cuBLAS, plain C++ — no device code, no nvcc needed). Compiled only
// with CLUSTERLM_BENCH_CUDA=ON and a found CUDA toolkit. There is no GPU on the development host, so the
// runtime path of this file is exercised on the target machines; the no-device path (cudaGetDeviceCount
// failing or returning 0) is what runs elsewhere and reports a clean "no CUDA device".
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "clusterlm/common/clock.hpp"
#include "gpu_probe.hpp"
#include "memory_probe.hpp"

namespace clusterlm::bench {

namespace {

Status cuda_error(cudaError_t e, const char* what) {
  if (e == cudaSuccess) return Status::ok();
  ErrorCode code = ErrorCode::kInternal;
  if (e == cudaErrorNoDevice || e == cudaErrorInsufficientDriver || e == cudaErrorInvalidDevice)
    code = ErrorCode::kHardwareUnavailable;
  else if (e == cudaErrorMemoryAllocation || e == cudaErrorHostMemoryAlreadyRegistered)
    code = ErrorCode::kResourceExhausted;
  (void)cudaGetLastError();  // clear the sticky error so later probes start clean
  return make_error(code, std::string(what) + ": " + cudaGetErrorString(e));
}

Status cublas_error(cublasStatus_t s, const char* what) {
  if (s == CUBLAS_STATUS_SUCCESS) return Status::ok();
  return make_error(ErrorCode::kInternal, std::string(what) + ": cuBLAS status " + std::to_string(static_cast<int>(s)));
}

#define CLM_CUDA(call) CLM_RETURN_IF_ERROR(cuda_error((call), #call))
#define CLM_CUBLAS(call) CLM_RETURN_IF_ERROR(cublas_error((call), #call))

struct DeviceBuffer {
  void* ptr = nullptr;
  ~DeviceBuffer() {
    if (ptr) (void)cudaFree(ptr);
  }
  Status alloc(std::size_t bytes) { return cuda_error(cudaMalloc(&ptr, bytes), "cudaMalloc"); }
};

struct Events {
  cudaEvent_t a = nullptr, b = nullptr;
  Events() {
    (void)cudaEventCreate(&a);
    (void)cudaEventCreate(&b);
  }
  ~Events() {
    if (a) (void)cudaEventDestroy(a);
    if (b) (void)cudaEventDestroy(b);
  }
};

struct Stream {
  cudaStream_t s = nullptr;
  Status create() { return cuda_error(cudaStreamCreate(&s), "cudaStreamCreate"); }
  ~Stream() {
    if (s) (void)cudaStreamDestroy(s);
  }
};

class CudaProbe final : public GpuProbe {
 public:
  CudaProbe() {
    int n = 0;
    const cudaError_t e = cudaGetDeviceCount(&n);
    if (e != cudaSuccess) {
      reason_ = std::string("CUDA runtime reports no usable device: ") + cudaGetErrorString(e);
      (void)cudaGetLastError();
    } else if (n == 0) {
      reason_ = "no CUDA device";
    } else {
      count_ = n;
    }
  }
  bool available() const override { return count_ > 0; }
  std::string unavailable_reason() const override { return reason_; }

  Result<std::vector<GpuDeviceInfo>> enumerate() override {
    std::vector<GpuDeviceInfo> out;
    for (int i = 0; i < count_; ++i) {
      cudaDeviceProp prop{};
      CLM_CUDA(cudaGetDeviceProperties(&prop, i));
      CLM_CUDA(cudaSetDevice(i));
      GpuDeviceInfo d;
      d.index = i;
      d.name = prop.name;
      d.compute_major = prop.major;
      d.compute_minor = prop.minor;
      d.multiprocessors = prop.multiProcessorCount;
      std::size_t free_b = 0, total_b = 0;
      CLM_CUDA(cudaMemGetInfo(&free_b, &total_b));
      d.free_bytes = free_b;
      d.total_bytes = total_b;
      CLM_CUDA(cudaDriverGetVersion(&d.driver_version));
      CLM_CUDA(cudaRuntimeGetVersion(&d.runtime_version));
      out.push_back(std::move(d));
    }
    return out;
  }

  Result<std::vector<AllocOverheadPoint>> allocation_overhead(int device, const std::vector<std::uint64_t>& sizes,
                                                              unsigned reps) override {
    CLM_CUDA(cudaSetDevice(device));
    CLM_CUDA(cudaFree(nullptr));  // create the context outside the timed region
    std::size_t free_b = 0, total_b = 0;
    CLM_CUDA(cudaMemGetInfo(&free_b, &total_b));
    std::vector<AllocOverheadPoint> out;
    for (std::uint64_t size : sizes) {
      if (size > free_b / 10 * 8) continue;  // never ask for more than 80% of what is free
      AllocOverheadPoint pt;
      pt.bytes = size;
      for (unsigned r = 0; r < reps + 1; ++r) {
        void* p = nullptr;
        Stopwatch sw;
        CLM_CUDA(cudaMalloc(&p, static_cast<std::size_t>(size)));
        const double malloc_us = static_cast<double>(sw.elapsed_ns()) / 1000.0;
        sw.reset();
        CLM_CUDA(cudaFree(p));
        const double free_us = static_cast<double>(sw.elapsed_ns()) / 1000.0;
        if (r == 0) continue;  // first allocation of a size warms the allocator
        pt.malloc_us.add(malloc_us);
        pt.free_us.add(free_us);
      }
      out.push_back(std::move(pt));
    }
    return out;
  }

  Result<std::vector<TransferPoint>> transfers(int device, const std::vector<std::uint64_t>& rings,
                                               const std::vector<HostMemoryKind>& kinds, unsigned reps) override {
    CLM_CUDA(cudaSetDevice(device));
    Stream stream;
    CLM_RETURN_IF_ERROR(stream.create());
    Events ev;
    std::size_t free_b = 0, total_b = 0;
    CLM_CUDA(cudaMemGetInfo(&free_b, &total_b));
    const MemoryInfo mem = probe_memory();
    std::vector<TransferPoint> out;
    for (auto kind : kinds)
      for (std::uint64_t bytes : rings) {
        // Bounded: device memory (80% of free) and host memory (leave 2 GiB available) policies.
        if (bytes > free_b / 10 * 8) continue;
        if (mem.available_physical < bytes + (2ull << 30)) continue;
        DeviceBuffer dev;
        CLM_RETURN_IF_ERROR(dev.alloc(static_cast<std::size_t>(bytes)));
        void* host = nullptr;
        bool registered = false;
        if (kind == HostMemoryKind::kPinnedAlloc) {
          if (cudaHostAlloc(&host, static_cast<std::size_t>(bytes), cudaHostAllocDefault) != cudaSuccess) {
            (void)cudaGetLastError();
            continue;  // the pinned-limit probe reports this case
          }
        } else {
          host = std::malloc(static_cast<std::size_t>(bytes));
          if (!host) continue;
          std::memset(host, 0x5A, static_cast<std::size_t>(bytes));  // touch every page up front
          if (kind == HostMemoryKind::kPinnedRegister) {
            if (cudaHostRegister(host, static_cast<std::size_t>(bytes), cudaHostRegisterDefault) != cudaSuccess) {
              (void)cudaGetLastError();
              std::free(host);
              continue;
            }
            registered = true;
          }
        }
        auto release = [&] {
          if (kind == HostMemoryKind::kPinnedAlloc) (void)cudaFreeHost(host);
          else {
            if (registered) (void)cudaHostUnregister(host);
            std::free(host);
          }
        };
        TransferPoint pt;
        pt.bytes = bytes;
        pt.kind = kind;
        for (unsigned r = 0; r < reps + 1; ++r) {
          for (int dir = 0; dir < 2; ++dir) {
            cudaEventRecord(ev.a, stream.s);
            const cudaError_t e = dir == 0 ? cudaMemcpyAsync(dev.ptr, host, static_cast<std::size_t>(bytes), cudaMemcpyHostToDevice, stream.s)
                                           : cudaMemcpyAsync(host, dev.ptr, static_cast<std::size_t>(bytes), cudaMemcpyDeviceToHost, stream.s);
            cudaEventRecord(ev.b, stream.s);
            const cudaError_t s = cudaStreamSynchronize(stream.s);
            if (e != cudaSuccess || s != cudaSuccess) {
              release();
              return cuda_error(e != cudaSuccess ? e : s, "cudaMemcpyAsync");
            }
            float ms = 0;
            cudaEventElapsedTime(&ms, ev.a, ev.b);
            if (r == 0 || ms <= 0) continue;  // first pass warms mappings
            (dir == 0 ? pt.h2d_bytes_per_s : pt.d2h_bytes_per_s).add(static_cast<double>(bytes) / (static_cast<double>(ms) * 1e-3));
          }
        }
        release();
        out.push_back(std::move(pt));
      }
    return out;
  }

  Result<PinnedProbeResult> pinned_limit(int device, std::uint64_t cap_bytes, std::uint64_t step_bytes) override {
    CLM_CUDA(cudaSetDevice(device));
    if (step_bytes == 0) return make_error(ErrorCode::kInvalidArgument, "pinned step must be positive");
    PinnedProbeResult r;
    const MemoryInfo mem = probe_memory();
    const std::uint64_t policy = mem.available_physical > (2ull << 30) ? mem.available_physical - (2ull << 30) : 0;
    r.cap_bytes = std::min(cap_bytes, policy);
    std::vector<void*> chunks;
    bool failed = false;
    for (std::uint64_t total = step_bytes; total <= r.cap_bytes; total += step_bytes) {
      void* p = nullptr;
      const bool ok = cudaHostAlloc(&p, static_cast<std::size_t>(step_bytes), cudaHostAllocDefault) == cudaSuccess;
      if (!ok) (void)cudaGetLastError();
      r.steps.emplace_back(total, ok);
      if (!ok) {
        failed = true;
        break;
      }
      chunks.push_back(p);
      r.largest_ok_bytes = total;
    }
    r.stopped_by_cap = !failed;
    for (void* p : chunks) (void)cudaFreeHost(p);
    return r;
  }

  Result<std::vector<GemmPoint>> gemm(int device, const std::vector<GemmShape>& shapes,
                                      const std::vector<GemmPrecision>& precisions, unsigned reps) override {
    CLM_CUDA(cudaSetDevice(device));
    cublasHandle_t handle = nullptr;
    CLM_CUBLAS(cublasCreate(&handle));
    struct HandleGuard {
      cublasHandle_t h;
      ~HandleGuard() { cublasDestroy(h); }
    } guard{handle};
    Events ev;
    std::vector<GemmPoint> out;
    const int inner = 8;
    for (const auto& shape : shapes)
      for (auto prec : precisions) {
        const std::size_t elem = prec == GemmPrecision::kF16 ? 2 : 4;
        const std::size_t a_n = std::size_t{shape.rows} * shape.cols, b_n = std::size_t{shape.cols} * shape.batch,
                          c_n = std::size_t{shape.rows} * shape.batch;
        DeviceBuffer a, b, c;
        CLM_RETURN_IF_ERROR(a.alloc(a_n * elem));
        CLM_RETURN_IF_ERROR(b.alloc(b_n * elem));
        CLM_RETURN_IF_ERROR(c.alloc(c_n * elem));
        // Non-zero, finite operands (all-zero data would under-report power-limited throughput).
        auto fill = [&](DeviceBuffer& buf, std::size_t n) -> Status {
          std::vector<unsigned char> host(n * elem);
          for (std::size_t i = 0; i < n; ++i) {
            if (prec == GemmPrecision::kF16) {
              const std::uint16_t h = static_cast<std::uint16_t>(0x2800u + (i % 0x1000u));  // finite halves in [~0.03, 0.5]
              std::memcpy(&host[i * 2], &h, 2);
            } else {
              const float f = 0.01f + static_cast<float>(i % 97) * 0.001f;
              std::memcpy(&host[i * 4], &f, 4);
            }
          }
          return cuda_error(cudaMemcpy(buf.ptr, host.data(), host.size(), cudaMemcpyHostToDevice), "cudaMemcpy");
        };
        CLM_RETURN_IF_ERROR(fill(a, a_n));
        CLM_RETURN_IF_ERROR(fill(b, b_n));
        const float alpha = 1.0f, beta = 0.0f;
        const int m = static_cast<int>(shape.rows), n = static_cast<int>(shape.batch), k = static_cast<int>(shape.cols);
        // Weights are row-major [rows][cols] == column-major (cols x rows) with lda = cols; y = W x via OP_T.
        auto call = [&]() -> Status {
          if (prec == GemmPrecision::kF32) {
            if (n == 1)
              return cublas_error(cublasSgemv(handle, CUBLAS_OP_T, k, m, &alpha, static_cast<const float*>(a.ptr), k,
                                              static_cast<const float*>(b.ptr), 1, &beta, static_cast<float*>(c.ptr), 1),
                                  "cublasSgemv");
            return cublas_error(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, static_cast<const float*>(a.ptr), k,
                                            static_cast<const float*>(b.ptr), k, &beta, static_cast<float*>(c.ptr), m),
                                "cublasSgemm");
          }
          return cublas_error(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, a.ptr, CUDA_R_16F, k, b.ptr,
                                           CUDA_R_16F, k, &beta, c.ptr, CUDA_R_16F, m, CUBLAS_COMPUTE_32F,
                                           CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                              "cublasGemmEx");
        };
        for (int w = 0; w < 3; ++w) CLM_RETURN_IF_ERROR(call());
        CLM_CUDA(cudaDeviceSynchronize());
        GemmPoint pt;
        pt.shape = shape;
        pt.precision = prec;
        for (unsigned r = 0; r < reps; ++r) {
          cudaEventRecord(ev.a, nullptr);
          for (int i = 0; i < inner; ++i) CLM_RETURN_IF_ERROR(call());
          cudaEventRecord(ev.b, nullptr);
          CLM_CUDA(cudaEventSynchronize(ev.b));
          float ms = 0;
          cudaEventElapsedTime(&ms, ev.a, ev.b);
          pt.ms.add(static_cast<double>(ms) / inner);
        }
        const double med = pt.ms.median();
        if (med > 0) {
          pt.weight_bytes_per_s = static_cast<double>(a_n * elem) / (med * 1e-3);
          pt.tflops = 2.0 * static_cast<double>(a_n) * shape.batch / (med * 1e-3) / 1e12;
        }
        out.push_back(std::move(pt));
      }
    return out;
  }

  Result<std::vector<KernelTimePoint>> time_kernels(int device, const std::vector<GpuKernelCase>& cases, unsigned reps) override {
    CLM_CUDA(cudaSetDevice(device));
    Stream stream;
    CLM_RETURN_IF_ERROR(stream.create());
    Events ev;
    std::vector<KernelTimePoint> out;
    for (const auto& c : cases) {
      if (c.setup) CLM_RETURN_IF_ERROR(c.setup());
      KernelTimePoint pt;
      pt.name = c.name;
      Status st = c.launch(stream.s);  // warm-up
      CLM_CUDA(cudaStreamSynchronize(stream.s));
      for (unsigned r = 0; r < reps && st.is_ok(); ++r) {
        cudaEventRecord(ev.a, stream.s);
        st = c.launch(stream.s);
        cudaEventRecord(ev.b, stream.s);
        CLM_CUDA(cudaStreamSynchronize(stream.s));
        float ms = 0;
        cudaEventElapsedTime(&ms, ev.a, ev.b);
        pt.ms.add(static_cast<double>(ms));
      }
      if (c.teardown) c.teardown();
      CLM_RETURN_IF_ERROR(st);
      out.push_back(std::move(pt));
    }
    return out;
  }

 private:
  int count_ = 0;
  std::string reason_;
};

}  // namespace

std::unique_ptr<GpuProbe> make_gpu_probe() { return std::make_unique<CudaProbe>(); }
bool gpu_probe_built_with_cuda() { return true; }

}  // namespace clusterlm::bench
