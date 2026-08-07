// Tenstorrent backend, mirroring CudaBackend.cu. Requires tt-metal.
//
// Blackhole has no FP64 fabric. That does not matter here: gather and scatter
// move bytes and do no arithmetic, so each double is treated as an opaque
// 8-byte payload.

#include "TenstorrentBackend.hh"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/circular_buffer_config.hpp>

using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

namespace {

// One element per DRAM page, so element index == page id. Costs 8x the host
// footprint at the 64 B alignment below; the alternative needs an integer
// divide per element, which distorts what a gather benchmark measures.
constexpr uint32_t kElemBytes = sizeof(double);

// A DRAM->L1 noc_async_read needs its L1 destination on the DRAM alignment
// boundary, 64 B on Blackhole. Packing slots tighter corrupts them silently.
constexpr uint32_t kSlotBytes = 64;

// Chunked so the index array need not fit in L1, and so a chunk's reads are
// issued before one barrier rather than one at a time.
constexpr uint32_t kChunk = 256;

// ---------------------------------------------------------------- device ctx
struct Context {
    std::shared_ptr<MeshDevice> mesh;
    CoreCoord grid{0, 0};

    // Deliberately never destroyed. tt-metal's own singletons have static
    // lifetime in libtt_metal, and the teardown order between them and ours is
    // unspecified, so releasing a MeshWorkload or MeshBuffer from a static
    // destructor can touch an already-dead device.
    static Context &instance() {
        static Context *ctx = new Context();
        static std::once_flag once;
        std::call_once(once, [] {
            const int device_id = [] {
                const char *e = std::getenv("SPATTER_TT_DEVICE");
                return e ? std::atoi(e) : 0;
            }();
            ctx->mesh = MeshDevice::create_unit_mesh(device_id);
            if (!ctx->mesh) {
                throw std::runtime_error("Tenstorrent: create_unit_mesh failed");
            }
            ctx->grid = ctx->mesh->compute_with_storage_grid_size();
        });
        return *ctx;
    }

    MeshCommandQueue &cq() { return mesh->mesh_command_queue(); }
    uint32_t cores() const { return grid.x * grid.y; }

private:
    Context() = default;
};

// Spatter stores device allocations in `double*` / `size_t*` slots. Those slots
// hold MeshBuffer* here; the owning shared_ptr lives in this map.
std::map<void *, std::shared_ptr<MeshBuffer>> &registry() {
    static auto *r = new std::map<void *, std::shared_ptr<MeshBuffer>>();
    return *r;
}
std::mutex &registry_mutex() {
    static std::mutex m;
    return m;
}

std::shared_ptr<MeshBuffer> lookup(const void *handle) {
    std::lock_guard<std::mutex> lock(registry_mutex());
    auto it = registry().find(const_cast<void *>(handle));
    if (it == registry().end()) {
        throw std::runtime_error("Tenstorrent: unknown device buffer handle");
    }
    return it->second;
}

// Embedded so the binary is self-contained. tt-metal JIT-compiles these for the
// data-movement RISC-V at first enqueue.
const char *kGatherKernel = R"KERNEL(
#include <cstdint>
#include "dataflow_api.h"

// dense[i*pattern_length + j] = sparse[pattern[j] + delta*i]
void kernel_main() {
    uint32_t sparse_addr  = get_arg_val<uint32_t>(0);
    uint32_t pattern_addr = get_arg_val<uint32_t>(1);
    uint32_t dense_addr   = get_arg_val<uint32_t>(2);
    uint32_t pattern_len  = get_arg_val<uint32_t>(3);
    uint32_t delta        = get_arg_val<uint32_t>(4);
    uint32_t i_start      = get_arg_val<uint32_t>(5);
    uint32_t i_count      = get_arg_val<uint32_t>(6);

    constexpr uint32_t ELEM_BYTES = get_compile_time_arg_val(0);
    constexpr uint32_t SLOT_BYTES = get_compile_time_arg_val(1);
    constexpr uint32_t WRITEBACK  = get_compile_time_arg_val(2);

    constexpr uint32_t cb_pattern = 0;
    constexpr uint32_t cb_scratch = 1;

    const InterleavedAddrGen<true> gp = {
        .bank_base_address = pattern_addr, .page_size = pattern_len * 4};
    const InterleavedAddrGen<true> gs = {
        .bank_base_address = sparse_addr, .page_size = ELEM_BYTES};
    const InterleavedAddrGen<true> gd = {
        .bank_base_address = dense_addr, .page_size = ELEM_BYTES};

    cb_reserve_back(cb_pattern, 1);
    const uint32_t pat_l1 = get_write_ptr(cb_pattern);
    noc_async_read(get_noc_addr(0, gp), pat_l1, pattern_len * 4);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t *pat =
        reinterpret_cast<volatile tt_l1_ptr uint32_t *>(pat_l1);

    cb_reserve_back(cb_scratch, 1);
    const uint32_t dst_l1 = get_write_ptr(cb_scratch);

    for (uint32_t k = 0; k < i_count; ++k) {
        const uint32_t i = i_start + k;
        const uint32_t base = delta * i;

        for (uint32_t j = 0; j < pattern_len; ++j) {
            noc_async_read(get_noc_addr(pat[j] + base, gs),
                           dst_l1 + j * SLOT_BYTES, ELEM_BYTES);
        }
        noc_async_read_barrier();

        if constexpr (WRITEBACK != 0) {
            const uint32_t out = i * pattern_len;
            for (uint32_t j = 0; j < pattern_len; ++j) {
                noc_async_write(dst_l1 + j * SLOT_BYTES,
                                get_noc_addr(out + j, gd), ELEM_BYTES);
            }
            noc_async_write_barrier();
        }
    }
}
)KERNEL";

// sparse[pattern[j] + delta*i] = dense[j + pattern_length*(i%wrap)]
const char *kScatterKernel = R"KERNEL(
#include <cstdint>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t sparse_addr  = get_arg_val<uint32_t>(0);
    uint32_t pattern_addr = get_arg_val<uint32_t>(1);
    uint32_t dense_addr   = get_arg_val<uint32_t>(2);
    uint32_t pattern_len  = get_arg_val<uint32_t>(3);
    uint32_t delta        = get_arg_val<uint32_t>(4);
    uint32_t i_start      = get_arg_val<uint32_t>(5);
    uint32_t i_count      = get_arg_val<uint32_t>(6);
    uint32_t wrap         = get_arg_val<uint32_t>(7);

    constexpr uint32_t ELEM_BYTES = get_compile_time_arg_val(0);
    constexpr uint32_t SLOT_BYTES = get_compile_time_arg_val(1);

    constexpr uint32_t cb_pattern = 0;
    constexpr uint32_t cb_scratch = 1;

    const InterleavedAddrGen<true> gp = {
        .bank_base_address = pattern_addr, .page_size = pattern_len * 4};
    const InterleavedAddrGen<true> gs = {
        .bank_base_address = sparse_addr, .page_size = ELEM_BYTES};
    const InterleavedAddrGen<true> gd = {
        .bank_base_address = dense_addr, .page_size = ELEM_BYTES};

    cb_reserve_back(cb_pattern, 1);
    const uint32_t pat_l1 = get_write_ptr(cb_pattern);
    noc_async_read(get_noc_addr(0, gp), pat_l1, pattern_len * 4);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t *pat =
        reinterpret_cast<volatile tt_l1_ptr uint32_t *>(pat_l1);

    cb_reserve_back(cb_scratch, 1);
    const uint32_t src_l1 = get_write_ptr(cb_scratch);

    for (uint32_t k = 0; k < i_count; ++k) {
        const uint32_t i = i_start + k;
        const uint32_t base = delta * i;
        const uint32_t in = pattern_len * (i % wrap);

        for (uint32_t j = 0; j < pattern_len; ++j) {
            noc_async_read(get_noc_addr(in + j, gd),
                           src_l1 + j * SLOT_BYTES, ELEM_BYTES);
        }
        noc_async_read_barrier();

        for (uint32_t j = 0; j < pattern_len; ++j) {
            noc_async_write(src_l1 + j * SLOT_BYTES,
                            get_noc_addr(pat[j] + base, gs), ELEM_BYTES);
        }
        noc_async_write_barrier();
    }
}
)KERNEL";

// ------------------------------------------------------------- program cache
struct Key {
    const char *kernel;
    uint32_t sparse, pattern, dense;      // device addresses
    uint32_t pattern_len, delta, wrap, count;
    bool operator<(const Key &o) const {
        return std::tie(kernel, sparse, pattern, dense, pattern_len, delta, wrap, count) <
               std::tie(o.kernel, o.sparse, o.pattern, o.dense, o.pattern_len, o.delta,
                        o.wrap, o.count);
    }
};

std::map<Key, std::shared_ptr<MeshWorkload>> &program_cache() {
    static auto *c = new std::map<Key, std::shared_ptr<MeshWorkload>>();
    return *c;
}

std::shared_ptr<MeshWorkload> get_or_build(const Key &key, bool is_scatter) {
    auto it = program_cache().find(key);
    if (it != program_cache().end()) {
        return it->second;
    }

    Context &ctx = Context::instance();
    const uint32_t gx = ctx.grid.x, gy = ctx.grid.y;
    const uint32_t ncores = gx * gy;

    Program program = CreateProgram();
    const CoreRange all(CoreCoord{0, 0}, CoreCoord{gx - 1, gy - 1});

    auto add_cb = [&](uint8_t index, uint32_t bytes) {
        CircularBufferConfig cfg(bytes, {{index, tt::DataFormat::UInt32}});
        cfg.set_page_size(index, bytes);
        CreateCircularBuffer(program, all, cfg);
    };
    add_cb(0, key.pattern_len * 4);                 // pattern, one page
    add_cb(1, key.pattern_len * kSlotBytes);        // gather/scatter staging

    DataMovementConfig dm{};
    dm.processor = DataMovementProcessor::RISCV_0;
    dm.noc = NOC::RISCV_0_default;
    dm.compile_args = is_scatter
        ? std::vector<uint32_t>{kElemBytes, kSlotBytes}
        : std::vector<uint32_t>{kElemBytes, kSlotBytes, /*WRITEBACK=*/1u};

    // The JIT's default -I list stops at tt_metal/hw/inc, so "dataflow_api.h"
    // does not resolve without these.
    if (const char *root = std::getenv("TT_METAL_RUNTIME_ROOT")) {
        const std::string api = std::string(root) + "/tt_metal/hw/inc/api";
        dm.compiler_include_paths = {api, api + "/dataflow", api + "/compute",
                                     api + "/tensor", api + "/debug", api + "/numeric"};
    }

    KernelHandle k = CreateKernelFromString(program, key.kernel, all, dm);

    const uint32_t base = key.count / ncores;
    const uint32_t rem = key.count % ncores;
    uint32_t start = 0;
    for (uint32_t c = 0; c < ncores; ++c) {
        const CoreCoord core{c % gx, c / gx};
        const uint32_t n = base + (c < rem ? 1u : 0u);
        std::vector<uint32_t> rt = {key.sparse, key.pattern, key.dense,
                                    key.pattern_len, key.delta, start, n};
        if (is_scatter) {
            rt.push_back(key.wrap);
        }
        SetRuntimeArgs(program, k, core, rt);
        start += n;
    }

    auto workload = std::make_shared<MeshWorkload>();
    workload->add_program(MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(0, 0)),
                          std::move(program));
    program_cache().emplace(key, workload);
    return workload;
}

uint32_t addr_of(const void *handle) {
    return static_cast<uint32_t>(lookup(handle)->address());
}

float launch(const Key &key, bool is_scatter) {
    Context &ctx = Context::instance();
    auto workload = get_or_build(key, is_scatter);   // JIT happens here, untimed

    const auto t0 = std::chrono::high_resolution_clock::now();
    EnqueueMeshWorkload(ctx.cq(), *workload, /*blocking=*/false);
    Finish(ctx.cq());
    const auto t1 = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<float, std::milli>(t1 - t0).count();
}

constexpr float kUnimplemented = -1.0f;

}  // namespace

// ---------------------------------------------------------------- public API
void *tt_device_alloc(size_t bytes, size_t page_size) {
    Context &ctx = Context::instance();

    // A page smaller than the 64 B DRAM alignment still occupies a full granule,
    // so an 8-byte-element buffer costs 8x its logical size. Check before asking:
    // the standard GPU test suite sizes its patterns for ~1e9 elements, which is
    // 8 GB on the host and 64 GB here, and a request that large is better
    // refused with a number than handed to the driver.
    const size_t pages = (bytes + page_size - 1) / page_size;
    const size_t granule = page_size < kSlotBytes
        ? kSlotBytes
        : (page_size + kSlotBytes - 1) / kSlotBytes * kSlotBytes;
    const size_t on_device = pages * granule;
    const size_t capacity = static_cast<size_t>(ctx.mesh->num_dram_channels()) *
                            ctx.mesh->dram_size_per_channel();
    if (capacity && on_device > capacity * 9 / 10) {
        throw std::runtime_error(
            "Tenstorrent: allocation of " + std::to_string(on_device >> 20) +
            " MiB exceeds device DRAM (" + std::to_string(capacity >> 20) +
            " MiB). " + std::to_string(bytes >> 20) + " MiB of " +
            std::to_string(page_size) + "-byte elements expands to " +
            std::to_string(granule) + " bytes each at the DRAM alignment. "
            "Reduce -l, or use a pattern whose elements pack into larger pages.");
    }

    DeviceLocalBufferConfig local{};
    local.page_size = page_size;
    local.buffer_type = BufferType::DRAM;
    ReplicatedBufferConfig global{};
    global.size = bytes;

    auto buf = MeshBuffer::create(global, local, ctx.mesh.get());
    void *handle = buf.get();
    {
        std::lock_guard<std::mutex> lock(registry_mutex());
        registry().emplace(handle, std::move(buf));
    }
    return handle;
}

void tt_device_free(void *handle) {
    if (!handle) {
        return;
    }
    std::lock_guard<std::mutex> lock(registry_mutex());
    registry().erase(handle);
}

void tt_memcpy_h2d(void *handle, const void *src, size_t bytes) {
    Context &ctx = Context::instance();
    auto buf = lookup(handle);
    std::vector<uint8_t> staging(static_cast<const uint8_t *>(src),
                                 static_cast<const uint8_t *>(src) + bytes);
    staging.resize(buf->size(), 0);
    EnqueueWriteMeshBuffer(ctx.cq(), buf, staging, /*blocking=*/true);
}

void tt_memcpy_d2h(void *dst, const void *handle, size_t bytes) {
    Context &ctx = Context::instance();
    auto buf = lookup(handle);
    std::vector<uint8_t> staging;
    EnqueueReadMeshBuffer(ctx.cq(), staging, buf, /*blocking=*/true);
    std::memcpy(dst, staging.data(), bytes);
}

// The device kernel indexes with uint32; 32 GB of DRAM cannot hold more than
// 2^32 8-byte pages anyway.
void tt_pattern_upload(void *handle, const size_t *pattern, size_t length) {
    std::vector<uint32_t> narrowed(length);
    for (size_t i = 0; i < length; ++i) {
        narrowed[i] = static_cast<uint32_t>(pattern[i]);
    }
    tt_memcpy_h2d(handle, narrowed.data(), length * sizeof(uint32_t));
}

float tt_gather_wrapper(const size_t *pattern, const double *sparse,
    double *dense, const size_t pattern_length, const size_t delta,
    const size_t wrap, const size_t count) {
    (void)wrap;
    Key key{kGatherKernel, addr_of(sparse), addr_of(pattern), addr_of(dense),
            static_cast<uint32_t>(pattern_length), static_cast<uint32_t>(delta),
            1u, static_cast<uint32_t>(count)};
    return launch(key, /*is_scatter=*/false);
}

float tt_scatter_wrapper(const size_t *pattern, double *sparse,
    const double *dense, const size_t pattern_length, const size_t delta,
    const size_t wrap, const size_t count) {
    Key key{kScatterKernel, addr_of(sparse), addr_of(pattern), addr_of(dense),
            static_cast<uint32_t>(pattern_length), static_cast<uint32_t>(delta),
            static_cast<uint32_t>(wrap), static_cast<uint32_t>(count)};
    return launch(key, /*is_scatter=*/true);
}

// Not implemented: atomics need a NoC atomic path, multi_* a second level of
// indirection. Neither is exercised by the stream or ustride suites.
float tt_scatter_atomic_wrapper(const size_t *, double *, const double *,
    const size_t, const size_t, const size_t, const size_t) {
    return kUnimplemented;
}
float tt_gather_scatter_wrapper(const size_t *, double *, const size_t *,
    const double *, const size_t, const size_t, const size_t, const size_t,
    const size_t) {
    return kUnimplemented;
}
float tt_gather_scatter_atomic_wrapper(const size_t *, double *, const size_t *,
    const double *, const size_t, const size_t, const size_t, const size_t,
    const size_t) {
    return kUnimplemented;
}
float tt_multi_gather_wrapper(const size_t *, const size_t *, const double *,
    double *, const size_t, const size_t, const size_t, const size_t) {
    return kUnimplemented;
}
float tt_multi_scatter_wrapper(const size_t *, const size_t *, double *,
    const double *, const size_t, const size_t, const size_t, const size_t) {
    return kUnimplemented;
}
float tt_multi_scatter_atomic_wrapper(const size_t *, const size_t *, double *,
    const double *, const size_t, const size_t, const size_t, const size_t) {
    return kUnimplemented;
}
