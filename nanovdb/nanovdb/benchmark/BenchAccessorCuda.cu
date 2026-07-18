// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// GPU (CUDA) counterpart of BenchAccessor.cc. It measures the same four access
// patterns against the same sphere grid, but on the device, so CPU and GPU
// ReadAccessor throughput can be compared side by side under both the OLD and
// NEW accessor-caching semantics (selected via the NANOVDB_USE_OLD_ACCESSOR /
// NANOVDB_NO_OLD_ACCESSOR compile definitions).
//
// To keep the accessor-cache reuse that the optimisation targets, each thread
// walks a *contiguous chunk* of the coordinate stream with its own accessor --
// mirroring the CPU's serial loop, but fanned out across many threads.

#include "BenchPatterns.h"

#include <nanovdb/GridHandle.h>
#include <nanovdb/cuda/DeviceBuffer.h>
#include <nanovdb/tools/CreatePrimitives.h>

#include <algorithm>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <vector>

// Number of consecutive coordinates each thread processes with a single
// accessor. Large enough to exercise cache reuse, small enough to keep the GPU
// saturated with threads.
static constexpr int CHUNK = 32;

// Each thread owns coords[base .. base+CHUNK) and accumulates their values.
__global__ void benchKernel(const nanovdb::NanoGrid<float>* grid,
                            const nanovdb::Coord*           coords,
                            int                             count,
                            float*                          out)
{
    const int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    const int base = tid * CHUNK;
    if (base >= count) return;

    auto  acc = grid->getAccessor();
    float sum = 0.0f;
    const int end = min(base + CHUNK, count);
    for (int i = base; i < end; ++i)
        sum += acc.getValue(coords[i]);
    out[tid] = sum;
}

// Each thread processes a chunk of centres; each centre expands into a 3x3x3
// (27-neighbour) lookup sharing the thread's accessor.
__global__ void stencilKernel(const nanovdb::NanoGrid<float>* grid,
                             const nanovdb::Coord*           centers,
                             int                             count,
                             float*                          out)
{
    const int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    const int base = tid * CHUNK;
    if (base >= count) return;

    auto      acc = grid->getAccessor();
    const int end = min(base + CHUNK, count);
    for (int i = base; i < end; ++i) {
        const nanovdb::Coord c = centers[i];
        float s = 0.0f;
        for (int dz = -1; dz <= 1; ++dz)
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx)
                    s += acc.getValue(c + nanovdb::Coord(dx, dy, dz));
        out[i] = s;
    }
}

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error " << cudaGetErrorString(err)               \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";         \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

// Time one pattern on the GPU; returns ns/access (total kernel time / N).
static double benchPattern(const nanovdb::NanoGrid<float>* dGrid,
                           const std::vector<nanovdb::Coord>& coords,
                           cudaStream_t stream, int nTrials = 7)
{
    const int   N       = static_cast<int>(coords.size());
    const int   nThread = (N + CHUNK - 1) / CHUNK;
    const int   block   = 128;
    const int   grid    = (nThread + block - 1) / block;

    nanovdb::Coord* dCoords = nullptr;
    float*          dOut    = nullptr;
    CUDA_CHECK(cudaMalloc(&dCoords, N * sizeof(nanovdb::Coord)));
    CUDA_CHECK(cudaMalloc(&dOut,    nThread * sizeof(float)));
    CUDA_CHECK(cudaMemcpyAsync(dCoords, coords.data(), N * sizeof(nanovdb::Coord),
                               cudaMemcpyHostToDevice, stream));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm-up launch (JIT / caches) not counted.
    benchKernel<<<grid, block, 0, stream>>>(dGrid, dCoords, N, dOut);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<double> times(nTrials);
    for (int t = 0; t < nTrials; ++t) {
        CUDA_CHECK(cudaEventRecord(start, stream));
        benchKernel<<<grid, block, 0, stream>>>(dGrid, dCoords, N, dOut);
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times[t] = (static_cast<double>(ms) * 1.0e6) / static_cast<double>(N); // ns/access
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(dCoords));
    CUDA_CHECK(cudaFree(dOut));

    std::nth_element(times.begin(), times.begin() + nTrials / 2, times.end());
    return times[nTrials / 2];
}

// Time the 3x3x3 stencil sweep on the GPU; returns ns per stencil (27 lookups).
static double benchStencil(const nanovdb::NanoGrid<float>* dGrid,
                           const std::vector<nanovdb::Coord>& centers,
                           cudaStream_t stream, int nTrials = 7)
{
    const int N       = static_cast<int>(centers.size());
    const int nThread = (N + CHUNK - 1) / CHUNK;
    const int block   = 128;
    const int grid    = (nThread + block - 1) / block;

    nanovdb::Coord* dCenters = nullptr;
    float*          dOut     = nullptr;
    CUDA_CHECK(cudaMalloc(&dCenters, N * sizeof(nanovdb::Coord)));
    CUDA_CHECK(cudaMalloc(&dOut,     N * sizeof(float)));
    CUDA_CHECK(cudaMemcpyAsync(dCenters, centers.data(), N * sizeof(nanovdb::Coord),
                               cudaMemcpyHostToDevice, stream));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    stencilKernel<<<grid, block, 0, stream>>>(dGrid, dCenters, N, dOut); // warm-up
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<double> times(nTrials);
    for (int t = 0; t < nTrials; ++t) {
        CUDA_CHECK(cudaEventRecord(start, stream));
        stencilKernel<<<grid, block, 0, stream>>>(dGrid, dCenters, N, dOut);
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times[t] = (static_cast<double>(ms) * 1.0e6) / static_cast<double>(N); // ns/stencil
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(dCenters));
    CUDA_CHECK(cudaFree(dOut));

    std::nth_element(times.begin(), times.begin() + nTrials / 2, times.end());
    return times[nTrials / 2];
}

int main()
{
    std::cout << "=== NanoVDB ReadAccessor GPU benchmark: " << bench::accessorMode() << " ===\n\n";

    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    if (devCount == 0) {
        std::cerr << "No CUDA device found\n";
        return 1;
    }
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Device: " << prop.name << " (SM " << prop.major << "." << prop.minor << ")\n";

    // Build the sphere with a device-capable buffer, then upload to the GPU.
    auto handle = nanovdb::tools::createFogVolumeSphere<float, nanovdb::cuda::DeviceBuffer>(
        /*radius=*/500.0, /*center=*/{0, 0, 0}, /*voxelSize=*/1.0,
        /*halfWidth=*/3.0, /*origin=*/{0, 0, 0}, "sphere");

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    handle.deviceUpload(stream, /*sync=*/true);

    auto* dGrid = handle.deviceGrid<float>();
    if (!dGrid) {
        std::cerr << "Failed to obtain device grid\n";
        return 1;
    }

    const int N = 1 << 20; // 1M accesses per pattern (matches CPU benchmark)

    std::cout << "Access count per pattern: " << N
              << "  (chunk=" << CHUNK << ", threads=" << (N + CHUNK - 1) / CHUNK << ")\n\n";
    std::cout << "Pattern          ns/access\n";
    std::cout << "-----------------------------\n";

    const bench::Pattern patterns[] = {
        bench::Pattern::Sequential, bench::Pattern::LeafJump,
        bench::Pattern::NodeJump,   bench::Pattern::Random};

    for (auto p : patterns) {
        auto   coords = bench::makePattern(p, N);
        double ns     = benchPattern(dGrid, coords, stream);
        std::cout << "  " << std::left << std::setw(16) << bench::name(p)
                  << std::right << std::setw(10) << std::fixed << std::setprecision(3) << ns
                  << " ns/access\n";
    }

    // 3x3x3 stencil sweep (27 neighbour lookups per centre).
    {
        auto   centers = bench::makeStencilCenters();
        double ns      = benchStencil(dGrid, centers, stream);
        std::cout << "\nStencil 3x3x3 (27 lookups/centre), centres = " << centers.size() << "\n";
        std::cout << "  " << std::fixed << std::setprecision(4) << ns << " ns/stencil  ("
                  << std::setprecision(4) << ns / 27.0 << " ns/lookup)\n";
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
