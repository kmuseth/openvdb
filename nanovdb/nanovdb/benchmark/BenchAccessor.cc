// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// CPU benchmark comparing ReadAccessor with NANOVDB_USE_OLD_ACCESSOR vs without.
//
// The key difference:
//   OLD: when a leaf-value access misses the leaf cache, the level-1 and level-2
//        cache checks are compiled away (if-constexpr + else), so the accessor
//        falls straight to a root traversal.
//   NEW: all applicable cache levels are always checked, so a leaf-cache miss can
//        still hit the level-1 or level-2 cache instead of restarting at the root.
//
// Two CPU modes are timed:
//   1T  -- single-threaded serial walk (latency).
//   MT  -- multi-threaded via nanovdb::util::forEach (tbb::parallel_for),
//          each grain using its own accessor so the cache-reuse semantics match
//          the serial walk. This makes the CPU-vs-GPU comparison fair: all CPU
//          cores against the full GPU.

#include "BenchPatterns.h"

#include <nanovdb/tools/CreatePrimitives.h>
#include <nanovdb/util/ForEach.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;

static double elapsed_ns(Clock::time_point t0, Clock::time_point t1)
{
    return static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
}

// Prevent the compiler from optimising away the accumulation result.
static volatile float g_sink = 0.0f;

// Contiguous coords per grain -- large enough for accessor cache reuse, small
// enough to spread work evenly across cores.
static constexpr int GRAIN = 4096;

// --- single-threaded (latency) ----------------------------------------------

template<typename GridT>
static double runTrial1T(const GridT* grid, const std::vector<nanovdb::Coord>& coords)
{
    auto  acc = grid->getAccessor();
    float sum = 0.0f;
    auto  t0  = Clock::now();
    for (const auto& c : coords)
        sum += acc.getValue(c);
    auto t1 = Clock::now();
    g_sink = sum;
    return elapsed_ns(t0, t1) / static_cast<double>(coords.size());
}

// --- multi-threaded (throughput) --------------------------------------------

template<typename GridT>
static double runTrialMT(const GridT* grid, const std::vector<nanovdb::Coord>& coords,
                         std::vector<float>& out)
{
    const nanovdb::Coord* pc = coords.data();
    float*                po = out.data();
    auto t0 = Clock::now();
    nanovdb::util::forEach(0, coords.size(), GRAIN, [&](const nanovdb::util::Range1D& r) {
        auto acc = grid->getAccessor(); // one accessor per grain -> per-thread cache
        for (size_t i = r.begin(); i != r.end(); ++i)
            po[i] = acc.getValue(pc[i]);
    });
    auto t1 = Clock::now();
    float sum = 0.0f;
    for (float v : out) sum += v;
    g_sink = sum;
    return elapsed_ns(t0, t1) / static_cast<double>(coords.size());
}

// --- 3x3x3 stencil: 27-neighbour lookup around each centre ------------------

template<typename AccT>
static float stencilSum(AccT& acc, const nanovdb::Coord& c)
{
    float s = 0.0f;
    for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx)
                s += acc.getValue(c + nanovdb::Coord(dx, dy, dz));
    return s;
}

template<typename GridT>
static double runStencil1T(const GridT* grid, const std::vector<nanovdb::Coord>& centers)
{
    auto  acc = grid->getAccessor();
    float sum = 0.0f;
    auto  t0  = Clock::now();
    for (const auto& c : centers)
        sum += stencilSum(acc, c);
    auto t1 = Clock::now();
    g_sink = sum;
    return elapsed_ns(t0, t1) / static_cast<double>(centers.size()); // ns per stencil
}

template<typename GridT>
static double runStencilMT(const GridT* grid, const std::vector<nanovdb::Coord>& centers,
                           std::vector<float>& out)
{
    const nanovdb::Coord* pc = centers.data();
    float*                po = out.data();
    auto t0 = Clock::now();
    nanovdb::util::forEach(0, centers.size(), GRAIN, [&](const nanovdb::util::Range1D& r) {
        auto acc = grid->getAccessor();
        for (size_t i = r.begin(); i != r.end(); ++i)
            po[i] = stencilSum(acc, pc[i]);
    });
    auto t1 = Clock::now();
    float sum = 0.0f;
    for (size_t i = 0; i < centers.size(); ++i) sum += po[i];
    g_sink = sum;
    return elapsed_ns(t0, t1) / static_cast<double>(centers.size()); // ns per stencil
}

template<typename FnT>
static double median(FnT trial, int nTrials = 7)
{
    std::vector<double> times(nTrials);
    for (int i = 0; i < nTrials; ++i)
        times[i] = trial();
    std::nth_element(times.begin(), times.begin() + nTrials / 2, times.end());
    return times[nTrials / 2];
}

int main()
{
    std::cout << "=== NanoVDB ReadAccessor CPU benchmark: " << bench::accessorMode() << " ===\n\n";

    auto handle = nanovdb::tools::createFogVolumeSphere<float>(
        /*radius=*/500.0, /*center=*/{0, 0, 0}, /*voxelSize=*/1.0,
        /*halfWidth=*/3.0, /*origin=*/{0, 0, 0}, "sphere");
    auto* grid = handle.grid<float>();
    if (!grid) {
        std::cerr << "Failed to create grid\n";
        return 1;
    }

    const int N = 1 << 20; // 1M accesses per pattern

    std::cout << "Access count per pattern: " << N
              << "   HW threads: " << std::thread::hardware_concurrency() << "\n\n";
    std::cout << "Pattern           1T ns/acc    MT ns/acc    MT speedup\n";
    std::cout << "-----------------------------------------------------\n";

    const bench::Pattern patterns[] = {
        bench::Pattern::Sequential, bench::Pattern::LeafJump,
        bench::Pattern::NodeJump,   bench::Pattern::Random};

    std::vector<float> out(N);
    for (auto p : patterns) {
        auto   coords = bench::makePattern(p, N);
        double ns1 = median([&] { return runTrial1T(grid, coords); });
        double nsM = median([&] { return runTrialMT(grid, coords, out); });
        std::cout << "  " << std::left << std::setw(14) << bench::name(p)
                  << std::right << std::setw(10) << std::fixed << std::setprecision(2) << ns1
                  << std::setw(13) << nsM
                  << std::setw(11) << std::setprecision(1) << (ns1 / nsM) << "x\n";
    }

    // 3x3x3 stencil sweep (27 neighbour lookups per centre).
    {
        auto   centers = bench::makeStencilCenters();
        double ns1 = median([&] { return runStencil1T(grid, centers); });
        double nsM = median([&] { return runStencilMT(grid, centers, out); });
        std::cout << "\nStencil 3x3x3 (27 lookups/centre), centres = " << centers.size() << "\n";
        std::cout << std::fixed;
        std::cout << "  1T: " << std::setprecision(2) << std::setw(8) << ns1 << " ns/stencil  ("
                  << std::setprecision(3) << ns1 / 27.0 << " ns/lookup)\n";
        std::cout << "  MT: " << std::setprecision(2) << std::setw(8) << nsM << " ns/stencil  ("
                  << std::setprecision(3) << nsM / 27.0 << " ns/lookup),  speedup "
                  << std::setprecision(1) << ns1 / nsM << "x\n";
    }

    return 0;
}
