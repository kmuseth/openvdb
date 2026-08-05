// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// CPU benchmark comparing ReadAccessor with NANOVDB_USE_OLD_ACCESSOR vs without,
// and ReadAccessor<0,1,2> (full 3-level cache) vs ReadAccessor<0> (leaf-only cache).
//
// Two CPU modes are timed:
//   1T  -- single-threaded serial walk (latency).
//   MT  -- multi-threaded via nanovdb::util::forEach (tbb::parallel_for),
//          each grain using its own accessor so the cache-reuse semantics match
//          the serial walk.

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

static volatile float g_sink = 0.0f;

static constexpr int GRAIN = 4096;

// --- single-threaded (latency) ----------------------------------------------

template<typename AccT, typename RootT>
static double runTrial1T(const RootT& root, const std::vector<nanovdb::Coord>& coords)
{
    AccT  acc(root);
    float sum = 0.0f;
    auto  t0  = Clock::now();
    for (const auto& c : coords)
        sum += acc.getValue(c);
    auto t1 = Clock::now();
    g_sink = sum;
    return elapsed_ns(t0, t1) / static_cast<double>(coords.size());
}

// --- multi-threaded (throughput) --------------------------------------------

template<typename AccT, typename GridT>
static double runTrialMT(const GridT* grid, const std::vector<nanovdb::Coord>& coords,
                         std::vector<float>& out)
{
    const nanovdb::Coord* pc = coords.data();
    float*                po = out.data();
    auto t0 = Clock::now();
    nanovdb::util::forEach(0, coords.size(), GRAIN, [&](const nanovdb::util::Range1D& r) {
        AccT acc(grid->tree().root());
        for (size_t i = r.begin(); i != r.end(); ++i)
            po[i] = acc.getValue(pc[i]);
    });
    auto t1 = Clock::now();
    float sum = 0.0f;
    for (float v : out) sum += v;
    g_sink = sum;
    return elapsed_ns(t0, t1) / static_cast<double>(coords.size());
}

// --- 3x3x3 stencil ----------------------------------------------------------

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

template<typename AccT, typename RootT>
static double runStencil1T(const RootT& root, const std::vector<nanovdb::Coord>& centers)
{
    AccT  acc(root);
    float sum = 0.0f;
    auto  t0  = Clock::now();
    for (const auto& c : centers)
        sum += stencilSum(acc, c);
    auto t1 = Clock::now();
    g_sink = sum;
    return elapsed_ns(t0, t1) / static_cast<double>(centers.size());
}

template<typename AccT, typename GridT>
static double runStencilMT(const GridT* grid, const std::vector<nanovdb::Coord>& centers,
                           std::vector<float>& out)
{
    const nanovdb::Coord* pc = centers.data();
    float*                po = out.data();
    auto t0 = Clock::now();
    nanovdb::util::forEach(0, centers.size(), GRAIN, [&](const nanovdb::util::Range1D& r) {
        AccT acc(grid->tree().root());
        for (size_t i = r.begin(); i != r.end(); ++i)
            po[i] = stencilSum(acc, pc[i]);
    });
    auto t1 = Clock::now();
    float sum = 0.0f;
    for (size_t i = 0; i < centers.size(); ++i) sum += po[i];
    g_sink = sum;
    return elapsed_ns(t0, t1) / static_cast<double>(centers.size());
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

template<typename AccT, typename GridT>
static void runSuite(const GridT* grid, const char* accLabel, int N,
                     std::vector<float>& out, const bench::Pools& pools)
{
    const bench::Pattern patterns[] = {
        bench::Pattern::Sequential, bench::Pattern::LeafJump,
        bench::Pattern::NodeJump,   bench::Pattern::Random};

    for (auto p : patterns) {
        auto   coords = bench::makePattern(p, N, pools);
        double ns1 = median([&] { return runTrial1T<AccT>(grid->tree().root(), coords); });
        double nsM = median([&] { return runTrialMT<AccT>(grid, coords, out); });
        std::cout << "  " << std::left << std::setw(14) << bench::name(p)
                  << std::setw(8) << accLabel
                  << std::right << std::setw(10) << std::fixed << std::setprecision(2) << ns1
                  << std::setw(13) << nsM
                  << std::setw(11) << std::setprecision(1) << (ns1 / nsM) << "x\n";
    }

    auto   centers = bench::makeStencilCenters(pools, grid);
    double ns1 = median([&] { return runStencil1T<AccT>(grid->tree().root(), centers); });
    double nsM = median([&] { return runStencilMT<AccT>(grid, centers, out); });
    std::cout << "  " << std::left << std::setw(14) << "Stencil(27pt)"
              << std::setw(8) << accLabel
              << std::right << std::setw(10) << std::fixed << std::setprecision(2) << ns1
              << std::setw(13) << nsM
              << std::setw(11) << std::setprecision(1) << (ns1 / nsM) << "x"
              << "   [" << std::setprecision(3) << ns1/27.0 << " / "
              << nsM/27.0 << " ns/lookup]\n";
}

int main()
{
    std::cout << "=== NanoVDB ReadAccessor CPU benchmark: " << bench::accessorMode() << " ===\n\n";

    // A narrow-band level set sphere: its only active voxels are the band leaves,
    // so every harvested coordinate resolves at a leaf node (a fog sphere would
    // store its uniform interior as active tiles at internal nodes, which never
    // descend into a leaf and make the leaf-only accessor look artificially bad).
    auto handle = nanovdb::tools::createLevelSetSphere<float>(
        /*radius=*/256.0, /*center=*/{0, 0, 0}, /*voxelSize=*/1.0,
        /*halfWidth=*/3.0, /*origin=*/{0, 0, 0}, "sphere");
    auto* grid = handle.grid<float>();
    if (!grid) {
        std::cerr << "Failed to create grid\n";
        return 1;
    }

    // Harvest the active leaf voxels once and drive every pattern from them.
    const bench::Pools pools = bench::harvest(grid);
    std::cout << "Active leaf voxels: " << pools.all.size()
              << "   leaf nodes: " << pools.leafReps.size()
              << "   lower nodes: " << pools.lowerReps.size() << "\n";

    const int N = 1 << 20; // 1M accesses per pattern

    std::cout << "Access count per pattern: " << N
              << "   HW threads: " << std::thread::hardware_concurrency() << "\n\n";
    std::cout << "Pattern           Accessor  1T ns/acc    MT ns/acc    MT speedup\n";
    std::cout << "-------------------------------------------------------------------\n";

    std::vector<float> out(N);

    using Acc012 = nanovdb::ReadAccessor<float, 0, 1, 2>;
    using Acc0   = nanovdb::ReadAccessor<float, 0>;

    runSuite<Acc012>(grid, "<0,1,2>", N, out, pools);
    std::cout << "\n";
    runSuite<Acc0>(grid, "<0>", N, out, pools);

    return 0;
}
