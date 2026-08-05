// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Shared access-pattern generation for the CPU and GPU ReadAccessor benchmarks,
// so both measure exactly the same coordinate streams.
//
// IMPORTANT: every benchmarked coordinate must resolve at a LEAF node, otherwise
// the comparison is meaningless. A sphere's uniform interior is stored as active
// TILES at the upper internal nodes, so sampling a cube of coordinates there
// never descends into a leaf -- it makes ReadAccessor<0> (leaf-only cache) look
// artificially bad because the value it caches (a leaf) is never the node that
// resolves the lookup. To avoid this, all coordinate streams below are drawn
// from the grid's ACTUAL active voxels, which live exclusively in leaf nodes
// (use a narrow-band level set, whose only active voxels are the band leaves).

#ifndef NANOVDB_BENCH_PATTERNS_H
#define NANOVDB_BENCH_PATTERNS_H

#include <nanovdb/NanoVDB.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

namespace bench {

// Node extents (NanoVDB default topology: 8 -> 128 -> 4096 voxels per axis).
static constexpr int LEAF_DIM  = 1 << 3;                 // 8    (leaf)
static constexpr int NODE1_DIM = LEAF_DIM  * (1 << 4);   // 128  (lower internal)
static constexpr int NODE2_DIM = NODE1_DIM * (1 << 5);   // 4096 (upper internal)

enum class Pattern { Sequential, LeafJump, NodeJump, Random };

inline const char* name(Pattern p)
{
    switch (p) {
        case Pattern::Sequential: return "Sequential";
        case Pattern::LeafJump:   return "LeafJump";
        case Pattern::NodeJump:   return "NodeJump";
        case Pattern::Random:     return "Random";
    }
    return "?";
}

// Active-voxel coordinate pools harvested from the grid. Every coordinate here is
// an active voxel and therefore lives in a leaf node, so all lookups descend the
// full tree and exercise all cache levels. The three pools drive the patterns:
//   all       -- every active leaf voxel in storage (Morton) order. Consecutive
//                entries usually share a leaf, so walking it in order keeps the
//                level-0 (leaf) cache hot -> the Sequential pattern.
//   leafReps  -- one active voxel per leaf node, in storage order. Consecutive
//                entries are in different but spatially-adjacent leaves (same
//                lower node), so the leaf cache misses while level-1 stays warm
//                -> the LeafJump pattern (the case the NEW accessor rescues).
//   lowerReps -- one active voxel per lower internal node. Consecutive entries
//                are in different lower nodes (same upper node), so leaf+level-1
//                miss while level-2 stays warm -> the NodeJump pattern.
struct Pools
{
    std::vector<nanovdb::Coord> all;
    std::vector<nanovdb::Coord> leafReps;
    std::vector<nanovdb::Coord> lowerReps;
};

// 128-aligned (lower-node) key for a coordinate. Assumes |coord| < 2^20 per axis,
// which holds comfortably for the benchmark grids.
inline uint64_t lowerKey(const nanovdb::Coord& c)
{
    const uint64_t x = static_cast<uint64_t>((c[0] >> 7) + (1 << 19)) & 0xFFFFF;
    const uint64_t y = static_cast<uint64_t>((c[1] >> 7) + (1 << 19)) & 0xFFFFF;
    const uint64_t z = static_cast<uint64_t>((c[2] >> 7) + (1 << 19)) & 0xFFFFF;
    return (x << 40) | (y << 20) | z;
}

// Harvest the active-voxel pools by iterating the grid's leaf nodes directly
// (NanoVDB stores all leaves contiguously). Active TILES at internal nodes are
// deliberately ignored -- only true leaf voxels are collected.
template<typename GridT>
inline Pools harvest(const GridT* grid)
{
    Pools pools;
    const auto* leaf0 = grid->tree().getFirstLeaf();
    const uint64_t nLeaf = grid->tree().nodeCount(0);
    std::unordered_set<uint64_t> seenLower;
    for (uint64_t li = 0; li < nLeaf; ++li) {
        const auto& leaf = leaf0[li];
        bool firstInLeaf = true;
        for (uint32_t n = 0; n < 512u; ++n) {
            if (!leaf.isActive(n)) continue;
            const nanovdb::Coord c = leaf.offsetToGlobalCoord(n);
            pools.all.push_back(c);
            if (firstInLeaf) {
                firstInLeaf = false;
                pools.leafReps.push_back(c);
                if (seenLower.insert(lowerKey(c)).second) pools.lowerReps.push_back(c);
            }
        }
    }
    return pools;
}

// Build a length-`count` coordinate stream for a pattern by cycling through the
// appropriate pool (all pools are non-empty for a populated grid). Cycling with
// modulo preserves each pattern's per-step cache locality even when count exceeds
// the pool size.
inline std::vector<nanovdb::Coord> makePattern(Pattern p, int count, const Pools& pools)
{
    std::vector<nanovdb::Coord> v;
    v.reserve(count);
    auto cycle = [&](const std::vector<nanovdb::Coord>& src) {
        if (src.empty()) return;
        for (int i = 0; i < count; ++i) v.push_back(src[static_cast<size_t>(i) % src.size()]);
    };
    switch (p) {
        case Pattern::Sequential: cycle(pools.all);       break; // leaf cache hot
        case Pattern::LeafJump:   cycle(pools.leafReps);  break; // leaf cold, level-1 warm
        case Pattern::NodeJump:   cycle(pools.lowerReps); break; // leaf+level-1 cold, level-2 warm
        case Pattern::Random: {                                  // all caches cold
            if (pools.all.empty()) break;
            std::vector<uint32_t> idx(pools.all.size());
            std::iota(idx.begin(), idx.end(), 0u);
            std::mt19937 rng(42);
            std::shuffle(idx.begin(), idx.end(), rng);
            for (int i = 0; i < count; ++i)
                v.push_back(pools.all[idx[static_cast<size_t>(i) % idx.size()]]);
            break;
        }
    }
    return v;
}

// Number of stencil centres (each expanded into a 3x3x3 = 27-neighbour lookup).
static constexpr int STENCIL_COUNT = 1 << 18; // 262144 centres

// Dense coherent sweep of stencil centres. A centre is only kept if its full
// 3x3x3 neighbourhood is active (all 27 taps land in leaf nodes), so the stencil
// never spills into an inactive tile at an internal node. Centres are taken in
// storage order, so consecutive centres share leaves (the realistic
// finite-difference / convolution access pattern) while boundary-crossing
// neighbours spill into adjacent leaves -- the case the NEW accessor rescues.
template<typename GridT>
inline std::vector<nanovdb::Coord> makeStencilCenters(const Pools& pools, const GridT* grid,
                                                      int count = STENCIL_COUNT)
{
    std::vector<nanovdb::Coord> v;
    if (pools.all.empty()) return v;
    v.reserve(count);
    auto acc = grid->getAccessor();
    auto fullyActive = [&](const nanovdb::Coord& c) {
        for (int dz = -1; dz <= 1; ++dz)
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx)
                    if (!acc.isActive(c + nanovdb::Coord(dx, dy, dz))) return false;
        return true;
    };
    for (size_t i = 0; i < pools.all.size() && static_cast<int>(v.size()) < count; ++i)
        if (fullyActive(pools.all[i])) v.push_back(pools.all[i]);
    // If the band is too thin to supply enough interior centres, cycle what we have.
    for (size_t i = 0; !v.empty() && static_cast<int>(v.size()) < count; ++i)
        v.push_back(v[i]);
    return v;
}

inline const char* accessorMode()
{
#ifdef NANOVDB_USE_OLD_ACCESSOR
    return "OLD (NANOVDB_USE_OLD_ACCESSOR defined)";
#else
    return "NEW (NANOVDB_USE_OLD_ACCESSOR not defined)";
#endif
}

} // namespace bench

#endif // NANOVDB_BENCH_PATTERNS_H
