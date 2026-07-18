// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Shared access-pattern generation for the CPU and GPU ReadAccessor benchmarks,
// so both measure exactly the same coordinate streams.

#ifndef NANOVDB_BENCH_PATTERNS_H
#define NANOVDB_BENCH_PATTERNS_H

#include <nanovdb/NanoVDB.h>

#include <random>
#include <string>
#include <vector>

namespace bench {

// Node extents (NanoVDB default topology: 8 -> 128 -> 4096 voxels per axis).
static constexpr int LEAF_DIM  = 1 << 3;                 // 8   (leaf)
static constexpr int NODE1_DIM = LEAF_DIM  * (1 << 4);   // 128 (lower internal)
static constexpr int NODE2_DIM = NODE1_DIM * (1 << 5);   // 4096 (upper internal)

// Half-extent of the sampled cube. Kept well inside the sphere radius so the
// traversals hit populated nodes instead of escaping into empty space (where a
// root lookup is trivial and the accessor caches never pay off).
static constexpr int DOMAIN = 256;

// Wrap a walking coordinate back into [-DOMAIN, DOMAIN] so long strided walks
// keep re-entering the populated region instead of leaving the grid.
inline int wrap(long x)
{
    const long span = 2L * DOMAIN;
    long m = ((x + DOMAIN) % span + span) % span;
    return static_cast<int>(m - DOMAIN);
}

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

// Side length of the dense cube of stencil centres. STENCIL_SIDE^3 centres, each
// expanded into a 3x3x3 = 27-neighbour lookup. A dense sweep is the realistic
// finite-difference / convolution access pattern: consecutive centres share
// leaves (level-0 cache hot) while boundary-crossing neighbours spill into
// adjacent leaves (level-1 cache -- the case the NEW accessor rescues).
static constexpr int STENCIL_SIDE = 64; // 64^3 = 262144 centres

// A dense cube of centres near the origin, well inside the populated region.
inline std::vector<nanovdb::Coord> makeStencilCenters()
{
    std::vector<nanovdb::Coord> v;
    v.reserve(static_cast<size_t>(STENCIL_SIDE) * STENCIL_SIDE * STENCIL_SIDE);
    for (int z = 0; z < STENCIL_SIDE; ++z)
        for (int y = 0; y < STENCIL_SIDE; ++y)
            for (int x = 0; x < STENCIL_SIDE; ++x)
                v.emplace_back(x, y, z);
    return v;
}

// Build the coordinate stream for a given pattern.
//   Sequential: stride 1        -> level-0 (leaf) cache always hot
//   LeafJump:   stride LEAF_DIM  -> leaf cache cold, level-1 warm (NEW only)
//   NodeJump:   stride NODE1_DIM -> leaf+level-1 cold, level-2 warm (NEW only)
//   Random:     uniform in cube  -> all caches cold, both fall to root
inline std::vector<nanovdb::Coord> makePattern(Pattern p, int count)
{
    std::vector<nanovdb::Coord> v;
    v.reserve(count);
    switch (p) {
        case Pattern::Sequential:
            for (int i = 0; i < count; ++i) v.emplace_back(wrap(i), 0, 0);
            break;
        case Pattern::LeafJump:
            for (int i = 0; i < count; ++i)
                v.emplace_back(wrap(static_cast<long>(i) * LEAF_DIM), 0, 0);
            break;
        case Pattern::NodeJump:
            for (int i = 0; i < count; ++i)
                v.emplace_back(wrap(static_cast<long>(i) * NODE1_DIM), 0, 0);
            break;
        case Pattern::Random: {
            std::mt19937 rng(42);
            std::uniform_int_distribution<int> dist(-DOMAIN, DOMAIN);
            for (int i = 0; i < count; ++i)
                v.emplace_back(dist(rng), dist(rng), dist(rng));
            break;
        }
    }
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
