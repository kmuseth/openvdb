// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file   MarchingTetrahedra.h
///
/// @brief  Extract a triangular isosurface mesh from a scalar volume using the
///         classic Marching Tetrahedra algorithm.
///
/// @author Ken Museth
///
/// @details Each grid cell (the cube spanned by eight neighboring voxels) is
///          split into six tetrahedra (Kuhn / Freudenthal decomposition, all
///          sharing the cube's main diagonal). Every tetrahedron is polygonized
///          independently by linearly interpolating the isosurface crossing
///          along its edges. Compared with Marching Cubes this is topologically
///          unambiguous (no lookup-table face-ambiguity) at the cost of more,
///          and more slivery, triangles. Vertices are shared between adjacent
///          tetrahedra and cells (welded per edge), so the output is a
///          watertight indexed triangle mesh.
///
///          Use the MarchingTetrahedra class directly to keep the extracted mesh
///          for repeated queries, or the free-standing marchingTetrahedra()
///          function to write the result straight into std::vector lists of
///          world-space vertex coordinates and triangle vertex indices (like
///          tools::volumeToMesh).

#ifndef OPENVDB_TOOLS_MARCHING_TETRAHEDRA_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_MARCHING_TETRAHEDRA_HAS_BEEN_INCLUDED

#include <openvdb/openvdb.h>
#include <openvdb/Types.h>
#include <openvdb/math/Math.h>       // for math::Clamp, math::isApproxZero
#include <openvdb/math/Transform.h>
#include <openvdb/math/Vec3.h>

#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>    // for std::pair
#include <vector>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {


////////////////////////////////////////


/// @brief Mesh a scalar grid's isosurface with the Marching Tetrahedra algorithm.
///
/// @param grid       a scalar (floating-point) grid to mesh
/// @param points     output list of world-space vertex positions
/// @param triangles  output triangle index list (indices into @a points)
/// @param isovalue   the isosurface to mesh (defaults to zero)
///
/// @details Both output vectors are cleared before use. A thin wrapper around the
///          MarchingTetrahedra class whose result is swapped into @a points and
///          @a triangles (no copy).
template<typename GridType>
void marchingTetrahedra(
    const GridType& grid,
    std::vector<Vec3s>& points,
    std::vector<Vec3I>& triangles,
    double isovalue = 0.0);


////////////////////////////////////////


/// @brief Extract a triangular isosurface mesh from a scalar grid with the
///        classic Marching Tetrahedra algorithm.
///
/// @details Construct the mesher with the grid to be meshed, then call it with
///          the desired isovalue; the resulting welded, watertight triangle mesh
///          is available through points() and triangles(). The same instance can
///          be re-invoked with a different isovalue on the same grid.
///
/// @note Only regions where the value varies (i.e. the active narrow band of a
///       level set, or the varying shell of a fog volume) are polygonized;
///       constant active tiles contain no crossing and are skipped. Triangles are
///       wound so their normals point toward increasing value (out of the
///       isosurface for a level set stored negative-inside).
template<typename GridType>
class MarchingTetrahedra
{
public:
    using ValueType = typename GridType::ValueType;

    static_assert(std::is_floating_point<ValueType>::value,
        "MarchingTetrahedra requires a scalar floating-point grid");

    /// @brief Construct a mesher bound to @a grid.
    explicit MarchingTetrahedra(const GridType& grid): mGrid(grid) {}

    /// @brief Extract the @a isovalue isosurface. Replaces any previous result;
    ///        afterwards the mesh is available through points() and triangles().
    void operator()(double isovalue = 0.0);

    /// @brief World-space vertex positions of the most recent extraction.
    /// @{
    const std::vector<Vec3s>& points() const { return mPoints; }
    std::vector<Vec3s>&       points()       { return mPoints; }
    /// @}

    /// @brief Triangle vertex indices (into points()) of the most recent extraction.
    /// @{
    const std::vector<Vec3I>& triangles() const { return mTriangles; }
    std::vector<Vec3I>&       triangles()       { return mTriangles; }
    /// @}

private:
    /// @brief An isosurface crossing vertex is uniquely identified by the edge it
    ///        lies on, i.e. the (canonically ordered) pair of endpoint voxel
    ///        coordinates.
    using EdgeKey = std::pair<Coord, Coord>;

    /// @brief Hash for EdgeKey, mixing the six integer coordinates.
    struct EdgeKeyHash
    {
        std::size_t operator()(const EdgeKey& e) const
        {
            auto h = [](const Coord& c) -> std::size_t {
                return (static_cast<std::size_t>(static_cast<uint32_t>(c.x())) * 73856093u)
                     ^ (static_cast<std::size_t>(static_cast<uint32_t>(c.y())) * 19349663u)
                     ^ (static_cast<std::size_t>(static_cast<uint32_t>(c.z())) * 83492791u);
            };
            const std::size_t h0 = h(e.first), h1 = h(e.second);
            return h0 ^ (h1 + 0x9e3779b97f4a7c15ULL + (h0 << 6) + (h0 >> 2));
        }
    };

    /// @brief Per-thread mesh fragment produced during the parallel phase.
    ///        Vertices are welded locally and tagged with the edge they lie on
    ///        (@c keys) so the global shard merge can re-weld boundaries.
    ///
    ///        The weld table is a flat open-addressing hash map (linear probing,
    ///        load factor ≤ 2/3). It stores vertex indices into @c points, using
    ///        @c keys for collision resolution. Compared with std::unordered_map
    ///        this eliminates per-entry heap allocations and makes teardown O(1)
    ///        (one contiguous free) rather than O(n) linked-node frees.
    struct LocalMesh
    {
        std::vector<Vec3s>   points;
        std::vector<EdgeKey> keys;       ///< canonical edge key per local vertex
        std::vector<Vec3I>   triangles;  ///< indices into @c points
        std::vector<Index32> htSlots;    ///< weld table slots (vertex index or HT_EMPTY)
        std::size_t          htMask = 0;

        static constexpr Index32 HT_EMPTY = ~Index32(0);

        LocalMesh()
        {
            constexpr std::size_t kInit = 64;
            htSlots.assign(kInit, HT_EMPTY);
            htMask = kInit - 1;
        }
    };

    /// @brief Local corner @a c of a cell has index-space offset
    ///        (c&1, (c>>1)&1, (c>>2)&1), so corner 0 is the cell origin and corner
    ///        7 the far corner along the main diagonal.
    static Coord cornerOffset(int c) { return Coord(c & 1, (c >> 1) & 1, (c >> 2) & 1); }

    /// @brief Kuhn/Freudenthal decomposition of a cube into six tetrahedra, each
    ///        four local corner indices sharing the main diagonal (0 to 7), so the
    ///        tetrahedra tile the cube and share faces with neighboring cells.
    static constexpr int sTetrahedra[6][4] = {
        {0, 1, 3, 7}, {0, 1, 5, 7}, {0, 4, 5, 7},
        {0, 4, 6, 7}, {0, 2, 6, 7}, {0, 2, 3, 7}
    };

    /// @brief Sign (+1/-1/0) of six times the signed volume of the tetrahedron
    ///        (@a a,@a b,@a c,@a d), evaluated exactly in integer index space.
    /// @details Orients the output triangles combinatorially: because it depends
    ///          only on the (never-degenerate) integer cell geometry, and not on
    ///          the interpolated crossing positions, it yields a globally
    ///          consistent winding even for slivery tetrahedra.
    static int tetSign(const Coord& a, const Coord& b, const Coord& c, const Coord& d);

    /// @brief Return the (locally welded) vertex index for the crossing on edge
    ///        (@a ca,@a cb), creating it if new. Keyed on the canonical edge so
    ///        every tetrahedron/cell/thread sharing it gets an identical position.
    static Index32 localCrossing(LocalMesh& mesh, const math::Transform& xform,
        double isovalue, const Coord& ca, ValueType va, const Coord& cb, ValueType vb);

    /// @brief Polygonize one cell (its eight @a corner coordinates and @a value s)
    ///        into @a mesh by splitting it into six tetrahedra and marching each.
    static void marchCell(LocalMesh& mesh, const math::Transform& xform,
        double isovalue, const Coord corner[8], const ValueType value[8]);

    /// @brief Collect the origins of every cell that may contain the isosurface
    ///        (a cell with at least one active corner). Independent of isovalue.
    void gatherCells(std::vector<Coord>& cells) const;

    /// @brief March the given cells (in parallel) and merge the result into
    ///        mPoints / mTriangles.
    void extract(const std::vector<Coord>& cells, double isovalue);

    const GridType&    mGrid;
    std::vector<Vec3s> mPoints;
    std::vector<Vec3I> mTriangles;
}; // class MarchingTetrahedra


////////////////////////////////////////

// MarchingTetrahedra implementation


template<typename GridType>
constexpr int MarchingTetrahedra<GridType>::sTetrahedra[6][4];


template<typename GridType>
int
MarchingTetrahedra<GridType>::tetSign(const Coord& a, const Coord& b,
                                      const Coord& c, const Coord& d)
{
    const long bx = b.x()-a.x(), by = b.y()-a.y(), bz = b.z()-a.z();
    const long cx = c.x()-a.x(), cy = c.y()-a.y(), cz = c.z()-a.z();
    const long dx = d.x()-a.x(), dy = d.y()-a.y(), dz = d.z()-a.z();
    const long det = bx*(cy*dz - cz*dy) - by*(cx*dz - cz*dx) + bz*(cx*dy - cy*dx);
    return (det > 0) - (det < 0);
}


template<typename GridType>
Index32
MarchingTetrahedra<GridType>::localCrossing(LocalMesh& mesh, const math::Transform& xform,
    double isovalue, const Coord& ca, ValueType va, const Coord& cb, ValueType vb)
{
    static constexpr EdgeKeyHash hasher{};
    const EdgeKey key = (ca < cb) ? EdgeKey(ca, cb) : EdgeKey(cb, ca);

    // Probe the flat weld table.
    std::size_t slot = hasher(key) & mesh.htMask;
    for (;;) {
        const Index32 g = mesh.htSlots[slot];
        if (g == LocalMesh::HT_EMPTY) break;       // empty slot — new vertex
        if (mesh.keys[g] == key) return g;          // found existing vertex
        slot = (slot + 1) & mesh.htMask;
    }

    // Compute interpolated world-space position for the new vertex.
    const Coord& c0 = key.first;
    const Coord& c1 = key.second;
    const double f0 = double((c0 == ca) ? va : vb);
    const double f1 = double((c1 == ca) ? va : vb);
    const double denom = f1 - f0;
    double t = math::isApproxZero(denom) ? 0.5 : (isovalue - f0) / denom;
    t = math::Clamp(t, 0.0, 1.0);

    const Vec3d ip(
        double(c0.x()) + t * double(c1.x() - c0.x()),
        double(c0.y()) + t * double(c1.y() - c0.y()),
        double(c0.z()) + t * double(c1.z() - c0.z()));
    const Vec3d wp = xform.indexToWorld(ip);

    const Index32 idx = static_cast<Index32>(mesh.points.size());

    // Grow the table when insertion would push load factor above 2/3.
    if ((mesh.points.size() + 1) * 3 > mesh.htSlots.size() * 2) {
        const std::size_t newCap = mesh.htSlots.size() * 2;
        mesh.htSlots.assign(newCap, LocalMesh::HT_EMPTY);
        mesh.htMask = newCap - 1;
        for (Index32 i = 0; i < idx; ++i) {
            std::size_t s = hasher(mesh.keys[i]) & mesh.htMask;
            while (mesh.htSlots[s] != LocalMesh::HT_EMPTY) s = (s + 1) & mesh.htMask;
            mesh.htSlots[s] = i;
        }
        // Re-probe for the insertion slot after rehash.
        slot = hasher(key) & mesh.htMask;
        while (mesh.htSlots[slot] != LocalMesh::HT_EMPTY) slot = (slot + 1) & mesh.htMask;
    }

    mesh.htSlots[slot] = idx;
    mesh.points.emplace_back(static_cast<float>(wp.x()),
                             static_cast<float>(wp.y()),
                             static_cast<float>(wp.z()));
    mesh.keys.push_back(key);
    return idx;
}


template<typename GridType>
void
MarchingTetrahedra<GridType>::marchCell(LocalMesh& mesh, const math::Transform& xform,
    double isovalue, const Coord corner[8], const ValueType value[8])
{
    for (int t = 0; t < 6; ++t) {
        const int* tet = sTetrahedra[t];

        // Partition the four tetrahedron corners by side of the isosurface.
        int inside[4],  nIn  = 0;
        int outside[4], nOut = 0;
        for (int i = 0; i < 4; ++i) {
            if (double(value[tet[i]]) < isovalue) inside[nIn++]  = tet[i];
            else                                  outside[nOut++] = tet[i];
        }
        (void)nOut;
        if (nIn == 0 || nIn == 4) continue; // no crossing

        auto edgeVtx = [&](int i, int j) {
            return localCrossing(mesh, xform, isovalue, corner[i], value[i], corner[j], value[j]);
        };

        // Winding is chosen combinatorially from the tetrahedron's integer
        // orientation (tetSign) so the normal points from inside (< isovalue) to
        // outside, consistently across every tetrahedron and cell.
        if (nIn == 1 || nIn == 3) {
            // One "lone" corner on the minority side; the three crossings on the
            // edges from it to the other three corners form one triangle.
            const int a = (nIn == 1) ? inside[0]  : outside[0];
            const int b = (nIn == 1) ? outside[0] : inside[0];
            const int c = (nIn == 1) ? outside[1] : inside[1];
            const int d = (nIn == 1) ? outside[2] : inside[2];
            const Index32 ab = edgeVtx(a, b);
            const Index32 ac = edgeVtx(a, c);
            const Index32 ad = edgeVtx(a, d);
            // For a lone inside vertex the normal points away from it; for a lone
            // outside vertex it points toward it (hence the reversal).
            int s = tetSign(corner[a], corner[b], corner[c], corner[d]);
            if (nIn == 3) s = -s;
            if (s >= 0) mesh.triangles.emplace_back(ab, ac, ad);
            else        mesh.triangles.emplace_back(ab, ad, ac);
        } else { // nIn == 2
            // Two-vs-two: the four crossings form a quad, split into two triangles
            // around the loop (a,c)-(b,c)-(b,d)-(a,d).
            const int a = inside[0],  b = inside[1];
            const int c = outside[0], d = outside[1];
            const Index32 ac = edgeVtx(a, c);
            const Index32 ad = edgeVtx(a, d);
            const Index32 bc = edgeVtx(b, c);
            const Index32 bd = edgeVtx(b, d);
            const int s = tetSign(corner[a], corner[b], corner[c], corner[d]);
            if (s >= 0) {
                mesh.triangles.emplace_back(ac, bd, bc);
                mesh.triangles.emplace_back(ac, ad, bd);
            } else {
                mesh.triangles.emplace_back(ac, bc, bd);
                mesh.triangles.emplace_back(ac, bd, ad);
            }
        }
    }
}


template<typename GridType>
void
MarchingTetrahedra<GridType>::gatherCells(std::vector<Coord>& cells) const
{
    // A cell (identified by its min-corner origin o) may contain the isosurface
    // only if at least one of its eight corners is active. Corner c of cell o
    // sits at o + cornerOffset(c), so o is a candidate iff some active voxel v
    // satisfies o == v - cornerOffset(c).
    //
    // Parallel phase: each thread marks candidate cell origins into its own
    // thread-local MaskGrid (no contention). The MaskGrid naturally deduplicates
    // repeated writes for voxels sharing the same candidate cell within a thread's
    // leaf range.
    //
    // Serial union: the thread-local MaskGrids are merged into a single master by
    // iterating each local grid's active voxels and stamping them into the master.
    // Because each thread's domain is a spatially contiguous leaf range, these
    // writes arrive in approximately Morton order, keeping the master accessor warm.
    //
    // Final collection: cbeginValueOn() on the master iterates in tree-traversal
    // (Morton/Z-curve) order, which is the same spatial ordering the original
    // serial MaskGrid approach produced — important for accessor cache efficiency
    // in the subsequent extract() phase.
    using LeafNode = typename GridType::TreeType::LeafNodeType;

    std::vector<const LeafNode*> leaves;
    leaves.reserve(256);
    for (auto it = mGrid.tree().cbeginLeaf(); it; ++it) leaves.push_back(&*it);

    tbb::enumerable_thread_specific<MaskGrid::Ptr> pool(
        [] { return MaskGrid::create(); });

    tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()),
        [&](const tbb::blocked_range<size_t>& range)
    {
        MaskGrid::Accessor acc = pool.local()->getAccessor();
        for (size_t li = range.begin(); li != range.end(); ++li) {
            for (auto vit = leaves[li]->cbeginValueOn(); vit; ++vit) {
                const Coord v = vit.getCoord();
                for (int c = 0; c < 8; ++c) acc.setValueOn(v - cornerOffset(c));
            }
        }
    });

    // Union all thread-local MaskGrids into a single master.
    MaskGrid::Ptr master = MaskGrid::create();
    {
        MaskGrid::Accessor masterAcc = master->getAccessor();
        for (const MaskGrid::Ptr& local : pool) {
            for (auto leafIt = local->tree().cbeginLeaf(); leafIt; ++leafIt)
                for (auto vit = leafIt->cbeginValueOn(); vit; ++vit)
                    masterAcc.setValueOn(vit.getCoord());
        }
    }

    cells.clear();
    cells.reserve(static_cast<size_t>(master->activeVoxelCount()));
    for (auto leafIt = master->tree().cbeginLeaf(); leafIt; ++leafIt)
        for (auto vit = leafIt->cbeginValueOn(); vit; ++vit)
            cells.push_back(vit.getCoord());
}


template<typename GridType>
void
MarchingTetrahedra<GridType>::extract(const std::vector<Coord>& cells, double isovalue)
{
    using AccessorT = typename GridType::ConstAccessor;

    const math::Transform& xform = mGrid.transform();

    // ---- March candidate cells in parallel ---------------------------------
    // Each thread accumulates a locally welded mesh fragment; vertices are welded
    // globally in the serial merge below.
    // Estimate unique vertices per thread to pre-size the local weld table and
    // avoid repeated rehashing during the march.  Each cell yields at most one
    // unique vertex per edge (6 tets × up to 3 new crossings each), but most
    // are shared, so cells.size()/nThreads is a conservative per-thread upper
    // bound.  Round up to the next power-of-two capacity that keeps load ≤ 2/3.
    const std::size_t nThreads = std::max(std::size_t(1),
        static_cast<std::size_t>(tbb::this_task_arena::max_concurrency()));
    const std::size_t vertsPerThread = cells.size() / nThreads + 1;
    std::size_t initCap = 64;
    while (initCap * 2 < vertsPerThread * 3) initCap <<= 1; // load factor <= 2/3

    tbb::enumerable_thread_specific<LocalMesh>  pool([initCap] {
        LocalMesh m;
        m.htSlots.assign(initCap, LocalMesh::HT_EMPTY);
        m.htMask = initCap - 1;
        return m;
    });
    tbb::enumerable_thread_specific<AccessorT> accPool(
        [&] { return mGrid.getConstAccessor(); });
    tbb::parallel_for(tbb::blocked_range<size_t>(0, cells.size()),
        [&](const tbb::blocked_range<size_t>& range)
    {
        LocalMesh& mesh = pool.local();
        AccessorT& acc  = accPool.local();
        Coord     corner[8];
        ValueType value[8];
        for (size_t ci = range.begin(); ci != range.end(); ++ci) {
            const Coord origin = cells[ci];
            for (int c = 0; c < 8; ++c) {
                corner[c] = origin + cornerOffset(c);
                value[c]  = acc.getValue(corner[c]);
            }
            marchCell(mesh, xform, isovalue, corner, value);
        }
    });

    // ---- Merge thread-local fragments (parallel shard merge) ----------------
    // Re-weld vertices by edge key so that boundary vertices shared between
    // thread ranges collapse to one global vertex.
    //
    // The global edge-key hash space is split into numShards disjoint buckets.
    // Each vertex belongs to exactly one shard (determined by its hash), so
    // shards can be deduped and indexed in parallel without synchronisation.
    //
    // To avoid a serial scatter bottleneck, we use a two-pass parallel scatter:
    //   Pass 1 (parallel): each fragment counts its vertices per shard.
    //   Prefix sum (serial, tiny): compute per-(fragment,shard) offsets.
    //   Pass 2 (parallel): each fragment fills binnedIdx at those offsets.
    // This replaces the serial O(N) scatter with two O(N/nFrags) parallel passes.
    //
    //  1. Count  – parallel per fragment: fragShardCount[f*S+s]
    //  2. Prefix sum – serial over nFrags*numShards entries (trivially small)
    //  3. Fill   – parallel per fragment: scatter local vertex indices into binnedIdx
    //  4. Dedup  – parallel per shard: open-addressing hash table, assign local idx
    //  5. Global offsets – serial prefix sum over numShards shard sizes
    //  6. Assemble – parallel: copy shard points into mPoints, patch remap values
    //  7. Triangles – serial, linear in triangle count

    // Flatten ETS for indexed access.
    std::vector<const LocalMesh*> frags;
    for (const LocalMesh& m : pool) frags.push_back(&m);
    const std::size_t nFrags = frags.size();

    // Per-fragment vertex offsets into the flat remap array.
    std::vector<uint32_t> fragVertOff(nFrags + 1, 0);
    std::size_t totalTris = 0;
    for (std::size_t f = 0; f < nFrags; ++f) {
        fragVertOff[f + 1] = fragVertOff[f] +
                              static_cast<uint32_t>(frags[f]->points.size());
        totalTris += frags[f]->triangles.size();
    }
    const uint32_t totalFragVerts = fragVertOff[nFrags];

    // numShards: power-of-2 >= nFrags.
    std::size_t numShards = 1;
    while (numShards < nFrags) numShards <<= 1;
    const std::size_t shardMask = numShards - 1;
    const EdgeKeyHash hasher;

    // 1. Count per (fragment, shard). Flat 2D array [nFrags × numShards].
    //    Reused as fill cursor in pass 3 (zeroed then re-incremented).
    std::vector<uint32_t> fragShardCount(nFrags * numShards, 0);
    tbb::parallel_for(std::size_t(0), nFrags, [&](std::size_t f) {
        uint32_t* cnt = fragShardCount.data() + f * numShards;
        for (const EdgeKey& key : frags[f]->keys)
            ++cnt[hasher(key) & shardMask];
    });

    // 2. Prefix sum → per-(fragment,shard) offsets into binnedIdx.
    std::vector<uint32_t> fragShardOff(nFrags * numShards + 1, 0);
    for (std::size_t i = 0; i < nFrags * numShards; ++i)
        fragShardOff[i + 1] = fragShardOff[i] + fragShardCount[i];
    const uint32_t totalBinned = fragShardOff[nFrags * numShards];

    std::vector<uint32_t> binnedIdx(totalBinned);

    // 3. Fill: scatter each fragment's local vertex indices by shard.
    tbb::parallel_for(std::size_t(0), nFrags, [&](std::size_t f) {
        uint32_t* fill = fragShardCount.data() + f * numShards; // reuse as cursor
        std::fill(fill, fill + numShards, 0u);
        const uint32_t* off = fragShardOff.data() + f * numShards;
        for (uint32_t i = 0, n = static_cast<uint32_t>(frags[f]->keys.size()); i < n; ++i) {
            const std::size_t s = hasher(frags[f]->keys[i]) & shardMask;
            binnedIdx[off[s] + fill[s]++] = i;
        }
    });

    // 4. Parallel per-shard dedup.
    // remap[fragVertOff[f] + i] = local-shard vertex index (patched to global in step 6).
    // Writes are safe: each (f,i) pair belongs to exactly one shard.
    std::vector<Index32> remap(totalFragVerts);
    std::vector<std::vector<Vec3s>> shardPts(numShards);

    tbb::parallel_for(std::size_t(0), numShards, [&](std::size_t s) {
        std::vector<Vec3s>& pts = shardPts[s];

        // Count this shard's total entries across all fragments.
        uint32_t shardTotal = 0;
        for (std::size_t f = 0; f < nFrags; ++f)
            shardTotal += fragShardOff[f * numShards + s + 1]
                        - fragShardOff[f * numShards + s];
        pts.reserve(shardTotal);

        // Open-addressing hash table, load factor ≤ 2/3.
        std::size_t htN = 1;
        while (htN < shardTotal + (shardTotal >> 1) + 1) htN <<= 1;
        const std::size_t htMask = htN - 1;
        const Index32 EMPTY = ~Index32(0);
        std::vector<Index32> ht(htN, EMPTY);
        std::vector<EdgeKey> htKey;
        htKey.reserve(shardTotal);

        for (std::size_t f = 0; f < nFrags; ++f) {
            const uint32_t base  = fragShardOff[f * numShards + s];
            const uint32_t count = fragShardOff[f * numShards + s + 1] - base;
            for (uint32_t bi = 0; bi < count; ++bi) {
                const uint32_t vi  = binnedIdx[base + bi];
                const EdgeKey& key = frags[f]->keys[vi];
                std::size_t slot = hasher(key) & htMask;
                for (;;) {
                    const Index32 g = ht[slot];
                    if (g == EMPTY) {
                        const Index32 li = static_cast<Index32>(pts.size());
                        ht[slot] = li;
                        remap[fragVertOff[f] + vi] = li;
                        pts.push_back(frags[f]->points[vi]);
                        htKey.push_back(key);
                        break;
                    }
                    if (htKey[g] == key) { remap[fragVertOff[f] + vi] = g; break; }
                    slot = (slot + 1) & htMask;
                }
            }
        }
    });

    // 5. Prefix sum → global point offsets per shard.
    std::vector<Index32> shardOffset(numShards + 1, 0);
    for (std::size_t s = 0; s < numShards; ++s)
        shardOffset[s + 1] = shardOffset[s] + static_cast<Index32>(shardPts[s].size());

    // 6. Assemble mPoints and patch remap values to global indices.
    mPoints.resize(shardOffset[numShards]);
    tbb::parallel_for(std::size_t(0), numShards, [&](std::size_t s) {
        const Index32 off = shardOffset[s];
        const std::size_t n = shardPts[s].size();
        for (std::size_t i = 0; i < n; ++i)
            mPoints[off + i] = shardPts[s][i];
        // Each (f,vi) in this shard is exclusively owned by shard s → no race.
        for (std::size_t f = 0; f < nFrags; ++f) {
            const uint32_t base  = fragShardOff[f * numShards + s];
            const uint32_t count = fragShardOff[f * numShards + s + 1] - base;
            for (uint32_t bi = 0; bi < count; ++bi)
                remap[fragVertOff[f] + binnedIdx[base + bi]] += off;
        }
    });

    // 7. Remap triangle indices (parallel over fragments).
    // Pre-compute per-fragment offsets into the output triangle list so each
    // fragment can write to its own slice without synchronisation.
    std::vector<std::size_t> triOff(nFrags + 1, 0);
    for (std::size_t f = 0; f < nFrags; ++f)
        triOff[f + 1] = triOff[f] + frags[f]->triangles.size();

    mTriangles.resize(totalTris);
    tbb::parallel_for(std::size_t(0), nFrags, [&](std::size_t f) {
        const uint32_t foff = fragVertOff[f];
        Vec3I* out = mTriangles.data() + triOff[f];
        for (const Vec3I& tri : frags[f]->triangles)
            *out++ = Vec3I(remap[foff + tri[0]], remap[foff + tri[1]], remap[foff + tri[2]]);
    });
}


template<typename GridType>
void
MarchingTetrahedra<GridType>::operator()(double isovalue)
{
    mPoints.clear();
    mTriangles.clear();

    std::vector<Coord> cells;
    this->gatherCells(cells);
    if (cells.empty()) return;
    this->extract(cells, isovalue);
}


////////////////////////////////////////


template<typename GridType>
void
marchingTetrahedra(
    const GridType& grid,
    std::vector<Vec3s>& points,
    std::vector<Vec3I>& triangles,
    double isovalue)
{
    MarchingTetrahedra<GridType> mesher(grid);
    mesher(isovalue);
    points.swap(mesher.points());
    triangles.swap(mesher.triangles());
}


} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_MARCHING_TETRAHEDRA_HAS_BEEN_INCLUDED
