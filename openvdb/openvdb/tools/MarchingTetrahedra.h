// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file   MarchingTetrahedra.h
///
/// @brief  Extract a triangular isosurface mesh from a scalar volume using the
///         classic Marching Tetrahedra algorithm.
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
#include <unordered_map>
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

    using EdgeVertexMap = std::unordered_map<EdgeKey, Index32, EdgeKeyHash>;

    /// @brief Per-thread mesh fragment produced during the parallel phase.
    ///        Vertices are welded locally (@c weld) and tagged with the edge they
    ///        lie on (@c keys) so the serial merge can re-weld them globally.
    struct LocalMesh
    {
        std::vector<Vec3s>   points;
        std::vector<EdgeKey> keys;      ///< edge key per local vertex
        std::vector<Vec3I>   triangles; ///< indices into @c points
        EdgeVertexMap        weld;
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
    EdgeKey key = (ca < cb) ? EdgeKey(ca, cb) : EdgeKey(cb, ca);
    auto it = mesh.weld.find(key);
    if (it != mesh.weld.end()) return it->second;

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
    mesh.points.emplace_back(static_cast<float>(wp.x()),
                             static_cast<float>(wp.y()),
                             static_cast<float>(wp.z()));
    mesh.keys.push_back(key);
    mesh.weld.emplace(key, idx);
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
    // A cell (identified by its origin/min corner o) can only contain the
    // isosurface if its value varies across the eight corners, which requires at
    // least one active corner. Corner c of cell o sits at o + cornerOffset(c), so
    // o is a candidate iff some active voxel v satisfies v == o + offset, i.e.
    // o == v - offset. We mark those eight origins per active voxel, then gather
    // them into a flat list to partition across threads.
    MaskGrid::Ptr cellMask = MaskGrid::create();
    MaskGrid::Accessor maskAcc = cellMask->getAccessor();
    for (auto leafIt = mGrid.tree().cbeginLeaf(); leafIt; ++leafIt) {
        for (auto voxIt = leafIt->cbeginValueOn(); voxIt; ++voxIt) {
            const Coord v = voxIt.getCoord();
            for (int c = 0; c < 8; ++c) maskAcc.setValueOn(v - cornerOffset(c));
        }
    }

    cells.clear();
    cells.reserve(static_cast<size_t>(cellMask->activeVoxelCount()));
    for (auto it = cellMask->cbeginValueOn(); it; ++it) cells.push_back(it.getCoord());
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
    tbb::enumerable_thread_specific<LocalMesh> pool;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, cells.size()),
        [&](const tbb::blocked_range<size_t>& range)
    {
        LocalMesh& mesh = pool.local();
        AccessorT  acc  = mGrid.getConstAccessor(); // per-invocation, thread-safe
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

    // ---- Merge thread-local fragments (serial) -----------------------------
    // Re-weld each fragment's vertices by their edge key so vertices on thread
    // (range) boundaries collapse to a single shared vertex, then append the
    // remapped triangles. The interpolated position for a given edge is identical
    // across fragments, so keeping the first occurrence is exact.
    //
    // A flat, open-addressing hash table (a pre-sized array of global vertex
    // indices resolved by linear probing) is used rather than a node-based
    // std::unordered_map: the latter's ~one-heap-allocation-per-vertex dominated
    // the merge. `mGlobalKey[g]` holds the edge key of global vertex `g`, used to
    // resolve probe collisions.
    std::size_t totalVerts = 0, totalTris = 0;
    for (const LocalMesh& mesh : pool) {
        totalVerts += mesh.points.size();
        totalTris  += mesh.triangles.size();
    }
    mPoints.reserve(totalVerts);
    mTriangles.reserve(totalTris);

    std::size_t n = 1;
    while (n < totalVerts + (totalVerts >> 1) + 1) n <<= 1; // load factor <= 2/3
    const std::size_t mask = n - 1;
    const Index32 EMPTY = ~Index32(0);
    std::vector<Index32> table(n, EMPTY);
    std::vector<EdgeKey> globalKey;
    globalKey.reserve(totalVerts);
    const EdgeKeyHash hasher;

    for (LocalMesh& mesh : pool) {
        std::vector<Index32> remap(mesh.points.size());
        for (std::size_t i = 0; i < mesh.points.size(); ++i) {
            const EdgeKey& key = mesh.keys[i];
            std::size_t slot = hasher(key) & mask;
            for (;;) {
                const Index32 g = table[slot];
                if (g == EMPTY) { // new vertex
                    const Index32 gi = static_cast<Index32>(mPoints.size());
                    table[slot] = gi;
                    mPoints.push_back(mesh.points[i]);
                    globalKey.push_back(key);
                    remap[i] = gi;
                    break;
                }
                if (globalKey[g] == key) { remap[i] = g; break; } // already welded
                slot = (slot + 1) & mask;
            }
        }
        for (const Vec3I& tri : mesh.triangles) {
            mTriangles.emplace_back(remap[tri[0]], remap[tri[1]], remap[tri[2]]);
        }
    }
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
