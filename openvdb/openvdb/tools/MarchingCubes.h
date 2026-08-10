// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file   MarchingCubes.h
///
/// @brief  Extract a triangular isosurface mesh from a scalar volume using the
///         topologically correct Marching Cubes 33 algorithm.
///
/// @author Ken Museth
///
/// @details Each grid cell (the cube spanned by eight neighboring voxels) is
///          classified by the sign configuration of its eight corners and
///          polygonized by looking the configuration up in a case table and
///          linearly interpolating the isosurface crossing along the twelve cube
///          edges. Unlike the original Marching Cubes (Lorensen & Cline 1987),
///          which leaves ambiguous faces and interiors unresolved and can produce
///          cracks or wrong topology, this implementation follows the Marching
///          Cubes 33 case analysis of Chernyaev (1995) as organized by Lewiner et
///          al. (2003), using the asymptotic-decider face test and the *corrected*
///          interior test of Vega, Abache & Coll (2019). The result is a
///          watertight, topologically consistent indexed triangle mesh. Vertices
///          are welded per edge, so adjacent cells share them exactly.
///
/// @note    The case table below encodes the combinatorial output of the above
///          case analysis. It is reformatted and documented here in OpenVDB
///          style; the numeric content is attributable to the cited papers and is
///          not derived independently. Only the algorithm — traversal, the face
///          and interior tests, interpolation and welding — is a fresh
///          implementation optimized for the sparse VDB data structure.
///
/// @par References
///  - W. Lorensen and H. Cline, "Marching Cubes: A High Resolution 3D Surface
///    Construction Algorithm", SIGGRAPH 1987.
///  - E. Chernyaev, "Marching Cubes 33: Construction of Topologically Correct
///    Isosurfaces", Technical Report CERN CN 95-17, 1995.
///  - T. Lewiner, H. Lopes, A. Vieira and G. Tavares, "Efficient Implementation
///    of Marching Cubes' Cases with Topological Guarantees", JGT 8(2), 2003.
///  - D. Vega, J. Abache and D. Coll, "A Fast and Memory-Saving Marching Cubes 33
///    Implementation with the Correct Interior Test", JCGT 8(3):1-18, 2019.
///
///          Use the MarchingCubes class directly to keep the extracted mesh for
///          repeated queries, or the free-standing marchingCubes() function to
///          write the result straight into std::vector lists of world-space vertex
///          coordinates and triangle vertex indices (like tools::volumeToMesh).

#ifndef OPENVDB_TOOLS_MARCHING_CUBES_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_MARCHING_CUBES_HAS_BEEN_INCLUDED

#include <openvdb/openvdb.h>
#include <openvdb/Types.h>
#include <openvdb/math/Math.h>       // for math::Clamp, math::isApproxZero
#include <openvdb/math/Transform.h>
#include <openvdb/math/Vec3.h>

#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include <openvdb/util/CpuTimer.h>

#include <algorithm>
#include <cmath>      // for std::signbit
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <type_traits>
#include <utility>    // for std::pair
#include <vector>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {


////////////////////////////////////////


/// @brief Mesh a scalar grid's isosurface with the Marching Cubes 33 algorithm.
///
/// @param grid       a scalar (floating-point) grid to mesh
/// @param points     output list of world-space vertex positions
/// @param triangles  output triangle index list (indices into @a points)
/// @param isovalue   the isosurface to mesh (defaults to zero)
///
/// @details Both output vectors are cleared before use. A thin wrapper around the
///          MarchingCubes class whose result is swapped into @a points and
///          @a triangles (no copy).
template<typename GridType>
void marchingCubes(
    const GridType& grid,
    std::vector<Vec3s>& points,
    std::vector<Vec3I>& triangles,
    double isovalue = 0.0);


////////////////////////////////////////


/// @brief Extract a triangular isosurface mesh from a scalar grid with the
///        topologically correct Marching Cubes 33 algorithm.
///
/// @details Construct the mesher with the grid to be meshed, then call it with the
///          desired isovalue; the resulting welded, watertight triangle mesh is
///          available through points() and triangles(). The same instance can be
///          re-invoked with a different isovalue on the same grid.
///
/// @note Only regions where the value varies (i.e. the active narrow band of a
///       level set, or the varying shell of a fog volume) are polygonized;
///       constant active tiles contain no crossing and are skipped. Triangles are
///       wound so their normals point toward increasing value (out of the
///       isosurface for a level set stored negative-inside).
template<typename GridType>
class MarchingCubes
{
public:
    using ValueType = typename GridType::ValueType;

    static_assert(std::is_floating_point<ValueType>::value,
        "MarchingCubes requires a scalar floating-point grid");

    /// @brief Construct a mesher bound to @a grid.
    explicit MarchingCubes(const GridType& grid): mGrid(grid) {}

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
    ///        coordinates. An interior ("tunnel") vertex at a cell center is keyed
    ///        by (origin, origin), which can never collide with a real edge.
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
    ///        load factor <= 2/3). It stores vertex indices into @c points, using
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

    /// @brief Index-space offset of cube corner @a c, in the Marching Cubes 33
    ///        vertex numbering (see the figure by sMC33Table): corner 0 is the
    ///        cell origin, corners 0-3 the x-low face and 4-7 the x-high face.
    static Coord cornerOffset(int c)
    {
        static constexpr int off[8][3] = {
            {0,0,0}, {0,1,0}, {0,1,1}, {0,0,1},
            {1,0,0}, {1,1,0}, {1,1,1}, {1,0,1}
        };
        return Coord(off[c][0], off[c][1], off[c][2]);
    }

    /// @brief The two corner indices spanned by each of the twelve cube edges, in
    ///        the Marching Cubes 33 edge numbering.
    static constexpr int sEdge[12][2] = {
        {0,1}, {1,2}, {2,3}, {0,3}, {4,5}, {5,6},
        {6,7}, {4,7}, {0,4}, {1,5}, {2,6}, {3,7}
    };

    /// @brief Marching Cubes 33 lookup table (config index + triangle patterns).
    ///        See the class comment for the encoding and references.
    static const unsigned short sMC33Table[2310];

    /// @brief Asymptotic-decider test on all six cube faces (Nielson-Hamann).
    ///        Returns per-face results in @a face (+1 if the positive-corner pair
    ///        is joined across the face, -1 if the negative pair, 0 if untested)
    ///        and their sum. @a ind is the (possibly complemented) configuration.
    static int faceTests(int face[6], int ind, const double v[8]);

    /// @brief Asymptotic-decider test on a single face (fast path for cases 3, 6).
    static int faceTest1(int face, const double v[8]);

    /// @brief Interior (body) test of Vega, Abache & Coll (2019). @a i selects the
    ///        interior diagonal (0: 0-6, 1: 1-7, 2: 2-4, 3: 3-5). With @a flag13
    ///        set (case 13) it distinguishes the 13.5.1 / 13.5.2 subcases.
    static int interiorTest(int i, int flag13, const double v[8]);

    /// @brief Weld a vertex with the given @a key and world-space @a pos into the
    ///        fragment's flat hash table, returning its (locally welded) index.
    static Index32 weldVertex(LocalMesh& mesh, const EdgeKey& key, const Vec3d& pos);

    /// @brief Return the (locally welded) vertex index for the crossing on the edge
    ///        (@a ca,@a cb), creating it if new.
    static Index32 localCrossing(LocalMesh& mesh, const math::Transform& xform,
        double isovalue, const Coord& ca, ValueType va, const Coord& cb, ValueType vb);

    /// @brief Return the (locally welded) vertex index for a cell's interior
    ///        "tunnel" vertex, placed at the cell center.
    static Index32 centerVertex(LocalMesh& mesh, const math::Transform& xform,
        const Coord& origin);

    /// @brief Polygonize one cell (its eight @a corner coordinates and @a value s,
    ///        with min-corner @a origin) into @a mesh via the MC33 case table.
    static void marchCell(LocalMesh& mesh, const math::Transform& xform,
        double isovalue, const Coord& origin, const Coord corner[8],
        const ValueType value[8]);

    using MaskLeaf = MaskGrid::TreeType::LeafNodeType;

    /// @brief Build a MaskGrid whose active voxels are the origins of every
    ///        candidate cell, and populate @a masterLeaves with pointers into
    ///        that grid's leaves. The returned MaskGrid must stay alive as long
    ///        as @a masterLeaves is used.
    MaskGrid::Ptr gatherCells(std::vector<const MaskLeaf*>& masterLeaves) const;

    /// @brief March the given master leaves (in parallel) and merge the result
    ///        into mPoints / mTriangles.
    void extract(const std::vector<const MaskLeaf*>& masterLeaves, double isovalue);

    const GridType&    mGrid;
    std::vector<Vec3s> mPoints;
    std::vector<Vec3I> mTriangles;
}; // class MarchingCubes


////////////////////////////////////////

// MarchingCubes implementation


template<typename GridType>
constexpr int MarchingCubes<GridType>::sEdge[12][2];


template<typename GridType>
const unsigned short MarchingCubes<GridType>::sMC33Table[2310] =
{
    // ---- Configuration index: 256 sign configurations (128 stored; the upper 128
    //   reuse the complement entry with reversed winding). Bits[0-10]=pattern
    //   offset/key, bit 11=reverse winding, bits[12-15]=MC33 case group.
    0x0000, 0x0885, 0x0886, 0x0895, 0x0883, 0x1816, 0x089D, 0x0943, 0x0884, 0x0897, 0x1814, 0x0916,
    0x0891, 0x094C, 0x091F, 0x048F, 0x0882, 0x089B, 0x1808, 0x0934, 0x2803, 0x3817, 0x3814, 0x0525,
    0x180E, 0x0928, 0x4802, 0x049D, 0x3815, 0x0541, 0x6004, 0x0110, 0x0881, 0x180A, 0x0899, 0x0922,
    0x1806, 0x4800, 0x0913, 0x0499, 0x2802, 0x380E, 0x3811, 0x053D, 0x380D, 0x6002, 0x0521, 0x010D,
    0x088B, 0x0946, 0x0937, 0x0493, 0x3813, 0x6003, 0x0545, 0x012E, 0x380F, 0x0531, 0x600A, 0x011C,
    0x5001, 0x3001, 0x3009, 0x0087, 0x0880, 0x2801, 0x1804, 0x3807, 0x088F, 0x380B, 0x092B, 0x0539,
    0x1802, 0x3806, 0x4806, 0x6001, 0x0925, 0x051D, 0x0495, 0x010A, 0x1812, 0x380A, 0x4805, 0x6009,
    0x3816, 0x5002, 0x6008, 0x3010, 0x4801, 0x6000, 0x7000, 0x4003, 0x6006, 0x3004, 0x4007, 0x1010,
    0x0889, 0x3808, 0x093A, 0x052D, 0x0949, 0x6005, 0x0491, 0x0131, 0x380C, 0x5000, 0x600B, 0x3012,
    0x0549, 0x3002, 0x0119, 0x008D, 0x0907, 0x0535, 0x04A1, 0x013D, 0x0529, 0x3005, 0x0140, 0x0093,
    0x6007, 0x3000, 0x4004, 0x1000, 0x3003, 0x2000, 0x100C, 0x007F,
    // ---- Case 1 (offset 128) ----------------------------------
    0x0380, 0x0109, 0x021A, 0x0B32, 0x0945, 0x0748, 0x07B6, 0x06A5,
    // ---- Case 2 (offset 136) ----------------------------------
    0x1189, 0x0138, 0x129A, 0x0092, 0x1B3A, 0x031A, 0x12B0, 0x0B80, 0x1045, 0x0105, 0x1975, 0x0987,
    0x1340, 0x0374, 0x1BA5, 0x07B5, 0x1486, 0x0B68, 0x1615, 0x0216, 0x1726, 0x0732, 0x146A, 0x094A,
    // ---- Case 3.1 (offset 160) --------------------------------
    0x1945, 0x0038, 0x1109, 0x0748, 0x16A5, 0x0109, 0x1945, 0x021A, 0x16A5, 0x032B, 0x17B6, 0x021A,
    0x17B6, 0x0038, 0x1B32, 0x0748, 0x121A, 0x0038, 0x1B32, 0x0109, 0x16A5, 0x0874, 0x1945, 0x07B6,
    // ---- Case 3.2 (offset 184) --------------------------------
    0x1905, 0x1035, 0x1453, 0x0843, 0x1974, 0x1917, 0x1871, 0x0081, 0x1605, 0x1950, 0x116A, 0x0106,
    0x1A45, 0x1942, 0x124A, 0x0192, 0x13A5, 0x1B56, 0x132A, 0x0B35, 0x11A6, 0x17B2, 0x1217, 0x0671,
    0x1786, 0x1806, 0x1B60, 0x03B0, 0x1834, 0x1324, 0x1742, 0x0B72, 0x123A, 0x138A, 0x11A8, 0x0018,
    0x1129, 0x12B9, 0x109B, 0x030B, 0x14A5, 0x1A86, 0x148A, 0x0768, 0x1B65, 0x1794, 0x17B9, 0x059B,
    // ---- Case 4.1.1 (offset 232) ------------------------------
    0x16A5, 0x0380, 0x17B6, 0x0109, 0x121A, 0x0748, 0x1945, 0x0B32,
    // ---- Case 4.1.2 (offset 240) ------------------------------
    0x10A5, 0x1805, 0x1863, 0x1685, 0x16A3, 0x03A0, 0x1796, 0x11B6, 0x1169, 0x1970, 0x17B0, 0x00B1,
    0x174A, 0x17A2, 0x1872, 0x1A41, 0x1481, 0x0182, 0x12B5, 0x1B34, 0x1943, 0x15B4, 0x1592, 0x0293,
    // ---- Case 5 (offset 264) ----------------------------------
    0x1B9A, 0x1930, 0x09B3, 0x180A, 0x101A, 0x0B8A, 0x189B, 0x1B12, 0x0B91, 0x128A, 0x1382, 0x09A8,
    0x1246, 0x1419, 0x0421, 0x18A5, 0x1485, 0x0BA8, 0x1026, 0x1067, 0x0078, 0x1845, 0x1538, 0x0135,
    0x176A, 0x1A87, 0x098A, 0x1715, 0x11B2, 0x017B, 0x1175, 0x1708, 0x0710, 0x1426, 0x1832, 0x0248,
    0x106A, 0x1460, 0x010A, 0x1149, 0x1174, 0x0371, 0x142B, 0x174B, 0x0024, 0x12A5, 0x1532, 0x0735,
    0x1635, 0x136B, 0x0153, 0x1695, 0x1609, 0x0206, 0x1935, 0x1390, 0x0753, 0x13B6, 0x1360, 0x0406,
    0x19BA, 0x17B4, 0x0B94, 0x17A6, 0x171A, 0x0317, 0x1A25, 0x1245, 0x0042, 0x1965, 0x19B6, 0x08B9,
    // ---- Case 6.1.1 (offset 336) ------------------------------
    0x146A, 0x1A94, 0x0380, 0x16A5, 0x1189, 0x0138, 0x16A5, 0x180B, 0x002B, 0x1BA5, 0x157B, 0x0038,
    0x1615, 0x1621, 0x0803, 0x16A5, 0x1034, 0x0374, 0x18B6, 0x1648, 0x0109, 0x1BA5, 0x17B5, 0x0109,
    0x129A, 0x17B6, 0x0092, 0x17B6, 0x1189, 0x0138, 0x1726, 0x1732, 0x0109, 0x1045, 0x17B6, 0x0105,
    0x129A, 0x1209, 0x0748, 0x1975, 0x121A, 0x0879, 0x1486, 0x121A, 0x08B6, 0x131A, 0x1B3A, 0x0874,
    0x121A, 0x1374, 0x0403, 0x1615, 0x1216, 0x0874, 0x1945, 0x12B0, 0x0B80, 0x1945, 0x1B3A, 0x031A,
    0x146A, 0x194A, 0x032B, 0x1975, 0x1987, 0x02B3, 0x1045, 0x1510, 0x0B32, 0x1945, 0x1726, 0x0732,
    // ---- Case 6.1.2 (offset 408) ------------------------------
    0x136A, 0x190A, 0x1094, 0x1804, 0x1684, 0x1386, 0x03A0, 0x1895, 0x11A5, 0x136A, 0x1591, 0x13A1,
    0x1863, 0x0856, 0x10A5, 0x1AB6, 0x102A, 0x1A2B, 0x186B, 0x1568, 0x0058, 0x10A5, 0x1785, 0x187B,
    0x138B, 0x1A3B, 0x103A, 0x0058, 0x1685, 0x1236, 0x1321, 0x1031, 0x1501, 0x1805, 0x0863, 0x10A5,
    0x1456, 0x1376, 0x1674, 0x1054, 0x13A0, 0x0A36, 0x1496, 0x1948, 0x1098, 0x1B08, 0x110B, 0x161B,
    0x0169, 0x1795, 0x11A5, 0x11BA, 0x1915, 0x1097, 0x1B07, 0x00B1, 0x19A6, 0x16A2, 0x1B62, 0x10B2,
    0x17B0, 0x1970, 0x0796, 0x1796, 0x113B, 0x1B38, 0x17B8, 0x1978, 0x1169, 0x01B6, 0x1796, 0x1730,
    0x1032, 0x1102, 0x1612, 0x1916, 0x0970, 0x1165, 0x1745, 0x1756, 0x1704, 0x1B61, 0x10B1, 0x0B07,
    0x149A, 0x1208, 0x1809, 0x1489, 0x174A, 0x127A, 0x0728, 0x19A5, 0x175A, 0x11A9, 0x1819, 0x1218,
    0x1728, 0x027A, 0x14A6, 0x18B2, 0x12B6, 0x1A26, 0x11A4, 0x1814, 0x0182, 0x141A, 0x1B7A, 0x17B3,
    0x1873, 0x1183, 0x1481, 0x04A7, 0x141A, 0x1014, 0x1103, 0x1213, 0x1723, 0x1A27, 0x04A7, 0x1645,
    0x1415, 0x1746, 0x1276, 0x1872, 0x1182, 0x0814, 0x1925, 0x1B84, 0x1480, 0x1940, 0x1290, 0x1B52,
    0x05B4, 0x19A5, 0x1319, 0x191A, 0x1B5A, 0x145B, 0x134B, 0x0439, 0x1B6A, 0x1B46, 0x12BA, 0x192A,
    0x1329, 0x1439, 0x034B, 0x12B5, 0x1398, 0x1387, 0x1B37, 0x15B7, 0x1925, 0x0293, 0x1B45, 0x1125,
    0x1210, 0x1320, 0x1430, 0x1B34, 0x0B52, 0x1265, 0x1574, 0x1567, 0x1347, 0x1943, 0x1293, 0x0925,
    // ---- Case 6.2 (offset 576) --------------------------------
    0x136A, 0x190A, 0x13A0, 0x1684, 0x0386, 0x1685, 0x136A, 0x1895, 0x113A, 0x0863, 0x10A5, 0x1856,
    0x102A, 0x186B, 0x0058, 0x10A5, 0x1785, 0x1058, 0x1A3B, 0x003A, 0x1685, 0x1236, 0x1863, 0x1501,
    0x0805, 0x10A5, 0x1A36, 0x1054, 0x1376, 0x03A0, 0x1496, 0x1169, 0x1B08, 0x110B, 0x061B, 0x1795,
    0x11BA, 0x10B1, 0x1097, 0x0B07, 0x19A6, 0x1796, 0x10B2, 0x17B0, 0x0970, 0x1796, 0x113B, 0x161B,
    0x1978, 0x0169, 0x1796, 0x1126, 0x1730, 0x1970, 0x0916, 0x1165, 0x1704, 0x1B07, 0x1B61, 0x00B1,
    0x149A, 0x1208, 0x1728, 0x174A, 0x027A, 0x1A75, 0x127A, 0x1819, 0x1218, 0x0728, 0x14A6, 0x18B2,
    0x1182, 0x11A4, 0x0814, 0x174A, 0x1B7A, 0x1183, 0x1481, 0x0A41, 0x141A, 0x1014, 0x1723, 0x1A27,
    0x04A7, 0x1415, 0x1276, 0x1814, 0x1872, 0x0182, 0x1925, 0x1B84, 0x15B4, 0x1290, 0x0B52, 0x1AB5,
    0x1319, 0x1439, 0x145B, 0x034B, 0x1B46, 0x192A, 0x134B, 0x1329, 0x0439, 0x12B5, 0x1398, 0x1293,
    0x15B7, 0x0925, 0x12B5, 0x1125, 0x1430, 0x1B34, 0x05B4, 0x1265, 0x1925, 0x1734, 0x1943, 0x0293,
    // ---- Case 7.1 (offset 696) --------------------------------
    0x1945, 0x121A, 0x07B6, 0x12B3, 0x1874, 0x0109, 0x16A5, 0x1B32, 0x0874, 0x1945, 0x121A, 0x0038,
    0x1945, 0x17B6, 0x0038, 0x16A5, 0x1B32, 0x0109, 0x16A5, 0x1109, 0x0748, 0x121A, 0x17B6, 0x0380,
    // ---- Case 7.2 (offset 720) --------------------------------
    0x1B65, 0x19B5, 0x121A, 0x1794, 0x0B97, 0x1B92, 0x1874, 0x1B30, 0x19B0, 0x0912, 0x14A5, 0x1A86,
    0x1B32, 0x1876, 0x08A4, 0x1945, 0x138A, 0x1801, 0x1A81, 0x0A23, 0x1B65, 0x1380, 0x1794, 0x1B97,
    0x09B5, 0x16A5, 0x1129, 0x1B92, 0x1B30, 0x09B0, 0x14A5, 0x1876, 0x1109, 0x18A4, 0x0A86, 0x17B6,
    0x123A, 0x18A3, 0x1801, 0x0A81, 0x1A45, 0x17B6, 0x124A, 0x1219, 0x0429, 0x1109, 0x1483, 0x1243,
    0x12B7, 0x0427, 0x16A5, 0x1274, 0x12B7, 0x1483, 0x0243, 0x1A45, 0x1038, 0x1219, 0x1429, 0x024A,
    0x1945, 0x1806, 0x1678, 0x103B, 0x060B, 0x1605, 0x1095, 0x116A, 0x132B, 0x0061, 0x1605, 0x1095,
    0x116A, 0x1748, 0x0061, 0x1786, 0x121A, 0x103B, 0x160B, 0x0068, 0x1945, 0x11A6, 0x1716, 0x17B2,
    0x0172, 0x132B, 0x1749, 0x1108, 0x1718, 0x0179, 0x13A5, 0x1B56, 0x1748, 0x135B, 0x032A, 0x1905,
    0x121A, 0x1350, 0x1384, 0x0534, 0x1905, 0x1534, 0x17B6, 0x1384, 0x0350, 0x13A5, 0x1B56, 0x1109,
    0x132A, 0x035B, 0x16A5, 0x1749, 0x1179, 0x1108, 0x0718, 0x11A6, 0x1380, 0x17B2, 0x1172, 0x0716,
    // ---- Case 7.3 (offset 840) --------------------------------
    0x1C65, 0x1C5A, 0x17C4, 0x1C7B, 0x1C94, 0x1C19, 0x1C21, 0x1CA2, 0x0CB6, 0x1C74, 0x1CB7, 0x1C2B,
    0x11C9, 0x1C12, 0x1C09, 0x1C30, 0x1C83, 0x0C48, 0x1CA5, 0x1C6A, 0x1C32, 0x1C83, 0x1C48, 0x1C54,
    0x1C76, 0x1CB7, 0x0C2B, 0x1AC5, 0x1C38, 0x1C23, 0x1CA2, 0x1C45, 0x1C94, 0x1C19, 0x1C01, 0x0C80,
    0x1C65, 0x19C5, 0x1CB6, 0x1C3B, 0x1C03, 0x1C80, 0x17C4, 0x1C78, 0x0C94, 0x16C5, 0x1C6A, 0x1C95,
    0x1C09, 0x1C30, 0x1CB3, 0x1C2B, 0x1C12, 0x0CA1, 0x1C95, 0x1C6A, 0x1C10, 0x1CA1, 0x1C48, 0x1C76,
    0x1C87, 0x1C54, 0x0C09, 0x1C1A, 0x17C6, 0x1C01, 0x1C80, 0x1C78, 0x1CB6, 0x1C3B, 0x1C23, 0x0CA2,
    0x1C65, 0x1CA6, 0x1C59, 0x11C2, 0x1CB2, 0x17C4, 0x1C7B, 0x1C94, 0x0C1A, 0x1C2B, 0x11C9, 0x1C12,
    0x1C49, 0x1C74, 0x1C87, 0x1C08, 0x1C30, 0x0CB3, 0x1CA5, 0x1C54, 0x1C76, 0x1C48, 0x1C2A, 0x1C32,
    0x1CB3, 0x1C6B, 0x0C87, 0x1C45, 0x12CA, 0x1C84, 0x1C38, 0x1C23, 0x1C59, 0x1C1A, 0x1C01, 0x0C90,
    0x1C65, 0x1C03, 0x1C90, 0x1C59, 0x1CB6, 0x17C4, 0x1C7B, 0x1C84, 0x0C38, 0x1CA5, 0x1C56, 0x1C09,
    0x1C30, 0x1CB3, 0x1C6B, 0x1C2A, 0x1C12, 0x0C91, 0x1CA5, 0x1C6A, 0x1C54, 0x1C76, 0x1C87, 0x1C08,
    0x11C9, 0x1C10, 0x0C49, 0x1CA6, 0x17C6, 0x1C1A, 0x1C01, 0x1C80, 0x1C38, 0x1C23, 0x1CB2, 0x0C7B,
    0x1AC5, 0x1CA6, 0x1C94, 0x1C19, 0x1C21, 0x1CB2, 0x1C7B, 0x1C67, 0x0C45, 0x1C2B, 0x11C9, 0x1C49,
    0x1C74, 0x1CB7, 0x1C32, 0x1C83, 0x1C08, 0x0C10, 0x1CA5, 0x1C56, 0x1C2A, 0x1C32, 0x1C83, 0x1C48,
    0x1C74, 0x1CB7, 0x0C6B, 0x1AC5, 0x12CA, 0x1C45, 0x1C84, 0x1C38, 0x1C03, 0x1C90, 0x1C19, 0x0C21,
    0x1C45, 0x1CB6, 0x1C3B, 0x1C03, 0x1C90, 0x1C59, 0x1C84, 0x1C78, 0x0C67, 0x16C5, 0x1C2A, 0x13CB,
    0x1C6B, 0x1C95, 0x1C09, 0x1C10, 0x1CA1, 0x0C32, 0x16C5, 0x1C6A, 0x1C95, 0x1C74, 0x17C8, 0x1C08,
    0x1C10, 0x1CA1, 0x0C49, 0x1CA6, 0x1C80, 0x1C78, 0x1C67, 0x1C1A, 0x1C21, 0x1CB2, 0x1C3B, 0x0C03,
    // ---- Case 7.4.1 (offset 1056) ------------------------------
    0x1A65, 0x1419, 0x11B2, 0x17B4, 0x04B1, 0x174B, 0x1149, 0x12B1, 0x11B4, 0x0830, 0x12A5, 0x1485,
    0x1B76, 0x1832, 0x0582, 0x1A25, 0x1238, 0x1458, 0x1528, 0x0019, 0x1965, 0x1390, 0x1B63, 0x1693,
    0x0784, 0x1695, 0x112A, 0x1093, 0x1B36, 0x0396, 0x1495, 0x176A, 0x110A, 0x1870, 0x07A0, 0x17A6,
    0x1780, 0x11A0, 0x1A70, 0x03B2,
    // ---- Case 7.4.2 (offset 1096) ------------------------------
    0x1465, 0x1459, 0x121A, 0x1AB2, 0x1BA6, 0x17B6, 0x1476, 0x15A1, 0x0951, 0x1084, 0x1748, 0x18B7,
    0x1B83, 0x12B3, 0x1109, 0x1130, 0x1123, 0x0904, 0x16A5, 0x132B, 0x1B83, 0x1874, 0x18B7, 0x1576,
    0x1547, 0x16B2, 0x0A62, 0x1945, 0x115A, 0x1038, 0x1023, 0x1201, 0x1A21, 0x1519, 0x1908, 0x0498,
    0x1465, 0x1380, 0x1890, 0x1984, 0x1594, 0x1647, 0x1B67, 0x1783, 0x0B73, 0x16A5, 0x1A95, 0x19A1,
    0x1091, 0x1312, 0x1301, 0x1B32, 0x12A6, 0x0B26, 0x16A5, 0x1109, 0x19A1, 0x1A95, 0x1754, 0x1765,
    0x1874, 0x1490, 0x0840, 0x1BA6, 0x1380, 0x1378, 0x173B, 0x167B, 0x1AB2, 0x11A2, 0x1230, 0x0120,
    // ---- Case 8 (offset 1168) ----------------------------------
    0x1B9A, 0x09B8, 0x1426, 0x0024, 0x1375, 0x0135,
    // ---- Case 9 (offset 1174) ----------------------------------
    0x17A6, 0x1180, 0x1781, 0x0A71, 0x1B42, 0x1129, 0x1492, 0x04B7, 0x1A35, 0x13A2, 0x1853, 0x0584,
    0x1965, 0x13B0, 0x10B6, 0x0069,
    // ---- Case 10.1.1 (offset 1190) -----------------------------
    0x146A, 0x194A, 0x1028, 0x082B, 0x17A5, 0x1189, 0x1381, 0x07BA, 0x1625, 0x1340, 0x1743, 0x0521,
    0x1846, 0x190A, 0x186B, 0x002A, 0x1795, 0x11BA, 0x1789, 0x03B1, 0x1015, 0x1236, 0x1540, 0x0763,
    // ---- Case 10.1.2 (offset 1214) -----------------------------
    0x126A, 0x1029, 0x12A9, 0x1B62, 0x16B8, 0x1468, 0x1489, 0x0809, 0x19A5, 0x1789, 0x1957, 0x11A9,
    0x1A13, 0x1BA3, 0x1B37, 0x0387, 0x1405, 0x1756, 0x1015, 0x1021, 0x1320, 0x1237, 0x1627, 0x0745,
    0x146A, 0x1A94, 0x1904, 0x1408, 0x1B80, 0x1B02, 0x1AB2, 0x0A6B, 0x1BA5, 0x1189, 0x1138, 0x13B8,
    0x18B7, 0x157B, 0x115A, 0x0195, 0x1465, 0x1156, 0x1621, 0x1231, 0x1130, 0x1403, 0x1437, 0x0647,
    // ---- Case 10.2 (offset 1262) -------------------------------
    0x19CA, 0x1AC6, 0x190C, 0x102C, 0x12BC, 0x18CB, 0x184C, 0x046C, 0x1C95, 0x1CBA, 0x1C7B, 0x1C57,
    0x19C8, 0x1C38, 0x1C13, 0x01CA, 0x1C15, 0x12C6, 0x1C21, 0x14C5, 0x1C76, 0x17C3, 0x1C03, 0x0C40,
    0x1C46, 0x1C2A, 0x1C94, 0x1CA9, 0x12C0, 0x1C80, 0x1CB8, 0x0BC6, 0x1CA5, 0x17C5, 0x1CBA, 0x1C3B,
    0x11C9, 0x13C1, 0x1C89, 0x08C7, 0x16C5, 0x12C6, 0x123C, 0x174C, 0x137C, 0x10C4, 0x101C, 0x015C,
    // ---- Case 11 (offset 1310) ---------------------------------
    0x16B5, 0x1B80, 0x15B0, 0x0150, 0x1786, 0x1189, 0x1126, 0x0681, 0x129A, 0x1974, 0x1792, 0x0372,
    0x14A5, 0x13BA, 0x13A4, 0x0340, 0x1975, 0x1902, 0x1927, 0x072B, 0x116A, 0x1846, 0x1861, 0x0813,
    // ---- Case 14 (offset 1334) ---------------------------------
    0x176A, 0x1A90, 0x17A0, 0x0370, 0x1B1A, 0x1140, 0x11B4, 0x074B, 0x1125, 0x1285, 0x12B8, 0x0458,
    0x1695, 0x1236, 0x1396, 0x0389, 0x1496, 0x1139, 0x1369, 0x063B, 0x12A5, 0x1785, 0x1825, 0x0802,
    // ---- Case 12.1.1 (offset 1358) -----------------------------
    0x1246, 0x1192, 0x1429, 0x0038, 0x1945, 0x181A, 0x1180, 0x0B8A, 0x16A5, 0x1129, 0x1B92, 0x089B,
    0x16A5, 0x1749, 0x1179, 0x0371, 0x123A, 0x17B6, 0x18A3, 0x09A8, 0x16A5, 0x1274, 0x172B, 0x0024,
    0x1715, 0x17B2, 0x1172, 0x0380, 0x19BA, 0x1794, 0x1B97, 0x0803, 0x1406, 0x121A, 0x103B, 0x060B,
    0x1905, 0x121A, 0x1350, 0x0753, 0x1345, 0x17B6, 0x1384, 0x0135, 0x1945, 0x1786, 0x1068, 0x0260,
    0x1246, 0x1384, 0x1342, 0x0190, 0x1A45, 0x14A8, 0x18AB, 0x0019, 0x16B5, 0x15B9, 0x112A, 0x09B8,
    0x1495, 0x116A, 0x1617, 0x0713, 0x1786, 0x168A, 0x1A89, 0x023B, 0x14A5, 0x1B76, 0x1A42, 0x0240,
    0x1715, 0x1180, 0x1817, 0x0B23, 0x19BA, 0x13B0, 0x10B9, 0x0784, 0x11A6, 0x1160, 0x1064, 0x03B2,
    0x1A35, 0x123A, 0x1537, 0x0019, 0x1B65, 0x1B53, 0x1351, 0x0784, 0x1065, 0x1905, 0x1602, 0x0784,
    // ---- Case 12.1.2 (offset 1454) -----------------------------
    0x1846, 0x1948, 0x1980, 0x1901, 0x1863, 0x1362, 0x1132, 0x0103, 0x1AB5, 0x1159, 0x11A5, 0x1190,
    0x15B4, 0x14B8, 0x1048, 0x0094, 0x1685, 0x126A, 0x1589, 0x11A5, 0x12B6, 0x16B8, 0x12A1, 0x0159,
    0x19A5, 0x1A36, 0x191A, 0x1A13, 0x1954, 0x1637, 0x1467, 0x0456, 0x126A, 0x1738, 0x137B, 0x1789,
    0x13B2, 0x1796, 0x169A, 0x02B6, 0x10A5, 0x1B6A, 0x1745, 0x1756, 0x1540, 0x176B, 0x1A02, 0x0BA2,
    0x1015, 0x1102, 0x1203, 0x123B, 0x1058, 0x1857, 0x1B87, 0x0B38, 0x13BA, 0x1784, 0x137B, 0x1738,
    0x13A0, 0x10A9, 0x1409, 0x0480, 0x1B6A, 0x1BA2, 0x1A64, 0x1B23, 0x1A41, 0x1140, 0x1310, 0x0321,
    0x19A5, 0x1320, 0x1021, 0x1237, 0x1019, 0x127A, 0x1A75, 0x091A, 0x1165, 0x1456, 0x1467, 0x1478,
    0x161B, 0x1B13, 0x18B3, 0x087B, 0x1265, 0x1809, 0x1894, 0x1902, 0x1847, 0x1925, 0x1756, 0x0745,
    0x1946, 0x1312, 0x1013, 0x1621, 0x1803, 0x1961, 0x1498, 0x0908, 0x1945, 0x115A, 0x1084, 0x1904,
    0x1B80, 0x11B0, 0x1AB1, 0x0195, 0x16A5, 0x1195, 0x1A15, 0x1891, 0x1281, 0x1B82, 0x1B26, 0x02A6,
    0x16A5, 0x1764, 0x1546, 0x1374, 0x1934, 0x1139, 0x119A, 0x095A, 0x12A6, 0x1B26, 0x19A2, 0x17B6,
    0x1392, 0x1893, 0x1837, 0x03B7, 0x16A5, 0x1B2A, 0x16BA, 0x102B, 0x1754, 0x170B, 0x1407, 0x0765,
    0x1B25, 0x178B, 0x13B8, 0x157B, 0x1038, 0x1152, 0x1120, 0x0230, 0x194A, 0x1490, 0x1840, 0x1380,
    0x17A4, 0x1BA7, 0x1B73, 0x0783, 0x1BA6, 0x1130, 0x1231, 0x1403, 0x1A21, 0x1B43, 0x164B, 0x0B2A,
    0x1A95, 0x119A, 0x1759, 0x121A, 0x1079, 0x1370, 0x1302, 0x0012, 0x1465, 0x13B8, 0x178B, 0x1138,
    0x167B, 0x1418, 0x1514, 0x0476, 0x1945, 0x1765, 0x1754, 0x1267, 0x1827, 0x1028, 0x1089, 0x0849,
    // ---- Case 12.2 (offset 1646) -------------------------------
    0x12C6, 0x1C19, 0x11C0, 0x13C2, 0x180C, 0x18C3, 0x194C, 0x046C, 0x1AC5, 0x119C, 0x14C9, 0x145C,
    0x1ABC, 0x10C8, 0x1B8C, 0x01C0, 0x1CA5, 0x156C, 0x12AC, 0x16BC, 0x1B8C, 0x11C9, 0x189C, 0x02C1,
    0x1CA5, 0x1AC6, 0x14C5, 0x16C7, 0x137C, 0x11C9, 0x113C, 0x09C4, 0x12CA, 0x1CB6, 0x13BC, 0x178C,
    0x167C, 0x189C, 0x19AC, 0x03C2, 0x1CA5, 0x154C, 0x176C, 0x1AC6, 0x140C, 0x1BC2, 0x102C, 0x07CB,
    0x1C15, 0x123C, 0x101C, 0x18C3, 0x180C, 0x1BC7, 0x157C, 0x02CB, 0x1CBA, 0x1C38, 0x17C4, 0x1BC7,
    0x1C84, 0x1C90, 0x1CA9, 0x03C0, 0x16CA, 0x11C2, 0x1C03, 0x12CB, 0x1C3B, 0x1C40, 0x1C64, 0x0AC1,
    0x1AC5, 0x1C19, 0x11C2, 0x13C0, 0x10C9, 0x1C37, 0x1C75, 0x02CA, 0x1C45, 0x17C6, 0x178C, 0x1BC3,
    0x16CB, 0x113C, 0x151C, 0x04C8, 0x1C45, 0x1C26, 0x184C, 0x190C, 0x159C, 0x102C, 0x17C6, 0x08C7,
    0x146C, 0x190C, 0x184C, 0x13C0, 0x138C, 0x11C2, 0x162C, 0x09C1, 0x1C45, 0x10C9, 0x14C8, 0x159C,
    0x1B8C, 0x11AC, 0x1ABC, 0x01C0, 0x16C5, 0x16AC, 0x15C9, 0x11CA, 0x189C, 0x12BC, 0x1B8C, 0x02C1,
    0x16C5, 0x16AC, 0x195C, 0x1A1C, 0x113C, 0x1C74, 0x137C, 0x09C4, 0x16CA, 0x12CB, 0x17BC, 0x17C6,
    0x19AC, 0x138C, 0x189C, 0x03C2, 0x16C5, 0x15CA, 0x1BC6, 0x1AC2, 0x102C, 0x174C, 0x140C, 0x07CB,
    0x17C5, 0x1C3B, 0x18C7, 0x103C, 0x10C8, 0x121C, 0x115C, 0x02CB, 0x19CA, 0x1C80, 0x1C94, 0x18C7,
    0x1C47, 0x1BC3, 0x1CBA, 0x03C0, 0x12CA, 0x1CB6, 0x1C23, 0x1BC3, 0x1C64, 0x1C01, 0x1C40, 0x0AC1,
    0x19C5, 0x1C1A, 0x11C0, 0x1C90, 0x1C75, 0x13C2, 0x1C37, 0x02CA, 0x1C65, 0x17C4, 0x1BC7, 0x1B6C,
    0x151C, 0x18C3, 0x113C, 0x04C8, 0x1C65, 0x17C4, 0x194C, 0x19C5, 0x126C, 0x180C, 0x102C, 0x08C7,
    // ---- Case 13.1 (offset 1838) -------------------------------
    0x1945, 0x121A, 0x17B6, 0x0380, 0x1A65, 0x1190, 0x1B23, 0x0784,
    // ---- Case 13.2 (offset 1846) -------------------------------
    0x1B65, 0x1B59, 0x121A, 0x1380, 0x1794, 0x0B97, 0x1945, 0x17B6, 0x181A, 0x1801, 0x1A38, 0x0A23,
    0x1945, 0x121A, 0x10B6, 0x103B, 0x1680, 0x0678, 0x1945, 0x11A6, 0x1038, 0x1172, 0x17B2, 0x0167,
    0x1A45, 0x17B6, 0x1380, 0x1429, 0x1219, 0x04A2, 0x1A65, 0x1794, 0x1197, 0x13B2, 0x1817, 0x0801,
    0x1905, 0x121A, 0x1345, 0x17B6, 0x1350, 0x0384, 0x1065, 0x1590, 0x11A6, 0x1784, 0x1B23, 0x0016,
    0x1B65, 0x135A, 0x1784, 0x1019, 0x1B53, 0x0A23, 0x1A65, 0x1190, 0x1342, 0x1384, 0x1472, 0x07B2,
    0x1A65, 0x1784, 0x10B9, 0x103B, 0x1B29, 0x0219, 0x1A45, 0x18A6, 0x1190, 0x13B2, 0x14A8, 0x0678,
    // ---- Case 13.3 (offset 1918) -------------------------------
    0x1C65, 0x1C59, 0x121A, 0x14C9, 0x10C8, 0x1C78, 0x1C47, 0x1C3B, 0x16CB, 0x0C03, 0x1945, 0x1CA6,
    0x13C0, 0x11C2, 0x1CB2, 0x1C3B, 0x1C80, 0x1C78, 0x17C6, 0x0C1A, 0x1C65, 0x1CA6, 0x19C5, 0x11AC,
    0x1C21, 0x1CB2, 0x17C4, 0x1BC7, 0x1C94, 0x0803, 0x1945, 0x12CA, 0x1CB6, 0x1C3B, 0x1C23, 0x1C1A,
    0x1C01, 0x1C78, 0x10C8, 0x0C67, 0x1C65, 0x1A6C, 0x17C4, 0x159C, 0x194C, 0x178C, 0x180C, 0x11AC,
    0x11C0, 0x03B2, 0x1945, 0x1CA6, 0x17BC, 0x18C3, 0x1C23, 0x1CB2, 0x1C67, 0x1C01, 0x1AC1, 0x0C80,
    0x1C65, 0x1C5A, 0x1CB6, 0x12CA, 0x17C4, 0x1C7B, 0x1C19, 0x14C9, 0x1C21, 0x0038, 0x1AC5, 0x1CA6,
    0x1C45, 0x17C6, 0x1C94, 0x1C19, 0x1CB2, 0x11C2, 0x1C7B, 0x0380, 0x1A65, 0x1C3B, 0x17C4, 0x18C7,
    0x180C, 0x103C, 0x1B2C, 0x1C19, 0x121C, 0x094C, 0x1AC5, 0x1C45, 0x17B6, 0x10C8, 0x14C9, 0x1C19,
    0x1C01, 0x1C38, 0x1C23, 0x02CA, 0x1A65, 0x119C, 0x11C0, 0x13C2, 0x138C, 0x180C, 0x194C, 0x17BC,
    0x17C4, 0x0B2C, 0x1AC5, 0x1A6C, 0x1C19, 0x194C, 0x145C, 0x167C, 0x180C, 0x18C7, 0x101C, 0x0B23,
    0x1C65, 0x16CA, 0x12CB, 0x159C, 0x121C, 0x11AC, 0x103C, 0x10C9, 0x13BC, 0x0784, 0x1C45, 0x17C6,
    0x1C59, 0x121A, 0x1C84, 0x1C78, 0x1CB6, 0x1C3B, 0x1C90, 0x03C0, 0x1C65, 0x19C5, 0x121A, 0x1C38,
    0x17C4, 0x1BC7, 0x1C84, 0x1C03, 0x1C90, 0x0CB6, 0x1C45, 0x16CA, 0x10C9, 0x14C8, 0x159C, 0x101C,
    0x11AC, 0x167C, 0x178C, 0x023B, 0x1C65, 0x12CA, 0x13C2, 0x159C, 0x11C0, 0x11AC, 0x13BC, 0x1B6C,
    0x190C, 0x0784, 0x19C5, 0x12CA, 0x1C45, 0x17B6, 0x1AC1, 0x1C01, 0x1C90, 0x1C84, 0x1C23, 0x08C3,
    0x1A65, 0x14C8, 0x10C9, 0x103C, 0x138C, 0x147C, 0x17BC, 0x121C, 0x12CB, 0x019C, 0x1AC5, 0x15C4,
    0x17B6, 0x1C19, 0x11C2, 0x13C0, 0x1C90, 0x1CA2, 0x1C84, 0x0C38, 0x1AC5, 0x1B6C, 0x1C19, 0x1A2C,
    0x121C, 0x190C, 0x103C, 0x1BC3, 0x165C, 0x0784, 0x1AC5, 0x16CA, 0x12CB, 0x145C, 0x167C, 0x17BC,
    0x123C, 0x138C, 0x14C8, 0x0019, 0x1C65, 0x15AC, 0x17C4, 0x17BC, 0x1B6C, 0x1A2C, 0x138C, 0x13C2,
    0x184C, 0x0190, 0x1AC5, 0x1B6C, 0x145C, 0x1C78, 0x1BC3, 0x167C, 0x184C, 0x1A2C, 0x123C, 0x0019,
    // ---- Case 13.4 (offset 2158) -------------------------------
    0x1C65, 0x1C5A, 0x1C90, 0x1C03, 0x1C38, 0x1C84, 0x1C47, 0x1C7B, 0x1CB6, 0x1CA2, 0x1C21, 0x0C19,
    0x1AC5, 0x1A6C, 0x138C, 0x123C, 0x1B2C, 0x145C, 0x17BC, 0x167C, 0x194C, 0x119C, 0x101C, 0x080C,
    0x1C65, 0x1CA6, 0x1CB2, 0x1C59, 0x1C21, 0x1C1A, 0x1C94, 0x1C47, 0x1C78, 0x1C80, 0x1C03, 0x0C3B,
    0x1C45, 0x1B6C, 0x159C, 0x11AC, 0x101C, 0x190C, 0x184C, 0x178C, 0x167C, 0x13BC, 0x123C, 0x0A2C,
    // ---- Case 13.5.2 (offset 2206) -----------------------------
    0x1A65, 0x1784, 0x1B87, 0x18B3, 0x1804, 0x1094, 0x1190, 0x1310, 0x1213, 0x0B23, 0x1945, 0x115A,
    0x17B6, 0x1380, 0x1302, 0x1012, 0x1908, 0x1498, 0x1951, 0x0A21, 0x1A65, 0x19A5, 0x1A91, 0x1A26,
    0x12B6, 0x13B2, 0x1132, 0x1031, 0x1901, 0x0784, 0x1945, 0x1BA6, 0x1380, 0x1837, 0x13B7, 0x1230,
    0x1120, 0x121A, 0x12AB, 0x067B, 0x1465, 0x1BA6, 0x1519, 0x121A, 0x12AB, 0x15A1, 0x1594, 0x1476,
    0x17B6, 0x0803, 0x1A65, 0x13B2, 0x18B3, 0x1574, 0x1B87, 0x1B62, 0x16A2, 0x1756, 0x1847, 0x0190,
    0x1465, 0x1459, 0x121A, 0x1380, 0x1089, 0x1849, 0x1783, 0x1B73, 0x17B6, 0x0764, 0x1A65, 0x1190,
    0x1A91, 0x19A5, 0x1940, 0x1480, 0x1784, 0x1574, 0x1675, 0x03B2,
    // ---- Case 13.5.1 (offset 2286) -----------------------------
    0x1A65, 0x1380, 0x1219, 0x1942, 0x14B2, 0x07B4, 0x1A35, 0x1584, 0x17B6, 0x1190, 0x123A, 0x0385,
    0x1965, 0x121A, 0x1B03, 0x1B60, 0x1690, 0x0784, 0x1945, 0x17A6, 0x13B2, 0x1018, 0x1178, 0x01A7
};


template<typename GridType>
int
MarchingCubes<GridType>::faceTests(int face[6], int ind, const double v[8])
{
    // Asymptotic decider (Nielson & Hamann) per face. For each ambiguous face the
    // bilinear saddle joins either the two positive or the two negative corners;
    // the product comparison decides which. ind (possibly complemented) tells which
    // corner pattern is present so only genuinely ambiguous faces are tested.
    if (ind & 0x80) { // corner 0 present
        face[0] = ((ind & 0xCC) == 0x84) ? (v[0]*v[5] < v[1]*v[4] ? -1 : 1) : 0;
        face[3] = ((ind & 0x99) == 0x81) ? (v[0]*v[7] < v[3]*v[4] ? -1 : 1) : 0;
        face[4] = ((ind & 0xF0) == 0xA0) ? (v[0]*v[2] < v[1]*v[3] ? -1 : 1) : 0;
    } else {
        face[0] = ((ind & 0xCC) == 0x48) ? (v[0]*v[5] < v[1]*v[4] ? 1 : -1) : 0;
        face[3] = ((ind & 0x99) == 0x18) ? (v[0]*v[7] < v[3]*v[4] ? 1 : -1) : 0;
        face[4] = ((ind & 0xF0) == 0x50) ? (v[0]*v[2] < v[1]*v[3] ? 1 : -1) : 0;
    }
    if (ind & 0x02) { // corner 6 present
        face[1] = ((ind & 0x66) == 0x42) ? (v[1]*v[6] < v[2]*v[5] ? -1 : 1) : 0;
        face[2] = ((ind & 0x33) == 0x12) ? (v[3]*v[6] < v[2]*v[7] ? -1 : 1) : 0;
        face[5] = ((ind & 0x0F) == 0x0A) ? (v[4]*v[6] < v[5]*v[7] ? -1 : 1) : 0;
    } else {
        face[1] = ((ind & 0x66) == 0x24) ? (v[1]*v[6] < v[2]*v[5] ? 1 : -1) : 0;
        face[2] = ((ind & 0x33) == 0x21) ? (v[3]*v[6] < v[2]*v[7] ? 1 : -1) : 0;
        face[5] = ((ind & 0x0F) == 0x05) ? (v[4]*v[6] < v[5]*v[7] ? 1 : -1) : 0;
    }
    return face[0] + face[1] + face[2] + face[3] + face[4] + face[5];
}


template<typename GridType>
int
MarchingCubes<GridType>::faceTest1(int face, const double v[8])
{
    switch (face) {
        case 0:  return v[0]*v[5] < v[1]*v[4] ? 0x48 : 0x84;
        case 1:  return v[1]*v[6] < v[2]*v[5] ? 0x24 : 0x42;
        case 2:  return v[3]*v[6] < v[2]*v[7] ? 0x21 : 0x12;
        case 3:  return v[0]*v[7] < v[3]*v[4] ? 0x18 : 0x81;
        case 4:  return v[0]*v[2] < v[1]*v[3] ? 0x50 : 0xA0;
        default: return v[4]*v[6] < v[5]*v[7] ? 0x05 : 0x0A;
    }
}


template<typename GridType>
int
MarchingCubes<GridType>::interiorTest(int i, int flag13, const double v[8])
{
    // Corrected interior test (Vega, Abache & Coll 2019). The bilinear interpolant
    // along the interior diagonal is a quadratic in t; its extremum t = -b/2a is
    // where the two sheets are closest. If that point lies inside the cube and both
    // sheets have the same sign there, the corners are joined (a tunnel exists).
    double At = v[4]-v[0], Bt = v[5]-v[1], Ct = v[6]-v[2], Dt = v[7]-v[3];
    double t = At*Ct - Bt*Dt; // the quadratic's leading coefficient "a"
    if (std::signbit(t)) {
        if (i & 0x01) return 0;
    } else {
        if (!(i & 0x01) || t == 0.0) return 0;
    }
    t = 0.5*(v[3]*Bt - v[2]*At + v[1]*Dt - v[0]*Ct)/t; // t = -b/2a
    if (t > 0.0 && t < 1.0) {
        At = v[0] + At*t;
        Bt = v[1] + Bt*t;
        Ct = v[2] + Ct*t;
        Dt = v[3] + Dt*t;
        Ct *= At;
        Dt *= Bt;
        if (i & 0x01) {
            if (Ct < Dt && !std::signbit(Dt))
                return (std::signbit(Bt) == std::signbit(v[i])) + flag13;
        } else {
            if (Ct > Dt && !std::signbit(Ct))
                return (std::signbit(At) == std::signbit(v[i])) + flag13;
        }
    }
    return 0;
}


template<typename GridType>
Index32
MarchingCubes<GridType>::weldVertex(LocalMesh& mesh, const EdgeKey& key, const Vec3d& pos)
{
    static constexpr EdgeKeyHash hasher{};

    // Probe the flat weld table.
    std::size_t slot = hasher(key) & mesh.htMask;
    while (true) {
        const Index32 g = mesh.htSlots[slot];
        if (g == LocalMesh::HT_EMPTY) break;   // empty slot — new vertex
        if (mesh.keys[g] == key) return g;       // found existing vertex
        slot = (slot + 1) & mesh.htMask;
    }

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
        slot = hasher(key) & mesh.htMask;
        while (mesh.htSlots[slot] != LocalMesh::HT_EMPTY) slot = (slot + 1) & mesh.htMask;
    }

    mesh.htSlots[slot] = idx;
    mesh.points.emplace_back(static_cast<float>(pos.x()),
                             static_cast<float>(pos.y()),
                             static_cast<float>(pos.z()));
    mesh.keys.push_back(key);
    return idx;
}


template<typename GridType>
Index32
MarchingCubes<GridType>::localCrossing(LocalMesh& mesh, const math::Transform& xform,
    double isovalue, const Coord& ca, ValueType va, const Coord& cb, ValueType vb)
{
    // Canonical edge key so every cell/thread sharing this edge produces an
    // identical position and welds to the same vertex.
    const EdgeKey key = (ca < cb) ? EdgeKey(ca, cb) : EdgeKey(cb, ca);
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
    return weldVertex(mesh, key, xform.indexToWorld(ip));
}


template<typename GridType>
Index32
MarchingCubes<GridType>::centerVertex(LocalMesh& mesh, const math::Transform& xform,
    const Coord& origin)
{
    // Interior "tunnel" vertex at the cell center. Keyed by (origin, origin), which
    // cannot collide with any real edge (whose endpoints are always distinct).
    const EdgeKey key(origin, origin);
    const Vec3d ip(double(origin.x()) + 0.5,
                   double(origin.y()) + 0.5,
                   double(origin.z()) + 0.5);
    return weldVertex(mesh, key, xform.indexToWorld(ip));
}


template<typename GridType>
void
MarchingCubes<GridType>::marchCell(LocalMesh& mesh, const math::Transform& xform,
    double isovalue, const Coord& origin, const Coord corner[8], const ValueType value[8])
{
    // Sign-inverted corner values (v = isovalue - value) so std::signbit(v[k]) is
    // set exactly when the corner is above the isovalue, matching the face/interior
    // tests and the table's derivation.
    double v[8];
    unsigned int idx = 0;
    for (int c = 0; c < 8; ++c) {
        v[c] = isovalue - double(value[c]);
        if (double(value[c]) > isovalue) idx |= (1u << (7 - c)); // corner c "positive"
    }
    if (idx == 0 || idx == 0xFF) return; // no crossing

    // Read the configuration descriptor. Configurations with corner 0 positive
    // reuse the complement entry (i^0xFF) with reversed winding.
    const unsigned short* pcase = sMC33Table;
    unsigned int c, m, n;
    if (idx & 0x80) { c = sMC33Table[idx ^ 0xFF]; m = (c & 0x800) == 0; n = !m; }
    else            { c = sMC33Table[idx];        n = (c & 0x800) == 0; m = !n; }
    unsigned int k = c & 0x7FF;
    int face[6];

    // Select the triangle pattern for this configuration, resolving face and
    // interior ambiguities per the MC33 case analysis. Offsets index sMC33Table;
    // the emit loop below pre-increments so each points one before its pattern.
    switch (c >> 12) {
        case 0: // cases 1, 2, 5, 8, 9, 11, 14 — unambiguous
            pcase += k;
            break;
        case 1: // case 3
            pcase += ((m ? idx : idx ^ 0xFF) & faceTest1(k >> 2, v))
                     ? 183 + (k << 1) : 159 + k;
            break;
        case 2: // case 4
            pcase += interiorTest(k, 0, v) ? 239 + 6*k : 231 + (k << 1);
            break;
        case 3: // case 6
            if ((m ? idx : idx ^ 0xFF) & faceTest1(k % 6, v))
                pcase += 575 + 5*k;                                    // 6.2
            else
                pcase += interiorTest(k / 6, 0, v) ? 407 + 7*k : 335 + 3*k; // 6.1
            break;
        case 4: // case 7
            switch (faceTests(face, (m ? idx : idx ^ 0xFF), v)) {
                case -3: pcase += 695 + 3*k; break;                    // 7.1
                case -1:                                               // 7.2
                    pcase += (face[4] + face[5] < 0
                             ? (face[0] + face[2] < 0 ? 759 : 799) : 719) + 5*k;
                    break;
                case 1:                                                // 7.3
                    pcase += (face[4] + face[5] < 0
                             ? 983 : (face[0] + face[2] < 0 ? 839 : 911)) + 9*k;
                    break;
                default:                                               // 7.4
                    pcase += interiorTest(k >> 1, 0, v) ? 1095 + 9*k : 1055 + 5*k;
            }
            break;
        case 5: // case 10
            switch (faceTests(face, (m ? idx : idx ^ 0xFF), v)) {
                case -2:
                    if (k == 2 ? interiorTest(0, 0, v)
                               : (interiorTest(0, 0, v) || interiorTest(k ? 1 : 3, 0, v)))
                        pcase += 1213 + (k << 3);                      // 10.1.2
                    else
                        pcase += 1189 + (k << 2);                      // 10.1.1
                    break;
                case 0:                                                // 10.2
                    pcase += (face[2 + k] < 0 ? 1261 : 1285) + (k << 3);
                    break;
                default:
                    if (k == 2 ? interiorTest(1, 0, v)
                               : (interiorTest(2, 0, v) || interiorTest(k ? 3 : 1, 0, v)))
                        pcase += 1237 + (k << 3);                      // 10.1.2
                    else
                        pcase += 1201 + (k << 2);                      // 10.1.1
            }
            break;
        case 6: // case 12
            switch (faceTests(face, (m ? idx : idx ^ 0xFF), v)) {
                case -2:                                               // 12.1
                    pcase += interiorTest((0xDA010C >> (k << 1)) & 3, 0, v)
                             ? 1453 + (k << 3) : 1357 + (k << 2);
                    break;
                case 0:                                                // 12.2
                    pcase += (face[k >> 1] < 0 ? 1645 : 1741) + (k << 3);
                    break;
                default:                                               // 12.1
                    pcase += interiorTest((0xA7B7E5 >> (k << 1)) & 3, 0, v)
                             ? 1549 + (k << 3) : 1405 + (k << 2);
            }
            break;
        default: // case 13
            switch (std::abs(faceTests(face, 165, v))) {
                case 0:
                    k = ((face[1] < 0) << 1) | (face[5] < 0);
                    if (face[0]*face[1] == face[5]) {                  // 13.4
                        pcase += 2157 + 12*k;
                    } else {
                        c = interiorTest(k, 1, v); // 0: 13.5.1, else 13.5.2
                        pcase += 2285 + (c ? 10*int(k) - 40*int(c) : 6*int(k));
                    }
                    break;
                case 2:                                                // 13.3
                    pcase += 1917 + 10*((face[0] < 0 ? face[2] > 0 : 12 + (face[2] < 0))
                                      + (face[1] < 0 ? face[3] < 0 : 6 + (face[3] > 0)));
                    if (face[4] > 0) pcase += 30;
                    break;
                case 4:                                                // 13.2
                    k = 21 + 11*face[0] + 4*face[1] + 3*face[2] + 2*face[3] + face[4];
                    if (k >> 4) k -= (k & 32 ? 20 : 10);
                    pcase += 1845 + 3*int(k);
                    break;
                default:                                               // 13.1
                    pcase += 1839 + 2*face[0];
            }
    }

    // Emit triangles. Each pattern short encodes one triangle in its three low
    // nibbles (edge codes 0-11, or 12 for the cell-center tunnel vertex); the top
    // nibble is nonzero except on the final triangle, so it doubles as the loop
    // guard. Winding (ti[n], ti[m]) with n != m applies the case's reversal.
    Index32 p[13];
    for (int e = 0; e < 13; ++e) p[e] = LocalMesh::HT_EMPTY;
    unsigned int word = idx; // nonzero: enters the loop, then reloaded below
    while (word) {
        word = *(++pcase);
        Index32 ti[3];
        for (int slot = 3; slot; ) {
            const unsigned int e = word & 0x0F;
            word >>= 4;
            if (p[e] == LocalMesh::HT_EMPTY) {
                p[e] = (e == 12) ? centerVertex(mesh, xform, origin)
                                 : localCrossing(mesh, xform, isovalue,
                                       corner[sEdge[e][0]], value[sEdge[e][0]],
                                       corner[sEdge[e][1]], value[sEdge[e][1]]);
            }
            ti[--slot] = p[e];
        }
        if (ti[0] != ti[1] && ti[0] != ti[2] && ti[1] != ti[2]) { // skip degenerate
            // Wound so normals point toward increasing value (outward for a level
            // set stored negative-inside): swap of (n,m) relative to the table's
            // native winding.
            mesh.triangles.emplace_back(ti[m], ti[n], ti[2]);
        }
    }
}


template<typename GridType>
MaskGrid::Ptr
MarchingCubes<GridType>::gatherCells(std::vector<const MaskLeaf*>& masterLeaves) const
{
    using LeafNode = typename GridType::TreeType::LeafNodeType;

    util::CpuTimer gt;

    gt.start();
    std::vector<const LeafNode*> leaves;
    leaves.reserve(256);
    for (auto it = mGrid.tree().cbeginLeaf(); it; ++it) leaves.push_back(&*it);
    const double msLeafCollect = gt.milliseconds();

    tbb::enumerable_thread_specific<MaskGrid::Ptr> pool(
        [] { return MaskGrid::create(); });

    gt.start();
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
    const double msMark = gt.milliseconds();

    gt.start();
    MaskGrid::Ptr master = MaskGrid::create();
    for (const MaskGrid::Ptr& local : pool)
        master->tree().topologyUnion(local->tree());
    const double msUnion = gt.milliseconds();

    gt.start();
    masterLeaves.clear();
    masterLeaves.reserve(master->tree().leafCount());
    for (auto it = master->tree().cbeginLeaf(); it; ++it)
        masterLeaves.push_back(&*it);
    const double msMasterCollect = gt.milliseconds();

    std::cerr << "[gatherCells profile]"
              << "  inputLeaves=" << leaves.size()
              << "  masterLeaves=" << masterLeaves.size() << "\n"
              << "  leafCollect(serial)  " << msLeafCollect   << " ms\n"
              << "  mark(parallel)       " << msMark          << " ms\n"
              << "  topologyUnion(serial)" << msUnion         << " ms\n"
              << "  masterCollect(serial)" << msMasterCollect << " ms\n"
              << "  total                " << (msLeafCollect+msMark+msUnion+msMasterCollect) << " ms\n";
    return master;
}


template<typename GridType>
void
MarchingCubes<GridType>::extract(const std::vector<const MaskLeaf*>& masterLeaves,
                                  double isovalue)
{
    using AccessorT = typename GridType::ConstAccessor;

    const math::Transform& xform = mGrid.transform();
    util::CpuTimer stepTimer;

    // ---- March candidate cells in parallel ---------------------------------
    // Iterate master leaves directly — each leaf's active voxels are cell origins.
    // This avoids materialising a flat Coord vector (2.5 GB at crawler scale).
    const std::size_t nMasterLeaves = masterLeaves.size();
    const std::size_t nThreads = std::max(std::size_t(1),
        static_cast<std::size_t>(tbb::this_task_arena::max_concurrency()));
    const std::size_t vertsPerThread = nMasterLeaves * 256 / nThreads + 1;
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
    stepTimer.start();
    tbb::parallel_for(tbb::blocked_range<size_t>(0, nMasterLeaves),
        [&](const tbb::blocked_range<size_t>& range)
    {
        LocalMesh& mesh = pool.local();
        AccessorT& acc  = accPool.local();
        Coord     corner[8];
        ValueType value[8];
        for (size_t li = range.begin(); li != range.end(); ++li) {
            for (auto vit = masterLeaves[li]->cbeginValueOn(); vit; ++vit) {
                const Coord origin = vit.getCoord();
                for (int c = 0; c < 8; ++c) {
                    corner[c] = origin + cornerOffset(c);
                    value[c]  = acc.getValue(corner[c]);
                }
                marchCell(mesh, xform, isovalue, origin, corner, value);
            }
        }
    });
    const double msMarch = stepTimer.milliseconds();

    // ---- Merge thread-local fragments (parallel shard merge) ----------------
    stepTimer.start();
    std::vector<const LocalMesh*> frags;
    for (const LocalMesh& m : pool) frags.push_back(&m);
    const std::size_t nFrags = frags.size();

    std::vector<uint32_t> fragVertOff(nFrags + 1, 0);
    std::size_t totalTris = 0;
    for (std::size_t f = 0; f < nFrags; ++f) {
        fragVertOff[f + 1] = fragVertOff[f] +
                              static_cast<uint32_t>(frags[f]->points.size());
        totalTris += frags[f]->triangles.size();
    }
    const uint32_t totalFragVerts = fragVertOff[nFrags];

    std::size_t numShards = 1;
    while (numShards < nFrags) numShards <<= 1;
    const std::size_t shardMask = numShards - 1;
    const EdgeKeyHash hasher;
    const double msFragSetup = stepTimer.milliseconds();

    // 1. Count per (fragment, shard).
    stepTimer.start();
    std::vector<uint32_t> fragShardCount(nFrags * numShards, 0);
    tbb::parallel_for(std::size_t(0), nFrags, [&](std::size_t f) {
        uint32_t* cnt = fragShardCount.data() + f * numShards;
        for (const EdgeKey& key : frags[f]->keys)
            ++cnt[hasher(key) & shardMask];
    });
    const double msCount = stepTimer.milliseconds();

    // 2. Prefix sum -> per-(fragment,shard) offsets into binnedIdx.
    stepTimer.start();
    std::vector<uint32_t> fragShardOff(nFrags * numShards + 1, 0);
    for (std::size_t i = 0; i < nFrags * numShards; ++i)
        fragShardOff[i + 1] = fragShardOff[i] + fragShardCount[i];
    const uint32_t totalBinned = fragShardOff[nFrags * numShards];
    const double msPrefixSum = stepTimer.milliseconds();

    std::vector<uint32_t> binnedIdx(totalBinned);

    // 3. Fill: scatter each fragment's local vertex indices by shard.
    stepTimer.start();
    tbb::parallel_for(std::size_t(0), nFrags, [&](std::size_t f) {
        uint32_t* fill = fragShardCount.data() + f * numShards; // reuse as cursor
        std::fill(fill, fill + numShards, 0u);
        const uint32_t* off = fragShardOff.data() + f * numShards;
        for (uint32_t i = 0, nk = static_cast<uint32_t>(frags[f]->keys.size()); i < nk; ++i) {
            const std::size_t s = hasher(frags[f]->keys[i]) & shardMask;
            binnedIdx[off[s] + fill[s]++] = i;
        }
    });
    const double msScatter = stepTimer.milliseconds();

    // 4. Parallel per-shard dedup.
    stepTimer.start();
    std::vector<Index32> remap(totalFragVerts);
    std::vector<std::vector<Vec3s>> shardPts(numShards);

    tbb::parallel_for(std::size_t(0), numShards, [&](std::size_t s) {
        std::vector<Vec3s>& pts = shardPts[s];

        uint32_t shardTotal = 0;
        for (std::size_t f = 0; f < nFrags; ++f)
            shardTotal += fragShardOff[f * numShards + s + 1]
                        - fragShardOff[f * numShards + s];
        pts.reserve(shardTotal);

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
                while (true) {
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
    const double msDedup = stepTimer.milliseconds();

    // 5. Prefix sum -> global point offsets per shard.
    stepTimer.start();
    std::vector<Index32> shardOffset(numShards + 1, 0);
    for (std::size_t s = 0; s < numShards; ++s)
        shardOffset[s + 1] = shardOffset[s] + static_cast<Index32>(shardPts[s].size());

    // 6. Assemble mPoints and patch remap values to global indices.
    mPoints.resize(shardOffset[numShards]);
    tbb::parallel_for(std::size_t(0), numShards, [&](std::size_t s) {
        const Index32 off = shardOffset[s];
        for (std::size_t i = 0, nn = shardPts[s].size(); i < nn; ++i)
            mPoints[off + i] = shardPts[s][i];
        for (std::size_t f = 0; f < nFrags; ++f) {
            const uint32_t base  = fragShardOff[f * numShards + s];
            const uint32_t count = fragShardOff[f * numShards + s + 1] - base;
            for (uint32_t bi = 0; bi < count; ++bi)
                remap[fragVertOff[f] + binnedIdx[base + bi]] += off;
        }
    });
    const double msAssemble = stepTimer.milliseconds();

    // 7. Remap triangle indices (parallel over fragments).
    stepTimer.start();
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
    const double msRemap = stepTimer.milliseconds();

    const double msTotal = msMarch + msFragSetup + msCount + msPrefixSum
                         + msScatter + msDedup + msAssemble + msRemap;
    std::cerr << "[MarchingCubes extract profile]  nFrags=" << nFrags
              << "  numShards=" << numShards << "\n"
              << "  march(parallel)  " << msMarch     << " ms\n"
              << "  fragSetup        " << msFragSetup  << " ms\n"
              << "  count(parallel)  " << msCount      << " ms\n"
              << "  prefixSum        " << msPrefixSum  << " ms\n"
              << "  scatter(parallel)" << msScatter    << " ms\n"
              << "  dedup(parallel)  " << msDedup      << " ms\n"
              << "  assemble+patch   " << msAssemble   << " ms\n"
              << "  triRemap(parallel)" << msRemap     << " ms\n"
              << "  total            " << msTotal      << " ms\n";
}


template<typename GridType>
void
MarchingCubes<GridType>::operator()(double isovalue)
{
    mPoints.clear();
    mTriangles.clear();

    util::CpuTimer t;
    std::vector<const MaskLeaf*> masterLeaves;
    t.start();
    MaskGrid::Ptr master = this->gatherCells(masterLeaves); // keep alive for extract
    const double msGather = t.milliseconds();
    if (masterLeaves.empty()) return;
    t.start();
    this->extract(masterLeaves, isovalue);
    const double msExtract = t.milliseconds();
    std::cerr << "[MarchingCubes profile]"
              << "  gatherCells=" << msGather << " ms"
              << "  extract="     << msExtract << " ms"
              << "  masterLeaves=" << masterLeaves.size()
              << "  pts="         << mPoints.size()
              << "  tris="        << mTriangles.size() << "\n";
}


////////////////////////////////////////


template<typename GridType>
void
marchingCubes(
    const GridType& grid,
    std::vector<Vec3s>& points,
    std::vector<Vec3I>& triangles,
    double isovalue)
{
    MarchingCubes<GridType> mesher(grid);
    mesher(isovalue);
    points.swap(mesher.points());
    triangles.swap(mesher.triangles());
}


} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_MARCHING_CUBES_HAS_BEEN_INCLUDED
