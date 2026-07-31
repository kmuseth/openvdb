// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file OpenFileToNanoGrid.h
///
/// @brief Reads a named grid directly from an OpenVDB .vdb file into a
///        NanoVDB GridHandle<BufferT> without first building a fully loaded
///        openvdb::Grid in memory.
///
/// The name distinguishes this from CreateNanoGrid.h, which converts an
/// already-loaded openvdb::Grid.  Here the source is a file on disk, and
/// the implementation avoids materialising the full OpenVDB grid before
/// building the NanoVDB buffer.
///
/// Strategy
/// --------
///  Pass 1  Open the file with delayed-load enabled.  Only tree topology
///          (internal-node masks + coordinates, leaf node masks) is read;
///          leaf VALUE buffers remain on disk.  Walk the topology to:
///            - count nodes at every level
///            - assign each node a stable index within its level
///
///  Alloc   Compute the NanoVDB buffer size from those counts (same formula
///          as CreateNanoGrid), allocate, and zero-initialise it.
///
///  Pass 2  Walk the topology again, bottom-up:
///            leaves   – copy masks + values (each leaf's value buffer is
///                       loaded from disk on demand by OpenVDB's delayed-load
///                       mechanism, and immediately copied into NanoVDB)
///            lower    – copy masks; set child offsets from leaf indices;
///                       propagate min/max stats (pure topology, no new I/O)
///            upper    – same for lower children
///            root     – write tiles and child references
///            tree     – set per-level node-count & offset fields
///            grid     – write transform, name, class, checksum header
///          Then calls updateGridStats(StatsMode::All) to tighten bboxes,
///          fill accurate tile counts, and compute avg/stddev.
///
/// Peak memory
/// -----------
///  OpenVDB topology  +  NanoVDB buffer  +  incremental leaf value loads
///
///  This contrasts with the traditional path:
///    full openvdb::Grid  +  full NanoVDB buffer  (both simultaneously).

#ifndef NANOVDB_TOOLS_OPEN_FILE_TO_NANO_GRID_H_HAS_BEEN_INCLUDED
#define NANOVDB_TOOLS_OPEN_FILE_TO_NANO_GRID_H_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h>
#include <nanovdb/GridHandle.h>
#include <nanovdb/HostBuffer.h>
#include <nanovdb/tools/CreateNanoGrid.h> // toGridType, toGridClass
#include <nanovdb/tools/GridStats.h>      // updateGridStats

#include <openvdb/openvdb.h>
#include <openvdb/io/File.h>
#include <openvdb/math/Maps.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace nanovdb {
namespace tools {
namespace detail {

// ============================================================================
//  OpenFileToNanoGridImpl<SrcGridT, DstBuildT, BufferT>
// ============================================================================

template<typename SrcGridT,
         typename DstBuildT,
         typename BufferT>
class OpenFileToNanoGridImpl
{
public:
    // ── Source (OpenVDB) types ──────────────────────────────────────────────
    using SrcValueT = typename SrcGridT::ValueType;
    using SrcTreeT  = typename SrcGridT::TreeType;
    using SrcRootT  = typename SrcTreeT::RootNodeType;
    using SrcUpperT = typename SrcRootT::ChildNodeType;   // InternalNode LOG2DIM=5
    using SrcLowerT = typename SrcUpperT::ChildNodeType;  // InternalNode LOG2DIM=4
    using SrcLeafT  = typename SrcLowerT::ChildNodeType;  // LeafNode     LOG2DIM=3

    // ── Destination (NanoVDB) types ─────────────────────────────────────────
    using DstValueT     = DstBuildT;
    using DstGridT      = NanoGrid<DstBuildT>;
    using DstTreeT      = NanoTree<DstBuildT>;
    using DstRootT      = NanoRoot<DstBuildT>;
    using DstUpperT     = NanoUpper<DstBuildT>;
    using DstLowerT     = NanoLower<DstBuildT>;
    using DstLeafT      = NanoLeaf<DstBuildT>;
    using DstGridDataT  = typename DstGridT::DataType;  // GridData
    using DstTreeDataT  = typename DstTreeT::DataType;  // TreeData
    using DstRootDataT  = typename DstRootT::DataType;  // RootData<NanoUpper<T>>
    using DstUpperDataT = typename DstUpperT::DataType; // InternalData<NanoLower<T>,5>
    using DstLowerDataT = typename DstLowerT::DataType; // InternalData<NanoLeaf<T>,4>
    using DstLeafDataT  = typename DstLeafT::DataType;  // LeafData<T,...>

    // ── Entry point ─────────────────────────────────────────────────────────
    GridHandle<BufferT> read(const std::string& filePath,
                             const std::string& gridName,
                             const BufferT&     pool)
    {
        // ── Pass 1: open with delayed-load, read topology only ──────────────
        openvdb::io::File file(filePath);
        file.open(/*delayLoad=*/true);

        openvdb::GridBase::Ptr baseGrid;
        if (gridName.empty()) {
            // Use beginName() to read only the first grid's topology rather
            // than calling getGrids(), which would load every grid in the file.
            auto nameIt = file.beginName();
            if (nameIt == file.endName())
                throw std::runtime_error(
                    "OpenFileToNanoGrid: no grids found in '" + filePath + "'");
            baseGrid = file.readGrid(*nameIt);
        } else {
            try {
                baseGrid = file.readGrid(gridName);
            } catch (const std::exception&) {
                throw std::runtime_error(
                    "OpenFileToNanoGrid: grid '" + gridName + "' not found in '" + filePath + "'");
            }
            if (!baseGrid)
                throw std::runtime_error(
                    "OpenFileToNanoGrid: grid '" + gridName + "' not found in '" + filePath + "'");
        }

        auto srcGrid = openvdb::gridPtrCast<SrcGridT>(baseGrid);
        if (!srcGrid)
            throw std::runtime_error("OpenFileToNanoGrid: grid type mismatch for '"
                + (gridName.empty() ? std::string("(first grid)") : gridName) + "'");

        const SrcTreeT& srcTree = srcGrid->tree();
        const SrcRootT& srcRoot = srcTree.root();

        // BFS traversal: count and index every node (no leaf values loaded)
        collectNodes(srcRoot);

        const uint32_t nUpper    = static_cast<uint32_t>(mUpperNodes.size());
        const uint32_t nLower    = static_cast<uint32_t>(mLowerNodes.size());
        const uint32_t nLeaf     = static_cast<uint32_t>(mLeafNodes.size());
        const uint32_t rootTable = srcRoot.getTableSize();

        // ── Compute NanoVDB buffer layout (mirrors CreateNanoGrid::initHandle)
        mOff.grid  = 0;
        mOff.tree  = mOff.grid  + DstGridT::memUsage();
        mOff.root  = mOff.tree  + DstTreeT::memUsage();
        mOff.upper = mOff.root  + DstRootT::memUsage(rootTable);
        mOff.lower = mOff.upper + DstUpperT::memUsage() * nUpper;
        mOff.leaf  = mOff.lower + DstLowerT::memUsage() * nLower;
        mOff.total = mOff.leaf  + DstLeafT::DataType::memUsage() * nLeaf;

        // Allocate and zero-initialise
        auto buffer = BufferT::create(mOff.total, &pool);
        mBufPtr = static_cast<uint8_t*>(buffer.data());
        std::memset(mBufPtr, 0, mOff.total);

        // ── Pass 2: fill nodes bottom-up ────────────────────────────────────
        // Leaves trigger per-leaf delayed-loads as values are accessed
        for (uint32_t i = 0; i < nLeaf;  ++i) processLeaf (*mLeafNodes[i],  i);
        for (uint32_t i = 0; i < nLower; ++i) processLower(*mLowerNodes[i], i);
        for (uint32_t i = 0; i < nUpper; ++i) processUpper(*mUpperNodes[i], i);
        processRoot(srcRoot);
        processTree(srcTree, nUpper, nLower, nLeaf);
        processGrid(*srcGrid);

        file.close();

        // Recompute tight bboxes (internal nodes were filled with full-node
        // extents) and accurate tile counts (left at 0 above).  StatsMode::All
        // also adds avg/stddev for every node level at marginal extra cost.
        auto handle = GridHandle<BufferT>(std::move(buffer));
        updateGridStats(handle.template grid<DstBuildT>(), StatsMode::All);
        return handle;
    }

private:
    // ── Indexed node arrays (position = NanoVDB array index) ───────────────
    std::vector<const SrcUpperT*> mUpperNodes;
    std::vector<const SrcLowerT*> mLowerNodes;
    std::vector<const SrcLeafT*>  mLeafNodes;
    std::unordered_map<const SrcUpperT*, uint32_t> mUpperIdx;
    std::unordered_map<const SrcLowerT*, uint32_t> mLowerIdx;
    std::unordered_map<const SrcLeafT*,  uint32_t> mLeafIdx;

    uint8_t* mBufPtr = nullptr;

    struct Offsets { uint64_t grid, tree, root, upper, lower, leaf, total; } mOff{};

    // ── Typed buffer accessors ──────────────────────────────────────────────
    DstGridDataT*  dstGrid()            { return util::PtrAdd<DstGridDataT>(mBufPtr, mOff.grid);  }
    DstTreeDataT*  dstTree()            { return util::PtrAdd<DstTreeDataT>(mBufPtr, mOff.tree);  }
    DstRootDataT*  dstRoot()            { return util::PtrAdd<DstRootDataT>(mBufPtr, mOff.root);  }
    DstUpperDataT* dstUpper(uint32_t i) { return util::PtrAdd<DstUpperDataT>(mBufPtr, mOff.upper) + i; }
    DstLowerDataT* dstLower(uint32_t i) { return util::PtrAdd<DstLowerDataT>(mBufPtr, mOff.lower) + i; }
    DstLeafDataT*  dstLeaf (uint32_t i) { return util::PtrAdd<DstLeafDataT> (mBufPtr, mOff.leaf)  + i; }

    // ── BFS collection: stable index assignment per level ──────────────────
    void collectNodes(const SrcRootT& root)
    {
        for (auto it = root.cbeginChildOn(); it; ++it) {
            const SrcUpperT* n = &(*it);
            mUpperIdx[n] = static_cast<uint32_t>(mUpperNodes.size());
            mUpperNodes.push_back(n);
        }
        for (const SrcUpperT* up : mUpperNodes)
            for (auto it = up->cbeginChildOn(); it; ++it) {
                const SrcLowerT* n = &(*it);
                mLowerIdx[n] = static_cast<uint32_t>(mLowerNodes.size());
                mLowerNodes.push_back(n);
            }
        for (const SrcLowerT* lo : mLowerNodes)
            for (auto it = lo->cbeginChildOn(); it; ++it) {
                const SrcLeafT* n = &(*it);
                mLeafIdx[n] = static_cast<uint32_t>(mLeafNodes.size());
                mLeafNodes.push_back(n);
            }
    }

    // Copy an OpenVDB NodeMask into a NanoVDB Mask.
    // Both store bits in contiguous uint64_t words with identical per-bit
    // semantics, so a plain memcpy is correct and portable.
    template<uint32_t LOG2DIM>
    static void copyMask(const openvdb::util::NodeMask<LOG2DIM>& src,
                         nanovdb::Mask<LOG2DIM>&                  dst)
    {
        static_assert(sizeof(src) == sizeof(dst), "OpenVDB/NanoVDB mask size mismatch");
        std::memcpy(dst.words(), &src, sizeof(dst));
    }

    // ── Leaf fill: accessing getValue() triggers the delayed-load ───────────
    void processLeaf(const SrcLeafT& src, uint32_t idx)
    {
        DstLeafDataT* dst = dstLeaf(idx);

        const openvdb::Coord o = src.origin();
        dst->mBBoxMin = Coord(o.x(), o.y(), o.z());
        constexpr uint8_t kMaxDif = static_cast<uint8_t>((1u << SrcLeafT::LOG2DIM) - 1u); // 7
        dst->mBBoxDif[0] = dst->mBBoxDif[1] = dst->mBBoxDif[2] = kMaxDif;

        // Value mask is part of topology — loads without triggering data I/O
        copyMask(src.valueMask(), dst->mValueMask);

        // Copy voxel values; each call to getValue() on a delayed-load leaf
        // triggers a single leaf buffer read the first time it is called.
        DstValueT vMin =  std::numeric_limits<DstValueT>::max();
        DstValueT vMax = -std::numeric_limits<DstValueT>::max();
        double sum = 0.0, sum2 = 0.0;
        uint32_t n = 0;

        for (uint32_t i = 0; i < SrcLeafT::SIZE; ++i) {
            const DstValueT v = static_cast<DstValueT>(src.getValue(i));
            dst->mValues[i] = v;
            if (dst->mValueMask.isOn(i)) {
                if (v < vMin) vMin = v;
                if (v > vMax) vMax = v;
                sum  += double(v);
                sum2 += double(v) * double(v);
                ++n;
            }
        }

        if (n > 0) {
            dst->mMinimum = vMin;
            dst->mMaximum = vMax;
            const double mean = sum / n;
            dst->mAverage = static_cast<DstValueT>(mean);
            dst->mStdDevi = static_cast<DstValueT>(
                std::sqrt(std::max(0.0, sum2 / n - mean * mean)));
        }
    }

    // ── Lower internal node (LOG2DIM=4, 16^3 subtree) ──────────────────────
    void processLower(const SrcLowerT& src, uint32_t idx)
    {
        DstLowerDataT* dst = dstLower(idx);

        const openvdb::Coord o = src.origin();
        constexpr int kSpan = static_cast<int>((1u << SrcLowerT::TOTAL) - 1u); // 127
        dst->mBBox.min() = Coord(o.x(),          o.y(),          o.z());
        dst->mBBox.max() = Coord(o.x() + kSpan,  o.y() + kSpan,  o.z() + kSpan);

        copyMask(src.getValueMask(), dst->mValueMask);
        copyMask(src.getChildMask(), dst->mChildMask);

        DstValueT vMin =  std::numeric_limits<DstValueT>::max();
        DstValueT vMax = -std::numeric_limits<DstValueT>::max();

        // Value tiles (cbeginValueAll skips child positions)
        for (auto it = src.cbeginValueAll(); it; ++it) {
            const uint32_t  pos = static_cast<uint32_t>(it.pos());
            const DstValueT v   = static_cast<DstValueT>(*it);
            dst->mTable[pos].value = v;
            if (dst->mValueMask.isOn(pos)) {
                if (v < vMin) vMin = v;
                if (v > vMax) vMax = v;
            }
        }

        // Child leaf nodes — setChild requires mChildMask already populated
        for (auto it = src.cbeginChildOn(); it; ++it) {
            const uint32_t  pos      = static_cast<uint32_t>(it.pos());
            const SrcLeafT* srcChild = &(*it);
            const uint32_t  ci       = mLeafIdx.at(srcChild);
            DstLeafDataT*   dstChild = dstLeaf(ci);
            dst->setChild(pos, dstChild);
            if (dstChild->mMinimum < vMin) vMin = dstChild->mMinimum;
            if (dstChild->mMaximum > vMax) vMax = dstChild->mMaximum;
        }

        if (vMin <= vMax) {
            dst->mMinimum = vMin;
            dst->mMaximum = vMax;
        }
    }

    // ── Upper internal node (LOG2DIM=5, 32^3 subtree) ──────────────────────
    void processUpper(const SrcUpperT& src, uint32_t idx)
    {
        DstUpperDataT* dst = dstUpper(idx);

        const openvdb::Coord o = src.origin();
        constexpr int kSpan = static_cast<int>((1u << SrcUpperT::TOTAL) - 1u); // 4095
        dst->mBBox.min() = Coord(o.x(),          o.y(),          o.z());
        dst->mBBox.max() = Coord(o.x() + kSpan,  o.y() + kSpan,  o.z() + kSpan);

        copyMask(src.getValueMask(), dst->mValueMask);
        copyMask(src.getChildMask(), dst->mChildMask);

        DstValueT vMin =  std::numeric_limits<DstValueT>::max();
        DstValueT vMax = -std::numeric_limits<DstValueT>::max();

        // Value tiles
        for (auto it = src.cbeginValueAll(); it; ++it) {
            const uint32_t  pos = static_cast<uint32_t>(it.pos());
            const DstValueT v   = static_cast<DstValueT>(*it);
            dst->mTable[pos].value = v;
            if (dst->mValueMask.isOn(pos)) {
                if (v < vMin) vMin = v;
                if (v > vMax) vMax = v;
            }
        }

        // Child lower nodes
        for (auto it = src.cbeginChildOn(); it; ++it) {
            const uint32_t   pos      = static_cast<uint32_t>(it.pos());
            const SrcLowerT* srcChild = &(*it);
            const uint32_t   ci       = mLowerIdx.at(srcChild);
            DstLowerDataT*   dstChild = dstLower(ci);
            dst->setChild(pos, dstChild);
            if (dstChild->mMinimum < vMin) vMin = dstChild->mMinimum;
            if (dstChild->mMaximum > vMax) vMax = dstChild->mMaximum;
        }

        if (vMin <= vMax) {
            dst->mMinimum = vMin;
            dst->mMaximum = vMax;
        }
    }

    // ── Root node ───────────────────────────────────────────────────────────
    void processRoot(const SrcRootT& srcRoot)
    {
        DstRootDataT* dst    = dstRoot();
        dst->mTableSize      = srcRoot.getTableSize();
        dst->mBackground     = static_cast<DstValueT>(srcRoot.background());

        DstValueT vMin =  std::numeric_limits<DstValueT>::max();
        DstValueT vMax = -std::numeric_limits<DstValueT>::max();

        // Compute root index-space bbox as union of child/tile extents
        bool bboxInit = false;
        Coord bMin, bMax;

        uint32_t tileIdx = 0;
        for (auto it = srcRoot.cbeginChildAll(); it; ++it, ++tileIdx) {
            SrcValueT tileValue{};
            const SrcUpperT* srcChild = it.probeChild(tileValue);
            if (srcChild) {
                const uint32_t   ci        = mUpperIdx.at(srcChild);
                DstUpperDataT*   dstChild  = dstUpper(ci);
                dst->tile(tileIdx)->setChild(srcChild->origin(), dstChild, dst);
                if (dstChild->mMinimum < vMin) vMin = dstChild->mMinimum;
                if (dstChild->mMaximum > vMax) vMax = dstChild->mMaximum;
                const Coord& cMin = dstChild->mBBox.min();
                const Coord& cMax = dstChild->mBBox.max();
                if (!bboxInit) { bMin = cMin; bMax = cMax; bboxInit = true; }
                else {
                    bMin = Coord(std::min(bMin[0], cMin[0]),
                                 std::min(bMin[1], cMin[1]),
                                 std::min(bMin[2], cMin[2]));
                    bMax = Coord(std::max(bMax[0], cMax[0]),
                                 std::max(bMax[1], cMax[1]),
                                 std::max(bMax[2], cMax[2]));
                }
            } else {
                const DstValueT v = static_cast<DstValueT>(tileValue);
                const openvdb::Coord oc = it.getCoord();
                dst->tile(tileIdx)->setValue(
                    Coord(oc.x(), oc.y(), oc.z()), it.isValueOn(), v);
                if (it.isValueOn()) {
                    if (v < vMin) vMin = v;
                    if (v > vMax) vMax = v;
                }
            }
        }

        if (vMin <= vMax) {
            dst->mMinimum = vMin;
            dst->mMaximum = vMax;
        }
        if (bboxInit) {
            dst->mBBox.min() = bMin;
            dst->mBBox.max() = bMax;
        }
    }

    // ── Tree node ───────────────────────────────────────────────────────────
    void processTree(const SrcTreeT& srcTree,
                     uint32_t nUpper, uint32_t nLower, uint32_t nLeaf)
    {
        DstTreeDataT* dst = dstTree();

        // Byte offsets from TreeData to the first node at each level
        // (setFirstNode uses NodeT::LEVEL to pick the right mNodeOffset slot)
        dst->setRoot(dstRoot());
        if (nUpper > 0)
            dst->template setFirstNode<DstUpperT>(
                util::PtrAdd<DstUpperT>(mBufPtr, mOff.upper));
        if (nLower > 0)
            dst->template setFirstNode<DstLowerT>(
                util::PtrAdd<DstLowerT>(mBufPtr, mOff.lower));
        if (nLeaf > 0)
            dst->template setFirstNode<DstLeafT>(
                util::PtrAdd<DstLeafT>(mBufPtr, mOff.leaf));

        dst->mNodeCount[0] = nLeaf;
        dst->mNodeCount[1] = nLower;
        dst->mNodeCount[2] = nUpper;

        // mTileCount is recomputed by updateGridStats after read() returns.
        dst->mTileCount[0] = 0;
        dst->mTileCount[1] = 0;
        dst->mTileCount[2] = 0;

        // Active voxel count from OpenVDB uses only leaf value masks
        // (topology), so it does not trigger further delayed data loads.
        dst->mVoxelCount = srcTree.activeVoxelCount();
    }

    // ── Grid header ─────────────────────────────────────────────────────────
    void processGrid(const SrcGridT& srcGrid)
    {
        // Build NanoVDB Map from the OpenVDB affine transform.
        // getAffineMap() converts any MapBase to an AffineMap; getMat4()
        // returns the 4×4 index-to-world matrix with translation in row 3.
        // Map::set(mat4, invMat4) fills both double- and float-precision
        // copies and extracts the translation from mat4[3].
        const auto mat4 = srcGrid.transform().baseMap()->getAffineMap()->getMat4();
        Map nanoMap;
        nanoMap.set(mat4, mat4.inverse());

        // Map OpenVDB grid class to NanoVDB
        GridClass gridClass;
        switch (srcGrid.getGridClass()) {
        case openvdb::GRID_LEVEL_SET:  gridClass = GridClass::LevelSet;  break;
        case openvdb::GRID_FOG_VOLUME: gridClass = GridClass::FogVolume; break;
        case openvdb::GRID_STAGGERED:  gridClass = GridClass::Staggered; break;
        default:                        gridClass = GridClass::Unknown;   break;
        }

        DstGridDataT* dst = dstGrid();
        dst->init({GridFlags::IsBreadthFirst, GridFlags::HasMinMax},
                  mOff.total, nanoMap,
                  toGridType<DstBuildT>(), gridClass);

        // Grid name (null-padded, max 255 chars; long names need blind data)
        std::memset(dst->mGridName, 0, GridData::MaxNameSize);
        std::strncpy(dst->mGridName,
                     srcGrid.getName().c_str(),
                     GridData::MaxNameSize - 1);

        // World bounding box from active-voxel CoordBBox.
        // evalActiveVoxelBoundingBox traverses only masks (topology), so it
        // does not trigger any further delayed leaf-data reads.
        const openvdb::CoordBBox iBBox = srcGrid.evalActiveVoxelBoundingBox();
        if (!iBBox.empty()) {
            const auto& xf   = srcGrid.transform();
            const auto  wMin = xf.indexToWorld(iBBox.min().asVec3d());
            const auto  wMax = xf.indexToWorld(iBBox.max().asVec3d());
            dst->mWorldBBox.min() = Vec3d(wMin[0], wMin[1], wMin[2]);
            dst->mWorldBBox.max() = Vec3d(wMax[0], wMax[1], wMax[2]);
            dst->setBBoxOn(true);
        }
    }
}; // OpenFileToNanoGridImpl

// ============================================================================
//  OpenFileToNanoIndexImpl<SrcGridT, DstBuildT, BufferT>
// ============================================================================
//  DstBuildT must be ValueIndex (all voxels) or ValueOnIndex (active only).
//  The output buffer contains:
//    [Grid][Tree][Root][Upper×N][Lower×N][LeafIndexBase×N]
//    [GridBlindMetaData×channels][float_channel_0][float_channel_1]...
//
//  Channel layout (float array of length mTotalValues):
//    [0]          = background value
//    [tiles...]   = active root / upper / lower tile values  (if includeTiles)
//    [leaves...]  = per-leaf active-voxel values (ValueOnIndex)
//                   or all-512 values (ValueIndex), one leaf after another
//    [+4 per leaf] = min, max, avg, stddev  (if includeStats)
// ============================================================================

template<typename SrcGridT, typename DstBuildT, typename BufferT>
class OpenFileToNanoIndexImpl
{
public:
    static_assert(BuildTraits<DstBuildT>::is_index,
                  "DstBuildT must be ValueIndex or ValueOnIndex");

    // ── Source types ────────────────────────────────────────────────────────
    using SrcValueT = typename SrcGridT::ValueType;
    using SrcTreeT  = typename SrcGridT::TreeType;
    using SrcRootT  = typename SrcTreeT::RootNodeType;
    using SrcUpperT = typename SrcRootT::ChildNodeType;
    using SrcLowerT = typename SrcUpperT::ChildNodeType;
    using SrcLeafT  = typename SrcLowerT::ChildNodeType;

    // ── Destination types ───────────────────────────────────────────────────
    using DstGridT      = NanoGrid<DstBuildT>;
    using DstTreeT      = NanoTree<DstBuildT>;
    using DstRootT      = NanoRoot<DstBuildT>;
    using DstUpperT     = NanoUpper<DstBuildT>;
    using DstLowerT     = NanoLower<DstBuildT>;
    using DstLeafT      = NanoLeaf<DstBuildT>;
    using DstGridDataT  = typename DstGridT::DataType;
    using DstTreeDataT  = typename DstTreeT::DataType;
    using DstRootDataT  = typename DstRootT::DataType;
    using DstUpperDataT = typename DstUpperT::DataType;
    using DstLowerDataT = typename DstLowerT::DataType;
    using DstLeafDataT  = typename DstLeafT::DataType; // LeafIndexBase

    GridHandle<BufferT> read(const std::string& filePath,
                             const std::string& gridName,
                             uint32_t           channels,
                             bool               includeTiles,
                             bool               includeStats,
                             const BufferT&     pool)
    {
        mChannels     = channels;
        mIncludeTiles = includeTiles;
        mIncludeStats = includeStats;

        // ── Pass 1: open with delayed-load, read topology ───────────────────
        openvdb::io::File file(filePath);
        file.open(/*delayLoad=*/true);

        openvdb::GridBase::Ptr baseGrid;
        if (gridName.empty()) {
            auto nameIt = file.beginName();
            if (nameIt == file.endName())
                throw std::runtime_error(
                    "OpenFileToNanoGrid: no grids found in '" + filePath + "'");
            baseGrid = file.readGrid(*nameIt);
        } else {
            try {
                baseGrid = file.readGrid(gridName);
            } catch (const std::exception&) {
                throw std::runtime_error(
                    "OpenFileToNanoGrid: grid '" + gridName + "' not found in '" + filePath + "'");
            }
            if (!baseGrid)
                throw std::runtime_error(
                    "OpenFileToNanoGrid: grid '" + gridName + "' not found in '" + filePath + "'");
        }

        auto srcGrid = openvdb::gridPtrCast<SrcGridT>(baseGrid);
        if (!srcGrid)
            throw std::runtime_error(
                "OpenFileToNanoGrid: type mismatch for '"
                + (gridName.empty() ? std::string("(first grid)") : gridName) + "'");

        const SrcTreeT& srcTree = srcGrid->tree();
        const SrcRootT& srcRoot = srcTree.root();

        collectNodes(srcRoot);

        const uint32_t nUpper    = static_cast<uint32_t>(mUpperNodes.size());
        const uint32_t nLower    = static_cast<uint32_t>(mLowerNodes.size());
        const uint32_t nLeaf     = static_cast<uint32_t>(mLeafNodes.size());
        const uint32_t rootTable = srcRoot.getTableSize();

        // Assign channel-array indices to every value slot
        countValues(srcRoot);

        // ── Buffer layout ───────────────────────────────────────────────────
        const uint64_t blindDataSize =
            math::AlignUp<NANOVDB_DATA_ALIGNMENT>(mTotalValues * sizeof(SrcValueT));

        mOff.grid  = 0;
        mOff.tree  = mOff.grid  + DstGridT::memUsage();
        mOff.root  = mOff.tree  + DstTreeT::memUsage();
        mOff.upper = mOff.root  + DstRootT::memUsage(rootTable);
        mOff.lower = mOff.upper + DstUpperT::memUsage() * nUpper;
        mOff.leaf  = mOff.lower + DstLowerT::memUsage() * nLower;
        mOff.meta  = mOff.leaf  + DstLeafT::DataType::memUsage() * nLeaf; // 96B each
        mOff.blind = mOff.meta  + sizeof(GridBlindMetaData) * mChannels;
        mOff.total = mOff.blind + blindDataSize * mChannels;

        auto buffer = BufferT::create(mOff.total, &pool);
        mBufPtr = static_cast<uint8_t*>(buffer.data());
        std::memset(mBufPtr, 0, mOff.total);

        // ── Pass 2: fill tree structure (bottom-up, no delayed-loads yet) ───
        for (uint32_t i = 0; i < nLeaf;  ++i) processLeaf(i);
        for (uint32_t i = 0; i < nLower; ++i) processLower(*mLowerNodes[i], i);
        for (uint32_t i = 0; i < nUpper; ++i) processUpper(*mUpperNodes[i], i);
        processRoot(srcRoot);
        processTree(srcTree, nUpper, nLower, nLeaf);
        processGrid(*srcGrid);

        // ── Blind metadata headers ──────────────────────────────────────────
        if (mChannels > 0) {
            setupBlindMeta(blindDataSize);
            // Fill channel 0 values (triggers delayed leaf loads), then copy to channels 1..N-1
            fillChannel0(*srcGrid, srcRoot);
            for (uint32_t ch = 1; ch < mChannels; ++ch)
                std::memcpy(channelPtr(ch), channelPtr(0), mTotalValues * sizeof(SrcValueT));
        }

        file.close();

        // BBox-only stats, matching createNanoGrid/openToIndexVDB behavior for index grids
        auto handle = GridHandle<BufferT>(std::move(buffer));
        updateGridStats(handle.template grid<DstBuildT>(), StatsMode::BBox);
        return handle;
    }

private:
    uint32_t mChannels     = 1;
    bool     mIncludeTiles = true;
    bool     mIncludeStats = true;

    // ── Indexed node arrays (BFS order) ────────────────────────────────────
    std::vector<const SrcUpperT*> mUpperNodes;
    std::vector<const SrcLowerT*> mLowerNodes;
    std::vector<const SrcLeafT*>  mLeafNodes;
    std::unordered_map<const SrcUpperT*, uint32_t> mUpperIdx;
    std::unordered_map<const SrcLowerT*, uint32_t> mLowerIdx;
    std::unordered_map<const SrcLeafT*,  uint32_t> mLeafIdx;

    uint8_t* mBufPtr = nullptr;
    struct Offsets { uint64_t grid, tree, root, upper, lower, leaf, meta, blind, total; } mOff{};

    // ── Channel index tracking ──────────────────────────────────────────────
    uint64_t              mRootTileStart  = 0; // channel idx for first active root tile
    std::vector<uint64_t> mUpperTileStart;     // per-upper-node: channel idx for first active tile
    std::vector<uint64_t> mLowerTileStart;     // per-lower-node: channel idx for first active tile
    std::vector<uint64_t> mLeafOffset;         // per-leaf: channel idx for first value
    uint64_t              mTotalValues    = 0;

    // ── Buffer accessors ───────────────────────────────────────────────────
    DstGridDataT*  dstGrid()            { return util::PtrAdd<DstGridDataT>(mBufPtr, mOff.grid);  }
    DstTreeDataT*  dstTree()            { return util::PtrAdd<DstTreeDataT>(mBufPtr, mOff.tree);  }
    DstRootDataT*  dstRoot()            { return util::PtrAdd<DstRootDataT>(mBufPtr, mOff.root);  }
    DstUpperDataT* dstUpper(uint32_t i) { return util::PtrAdd<DstUpperDataT>(mBufPtr, mOff.upper) + i; }
    DstLowerDataT* dstLower(uint32_t i) { return util::PtrAdd<DstLowerDataT>(mBufPtr, mOff.lower) + i; }
    DstLeafDataT*  dstLeaf (uint32_t i) { return util::PtrAdd<DstLeafDataT> (mBufPtr, mOff.leaf)  + i; }

    SrcValueT* channelPtr(uint32_t ch) {
        const uint64_t stride =
            math::AlignUp<NANOVDB_DATA_ALIGNMENT>(mTotalValues * sizeof(SrcValueT));
        return reinterpret_cast<SrcValueT*>(mBufPtr + mOff.blind + ch * stride);
    }

    // ── BFS node collection ────────────────────────────────────────────────
    void collectNodes(const SrcRootT& root)
    {
        for (auto it = root.cbeginChildOn(); it; ++it) {
            const SrcUpperT* n = &(*it);
            mUpperIdx[n] = static_cast<uint32_t>(mUpperNodes.size());
            mUpperNodes.push_back(n);
        }
        for (const SrcUpperT* up : mUpperNodes)
            for (auto it = up->cbeginChildOn(); it; ++it) {
                const SrcLowerT* n = &(*it);
                mLowerIdx[n] = static_cast<uint32_t>(mLowerNodes.size());
                mLowerNodes.push_back(n);
            }
        for (const SrcLowerT* lo : mLowerNodes)
            for (auto it = lo->cbeginChildOn(); it; ++it) {
                const SrcLeafT* n = &(*it);
                mLeafIdx[n] = static_cast<uint32_t>(mLeafNodes.size());
                mLeafNodes.push_back(n);
            }
    }

    template<uint32_t LOG2DIM>
    static void copyMask(const openvdb::util::NodeMask<LOG2DIM>& src,
                         nanovdb::Mask<LOG2DIM>& dst)
    {
        static_assert(sizeof(src) == sizeof(dst), "mask size mismatch");
        std::memcpy(&dst, &src, sizeof(dst));
    }

    // ── Assign a channel-array index to every value slot ──────────────────
    // Order: [0]=background, [root tiles], [upper tiles], [lower tiles],
    //        then per-leaf: [active voxels] + optional [min,max,avg,dev].
    // This order must match fillChannel0().
    void countValues(const SrcRootT& srcRoot)
    {
        uint64_t idx = 1; // slot 0 = background

        if (mIncludeTiles) {
            mRootTileStart = idx;
            for (auto it = srcRoot.cbeginValueOn(); it; ++it) ++idx;

            mUpperTileStart.resize(mUpperNodes.size());
            for (uint32_t i = 0; i < (uint32_t)mUpperNodes.size(); ++i) {
                mUpperTileStart[i] = idx;
                for (auto it = mUpperNodes[i]->cbeginValueOn(); it; ++it) ++idx;
            }

            mLowerTileStart.resize(mLowerNodes.size());
            for (uint32_t i = 0; i < (uint32_t)mLowerNodes.size(); ++i) {
                mLowerTileStart[i] = idx;
                for (auto it = mLowerNodes[i]->cbeginValueOn(); it; ++it) ++idx;
            }
        }

        mLeafOffset.resize(mLeafNodes.size());
        for (uint32_t i = 0; i < (uint32_t)mLeafNodes.size(); ++i) {
            mLeafOffset[i] = idx;
            uint64_t leafSlots;
            if constexpr (BuildTraits<DstBuildT>::is_onindex)
                leafSlots = mLeafNodes[i]->valueMask().countOn(); // active only
            else
                leafSlots = SrcLeafT::SIZE;                       // all 512
            idx += leafSlots + (mIncludeStats ? 4u : 0u);
        }

        mTotalValues = idx;
    }

    // ── Leaf: set mOffset, mPrefixSum (ValueOnIndex), masks ────────────────
    // No delayed-load triggered here — value mask is part of topology.
    void processLeaf(uint32_t i)
    {
        const SrcLeafT& src = *mLeafNodes[i];
        DstLeafDataT*   dst = dstLeaf(i);

        const openvdb::Coord o = src.origin();
        dst->mBBoxMin = Coord(o.x(), o.y(), o.z());
        constexpr uint8_t kMaxDif = static_cast<uint8_t>((1u << SrcLeafT::LOG2DIM) - 1u);
        dst->mBBoxDif[0] = dst->mBBoxDif[1] = dst->mBBoxDif[2] = kMaxDif;

        copyMask(src.valueMask(), dst->mValueMask);

        dst->mOffset = mLeafOffset[i];

        if constexpr (BuildTraits<DstBuildT>::is_onindex) {
            // Pack 7 × 9-bit cumulative popcounts of the value-mask words.
            // Word 7's cumulative count is recovered at access time via valueCount().
            uint64_t ps = 0, cum = 0;
            for (int w = 0; w < 7; ++w) {
                cum += util::countOn(dst->mValueMask.words()[w]);
                ps  |= (cum & 511u) << (9u * w);
            }
            dst->mPrefixSum = ps;
        }

        if (mIncludeStats) dst->mFlags |= uint8_t(1u << 4); // hasStats bit
    }

    // ── Lower internal node ─────────────────────────────────────────────────
    void processLower(const SrcLowerT& src, uint32_t i)
    {
        DstLowerDataT* dst = dstLower(i);

        const openvdb::Coord o = src.origin();
        constexpr int kSpan = static_cast<int>((1u << SrcLowerT::TOTAL) - 1u);
        dst->mBBox.min() = Coord(o.x(), o.y(), o.z());
        dst->mBBox.max() = Coord(o.x()+kSpan, o.y()+kSpan, o.z()+kSpan);

        copyMask(src.getValueMask(), dst->mValueMask);
        copyMask(src.getChildMask(), dst->mChildMask);

        // Active value tiles: store their channel-array index as mTable[pos].value
        if (mIncludeTiles) {
            uint64_t tileIdx = mLowerTileStart[i];
            for (auto it = src.cbeginValueOn(); it; ++it)
                dst->mTable[static_cast<uint32_t>(it.pos())].value = tileIdx++;
        }
        // Inactive tiles remain 0 (= background index) from memset

        // Child leaf nodes (setChild requires mChildMask already set)
        for (auto it = src.cbeginChildOn(); it; ++it) {
            const uint32_t pos = static_cast<uint32_t>(it.pos());
            dst->setChild(pos, dstLeaf(mLeafIdx.at(&(*it))));
        }
    }

    // ── Upper internal node ─────────────────────────────────────────────────
    void processUpper(const SrcUpperT& src, uint32_t i)
    {
        DstUpperDataT* dst = dstUpper(i);

        const openvdb::Coord o = src.origin();
        constexpr int kSpan = static_cast<int>((1u << SrcUpperT::TOTAL) - 1u);
        dst->mBBox.min() = Coord(o.x(), o.y(), o.z());
        dst->mBBox.max() = Coord(o.x()+kSpan, o.y()+kSpan, o.z()+kSpan);

        copyMask(src.getValueMask(), dst->mValueMask);
        copyMask(src.getChildMask(), dst->mChildMask);

        if (mIncludeTiles) {
            uint64_t tileIdx = mUpperTileStart[i];
            for (auto it = src.cbeginValueOn(); it; ++it)
                dst->mTable[static_cast<uint32_t>(it.pos())].value = tileIdx++;
        }

        for (auto it = src.cbeginChildOn(); it; ++it) {
            const uint32_t pos = static_cast<uint32_t>(it.pos());
            dst->setChild(pos, dstLower(mLowerIdx.at(&(*it))));
        }
    }

    // ── Root node ───────────────────────────────────────────────────────────
    void processRoot(const SrcRootT& srcRoot)
    {
        DstRootDataT* dst = dstRoot();
        dst->mTableSize  = srcRoot.getTableSize();
        dst->mBackground = uint64_t(0); // index 0 = background slot in channel array

        uint64_t rootActiveIdx = mRootTileStart;
        uint32_t tilePos = 0;
        for (auto it = srcRoot.cbeginChildAll(); it; ++it, ++tilePos) {
            SrcValueT tileValue{};
            const SrcUpperT* srcChild = it.probeChild(tileValue);
            if (srcChild) {
                const uint32_t ci = mUpperIdx.at(srcChild);
                dst->tile(tilePos)->setChild(srcChild->origin(), dstUpper(ci), dst);
            } else {
                const openvdb::Coord oc  = it.getCoord();
                const bool           act = it.isValueOn();
                const uint64_t chIdx = (act && mIncludeTiles) ? rootActiveIdx++ : uint64_t(0);
                dst->tile(tilePos)->setValue(Coord(oc.x(), oc.y(), oc.z()), act, chIdx);
            }
        }
    }

    // ── Tree node ───────────────────────────────────────────────────────────
    void processTree(const SrcTreeT& srcTree, uint32_t nUpper, uint32_t nLower, uint32_t nLeaf)
    {
        DstTreeDataT* dst = dstTree();
        dst->setRoot(dstRoot());
        if (nUpper > 0) dst->template setFirstNode<DstUpperT>(util::PtrAdd<DstUpperT>(mBufPtr, mOff.upper));
        if (nLower > 0) dst->template setFirstNode<DstLowerT>(util::PtrAdd<DstLowerT>(mBufPtr, mOff.lower));
        if (nLeaf  > 0) dst->template setFirstNode<DstLeafT> (util::PtrAdd<DstLeafT> (mBufPtr, mOff.leaf));
        dst->mNodeCount[0] = nLeaf;
        dst->mNodeCount[1] = nLower;
        dst->mNodeCount[2] = nUpper;
        dst->mTileCount[0] = dst->mTileCount[1] = dst->mTileCount[2] = 0; // set by updateGridStats
        dst->mVoxelCount   = srcTree.activeVoxelCount();
    }

    // ── Grid header ─────────────────────────────────────────────────────────
    void processGrid(const SrcGridT& srcGrid)
    {
        const auto mat4 = srcGrid.transform().baseMap()->getAffineMap()->getMat4();
        Map nanoMap;
        nanoMap.set(mat4, mat4.inverse());

        DstGridDataT* dst = dstGrid();
        dst->init({GridFlags::IsBreadthFirst}, mOff.total, nanoMap,
                  toGridType<DstBuildT>(), GridClass::IndexGrid);

        std::memset(dst->mGridName, 0, GridData::MaxNameSize);
        std::strncpy(dst->mGridName, srcGrid.getName().c_str(), GridData::MaxNameSize - 1);

        dst->mData1 = mTotalValues; // total indexed values — accessed via grid.valueCount()

        const openvdb::CoordBBox iBBox = srcGrid.evalActiveVoxelBoundingBox();
        if (!iBBox.empty()) {
            const auto& xf   = srcGrid.transform();
            const auto  wMin = xf.indexToWorld(iBBox.min().asVec3d());
            const auto  wMax = xf.indexToWorld(iBBox.max().asVec3d());
            dst->mWorldBBox.min() = Vec3d(wMin[0], wMin[1], wMin[2]);
            dst->mWorldBBox.max() = Vec3d(wMax[0], wMax[1], wMax[2]);
            dst->setBBoxOn(true);
        }

        if (mChannels > 0) {
            auto* firstMeta = util::PtrAdd<GridBlindMetaData>(mBufPtr, mOff.meta);
            dst->mBlindMetadataOffset = util::PtrDiff(firstMeta, dst);
            dst->mBlindMetadataCount  = mChannels;
        }
    }

    // ── GridBlindMetaData headers ────────────────────────────────────────────
    void setupBlindMeta(uint64_t blindDataSize)
    {
        for (uint32_t ch = 0; ch < mChannels; ++ch) {
            auto*    meta  = util::PtrAdd<GridBlindMetaData>(mBufPtr, mOff.meta) + ch;
            uint8_t* blind = mBufPtr + mOff.blind + ch * blindDataSize;
            std::memset(meta, 0, sizeof(GridBlindMetaData));
            meta->mValueCount = mTotalValues;
            meta->mValueSize  = sizeof(SrcValueT);
            meta->mDataClass  = GridBlindDataClass::ChannelArray;
            meta->mDataType   = toGridType<SrcValueT>();
            meta->mSemantic   = GridBlindDataSemantic::Unknown;
            meta->setName(("channel_" + std::to_string(ch)).c_str());
            meta->setBlindData(blind);
        }
    }

    // ── Fill channel 0 with actual values (triggers delayed leaf loads) ─────
    void fillChannel0(const SrcGridT& srcGrid, const SrcRootT& srcRoot)
    {
        SrcValueT* ch = channelPtr(0);

        // Slot 0 = background
        ch[0] = static_cast<SrcValueT>(srcRoot.background());

        // Active root tile values
        if (mIncludeTiles) {
            uint64_t idx = mRootTileStart;
            for (auto it = srcRoot.cbeginValueOn(); it; ++it)
                ch[idx++] = static_cast<SrcValueT>(*it);

            // Upper active tile values (from topology — no delayed load)
            for (uint32_t i = 0; i < (uint32_t)mUpperNodes.size(); ++i) {
                idx = mUpperTileStart[i];
                for (auto it = mUpperNodes[i]->cbeginValueOn(); it; ++it)
                    ch[idx++] = static_cast<SrcValueT>(*it);
            }

            // Lower active tile values
            for (uint32_t i = 0; i < (uint32_t)mLowerNodes.size(); ++i) {
                idx = mLowerTileStart[i];
                for (auto it = mLowerNodes[i]->cbeginValueOn(); it; ++it)
                    ch[idx++] = static_cast<SrcValueT>(*it);
            }
        }

        // Leaf values — each leaf's first getValue() triggers its delayed load
        for (uint32_t i = 0; i < (uint32_t)mLeafNodes.size(); ++i) {
            const SrcLeafT& leaf = *mLeafNodes[i];
            uint64_t idx = mLeafOffset[i];

            SrcValueT vMin =  std::numeric_limits<SrcValueT>::max();
            SrcValueT vMax = -std::numeric_limits<SrcValueT>::max();
            double    sum  = 0.0, sum2 = 0.0;

            if constexpr (BuildTraits<DstBuildT>::is_onindex) {
                // Only active voxels — isValueOn() uses the already-loaded mask
                for (uint32_t v = 0; v < SrcLeafT::SIZE; ++v) {
                    if (!leaf.isValueOn(v)) continue;
                    const SrcValueT val = static_cast<SrcValueT>(leaf.getValue(v));
                    ch[idx++] = val;
                    if (mIncludeStats) {
                        if (val < vMin) vMin = val;
                        if (val > vMax) vMax = val;
                        sum  += double(val);
                        sum2 += double(val) * double(val);
                    }
                }
            } else {
                // All 512 voxels
                for (uint32_t v = 0; v < SrcLeafT::SIZE; ++v) {
                    const SrcValueT val = static_cast<SrcValueT>(leaf.getValue(v));
                    ch[idx++] = val;
                    if (mIncludeStats && leaf.isValueOn(v)) {
                        if (val < vMin) vMin = val;
                        if (val > vMax) vMax = val;
                        sum  += double(val);
                        sum2 += double(val) * double(val);
                    }
                }
            }

            // Stats slots (always written when includeStats, even if no active voxels)
            if (mIncludeStats) {
                const uint64_t activeCount = idx - mLeafOffset[i];
                if (activeCount > 0) {
                    const double mean = sum / double(activeCount);
                    ch[idx++] = vMin;
                    ch[idx++] = vMax;
                    ch[idx++] = static_cast<SrcValueT>(mean);
                    ch[idx++] = static_cast<SrcValueT>(
                        std::sqrt(std::max(0.0, sum2 / double(activeCount) - mean * mean)));
                } else {
                    ch[idx++] = SrcValueT(0);
                    ch[idx++] = SrcValueT(0);
                    ch[idx++] = SrcValueT(0);
                    ch[idx++] = SrcValueT(0);
                }
            }
        }
    }
}; // OpenFileToNanoIndexImpl

} // namespace detail

// ============================================================================
//  Public API
// ============================================================================

/// @brief Read a grid from an OpenVDB .vdb file directly into a NanoVDB
///        GridHandle, without first building a fully loaded openvdb::Grid.
///
/// Unlike createNanoGrid() (which converts an already-loaded openvdb::Grid),
/// this function reads topology and voxel data incrementally from the file,
/// keeping peak memory to:  OpenVDB topology + NanoVDB buffer.
///
/// @tparam SrcGridT   OpenVDB grid type to read, e.g. openvdb::FloatGrid
/// @tparam DstBuildT  NanoVDB build/value type (default: SrcGridT::ValueType)
/// @tparam BufferT    Host buffer type for the returned GridHandle
///                    (default: nanovdb::HostBuffer)
///
/// @param filePath  Path to the .vdb file on disk
/// @param gridName  Name of the grid to read.  If empty, the first grid in
///                  the file is used.
/// @param pool      Optional buffer allocator / memory pool
///
/// @return A GridHandle<BufferT> wrapping the NanoVDB grid.
///
/// @throw std::runtime_error if the file cannot be opened, the grid is not
///        found, or the OpenVDB grid type does not match SrcGridT.
template<typename SrcGridT,
         typename DstBuildT = typename SrcGridT::ValueType,
         typename BufferT   = HostBuffer>
GridHandle<BufferT>
openFileToNanoGrid(const std::string& filePath,
                   const std::string& gridName = "",
                   const BufferT&     pool     = BufferT())
{
    openvdb::initialize();
    detail::OpenFileToNanoGridImpl<SrcGridT, DstBuildT, BufferT> impl;
    return impl.read(filePath, gridName, pool);
}

/// @brief Read a grid from an OpenVDB .vdb file directly into a NanoVDB
///        index GridHandle + sidecar value channel(s), without first building
///        a fully loaded openvdb::Grid.
///
/// An index grid stores a dense or sparse mapping from voxel position to an
/// integer slot in a separate value channel (blind data array).  This layout
/// lets multiple value channels (float, Vec3f, …) share one topology, and
/// enables GPU-friendly scatter/gather access via ChannelAccessor<T>.
///
/// @tparam SrcGridT   OpenVDB source grid type, e.g. openvdb::FloatGrid
/// @tparam DstBuildT  Index build type: ValueOnIndex (active voxels only,
///                    default) or ValueIndex (all 512 voxels per leaf)
/// @tparam BufferT    Host buffer type (default: nanovdb::HostBuffer)
///
/// @param filePath     Path to the .vdb file on disk
/// @param gridName     Grid to read; empty = first grid in the file
/// @param channels     Number of sidecar value copies to embed as blind data
///                     (default 1). Channel 0 holds the original values;
///                     channels 1..N-1 are identical copies.
/// @param includeTiles If true, active internal-node tile values are indexed
///                     in the channel array (default true)
/// @param includeStats If true, each leaf appends 4 extra channel slots
///                     (min, max, avg, stddev of its active voxels)
///                     and sets the hasStats flag on the leaf (default true)
/// @param pool         Optional buffer allocator / memory pool
///
/// @return GridHandle<BufferT> containing the NanoVDB index grid and its
///         embedded float sidecar channel(s).
///
/// @note Use nanovdb::ChannelAccessor<SrcValueT, DstBuildT> to look up values:
///   @code
///   auto handle = openFileToNanoIndexGrid<openvdb::FloatGrid>("in.vdb");
///   auto* grid  = handle.grid<nanovdb::ValueOnIndex>();
///   nanovdb::ChannelAccessor<float, nanovdb::ValueOnIndex> acc(*grid, 0u);
///   float v = acc(10, 20, 30);
///   @endcode
///
/// @throw std::runtime_error on file/grid/type errors.
template<typename SrcGridT,
         typename DstBuildT = ValueOnIndex,
         typename BufferT   = HostBuffer>
typename util::enable_if<BuildTraits<DstBuildT>::is_index, GridHandle<BufferT>>::type
openFileToNanoIndexGrid(const std::string& filePath,
                        const std::string& gridName    = "",
                        uint32_t           channels    = 1u,
                        bool               includeTiles = true,
                        bool               includeStats = true,
                        const BufferT&     pool         = BufferT())
{
    openvdb::initialize();
    detail::OpenFileToNanoIndexImpl<SrcGridT, DstBuildT, BufferT> impl;
    return impl.read(filePath, gridName, channels, includeTiles, includeStats, pool);
}

} // namespace tools
} // namespace nanovdb

#endif // NANOVDB_TOOLS_OPEN_FILE_TO_NANO_GRID_H_HAS_BEEN_INCLUDED
