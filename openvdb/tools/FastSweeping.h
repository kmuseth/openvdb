// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
//
/// @file    FastSweeping.h
///
/// @author  Ken Museth
///
/// @brief   Defined the six functions {fog,sdf}To{Sdf,Ext,SdfAndExt} in
///          addition to the two functions maskSdf and dilateSdf. Sdf denotes
///          a signed-distance field (i.e. negative values are insdie), fog
///          is a scalar fog volume (i.e. higher values are inside), and Ext is
///          a field (currently limited to a scalar) that is extended off the
///          iso-surface. All these functions are implemented by the methods of
///          the class dubbed FastSweeping.
///
/// @todo    1) Sort uint32 offsets instead of array of Coord
///          2) Concurrent bi-directional sweeping
///          3) Rebuild narrow-band level set
///          4) Allow for the grid types of the sdf and ext to be different
///
/// @note    Solves the (simplified) Eikonal Eq: @f$|\nabla \phi|^2 = 1@f$ and
///          performs velocity extension,  @f$\nable f\nabla \phi = 0@f$, both
///          by means of the fast sweeping algorithm detailed in:
///          "A Fast Sweeping Method For Eikonal Equations"
///          by H. Zhao, Mathematics of Computation, Vol 74(230), pp 603-627, 2004
///
/// @details The algorithm used below for parallel fast sweeping was first publised in:
///          "New Algorithm for Sparse and Parallel Fast Sweeping: Efficient
///          Computation of Sparse Distance Fields" by K. Museth, ACM SIGGRAPH Talk,
///          2017, http://www.museth.org/Ken/Publications_files/Museth_SIG17.pdf

#ifndef OPENVDB_TOOLS_FASTSWEEPING_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_FASTSWEEPING_HAS_BEEN_INCLUDED

//#define BENCHMARK_FAST_SWEEPING

#include <type_traits>// for static_assert
#include <cmath>
#include <limits>
#include <unordered_map>

#include <tbb/parallel_for.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/task_group.h>

#include <openvdb/math/Math.h> // for Abs() and isExactlyEqual()
#include <openvdb/math/Stencils.h> // for GradStencil
#include <openvdb/tree/LeafManager.h>
#include "LevelSetUtil.h"
#include "Morphology.h"

#include "Statistics.h"
#ifdef BENCHMARK_FAST_SWEEPING
#include <openvdb/util/CpuTimer.h>
#endif

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// @brief Converts a scalar fog volume into a signed distance function. Active input voxels
///        with scalar values above the given isoValue will have NEGATIVE distance
///        values on output, i.e. they are assumed to be INSIDE the iso-surface.
///
/// @return A shared pointer to a signed-distance field defined on the active values
///         of the input fog volume.
///
/// @param fogGrid  Scalar (floating-point) volume from which an
///                 iso-surface can be defined.
///
/// @param isoValue A value which defines a smooth iso-surface that
///                 intersects active voxels in @a fogGrid.
///
/// @param nIter    Number of iterations of the fast sweeping algorithm.
///                 each performing 2^3 = 8 sweeps.
///
/// @note Strictly speaking a fog volume is normalized to the range [0,1] but this
///       method accepts a scalar volume with an arbitary range, as long as the it
///       includes the @a isoValue.
///
/// @details Topology of output grid is identical to that of the input grid, except
///          active tiles in the input grid will be converted to active voxels
///          in the output grid!
///
/// @throw  RuntimeError if the iso-surface does not intersected any active
///         voxels or if it intersects any active tiles in @a fogGrid.
template<typename GridT>
typename GridT::Ptr
fogToSdf(const GridT &fogGrid,
         typename GridT::ValueType isoValue,
         int nIter = 1);

/// @brief Given an existing approximate SDF it solves the Eikonal equation for all its
///        active voxels. Active input voxels with a signed distance value above the
///        given isoValue will have POSITIVE distance values on output, i.e. they are
///        assumed to be OUTSIDE the iso-surface.
///
/// @return A shared pointer to a signed-distance field defined on the active values
///         of the input sdf volume.
///
/// @param sdfGrid  An approximate signed distance field to the specified iso-surface.
///
/// @param isoValue A value which defines a smooth iso-surface that
///                 intersects active voxels in @a sdfGrid.
///
/// @param nIter    Number of iterations of the fast sweeping algorithm.
///                 each performing 2^3 = 8 sweeps.
///
/// @note The only difference between this method and fogToSdf, defined above, is the
///       convention of the sign of the output distance field.
///
/// @details Topology of output grid is identical to that of the input grid, except
///          active tiles in the input grid will be converted to active voxels
///          in the output grid!
///
/// @throw  RuntimeError if the iso-surface does not intersected any active
///         voxels or if it intersects any active tiles in @a sdfGrid.
template<typename GridT>
typename GridT::Ptr
sdfToSdf(const GridT &sdfGrid,
         typename GridT::ValueType isoValue = 0,
         int nIter = 1);

/// @brief Computes the extension of a scalar field, defined by the specified functor,
///        off an iso-surface from an input FOG volume.
///
/// @return A shared pointer to the extension field defined from the active values in
///         the input fog volume.
///
/// @param fogGrid  Scalar (floating-point) volume from which an
///                 iso-surface can be defined.
///
/// @param op       Functor with signature [](const Vec3R &xyz)->float that
///                 defines the Dirichlet boundary condition, on the iso-surface,
///                 of the field to be extended.
///
/// @param isoValue A value which defines a smooth iso-surface that
///                 intersects active voxels in @a fogGrid.
///
/// @param nIter    Number of iterations of the fast sweeping algorithm.
///                 each performing 2^3 = 8 sweeps.
///
/// @note Strictly speaking a fog volume is normalized to the range [0,1] but this
///       method accepts a scalar volume with an arbitary range, as long as the it
///       includes the @a isoValue.
///
/// @details Topology of output grid is identical to that of the input grid, except
///          active tiles in the input grid will be converted to active voxels
///          in the output grid!
///
/// @throw  RuntimeError if the iso-surface does not intersected any active
///         voxels or if it intersects any active tiles in @a fogGrid.
template<typename GridT, typename OpT>
typename GridT::Ptr
fogToExt(const GridT &fogGrid,
         const OpT &op,
         typename GridT::ValueType isoValue,
         int nIter = 1);

/// @brief Computes the extension of a scalar field, defined by the specified functor,
///        off an iso-surface from an input SDF volume.
///
/// @return A shared pointer to the extension field defined on the active values in the
///         input signed distance field.
///
/// @param sdfGrid  An approximate signed distance field to the specified iso-surface.
///
/// @param op       Functor with signature [](const Vec3R &xyz)->float that
///                 defines the Dirichlet boundary condition, on the iso-surface,
///                 of the field to be extended.
///
/// @param isoValue A value which defines a smooth iso-surface that
///                 intersects active voxels in @a sdfGrid.
///
/// @param nIter    Number of iterations of the fast sweeping algorithm.
///                 each performing 2^3 = 8 sweeps.
///
/// @note The only difference between this method and fogToEXT, defined above, is the
///       convention of the sign of the signed distance field.
///
/// @details Topology of output grid is identical to that of the input grid, except
///          active tiles in the input grid will be converted to active voxels
///          in the output grid!
///
/// @throw  RuntimeError if the iso-surface does not intersected any active
///         voxels or if it intersects any active tiles in @a sdfGrid.
template<typename GridT, typename OpT>
typename GridT::Ptr
sdfToExt(const GridT &sdfGrid,
         const OpT &op,
         typename GridT::ValueType isoValue = 0,
         int nIter = 1);

/// @brief Computes the signed distance field and the extension of a scalar field,
///        defined by the specified functor, off an iso-surface from an input FOG volume.
///
/// @return An array of two shared pointers to respectively the SDF and extension field
///
/// @param fogGrid  Scalar (floating-point) volume from which an
///                 iso-surface can be defined.
///
/// @param op       Functor with signature [](const Vec3R &xyz)->float that
///                 defines the Dirichlet boundary condition, on the iso-surface,
///                 of the field to be extended.
///
/// @param isoValue A value which defines a smooth iso-surface that
///                 intersects active voxels in @a fogGrid.
///
/// @param nIter    Number of iterations of the fast sweeping algorithm.
///                 each performing 2^3 = 8 sweeps.
///
/// @note Strictly speaking a fog volume is normalized to the range [0,1] but this
///       method accepts a scalar volume with an arbitary range, as long as the it
///       includes the @a isoValue.
///
/// @details Topology of output grids are identical to that of the input grid, except
///          active tiles in the input grid will be converted to active voxels
///          in the output grids!
///
/// @throw  RuntimeError if the iso-surface does not intersected any active
///         voxels or if it intersects any active tiles in @a fogGrid.
template<typename GridT, typename OpT>
std::array<typename GridT::Ptr, 2>
fogToSdfAndExt(const GridT &fogGrid,
               const OpT &op,
               typename GridT::ValueType isoValue,
               int nIter = 1);

/// @brief Computes the signed distance field and the extension of a scalar field,
///        defined by the specified functor, off an iso-surface from an input SDF volume.
///
/// @return An array of two shared pointers to respectively the SDF and extension field
///
/// @param sdfGrid  Scalar (floating-point) volume from which an
///                 iso-surface can be defined.
///
/// @param op       Functor with signature [](const Vec3R &xyz)->float that
///                 defines the Dirichlet boundary condition, on the iso-surface,
///                 of the field to be extended.
///
/// @param isoValue A value which defines a smooth iso-surface that
///                 intersects active voxels in @a sdfGrid.
///
/// @param nIter    Number of iterations of the fast sweeping algorithm.
///                 each performing 2^3 = 8 sweeps.
///
/// @note Strictly speaking a fog volume is normalized to the range [0,1] but this
///       method accepts a scalar volume with an arbitary range, as long as the it
///       includes the @a isoValue.
///
/// @details Topology of output grids are identical to that of the input grid, except
///          active tiles in the input grid will be converted to active voxels
///          in the output grids!
///
/// @throw  RuntimeError if the iso-surface does not intersected any active
///         voxels or if it intersects any active tiles in @a fogGrid.
template<typename GridT, typename OpT>
std::array<typename GridT::Ptr, 2>
sdfToSdfAndExt(const GridT &sdfGrid,
               const OpT &op,
               typename GridT::ValueType isoValue = 0,
               int nIter = 1);

/// @brief Dilates an existing signed distance filed by a specified number of voxels
///
/// @return A shared pointer to the dilated signed distance field.
///
/// @param sdfGrid  Input signed distance field to to be dilated.
///
/// @param dilation Numer of voxels that the input SDF will be dilated.
///
/// @param nn       Stencil-pattern used for dilation
///
/// @param nIter    Number of iterations of the fast sweeping algorithm.
///                 each performing 2^3 = 8 sweeps.
///
/// @details Topology will change as a result of the dilation
template<typename GridT>
typename GridT::Ptr
dilateSdf(const GridT &sdfGrid,
          int dilation,
          NearestNeighbors nn = NN_FACE,
          int nIter = 1);

/// @brief Fills mask by extending an existing signed distance field into it.
///
/// @return A shared pointer to the masked signed distance field.
///
/// @param sdfGrid  Input signed distance field to to be dilated.
///
/// @param mask     Mask used to idetify the topology of the output SDF.
///                 Note this mask is assume to overlap with the sdfGrid.
///
/// @param ignoreActiveTiles If false, active tiles in the mask are treated
///                 as active voxels. Else they are ignored.
///
/// @param nIter    Number of iterations of the fast sweeping algorithm.
///                 each performing 2^3 = 8 sweeps.
///
/// @details Topology of the output SDF is determined by the union of the active
///          voxels (or optionally values) in @a sdfGrid and @a mask.
template<typename GridT, typename MaskTreeT>
typename GridT::Ptr
maskSdf(const GridT &sdfGrid,
        const Grid<MaskTreeT> &mask,
        bool ignoreActiveTiles = false,
        int nIter = 1);

////////////////////////////////////////////////////////////////////////////////
/// @brief Computes signed distance values from an initial narrow-band
/// levelset. This is done by means of a sparse and parallel fast sweeping
/// algorithm based on a first order Goudonov's scheme.
///
/// Solves: @f$|\nabla \phi|^2 = 1 @f$
///
/// @details The grid needs to have correct values set near the zero level set, where "near"
/// means 1.5 times the size of a grid voxel.  These near values are not modified.
/// Use multiple iterations (@a nIter > 1) for complex geometry where characteristics cross
/// (along the medial axis). Each iteration performs eight sweeps corresponding to
/// the grouping of characteristics in the eight coordinate quadrants.
///
/// @param grid   a grid containing initial zero-crossing signed distance values
///               (@f$|\phi|@f$ < 1.5*voxel size).
/// @param nIter  the number of interations to be performed (each with 8 sweeps).
///
/// @note Multi-threaded and sparse!
///
/// @warning This method only works for scalar, floating-point grids
/// and it is @e not the preferred way to solve the eikonal equation for
/// narrow-band level sets (of width ~2x3). Rather it is an efficient
/// technique when distance values must be computed far from known
/// distance values (typically close to an interface).
///
/// @warning This class is for experts only (the sequence in with the different
///          methods are called is important). Instead call one of the free-standing
///          functions listed above!
template<typename GridT>
class FastSweeping
{
    using ValueT = typename GridT::ValueType;
    using TreeT  = typename GridT::TreeType;
    using AccT   = tree::ValueAccessor<TreeT, false>;//don't register accessors
    using SweepMaskTreeT = typename TreeT::template ValueConverter<ValueMask>::Type;
    using SweepMaskAccT = tree::ValueAccessor<SweepMaskTreeT, false>;//don't register accessors

    // This class can only be templated on a grid with floating-point values!
    static_assert(std::is_floating_point<ValueT>::value,
                  "FastSweeping requires a grid with floating-point values");
public:

    /// @brief Constructor
    FastSweeping();

     /// @brief Destructor.
    ~FastSweeping() { this->clear(); }

    /// @brief Disallow copy construction.
    FastSweeping(const FastSweeping&) = delete;

    /// @brief Disallow copy assignment.
    FastSweeping& operator=(const FastSweeping&) = delete;

    /// @brief Returns a shared pointer to the signed distance field computed
    ///        by this class.
    /// @warning This shared pointer might point to NULL if the grid has not been
    ///          initialize (by one of the init methods) or computed (by the sweep
    ///          method).
    typename GridT::Ptr sdfGrid() { return mGrid1; }
    typename GridT::Ptr extGrid() { return mGrid2; }

    bool initSdf(const GridT &fogGrid, ValueT isoValue, bool isInputSdf);

    template <typename OpT>
    bool initExt(const GridT &fogGrid, const OpT &op, ValueT isoValue, bool isInputSdf);

    // use tools::NN_FACE_EDGE for improved dilation
    bool initDilate(const GridT &sdfGrid, int dilation, NearestNeighbors nn = NN_FACE);

    template<typename MaskTreeT>
    bool initMask(const GridT &sdfGrid, const Grid<MaskTreeT> &mask, bool ignoreActiveTiles = false);

    /// @brief Perform @a nIter iterations of the fast sweeping algorithm.
    void sweep(int nIter = 1, bool finalize = true);

    void clear();

    /// @brief Return the number of voxels that will be solved for.
    size_t voxelCount() const;

    /// @brief Return the number of voxels that defined the boundary condition.
    size_t boundaryCount() const;

    /// @brief Prune the sweep mask and cache leaf origins.
    void computeSweepMaskLeafOrigins();

    /// @brief Return true if there are voxels and boundaries to solve for
    bool isValid() const { return this->voxelCount() > 0 && this->boundaryCount() > 0; }
private:

    // Private classes to initialize the grid and construct
    template<typename>
    struct MaskKernel;//   initialization to extand a SDF into a mask
    template<typename>
    struct InitExt;
    struct InitSdf;
    struct DilateKernel;// initialization to dilate a SDF
    struct MinMaxKernel;
    struct SweepingKernel;// Private class to perform the actual concurrent sparse fast sweeping

    // Define the topology (i.e. stencil) of the neighboring grid points
    static const Coord mOffset[6];// = {{-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1}};

    // Private member data of FastSweeping
    typename GridT::Ptr mGrid1, mGrid2;//shared pointers, so using atomic counters!
    SweepMaskTreeT mSweepMask; // mask tree containing all non-boundary active voxels
    std::vector<Coord> mSweepMaskLeafOrigins; // cache of leaf node origins for mask tree
};// FastSweeping

// Static member data initialization
template <typename GridT>
const Coord FastSweeping<GridT>::mOffset[6] = {{-1,0,0},{1,0,0},
                                               {0,-1,0},{0,1,0},
                                               {0,0,-1},{0,0,1}};

template <typename GridT>
FastSweeping<GridT>::FastSweeping()
    : mGrid1(nullptr), mGrid2(nullptr)
{
}

template <typename GridT>
void FastSweeping<GridT>::clear()
{
    mGrid1.reset();
    mGrid2.reset();
    mSweepMask.clear();
}

template <typename GridT>
size_t FastSweeping<GridT>::voxelCount() const
{
    return mSweepMask.activeVoxelCount();
}

template <typename GridT>
size_t FastSweeping<GridT>::boundaryCount() const
{
    if (!mGrid1)  return size_t(0);
    return mGrid1->constTree().activeVoxelCount() - this->voxelCount();
}

template <typename GridT>
void FastSweeping<GridT>::computeSweepMaskLeafOrigins()
{
    // replace any inactive leaf nodes with tiles and voxelize any active tiles

    pruneInactive(mSweepMask);
    mSweepMask.voxelizeActiveTiles();

    using LeafManagerT = tree::LeafManager<SweepMaskTreeT>;
    using LeafT = typename SweepMaskTreeT::LeafNodeType;
    LeafManagerT leafManager(mSweepMask);

    mSweepMaskLeafOrigins.resize(leafManager.leafCount());
    leafManager.foreach(
        [&](const LeafT& leaf, size_t leafIdx)
        {
            mSweepMaskLeafOrigins[leafIdx] = leaf.origin();
        }, /*threaded=*/true, /*grainsize=*/1024
    );
}

template <typename GridT>
bool FastSweeping<GridT>::initSdf(const GridT &fogGrid, ValueT isoValue, bool isInputSdf)
{
    this->clear();
    mGrid1 = fogGrid.deepCopy();//very fast
    InitSdf kernel(*this);
    kernel.run(isoValue, isInputSdf);
    return this->isValid();
}

template <typename GridT>
template <typename OpT>
bool FastSweeping<GridT>::initExt(const GridT &fogGrid, const OpT &op, ValueT isoValue, bool isInputSdf)
{
    this->clear();
    mGrid1 = fogGrid.deepCopy();//very fast
    mGrid2 = fogGrid.deepCopy();//very fast
    InitExt<OpT> kernel(*this);
    kernel.run(isoValue, op, isInputSdf);
    return this->isValid();
}

template <typename GridT>
bool FastSweeping<GridT>::initDilate(const GridT &sdfGrid, int dilate, NearestNeighbors nn)
{
    this->clear();
    mGrid1 = sdfGrid.deepCopy();//very fast
    DilateKernel kernel(*this);
    kernel.run(dilate, nn);
    return this->isValid();
}

template <typename GridT>
template<typename MaskTreeT>
bool FastSweeping<GridT>::initMask(const GridT &sdfGrid, const Grid<MaskTreeT> &mask, bool ignoreActiveTiles)
{
    this->clear();
    mGrid1 = sdfGrid.deepCopy();//very fast

    if (mGrid1->transform() != mask.transform()) {
        OPENVDB_THROW(RuntimeError, "FastSweeping: Mask not aligned with the grid!");
    }

    if (mask.getGridClass() == GRID_LEVEL_SET) {
        using T = typename MaskTreeT::template ValueConverter<bool>::Type;
        typename Grid<T>::Ptr tmp = sdfInteriorMask(mask);//might have active tiles
        tmp->tree().voxelizeActiveTiles();//multi-threaded
        MaskKernel<T> kernel(*this);
        kernel.run(tmp->tree());//multi-threaded
    } else {
        if (ignoreActiveTiles || !mask.tree().hasActiveTiles()) {
            MaskKernel<MaskTreeT> kernel(*this);
            kernel.run(mask.tree());//multi-threaded
        } else {
            using T = typename MaskTreeT::template ValueConverter<ValueMask>::Type;
            T tmp(mask.tree(), false, TopologyCopy());//multi-threaded
            tmp.voxelizeActiveTiles(true);//multi-threaded
            MaskKernel<T> kernel(*this);
            kernel.run(tmp);//multi-threaded
        }
    }
    return this->isValid();
}// FastSweeping::initMaskSdf

template <typename GridT>
void FastSweeping<GridT>::sweep(int nIter, bool finalize)
{
    if (!mGrid1) {
      OPENVDB_THROW(RuntimeError, "FastSweeping::sweep called before initialization");
    }
    if (this->boundaryCount() == 0) {
        OPENVDB_THROW(RuntimeError, "FastSweeping: No boundary voxels found!");
    } else if (this->voxelCount() == 0) {
        OPENVDB_THROW(RuntimeError, "FastSweeping: No computing voxels found!");
    }

    // note: SweepingKernel is non copy-constructible, so use a deque instead of a vector
    std::deque<SweepingKernel> kernels;
    for (int i = 0; i < 4; i++)     kernels.emplace_back(*this);

    { // compute voxel slices
#ifdef BENCHMARK_FAST_SWEEPING
        util::CpuTimer timer("Computing voxel slices");
#endif

        // Exploiting nested parallelism - all voxel slice data is precomputed
        tbb::task_group tasks;
        tasks.run([&] { kernels[0].computeVoxelSlices([](const Coord &a){ return a[0]+a[1]+a[2]; });/*+++ & ---*/ });
        tasks.run([&] { kernels[1].computeVoxelSlices([](const Coord &a){ return a[0]+a[1]-a[2]; });/*++- & --+*/ });
        tasks.run([&] { kernels[2].computeVoxelSlices([](const Coord &a){ return a[0]-a[1]+a[2]; });/*+-+ & -+-*/ });
        tasks.run([&] { kernels[3].computeVoxelSlices([](const Coord &a){ return a[0]-a[1]-a[2]; });/*+-- & -++*/ });
        tasks.wait();

#ifdef BENCHMARK_FAST_SWEEPING
        timer.stop();
#endif
    }

    // perform nIter iterations of bi-directional sweeping in all directions
    for (int i = 0; i < nIter; ++i) {
        for (SweepingKernel& kernel : kernels)   kernel.sweep();
    }

    if (finalize) {
#ifdef BENCHMARK_FAST_SWEEPING
      util::CpuTimer timer("Computing extrema values");
#endif
      MinMaxKernel kernel;
      auto e = kernel.run(*mGrid1);//multi-threaded
      //auto e = extrema(mGrid->beginValueOn());// 100x slower!!!!
#ifdef BENCHMARK_FAST_SWEEPING
      std::cerr << "Min = " << e.min() << " Max = " << e.max() << std::endl;
      timer.restart("Changing asymmetric background value");
#endif
      changeAsymmetricLevelSetBackground(mGrid1->tree(), e.max(), e.min());//multi-threaded

#ifdef BENCHMARK_FAST_SWEEPING
      timer.stop();
#endif
    }
}// FastSweeping::sweep

/// Private class of FastSweeping to quickly compute the extrema
/// values of the active voxels in the leaf nodes. Several orders
/// of magnitude faster than tools::extrema!
template <typename GridT>
struct FastSweeping<GridT>::MinMaxKernel
{
    using LeafMgr = tree::LeafManager<const TreeT>;
    using LeafRange = typename LeafMgr::LeafRange;
    MinMaxKernel() : mMin(std::numeric_limits<ValueT>::max()), mMax(-mMin) {}
    MinMaxKernel(MinMaxKernel& other, tbb::split) : mMin(other.mMin), mMax(other.mMax) {}
    math::MinMax<ValueT> run(const GridT &grid)
    {
        LeafMgr mgr(grid.tree());// super fast
        tbb::parallel_reduce(mgr.leafRange(), *this);
        return math::MinMax<ValueT>(mMin, mMax);
    }
    void operator()(const LeafRange& r)
    {
        for (auto leafIter = r.begin(); leafIter; ++leafIter) {
            for (auto voxelIter = leafIter->beginValueOn(); voxelIter; ++voxelIter) {
                const ValueT v = *voxelIter;
                if (v < mMin) mMin = v;
                if (v > mMax) mMax = v;
            }
        }
    }
    void join(const MinMaxKernel& other)
    {
        if (other.mMin < mMin) mMin = other.mMin;
        if (other.mMax > mMax) mMax = other.mMax;
    }
    ValueT mMin, mMax;
};//MinMaxKernel

////////////////////////////////////////////////////////////////////////////////

/// Private class of FastSweeping to perform multi-threaded initialization
template <typename GridT>
struct FastSweeping<GridT>::DilateKernel
{
    using LeafRange = typename tree::LeafManager<TreeT>::LeafRange;
    DilateKernel(FastSweeping &parent)
        : mParent(&parent), mBackground(parent.mGrid1->background())
    {
    }
    DilateKernel(const DilateKernel &parent) = default;// for tbb::parallel_for
    DilateKernel& operator=(const DilateKernel&) = delete;

    void run(int dilation, NearestNeighbors nn)
    {
#ifdef BENCHMARK_FAST_SWEEPING
        util::CpuTimer timer("Construct LeafManager");
#endif
        tree::LeafManager<TreeT> mgr(mParent->mGrid1->tree());// super fast

#ifdef BENCHMARK_FAST_SWEEPING
        timer.restart("Changing background value");
#endif
        static const ValueT Unknown = std::numeric_limits<ValueT>::max();
        changeLevelSetBackground(mgr, Unknown);//multi-threaded

 #ifdef BENCHMARK_FAST_SWEEPING
        timer.restart("Dilating and updating mgr (parallel)");
        //timer.restart("Dilating and updating mgr (serial)");
#endif

        const int delta = 5;
        for (int i=0, d = dilation/delta; i<d; ++i) dilateActiveValues(mgr, delta, nn, IGNORE_TILES);
        dilateActiveValues(mgr, dilation % delta, nn, IGNORE_TILES);
        //for (int i=0, n=5, d=dilation/n; i<d; ++i) dilateActiveValues(mgr, n, nn, IGNORE_TILES);
        //dilateVoxels(mgr, dilation, nn);

#ifdef BENCHMARK_FAST_SWEEPING
        timer.restart("Initializing grid and sweep mask");
#endif

        mParent->mSweepMask.clear();
        mParent->mSweepMask.topologyUnion(mParent->mGrid1->constTree());

        using LeafManagerT = tree::LeafManager<typename GridT::TreeType>;
        using LeafT = typename GridT::TreeType::LeafNodeType;
        LeafManagerT leafManager(mParent->mGrid1->tree());

        leafManager.foreach(
            [&](LeafT& leaf, size_t /*leafIdx*/)
            {
                static const ValueT Unknown = std::numeric_limits<ValueT>::max();
                const ValueT background = mBackground;//local copy
                auto* maskLeaf = mParent->mSweepMask.probeLeaf(leaf.origin());
                assert(maskLeaf);
                for (auto voxelIter = leaf.beginValueOn(); voxelIter; ++voxelIter) {
                    const ValueT value = *voxelIter;
                    if (math::Abs(value) < background) {
                        // disable boundary voxels from the mask tree
                        maskLeaf->setValueOff(voxelIter.pos());
                    } else {
                        voxelIter.setValue(value > 0 ? Unknown : -Unknown);
                    }
                }
            }
        );

        // cache the leaf node origins for fast lookup in the sweeping kernels

        mParent->computeSweepMaskLeafOrigins();

#ifdef BENCHMARK_FAST_SWEEPING
        timer.stop();
#endif
    }

    // Private member data of DilateKernel
    FastSweeping *mParent;
    const ValueT  mBackground;
};// DilateKernel

////////////////////////////////////////////////////////////////////////////////
template <typename GridT>
struct FastSweeping<GridT>::InitSdf
{
    using LeafRange = typename tree::LeafManager<TreeT>::LeafRange;
    InitSdf(FastSweeping &parent): mParent(&parent),
      mGrid1(parent.mGrid1.get()), mIsoValue(0), mAboveSign(0) {}
    InitSdf(const InitSdf&) = default;// for tbb::parallel_for
    InitSdf& operator=(const InitSdf&) = delete;

    void run(ValueT isoValue, bool isInputSdf)
    {
        mIsoValue   = isoValue;
        mAboveSign  = isInputSdf ? ValueT(1) : ValueT(-1);
        TreeT &tree = mGrid1->tree();//sdf
        const bool hasActiveTiles = tree.hasActiveTiles();

        if (isInputSdf && hasActiveTiles) {
          OPENVDB_THROW(RuntimeError, "FastSweeping: A SDF should not have active tiles!");
        }

#ifdef BENCHMARK_FAST_SWEEPING
        util::CpuTimer  timer("Initialize voxels");
#endif
        mParent->mSweepMask.clear();
        mParent->mSweepMask.topologyUnion(mParent->mGrid1->constTree());

        {// Process all voxels
          tree::LeafManager<TreeT> mgr(tree, 1);// we need one auxiliary buffer
          tbb::parallel_for(mgr.leafRange(32), *this);//multi-threaded
          mgr.swapLeafBuffer(1);//swap voxel values
        }

#ifdef BENCHMARK_FAST_SWEEPING
        timer.restart("Initialize tiles - new");
#endif
        // Process all tiles
        tree::NodeManager<TreeT, TreeT::RootNodeType::LEVEL-1> mgr(tree);
        mgr.foreachBottomUp(*this);//multi-threaded
        tree.root().setBackground(std::numeric_limits<ValueT>::max(), false);
        if (hasActiveTiles) tree.voxelizeActiveTiles();//multi-threaded

        // cache the leaf node origins for fast lookup in the sweeping kernels

        mParent->computeSweepMaskLeafOrigins();
    }
    void operator()(const LeafRange& r) const
    {
        SweepMaskAccT sweepMaskAcc(mParent->mSweepMask);
        math::GradStencil<GridT, false> stencil(*mGrid1);
        const ValueT isoValue = mIsoValue, above = mAboveSign*std::numeric_limits<ValueT>::max();//local copy
        const ValueT h = mAboveSign*static_cast<ValueT>(mGrid1->voxelSize()[0]);//Voxel size
        for (auto leafIter = r.begin(); leafIter; ++leafIter) {
            ValueT* sdf = leafIter.buffer(1).data();
            for (auto voxelIter = leafIter->beginValueAll(); voxelIter; ++voxelIter) {
                const ValueT value = *voxelIter;
                const bool isAbove = value > isoValue;
                if (!voxelIter.isValueOn()) {// inactive voxels
                  sdf[voxelIter.pos()] = isAbove ? above : -above;
                } else {// active voxels
                  const Coord ijk = voxelIter.getCoord();
                  stencil.moveTo(ijk, value);
                  const auto mask = stencil.intersectionMask( isoValue );
                  if (mask.none()) {// most common case
                    sdf[voxelIter.pos()] = isAbove ? above : -above;
                  } else {// compute distance to iso-surface
                    // disable boundary voxels from the mask tree
                    sweepMaskAcc.setValueOff(ijk);
                    const ValueT delta = value - isoValue;//offset relative to iso-value
                    if (math::isApproxZero(delta)) {//voxel is on the iso-surface
                      sdf[voxelIter.pos()] = 0;
                    } else {//voxel is neighboring the iso-surface
                      ValueT sum = 0;
                      for (int i=0; i<6;) {
                        ValueT d = std::numeric_limits<ValueT>::max(), d2;
                        if (mask.test(i++)) d = math::Abs(delta/(value-stencil.getValue(i)));
                        if (mask.test(i++)) {
                          d2 = math::Abs(delta/(value-stencil.getValue(i)));
                          if (d2 < d) d = d2;
                        }
                        if (d < std::numeric_limits<ValueT>::max()) sum += 1/(d*d);
                      }
                      sdf[voxelIter.pos()] = isAbove ? h / math::Sqrt(sum) : -h / math::Sqrt(sum);
                    }// voxel is neighboring the iso-surface
                  }// intersecting voxels
                }// active voxels
            }// loop over voxels
        }// loop over leaf nodes
    }// operator(const LeafRange& r)
    template<typename RootOrInternalNodeT>
    void operator()(const RootOrInternalNodeT& node) const
    {
        const ValueT isoValue = mIsoValue, above = mAboveSign*std::numeric_limits<ValueT>::max();
        for (auto it = node.cbeginValueAll(); it; ++it) {
          ValueT& v = const_cast<ValueT&>(*it);
          v = v > isoValue ? above : -above;
        }//loop over all tiles
    }
    // Public member data
    FastSweeping *mParent;
    GridT        *mGrid1;//raw pointer, i.e. lock free
    ValueT        mIsoValue;
    ValueT        mAboveSign;//sign of distance values above the iso-value
};// InitSdf

/// Private class of FastSweeping to perform multi-threaded initialization
template <typename GridT>
template <typename OpT>
struct FastSweeping<GridT>::InitExt
{
    using LeafRange = typename tree::LeafManager<TreeT>::LeafRange;
    using OpPoolT = tbb::enumerable_thread_specific<OpT>;
    InitExt(FastSweeping &parent) : mParent(&parent),
      mOpPool(nullptr), mGrid1(parent.mGrid1.get()),
      mGrid2(parent.mGrid2.get()), mIsoValue(0), mAboveSign(0) {}
    InitExt(const InitExt&) = default;// for tbb::parallel_for
    InitExt& operator=(const InitExt&) = delete;
    void run(ValueT isoValue, const OpT &opPrototype, bool isInputSdf)
    {
        static_assert(std::is_same<ValueT, decltype(opPrototype(Vec3d(0)))>::value, "Invalid return type of functor");
        if (mGrid2 == nullptr) {
          OPENVDB_THROW(RuntimeError, "FastSweeping::InitExt expected an extension grid!");
        }

        mAboveSign  = isInputSdf ? ValueT(1) : ValueT(-1);
        mIsoValue = isoValue;
        TreeT &tree1 = mGrid1->tree(), &tree2 = mGrid2->tree();
        const bool hasActiveTiles = tree1.hasActiveTiles();//very fast

        if (isInputSdf && hasActiveTiles) {
          OPENVDB_THROW(RuntimeError, "FastSweeping: A SDF should not have active tiles!");
        }

#ifdef BENCHMARK_FAST_SWEEPING
        util::CpuTimer  timer("Initialize voxels");
#endif

        mParent->mSweepMask.clear();
        mParent->mSweepMask.topologyUnion(mParent->mGrid1->constTree());

        {// Process all voxels
          // Define thread-local operators
          OpPoolT opPool(opPrototype);
          mOpPool = &opPool;

          tree::LeafManager<TreeT> mgr(tree1, 1);// we need one auxiliary buffer
          tbb::parallel_for(mgr.leafRange(32), *this);//multi-threaded
          mgr.swapLeafBuffer(1);//swap out auxiliary buffer
        }

#ifdef BENCHMARK_FAST_SWEEPING
        timer.restart("Initialize tiles");
#endif
        {// Process all tiles
          tree::NodeManager<TreeT, TreeT::RootNodeType::LEVEL-1> mgr(tree1);
          mgr.foreachBottomUp(*this);//multi-threaded
          tree1.root().setBackground(std::numeric_limits<ValueT>::max(), false);
          tree2.root().setBackground(0, true);
          if (hasActiveTiles) {
#ifdef BENCHMARK_FAST_SWEEPING
            timer.restart("Voxelizing active tiles");
#endif
            tree1.voxelizeActiveTiles();//multi-threaded
            tree2.voxelizeActiveTiles();//multi-threaded
          }
        }

        // cache the leaf node origins for fast lookup in the sweeping kernels

        mParent->computeSweepMaskLeafOrigins();

#ifdef BENCHMARK_FAST_SWEEPING
        timer.stop();
#endif
    }
    void operator()(const LeafRange& r) const
    {
        SweepMaskAccT sweepMaskAcc(mParent->mSweepMask);
        math::GradStencil<GridT, false> stencil(*mGrid1);
        const math::Transform& xform = mGrid2->transform();
        AccT acc(mGrid2->tree());
        typename OpPoolT::reference op = mOpPool->local();
        const ValueT isoValue = mIsoValue, above = mAboveSign*std::numeric_limits<ValueT>::max();//local copy
        const ValueT h = mAboveSign*static_cast<ValueT>(mGrid1->voxelSize()[0]);//Voxel size
        for (auto leafIter = r.begin(); leafIter; ++leafIter) {
            ValueT *sdf = leafIter.buffer(1).data();
            ValueT *ext = acc.probeLeaf(leafIter->origin())->buffer().data();//should be safe!
            for (auto voxelIter = leafIter->beginValueAll(); voxelIter; ++voxelIter) {
                const ValueT value = *voxelIter;
                const bool isAbove = value > isoValue;
                if (!voxelIter.isValueOn()) {// inactive voxels
                  sdf[voxelIter.pos()] = isAbove ? above : -above;
                } else {// active voxels
                  const Coord ijk = voxelIter.getCoord();
                  stencil.moveTo(ijk, value);
                  const auto mask = stencil.intersectionMask( isoValue );
                  if (mask.none()) {// no zero-crossing neighbors, most common case
                    sdf[voxelIter.pos()] = isAbove ? above : -above;
                  } else {// compute distance to iso-surface
                    // disable boundary voxels from the mask tree
                    sweepMaskAcc.setValueOff(ijk);
                    const ValueT delta = value - isoValue;//offset relative to iso-value
                    if (math::isApproxZero(delta)) {//voxel is on the iso-surface
                      sdf[voxelIter.pos()] = 0;
                      ext[voxelIter.pos()] = op(xform.indexToWorld(ijk));
                    } else {//voxel is neighboring the iso-surface
                      ValueT sum1 = 0, sum2 = 0;
                      for (int n, i=0; i<6;) {
                        ValueT d = std::numeric_limits<ValueT>::max(), d2;
                        if (mask.test(i++)) {
                          d = math::Abs(delta/(value-stencil.getValue(i)));
                          n = i - 1;
                        }
                        if (mask.test(i++)) {
                          d2 = math::Abs(delta/(value-stencil.getValue(i)));
                          if (d2 < d) {
                            d = d2;
                            n = i - 1;
                          }
                        }
                        if (d < std::numeric_limits<ValueT>::max()) {
                          d2 = 1/(d*d);
                          sum1 += d2;
                          const Vec3R xyz(
                              static_cast<ValueT>(ijk[0])+d*static_cast<ValueT>(FastSweeping::mOffset[n][0]),
                              static_cast<ValueT>(ijk[1])+d*static_cast<ValueT>(FastSweeping::mOffset[n][1]),
                              static_cast<ValueT>(ijk[2])+d*static_cast<ValueT>(FastSweeping::mOffset[n][2]));
                          sum2 += op(xform.indexToWorld(xyz))*d2;
                        }
                      }//look over six cases
                      ext[voxelIter.pos()] = sum2 / sum1;
                      sdf[voxelIter.pos()] = isAbove ? h / math::Sqrt(sum1) : -h / math::Sqrt(sum1);
                    }// voxel is neighboring the iso-surface
                  }// intersecting voxels
                }// active voxels
            }// loop over voxels
        }// loop over leaf nodes
    }// operator(const LeafRange& r)
    template<typename RootOrInternalNodeT>
    void operator()(const RootOrInternalNodeT& node) const
    {
        const ValueT isoValue = mIsoValue, above = mAboveSign*std::numeric_limits<ValueT>::max();
        for (auto it = node.cbeginValueAll(); it; ++it) {
          ValueT& v = const_cast<ValueT&>(*it);
          v = v > isoValue ? above : -above;
        }//loop over all tiles
    }
    // Public member data
    FastSweeping *mParent;
    OpPoolT      *mOpPool;
    GridT        *mGrid1, *mGrid2;//raw pointers, i.e. lock free
    ValueT        mIsoValue;
    ValueT        mAboveSign;//sign of distance values above the iso-value
};// InitExt

/// Private class of FastSweeping to perform multi-threaded initialization
template <typename GridT>
template <typename MaskTreeT>
struct FastSweeping<GridT>::MaskKernel
{
    using LeafRange = typename tree::LeafManager<const MaskTreeT>::LeafRange;
    MaskKernel(FastSweeping &parent) : mParent(&parent),
      mGrid1(parent.mGrid1.get()) {}
    MaskKernel(const MaskKernel &parent) = default;// for tbb::parallel_for
    MaskKernel& operator=(const MaskKernel&) = delete;

    void run(const MaskTreeT &mask)
    {
#ifdef BENCHMARK_FAST_SWEEPING
        util::CpuTimer timer;
#endif
        auto &lsTree = mGrid1->tree();

        static const ValueT Unknown = std::numeric_limits<ValueT>::max();

#ifdef BENCHMARK_FAST_SWEEPING
        timer.restart("Changing background value");
#endif
        changeLevelSetBackground(lsTree, Unknown);//multi-threaded

#ifdef BENCHMARK_FAST_SWEEPING
        timer.restart("Union with mask");//multi-threaded
#endif
        lsTree.topologyUnion(mask);//multi-threaded

        // ignore active tiles since the input grid is assumed to be a level set
        tree::LeafManager<const MaskTreeT> mgr(mask);// super fast

#ifdef BENCHMARK_FAST_SWEEPING
        timer.restart("Initializing grid and sweep mask");
#endif

        mParent->mSweepMask.clear();
        mParent->mSweepMask.topologyUnion(mParent->mGrid1->constTree());

        using LeafManagerT = tree::LeafManager<SweepMaskTreeT>;
        using LeafT = typename SweepMaskTreeT::LeafNodeType;
        LeafManagerT leafManager(mParent->mSweepMask);

        leafManager.foreach(
            [&](LeafT& leaf, size_t /*leafIdx*/)
            {
                static const ValueT Unknown = std::numeric_limits<ValueT>::max();
                AccT acc(mGrid1->tree());
                // The following hack is safe due to the topoloyUnion in
                // init and the fact that ValueT is known to be a floating point!
                ValueT *data = acc.probeLeaf(leaf.origin())->buffer().data();
                for (auto voxelIter = leaf.beginValueOn(); voxelIter; ++voxelIter) {// mask voxels
                    if (math::Abs( data[voxelIter.pos()] ) < Unknown ) {
                        // disable boundary voxels from the mask tree
                        voxelIter.setValue(false);
                    }
                }
            }
        );

        // cache the leaf node origins for fast lookup in the sweeping kernels

        mParent->computeSweepMaskLeafOrigins();

#ifdef BENCHMARK_FAST_SWEEPING
        timer.stop();
#endif
    }

    // Private member data of MaskKernel
    FastSweeping *mParent;
    GridT        *mGrid1;//raw pointer, i.e. lock free
};// MaskKernel

/// @brief Private class of FastSweeping to perform concurrent fast sweeping in two directions
template <typename GridT>
struct FastSweeping<GridT>::SweepingKernel
{
    SweepingKernel(FastSweeping &parent) : mParent(&parent) { }
    SweepingKernel(const SweepingKernel&) = delete;
    SweepingKernel& operator=(const SweepingKernel&) = delete;

    /// Main method that performs concurrent bi-directional sweeps
    template<typename HashOp>
    void computeVoxelSlices(HashOp hash)
    {
#ifdef BENCHMARK_FAST_SWEEPING
        util::CpuTimer timer;
#endif

        // this mask tree only stores the active voxels in use in the sweeping, not the boundary voxels
        const SweepMaskTreeT& maskTree = mParent->mSweepMask;

        using LeafManagerT = typename tree::LeafManager<const SweepMaskTreeT>;
        using LeafT = typename SweepMaskTreeT::LeafNodeType;
        LeafManagerT leafManager(maskTree);

        ////////////////////////////////////////////////
        // compute the leaf node slices that have active voxels in them
        // the sliding window of the slices is -14 => 22 (based on an 8x8x8 leaf node),
        // but we use a larger mask window here to easily accomodate any leaf dimension.
        // the mask offset is used to be able to store this in a fixed-size byte array
        ////////////////////////////////////////////////

        constexpr int maskOffset = LeafT::DIM * 3;
        constexpr int maskRange = maskOffset * 2;

        // mark each possible slice in each leaf node that has one or more active voxels in it

        std::vector<int8_t> leafSliceMasks(leafManager.leafCount()*maskRange);

        leafManager.foreach(
            [&](const LeafT& leaf, size_t leafIdx)
            {
                const size_t leafOffset = leafIdx * maskRange;
                for (auto voxelIter = leaf.cbeginValueOn(); voxelIter; ++voxelIter) {
                    const Index voxelIdx = voxelIter.pos();
                    const Coord ijk = LeafT::offsetToLocalCoord(voxelIdx);
                    const size_t key = hash(ijk) + maskOffset;
                    leafSliceMasks[leafOffset + key] = uint8_t(1);
                }
            }
        );

        ////////////////////////////////////////////////
        // compute the voxel slice map using a thread-local-storage hash map
        // the key of the hash map is the slice index of the voxel coord (ijk.x() + ijk.y() + ijk.z())
        // the values are an array of indices for every leaf that has active voxels with this slice index
        ////////////////////////////////////////////////

        using ThreadLocalMap = std::unordered_map</*voxelSliceKey=*/int64_t, /*leafIdx=*/std::deque<size_t>>;
        tbb::enumerable_thread_specific<ThreadLocalMap> pool;

        leafManager.foreach(
            [&](const LeafT& leaf, size_t leafIdx)
            {
                ThreadLocalMap& map = pool.local();
                const Coord& origin = leaf.origin();
                const int64_t leafKey = hash(origin);
                const size_t leafOffset = leafIdx * maskRange;
                for (int sliceIdx = 0; sliceIdx < maskRange; sliceIdx++) {
                    if (leafSliceMasks[leafOffset + sliceIdx] == uint8_t(1)) {
                        const int64_t voxelSliceKey = leafKey+sliceIdx-maskOffset;
                        assert(voxelSliceKey >= 0);
                        map[voxelSliceKey].emplace_back(leafIdx);
                    }
                }
            }
        );

        ////////////////////////////////////////////////
        // combine into a single ordered map keyed by the voxel slice key
        // note that this is now stored in a map ordered by voxel slice key,
        // so sweep slices can be processed in order
        ////////////////////////////////////////////////

        for (auto poolIt = pool.begin(); poolIt != pool.end(); ++poolIt) {
            const ThreadLocalMap& map = *poolIt;
            for (const auto& it : map) {
                for (const size_t leafIdx : it.second) {
                    mVoxelSliceMap[it.first].emplace_back(leafIdx, NodeMaskPtrT());
                }
            }
        }

        ////////////////////////////////////////////////
        // extract the voxel slice keys for random access into the map
        ////////////////////////////////////////////////

        mVoxelSliceKeys.reserve(mVoxelSliceMap.size());
        for (const auto& it : mVoxelSliceMap) {
            mVoxelSliceKeys.push_back(it.first);
        }

        ////////////////////////////////////////////////
        // allocate the node masks in parallel, as the map is populated in serial
        ////////////////////////////////////////////////

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, mVoxelSliceKeys.size()),
            [&](tbb::blocked_range<size_t>& range)
            {
                for (size_t i = range.begin(); i < range.end(); i++) {
                    const int64_t key = mVoxelSliceKeys[i];
                    for (auto& it : mVoxelSliceMap[key]) {
                        it.second = std::make_unique<NodeMaskT>();
                    }
                }
            }
        );

        ////////////////////////////////////////////////
        // each voxel slice contains a leafIdx-nodeMask pair,
        // this routine populates these node masks to select only the active voxels
        // from the mask tree that have the same voxel slice key
        // TODO: a small optimization here would be to union this leaf node mask with
        // a pre-computed one for this particular slice pattern
        ////////////////////////////////////////////////

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, mVoxelSliceKeys.size()),
            [&](tbb::blocked_range<size_t>& range)
            {
                for (size_t i = range.begin(); i < range.end(); i++) {
                    const int64_t voxelSliceKey = mVoxelSliceKeys[i];
                    LeafSliceArray& leafSliceArray = mVoxelSliceMap[voxelSliceKey];
                    for (LeafSlice& leafSlice : leafSliceArray) {
                        const size_t leafIdx = leafSlice.first;
                        NodeMaskPtrT& nodeMask = leafSlice.second;
                        const LeafT& leaf = leafManager.leaf(leafIdx);
                        const Coord& origin = leaf.origin();
                        const int64_t leafKey = hash(origin);
                        for (auto voxelIter = leaf.cbeginValueOn(); voxelIter; ++voxelIter) {
                            const Index voxelIdx = voxelIter.pos();
                            const Coord ijk = LeafT::offsetToLocalCoord(voxelIdx);
                            const int64_t key = leafKey + hash(ijk);
                            if (key == voxelSliceKey) {
                                nodeMask->setOn(voxelIdx);
                            }
                        }
                    }
                }
            }
        );
    }

    // Private struct for nearest neighbor grid points (very memory light!)
    struct NN {
        ValueT v;
        int    n;
        inline static Coord ijk(const Coord &p, int i) { return p + FastSweeping::mOffset[i]; }
        NN() : v(), n() {}
        NN(const AccT &a, const Coord &p, int i) : v(math::Abs(a.getValue(ijk(p,i)))), n(i) {}
        inline Coord operator()(const Coord &p) const { return ijk(p, n); }
        inline bool operator<(const NN &rhs) const { return v < rhs.v; }
        inline operator bool() const { return v < ValueT(1000); }
    };

    void sweep()
    {
        TreeT* tree2 = mParent->mGrid2 ? &mParent->mGrid2->tree() : nullptr;

        const ValueT h = static_cast<ValueT>(mParent->mGrid1->voxelSize()[0]);
        const ValueT sqrt2h = math::Sqrt(ValueT(2))*h;

        const std::vector<Coord>& leafNodeOrigins = mParent->mSweepMaskLeafOrigins;

        int64_t voxelSliceIndex(0);

        auto sweepOp = [&](const tbb::blocked_range<size_t>& range)
        {
            using LeafT = typename GridT::TreeType::LeafNodeType;

            AccT acc1(mParent->mGrid1->tree());
            auto acc2 = std::unique_ptr<AccT>(tree2 ? new AccT(*tree2) : nullptr);
            ValueT absV, sign, update, D;
            NN d1, d2, d3;//distance values and coordinates of closest neighbor points

            const LeafSliceArray& leafSliceArray = mVoxelSliceMap[voxelSliceIndex];

            // Solves Goudonov's scheme: [x-d1]^2 + [x-d2]^2  + [x-d3]^2 = h^2
            // where [X] = (X>0?X:0) and ai=min(di+1,di-1)
            for (size_t i = range.begin(); i < range.end(); ++i) {

                // iterate over all leafs in the slice and extract the leaf
                // and node mask for each slice pattern

                const LeafSlice& leafSlice = leafSliceArray[i];
                const size_t leafIdx = leafSlice.first;
                const NodeMaskPtrT& nodeMask = leafSlice.second;

                const Coord& origin = leafNodeOrigins[leafIdx];

                Coord ijk;
                for (auto indexIter = nodeMask->beginOn(); indexIter; ++indexIter) {

                    // Get coordinate of center point of the FD stencil
                    ijk = origin + LeafT::offsetToLocalCoord(indexIter.pos());

                    // Find the closes neighbors in the three axial directions
                    d1 = std::min(NN(acc1, ijk, 0), NN(acc1, ijk, 1));
                    d2 = std::min(NN(acc1, ijk, 2), NN(acc1, ijk, 3));
                    d3 = std::min(NN(acc1, ijk, 4), NN(acc1, ijk, 5));

                    if (!(d1 || d2 || d3)) continue;//no valid neighbors

                    // Get the center point of the FD stencil (assumed to be an active voxel)
                    // Note this const_cast is normally unsafe but by design we know the tree
                    // to be static, of floating-point type and containing active voxels only!
                    ValueT &value = const_cast<ValueT&>(acc1.getValue(ijk));

                    // Extract the sign
                    sign = value >= ValueT(0) ? ValueT(1) : ValueT(-1);

                    // Absolute value
                    absV = math::Abs(value);

                    // sort values so d1 <= d2 <= d3
                    if (d2 < d1) std::swap(d1, d2);
                    if (d3 < d2) std::swap(d2, d3);
                    if (d2 < d1) std::swap(d1, d2);

                    // Test if there is a solution depending on ONE of the neighboring voxels
                    // if d2 - d1 >= h  => d2 >= d1 + h  then:
                    // (x-d1)^2=h^2 => x = d1 + h
                    update = d1.v + h;
                    if (update <= d2.v) {
                        if (update < absV) {
                          value = sign * update;
                          if (acc2) acc2->setValue(ijk, acc2->getValue(d1(ijk)));//update ext?
                        }//update sdf?
                        continue;
                    }// one neighbor case

                    // Test if there is a solution depending on TWO of the neighboring voxels
                    // (x-d1)^2 + (x-d2)^2 = h^2
                    //D = ValueT(2) * h * h - math::Pow2(d1.v - d2.v);// = 2h^2-(d1-d2)^2
                    //if (D >= ValueT(0)) {// non-negative discriminant
                    if (d2.v <= sqrt2h + d1.v) {
                        D = ValueT(2) * h * h - math::Pow2(d1.v - d2.v);// = 2h^2-(d1-d2)^2
                        update = ValueT(0.5) * (d1.v + d2.v + std::sqrt(D));
                        if (update > d2.v && update <= d3.v) {
                            if (update < absV) {
                              value = sign * update;
                              if (acc2) {
                                d1.v -= update; d2.v -= update;
                                acc2->setValue(ijk, (acc2->getValue(d1(ijk))*d1.v +
                                                     acc2->getValue(d2(ijk))*d2.v)/(d1.v+d2.v));
                              }//update ext?
                            }//update sdf?
                            continue;
                        }//test for two neighbor case
                    }//test for non-negative determinant

                    // Test if there is a solution depending on THREE of the neighboring voxels
                    // (x-d1)^2 + (x-d2)^2  + (x-d3)^2 = h^2
                    // 3x^2 - 2(d1 + d2 + d3)x + d1^2 + d2^2 + d3^2 = h^2
                    // ax^2 + bx + c=0, a=3, b=-2(d1+d2+d3), c=d1^2 + d2^2 + d3^2 - h^2
                    const ValueT d123 = d1.v + d2.v + d3.v;
                    D = d123*d123 - ValueT(3)*(d1.v*d1.v + d2.v*d2.v + d3.v*d3.v - h * h);
                    if (D >= ValueT(0)) {// non-negative discriminant
                        update = ValueT(1.0/3.0) * (d123 + std::sqrt(D));//always passes test
                        //if (update > d3.v) {//disabled due to round-off errors
                        if (update < absV) {
                          value = sign * update;
                          if (acc2) {
                            d1.v -= update; d2.v -= update; d3.v -= update;
                            acc2->setValue(ijk, (acc2->getValue(d1(ijk))*d1.v +
                                                 acc2->getValue(d2(ijk))*d2.v +
                                                 acc2->getValue(d3(ijk))*d3.v)/(d1.v+d2.v+d3.v));
                          }//update ext?
                        }//update sdf?
                    }//test for non-negative determinant
                }//loop over coordinates
            }
        };

#ifdef BENCHMARK_FAST_SWEEPING
        util::CpuTimer timer("Forward  sweep");
#endif

        for (size_t i = 0; i < mVoxelSliceKeys.size(); i++) {
            voxelSliceIndex = mVoxelSliceKeys[i];
            tbb::parallel_for(tbb::blocked_range<size_t>(0, mVoxelSliceMap[voxelSliceIndex].size()), sweepOp);
        }

#ifdef BENCHMARK_FAST_SWEEPING
        timer.restart("Backward sweeps");
#endif
        for (size_t i = mVoxelSliceKeys.size(); i > 0; i--) {
            voxelSliceIndex = mVoxelSliceKeys[i-1];
            tbb::parallel_for(tbb::blocked_range<size_t>(0, mVoxelSliceMap[voxelSliceIndex].size()), sweepOp);
        }

#ifdef BENCHMARK_FAST_SWEEPING
        timer.stop();
#endif
    }

private:
    using NodeMaskT = typename SweepMaskTreeT::LeafNodeType::NodeMaskType;
    using NodeMaskPtrT = std::unique_ptr<NodeMaskT>;
    // using a unique ptr for the NodeMask allows for parallel allocation,
    // but makes this class not copy-constructible
    using LeafSlice = std::pair</*leafIdx=*/size_t, /*leafMask=*/NodeMaskPtrT>;
    using LeafSliceArray = std::deque<LeafSlice>;
    using VoxelSliceMap = std::map</*voxelSliceKey=*/int64_t, LeafSliceArray>;

    // Private member data of SweepingKernel
    FastSweeping *mParent;
    VoxelSliceMap mVoxelSliceMap;
    std::vector<int64_t> mVoxelSliceKeys;

};// SweepingKernel

////////////////////////////////////////////////////////////////////////////////

template<typename GridT>
typename GridT::Ptr
fogToSdf(const GridT &fogGrid,
         typename GridT::ValueType isoValue,
         int nIter)
{
    FastSweeping<GridT> fs;
    if (fs.initSdf(fogGrid, isoValue, /*isInputSdf*/false)) {
      fs.sweep(nIter);
    } else {
      return typename GridT::Ptr(nullptr);
    }
    return fs.sdfGrid();
}

template<typename GridT>
typename GridT::Ptr
sdfToSdf(const GridT &sdfGrid,
         typename GridT::ValueType isoValue,
         int nIter)
{
    FastSweeping<GridT> fs;
    fs.initSdf(sdfGrid, isoValue, /*isInputSdf*/true);
    fs.sweep(nIter);
    return fs.sdfGrid();
}


template<typename GridT, typename OpT>
typename GridT::Ptr
fogToExt(const GridT &fogGrid,
         const OpT &op,
         typename GridT::ValueType isoValue,
         int nIter)
{
  FastSweeping<GridT> fs;
  fs.initExt(fogGrid, op, isoValue, /*isInputSdf*/false);
  fs.sweep(nIter);
  return fs.extGrid();
}

template<typename GridT, typename OpT>
std::array<typename GridT::Ptr, 2>
fogToSdfAndExt(const GridT &fogGrid,
               const OpT &op,
               typename GridT::ValueType isoValue,
               int nIter)
{
  FastSweeping<GridT> fs;
  fs.initExt(fogGrid, op, isoValue, /*isInputSdf*/false);
  fs.sweep(nIter);
  return std::array<typename GridT::Ptr, 2>{{fs.sdfGrid(), fs.extGrid()}};// double-braces required in C++11 (not in C++14)
}

template<typename GridT, typename OpT>
std::array<typename GridT::Ptr, 2>
sdfToSdfAndExt(const GridT &sdfGrid,
               const OpT &op,
               typename GridT::ValueType isoValue,
               int nIter)
{
  FastSweeping<GridT> fs;
  fs.initExt(sdfGrid, op, isoValue, /*isInputSdf*/true);
  fs.sweep(nIter);
  return std::array<typename GridT::Ptr, 2>{{fs.sdfGrid(), fs.extGrid()}};// double-braces required in C++11 (not in C++14)
}

template<typename GridT, typename OpT>
typename GridT::Ptr
sdfToExt(const GridT &sdfGrid,
         const OpT &op,
         typename GridT::ValueType isoValue,
         int nIter)
{
  FastSweeping<GridT> fs;
  fs.initExt(sdfGrid, op, isoValue, /*isInputSdf*/true);
  fs.sweep(nIter);
  return fs.extGrid();
}

template<typename GridT>
typename GridT::Ptr
dilateSdf(const GridT &sdfGrid,
          int dilation,
          NearestNeighbors nn,
          int nIter)
{
    FastSweeping<GridT> fs;
    fs.initDilate(sdfGrid, dilation, nn);
    fs.sweep(nIter);
    return fs.sdfGrid();
}

template<typename GridT, typename MaskTreeT>
typename GridT::Ptr
maskSdf(const GridT &sdfGrid,
        const Grid<MaskTreeT> &mask,
        bool ignoreActiveTiles,
        int nIter)
{
    FastSweeping<GridT> fs;
    fs.initMask(sdfGrid, mask, ignoreActiveTiles);
    fs.sweep(nIter);
    return fs.sdfGrid();
}

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_FASTSWEEPING_HAS_BEEN_INCLUDED