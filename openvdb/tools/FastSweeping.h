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
#include <thread>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>
#include <tbb/parallel_invoke.h>
#include <tbb/enumerable_thread_specific.h>

#include <openvdb/math/Math.h> // for Abs() and isExactlyEqual()
#include <openvdb/math/Stencils.h> // for GradStencil
#include <openvdb/util/PagedArray.h>
#include <openvdb/tree/LeafManager.h>
#include "LevelSetUtil.h"
#include "Morphology.h"

#include "Statistics.h"
#ifdef BENCHMARK_FAST_SWEEPING
#include <openvdb/util/CpuTimer.h>
#endif

#include <openvdb/math/Stats.h>

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

    void clear();// { this->init(nullptr); }

    /// @brief Return the number of voxels that will be solved for.
    size_t voxelCount() const { return mPagedArray.size(); }

    /// @brief Return the number of voxels that defined the boundary condition.
    size_t boundaryCount() const { return mBoundaryCount; }

    /// @brief Return true if there are voxels and boundaries to solve for
    bool isValid() const { return this->voxelCount() > 0 && this->boundaryCount() > 0; }
private:

    // Private classes to initialize the grid and construct
    // mPagedArray with voxel coordinates.
    template<typename>
    struct MaskKernel;//   initialization to extand a SDF into a mask
    template<typename>
    struct InitExt;
    struct InitSdf;
    struct DilateKernel;// initialization to dilate a SDF
    struct MinMaxKernel;
    struct SweepingKernel;// Private class to perform the actual concurrent sparse fast sweeping

    using CoordArrayT = util::PagedArray<Coord, 12>;//unsorted PagedArray with a page size of 4096

    // Define the topology (i.e. stencil) of the neighboring grid points
    static const Coord mOffset[6];// = {{-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1}};

    // Private member data of FastSweeping
    typename GridT::Ptr mGrid1, mGrid2;//shared pointers, so using atomic counters!
    std::unique_ptr<Coord[]> mCoords;//sorted c-style array (this pointer is lock free)
    CoordArrayT mPagedArray;//unsorted PagedArray of voxel coordinates
    tbb::atomic<size_t> mBoundaryCount;// number of voxels defining the bounday condition
};// FastSweeping

// Static member data initialization
template <typename GridT>
const Coord FastSweeping<GridT>::mOffset[6] = {{-1,0,0},{1,0,0},
                                               {0,-1,0},{0,1,0},
                                               {0,0,-1},{0,0,1}};

template <typename GridT>
FastSweeping<GridT>::FastSweeping()
    : mGrid1(nullptr), mGrid2(nullptr), mCoords(nullptr), mPagedArray(), mBoundaryCount(0)
{
}

template <typename GridT>
void FastSweeping<GridT>::clear()
{
    mGrid1.reset();
    mGrid2.reset();
    mCoords.reset();
    mPagedArray.clear();
    mBoundaryCount = 0;
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

    // Coord has a non-empty default constructor so we use the following trick to
    // avoid initialization of the large array of Coords. The resulting
    // allocation is virtually instantaneous. Note, we use mCoords as
    // temporal storage of mPagedArray since this allows us to avoid
    // performing multiple sorts directly on mPagedArray, which has poor
    // performance due to the fact that the sorts are done with
    // respect to changing hash functions (resulting in worst-case inputs).
    mCoords.reset(reinterpret_cast<Coord*>(new char[sizeof(Coord)*mPagedArray.size()]));

    SweepingKernel kernel(*this);
    for (int i = 0; i < nIter; ++i) {
        kernel.sweep([](const Coord &a){ return a[0]+a[1]+a[2]; });//+++ & ---
        kernel.sweep([](const Coord &a){ return a[0]+a[1]-a[2]; });//++- & --+
        kernel.sweep([](const Coord &a){ return a[0]-a[1]+a[2]; });//+-+ & -+-
        kernel.sweep([](const Coord &a){ return a[0]-a[1]-a[2]; });//+-- & -++
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
    using BufferT = typename FastSweeping::CoordArrayT::ValueBuffer;//util::PagedArray<Coord, 10>::ValueBuffer;
    using PoolT = tbb::enumerable_thread_specific<BufferT>;
    DilateKernel(FastSweeping &parent)
        : mParent(&parent), mPool(nullptr), mBackground(parent.mGrid1->background())
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
        timer.restart("Initializing grid and coords");
#endif
        BufferT prototype(mParent->mPagedArray);//exemplar used for initialization
        PoolT pool(prototype);//thread local storage pool of ValueBuffers
        mPool = &pool;
        tbb::parallel_for(mgr.leafRange(32), *this);//multi-threaded
        for (auto i = pool.begin(); i != pool.end(); ++i) i->flush();

#ifdef BENCHMARK_FAST_SWEEPING
        timer.stop();
#endif
    }

    void operator()(const LeafRange& r) const
    {
        typename PoolT::reference buffer = mPool->local();
        static const ValueT Unknown = std::numeric_limits<ValueT>::max();
        const ValueT background = mBackground;//local copy
        size_t boundaryCount = 0;
        for (auto leafIter = r.begin(); leafIter; ++leafIter) {
            for (auto voxelIter = leafIter->beginValueOn(); voxelIter; ++voxelIter) {
                const ValueT value = *voxelIter;
                if (math::Abs(value) < background) {
                    ++boundaryCount;
                } else {
                    buffer.push_back(voxelIter.getCoord());
                    voxelIter.setValue(value > 0 ? Unknown : -Unknown);
                }
            }
        }
        mParent->mBoundaryCount += boundaryCount;//reduces pressure on the atomic
    }
    // Private member data of DilateKernel
    FastSweeping *mParent;
    PoolT        *mPool;
    const ValueT  mBackground;
};// DilateKernel

////////////////////////////////////////////////////////////////////////////////
template <typename GridT>
struct FastSweeping<GridT>::InitSdf
{
    using LeafRange = typename tree::LeafManager<TreeT>::LeafRange;
    using BufferT = typename FastSweeping::CoordArrayT::ValueBuffer;//util::PagedArray<Coord, 10>::ValueBuffer;
    using BufferPoolT = tbb::enumerable_thread_specific<BufferT>;
    using StencilT = math::GradStencil<GridT, false>;
    using StencilPoolT = tbb::enumerable_thread_specific<StencilT>;
    InitSdf(FastSweeping &parent): mParent(&parent), mBufferPool(nullptr),
      mStencilPool(nullptr), mGrid1(parent.mGrid1.get()), mIsoValue(0), mAboveSign(0) {}
    InitSdf(const InitSdf&) = default;// for tbb::parallel_for
    InitSdf& operator=(const InitSdf&) = delete;

    void run(ValueT isoValue, bool isInputSdf)
    {
        mIsoValue   = isoValue;
        mAboveSign  = isInputSdf ? 1 : -1;
        TreeT &tree = mGrid1->tree();//sdf
        const bool hasActiveTiles = tree.hasActiveTiles();

        if (isInputSdf && hasActiveTiles) {
          OPENVDB_THROW(RuntimeError, "FastSweeping: A SDF should not have active tiles!");
        }

#ifdef BENCHMARK_FAST_SWEEPING
        util::CpuTimer  timer("Initialize voxels");
#endif
        // Define thread-local container for coordinates
        BufferT prototype(mParent->mPagedArray);//exemplar used for initialization
        BufferPoolT bufferPool(prototype);//thread local storage pool of ValueBuffers
        mBufferPool = &bufferPool;

        // Define thread-local stencil
        StencilT stencilPrototype(*(mGrid1));
        StencilPoolT stencilPool(stencilPrototype);
        mStencilPool = &stencilPool;

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

        // Merge all coordinate buffers
        for (auto i = bufferPool.begin(); i != bufferPool.end(); ++i) i->flush();

#ifdef BENCHMARK_FAST_SWEEPING
        timer.stop();
#endif
    }
    void operator()(const LeafRange& r) const
    {
        typename BufferPoolT::reference buffer = mBufferPool->local();
        typename StencilPoolT::reference stencil = mStencilPool->local();
        const ValueT isoValue = mIsoValue, above = mAboveSign*std::numeric_limits<ValueT>::max();//local copy
        const ValueT h = mAboveSign*static_cast<ValueT>(mGrid1->voxelSize()[0]);//Voxel size
        size_t boundaryCount = 0;
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
                    buffer.push_back(ijk);
                    sdf[voxelIter.pos()] = isAbove ? above : -above;
                  } else {// compute distance to iso-surface
                    ++boundaryCount;
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
        mParent->mBoundaryCount += boundaryCount;//reduces pressure on the atomic
    }// operator(const LeafRange& r)
    template<typename RootOrInternalNodeT>
    void operator()(const RootOrInternalNodeT& node) const
    {
        const ValueT isoValue = mIsoValue, above = mAboveSign*std::numeric_limits<ValueT>::max();
        for (auto it = node.cbeginValueAll(); it; ++it) {
          ValueT& v = const_cast<ValueT&>(*it);
          v = v > isoValue ? above : -above;
          if (it.isValueOn()) {//add coordinates of active tiles to buffer
            typename BufferPoolT::reference buffer = mBufferPool->local();
            const auto bbox = CoordBBox::createCube(it.getCoord(), node.getChildDim());
            for (auto i = bbox.begin(); i; ++i) buffer.push_back(*i);
          }//active tiles
        }//loop over all tiles
    }
    // Public member data
    FastSweeping *mParent;
    BufferPoolT  *mBufferPool;
    StencilPoolT *mStencilPool;
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
    //using BufferT = typename util::PagedArray<Coord, 10>::ValueBuffer;
    using BufferT = typename FastSweeping::CoordArrayT::ValueBuffer;
    using BufferPoolT = tbb::enumerable_thread_specific<BufferT>;
    using StencilT = math::GradStencil<GridT, false>;
    using StencilPoolT = tbb::enumerable_thread_specific<StencilT>;
    using OpPoolT = tbb::enumerable_thread_specific<OpT>;
    InitExt(FastSweeping &parent) : mParent(&parent), mBufferPool(nullptr),
      mStencilPool(nullptr), mOpPool(nullptr), mGrid1(parent.mGrid1.get()),
      mGrid2(parent.mGrid2.get()), mIsoValue(0), mAboveSign(0) {}
    InitExt(const InitExt&) = default;// for tbb::parallel_for
    InitExt& operator=(const InitExt&) = delete;
    void run(ValueT isoValue, const OpT &opPrototype, bool isInputSdf)
    {
        static_assert(std::is_same<ValueT, decltype(opPrototype(Vec3d(0)))>::value, "Invalid return type of functor");
        if (mGrid2 == nullptr) {
          OPENVDB_THROW(RuntimeError, "FastSweeping::InitExt expected an extension grid!");
        }

        mAboveSign  = isInputSdf ? 1.0f : -1.0f;
        mIsoValue = isoValue;
        TreeT &tree1 = mGrid1->tree(), &tree2 = mGrid2->tree();
        const bool hasActiveTiles = tree1.hasActiveTiles();//very fast

        if (isInputSdf && hasActiveTiles) {
          OPENVDB_THROW(RuntimeError, "FastSweeping: A SDF should not have active tiles!");
        }

#ifdef BENCHMARK_FAST_SWEEPING
        util::CpuTimer  timer("Initialize voxels");
#endif
        // Define thread-local container for coordinates
        BufferT prototype(mParent->mPagedArray);//exemplar used for initialization
        BufferPoolT bufferPool(prototype);//thread local storage pool of ValueBuffers
        mBufferPool = &bufferPool;

        {// Process all voxels
          // Define thread-local stencils
          StencilT stencilPrototype(*(mGrid1));
          StencilPoolT stencilPool(stencilPrototype);
          mStencilPool = &stencilPool;

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
#ifdef BENCHMARK_FAST_SWEEPING
        timer.restart("Merging coordinates");
#endif
        // Merge all coordinate buffers
        for (auto i = bufferPool.begin(); i != bufferPool.end(); ++i) i->flush();//fast

#ifdef BENCHMARK_FAST_SWEEPING
        timer.stop();
#endif
    }
    void operator()(const LeafRange& r) const
    {
        const math::Transform& xform = mGrid2->transform();
        AccT acc(mGrid2->tree());
        typename BufferPoolT::reference buffer = mBufferPool->local();
        typename StencilPoolT::reference stencil = mStencilPool->local();
        typename OpPoolT::reference op = mOpPool->local();
        const ValueT isoValue = mIsoValue, above = mAboveSign*std::numeric_limits<ValueT>::max();//local copy
        const ValueT h = mAboveSign*static_cast<ValueT>(mGrid1->voxelSize()[0]);//Voxel size
        size_t boundaryCount = 0;
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
                    buffer.push_back(ijk);
                    sdf[voxelIter.pos()] = isAbove ? above : -above;
                  } else {// compute distance to iso-surface
                    ++boundaryCount;
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
                          const Vec3R xyz(ijk[0]+d*FastSweeping::mOffset[n][0],
                                          ijk[1]+d*FastSweeping::mOffset[n][1],
                                          ijk[2]+d*FastSweeping::mOffset[n][2]);
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
        mParent->mBoundaryCount += boundaryCount;//reduces pressure on the atomic
    }// operator(const LeafRange& r)
    template<typename RootOrInternalNodeT>
    void operator()(const RootOrInternalNodeT& node) const
    {
        const ValueT isoValue = mIsoValue, above = mAboveSign*std::numeric_limits<ValueT>::max();
        for (auto it = node.cbeginValueAll(); it; ++it) {
          ValueT& v = const_cast<ValueT&>(*it);
          v = v > isoValue ? above : -above;
          if (it.isValueOn()) {//add coordinates of active tiles to buffer
            typename BufferPoolT::reference buffer = mBufferPool->local();
            const auto bbox = CoordBBox::createCube(it.getCoord(), node.getChildDim());
            for (auto i = bbox.begin(); i; ++i) buffer.push_back(*i);
          }//active tiles
        }//loop over all tiles
    }
    // Public member data
    FastSweeping *mParent;
    BufferPoolT  *mBufferPool;
    StencilPoolT *mStencilPool;
    OpPoolT      *mOpPool;
    GridT        *mGrid1, *mGrid2;//raw pointers, i.e. lock free
    ValueT        mIsoValue;
    int           mAboveSign;//sign of distance values above the iso-value
};// InitExt

/// Private class of FastSweeping to perform multi-threaded initialization
template <typename GridT>
template <typename MaskTreeT>
struct FastSweeping<GridT>::MaskKernel
{
    using LeafRange = typename tree::LeafManager<const MaskTreeT>::LeafRange;
    using BufferT = typename FastSweeping::CoordArrayT::ValueBuffer;
    //using BufferT = typename util::PagedArray<Coord, 10>::ValueBuffer;
    using BufferPoolT = tbb::enumerable_thread_specific<BufferT>;
    MaskKernel(FastSweeping &parent) : mParent(&parent), mBufferPool(nullptr),
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
        timer.restart("Initializing grid and coords");
#endif
        // Define thread-local container for coordinates
        BufferT prototype(mParent->mPagedArray);//exemplar used for initialization
        BufferPoolT bufferPool(prototype);//thread local storage pool of ValueBuffers
        mBufferPool = &bufferPool;

        tbb::parallel_for(mgr.leafRange(32), *this);//multi-threaded
        for (auto i = bufferPool.begin(); i != bufferPool.end(); ++i) i->flush();

#ifdef BENCHMARK_FAST_SWEEPING
        timer.stop();
#endif
    }

    void operator()(const LeafRange& r) const
    {
        typename BufferPoolT::reference buffer = mBufferPool->local();
        static const ValueT Unknown = std::numeric_limits<ValueT>::max();
        AccT acc(mGrid1->tree());
        size_t boundaryCount = 0;
        for (auto leafIter = r.begin(); leafIter; ++leafIter) {// mask leafs
            // The following hack is safe due to the topoloyUnion in
            // init and the fact that ValueT is known to be a floating point!
            ValueT *data = acc.probeLeaf(leafIter->origin())->buffer().data();
            for (auto voxelIter = leafIter->cbeginValueOn(); voxelIter; ++voxelIter) {// mask voxels
                if (math::Abs( data[voxelIter.pos()] ) < Unknown ) {
                    ++boundaryCount;
                } else {
                    buffer.push_back(voxelIter.getCoord());
                }
            }
        }
        mParent->mBoundaryCount += boundaryCount;//reduces pressure on the atomic
    }

    // Private member data of MaskKernel
    FastSweeping *mParent;
    BufferPoolT  *mBufferPool;
    GridT        *mGrid1;//raw pointer, i.e. lock free
};// MaskKernel

/// @brief Private class of FastSweeping to perform concurrent fast sweeping in two directions
template <typename GridT>
struct FastSweeping<GridT>::SweepingKernel
{
    SweepingKernel(FastSweeping &parent) : mParent(&parent),
      mAcc1(parent.mGrid1->tree()), mTree2(parent.mGrid2?&parent.mGrid2->tree():nullptr),
      mVoxelSize(static_cast<ValueT>(parent.mGrid1->voxelSize()[0])) {}
    SweepingKernel(const SweepingKernel&) = default;
    SweepingKernel& operator=(const SweepingKernel&) = delete;

    /// Main method that performs concurrent bi-directional sweeps
    template<typename HashOp>
    void sweep(HashOp hash)
    {
#ifdef BENCHMARK_FAST_SWEEPING
        util::CpuTimer timer("\nConcurrent copy of coordinates");
#endif
        Coord *coords = mParent->mCoords.get();
        mParent->mPagedArray.copy(coords);

#ifdef BENCHMARK_FAST_SWEEPING
        timer.restart("Alternative init");
#endif
        assert( mParent->mPagedArray.size() < static_cast<size_t>(std::numeric_limits<uint32_t>::max()) );

        //const uint32_t count = mParent->mPagedArray.size();
        //auto tmp = std::make_unique<uint32_t[]>(count);
        //tbb::parallel_for(tbb::blocked_range<uint32_t>(0, count, 64),
        //                  [&](const tbb::blocked_range<uint32_t>& r){auto *p=&tmp[r.begin()]; for (uint32_t i = r.begin(); i < r.end(); ++i) *p++=i;});
        //if (tmp[134] != 134) std::cerr << "ERROR" << std::endl;
        //auto hashComp2 = [&](uint32_t a, uint32_t b){return hash(coords[a]) < hash(coords[b]);};
#ifdef BENCHMARK_FAST_SWEEPING
        timer.restart("Alternative sort");
#endif
        //tbb::parallel_sort(&tmp[0], &tmp[0] + mParent->voxelCount(), hashComp2);

#ifdef BENCHMARK_FAST_SWEEPING
        timer.restart("Sorting by sweep plane");
#endif
        auto hashComp = [&hash](const Coord &a, const Coord &b){return hash(a) < hash(b);};
        tbb::parallel_sort(coords, coords + mParent->voxelCount(), hashComp);

#ifdef BENCHMARK_FAST_SWEEPING
        timer.restart("Computing number of sweep planes");
#endif
        util::PagedArray<size_t, 6> planes;
        this->buildPlanes(hash, planes);

        auto range = [&](size_t i) {
          static const size_t min = 16, size = 100*std::thread::hardware_concurrency();
          const size_t grainSize = std::max((planes[i] - planes[i-1])/size, min);
          //const size_t grainSize = 1024;//2048;//512;//seems very sensitive to the grain size!
          return tbb::blocked_range<size_t>(planes[i-1], planes[i], grainSize);
        };

#if 1
#ifdef BENCHMARK_FAST_SWEEPING
        timer.restart("Forward  sweep");
#endif
        for (size_t i = 1; i < planes.size(); ++i) tbb::parallel_for(range(i), *this);
#ifdef BENCHMARK_FAST_SWEEPING
        timer.restart("Backward sweeps");
#endif
        for (size_t i = planes.size()-1; i>0; --i) tbb::parallel_for(range(i), *this);
#else
#ifdef BENCHMARK_FAST_SWEEPING
        timer.restart("Forward and backward sweeps");
#endif
        tbb::parallel_invoke([&](){for (size_t i = 1; i < planes.size(); ++i) tbb::parallel_for(range(i), *this);},
                             [&](){for (size_t i = planes.size()-1; i>0; --i) tbb::parallel_for(range(i), *this);});
#endif
#ifdef BENCHMARK_FAST_SWEEPING
        timer.stop();
#endif
    }
    template<typename HashOp, typename ArrayType>
    void buildPlanes(HashOp &hash, ArrayType &planes)
    {
        using BufferT = typename ArrayType::ValueBuffer;
        using PoolT = tbb::enumerable_thread_specific<BufferT>;
        BufferT exemplar(planes);//dummy used for initialization
        PoolT pool(exemplar);//thread local storage pool of ValueBuffers
        auto func = [&](const tbb::blocked_range<size_t>& r) {
            Coord *p = mParent->mCoords.get();
            typename PoolT::reference b = pool.local();
            for (size_t i=r.begin(); i!=r.end(); ++i) if (hash(p[i-1])!=hash(p[i])) b.push_back(i);
        };
        planes.push_back(0);
        tbb::parallel_for(tbb::blocked_range<size_t>(1, mParent->voxelCount(), 1<<12), func);// 4096
        for (auto iter=pool.begin(); iter!=pool.end(); ++iter) iter->flush();
        planes.push_back(mParent->voxelCount());
        planes.sort();
    }
    /// @brief Locally solves @f$|\nabla \phi|^2 = 1 @f$ by means
    /// of Godunov's upwind finite difference scheme
#if 0
    void operator()(const tbb::blocked_range<size_t>& range) const
    {
        static const ValueT Unknown = std::numeric_limits<ValueT>::max();
        typename GridT::Accessor acc(mParent->mGrid1->tree());
        const ValueT h = static_cast<ValueT>(mParent->mGrid1->voxelSize()[0]), h2 = h * h;
        ValueT absValue, sign, a1, a2, a3, update, D;
        Coord *coords = mParent->mCoords.get();

        // Solves Goudonov's scheme: [x-a1]^2 + [x-a2]^2  + [x-a3]^2 = h^2
        // where [X] = (X>0?X:0) and ai=min(ai+1,ai-1)
        for (size_t i=range.begin(); i!=range.end(); ++i) {

            // Get coordinate of center point of the FD stencil
            const Coord &xyz = coords[i];

            // Get the center point of the FD stencil (assumed to be an active voxel)
            // Note this const_cast is normally unsafe but by design we know the tree
            // to be static, of floating-point type and containing active voxels only!
            ValueT &value = const_cast<ValueT&>(acc.getValue(xyz));

            // Extract the sign
            sign = value >= ValueT(0) ? ValueT(1) : ValueT(-1);

            // Absolute value
            absValue = math::Abs(value);

            // Find the closes neighbors in the three axial directions
            a1 = std::min(math::Abs(acc.getValue(xyz.offsetBy(-1, 0, 0))),
                          math::Abs(acc.getValue(xyz.offsetBy( 1, 0, 0))));
            a2 = std::min(math::Abs(acc.getValue(xyz.offsetBy( 0,-1, 0))),
                          math::Abs(acc.getValue(xyz.offsetBy( 0, 1, 0))));
            a3 = std::min(math::Abs(acc.getValue(xyz.offsetBy( 0, 0,-1))),
                          math::Abs(acc.getValue(xyz.offsetBy( 0, 0, 1))));

            // sort values so a1 <= a2 <= a3
            if (a1 > a2) std::swap(a1, a2);
            if (a2 > a3) std::swap(a2, a3);
            if (a1 > a2) std::swap(a1, a2);

            if (math::isExactlyEqual(a1, Unknown)) continue;//no valid neighbors

            // Test if there is a solution depending on ONE of the neighboring voxels
            // if a2 - a1 >= h  => a2 >= a1 + h  then:
            // (x-a1)^2=h^2 => x = h + a1
            update = a1 + h;
            if (update <= a2) {
                value = sign * math::Min(update, absValue);
                continue;
            }

            // Test if there is a solution depending on TWO of the neighboring voxels
            // (x-a1)^2 + (x-a2)^2 = h^2
            D = ValueT(2) * h2 - math::Pow2(a1-a2);// = 2h^2-(a1-a2)^2
            if (D >= ValueT(0)) {// non-negative discriminant
                update = ValueT(0.5) * (a1 + a2 + std::sqrt(D));
                if (update > a2 && update <= a3) {
                    value = sign * math::Min(update, absValue);
                    continue;
                }
            }

            // Test if there is a solution depending on THREE of the neighboring voxels
            // (x-a1)^2 + (x-a2)^2  + (x-a3)^2 = h^2
            // 3x^2 - 2(a1 + a2 + a3)x + a1^2 + a2^2 + a3^2 = h^2
            // ax^2 + bx + c=0, a=3, b=-2(a1+a2+a3), c=a1^2 + a2^2 + a3^2 - h^2
            const ValueT a123 = a1 + a2 + a3;
            D = a123*a123 - ValueT(3)*(a1*a1 + a2*a2 + a3*a3 - h2);
            if (D >= ValueT(0)) {// non-negative discriminant
                update = ValueT(1.0/3.0) * (a123 + std::sqrt(D));
                value = sign * math::Min(update, absValue);
            }
        }//loop over coordinates
    }// SweepingKernel::operator()
#else
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
    void operator()(const tbb::blocked_range<size_t>& range) const
    {
      //static const ValueT Unknown = std::numeric_limits<ValueT>::max();
      //AccT acc1(*mTree1);
      //AccT *acc2 = mTree2 ? new AccT(*mTree2) : nullptr;
      auto acc2 = std::unique_ptr<AccT>(mTree2 ? new AccT(*mTree2) : nullptr);
      const ValueT h = mVoxelSize, sqrt2h = math::Sqrt(ValueT(2))*h;
      ValueT absV, sign, update, D;
      NN d1, d2, d3;//distance values and coordinates of closest neighbor points
      Coord *coords = mParent->mCoords.get();

      // Solves Goudonov's scheme: [x-d1]^2 + [x-d2]^2  + [x-d3]^2 = h^2
      // where [X] = (X>0?X:0) and ai=min(di+1,di-1)
      for (size_t i=range.begin(); i!=range.end(); ++i) {

        // Get coordinate of center point of the FD stencil
        const Coord &ijk = coords[i];

        // Find the closes neighbors in the three axial directions
        d1 = std::min(NN(mAcc1, ijk, 0), NN(mAcc1, ijk, 1));
        d2 = std::min(NN(mAcc1, ijk, 2), NN(mAcc1, ijk, 3));
        d3 = std::min(NN(mAcc1, ijk, 4), NN(mAcc1, ijk, 5));

        if (!(d1 || d2 || d3)) continue;//no valid neighbors

        // Get the center point of the FD stencil (assumed to be an active voxel)
        // Note this const_cast is normally unsafe but by design we know the tree
        // to be static, of floating-point type and containing active voxels only!
        ValueT &value = const_cast<ValueT&>(mAcc1.getValue(ijk));

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
    //delete acc2;
}// SweepingKernel::operator()
#endif
    // Private member data of SweepingKernel
    FastSweeping *mParent;
    AccT   mAcc1;
    TreeT *mTree2;//raw pointers, i.e. lock free
    const ValueT mVoxelSize;
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