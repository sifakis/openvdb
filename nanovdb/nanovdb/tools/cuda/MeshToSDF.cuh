// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/tools/cuda/MeshToSDF.cuh

    \authors Efty Sifakis and JaeHyun Lee

    \brief GPU conversion of a triangle mesh into a narrow-band signed distance field on NanoVDB
           index grids. Builds on two domain-agnostic primitives:
             - nanovdb::tools::cuda::MeshToGrid            (mesh -> narrow-band ValueOnIndex + UDF),
             - nanovdb::tools::cuda::ConnectedComponents   (generic CC labeling of such a grid).

           MeshToSDF is the entry point. It partitions the rasterized band into closed surfaces,
           signs each one independently, and composes the results:

             1  rasterize                    MeshToGrid -> band + UDF + nearest-triangle index
             2  partition                    ConnectedComponents on the UN-PRUNED band. The barrier
                                             shell glues a surface's inner and outer sides together,
                                             so each closed surface is exactly one component
             3  per surface: carve it out and sign it alone -> its own complete sign field
             4  compose by nesting parity     a point wrapped by k surfaces is inside iff k is odd
             5  fill                          extend the composed sign off the band

           Step 3 is sdf_detail::SurfaceSigner, whose stages are, in call order:

             computeDerivedTopology()  drop the surface shell, leaving the grid CC labels
             signNonBarrier()          the min-x component is the exterior; the rest is interior
             injectSignsToOriginal()   carry those signs back onto the un-pruned grid
             signBarrier()             sign the shell the first stage set aside
             fillLeafInvertMask()      \
             fillCoarseInvertMasks()    | extend the sign off the band, level by level
             fillRootInteriorMask()    /

           Partitioning first is what makes signNonBarrier's rule — the leftmost active voxel is
           outside — legitimate: it is a statement about ONE closed surface, and a carved grid is
           where it holds. A mesh with a single closed surface skips the carve, since the rasterized
           band already is that surface's band.

           A single signedSignAt() query composes the resulting sidecars into the sign at any
           coordinate. The grid is never rebuilt: everything the stages produce is a sidecar indexed
           by leaf.getValue(n), or a per-node invert mask.

           Reading order below: MeshToSDF first, then sdf_detail (SurfaceSigner, the CUDA functors
           grouped by the stage that launches them, and the query), then the method definitions.

    \warning The header file contains cuda device code so be sure
             to only include it in .cu files (or other .cuh files)
*/

#ifndef NVIDIA_TOOLS_CUDA_MESHTOSDF_CUH_HAS_BEEN_INCLUDED
#define NVIDIA_TOOLS_CUDA_MESHTOSDF_CUH_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h>
#include <nanovdb/GridHandle.h>
#include <nanovdb/cuda/DeviceBuffer.h>
#include <nanovdb/math/Proximity.h>                    // closestPointOnTriangleToPoint
#include <nanovdb/tools/cuda/ConnectedComponents.cuh>
#include <nanovdb/tools/cuda/MeshToGrid.cuh>
#include <nanovdb/tools/cuda/PruneGrid.cuh>
#include <nanovdb/util/cuda/Injection.cuh>             // InjectGridDataFunctor
#include <nanovdb/util/cuda/DeviceGridTraits.cuh>
#include <nanovdb/util/cuda/Timer.h>
#include <nanovdb/util/cuda/Util.h>                    // operatorKernel, cudaCheck

#include <memory>
#include <utility>
#include <vector>

namespace nanovdb {

namespace tools::cuda {

namespace sdf_detail { template <typename BuildT> class SurfaceSigner; }

/// @brief Convert a triangle mesh into a narrow-band signed distance field on a NanoVDB index grid,
///        correctly for any number of closed surfaces — separate objects, cavities and nesting alike.
///        See the file notes above for the pipeline.
/// @tparam BuildT Build type of the index grid (e.g. nanovdb::ValueOnIndex).
template <typename BuildT>
class MeshToSDF
{
    using GridT   = NanoGrid<BuildT>;
    using Handle  = GridHandle<nanovdb::cuda::DeviceBuffer>;
    using Buffer  = nanovdb::cuda::DeviceBuffer;
    using Signer  = sdf_detail::SurfaceSigner<BuildT>;

public:

    /// @brief Constructor. Mirrors MeshToGrid: the mesh lives on the device, the map is the
    ///        world<->index transform, and nothing runs until build().
    /// @param d_points       device vertex list of the input triangle surface (WORLD space)
    /// @param pointCount     vertex count
    /// @param d_triangles    device triangle vertex-index list
    /// @param triangleCount  triangle count
    /// @param map            affine map used for the conversion
    /// @param stream         optional CUDA stream (defaults to CUDA stream 0)
    MeshToSDF(const nanovdb::Vec3f* d_points, uint32_t pointCount,
              const nanovdb::Vec3i* d_triangles, uint32_t triangleCount,
              const nanovdb::Map& map = nanovdb::Map(), cudaStream_t stream = 0)
        : mPoints(d_points), mPointCount(pointCount)
        , mTriangles(d_triangles), mTriangleCount(triangleCount)
        , mMap(map), mStream(stream) {}

    /// @brief Toggle on and off verbose mode
    /// @param level Verbose level: 0=quiet, 1=timing
    void setVerbose(int level = 1) { mVerbose = level; }

    /// @brief Set desired width of the narrow band
    /// @param bandWidth Narrow band width in cell units
    void setNarrowBandWidth(float bandWidth = 3.f) { mBandWidth = bandWidth; }

    /// @brief Run the whole pipeline. Afterwards the accessors below describe a complete sign field
    ///        over the rasterized band, extended off it by the invert masks.
    void build();

    /// @brief The rasterized narrow band (all surfaces together), valid after build().
    const GridT* deviceGrid() const { return mGridHandle.template deviceGrid<BuildT>(); }
    /// @brief Handle owning that grid, valid after build().
    const Handle& gridHandle() const { return mGridHandle; }
    /// @brief Per-active-voxel unsigned distance in WORLD units, valid after build().
    const float* deviceUDF() const { return static_cast<const float*>(mUDF.deviceData()); }
    /// @brief Per-active-voxel nearest-triangle index (0xFFFFFFFF = none), valid after build().
    const uint32_t* deviceIndex() const { return static_cast<const uint32_t*>(mIndex.deviceData()); }
    /// @brief Per-active-voxel sign over the rasterized band (+1 outside / -1 inside), valid after
    ///        build(). Length activeVoxelCount+1, indexed by leaf.getValue(n); slot 0 = +1.
    const int8_t* deviceSign() const { return mSign; }
    /// @brief The world<->index transform the grid was built with.
    const nanovdb::Map& map() const { return mMap; }
    /// @brief The narrow-band width, in cell units, the grid was built with.
    float narrowBandWidth() const { return mBandWidth; }

    /// @brief Number of closed surfaces the band was partitioned into, valid after build().
    uint32_t surfaceCount() const { return uint32_t(mSurfaces.size()); }
    /// @brief Per-active-voxel closed-surface id in [0, surfaceCount()), on the rasterized band.
    const uint32_t* deviceSurfaceLabels() const { return mSurfaceLabels.first; }
    /// @brief Per surface: 1 iff its nesting depth is odd, so its own signs were negated.
    const std::vector<uint8_t>& nestingParity() const { return mParity; }

    /// @name Invert masks — the sign of everything the band does not cover, valid after build().
    ///       Consumed together with deviceGrid() and deviceSign() by sdf_detail::signedSignAt().
    /// @{
    const nanovdb::Mask<3>* deviceLeafInvertMask() const;
    const nanovdb::Mask<4>* deviceLowerInvertMask() const;
    const nanovdb::Mask<5>* deviceUpperInvertMask() const;
    const uint8_t*          deviceRootInterior() const;
    nanovdb::Coord          rootTileMin() const;
    nanovdb::Coord          rootTileDims() const;
    /// @}

    /// @brief The signer that produced surface @a i's own field, for tests and debugging.
    Signer& surfaceSigner(uint32_t i) const { return *mSurfaces[i].signer; }
    /// @brief Surface @a i's carved band, or the rasterized band when there is only one surface.
    const Handle& surfaceGridHandle(uint32_t i) const;
    /// @brief Surface @a i's barrier-pruned grid (the connected-components input).
    const Handle& derivedGridHandle(uint32_t i) const { return mSurfaces[i].derived; }
    /// @brief Surface @a i's per-active-voxel component labels, on its derived grid.
    const uint32_t* deviceComponentLabels(uint32_t i) const { return mSurfaces[i].ccLabels.first; }
    /// @brief How many components surface @a i's derived grid was labeled into.
    uint64_t componentCount(uint32_t i) const { return mSurfaces[i].ccLabels.second; }
    /// @brief Surface @a i's nearest-triangle index sidecar, re-indexed onto its carved band.
    const uint32_t* surfaceIndex(uint32_t i) const;

private:

    /// @brief Everything the pipeline produces for ONE closed surface: that surface's band as a
    ///        stand-alone grid, plus the complete sign field it would have if it were the only object
    ///        in the scene. Nothing here knows that other surfaces exist, which is exactly what makes
    ///        a single global exterior seed valid again.
    struct SurfaceField {
        Handle  subGrid;   // this surface's band, carved out of the rasterized one. EMPTY when the
                           // mesh has a single closed surface — that band is then used as-is.
        Buffer  subUdf;    // udf / nearest-triangle index re-indexed onto subGrid (carving renumbers
        Buffer  subIndex;  // the value slots). Both empty in that same single-surface case.
        Handle  derived;   // barrier-pruned copy of the surface grid = connected-components input
        std::unique_ptr<ConnectedComponents<BuildT>> cc;
        std::pair<uint32_t*, uint64_t>               ccLabels{nullptr, 0};
        std::unique_ptr<Signer>                      signer;
    };

    void rasterize();                    // step 1
    void partition();                    // step 2
    void signSurface(uint32_t surface);  // step 3, once per closed surface
    void composeByInclusion();           // step 4
    void fillOnOriginal();               // step 5

    // Surface i's grid / sidecars: its own carved band, or the rasterized band when uncarved.
    const GridT*    surfaceGrid(uint32_t i) const;
    const float*    surfaceUdf(uint32_t i) const;

    const nanovdb::Vec3f* mPoints{nullptr};
    uint32_t              mPointCount{0};
    const nanovdb::Vec3i* mTriangles{nullptr};
    uint32_t              mTriangleCount{0};
    nanovdb::Map          mMap{};
    cudaStream_t          mStream{0};
    int                   mVerbose{0};
    float                 mBandWidth{3.f};

    Handle mGridHandle;   // step 1: the rasterized narrow band, all surfaces together
    Buffer mUDF, mIndex;  // its per-active-voxel unsigned distance and nearest-triangle index

    std::unique_ptr<ConnectedComponents<BuildT>> mSurfaceCC;                   // step 2
    std::pair<uint32_t*, uint64_t>               mSurfaceLabels{nullptr, 0};

    std::vector<SurfaceField> mSurfaces;   // step 3, one per closed surface

    std::vector<uint8_t> mParity;          // step 4: per surface, 1 iff its nesting depth is odd
    Buffer               mComposedSign;    // gathered signs on the rasterized band. EMPTY when a lone
                                           // uncarved surface's own array is used instead — which is
                                           // also how fillOnOriginal knows its fill is already done.
    int8_t*              mSign{nullptr};   // the signs every later stage reads

    std::unique_ptr<Signer> mOrigSigner;   // step 5; empty when surfaces[0]'s fill is adopted
    Signer*                 mFinalSigner{nullptr};

}; // tools::cuda::MeshToSDF<BuildT>

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace sdf_detail {

/// @brief Signs ONE closed surface: the stages that turn a narrow-band ValueOnIndex grid holding a
///        single closed surface, plus its UDF sidecar, into a complete sign field for that surface —
///        band signs plus the per-level invert masks that extend them off the band. Every rule here
///        assumes exactly one closed surface (see the file notes); MeshToSDF partitions and drives it.
/// @tparam BuildT Build type of the index grid (e.g. nanovdb::ValueOnIndex).
template <typename BuildT>
class SurfaceSigner
{
    using GridT = NanoGrid<BuildT>;

public:

    /// @brief Constructor
    /// @param stream optional CUDA stream (defaults to CUDA stream 0)
    SurfaceSigner(cudaStream_t stream = 0) : mStream(stream), mTimer(stream) {}

    /// @brief Toggle on and off verbose mode
    /// @param level Verbose level: 0=quiet, 1=timing
    void setVerbose(int level = 1) { mVerbose = level; }

    /// @brief Prune the surface/barrier shell of a narrow-band UDF index grid, producing a clean
    ///        ValueOnIndex grid of the non-barrier voxels (the input to connected components).
    ///        A voxel is dropped iff udf^2 < 0.75·voxelSize^2 (within √3/2 voxels of the surface).
    /// @param d_srcGrid device narrow-band ValueOnIndex grid
    /// @param d_udf     device UDF sidecar (WORLD units), indexed by leaf.getValue(n); slot 0 = background
    /// @param voxelSize world-space voxel size (to convert the √3/2-voxel barrier into world units)
    /// @return a handle to the derived (barrier-pruned) ValueOnIndex grid
    template <typename BufferT = nanovdb::cuda::DeviceBuffer>
    GridHandle<BufferT> computeDerivedTopology(const GridT* d_srcGrid, const float* d_udf,
                                               float voxelSize, const BufferT& buffer = BufferT());

    /// @brief Sign the non-barrier voxels of a CC-labeled grid: the component containing the grid's
    ///        minimum-x active voxel is the exterior (+); every other component is interior (-).
    ///        Convention: +outside / -inside.
    ///
    ///        The rule is a statement about ONE closed surface, and holds only on a grid that carries
    ///        one — the leftmost voxel of a lone closed surface is necessarily outside it. Several
    ///        objects at once would need one seed each, which is why the caller partitions the band
    ///        into closed surfaces first and runs this per surface (see the header notes above).
    /// @param d_grid        the CC-labeled (derived) device grid
    /// @param d_voxelLabel  per-active-voxel component-label sidecar for @a d_grid (from
    ///                      ConnectedComponents::getVoxelLabelsAndCount()), indexed by leaf.getValue(n).
    void signNonBarrier(const GridT* d_grid, const uint32_t* d_voxelLabel);

    /// @brief Carry the derived-grid signs (from signNonBarrier) back onto the original grid. The
    ///        derived grid is the barrier-pruned subset of the original, so the injection covers all
    ///        non-barrier voxels; barrier voxels (present only in the original) keep the sentinel 0
    ///        ("unsigned barrier") for step 5 to fill. Requires signNonBarrier() first.
    /// @param d_origGrid    the original (pre-prune) grid — injection target.
    /// @param d_derivedGrid the barrier-pruned grid that was signed.
    void injectSignsToOriginal(const GridT* d_origGrid, const GridT* d_derivedGrid);

    /// @brief Sign every barrier voxel (sign == 0) of the original grid, completing the sign field so
    ///        no sentinel-0 voxel remains. Faithful mirror of OpenVDB MeshToVolume.h
    ///        ComputeIntersectingVoxelSign: a barrier voxel is exterior (+1) iff some exterior (+1)
    ///        neighbor's nearest triangle places it on the same side of the surface, else interior (-1).
    ///        Requires injectSignsToOriginal() first (it reads those signs as the anchor snapshot); the
    ///        completed signs land in a new buffer exposed via deviceSignedVoxelSign().
    /// @param d_grid      the original (post-injection) grid being signed.
    /// @param d_index     nearest-triangle-index sidecar of the original grid (uint32; 0xFFFFFFFF = none).
    /// @param d_points    device mesh vertices (WORLD space).
    /// @param d_triangles device triangle vertex-index list.
    /// @param map         the world<->index transform used to build the grid.
    void signBarrier(const GridT* d_grid, const uint32_t* d_index,
                     const nanovdb::Vec3f* d_points, const nanovdb::Vec3i* d_triangles,
                     const nanovdb::Map& map);

    /// @brief Step 6 (chunk A, leaf level): build the per-leaf invert masks that sign the INACTIVE
    ///        voxels inside materialized leaves — bit ON => interior (-background), bit OFF =>
    ///        exterior (+background, the default). Floods interior signs from the interior active
    ///        band through each leaf's inactive voxels (active voxels are walls). Needs
    ///        completed signs (from signBarrier(), or supplied via @a d_sign). Coarser levels (lower/upper/
    ///        root) are later pipeline chunks.
    /// @param d_grid the original (fully signed) device grid.
    /// @param d_sign optional per-active-voxel sign array for @a d_grid, overriding the one this
    ///               object computed. Lets the fill run on a grid it did not sign — a per-component
    ///               sub-grid, or the original grid after the signs were recomposed.
    void fillLeafInvertMask(const GridT* d_grid, const int8_t* d_sign = nullptr);

    /// @brief Step 6 (chunk B, lower + upper levels): fill the coarse invert masks that sign the
    ///        CHILDLESS child slots of lower and upper internal nodes (bit ON => that tile is
    ///        interior / -background). Bottom-up per plan §7a: leaf faces seed lower/upper tiles ->
    ///        flood lower -> lower faces seed upper tiles -> flood upper. Root-level tiles and the
    ///        topology rebuild are chunk C. Requires fillLeafInvertMask() first.
    /// @param d_grid the original (fully signed) device grid.
    /// @param d_sign optional per-active-voxel sign array for @a d_grid, overriding the one this
    ///               object computed. Lets the fill run on a grid it did not sign — a per-component
    ///               sub-grid, or the original grid after the signs were recomposed.
    void fillCoarseInvertMasks(const GridT* d_grid, const int8_t* d_sign = nullptr);

    /// @brief Step 6 (chunk C, root level): build the root-interior SIDECAR that signs the deep
    ///        interior beyond any upper node. A small P×Q×R cell array over the grid's root-tile
    ///        bounding range (one uint8 per 4096^3 root region): pre-existing root entries are walls,
    ///        faces from ALL THREE levels seed interior evidence into abutting ABSENT cells, and a
    ///        multi-seed flood fills enclosed interiors. The GRID IS NOT MUTATED — queries consult the
    ///        sidecar via signedSignAt(). Requires fillCoarseInvertMasks() first.
    /// @param d_grid the original (fully signed) device grid.
    /// @param d_sign optional per-active-voxel sign array for @a d_grid, overriding the one this
    ///               object computed. Lets the fill run on a grid it did not sign — a per-component
    ///               sub-grid, or the original grid after the signs were recomposed.
    void fillRootInteriorMask(const GridT* d_grid, const int8_t* d_sign = nullptr);

    /// @brief Device pointer to the per-active-voxel sign array (+1 exterior / -1 interior),
    ///        valid after signNonBarrier(). Length activeVoxelCount+1, indexed by leaf.getValue(n);
    ///        slot 0 is the background (+1).
    int8_t* deviceVoxelSign() { return static_cast<int8_t*>(mVoxelSign.deviceData()); }

    /// @brief Representative (slot) of the exterior component, valid after signNonBarrier().
    uint64_t exteriorRepresentative() const { return mExteriorRep; }

    /// @brief Device pointer to the per-active-voxel sign array on the ORIGINAL grid, valid after
    ///        injectSignsToOriginal(). Length origActiveVoxelCount+1, indexed by leaf.getValue(n);
    ///        slot 0 = background (+1); non-barrier voxels carry +1/-1; barrier voxels carry the
    ///        sentinel 0 (still unsigned, awaiting step 5).
    int8_t* deviceOriginalVoxelSign() { return static_cast<int8_t*>(mOriginalVoxelSign.deviceData()); }

    /// @brief Device pointer to the fully-signed per-active-voxel array on the ORIGINAL grid, valid
    ///        after signBarrier(): every active voxel is +1 (exterior) or -1 (interior), no sentinel-0
    ///        remains. Length origActiveVoxelCount+1, indexed by leaf.getValue(n); slot 0 = +1.
    int8_t* deviceSignedVoxelSign() { return static_cast<int8_t*>(mSignedVoxelSign.deviceData()); }

    /// @brief Device pointer to the per-leaf invert masks (nodeCount[0] × Mask<3>), valid after
    ///        fillLeafInvertMask(). For each materialized leaf, bit n ON means the INACTIVE voxel n
    ///        is interior (-background); OFF means exterior (+background). Bits of active voxels are
    ///        always OFF (their sign lives in the sign sidecar).
    nanovdb::Mask<3>* deviceLeafInvertMask() { return static_cast<nanovdb::Mask<3>*>(mLeafInvertMask.deviceData()); }

    /// @brief Device pointer to the lower-level invert masks (nodeCount[1] × Mask<4>), valid after
    ///        fillCoarseInvertMasks(). Bit n ON means the CHILDLESS lower slot n (an 8^3-voxel tile)
    ///        is interior; OFF means exterior. Refined slots (childMask ON) carry no invert bit.
    nanovdb::Mask<4>* deviceLowerInvertMask() { return static_cast<nanovdb::Mask<4>*>(mLowerInvertMask.deviceData()); }

    /// @brief Device pointer to the upper-level invert masks (nodeCount[2] × Mask<5>), valid after
    ///        fillCoarseInvertMasks(). Bit n ON means the CHILDLESS upper slot n (a 128^3-voxel tile)
    ///        is interior; OFF means exterior. Refined slots carry no invert bit.
    nanovdb::Mask<5>* deviceUpperInvertMask() { return static_cast<nanovdb::Mask<5>*>(mUpperInvertMask.deviceData()); }

    /// @brief Device pointer to the root-interior sidecar (rootTileDims() cells, one uint8 per 4096^3
    ///        root region; 1 = deep interior), valid after fillRootInteriorMask(). Cell (i,j,k) covers
    ///        the root region at (rootTileMin()+(i,j,k))*4096. Cells outside the array are exterior.
    uint8_t* deviceRootInterior() { return static_cast<uint8_t*>(mRootInterior.deviceData()); }

    /// @brief Origin of the root-cell array, in 4096-tile units (valid after fillRootInteriorMask()).
    nanovdb::Coord rootTileMin() const { return mRootTileMin; }

    /// @brief Dimensions P×Q×R of the root-cell array (valid after fillRootInteriorMask()).
    nanovdb::Coord rootTileDims() const { return mRootDims; }

private:

    // Shorthands for the two device-grid queries the stages keep asking for.
    static uint32_t leafCountOf(const GridT* g) {
        return util::cuda::DeviceGridTraits<BuildT>::getTreeData(g).mNodeCount[0];
    }
    static uint64_t activeCountOf(const GridT* g) {
        return util::cuda::DeviceGridTraits<BuildT>::getActiveVoxelCount(g);
    }

    cudaStream_t                 mStream{0};
    util::cuda::Timer            mTimer;
    int                          mVerbose{0};

    uint64_t                     mExteriorRep{0};  // the exterior component (see exteriorRepresentative())
    nanovdb::cuda::DeviceBuffer  mVoxelSign;       // (derived activeVoxelCount+1) × int8_t: +1 ext / -1 int
    nanovdb::cuda::DeviceBuffer  mOriginalVoxelSign; // (orig activeVoxelCount+1) × int8_t: +1/-1 non-barrier, 0 barrier
    nanovdb::cuda::DeviceBuffer  mSignedVoxelSign;   // (orig activeVoxelCount+1) × int8_t: +1/-1 everywhere (barriers signed)
    nanovdb::cuda::DeviceBuffer  mLeafInvertMask;    // nodeCount[0] × Mask<3>: inactive-voxel interior bits
    nanovdb::cuda::DeviceBuffer  mLowerInvertMask;   // nodeCount[1] × Mask<4>: childless-lower-tile interior bits
    nanovdb::cuda::DeviceBuffer  mUpperInvertMask;   // nodeCount[2] × Mask<5>: childless-upper-tile interior bits
    nanovdb::cuda::DeviceBuffer  mRootInterior;      // P×Q×R × uint8: deep-interior bits of absent root regions
    nanovdb::Coord               mRootTileMin{0, 0, 0};  // root-cell array origin (4096-tile units)
    nanovdb::Coord               mRootDims{0, 0, 0};     // root-cell array dims P×Q×R

}; // SurfaceSigner<BuildT>

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

static constexpr int LEAF_SIZE = 512;  // 8^3 voxels per leaf

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Step 2 - prune the surface/barrier shell.
// Voxels within sqrt(3)/2 of the surface straddle it, so which side they are on is not yet decided.
// Dropping them splits each closed surface's band into a separate inner and outer shell, which is
// what connected components then labels.

/// @brief CUDA functor: build a per-leaf retain bitmask that drops the surface/barrier shell. A
///        voxel is PRUNED iff it is within √3/2 voxels of the surface (half a voxel space-diagonal —
///        the same barrier OpenVDB's MeshToVolume uses); every other active voxel is RETAINED.
///        Because the UDF sidecar is in WORLD units, the test is
///        udf^2 < (√3/2 · voxelSize)^2 = 0.75 · voxelSize^2, passed in precomputed.
///        Launched via operatorKernel, one block per leaf, 512 threads (one per voxel in the 8^3 leaf).
template <typename BuildT>
struct UDFBarrierPruneMaskFunctor
{
    static constexpr int MaxThreadsPerBlock         = LEAF_SIZE;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(
        const nanovdb::NanoGrid<BuildT>* d_grid,
        const float*                     d_udf,           // UDF sidecar, WORLD units
        float                            barrierSqWorld,  // (√3/2 · voxelSize)^2, world^2 units
        nanovdb::Mask<3>*                d_dstLeafMasks)
    {
        const int leafID   = blockIdx.x;
        const int threadID = threadIdx.x;

        const auto& leaf       = d_grid->tree().template getFirstNode<0>()[leafID];
        auto&       resultMask = d_dstLeafMasks[leafID];

        // Clear the leaf's mask words in parallel, then fill the retain bits.
        if (threadID < nanovdb::Mask<3>::WORD_COUNT)
            resultMask.words()[threadID] = 0UL;
        __syncthreads();

        if (auto n = leaf.data()->getValue(threadID)) {  // n != 0 => active voxel
            const float udf = d_udf[n];
            if (udf * udf >= barrierSqWorld)             // retain non-barrier voxels
                resultMask.setOnAtomic(threadID);
        }
    }
};

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Step 4 - sign the non-barrier voxels.
// Within one closed surface, the component holding that surface's minimum-x voxel is its exterior and
// every other component of the same surface is interior. Seeding per surface rather than once globally
// is what makes the rule hold for several disconnected objects.

/// @brief Reduce over the active voxels to the minimum x, carrying the voxel's component representative
///        into *d_minKey = (unsigned(x) << 32) | uint32(rep). The min-x voxels are all exterior (and
///        share one representative), so the low 32 bits resolve to the exterior rep.
template <typename BuildT>
struct FindExteriorRepFunctor
{
    static constexpr int MaxThreadsPerBlock         = LEAF_SIZE;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(const NanoGrid<BuildT>* d_grid, const uint32_t* d_voxelLabel,
                               unsigned long long* d_minKey)
    {
        const int leafID = blockIdx.x, n = threadIdx.x;
        const auto& leaf = d_grid->tree().template getFirstNode<0>()[leafID];
        if (!leaf.isActive(uint32_t(n))) return;
        const uint64_t v = leaf.getValue(uint32_t(n));
        const nanovdb::Coord ijk = leaf.origin() + nanovdb::NanoLeaf<BuildT>::OffsetToLocalCoord(uint32_t(n));
        const uint32_t rep = d_voxelLabel[v];                              // component representative slot
        const uint32_t ux  = uint32_t(int64_t(ijk[0]) + (int64_t(1) << 31));  // x, shifted to unsigned-comparable
        atomicMin(d_minKey, (static_cast<unsigned long long>(ux) << 32) | rep);
    }
};

/// @brief Write per-active-voxel signs (+1 exterior / -1 interior), indexed by leaf.getValue(n),
///        into d_sign (length activeVoxelCount+1; slot 0 = background, pre-filled +1). A voxel is
///        exterior iff its component is the exterior representative.
template <typename BuildT>
struct SignNonBarrierFunctor
{
    static constexpr int MaxThreadsPerBlock         = LEAF_SIZE;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(const NanoGrid<BuildT>* d_grid, const uint32_t* d_voxelLabel,
                               uint32_t exteriorRep, int8_t* d_sign)
    {
        const int leafID = blockIdx.x, n = threadIdx.x;
        const auto& leaf = d_grid->tree().template getFirstNode<0>()[leafID];
        if (!leaf.isActive(uint32_t(n))) return;
        const uint64_t v = leaf.getValue(uint32_t(n));
        d_sign[v] = (d_voxelLabel[v] == exteriorRep) ? int8_t(1) : int8_t(-1);
    }
};

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Step 5 - sign the barrier voxels.
// A faithful mirror of OpenVDB MeshToVolume's ComputeIntersectingVoxelSign: a barrier voxel is exterior
// iff some already-signed exterior neighbour's nearest triangle places both of them on the same side of
// the surface. Anchors come from an immutable snapshot, so the result is independent of execution order.

static constexpr uint32_t INVALID_TRIANGLE = 0xFFFFFFFFu;  // nearest-triangle-index sentinel

/// @brief Exterior-anchor proof used by signBarrier (a faithful mirror of one neighbor test in
///        OpenVDB MeshToVolume.h ComputeIntersectingVoxelSign): given a candidate barrier voxel @a q
///        (world/index-space center @a q_xyz) and a neighbor voxel @a n (value-index @a nv, index-space
///        coord @a nijk), returns true iff @a n is a confidently-EXTERIOR voxel (sign == +1) whose
///        nearest triangle places @a q on the SAME side of the surface as @a n — i.e.
///        normalize(n - cp) · normalize(q - cp) > 0, where cp is the closest point on n's nearest
///        triangle to n. Interior / barrier / no-hit neighbors never prove exteriority.
///        Host+device (__hostdev__) so the CPU oracle can reuse the identical math.
///
///        The geometry is done in DOUBLE precision, matching OpenVDB's Vec3d sign test: this is a
///        sign-of-dot decision, and at large index coordinates float cancellation can flip that sign
///        between host and device (different FMA contraction under -use_fast_math). Double precision
///        resolves it reproducibly, so the CPU oracle agrees with the GPU bit-robustly.
__hostdev__ inline bool
barrierExteriorProof(uint64_t nv, const nanovdb::Coord& nijk, const nanovdb::Vec3d& q_xyz,
                     const int8_t* d_sign, const uint32_t* d_index,
                     const nanovdb::Vec3f* d_points, const nanovdb::Vec3i* d_triangles,
                     const nanovdb::Map& map)
{
    if (d_sign[nv] != int8_t(1)) return false;       // only exterior (+1) neighbors anchor
    const uint32_t tid = d_index[nv];
    if (tid == INVALID_TRIANGLE) return false;        // neighbor had no nearest triangle (no-hit)

    const nanovdb::Vec3i& T  = d_triangles[tid];
    const nanovdb::Vec3f& p0 = d_points[T[0]];
    const nanovdb::Vec3f& p1 = d_points[T[1]];
    const nanovdb::Vec3f& p2 = d_points[T[2]];
    const nanovdb::Vec3d  v0 = map.applyInverseMap(nanovdb::Vec3d(p0[0], p0[1], p0[2]));  // world -> index
    const nanovdb::Vec3d  v1 = map.applyInverseMap(nanovdb::Vec3d(p1[0], p1[1], p1[2]));
    const nanovdb::Vec3d  v2 = map.applyInverseMap(nanovdb::Vec3d(p2[0], p2[1], p2[2]));
    const nanovdb::Vec3d  n_xyz(static_cast<double>(nijk[0]), static_cast<double>(nijk[1]), static_cast<double>(nijk[2]));

    double t0, t1;
    const nanovdb::Vec3d cp = nanovdb::math::closestPointOnTriangleToPoint(v0, v1, v2, n_xyz, t0, t1);
    nanovdb::Vec3d dn = n_xyz - cp; dn.normalize();   // surface -> neighbor (its confident side)
    nanovdb::Vec3d dq = q_xyz - cp; dq.normalize();   // surface -> q
    return dn.dot(dq) > 0.0;                           // same side => q is exterior
}

/// @brief Sign every barrier voxel (sign == 0) in place, as a faithful mirror of OpenVDB's
///        ComputeIntersectingVoxelSign. One block per leaf, one thread per voxel:
///          - non-barrier voxels (sign != 0): copy their sign through to the output unchanged;
///          - barrier voxels: search neighbors for the first EXTERIOR (+1) anchor that proves q
///            exterior (barrierExteriorProof). Pass 1 scans the in-leaf 3×3×3 directly off the leaf
///            buffer and early-outs; pass 2 (only for unresolved voxels touching the leaf boundary)
///            scans the 26-neighborhood crossing the boundary via ONE reused ReadAccessor. If no anchor
///            proves exteriority, q defaults to interior (-1).
///        Anchors are read from @a d_signIn (immutable post-injection snapshot: +1/-1 non-barrier,
///        0 barrier) and results written to a separate @a d_signOut, so a just-signed barrier voxel is
///        never used as an anchor (matching OpenVDB, which anchors only on confident voxels) and the
///        result is independent of thread/block execution order.
template <typename BuildT>
struct SignBarrierFunctor
{
    static constexpr int MaxThreadsPerBlock         = LEAF_SIZE;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(
        const NanoGrid<BuildT>* d_grid,
        const int8_t*           d_signIn,     // immutable anchors: +1 ext / -1 int / 0 barrier
        int8_t*                 d_signOut,    // result: every active voxel ±1 (slot 0 set by host)
        const uint32_t*         d_index,      // nearest-triangle index sidecar (original grid)
        const nanovdb::Vec3f*   d_points,     // mesh vertices, WORLD space
        const nanovdb::Vec3i*   d_triangles,  // triangle vertex indices
        nanovdb::Map            map)          // world<->index transform (by value)
    {
        const int leafID = blockIdx.x, n = threadIdx.x;
        const auto& leaf = d_grid->tree().template getFirstNode<0>()[leafID];
        if (!leaf.isActive(uint32_t(n))) return;

        const uint64_t qv = leaf.getValue(uint32_t(n));
        const int8_t   qs = d_signIn[qv];
        if (qs != int8_t(0)) { d_signOut[qv] = qs; return; }  // non-barrier: carry sign through

        const nanovdb::Coord local  = nanovdb::NanoLeaf<BuildT>::OffsetToLocalCoord(uint32_t(n));
        const int            lx = local[0], ly = local[1], lz = local[2];
        const nanovdb::Coord origin = leaf.origin();
        const nanovdb::Vec3d q_xyz(double(origin[0] + lx), double(origin[1] + ly), double(origin[2] + lz));

        bool exterior = false;

        // Pass 1: in-leaf 3×3×3 neighbors (direct leaf buffer), early-out on first proof.
        for (int dx = -1; dx <= 1 && !exterior; ++dx) {
            const int nx = lx + dx; if (nx < 0 || nx > 7) continue;
            for (int dy = -1; dy <= 1 && !exterior; ++dy) {
                const int ny = ly + dy; if (ny < 0 || ny > 7) continue;
                for (int dz = -1; dz <= 1; ++dz) {
                    const int nz = lz + dz; if (nz < 0 || nz > 7) continue;
                    if (dx == 0 && dy == 0 && dz == 0) continue;  // skip q itself
                    const uint32_t nOff = (uint32_t(nx) << 6) | (uint32_t(ny) << 3) | uint32_t(nz);
                    if (!leaf.isActive(nOff)) continue;
                    const nanovdb::Coord nijk(origin[0] + nx, origin[1] + ny, origin[2] + nz);
                    if (barrierExteriorProof(leaf.getValue(nOff), nijk, q_xyz,
                                             d_signIn, d_index, d_points, d_triangles, map)) {
                        exterior = true; break;
                    }
                }
            }
        }

        // Pass 2: 26-neighborhood crossing the leaf boundary (only if unresolved and q is on a face).
        if (!exterior && (lx == 0 || lx == 7 || ly == 0 || ly == 7 || lz == 0 || lz == 7)) {
            auto acc = d_grid->getAccessor();
            for (int dx = -1; dx <= 1 && !exterior; ++dx)
                for (int dy = -1; dy <= 1 && !exterior; ++dy)
                    for (int dz = -1; dz <= 1; ++dz) {
                        if (dx == 0 && dy == 0 && dz == 0) continue;
                        const int nx = lx + dx, ny = ly + dy, nz = lz + dz;
                        if (nx >= 0 && nx <= 7 && ny >= 0 && ny <= 7 && nz >= 0 && nz <= 7)
                            continue;  // in-leaf neighbor already handled by pass 1
                        const nanovdb::Coord nijk(origin[0] + nx, origin[1] + ny, origin[2] + nz);
                        if (!acc.isActive(nijk)) continue;
                        if (barrierExteriorProof(acc.getValue(nijk), nijk, q_xyz,
                                                 d_signIn, d_index, d_points, d_triangles, map)) {
                            exterior = true; break;
                        }
                    }
        }

        d_signOut[qv] = exterior ? int8_t(1) : int8_t(-1);
    }
};

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Step 6A - inactive voxels inside materialized leaves.
// The band only covers the surface's neighbourhood; everywhere else the sign is carried implicitly by a
// per-level invert mask whose bits mark the interior regions. Each level is flooded outwards from the
// signed band, with the active voxels acting as walls.

/// @brief Step 6 (chunk A, leaf level): build one invert Mask<3> per materialized leaf that signs the
///        INACTIVE voxels — bit ON => interior (-background), bit OFF => exterior (+background, the
///        default). One block per leaf, 512 threads (one per voxel), shared-memory Jacobi flood:
///          - injectors  = ACTIVE voxels with sign -1 (the interior side of the signed band);
///          - walls      = ALL active voxels (exterior/barrier active neighbors never turn a bit on);
///          - propagation runs through INACTIVE voxels only, via the 6 face neighbors, entirely
///            within the leaf. The active band bounds the flood, so it cannot leak from
///            interior-inactive to exterior-inactive.
///        A leaf with no interior voxels stays all-0. Cross-leaf / coarser-level fill is a later chunk.
template <typename BuildT>
struct FillLeafInvertMaskFunctor
{
    static constexpr int MaxThreadsPerBlock         = LEAF_SIZE;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(const NanoGrid<BuildT>* d_grid,
                               const int8_t*           d_sign,        // full sign sidecar (post step 5)
                               nanovdb::Mask<3>*       d_invertMasks) // one Mask<3> per leaf, output
    {
        const int leafID = blockIdx.x, n = threadIdx.x;
        const auto& leaf = d_grid->tree().template getFirstNode<0>()[leafID];

        __shared__ uint8_t sAct[LEAF_SIZE];  // active voxel (wall)
        __shared__ uint8_t sInj[LEAF_SIZE];  // interior active voxel (flood source)
        __shared__ uint8_t sInv[LEAF_SIZE];  // result: inactive voxel marked interior
        __shared__ int     sChanged;

        const bool act = leaf.isActive(uint32_t(n));
        sAct[n] = act ? 1 : 0;
        sInj[n] = (act && d_sign[leaf.getValue(uint32_t(n))] == int8_t(-1)) ? 1 : 0;
        sInv[n] = 0;
        __syncthreads();

        // Voxel offset layout n = (x<<6)|(y<<3)|z; face neighbors are n±64 / n±8 / n±1.
        const int x = n >> 6, y = (n >> 3) & 7, z = n & 7;
        auto feeds = [&](int m) { return sInj[m] || (!sAct[m] && sInv[m]); };

        // Jacobi flood to convergence. Sources are ≤1 band-thickness away through smooth inactive
        // pockets, so convergence takes far fewer than 64 sweeps; 64 is a safety cap.
        for (int it = 0; it < 64; ++it) {
            if (n == 0) sChanged = 0;
            __syncthreads();
            bool turnOn = false;
            if (!act && !sInv[n]) {
                if ((x > 0 && feeds(n - 64)) || (x < 7 && feeds(n + 64)) ||
                    (y > 0 && feeds(n -  8)) || (y < 7 && feeds(n +  8)) ||
                    (z > 0 && feeds(n -  1)) || (z < 7 && feeds(n +  1)))
                    turnOn = true;
            }
            __syncthreads();                          // all reads of sInv precede this sweep's writes
            if (turnOn) { sInv[n] = 1; sChanged = 1; }
            __syncthreads();                          // writes (incl. sChanged) visible to the break test
            const bool done = (sChanged == 0);        // latch into a register...
            __syncthreads();                          // ...so thread 0's next-iter reset can't race the read
            if (done) break;
        }

        // Pack the 512 result bits into the leaf's Mask<3> (bit n lives in word n>>6, bit n&63).
        if (n < int(nanovdb::Mask<3>::WORD_COUNT)) {
            uint64_t w = 0;
            for (int b = 0; b < 64; ++b)
                if (sInv[(n << 6) | b]) w |= (uint64_t(1) << b);
            d_invertMasks[leafID].words()[n] = w;
        }
    }
};

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Step 6B - childless lower (8^3) and upper (128^3) tiles.
// Built bottom-up: leaf faces seed the tiles they abut, those tiles flood among themselves, then lower
// faces seed the upper level and it floods in turn.

/// @brief Descend root -> upper -> lower for the 8^3 leaf-region at coord @a c and report where the
///        finest CHILDLESS tile containing it lives (the face->coarser seeding target, plan §7a).
/// @return 0 = skip: root-level value tile or refined all the way to a leaf (handled at the finer
///         level); 1 = childless LOWER slot; 2 = childless UPPER slot; 3 = ABSENT root region (no
///         root tile at all — the chunk-C root-cell target, cell = floorDiv(c,4096) per axis).
///         On 1/2, @a nodeIdx = linear node index at that level and @a slot = child-slot offset.
template <typename BuildT>
__hostdev__ inline int
probeChildlessSlot(const NanoGrid<BuildT>& grid, const nanovdb::Coord& c,
                   uint64_t& nodeIdx, uint32_t& slot)
{
    using UpperT = NanoUpper<BuildT>;
    using LowerT = NanoLower<BuildT>;
    const auto& tree = grid.tree();
    const auto* tile = tree.root().probeTile(c);
    if (!tile) return 3;                                        // absent root region (chunk C)
    if (!tile->isChild()) return 0;                             // root-level value tile: skip
    const UpperT* upper = tree.root().getChild(tile);
    const LowerT* lower = upper->probeChild(c);
    if (!lower) {                                               // childless upper slot
        nodeIdx = util::PtrDiff(upper, tree.template getFirstNode<2>()) / sizeof(UpperT);
        slot    = UpperT::CoordToOffset(c);
        return 2;
    }
    const uint32_t lOff = LowerT::CoordToOffset(c);
    if (!lower->childMask().isOn(lOff)) {                       // childless lower slot
        nodeIdx = util::PtrDiff(lower, tree.template getFirstNode<1>()) / sizeof(LowerT);
        slot    = lOff;
        return 1;
    }
    return 0;                                                   // refined to a leaf
}

/// @brief Chunk-B seeding pass 1 (plan §7a): each leaf classifies its 6 faces and accumulates
///        interior/exterior evidence into the childless coarse tile across each face. One block per
///        leaf, 384 threads = 6 faces × 64 face voxels. Face voxel classification:
///          interior = (active && sign == -1) || (inactive && leaf invert bit ON)
///          exterior = (active && sign == +1) || (inactive && leaf invert bit OFF)
///        (No barrier special-case: a face abutting a CHILDLESS tile is barrier-free by construction —
///        a barrier voxel forces the neighbor across to be refined, §6d rule 3.)
///        The whole 8×8 face abuts exactly one 8^3 region across, so one probe per face decides the
///        target: childless lower/upper slot -> atomically OR the per-tile sawInterior/sawExterior
///        bits; refined or root-level -> skip. The final seed gate (sawInt && !sawExt) is applied by
///        the flood kernel.
template <typename BuildT>
struct LeafFaceSeedFunctor
{
    static constexpr int MaxThreadsPerBlock         = 384;  // 6 faces × 64 cells
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(const NanoGrid<BuildT>*  d_grid,
                               const int8_t*            d_sign,        // completed signs (post step 5)
                               const nanovdb::Mask<3>*  d_leafInvert,  // chunk-A leaf invert masks
                               nanovdb::Mask<4>* d_lowerSawInt, nanovdb::Mask<4>* d_lowerSawExt,
                               nanovdb::Mask<5>* d_upperSawInt, nanovdb::Mask<5>* d_upperSawExt)
    {
        const int leafID = blockIdx.x, t = threadIdx.x;
        const auto& leaf = d_grid->tree().template getFirstNode<0>()[leafID];

        __shared__ int sInt[6], sExt[6];
        if (t < 6) { sInt[t] = 0; sExt[t] = 0; }
        __syncthreads();

        // Face voxel of this thread: face = t/64 (0..5 = -x,+x,-y,+y,-z,+z), (a,b) = 8×8 position.
        const int face = t >> 6, a = (t >> 3) & 7, b = t & 7;
        int n;  // voxel offset n = (x<<6)|(y<<3)|z
        switch (face) {
            case 0:  n = (0 << 6) | (a << 3) | b; break;
            case 1:  n = (7 << 6) | (a << 3) | b; break;
            case 2:  n = (a << 6) | (0 << 3) | b; break;
            case 3:  n = (a << 6) | (7 << 3) | b; break;
            case 4:  n = (a << 6) | (b << 3) | 0; break;
            default: n = (a << 6) | (b << 3) | 7; break;
        }
        const bool act      = leaf.isActive(uint32_t(n));
        const bool interior = act ? (d_sign[leaf.getValue(uint32_t(n))] == int8_t(-1))
                                  : d_leafInvert[leafID].isOn(uint32_t(n));
        if (interior) sInt[face] = 1; else sExt[face] = 1;  // benign race: all writers store 1
        __syncthreads();

        if (t < 6) {  // one probe per face
            const int off[6][3] = {{-8,0,0},{8,0,0},{0,-8,0},{0,8,0},{0,0,-8},{0,0,8}};
            const nanovdb::Coord c = leaf.origin().offsetBy(off[t][0], off[t][1], off[t][2]);
            uint64_t nodeIdx; uint32_t slot;
            const int level = probeChildlessSlot(*d_grid, c, nodeIdx, slot);
            if (level == 1) {
                if (sInt[t]) d_lowerSawInt[nodeIdx].setOnAtomic(slot);
                if (sExt[t]) d_lowerSawExt[nodeIdx].setOnAtomic(slot);
            } else if (level == 2) {
                if (sInt[t]) d_upperSawInt[nodeIdx].setOnAtomic(slot);
                if (sExt[t]) d_upperSawExt[nodeIdx].setOnAtomic(slot);
            }
        }
    }
};

/// @brief Chunk-B seeding pass 2 (plan §7a): each LOWER node classifies the childless slots on its 6
///        faces (via the already-flooded lower invert mask) and accumulates into the childless UPPER
///        tile across each face. Refined face slots are skipped — their finer content contributed via
///        LeafFaceSeedFunctor. A probe landing on another lower node's slot is a SAME-level neighbor:
///        cross-node same-level propagation is deferred (like leaf<->leaf), not chunk B. One block per
///        lower node, 512 threads striding 6 faces × 16×16 = 1536 face cells.
template <typename BuildT>
struct LowerFaceSeedFunctor
{
    static constexpr int MaxThreadsPerBlock         = 512;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(const NanoGrid<BuildT>*  d_grid,
                               const nanovdb::Mask<4>*  d_lowerInvert,  // flooded lower invert masks
                               nanovdb::Mask<5>* d_upperSawInt, nanovdb::Mask<5>* d_upperSawExt)
    {
        const int nodeID = blockIdx.x, t = threadIdx.x;
        const auto& node = d_grid->tree().template getFirstNode<1>()[nodeID];

        __shared__ int sInt[6], sExt[6];
        if (t < 6) { sInt[t] = 0; sExt[t] = 0; }
        __syncthreads();

        for (int u = t; u < 6 * 256; u += blockDim.x) {
            const int face = u >> 8, a = (u >> 4) & 15, b = u & 15;
            int n;  // lower slot offset n = (x<<8)|(y<<4)|z
            switch (face) {
                case 0:  n = ( 0 << 8) | (a << 4) | b; break;
                case 1:  n = (15 << 8) | (a << 4) | b; break;
                case 2:  n = (a << 8) | ( 0 << 4) | b; break;
                case 3:  n = (a << 8) | (15 << 4) | b; break;
                case 4:  n = (a << 8) | (b << 4) |  0; break;
                default: n = (a << 8) | (b << 4) | 15; break;
            }
            if (node.childMask().isOn(uint32_t(n))) continue;  // refined: leaf faces already contributed
            if (d_lowerInvert[nodeID].isOn(uint32_t(n))) sInt[face] = 1; else sExt[face] = 1;
        }
        __syncthreads();

        if (t < 6) {  // whole 128×128 face abuts exactly one 128^3 region across
            const int off[6][3] = {{-128,0,0},{128,0,0},{0,-128,0},{0,128,0},{0,0,-128},{0,0,128}};
            const nanovdb::Coord c = node.origin().offsetBy(off[t][0], off[t][1], off[t][2]);
            uint64_t nodeIdx; uint32_t slot;
            if (probeChildlessSlot(*d_grid, c, nodeIdx, slot) == 2) {
                if (sInt[t]) d_upperSawInt[nodeIdx].setOnAtomic(slot);
                if (sExt[t]) d_upperSawExt[nodeIdx].setOnAtomic(slot);
            }
        }
    }
};

/// @brief Chunk-B finalize + within-node flood at a coarse level (LEVEL 1 = lower/16^3, 2 = upper/
///        32^3), the coarse-granularity analogue of FillLeafInvertMaskFunctor. Per node: seed =
///        childless slots passing the §6d.4 gate (sawInterior && !sawExterior — errs to exterior on
///        mixed evidence, the safe side), then a monotone ON-flood over 6-adjacent CHILDLESS slots;
///        REFINED slots are walls. Safety invariant: two adjacent childless slots can never be on
///        opposite sides of the surface (the surface between them would force refinement), so the
///        flood cannot leak interior -> exterior. One block per node, 512 threads striding the slots;
///        slot bits live in shared memory as packed words (Mask<4> 512 B / Mask<5> 4 KB).
template <typename BuildT, int LEVEL>
struct CoarseInvertFloodFunctor
{
    static constexpr int LOG2DIM = (LEVEL == 1) ? 4 : 5;
    static constexpr int DIM     = 1 << LOG2DIM;             // 16 / 32
    static constexpr int SLOTS   = 1 << (3 * LOG2DIM);       // 4096 / 32768
    static constexpr int WORDS   = SLOTS >> 6;               // 64 / 512
    using MaskT = nanovdb::Mask<LOG2DIM>;

    static constexpr int MaxThreadsPerBlock         = 512;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(const NanoGrid<BuildT>* d_grid,
                               const MaskT* d_sawInt, const MaskT* d_sawExt, MaskT* d_invert)
    {
        const int nodeID = blockIdx.x, t = threadIdx.x;
        const auto& node = d_grid->tree().template getFirstNode<LEVEL>()[nodeID];

        __shared__ uint64_t sWall[WORDS];  // refined slots (childMask)
        __shared__ uint64_t sInv [WORDS];  // result bits (seeded, then flooded)
        __shared__ int      sChanged;

        for (int w = t; w < WORDS; w += blockDim.x) {
            const uint64_t wall = node.childMask().words()[w];
            sWall[w] = wall;
            sInv[w]  = d_sawInt[nodeID].words()[w] & ~d_sawExt[nodeID].words()[w] & ~wall;
        }
        __syncthreads();

        auto isOn = [&](const uint64_t* m, int s) { return (m[s >> 6] >> (s & 63)) & 1ull; };

        // Monotone ON-flood: racy same-sweep reads only accelerate legitimate propagation (a set bit
        // is final truth), so in-place atomicOr is safe. Cap = Manhattan diameter + slack.
        for (int it = 0; it < 3 * DIM + 16; ++it) {
            if (t == 0) sChanged = 0;
            __syncthreads();
            bool any = false;
            for (int s = t; s < SLOTS; s += blockDim.x) {
                if (isOn(sWall, s) || isOn(sInv, s)) continue;
                const int x = s >> (2 * LOG2DIM), y = (s >> LOG2DIM) & (DIM - 1), z = s & (DIM - 1);
                constexpr int dx = 1 << (2 * LOG2DIM), dy = 1 << LOG2DIM;
                // inv bits exist only on childless slots, so a set neighbor bit is a valid feeder
                if ((x > 0       && isOn(sInv, s - dx)) || (x < DIM - 1 && isOn(sInv, s + dx)) ||
                    (y > 0       && isOn(sInv, s - dy)) || (y < DIM - 1 && isOn(sInv, s + dy)) ||
                    (z > 0       && isOn(sInv, s -  1)) || (z < DIM - 1 && isOn(sInv, s +  1))) {
                    ::atomicOr(reinterpret_cast<unsigned long long*>(&sInv[s >> 6]), 1ull << (s & 63));
                    any = true;
                }
            }
            if (any) sChanged = 1;
            __syncthreads();
            const bool done = (sChanged == 0);  // latch, then barrier: next-iter reset can't race the read
            __syncthreads();
            if (done) break;
        }

        for (int w = t; w < WORDS; w += blockDim.x)
            d_invert[nodeID].words()[w] = sInv[w];
    }
};

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Step 6C - the deep interior beyond any upper node, held in a small
// provisional P×Q×R cell array over the grid's root-tile bounding range (one cell per 4096^3 root
// region) — a SIDECAR consulted by signedSignAt(); the grid itself is never mutated.

/// @brief Mark the WALL cells of the root-cell array: cells whose 4096^3 region has a pre-existing
///        root entry (an upper child — the band passes through). One thread per cell via lambdaKernel.
template <typename BuildT>
struct RootWallMarkFunctor
{
    const NanoGrid<BuildT>* dGrid;
    uint8_t*                dWall;
    nanovdb::Coord          tileMin;  // root-cell range origin, in 4096-tile units
    nanovdb::Coord          dims;     // P×Q×R

    __device__ void operator()(size_t idx) const
    {
        const int k = int(idx) % dims[2], j = (int(idx) / dims[2]) % dims[1], i = int(idx) / (dims[1] * dims[2]);
        const nanovdb::Coord c((tileMin[0] + i) << 12, (tileMin[1] + j) << 12, (tileMin[2] + k) << 12);
        const auto* tile = dGrid->tree().root().probeTile(c);
        dWall[idx] = (tile && tile->isChild()) ? 1 : 0;
    }
};

/// @brief Chunk-C seeding: accumulate interior/exterior evidence from node faces that abut an ABSENT
///        root region, from ALL THREE levels (a deep-interior root cell can be abutted by a childless
///        upper tile, a childless lower tile, OR a leaf's inactive-interior voxels — seeding from the
///        upper level alone would miss the finer abutments). One block per node; face cells classified
///        exactly as in the chunk-B seed functors (leaf: sign / leaf-invert; lower/upper: childless
///        slot invert bit, refined slots skipped); each whole face abuts exactly ONE root cell, probed
///        once. Out-of-range cells (beyond the array) are skipped — they stay exterior by default.
///        // classification duplicated from Leaf/LowerFaceSeedFunctor — keep in sync
template <typename BuildT, int LEVEL>  // 0 = leaf, 1 = lower, 2 = upper
struct RootFaceSeedFunctor
{
    static constexpr int MaxThreadsPerBlock         = (LEVEL == 0) ? 384 : 512;
    static constexpr int MinBlocksPerMultiprocessor = 1;
    static constexpr int NODE_DIM = (LEVEL == 0) ? 8 : (LEVEL == 1) ? 128 : 4096;

    __device__ void operator()(const NanoGrid<BuildT>*  d_grid,
                               const int8_t*            d_sign,
                               const nanovdb::Mask<3>*  d_leafInvert,
                               const nanovdb::Mask<4>*  d_lowerInvert,
                               const nanovdb::Mask<5>*  d_upperInvert,
                               uint8_t* d_sawInt, uint8_t* d_sawExt,
                               nanovdb::Coord tileMin, nanovdb::Coord dims)
    {
        const int nodeID = blockIdx.x, t = threadIdx.x;
        const auto& node = d_grid->tree().template getFirstNode<LEVEL>()[nodeID];

        __shared__ int sInt[6], sExt[6];
        if (t < 6) { sInt[t] = 0; sExt[t] = 0; }
        __syncthreads();

        if constexpr (LEVEL == 0) {
            const int face = t >> 6, a = (t >> 3) & 7, b = t & 7;
            int n;
            switch (face) {
                case 0:  n = (0 << 6) | (a << 3) | b; break;
                case 1:  n = (7 << 6) | (a << 3) | b; break;
                case 2:  n = (a << 6) | (0 << 3) | b; break;
                case 3:  n = (a << 6) | (7 << 3) | b; break;
                case 4:  n = (a << 6) | (b << 3) | 0; break;
                default: n = (a << 6) | (b << 3) | 7; break;
            }
            const bool act      = node.isActive(uint32_t(n));
            const bool interior = act ? (d_sign[node.getValue(uint32_t(n))] == int8_t(-1))
                                      : d_leafInvert[nodeID].isOn(uint32_t(n));
            if (interior) sInt[face] = 1; else sExt[face] = 1;
        } else if constexpr (LEVEL == 1) {
            for (int u = t; u < 6 * 256; u += blockDim.x) {
                const int face = u >> 8, a = (u >> 4) & 15, b = u & 15;
                int n;
                switch (face) {
                    case 0:  n = ( 0 << 8) | (a << 4) | b; break;
                    case 1:  n = (15 << 8) | (a << 4) | b; break;
                    case 2:  n = (a << 8) | ( 0 << 4) | b; break;
                    case 3:  n = (a << 8) | (15 << 4) | b; break;
                    case 4:  n = (a << 8) | (b << 4) |  0; break;
                    default: n = (a << 8) | (b << 4) | 15; break;
                }
                if (node.childMask().isOn(uint32_t(n))) continue;  // refined: finer level contributes
                if (d_lowerInvert[nodeID].isOn(uint32_t(n))) sInt[face] = 1; else sExt[face] = 1;
            }
        } else {
            for (int u = t; u < 6 * 1024; u += blockDim.x) {
                const int face = u >> 10, a = (u >> 5) & 31, b = u & 31;
                int n;
                switch (face) {
                    case 0:  n = ( 0 << 10) | (a << 5) | b; break;
                    case 1:  n = (31 << 10) | (a << 5) | b; break;
                    case 2:  n = (a << 10) | ( 0 << 5) | b; break;
                    case 3:  n = (a << 10) | (31 << 5) | b; break;
                    case 4:  n = (a << 10) | (b << 5) |  0; break;
                    default: n = (a << 10) | (b << 5) | 31; break;
                }
                if (node.childMask().isOn(uint32_t(n))) continue;
                if (d_upperInvert[nodeID].isOn(uint32_t(n))) sInt[face] = 1; else sExt[face] = 1;
            }
        }
        __syncthreads();

        if (t < 6) {  // one probe per face: the whole face abuts exactly one root cell
            const int off[6][3] = {{-NODE_DIM,0,0},{NODE_DIM,0,0},{0,-NODE_DIM,0},{0,NODE_DIM,0},{0,0,-NODE_DIM},{0,0,NODE_DIM}};
            const nanovdb::Coord c = node.origin().offsetBy(off[t][0], off[t][1], off[t][2]);
            uint64_t nodeIdx; uint32_t slot;
            if (probeChildlessSlot(*d_grid, c, nodeIdx, slot) == 3) {  // absent root region
                const int i = (c[0] >> 12) - tileMin[0], j = (c[1] >> 12) - tileMin[1], k = (c[2] >> 12) - tileMin[2];
                if (i >= 0 && i < dims[0] && j >= 0 && j < dims[1] && k >= 0 && k < dims[2]) {
                    const int idx = (i * dims[1] + j) * dims[2] + k;
                    if (sInt[t]) d_sawInt[idx] = 1;  // benign race: all writers store 1
                    if (sExt[t]) d_sawExt[idx] = 1;
                }
            }
        }
    }
};

/// @brief Chunk-C multi-seed flood on the P×Q×R root-cell array (single block; the array is a handful
///        of cells). Seed = sawInterior && !sawExterior && !wall (mixed evidence => exterior, the safe
///        side); then flood ON among non-wall cells over 6-adjacency — WALL cells (pre-existing root
///        entries) block. Every interior-abutting cell seeds (multi-seed), so disconnected interiors
///        all fill. The surface at 4096 granularity always lies inside a wall cell, so the flood
///        cannot leak interior -> exterior. Grid-independent (plain arrays) so it is unit-testable.
struct RootInteriorFloodFunctor
{
    static constexpr int MaxThreadsPerBlock         = 256;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(const uint8_t* d_wall, const uint8_t* d_sawInt, const uint8_t* d_sawExt,
                               uint8_t* d_on, nanovdb::Coord dims)
    {
        const int t = threadIdx.x;
        const int P = dims[0], Q = dims[1], R = dims[2], total = P * Q * R;
        __shared__ int sChanged;

        for (int c = t; c < total; c += blockDim.x)
            d_on[c] = (!d_wall[c] && d_sawInt[c] && !d_sawExt[c]) ? 1 : 0;
        __syncthreads();

        for (int it = 0; it < total + 2; ++it) {  // cap: any path length < total cells
            if (t == 0) sChanged = 0;
            __syncthreads();
            bool any = false;
            for (int c = t; c < total; c += blockDim.x) {
                if (d_wall[c] || d_on[c]) continue;
                const int k = c % R, j = (c / R) % Q, i = c / (Q * R);
                const bool on =
                    (i > 0     && d_on[c - Q * R]) || (i < P - 1 && d_on[c + Q * R]) ||
                    (j > 0     && d_on[c - R])     || (j < Q - 1 && d_on[c + R])     ||
                    (k > 0     && d_on[c - 1])     || (k < R - 1 && d_on[c + 1]);
                if (on) { d_on[c] = 1; any = true; }  // monotone: racy same-sweep reads only accelerate
            }
            if (any) sChanged = 1;
            __syncthreads();
            const bool done = (sChanged == 0);  // latch, then barrier: next-iter reset can't race the read
            __syncthreads();
            if (done) break;
        }
    }
};

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Query - compose the per-level sidecars into a sign at any coordinate.
// Descends the tree and answers from whichever level owns the coordinate: the sign sidecar for an active
// voxel, the leaf/lower/upper invert mask for an inactive voxel or a childless tile, and the
// root-interior array beyond the last upper node.

/// @brief Full-domain sign query (host + device): the completed level-set sign at ANY index coord,
///        composed from the grid plus the step-6 sidecars — no grid mutation anywhere. Descends:
///          active voxel            -> sign sidecar (±1)
///          inactive leaf voxel     -> leaf invert bit
///          childless lower slot    -> lower invert bit
///          childless upper slot    -> upper invert bit
///          ABSENT root region      -> root-interior sidecar (in-range && ON => interior, else exterior)
/// @return +1 outside / -1 inside.
template <typename BuildT>
__hostdev__ inline int8_t
signedSignAt(const NanoGrid<BuildT>& grid, const nanovdb::Coord& ijk,
             const int8_t* sign, const nanovdb::Mask<3>* leafInvert,
             const nanovdb::Mask<4>* lowerInvert, const nanovdb::Mask<5>* upperInvert,
             const uint8_t* rootInterior, const nanovdb::Coord& rootTileMin, const nanovdb::Coord& rootDims)
{
    using UpperT = NanoUpper<BuildT>;
    using LowerT = NanoLower<BuildT>;
    using LeafT  = NanoLeaf<BuildT>;
    const auto& tree = grid.tree();
    const auto* tile = tree.root().probeTile(ijk);
    if (!tile || !tile->isChild()) {  // absent root region (or root value tile): consult the sidecar
        const int i = (ijk[0] >> 12) - rootTileMin[0],
                  j = (ijk[1] >> 12) - rootTileMin[1],
                  k = (ijk[2] >> 12) - rootTileMin[2];
        const bool interior = rootInterior &&
            i >= 0 && i < rootDims[0] && j >= 0 && j < rootDims[1] && k >= 0 && k < rootDims[2] &&
            rootInterior[(i * rootDims[1] + j) * rootDims[2] + k];
        return interior ? int8_t(-1) : int8_t(1);
    }
    const UpperT* upper = tree.root().getChild(tile);
    const LowerT* lower = upper->probeChild(ijk);
    if (!lower) {
        const uint64_t u = util::PtrDiff(upper, tree.template getFirstNode<2>()) / sizeof(UpperT);
        return upperInvert[u].isOn(UpperT::CoordToOffset(ijk)) ? int8_t(-1) : int8_t(1);
    }
    const LeafT* leaf = lower->probeChild(ijk);
    if (!leaf) {
        const uint64_t l = util::PtrDiff(lower, tree.template getFirstNode<1>()) / sizeof(LowerT);
        return lowerInvert[l].isOn(LowerT::CoordToOffset(ijk)) ? int8_t(-1) : int8_t(1);
    }
    const uint32_t n = LeafT::CoordToOffset(ijk);
    if (leaf->isActive(n)) return sign[leaf->getValue(n)];
    const uint64_t lf = util::PtrDiff(leaf, tree.template getFirstNode<0>()) / sizeof(LeafT);
    return leafInvert[lf].isOn(n) ? int8_t(-1) : int8_t(1);
}

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Partition and composition - the stages MeshToSDF runs around the per-surface signing: carving one
// surface out of the rasterized band, picking a representative voxel for it, probing another
// surface's field at that voxel, and folding the resulting nesting parity into the signs.

// Retain mask selecting one surface's voxels out of the original grid: one block per leaf, one thread
// per voxel offset, bit ON iff the voxel carries the target surface label.
template <typename BuildT>
struct SurfaceMaskFunctor
{
    static constexpr int MaxThreadsPerBlock         = 512;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(const NanoGrid<BuildT>* d_grid, const uint32_t* d_surfaceLabel,
                               uint32_t target, Mask<3>* d_masks)
    {
        const int leafID = blockIdx.x, n = threadIdx.x;
        const auto& leaf = d_grid->tree().template getFirstNode<0>()[leafID];
        auto&       mask = d_masks[leafID];
        if (n < int(Mask<3>::WORD_COUNT)) mask.words()[n] = 0UL;
        __syncthreads();
        if (auto v = leaf.data()->getValue(uint32_t(n)))          // v != 0 => active voxel
            if (d_surfaceLabel[v] == target) mask.setOnAtomic(uint32_t(n));
    }
};

// Pack a voxel coordinate into one sortable key so a per-surface atomicMin picks a deterministic
// representative voxel. 21 bits per axis covers |coord| < 2^20, far beyond any rasterized grid.
__hostdev__ inline unsigned long long packCoord(const Coord& c)
{
    return ((unsigned long long)(c[0] + (1 << 20)) << 42) |
           ((unsigned long long)(c[1] + (1 << 20)) << 21) |
            (unsigned long long)(c[2] + (1 << 20));
}
// One representative voxel per surface (the packed-coordinate minimum, so it is deterministic).
template <typename BuildT>
struct SurfaceRepFunctor
{
    static constexpr int MaxThreadsPerBlock         = 512;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(const NanoGrid<BuildT>* d_grid, const uint32_t* d_surfaceLabel,
                               uint32_t surfaceCount, unsigned long long* d_repKey)
    {
        const int leafID = blockIdx.x, n = threadIdx.x;
        const auto& leaf = d_grid->tree().template getFirstNode<0>()[leafID];
        if (!leaf.isActive(uint32_t(n))) return;
        const uint32_t s = d_surfaceLabel[leaf.getValue(uint32_t(n))];
        if (s >= surfaceCount) return;
        const Coord ijk = leaf.origin() + NanoLeaf<BuildT>::OffsetToLocalCoord(uint32_t(n));
        atomicMin(&d_repKey[s], packCoord(ijk));
    }
};

// Ask one surface's sign field about every surface's representative voxel: out[j] < 0 means that
// surface's band lies inside this one.
template <typename BuildT>
struct InclusionProbeFunctor
{
    __device__ void operator()(size_t j, const NanoGrid<BuildT>* d_gridI,
                               const unsigned long long* d_repKey, const int8_t* d_signI,
                               const Mask<3>* d_leafI, const nanovdb::Mask<4>* d_lowI,
                               const nanovdb::Mask<5>* d_upI, const uint8_t* d_rootI,
                               Coord rootMin, Coord rootDims, int8_t* d_out) const
    {
        const unsigned long long k = d_repKey[j];
        const Coord ijk(int((k >> 42) & 0x1FFFFF) - (1 << 20),
                                 int((k >> 21) & 0x1FFFFF) - (1 << 20),
                                 int( k        & 0x1FFFFF) - (1 << 20));
        d_out[j] = signedSignAt<BuildT>(
            *d_gridI, ijk, d_signI, d_leafI, d_lowI, d_upI, d_rootI, rootMin, rootDims);
    }
};

// Flip each voxel's sign once per enclosing surface: sign *= (-1)^depth(surface).
struct FlipSignByDepthFunctor
{
    __device__ void operator()(size_t v, const uint32_t* d_surfaceLabel, const uint8_t* d_flip,
                               uint32_t surfaceCount, int8_t* d_sign) const
    {
        if (v == 0) return;                       // slot 0 is the background
        const uint32_t s = d_surfaceLabel[v];
        if (s < surfaceCount && d_flip[s]) d_sign[v] = int8_t(-d_sign[v]);
    }
};

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
template <typename BufferT>
GridHandle<BufferT>
SurfaceSigner<BuildT>::computeDerivedTopology(const GridT* d_srcGrid, const float* d_udf,
                                          float voxelSize, const BufferT& buffer)
{
    using PruneOp = UDFBarrierPruneMaskFunctor<BuildT>;

    // Barrier threshold √3/2 voxels expressed in the sidecar's WORLD units, squared.
    const float    barrierSqWorld = 0.75f * voxelSize * voxelSize;
    const uint32_t srcLeafCount = leafCountOf(d_srcGrid);

    // Leaf-indexed retain mask: one Mask<3> (512 bits) per source leaf (device-only).
    auto  retainMask   = nanovdb::cuda::DeviceBuffer::create(
        std::size_t(srcLeafCount) * sizeof(nanovdb::Mask<3>), nullptr, false);
    auto* d_retainMask = static_cast<nanovdb::Mask<3>*>(retainMask.deviceData());

    if (mVerbose==1) mTimer.start("Prune barrier shell -> derived topology");
    util::cuda::operatorKernel<PruneOp><<<srcLeafCount, PruneOp::MaxThreadsPerBlock, 0, mStream>>>(
        d_srcGrid, d_udf, barrierSqWorld, d_retainMask);
    cudaCheckError();

    // Topological pruning -> clean, topology-only derived index grid (UDF no longer needed).
    PruneGrid<BuildT> pruner(d_srcGrid, d_retainMask, mStream);
    pruner.setVerbose(mVerbose);
    auto handle = pruner.template getHandle<BufferT>(buffer);
    if (mVerbose==1) mTimer.stop();
    return handle;
}// SurfaceSigner<BuildT>::computeDerivedTopology

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
void SurfaceSigner<BuildT>::signNonBarrier(const GridT* d_grid, const uint32_t* d_voxelLabel)
{
    const uint32_t leafCount = leafCountOf(d_grid);
    mExteriorRep = 0;
    if (leafCount == 0) return;  // every materialized leaf has >=1 component, so leafCount==0 => K==0

    const uint64_t activeCount = activeCountOf(d_grid);

    // (1) The exterior representative = component of the grid's minimum-x active voxel.
    if (mVerbose==1) mTimer.start("Sign: find exterior component");
    auto minKeyBuf = nanovdb::cuda::DeviceBuffer::create(sizeof(unsigned long long), nullptr, false);
    auto* d_minKey = static_cast<unsigned long long*>(minKeyBuf.deviceData());
    cudaCheck(cudaMemsetAsync(d_minKey, 0xFF, sizeof(unsigned long long), mStream));   // ~0ull
    using FindOp = FindExteriorRepFunctor<BuildT>;
    util::cuda::operatorKernel<FindOp><<<leafCount, FindOp::MaxThreadsPerBlock, 0, mStream>>>(
        d_grid, d_voxelLabel, d_minKey);
    cudaCheckError();

    unsigned long long minKey = 0;   // low 32 bits of the min key = the exterior representative
    cudaCheck(cudaMemcpyAsync(&minKey, d_minKey, sizeof(minKey), cudaMemcpyDeviceToHost, mStream));
    cudaCheck(cudaStreamSynchronize(mStream));
    mExteriorRep = uint64_t(uint32_t(minKey & 0xFFFFFFFFull));
    if (mVerbose==1) mTimer.stop();

    // (2) Per-voxel sign: +1 exterior / -1 interior (slot 0 = background +1).
    mVoxelSign = nanovdb::cuda::DeviceBuffer::create((activeCount + 1) * sizeof(int8_t), nullptr, false);
    cudaCheck(cudaMemsetAsync(mVoxelSign.deviceData(), 1, (activeCount + 1) * sizeof(int8_t), mStream));// all +1
    using SignOp = SignNonBarrierFunctor<BuildT>;
    if (mVerbose==1) mTimer.start("Sign: write per-voxel signs");
    util::cuda::operatorKernel<SignOp><<<leafCount, SignOp::MaxThreadsPerBlock, 0, mStream>>>(
        d_grid, d_voxelLabel, uint32_t(mExteriorRep), deviceVoxelSign());
    cudaCheckError();
    if (mVerbose==1) mTimer.stop();
}// SurfaceSigner<BuildT>::signNonBarrier

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
void SurfaceSigner<BuildT>::injectSignsToOriginal(const GridT* d_origGrid, const GridT* d_derivedGrid)
{
    const uint64_t origActive = activeCountOf(d_origGrid);
    const uint32_t derivedLeafCount = leafCountOf(d_derivedGrid);

    // Sentinel 0 ("unsigned barrier") everywhere; slot 0 = background (+1). Non-barrier voxels are
    // overwritten by the injection below; barrier voxels (in original but not derived) keep 0.
    mOriginalVoxelSign = nanovdb::cuda::DeviceBuffer::create((origActive + 1) * sizeof(int8_t), nullptr, false);
    cudaCheck(cudaMemsetAsync(mOriginalVoxelSign.deviceData(), 0, (origActive + 1) * sizeof(int8_t), mStream));
    cudaCheck(cudaMemsetAsync(mOriginalVoxelSign.deviceData(), 1, sizeof(int8_t), mStream)); // slot 0 = +1

    if (derivedLeafCount == 0) return;  // nothing signed -> all voxels stay sentinel

    // Inject derived signs into the original sidecar at the intersection (= every non-barrier voxel,
    // since derived ⊂ original). One block per derived (source) leaf.
    using InjectOp = util::cuda::InjectGridDataFunctor<BuildT, int8_t>;
    if (mVerbose==1) mTimer.start("Inject derived signs -> original grid");
    util::cuda::operatorKernel<InjectOp><<<derivedLeafCount, InjectOp::MaxThreadsPerBlock, 0, mStream>>>(
        d_derivedGrid, d_origGrid, deviceVoxelSign(), deviceOriginalVoxelSign());
    cudaCheckError();
    if (mVerbose==1) mTimer.stop();
}// SurfaceSigner<BuildT>::injectSignsToOriginal

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
void SurfaceSigner<BuildT>::signBarrier(const GridT* d_grid, const uint32_t* d_index,
                                    const nanovdb::Vec3f* d_points, const nanovdb::Vec3i* d_triangles,
                                    const nanovdb::Map& map)
{
    const uint64_t activeCount = activeCountOf(d_grid);
    const uint32_t leafCount = leafCountOf(d_grid);

    // Output: every active voxel ends up ±1. Start at 0, set slot 0 (background) = +1; the kernel
    // writes every active voxel (carrying non-barrier signs through, filling barriers).
    mSignedVoxelSign = nanovdb::cuda::DeviceBuffer::create((activeCount + 1) * sizeof(int8_t), nullptr, false);
    cudaCheck(cudaMemsetAsync(mSignedVoxelSign.deviceData(), 0, (activeCount + 1) * sizeof(int8_t), mStream));
    cudaCheck(cudaMemsetAsync(mSignedVoxelSign.deviceData(), 1, sizeof(int8_t), mStream)); // slot 0 = +1
    if (leafCount == 0) return;

    using Op = SignBarrierFunctor<BuildT>;
    if (mVerbose==1) mTimer.start("Sign: barrier voxels (intersecting-voxel-sign mirror)");
    util::cuda::operatorKernel<Op><<<leafCount, Op::MaxThreadsPerBlock, 0, mStream>>>(
        d_grid, deviceOriginalVoxelSign(), deviceSignedVoxelSign(),
        d_index, d_points, d_triangles, map);
    cudaCheckError();
    if (mVerbose==1) mTimer.stop();
}// SurfaceSigner<BuildT>::signBarrier

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
void SurfaceSigner<BuildT>::fillLeafInvertMask(const GridT* d_grid, const int8_t* d_sign)
{
    const int8_t* sign = d_sign ? d_sign : deviceSignedVoxelSign();  // external signs override the member
    const uint32_t leafCount = leafCountOf(d_grid);
    if (leafCount == 0) { mLeafInvertMask = nanovdb::cuda::DeviceBuffer(); return; }

    mLeafInvertMask = nanovdb::cuda::DeviceBuffer::create(
        std::size_t(leafCount) * sizeof(nanovdb::Mask<3>), nullptr, false);
    cudaCheck(cudaMemsetAsync(mLeafInvertMask.deviceData(), 0,
                              std::size_t(leafCount) * sizeof(nanovdb::Mask<3>), mStream));

    using Op = FillLeafInvertMaskFunctor<BuildT>;
    if (mVerbose==1) mTimer.start("Fill leaf invert mask (inactive-voxel interior flood)");
    util::cuda::operatorKernel<Op><<<leafCount, Op::MaxThreadsPerBlock, 0, mStream>>>(
        d_grid, sign, deviceLeafInvertMask());
    cudaCheckError();
    if (mVerbose==1) mTimer.stop();
}// SurfaceSigner<BuildT>::fillLeafInvertMask

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
void SurfaceSigner<BuildT>::fillCoarseInvertMasks(const GridT* d_grid, const int8_t* d_sign)
{
    const int8_t* sign = d_sign ? d_sign : deviceSignedVoxelSign();  // external signs override the member
    const auto     treeData   = util::cuda::DeviceGridTraits<BuildT>::getTreeData(d_grid);
    const uint32_t leafCount  = treeData.mNodeCount[0];
    const uint32_t lowerCount = treeData.mNodeCount[1];
    const uint32_t upperCount = treeData.mNodeCount[2];

    const std::size_t lowerBytes = std::size_t(lowerCount) * sizeof(nanovdb::Mask<4>);
    const std::size_t upperBytes = std::size_t(upperCount) * sizeof(nanovdb::Mask<5>);
    mLowerInvertMask = lowerCount ? nanovdb::cuda::DeviceBuffer::create(lowerBytes, nullptr, false)
                                  : nanovdb::cuda::DeviceBuffer();
    mUpperInvertMask = upperCount ? nanovdb::cuda::DeviceBuffer::create(upperBytes, nullptr, false)
                                  : nanovdb::cuda::DeviceBuffer();
    if (lowerCount) cudaCheck(cudaMemsetAsync(mLowerInvertMask.deviceData(), 0, lowerBytes, mStream));
    if (upperCount) cudaCheck(cudaMemsetAsync(mUpperInvertMask.deviceData(), 0, upperBytes, mStream));
    if (leafCount == 0 || lowerCount == 0) return;  // nothing to seed from

    // Temporary per-tile evidence accumulators (freed at scope exit).
    auto lowSawIntBuf = nanovdb::cuda::DeviceBuffer::create(lowerBytes, nullptr, false);
    auto lowSawExtBuf = nanovdb::cuda::DeviceBuffer::create(lowerBytes, nullptr, false);
    auto upSawIntBuf  = nanovdb::cuda::DeviceBuffer::create(upperBytes, nullptr, false);
    auto upSawExtBuf  = nanovdb::cuda::DeviceBuffer::create(upperBytes, nullptr, false);
    auto* d_lowSawInt = static_cast<nanovdb::Mask<4>*>(lowSawIntBuf.deviceData());
    auto* d_lowSawExt = static_cast<nanovdb::Mask<4>*>(lowSawExtBuf.deviceData());
    auto* d_upSawInt  = static_cast<nanovdb::Mask<5>*>(upSawIntBuf.deviceData());
    auto* d_upSawExt  = static_cast<nanovdb::Mask<5>*>(upSawExtBuf.deviceData());
    cudaCheck(cudaMemsetAsync(d_lowSawInt, 0, lowerBytes, mStream));
    cudaCheck(cudaMemsetAsync(d_lowSawExt, 0, lowerBytes, mStream));
    cudaCheck(cudaMemsetAsync(d_upSawInt,  0, upperBytes, mStream));
    cudaCheck(cudaMemsetAsync(d_upSawExt,  0, upperBytes, mStream));

    // (1) Leaf faces seed the childless lower/upper tiles across them.
    using LeafSeedOp = LeafFaceSeedFunctor<BuildT>;
    if (mVerbose==1) mTimer.start("Coarse invert: seed from leaf faces");
    util::cuda::operatorKernel<LeafSeedOp><<<leafCount, LeafSeedOp::MaxThreadsPerBlock, 0, mStream>>>(
        d_grid, sign, deviceLeafInvertMask(),
        d_lowSawInt, d_lowSawExt, d_upSawInt, d_upSawExt);
    cudaCheckError();
    if (mVerbose==1) mTimer.stop();

    // (2) Finalize (sawInt && !sawExt gate) + flood the lower level.
    using LowerFloodOp = CoarseInvertFloodFunctor<BuildT, 1>;
    if (mVerbose==1) mTimer.start("Coarse invert: flood lower nodes");
    util::cuda::operatorKernel<LowerFloodOp><<<lowerCount, LowerFloodOp::MaxThreadsPerBlock, 0, mStream>>>(
        d_grid, d_lowSawInt, d_lowSawExt, deviceLowerInvertMask());
    cudaCheckError();
    if (mVerbose==1) mTimer.stop();

    // (3) Lower faces seed the childless upper tiles across them.
    using LowerSeedOp = LowerFaceSeedFunctor<BuildT>;
    if (mVerbose==1) mTimer.start("Coarse invert: seed from lower faces");
    util::cuda::operatorKernel<LowerSeedOp><<<lowerCount, LowerSeedOp::MaxThreadsPerBlock, 0, mStream>>>(
        d_grid, deviceLowerInvertMask(), d_upSawInt, d_upSawExt);
    cudaCheckError();
    if (mVerbose==1) mTimer.stop();

    // (4) Finalize + flood the upper level.
    if (upperCount) {
        using UpperFloodOp = CoarseInvertFloodFunctor<BuildT, 2>;
        if (mVerbose==1) mTimer.start("Coarse invert: flood upper nodes");
        util::cuda::operatorKernel<UpperFloodOp><<<upperCount, UpperFloodOp::MaxThreadsPerBlock, 0, mStream>>>(
            d_grid, d_upSawInt, d_upSawExt, deviceUpperInvertMask());
        cudaCheckError();
        if (mVerbose==1) mTimer.stop();
    }

    // The evidence accumulators go out of scope here; sync so their frees can't outrun the kernels.
    cudaCheck(cudaStreamSynchronize(mStream));
}// SurfaceSigner<BuildT>::fillCoarseInvertMasks

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
void SurfaceSigner<BuildT>::fillRootInteriorMask(const GridT* d_grid, const int8_t* d_sign)
{
    const int8_t* sign = d_sign ? d_sign : deviceSignedVoxelSign();  // external signs override the member
    const auto     treeData   = util::cuda::DeviceGridTraits<BuildT>::getTreeData(d_grid);
    const uint32_t leafCount  = treeData.mNodeCount[0];
    const uint32_t lowerCount = treeData.mNodeCount[1];
    const uint32_t upperCount = treeData.mNodeCount[2];
    if (leafCount == 0) { mRootInterior = nanovdb::cuda::DeviceBuffer(); mRootDims = nanovdb::Coord(0); return; }

    // C1: provisional cell range = the grid bbox at root-tile (4096^3) granularity. >>12 is floor
    // division by 4096 for negative coords too (arithmetic shift).
    const auto bbox = util::cuda::DeviceGridTraits<BuildT>::getIndexBBox(d_grid, treeData);
    mRootTileMin = nanovdb::Coord(bbox.min()[0] >> 12, bbox.min()[1] >> 12, bbox.min()[2] >> 12);
    const nanovdb::Coord tileMax(bbox.max()[0] >> 12, bbox.max()[1] >> 12, bbox.max()[2] >> 12);
    mRootDims = nanovdb::Coord(tileMax[0] - mRootTileMin[0] + 1,
                               tileMax[1] - mRootTileMin[1] + 1,
                               tileMax[2] - mRootTileMin[2] + 1);
    const std::size_t total = std::size_t(mRootDims[0]) * mRootDims[1] * mRootDims[2];

    mRootInterior    = nanovdb::cuda::DeviceBuffer::create(total, nullptr, false);
    auto wallBuf     = nanovdb::cuda::DeviceBuffer::create(total, nullptr, false);
    auto sawIntBuf   = nanovdb::cuda::DeviceBuffer::create(total, nullptr, false);
    auto sawExtBuf   = nanovdb::cuda::DeviceBuffer::create(total, nullptr, false);
    auto* d_wall     = static_cast<uint8_t*>(wallBuf.deviceData());
    auto* d_sawInt   = static_cast<uint8_t*>(sawIntBuf.deviceData());
    auto* d_sawExt   = static_cast<uint8_t*>(sawExtBuf.deviceData());
    cudaCheck(cudaMemsetAsync(mRootInterior.deviceData(), 0, total, mStream));
    cudaCheck(cudaMemsetAsync(d_wall,   0, total, mStream));
    cudaCheck(cudaMemsetAsync(d_sawInt, 0, total, mStream));
    cudaCheck(cudaMemsetAsync(d_sawExt, 0, total, mStream));

    if (mVerbose==1) mTimer.start("Root interior: mark walls + seed + flood");

    // C1: walls = pre-existing root entries.
    constexpr unsigned int kWallThreads = 128;
    util::cuda::lambdaKernel<<<unsigned((total + kWallThreads - 1) / kWallThreads), kWallThreads, 0, mStream>>>(
        total, RootWallMarkFunctor<BuildT>{ d_grid, d_wall, mRootTileMin, mRootDims });
    cudaCheckError();

    // C2: interior/exterior evidence from ALL THREE levels into abutting absent root cells.
    using Seed0 = RootFaceSeedFunctor<BuildT, 0>;
    using Seed1 = RootFaceSeedFunctor<BuildT, 1>;
    using Seed2 = RootFaceSeedFunctor<BuildT, 2>;
    util::cuda::operatorKernel<Seed0><<<leafCount, Seed0::MaxThreadsPerBlock, 0, mStream>>>(
        d_grid, sign, deviceLeafInvertMask(), deviceLowerInvertMask(),
        deviceUpperInvertMask(), d_sawInt, d_sawExt, mRootTileMin, mRootDims);
    cudaCheckError();
    if (lowerCount) {
        util::cuda::operatorKernel<Seed1><<<lowerCount, Seed1::MaxThreadsPerBlock, 0, mStream>>>(
            d_grid, sign, deviceLeafInvertMask(), deviceLowerInvertMask(),
            deviceUpperInvertMask(), d_sawInt, d_sawExt, mRootTileMin, mRootDims);
        cudaCheckError();
    }
    if (upperCount) {
        util::cuda::operatorKernel<Seed2><<<upperCount, Seed2::MaxThreadsPerBlock, 0, mStream>>>(
            d_grid, sign, deviceLeafInvertMask(), deviceLowerInvertMask(),
            deviceUpperInvertMask(), d_sawInt, d_sawExt, mRootTileMin, mRootDims);
        cudaCheckError();
    }

    // C3: gate + multi-seed flood (single block; the array is a handful of cells).
    using FloodOp = RootInteriorFloodFunctor;
    util::cuda::operatorKernel<FloodOp><<<1, FloodOp::MaxThreadsPerBlock, 0, mStream>>>(
        d_wall, d_sawInt, d_sawExt, deviceRootInterior(), mRootDims);
    cudaCheckError();
    if (mVerbose==1) mTimer.stop();

    // Temporaries go out of scope here; sync so their frees can't outrun the kernels.
    cudaCheck(cudaStreamSynchronize(mStream));
}// SurfaceSigner<BuildT>::fillRootInteriorMask

} // namespace sdf_detail

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// MeshToSDF - partition the rasterized band into closed surfaces, sign each one alone, compose.
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
void MeshToSDF<BuildT>::build()
{
    this->rasterize();
    this->partition();
    for (uint32_t i = 0; i < uint32_t(mSurfaces.size()); ++i) this->signSurface(i);
    this->composeByInclusion();
    this->fillOnOriginal();
}// MeshToSDF<BuildT>::build

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
void MeshToSDF<BuildT>::rasterize()
{
    MeshToGrid<BuildT> converter(mPoints, mPointCount, mTriangles, mTriangleCount, mMap, mStream);
    converter.setVerbose(mVerbose);
    converter.setNarrowBandWidth(mBandWidth);
    std::tie(mGridHandle, mUDF, mIndex) = converter.getHandleAndUDFAndIndex();
}// MeshToSDF<BuildT>::rasterize

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

/// Connected components on the UN-PRUNED band. The barrier shell is what glues a surface's inner and
/// outer sides together, so leaving it in place makes each closed surface exactly one component — and
/// the component count the surface count.
template <typename BuildT>
void MeshToSDF<BuildT>::partition()
{
    mSurfaceCC = std::make_unique<ConnectedComponents<BuildT>>(this->deviceGrid(), mStream);
    mSurfaceCC->setVerbose(mVerbose);
    mSurfaceLabels = mSurfaceCC->getVoxelLabelsAndCount();
    cudaCheck(cudaStreamSynchronize(mStream));
    mSurfaces.resize(std::size_t(mSurfaceLabels.second));
}// MeshToSDF<BuildT>::partition

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

/// Carve one closed surface out of the rasterized band and run the whole signing sequence on it, so
/// the result is the sign field that surface would have if it were the only object in the scene.
///
/// The global exterior seed is legitimate here precisely because the grid holds one closed surface:
/// its minimum-x active voxel is necessarily outside it. A mesh with a single closed surface needs no
/// carving at all — the rasterized band already IS that surface's band — so subGrid/subUdf/subIndex
/// stay empty and the rasterized arrays are used directly. That is a memory decision: carving
/// duplicates the grid plus the UDF, index and sign sidecars, ~9 bytes per voxel.
template <typename BuildT>
void MeshToSDF<BuildT>::signSurface(uint32_t surface)
{
    using Traits = util::cuda::DeviceGridTraits<BuildT>;

    SurfaceField&  sf         = mSurfaces[surface];
    const auto*    d_orig     = this->deviceGrid();
    const uint32_t origLeaves = Traits::getTreeData(d_orig).mNodeCount[0];
    const float    voxelSize  = float(mMap.getVoxelSize()[0]);

    // (a) Carve this surface out, and re-index onto the carved grid the two sidecars the stages below
    //     read. Carving renumbers the value slots, so the transfer goes through the injection functor
    //     (leaf-origin pairing + popcount rank) rather than a memcpy.
    if (mSurfaces.size() > 1) {
        util::cuda::Timer timer(mStream);
        if (mVerbose==1) timer.start("Carve closed surface out of the band");
        auto  maskBuf    = Buffer::create(std::size_t(origLeaves) * sizeof(Mask<3>), nullptr, false);
        auto* d_partMask = static_cast<Mask<3>*>(maskBuf.deviceData());
        using MaskOp = sdf_detail::SurfaceMaskFunctor<BuildT>;
        util::cuda::operatorKernel<MaskOp><<<origLeaves, MaskOp::MaxThreadsPerBlock, 0, mStream>>>(
            d_orig, mSurfaceLabels.first, surface, d_partMask);
        cudaCheckError();

        PruneGrid<BuildT> pruner(d_orig, d_partMask, mStream);
        sf.subGrid = pruner.getHandle();

        const auto*    d_sub     = sf.subGrid.template deviceGrid<BuildT>();
        const uint64_t subActive = Traits::getActiveVoxelCount(d_sub);
        sf.subUdf   = Buffer::create((subActive + 1) * sizeof(float), nullptr, false);
        sf.subIndex = Buffer::create((subActive + 1) * sizeof(uint32_t), nullptr, false);
        cudaCheck(cudaMemsetAsync(sf.subUdf.deviceData(),   0,    (subActive + 1) * sizeof(float), mStream));
        cudaCheck(cudaMemsetAsync(sf.subIndex.deviceData(), 0xFF, (subActive + 1) * sizeof(uint32_t), mStream));

        using InjectUdf   = util::cuda::InjectGridDataFunctor<BuildT, float>;
        using InjectIndex = util::cuda::InjectGridDataFunctor<BuildT, uint32_t>;
        util::cuda::operatorKernel<InjectUdf><<<origLeaves, InjectUdf::MaxThreadsPerBlock, 0, mStream>>>(
            d_orig, d_sub, this->deviceUDF(), static_cast<float*>(sf.subUdf.deviceData()));
        cudaCheckError();
        util::cuda::operatorKernel<InjectIndex><<<origLeaves, InjectIndex::MaxThreadsPerBlock, 0, mStream>>>(
            d_orig, d_sub, this->deviceIndex(), static_cast<uint32_t*>(sf.subIndex.deviceData()));
        cudaCheckError();
        cudaCheck(cudaStreamSynchronize(mStream));
        if (mVerbose==1) timer.stop();
    }

    const auto* d_grid = this->surfaceGrid(surface);
    sf.signer = std::make_unique<Signer>(mStream);
    sf.signer->setVerbose(mVerbose);

    // (b) Drop the barrier shell, then label what remains. This surface's inner and outer sides fall
    //     apart into separate components — the split the signing rule relies on.
    sf.derived = sf.signer->computeDerivedTopology(d_grid, this->surfaceUdf(surface), voxelSize);
    const auto* d_derived = sf.derived.template deviceGrid<BuildT>();

    sf.cc = std::make_unique<ConnectedComponents<BuildT>>(d_derived, mStream);
    sf.cc->setVerbose(mVerbose);
    sf.ccLabels = sf.cc->getVoxelLabelsAndCount();
    cudaCheck(cudaStreamSynchronize(mStream));

    // (c) Sign the non-barrier voxels, carry the signs onto the un-pruned surface grid, then sign the
    //     barrier shell that (b) set aside.
    sf.signer->signNonBarrier(d_derived, sf.ccLabels.first);
    sf.signer->injectSignsToOriginal(d_grid, d_derived);
    sf.signer->signBarrier(d_grid, this->surfaceIndex(surface), mPoints, mTriangles, mMap);

    // (d) Extend the sign off the band. That is what lets the inclusion test query this field at
    //     another surface's band — and, for a single-surface mesh, it is already the finished result.
    sf.signer->fillLeafInvertMask(d_grid);
    sf.signer->fillCoarseInvertMasks(d_grid);
    sf.signer->fillRootInteriorMask(d_grid);
    cudaCheck(cudaStreamSynchronize(mStream));
}// MeshToSDF<BuildT>::signSurface

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

/// Recover the nesting depth of every closed surface from the per-surface fields, then merge those
/// fields into one sign array on the rasterized band, negating the odd-depth surfaces on the way in.
/// Requires every field to be complete through its invert-mask fill.
template <typename BuildT>
void MeshToSDF<BuildT>::composeByInclusion()
{
    using Traits = util::cuda::DeviceGridTraits<BuildT>;

    const uint32_t N = uint32_t(mSurfaces.size());
    if (N == 0) return;

    // Nothing encloses a lone surface, and an uncarved one already carries its signs on the rasterized
    // band — so there is no depth to recover and nothing to gather. Aliasing here is what keeps the
    // single-surface case free of the extra full-length sign array a merge would allocate.
    if (N == 1 && !mSurfaces[0].subGrid.bufferSize()) {
        mParity.assign(1, 0);
        mSign = mSurfaces[0].signer->deviceSignedVoxelSign();
        return;
    }

    const auto*    d_orig     = this->deviceGrid();
    const uint32_t origLeaves = Traits::getTreeData(d_orig).mNodeCount[0];
    const uint64_t origActive = Traits::getActiveVoxelCount(d_orig);

    util::cuda::Timer timer(mStream);
    if (mVerbose==1) timer.start("Inclusion: nesting depth + merge onto the rasterized band");

    // (1) One representative voxel per surface. Any voxel of a surface's band serves: the band hugs
    //     its own surface, so it lies wholly inside, or wholly outside, every other surface.
    auto  repBuf   = Buffer::create(N * sizeof(unsigned long long), nullptr, false);
    auto* d_repKey = static_cast<unsigned long long*>(repBuf.deviceData());
    cudaCheck(cudaMemsetAsync(d_repKey, 0xFF, N * sizeof(unsigned long long), mStream));
    {
        using RepOp = sdf_detail::SurfaceRepFunctor<BuildT>;
        util::cuda::operatorKernel<RepOp><<<origLeaves, RepOp::MaxThreadsPerBlock, 0, mStream>>>(
            d_orig, mSurfaceLabels.first, N, d_repKey);
        cudaCheckError();
    }

    // (2) Ask each surface's own field about every surface's representative. Each was completed
    //     through the invert-mask fill, so it answers off its band too — including at the other bands.
    auto  incBuf = Buffer::create(std::size_t(N) * N * sizeof(int8_t), nullptr, false);
    auto* d_inc  = static_cast<int8_t*>(incBuf.deviceData());  // d_inc[i*N+j] = field i's sign at surface j
    for (uint32_t i = 0; i < N; ++i) {
        auto&       phi   = *mSurfaces[i].signer;
        const auto* d_sub = this->surfaceGrid(i);
        using ProbeOp = sdf_detail::InclusionProbeFunctor<BuildT>;
        util::cuda::lambdaKernel<<<1, N, 0, mStream>>>(
            N, ProbeOp{}, d_sub, d_repKey, phi.deviceSignedVoxelSign(),
            phi.deviceLeafInvertMask(), phi.deviceLowerInvertMask(), phi.deviceUpperInvertMask(),
            phi.deviceRootInterior(), phi.rootTileMin(), phi.rootTileDims(), d_inc + std::size_t(i) * N);
        cudaCheckError();
    }

    // (3) Nesting depth = how many other surfaces report this one as inside them. The inclusion forest
    //     itself is not needed for the signs — only the parity of the depth is.
    std::vector<int8_t> inc(std::size_t(N) * N);
    cudaCheck(cudaMemcpyAsync(inc.data(), d_inc, inc.size() * sizeof(int8_t), cudaMemcpyDeviceToHost, mStream));
    cudaCheck(cudaStreamSynchronize(mStream));

    std::vector<uint32_t> depth(N, 0);
    mParity.assign(N, 0);
    for (uint32_t j = 0; j < N; ++j) {
        for (uint32_t i = 0; i < N; ++i)
            if (i != j && inc[std::size_t(i) * N + j] < 0) ++depth[j];   // field i says surface j is inside
        mParity[j] = uint8_t(depth[j] & 1u);
    }

    // (4) Merge. Gather every surface's signs back onto the rasterized band — the surfaces partition
    //     its active voxels, so the N injections write disjoint slots and together cover all of them —
    //     then negate the odd-depth ones in place.
    mComposedSign = Buffer::create((origActive + 1) * sizeof(int8_t), nullptr, false);
    mSign = static_cast<int8_t*>(mComposedSign.deviceData());
    cudaCheck(cudaMemsetAsync(mSign, 1, (origActive + 1) * sizeof(int8_t), mStream));  // slot 0 = background +1
    using InjectOp = util::cuda::InjectGridDataFunctor<BuildT, int8_t>;
    for (uint32_t i = 0; i < N; ++i) {
        const auto*    d_sub     = this->surfaceGrid(i);
        const uint32_t subLeaves = Traits::getTreeData(d_sub).mNodeCount[0];
        util::cuda::operatorKernel<InjectOp><<<subLeaves, InjectOp::MaxThreadsPerBlock, 0, mStream>>>(
            d_sub, d_orig, mSurfaces[i].signer->deviceSignedVoxelSign(), mSign);
        cudaCheckError();
    }
    auto  flipBuf = Buffer::create(N * sizeof(uint8_t), nullptr, false);
    auto* d_flip  = static_cast<uint8_t*>(flipBuf.deviceData());
    cudaCheck(cudaMemcpyAsync(d_flip, mParity.data(), N * sizeof(uint8_t), cudaMemcpyHostToDevice, mStream));
    util::cuda::lambdaKernel<<<(unsigned int)((origActive + 256) / 256), 256, 0, mStream>>>(
        origActive + 1, sdf_detail::FlipSignByDepthFunctor{}, mSurfaceLabels.first, d_flip, N, mSign);
    cudaCheckError();
    cudaCheck(cudaStreamSynchronize(mStream));
    if (mVerbose==1) timer.stop();
}// MeshToSDF<BuildT>::composeByInclusion

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

/// Extend the composed sign off the band, over the whole rasterized grid. When the composition aliased
/// a lone uncarved surface, the fill it already ran covered this very grid with these very signs, so
/// it is adopted as-is.
template <typename BuildT>
void MeshToSDF<BuildT>::fillOnOriginal()
{
    if (mSurfaces.empty()) return;

    if (!mComposedSign.size()) {                 // composition aliased surface 0 -> its fill is the answer
        mFinalSigner = mSurfaces[0].signer.get();
        return;
    }

    const auto* d_orig = this->deviceGrid();
    mOrigSigner = std::make_unique<Signer>(mStream);
    mOrigSigner->setVerbose(mVerbose);
    mOrigSigner->fillLeafInvertMask(d_orig, mSign);
    mOrigSigner->fillCoarseInvertMasks(d_orig, mSign);
    mOrigSigner->fillRootInteriorMask(d_orig, mSign);
    cudaCheck(cudaStreamSynchronize(mStream));
    mFinalSigner = mOrigSigner.get();
}// MeshToSDF<BuildT>::fillOnOriginal

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
const GridHandle<nanovdb::cuda::DeviceBuffer>&
MeshToSDF<BuildT>::surfaceGridHandle(uint32_t i) const
{ return mSurfaces[i].subGrid.bufferSize() ? mSurfaces[i].subGrid : mGridHandle; }

template <typename BuildT>
const NanoGrid<BuildT>* MeshToSDF<BuildT>::surfaceGrid(uint32_t i) const
{ return this->surfaceGridHandle(i).template deviceGrid<BuildT>(); }

template <typename BuildT>
const float* MeshToSDF<BuildT>::surfaceUdf(uint32_t i) const
{ return static_cast<const float*>(mSurfaces[i].subUdf.size() ? mSurfaces[i].subUdf.deviceData()
                                                              : mUDF.deviceData()); }

template <typename BuildT>
const uint32_t* MeshToSDF<BuildT>::surfaceIndex(uint32_t i) const
{ return static_cast<const uint32_t*>(mSurfaces[i].subIndex.size() ? mSurfaces[i].subIndex.deviceData()
                                                                   : mIndex.deviceData()); }

template <typename BuildT>
const Mask<3>* MeshToSDF<BuildT>::deviceLeafInvertMask() const
{ return mFinalSigner->deviceLeafInvertMask(); }

template <typename BuildT>
const Mask<4>* MeshToSDF<BuildT>::deviceLowerInvertMask() const
{ return mFinalSigner->deviceLowerInvertMask(); }

template <typename BuildT>
const Mask<5>* MeshToSDF<BuildT>::deviceUpperInvertMask() const
{ return mFinalSigner->deviceUpperInvertMask(); }

template <typename BuildT>
const uint8_t* MeshToSDF<BuildT>::deviceRootInterior() const
{ return mFinalSigner->deviceRootInterior(); }

template <typename BuildT>
Coord MeshToSDF<BuildT>::rootTileMin() const { return mFinalSigner->rootTileMin(); }

template <typename BuildT>
Coord MeshToSDF<BuildT>::rootTileDims() const { return mFinalSigner->rootTileDims(); }

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

} // namespace tools::cuda

} // namespace nanovdb

#endif // NVIDIA_TOOLS_CUDA_MESHTOSDF_CUH_HAS_BEEN_INCLUDED
