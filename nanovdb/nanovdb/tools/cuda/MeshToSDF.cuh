// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/tools/cuda/MeshToSDF.cuh
    \authors Efty Sifakis and JaeHyun Lee
    \brief GPU conversion of triangle meshes to narrow-band signed distance fields on NanoVDB
           index grids.

    \details The pipeline rasterizes a UDF, partitions its band into surfaces, signs each surface,
             composes nested surfaces, and extends the sign beyond the active band. The returned
             grid stores the distance channel and sign masks as blind data.

    \warning This header contains CUDA device code and must be included from a .cu or .cuh file.
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
#include <nanovdb/tools/cuda/AddBlindData.cuh>

#include <chrono>
#include <memory>
#include <utility>
#include <vector>

namespace nanovdb {

namespace tools::cuda {

namespace sdf_detail { template <typename BuildT> class SurfaceSigner; }

/// @brief Convert closed triangle surfaces to a narrow-band signed distance field.
/// @tparam BuildT Build type of the index grid (e.g. nanovdb::ValueOnIndex).
template <typename BuildT>
class MeshToSDF
{
    using GridT   = NanoGrid<BuildT>;
    using Handle  = GridHandle<nanovdb::cuda::DeviceBuffer>;
    using Buffer  = nanovdb::cuda::DeviceBuffer;
    using Signer  = sdf_detail::SurfaceSigner<BuildT>;

public:

    /// @brief Construct from device-resident mesh data. Processing starts in build().
    /// @param d_points       device vertex list in world space
    /// @param pointCount     vertex count
    /// @param d_triangles    device triangle vertex-index list
    /// @param triangleCount  triangle count
    /// @param map            world-to-index transform
    /// @param stream         CUDA stream
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

    /// @brief Policy for signing voxels in the surface barrier.
    enum class BarrierSigning {
        Interior,   ///< classify all barrier voxels as interior
        Heuristic,  ///< use the OpenVDB intersecting-voxel heuristic
        Ball        ///< use ball-overlap certificates and classify unresolved voxels as interior
    };

    /// @brief Choose the barrier signing method (default Interior).
    void setBarrierSigning(BarrierSigning m) { mBarrierSigning = m; }

    /// @brief How a point enclosed by several of the input's closed surfaces is signed.
    enum class NestingRule {
        EvenOdd,  ///< inside when enclosed by an odd number of surfaces
        Solid     ///< inside when enclosed by one or more surfaces
    };

    /// @brief Choose the nesting rule (default EvenOdd).
    void setNestingRule(NestingRule r) { mNestingRule = r; }

    /// @brief Set the stencil half-width used by BarrierSigning::Ball (default is 1).
    void setBallStencilRadius(int radius) { mBallStencilRadius = radius; }

    /// @brief Sign the isosurface where the mesh UDF equals @a isoValue.
    /// @param isoValue Non-negative offset in world units; zero signs the input surface.
    /// @note Rasterization includes the offset, so time and memory increase with @a isoValue.
    void setIsoValue(float isoValue = 0.f) { mIsoValue = isoValue; }

    /// @brief Run the pipeline and return a self-contained index grid.
    /// @details The returned handle contains these blind-data channels:
    ///
    ///   0 "sdf"           float  per active voxel   sign * distance to the signed surface
    ///   1 "leaf_invert"   uint64 per leaf x 8       sign of that leaf's INACTIVE voxels
    ///   2 "lower_invert"  uint64 per lower x 64     sign of that node's childless tiles
    ///   3 "upper_invert"  uint64 per upper x 512    sign of that node's childless tiles
    ///   4 "root_interior" uint8  per root cell      sign of regions with no node at all
    ///   5 "root_extent"   int32  x 6                origin and dims of that cell array
    ///
    /// The accessors below continue to reference the pipeline's internal buffers.
    GridHandle<Buffer> build();

    /// @brief The rasterized narrow band (all surfaces together), valid after build().
    const GridT* deviceGrid() const { return mGridHandle.template deviceGrid<BuildT>(); }
    /// @brief Handle owning that grid, valid after build().
    const Handle& gridHandle() const { return mGridHandle; }
    /// @brief Per-active-voxel unsigned distance in WORLD units, valid after build().
    const float* deviceUDF() const { return static_cast<const float*>(mUDF.deviceData()); }
    /// @brief Per-active-voxel sign over the rasterized band (+1 outside / -1 inside), valid after
    ///        build(). Length activeVoxelCount+1, indexed by leaf.getValue(n); slot 0 = +1.
    const int8_t* deviceSign() const { return mSign; }
    /// @brief The world<->index transform the grid was built with.
    const nanovdb::Map& map() const { return mMap; }
    /// @brief The narrow-band width, in cell units, the grid was built with.
    float narrowBandWidth() const { return mBandWidth; }

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

    /// @brief Wall-clock milliseconds for rasterize, partition, per-surface signing, composition,
    ///        and finalization/fill. Blind-data assembly is not included.
    const float* phaseMs() const { return mPhaseMs; }

private:

    /// @brief Intermediate and final data for one surface.
    struct SurfaceField {
        Handle  subGrid;   // this surface's band, carved out of the rasterized one. EMPTY when the
                           // mesh has a single closed surface — that band is then used as-is.
        Buffer  subUdf;    // udf / nearest-triangle index re-indexed onto subGrid (carving renumbers
        Buffer  subIndex;  // the value slots). Both empty in that same single-surface case.
        Handle  derived;   // barrier-pruned connected-components input
        std::unique_ptr<ConnectedComponents<BuildT>> cc;
        std::pair<uint32_t*, uint64_t>               ccLabels{nullptr, 0};
        std::unique_ptr<Signer>                      signer;
    };

    void rasterize();
    void partition();
    void signSurface(uint32_t surface);
    void composeByInclusion();
    void postProcess();
    void fillOnOriginal();
    GridHandle<Buffer> bakeBlindData();

    const Handle&   surfaceGridHandle(uint32_t i) const;
    const GridT*    surfaceGrid(uint32_t i) const;
    const uint32_t* surfaceIndex(uint32_t i) const;
    const float*    surfaceUdf(uint32_t i) const;
    void            releaseIntermediates();

    const nanovdb::Vec3f* mPoints{nullptr};
    uint32_t              mPointCount{0};
    const nanovdb::Vec3i* mTriangles{nullptr};
    uint32_t              mTriangleCount{0};
    nanovdb::Map          mMap{};
    cudaStream_t          mStream{0};
    int                   mVerbose{0};
    float                 mBandWidth{3.f};
    BarrierSigning        mBarrierSigning{BarrierSigning::Interior};
    float                 mIsoValue{0.f};  // world units, >= 0; see setIsoValue()
    int                   mBallStencilRadius{1};  // see setBallStencilRadius()
    float                 mPhaseMs[5]{};   // per-phase wall time from the last build(), see phaseMs()
    // Sizes below use A = the rasterized band's active voxel count and N = the closed-surface count.
    // Every per-voxel sidecar is A+1 long and indexed by leaf.getValue(n), so slot 0 is the background.
    Handle mGridHandle;   // rasterized narrow band, all surfaces together
    Buffer mUDF, mIndex;  // (A+1) x float / (A+1) x uint32: unsigned distance, nearest-triangle index

    std::unique_ptr<ConnectedComponents<BuildT>> mSurfaceCC;
    std::pair<uint32_t*, uint64_t>               mSurfaceLabels{nullptr, 0};   // { (A+1) x uint32 surface
                                                                               // id, N }; owned by mSurfaceCC
    std::vector<SurfaceField> mSurfaces;   // one entry per surface

    NestingRule          mNestingRule{NestingRule::EvenOdd};
    Buffer               mComposedSign;    // (A+1) x int8_t: gathered signs on the rasterized band. EMPTY
                                           // when a lone uncarved surface's own array is used instead —
                                           // which is also how fillOnOriginal knows its fill is already done.
    int8_t*              mSign{nullptr};   // (A+1) x int8_t, NOT owned: the signs every later stage reads.
                                           // Points into mComposedSign, or into surface 0's signer.

    std::unique_ptr<Signer> mOrigSigner;   // empty when surfaces[0]'s fill is adopted
    Signer*                 mFinalSigner{nullptr};  // NOT owned: mOrigSigner, or surface 0's signer

}; // tools::cuda::MeshToSDF<BuildT>

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace sdf_detail {

/// @brief Build the complete sign field for one closed surface.
/// @tparam BuildT Build type of the index grid (e.g. nanovdb::ValueOnIndex).
template <typename BuildT>
class SurfaceSigner
{
    using GridT = NanoGrid<BuildT>;

public:

    /// @brief Construct on the given CUDA stream.
    SurfaceSigner(cudaStream_t stream = 0) : mStream(stream), mTimer(stream) {}

    /// @brief Toggle on and off verbose mode
    /// @param level Verbose level: 0=quiet, 1=timing
    void setVerbose(int level = 1) { mVerbose = level; }

    /// @brief Remove voxels within sqrt(3)/2 voxels of the surface being signed.
    /// @return The non-barrier topology used for connected-components labeling.
    template <typename BufferT = nanovdb::cuda::DeviceBuffer>
    GridHandle<BufferT> computeDerivedTopology(const GridT* d_srcGrid, const float* d_udf,
                                               float voxelSize, float isoValue = 0.f,
                                               const BufferT& buffer = BufferT());

    /// @brief Sign the minimum-x component exterior and all other non-barrier components interior.
    /// @note The input grid must contain exactly one closed surface.
    void signNonBarrier(const GridT* d_grid, const uint32_t* d_voxelLabel);

    /// @brief Transfer non-barrier signs to the source grid, leaving barrier signs at zero.
    void injectSignsToOriginal(const GridT* d_origGrid, const GridT* d_derivedGrid);

    /// @brief Sign all barrier voxels as interior.
    void signBarrierAsInterior(const GridT* d_grid);

    /// @brief Sign barrier voxels with the OpenVDB intersecting-voxel heuristic.
    void signBarrier(const GridT* d_grid, const uint32_t* d_index,
                     const nanovdb::Vec3f* d_points, const nanovdb::Vec3i* d_triangles,
                     const nanovdb::Map& map, float isoValue = 0.f,
                     float voxelSize = 1.f);

    /// @brief Sign barrier voxels with iterative ball-overlap certificates.
    /// @note Unresolved voxels are classified as interior in deviceSignedVoxelSign().
    void signBarrierByBalls(const GridT* d_grid, const float* d_udf, float voxelSize,
                            int maxRounds = 32, int radius = 1, float isoValue = 0.f);

    /// @brief Build interior masks for inactive voxels in materialized leaves.
    void fillLeafInvertMask(const GridT* d_grid, const int8_t* d_sign = nullptr);

    /// @brief Build interior masks for childless lower and upper tiles.
    void fillCoarseInvertMasks(const GridT* d_grid, const int8_t* d_sign = nullptr);

    /// @brief Build interior flags for absent 4096^3 root regions.
    void fillRootInteriorMask(const GridT* d_grid, const int8_t* d_sign = nullptr);

    /// @brief Completed source-grid signs.
    int8_t* deviceSignedVoxelSign() { return static_cast<int8_t*>(mSignedVoxelSign.deviceData()); }

    /// @brief Per-leaf interior masks for inactive voxels.
    nanovdb::Mask<3>* deviceLeafInvertMask() { return static_cast<nanovdb::Mask<3>*>(mLeafInvertMask.deviceData()); }

    /// @brief Per-lower-node interior masks for childless tiles.
    nanovdb::Mask<4>* deviceLowerInvertMask() { return static_cast<nanovdb::Mask<4>*>(mLowerInvertMask.deviceData()); }

    /// @brief Per-upper-node interior masks for childless tiles.
    nanovdb::Mask<5>* deviceUpperInvertMask() { return static_cast<nanovdb::Mask<5>*>(mUpperInvertMask.deviceData()); }

    /// @brief Interior flags for absent root regions.
    uint8_t* deviceRootInterior() { return static_cast<uint8_t*>(mRootInterior.deviceData()); }

    /// @brief Origin of the root-cell array in 4096-tile units.
    nanovdb::Coord rootTileMin() const { return mRootTileMin; }

    /// @brief Dimensions of the root-cell array.
    nanovdb::Coord rootTileDims() const { return mRootDims; }

private:

    // Shorthands for the two device-grid queries the stages keep asking for.
    static uint32_t leafCountOf(const GridT* g) {
        return util::cuda::DeviceGridTraits<BuildT>::getTreeData(g).mNodeCount[0];
    }
    static uint64_t activeCountOf(const GridT* g) {
        return util::cuda::DeviceGridTraits<BuildT>::getActiveVoxelCount(g);
    }

    int8_t* deviceVoxelSign() { return static_cast<int8_t*>(mVoxelSign.deviceData()); }
    int8_t* deviceOriginalVoxelSign() { return static_cast<int8_t*>(mOriginalVoxelSign.deviceData()); }

    cudaStream_t                 mStream{0};
    util::cuda::Timer            mTimer;
    int                          mVerbose{0};

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

/// @brief Combine magnitude and sign into a contiguous SDF channel.
struct SignedDistanceFunctor
{
    __device__ void operator()(const uint64_t slot, const float* d_udf, const int8_t* d_sign,
                               float* d_out) const
    {
        d_out[slot] = float(d_sign[slot]) * d_udf[slot];
    }
};// sdf_detail::SignedDistanceFunctor

/// @brief Convert the mesh UDF to distance from the requested isosurface.
/// @note Interior values away from a sign change are floored at half a voxel diagonal.
template <typename BuildT>
struct IsoMagnitudeFunctor
{
    static constexpr int MaxThreadsPerBlock         = LEAF_SIZE;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(const NanoGrid<BuildT>* d_grid, float* d_udf, const int8_t* d_sign,
                               float isoValue, float interiorFloor)
    {
        const int leafID = blockIdx.x, n = threadIdx.x;
        const auto& leaf = d_grid->tree().template getFirstNode<0>()[leafID];
        if (!leaf.isActive(uint32_t(n))) return;

        const uint64_t v = leaf.getValue(uint32_t(n));
        const float    m = fabsf(d_udf[v] - isoValue);
        d_udf[v] = m;
        if (d_sign[v] >= int8_t(0) || m >= interiorFloor) return;   // nothing to floor

        // Only a voxel with an exterior FACE neighbour can carry the interface: a marching-cubes
        // vertex lands on an axis edge, between two face-adjacent samples of opposite sign. Flooring
        // such a voxel would drag that crossing along, so leave it alone and floor the rest.
        const nanovdb::Coord local = nanovdb::NanoLeaf<BuildT>::OffsetToLocalCoord(uint32_t(n));
        const int lx = local[0], ly = local[1], lz = local[2];
        const int off[6][3] = {{-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1}};

        bool touchesExterior = false, needsAccessor = false;
        for (int k = 0; k < 6 && !touchesExterior; ++k) {
            const int nx = lx + off[k][0], ny = ly + off[k][1], nz = lz + off[k][2];
            if (nx < 0 || nx > 7 || ny < 0 || ny > 7 || nz < 0 || nz > 7) { needsAccessor = true; continue; }
            const uint32_t nOff = (uint32_t(nx) << 6) | (uint32_t(ny) << 3) | uint32_t(nz);
            if (leaf.isActive(nOff) && d_sign[leaf.getValue(nOff)] > int8_t(0)) touchesExterior = true;
        }
        if (!touchesExterior && needsAccessor) {
            const nanovdb::Coord origin = leaf.origin();
            auto acc = d_grid->getAccessor();
            for (int k = 0; k < 6 && !touchesExterior; ++k) {
                const int nx = lx + off[k][0], ny = ly + off[k][1], nz = lz + off[k][2];
                if (nx >= 0 && nx <= 7 && ny >= 0 && ny <= 7 && nz >= 0 && nz <= 7) continue;
                const nanovdb::Coord nijk(origin[0] + nx, origin[1] + ny, origin[2] + nz);
                if (acc.isActive(nijk) && d_sign[acc.getValue(nijk)] > int8_t(0)) touchesExterior = true;
            }
        }
        if (!touchesExterior) d_udf[v] = interiorFloor;
    }
};// sdf_detail::IsoMagnitudeFunctor

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Barrier pruning.

/// @brief Retain voxels outside the sqrt(3)/2-voxel barrier around {udf == isoValue}.
/// @note Distances and the precomputed squared threshold are in world units.
template <typename BuildT>
struct UDFBarrierPruneMaskFunctor
{
    static constexpr int MaxThreadsPerBlock         = LEAF_SIZE;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(
        const nanovdb::NanoGrid<BuildT>* d_grid,
        const float*                     d_udf,           // UDF sidecar, WORLD units
        float                            isoValue,        // signed surface = { udf == isoValue }, world
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
            const float d = d_udf[n] - isoValue;
            if (d * d >= barrierSqWorld)                 // retain non-barrier voxels
                resultMask.setOnAtomic(threadID);
        }
    }
};

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Non-barrier signing.
// For one closed surface, only the component containing its minimum-x voxel is exterior.

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
// Barrier signing.
// Heuristic anchors come from an immutable snapshot, making the result order-independent.

static constexpr uint32_t INVALID_TRIANGLE = 0xFFFFFFFFu;  // nearest-triangle-index sentinel

/// @brief Test whether an exterior neighbour's nearest triangle also places @a q outside.
/// @note Double precision prevents large-coordinate cancellation from changing the sign test.
__hostdev__ inline bool
barrierExteriorProof(uint64_t nv, const nanovdb::Coord& nijk, const nanovdb::Vec3d& q_xyz,
                     const int8_t* d_sign, const uint32_t* d_index,
                     const nanovdb::Vec3f* d_points, const nanovdb::Vec3i* d_triangles,
                     const nanovdb::Map& map, double isoValueIndex)
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

    // Move the reference point from the mesh to {udf == isoValue} along the local normal.
    const nanovdb::Vec3d base = cp + dn * isoValueIndex;

    nanovdb::Vec3d dq = q_xyz - base; dq.normalize();  // surface -> q
    return dn.dot(dq) > 0.0;                           // same side => q is exterior
}


/// @brief Perform one order-independent round of ball-intersection certification.
/// @note Strictly overlapping surface-free balls certify the same sign; tangency proves nothing.
/// Voxels certified both ways remain undecided.
template <typename BuildT>
struct BallCertifyFunctor
{
    static constexpr int MaxThreadsPerBlock         = LEAF_SIZE;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(
        const NanoGrid<BuildT>* d_grid,
        const int8_t*           d_labelIn,   // +1 ext / -1 int / 0 undecided
        int8_t*                 d_labelOut,
        const float*            d_udf,       // unsigned distance sidecar, WORLD units
        float                   voxelSize,
        uint32_t*               d_changed,       // incremented once per newly decided voxel
        int                     radius,          // stencil half-width in voxels; 1 = the 26 neighbours
        float                   isoValue)        // surface signed = { udf == isoValue }, world units
    {
        const int leafID = blockIdx.x, n = threadIdx.x;
        const auto& leaf = d_grid->tree().template getFirstNode<0>()[leafID];
        if (!leaf.isActive(uint32_t(n))) return;

        const uint64_t qv = leaf.getValue(uint32_t(n));
        const int8_t   ql = d_labelIn[qv];
        if (ql != int8_t(0)) { d_labelOut[qv] = ql; return; }   // already certain: carry through

        const nanovdb::Coord local  = nanovdb::NanoLeaf<BuildT>::OffsetToLocalCoord(uint32_t(n));
        const int            lx = local[0], ly = local[1], lz = local[2];
        const nanovdb::Coord origin = leaf.origin();
        // Radii are distances to the surface being signed, not to the mesh. |udf - isoValue| is a
        // 1-Lipschitz under-estimate of that near the medial axis, which is the safe direction: it
        // shrinks the balls, so it can only certify fewer voxels, never wrongly.
        const float          dq  = fabsf(d_udf[qv] - isoValue);
        const float          eps = 1e-5f * voxelSize;

        bool ext = false, inr = false;

        // The ball test needs only the neighbour's distance and its label, so both passes reduce to
        // two loads and a comparison against the offset length -- one of three constants.
        auto consider = [&] __device__ (uint64_t nv, int dx, int dy, int dz) {
            const int8_t ln = d_labelIn[nv];
            if (ln == int8_t(0)) return;                                   // neighbour not certain yet
            const float len = sqrtf(float(dx*dx + dy*dy + dz*dz)) * voxelSize;
            if (fabsf(d_udf[nv] - isoValue) + dq <= len + eps) return;      // balls do not overlap
            if (ln > 0) ext = true; else inr = true;
        };

        // Pass 1: neighbours inside this leaf, straight off the leaf buffer.
        for (int dx = -radius; dx <= radius; ++dx) {
            const int nx = lx + dx; if (nx < 0 || nx > 7) continue;
            for (int dy = -radius; dy <= radius; ++dy) {
                const int ny = ly + dy; if (ny < 0 || ny > 7) continue;
                for (int dz = -radius; dz <= radius; ++dz) {
                    const int nz = lz + dz; if (nz < 0 || nz > 7) continue;
                    if (!dx && !dy && !dz) continue;
                    const uint32_t nOff = (uint32_t(nx) << 6) | (uint32_t(ny) << 3) | uint32_t(nz);
                    if (leaf.isActive(nOff)) consider(leaf.getValue(nOff), dx, dy, dz);
                }
            }
        }

        // Pass 2: the rest of the stencil, which crosses the leaf boundary. One reused accessor.
        if (lx < radius || lx > 7 - radius || ly < radius || ly > 7 - radius ||
            lz < radius || lz > 7 - radius) {
            auto acc = d_grid->getAccessor();
            for (int dx = -radius; dx <= radius; ++dx) {
                const int nx = lx + dx;
                for (int dy = -radius; dy <= radius; ++dy) {
                    const int ny = ly + dy;
                    for (int dz = -radius; dz <= radius; ++dz) {
                        const int nz = lz + dz;
                        if (!dx && !dy && !dz) continue;
                        if (nx >= 0 && nx <= 7 && ny >= 0 && ny <= 7 && nz >= 0 && nz <= 7) continue;
                        const nanovdb::Coord nijk(origin[0] + nx, origin[1] + ny, origin[2] + nz);
                        if (acc.isActive(nijk)) consider(acc.getValue(nijk), dx, dy, dz);
                    }
                }
            }
        }

        if (ext && inr) {
            d_labelOut[qv] = int8_t(0);
            return;
        }
        if (!ext && !inr) { d_labelOut[qv] = int8_t(0); return; }
        d_labelOut[qv] = ext ? int8_t(1) : int8_t(-1);
        atomicAdd(d_changed, 1u);
    }
};

/// @brief Complete the ball-certified sign field, defaulting unresolved voxels to interior.
struct BallFinalizeFunctor
{
    __device__ void operator()(size_t v, const int8_t* d_ball, int8_t* d_signOut) const
    {
        if (v == 0) { d_signOut[0] = int8_t(1); return; }   // slot 0 = background = exterior
        const int8_t l = d_ball[v];
        if (l != int8_t(0)) { d_signOut[v] = l; return; }
        d_signOut[v] = int8_t(-1);
    }
};

/// @brief Preserve non-barrier signs and classify every barrier voxel as interior.
struct BarrierToInteriorFunctor
{
    __device__ void operator()(const uint64_t slot, const int8_t* d_signIn, int8_t* d_signOut) const
    {
        const int8_t s = d_signIn[slot];
        d_signOut[slot] = (s != int8_t(0)) ? s : int8_t(-1);
    }
};// sdf_detail::BarrierToInteriorFunctor

/// @brief Sign barrier voxels using exterior neighbours and their nearest triangles.
/// @note Input anchors are immutable; unresolved voxels default to interior.
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
        nanovdb::Map            map,          // world<->index transform (by value)
        double                  isoValueIndex)// surface signed = { udf == isoValue }, INDEX units
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
                    if (barrierExteriorProof(leaf.getValue(nOff), nijk, q_xyz, d_signIn, d_index,
                                             d_points, d_triangles, map, isoValueIndex)) {
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
                        if (barrierExteriorProof(acc.getValue(nijk), nijk, q_xyz, d_signIn, d_index,
                                                 d_points, d_triangles, map, isoValueIndex)) {
                            exterior = true; break;
                        }
                    }
        }

        d_signOut[qv] = exterior ? int8_t(1) : int8_t(-1);
    }
};

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Inactive voxels inside materialized leaves.

/// @brief Mark inactive interior voxels with a shared-memory Jacobi flood.
/// @note Active voxels seed or bound the flood and therefore act as walls.
template <typename BuildT>
struct FillLeafInvertMaskFunctor
{
    static constexpr int MaxThreadsPerBlock         = LEAF_SIZE;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(const NanoGrid<BuildT>* d_grid,
                               const int8_t*           d_sign,        // completed sign sidecar
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
// Childless lower (8^3) and upper (128^3) tiles, filled bottom-up.

/// @brief Find the finest childless tile containing the leaf region at @a c.
/// @return 0 for a root value tile or leaf, 1 for lower, 2 for upper, and 3 for an absent root region.
/// On 1 or 2, @a nodeIdx and @a slot identify the childless tile.
template <typename BuildT>
__hostdev__ inline int
probeChildlessSlot(const NanoGrid<BuildT>& grid, const nanovdb::Coord& c,
                   uint64_t& nodeIdx, uint32_t& slot)
{
    using UpperT = NanoUpper<BuildT>;
    using LowerT = NanoLower<BuildT>;
    const auto& tree = grid.tree();
    const auto* tile = tree.root().probeTile(c);
    if (!tile) return 3;                                        // absent root region -> root sidecar
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

/// @brief Record leaf-face interior/exterior evidence on adjacent childless tiles.
/// @note Each 8x8 face abuts one 8^3 region, so one probe resolves its target.
template <typename BuildT>
struct LeafFaceSeedFunctor
{
    static constexpr int MaxThreadsPerBlock         = 384;  // 6 faces × 64 cells
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(const NanoGrid<BuildT>*  d_grid,
                               const int8_t*            d_sign,        // completed signs
                               const nanovdb::Mask<3>*  d_leafInvert,  // leaf invert masks
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

/// @brief Record lower-node face evidence on adjacent childless upper tiles.
/// @note Refined slots are handled at the finer level; lower neighbours are handled by the flood.
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

/// @brief Flood interior state through adjacent childless slots, using refined slots as walls.
/// @note Mixed face evidence defaults to exterior.
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
// Absent root regions, represented by one flag per 4096^3 cell.

/// @brief Mark root cells containing upper children as walls.
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

/// @brief Accumulate interior and exterior evidence on absent root regions from every tree level.
/// @note Face classification mirrors LeafFaceSeedFunctor and LowerFaceSeedFunctor.
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

/// @brief Flood interior flags through non-wall root cells from unambiguous interior seeds.
/// @note Mixed interior/exterior evidence defaults to exterior.
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
// Full-domain sign query over the grid and its per-level sidecars.

/// @brief Query the completed sign at any index coordinate without mutating the grid.
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

/// @brief Compose per-surface signs under the selected nesting rule.
/// @note The enclosing count is the surface depth plus its local inside state.
struct ResolveNestingFunctor
{
    __device__ void operator()(size_t v, const uint32_t* d_surfaceLabel, const uint32_t* d_depth,
                               uint32_t surfaceCount, bool evenOdd, int8_t* d_sign) const
    {
        if (v == 0) return;                                  // slot 0 is the background
        const uint32_t s = d_surfaceLabel[v];
        if (s >= surfaceCount) return;
        const uint32_t enclosing = d_depth[s] + (d_sign[v] < int8_t(0) ? 1u : 0u);
        const bool interior = evenOdd ? ((enclosing & 1u) != 0u) : (enclosing != 0u);
        d_sign[v] = interior ? int8_t(-1) : int8_t(1);
    }
};// sdf_detail::ResolveNestingFunctor

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
template <typename BufferT>
GridHandle<BufferT>
SurfaceSigner<BuildT>::computeDerivedTopology(const GridT* d_srcGrid, const float* d_udf,
                                          float voxelSize, float isoValue, const BufferT& buffer)
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
        d_srcGrid, d_udf, isoValue, barrierSqWorld, d_retainMask);
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
    const uint32_t exteriorRep = uint32_t(minKey & 0xFFFFFFFFull);
    if (mVerbose==1) mTimer.stop();

    // (2) Per-voxel sign: +1 exterior / -1 interior (slot 0 = background +1).
    mVoxelSign = nanovdb::cuda::DeviceBuffer::create((activeCount + 1) * sizeof(int8_t), nullptr, false);
    cudaCheck(cudaMemsetAsync(mVoxelSign.deviceData(), 1, (activeCount + 1) * sizeof(int8_t), mStream));// all +1
    using SignOp = SignNonBarrierFunctor<BuildT>;
    if (mVerbose==1) mTimer.start("Sign: write per-voxel signs");
    util::cuda::operatorKernel<SignOp><<<leafCount, SignOp::MaxThreadsPerBlock, 0, mStream>>>(
        d_grid, d_voxelLabel, exteriorRep, deviceVoxelSign());
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
void SurfaceSigner<BuildT>::signBarrierAsInterior(const GridT* d_grid)
{
    const uint64_t activeCount = activeCountOf(d_grid);
    const uint64_t slots       = activeCount + 1;

    mSignedVoxelSign = nanovdb::cuda::DeviceBuffer::create(slots * sizeof(int8_t), nullptr, false);
    cudaCheck(cudaMemsetAsync(mSignedVoxelSign.deviceData(), 0, slots * sizeof(int8_t), mStream));
    cudaCheck(cudaMemsetAsync(mSignedVoxelSign.deviceData(), 1, sizeof(int8_t), mStream)); // slot 0 = +1
    if (leafCountOf(d_grid) == 0) return;

    if (mVerbose==1) mTimer.start("Sign: barrier voxels (all interior)");
    util::cuda::lambdaKernel<<<(unsigned int)((slots + 255) / 256), 256, 0, mStream>>>(
        slots, BarrierToInteriorFunctor{}, deviceOriginalVoxelSign(), deviceSignedVoxelSign());
    cudaCheckError();
    // The kernel also writes slot 0, which holds +1 rather than a voxel sign; restore it.
    const int8_t one = 1;
    cudaCheck(cudaMemcpyAsync(mSignedVoxelSign.deviceData(), &one, sizeof(int8_t),
                              cudaMemcpyHostToDevice, mStream));
    cudaCheck(cudaStreamSynchronize(mStream));
    if (mVerbose==1) mTimer.stop();
}// SurfaceSigner<BuildT>::signBarrierAsInterior

template <typename BuildT>
void SurfaceSigner<BuildT>::signBarrier(const GridT* d_grid, const uint32_t* d_index,
                                    const nanovdb::Vec3f* d_points, const nanovdb::Vec3i* d_triangles,
                                    const nanovdb::Map& map, float isoValue, float voxelSize)
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
        d_index, d_points, d_triangles, map,
        (voxelSize > 0.f) ? double(isoValue) / double(voxelSize) : 0.0);
    cudaCheckError();
    if (mVerbose==1) mTimer.stop();
}// SurfaceSigner<BuildT>::signBarrier

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

/// Ball-intersection certification of the barrier voxels. Seeds are the non-barrier signs already in
/// deviceOriginalVoxelSign(); each Jacobi round lets an undecided voxel take the sign of any certain
/// neighbour whose ball overlaps its own. Ping-pongs two label buffers so no round reads what it
/// writes, and stops when a round decides nothing.
template <typename BuildT>
void SurfaceSigner<BuildT>::signBarrierByBalls(const GridT* d_grid, const float* d_udf,
                                               float voxelSize, int maxRounds, int radius,
                                               float isoValue)
{
    const uint64_t activeCount = activeCountOf(d_grid);
    const uint32_t leafCount   = leafCountOf(d_grid);
    const std::size_t bytes    = std::size_t(activeCount + 1) * sizeof(int8_t);

    mSignedVoxelSign = nanovdb::cuda::DeviceBuffer::create(bytes, nullptr, false);
    if (leafCount == 0) {
        cudaCheck(cudaMemsetAsync(mSignedVoxelSign.deviceData(), 1, sizeof(int8_t), mStream));
        return;
    }

    // Seed both buffers from the non-barrier signs; barrier voxels start at 0.
    auto labels  = nanovdb::cuda::DeviceBuffer::create(bytes, nullptr, false);
    auto scratch = nanovdb::cuda::DeviceBuffer::create(bytes, nullptr, false);
    cudaCheck(cudaMemcpyAsync(labels.deviceData(), deviceOriginalVoxelSign(), bytes,
                              cudaMemcpyDeviceToDevice, mStream));
    cudaCheck(cudaMemcpyAsync(scratch.deviceData(), deviceOriginalVoxelSign(), bytes,
                              cudaMemcpyDeviceToDevice, mStream));

    auto  counter   = nanovdb::cuda::DeviceBuffer::create(sizeof(uint32_t), nullptr, false);
    auto* d_changed = static_cast<uint32_t*>(counter.deviceData());
    int8_t* labelIn  = static_cast<int8_t*>(labels.deviceData());
    int8_t* labelOut = static_cast<int8_t*>(scratch.deviceData());

    using Op = BallCertifyFunctor<BuildT>;
    if (mVerbose==1) mTimer.start("Sign: barrier voxels (ball certification)");
    uint32_t changed = 0;
    for (int r = 0; r < maxRounds; ++r) {
        cudaCheck(cudaMemsetAsync(d_changed, 0, sizeof(uint32_t), mStream));
        util::cuda::operatorKernel<Op><<<leafCount, Op::MaxThreadsPerBlock, 0, mStream>>>(
            d_grid, labelIn, labelOut, d_udf, voxelSize, d_changed, radius, isoValue);
        cudaCheckError();
        cudaCheck(cudaMemcpyAsync(&changed, d_changed, sizeof(uint32_t), cudaMemcpyDeviceToHost, mStream));
        cudaCheck(cudaStreamSynchronize(mStream));
        std::swap(labelIn, labelOut);
        if (changed == 0) break;
    }
    if (mVerbose==1) mTimer.stop();

    // Complete the field: unproven voxels default to interior.
    util::cuda::lambdaKernel<<<(unsigned int)((activeCount + 256) / 256), 256, 0, mStream>>>(
        activeCount + 1, BallFinalizeFunctor{}, labelIn, deviceSignedVoxelSign());
    cudaCheckError();
    // The temporary label buffers must outlive the finalization kernel.
    cudaCheck(cudaStreamSynchronize(mStream));
}// SurfaceSigner<BuildT>::signBarrierByBalls

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
typename MeshToSDF<BuildT>::Handle MeshToSDF<BuildT>::
    build()
{
    // Time the five compute phases. Blind-data assembly follows the final mark and is excluded.
    int  phase = 0;
    auto mark  = [&, prev = std::chrono::steady_clock::time_point{}]() mutable {
        cudaCheck(cudaStreamSynchronize(mStream));
        const auto now = std::chrono::steady_clock::now();
        if (phase) mPhaseMs[phase - 1] = std::chrono::duration<float, std::milli>(now - prev).count();
        prev = now;
        ++phase;
    };

    // The isovalue only ever moves the surface outward, and a negative one would ask for a level set
    // the unsigned distance does not have.
    if (mIsoValue < 0.f)
        throw std::runtime_error("MeshToSDF: setIsoValue() must be >= 0");

    mark();
    this->rasterize();          // mesh -> narrow band, with the UDF and nearest-triangle sidecars
    mark();
    this->partition();          // components of the UN-pruned band = one per closed surface
    mark();
    for (uint32_t i = 0; i < uint32_t(mSurfaces.size()); ++i)
        this->signSurface(i);   // carve surface i out, prune its barrier shell, label, and sign it alone
    mark();
    this->composeByInclusion();  // nesting parity per surface, then merge the signs onto the band
    mark();
    this->postProcess();         // signs are settled: fold the magnitudes, floor the interior
    this->fillOnOriginal();      // extend those signs off the band as invert masks
    mark();
    Handle handle = this->bakeBlindData();
    this->releaseIntermediates();
    return handle;
}// MeshToSDF<BuildT>::build

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
void MeshToSDF<BuildT>::rasterize()
{
    const float voxelSize = float(mMap.getVoxelSize()[0]);

    // The signed surface stands mIsoValue out from the mesh, so the band has to reach that far again
    // to still hold mBandWidth voxels beyond it. Rasterizing 3 + isoValue/voxelSize wide and signing
    // the isosurface lands mBandWidth voxels of band outside it -- the width the caller asked for,
    // measured where they meant it.
    const float extra = (voxelSize > 0.f) ? mIsoValue / voxelSize : 0.f;

    MeshToGrid<BuildT> converter(mPoints, mPointCount, mTriangles, mTriangleCount, mMap, mStream);
    converter.setVerbose(mVerbose);
    converter.setNarrowBandWidth(mBandWidth + extra);
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

/// @brief Sign one closed surface independently of the other surfaces.
/// @note A single-surface mesh reuses the rasterized grid and sidecars without carving.
template <typename BuildT>
void MeshToSDF<BuildT>::signSurface(uint32_t surface)
{
    using Traits = util::cuda::DeviceGridTraits<BuildT>;

    SurfaceField&  sf         = mSurfaces[surface];
    const auto*    d_orig     = this->deviceGrid();
    const uint32_t origLeaves = Traits::getTreeData(d_orig).mNodeCount[0];
    const float    voxelSize  = float(mMap.getVoxelSize()[0]);

    // Carving renumbers value slots, so transfer sidecars by injection rather than memcpy.
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
            d_orig, d_sub, static_cast<const uint32_t*>(mIndex.deviceData()),
            static_cast<uint32_t*>(sf.subIndex.deviceData()));
        cudaCheckError();
        cudaCheck(cudaStreamSynchronize(mStream));
        if (mVerbose==1) timer.stop();
    }

    const auto* d_grid = this->surfaceGrid(surface);
    sf.signer = std::make_unique<Signer>(mStream);
    sf.signer->setVerbose(mVerbose);

    // Pruning the barrier separates this surface's interior and exterior components.
    sf.derived = sf.signer->computeDerivedTopology(d_grid, this->surfaceUdf(surface), voxelSize, mIsoValue);
    const auto* d_derived = sf.derived.template deviceGrid<BuildT>();

    sf.cc = std::make_unique<ConnectedComponents<BuildT>>(d_derived, mStream);
    sf.cc->setVerbose(mVerbose);
    sf.ccLabels = sf.cc->getVoxelLabelsAndCount();
    cudaCheck(cudaStreamSynchronize(mStream));

    // Sign the components, inject them onto the unpruned grid, then resolve the barrier.
    sf.signer->signNonBarrier(d_derived, sf.ccLabels.first);
    sf.signer->injectSignsToOriginal(d_grid, d_derived);
    switch (mBarrierSigning) {
    case BarrierSigning::Ball:
        sf.signer->signBarrierByBalls(d_grid, this->surfaceUdf(surface), voxelSize,
                                      32, mBallStencilRadius, mIsoValue);
        break;
    case BarrierSigning::Heuristic:
        sf.signer->signBarrier(d_grid, this->surfaceIndex(surface), mPoints, mTriangles, mMap,
                               mIsoValue, voxelSize);
        break;
    case BarrierSigning::Interior:
    default:
        sf.signer->signBarrierAsInterior(d_grid);
        break;
    }

    // Extend signs off-band so other surfaces can query this field.
    sf.signer->fillLeafInvertMask(d_grid);
    sf.signer->fillCoarseInvertMasks(d_grid);
    sf.signer->fillRootInteriorMask(d_grid);
    cudaCheck(cudaStreamSynchronize(mStream));

    // These inputs are not needed by composition; retain only the surface grid and completed field.
    sf.subUdf   = Buffer();
    sf.subIndex = Buffer();
    sf.derived  = Handle();
    sf.cc.reset();
    sf.ccLabels = {nullptr, 0};
}// MeshToSDF<BuildT>::signSurface

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

/// Recover the nesting depth of every closed surface from the per-surface fields, then merge those
/// fields into one sign array on the rasterized band, negating the odd-depth surfaces on the way in.
/// Requires every field to be complete through its invert-mask fill.
template <typename BuildT>
void MeshToSDF<BuildT>::composeByInclusion()
{
    using Traits = util::cuda::DeviceGridTraits<BuildT>;

    const uint32_t numSurfaces = uint32_t(mSurfaces.size());
    if (numSurfaces == 0) return;

    // Nothing encloses a lone surface, and an uncarved one already carries its signs on the rasterized
    // band — so there is no depth to recover and nothing to gather. Aliasing here is what keeps the
    // single-surface case free of the extra full-length sign array a merge would allocate.
    if (numSurfaces == 1 && !mSurfaces[0].subGrid.bufferSize()) {
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
    auto  repBuf   = Buffer::create(numSurfaces * sizeof(unsigned long long), nullptr, false);
    auto* d_repKey = static_cast<unsigned long long*>(repBuf.deviceData());
    cudaCheck(cudaMemsetAsync(d_repKey, 0xFF, numSurfaces * sizeof(unsigned long long), mStream));
    {
        using RepOp = sdf_detail::SurfaceRepFunctor<BuildT>;
        util::cuda::operatorKernel<RepOp><<<origLeaves, RepOp::MaxThreadsPerBlock, 0, mStream>>>(
            d_orig, mSurfaceLabels.first, numSurfaces, d_repKey);
        cudaCheckError();
    }

    // (2) Ask each surface's own field about every surface's representative. Each was completed
    //     through the invert-mask fill, so it answers off its band too — including at the other bands.
    auto  incBuf = Buffer::create(std::size_t(numSurfaces) * numSurfaces * sizeof(int8_t), nullptr, false);
    auto* d_inc  = static_cast<int8_t*>(incBuf.deviceData());  // d_inc[i*numSurfaces+j] = field i's sign at surface j
    for (uint32_t i = 0; i < numSurfaces; ++i) {
        auto&       phi   = *mSurfaces[i].signer;
        const auto* d_sub = this->surfaceGrid(i);
        using ProbeOp = sdf_detail::InclusionProbeFunctor<BuildT>;
        util::cuda::lambdaKernel<<<1, numSurfaces, 0, mStream>>>(
            numSurfaces, ProbeOp{}, d_sub, d_repKey, phi.deviceSignedVoxelSign(),
            phi.deviceLeafInvertMask(), phi.deviceLowerInvertMask(), phi.deviceUpperInvertMask(),
            phi.deviceRootInterior(), phi.rootTileMin(), phi.rootTileDims(), d_inc + std::size_t(i) * numSurfaces);
        cudaCheckError();
    }

    // (3) Nesting depth = how many other surfaces report this one as inside them. Counting a column is
    //     enough: enclosure is transitive between non-intersecting surfaces, so a surface nested d deep
    //     is reported inside by exactly d others — the inclusion forest never has to be built.
    std::vector<int8_t> inc(std::size_t(numSurfaces) * numSurfaces);
    cudaCheck(cudaMemcpyAsync(inc.data(), d_inc, inc.size() * sizeof(int8_t), cudaMemcpyDeviceToHost, mStream));
    cudaCheck(cudaStreamSynchronize(mStream));

    std::vector<uint32_t> nestingDepth(numSurfaces, 0u);
    for (uint32_t j = 0; j < numSurfaces; ++j)
        for (uint32_t i = 0; i < numSurfaces; ++i)
            if (i != j && inc[std::size_t(i) * numSurfaces + j] < 0) ++nestingDepth[j];

    // (4) Merge. Gather every surface's signs back onto the rasterized band — the surfaces partition
    //     its active voxels, so the per-surface injections write disjoint slots and together cover all —
    //     then resolve each voxel against the nesting rule in place.
    mComposedSign = Buffer::create((origActive + 1) * sizeof(int8_t), nullptr, false);
    mSign = static_cast<int8_t*>(mComposedSign.deviceData());
    cudaCheck(cudaMemsetAsync(mSign, 1, (origActive + 1) * sizeof(int8_t), mStream));  // slot 0 = background +1
    using InjectOp = util::cuda::InjectGridDataFunctor<BuildT, int8_t>;
    for (uint32_t i = 0; i < numSurfaces; ++i) {
        const auto*    d_sub     = this->surfaceGrid(i);
        const uint32_t subLeaves = Traits::getTreeData(d_sub).mNodeCount[0];
        util::cuda::operatorKernel<InjectOp><<<subLeaves, InjectOp::MaxThreadsPerBlock, 0, mStream>>>(
            d_sub, d_orig, mSurfaces[i].signer->deviceSignedVoxelSign(), mSign);
        cudaCheckError();
    }
    auto  depthBuf = Buffer::create(numSurfaces * sizeof(uint32_t), nullptr, false);
    auto* d_depth  = static_cast<uint32_t*>(depthBuf.deviceData());
    cudaCheck(cudaMemcpyAsync(d_depth, nestingDepth.data(), numSurfaces * sizeof(uint32_t),
                              cudaMemcpyHostToDevice, mStream));
    util::cuda::lambdaKernel<<<(unsigned int)((origActive + 256) / 256), 256, 0, mStream>>>(
        origActive + 1, sdf_detail::ResolveNestingFunctor{}, mSurfaceLabels.first, d_depth,
        numSurfaces, mNestingRule == NestingRule::EvenOdd, mSign);
    cudaCheckError();
    cudaCheck(cudaStreamSynchronize(mStream));
    if (mVerbose==1) timer.stop();
}// MeshToSDF<BuildT>::composeByInclusion

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

/// @brief Fold UDF magnitudes around the isovalue after all signs are settled.
/// @note Interior magnitudes are floored at half a voxel diagonal.
template <typename BuildT>
void MeshToSDF<BuildT>::postProcess()
{
    if (mIsoValue == 0.f || mSurfaces.empty() || mSign == nullptr) return;

    const float voxelSize = float(mMap.getVoxelSize()[0]);

    // sqrt(3)/2 voxels: the same half-diagonal the barrier test uses, so the floor lands exactly at
    // the edge of the shell the oracle was responsible for.
    const float interiorFloor = 0.8660254f * voxelSize;

    using Op = sdf_detail::IsoMagnitudeFunctor<BuildT>;
    const uint32_t leaves = util::cuda::DeviceGridTraits<BuildT>::getTreeData(this->deviceGrid()).mNodeCount[0];
    if (leaves)
        util::cuda::operatorKernel<Op><<<leaves, Op::MaxThreadsPerBlock, 0, mStream>>>(
            this->deviceGrid(), static_cast<float*>(mUDF.deviceData()), mSign, mIsoValue, interiorFloor);
    cudaCheckError();

    // Slot 0 is the background sentinel, not a voxel. The kernel above walks active voxels and never
    // reaches it, but it is restated here so the value is the exterior background whatever the
    // rasterizer left, rather than something an earlier stage happened to write.
    const float bg = mBandWidth * voxelSize;
    cudaCheck(cudaMemcpyAsync(mUDF.deviceData(), &bg, sizeof(float), cudaMemcpyHostToDevice, mStream));
    cudaCheck(cudaStreamSynchronize(mStream));
}// MeshToSDF<BuildT>::postProcess

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

/// Extend the composed sign off the band, over the whole rasterized grid. When the composition aliased
/// a lone uncarved surface, the fill it already ran covered this very grid with these very signs, so
/// it is adopted as-is.
template <typename BuildT>
void MeshToSDF<BuildT>::fillOnOriginal()
{
    if (mSurfaces.empty()) return;

    if (!mComposedSign.size()) {                 // composition aliased surface 0 -> its fill is the answer
        mOrigSigner  = std::move(mSurfaces[0].signer);
        mFinalSigner = mOrigSigner.get();
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

/// @brief Append the SDF and sign-extension sidecars as blind-data channels.
template <typename BuildT>
typename MeshToSDF<BuildT>::Handle MeshToSDF<BuildT>::bakeBlindData()
{
    using Traits = util::cuda::DeviceGridTraits<BuildT>;
    namespace tc = nanovdb::tools::cuda;

    const auto*    d_grid = this->deviceGrid();
    const uint64_t slots  = Traits::getActiveVoxelCount(d_grid) + 1;
    const auto&    tree   = Traits::getTreeData(d_grid);
    const uint32_t leaves = tree.mNodeCount[0], lowers = tree.mNodeCount[1], uppers = tree.mNodeCount[2];

    // Channel 0. The pipeline keeps sign and magnitude apart; a consumer wants the product.
    auto  sdfBuf = Buffer::create(slots * sizeof(float), nullptr, false);
    auto* d_sdf  = static_cast<float*>(sdfBuf.deviceData());
    util::cuda::lambdaKernel<<<(unsigned int)((slots + 255) / 256), 256, 0, mStream>>>(
        slots, sdf_detail::SignedDistanceFunctor{}, this->deviceUDF(), mSign, d_sdf);
    cudaCheckError();

    Handle h = tc::addBlindData<BuildT, float>(d_grid, d_sdf, slots,
                   GridBlindDataClass::ChannelArray, GridBlindDataSemantic::LevelSet, "sdf",
                   Buffer(), mStream);

    // Channels 1-3. A Mask<N> is a flat bit array, so it travels as the uint64 words it already is.
    auto addMask = [&](const void* d_src, uint64_t words, const char* name) {
        if (!words) return;
        h = tc::addBlindData<BuildT, uint64_t>(h.template deviceGrid<BuildT>(),
                static_cast<const uint64_t*>(d_src), words,
                GridBlindDataClass::ChannelArray, GridBlindDataSemantic::Unknown, name,
                Buffer(), mStream);
    };
    addMask(this->deviceLeafInvertMask(),  uint64_t(leaves) * (sizeof(nanovdb::Mask<3>) / 8), "leaf_invert");
    addMask(this->deviceLowerInvertMask(), uint64_t(lowers) * (sizeof(nanovdb::Mask<4>) / 8), "lower_invert");
    addMask(this->deviceUpperInvertMask(), uint64_t(uppers) * (sizeof(nanovdb::Mask<5>) / 8), "upper_invert");

    // Channels 4-5. The root sidecar covers regions with no node at all, so unlike the masks above it
    // is not indexed by a node and needs its origin and dims carried alongside.
    const nanovdb::Coord tileMin = this->rootTileMin(), dims = this->rootTileDims();
    const uint64_t       cells   = uint64_t(dims[0]) * dims[1] * dims[2];
    if (cells) {
        h = tc::addBlindData<BuildT, uint8_t>(h.template deviceGrid<BuildT>(),
                this->deviceRootInterior(), cells,
                GridBlindDataClass::ChannelArray, GridBlindDataSemantic::Unknown, "root_interior",
                Buffer(), mStream);

        const int32_t extent[6] = {tileMin[0], tileMin[1], tileMin[2], dims[0], dims[1], dims[2]};
        int32_t*      d_extent  = nullptr;
        cudaCheck(cudaMalloc(&d_extent, sizeof(extent)));
        cudaCheck(cudaMemcpyAsync(d_extent, extent, sizeof(extent), cudaMemcpyHostToDevice, mStream));
        cudaCheck(cudaStreamSynchronize(mStream));
        h = tc::addBlindData<BuildT, int32_t>(h.template deviceGrid<BuildT>(), d_extent, 6,
                GridBlindDataClass::ChannelArray, GridBlindDataSemantic::Unknown, "root_extent",
                Buffer(), mStream);
        cudaCheck(cudaFree(d_extent));
    }
    return h;
}// MeshToSDF<BuildT>::bakeBlindData

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
void MeshToSDF<BuildT>::releaseIntermediates()
{
    mSurfaceCC.reset();
    mSurfaceLabels = {nullptr, 0};
    mSurfaces.clear();
    mIndex = Buffer();
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

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

template <typename BuildT>
const GridHandle<nanovdb::cuda::DeviceBuffer>&
MeshToSDF<BuildT>::surfaceGridHandle(uint32_t i) const
{ return mSurfaces[i].subGrid.bufferSize() ? mSurfaces[i].subGrid : mGridHandle; }

template <typename BuildT>
const uint32_t* MeshToSDF<BuildT>::surfaceIndex(uint32_t i) const
{ return static_cast<const uint32_t*>(mSurfaces[i].subIndex.size() ? mSurfaces[i].subIndex.deviceData()
                                                                   : mIndex.deviceData()); }

template <typename BuildT>
const float* MeshToSDF<BuildT>::surfaceUdf(uint32_t i) const
{ return static_cast<const float*>(mSurfaces[i].subUdf.size() ? mSurfaces[i].subUdf.deviceData()
                                                              : mUDF.deviceData()); }

template <typename BuildT>
const NanoGrid<BuildT>* MeshToSDF<BuildT>::surfaceGrid(uint32_t i) const
{ return this->surfaceGridHandle(i).template deviceGrid<BuildT>(); }

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

} // namespace tools::cuda

} // namespace nanovdb

#endif // NVIDIA_TOOLS_CUDA_MESHTOSDF_CUH_HAS_BEEN_INCLUDED
