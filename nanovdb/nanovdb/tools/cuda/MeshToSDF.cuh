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
#include <nanovdb/cuda/Buffer.h>
#include <nanovdb/cuda/DeviceResource.h>
#include <nanovdb/math/Proximity.h>                    // closestPointOnTriangleToPoint
#include <nanovdb/tools/cuda/ConnectedComponents.cuh>
#include <nanovdb/tools/cuda/MeshToGrid.cuh>
#include <nanovdb/tools/cuda/PruneGrid.cuh>
#include <nanovdb/util/cuda/Injection.cuh>             // InjectGridDataFunctor
#include <nanovdb/util/cuda/DeviceGridTraits.cuh>
#include <nanovdb/util/cuda/Timer.h>
#include <nanovdb/util/cuda/Util.h>                    // operatorKernel, cudaCheck
#include <nanovdb/tools/cuda/AddBlindData.cuh>

#include <cstddef>
#include <tuple>
#include <utility>

namespace nanovdb {

namespace tools::cuda {

namespace sdf_detail {

/// @brief Typed device storage of the pipeline, allocated stream-ordered from the default device resource.
template <typename T>
using BufferT = nanovdb::cuda::Buffer<T, nanovdb::cuda::ResourceRef<nanovdb::cuda::DeviceResource>>;

/// @brief Allocate @a count uninitialized elements ordered on @a stream.
/// @note A ResourceRef cannot be default-constructed, so an empty buffer (count 0) is also what
///       initializes a buffer member and what library calls receive as the prototype they take the
///       allocation resource from.
template <typename T>
BufferT<T> allocate(std::size_t count, cudaStream_t stream)
{
    return BufferT<T>(stream, nanovdb::cuda::default_resource<nanovdb::cuda::DeviceResource>(), count,
                      nanovdb::cuda::noInit);
}

} // namespace sdf_detail

/// @brief Convert closed triangle surfaces to a narrow-band signed distance field.
/// @tparam BuildT Build type of the index grid (e.g. nanovdb::ValueOnIndex).
template <typename BuildT>
class MeshToSDF
{
    using PointT         = nanovdb::Vec3f;
    using TriangleIndexT = nanovdb::Vec3i;
    using GridT          = NanoGrid<BuildT>;
    using TreeT          = NanoTree<BuildT>;
    using RootT          = NanoRoot<BuildT>;
    using UpperT         = NanoUpper<BuildT>;
    using LowerT         = NanoLower<BuildT>;
    using LeafT          = NanoLeaf<BuildT>;
    using TraitsT        = util::cuda::DeviceGridTraits<BuildT>;
    using SurfaceLabelT  = ConnectedComponentsBase::ComponentLabelT;
    template <typename T>
    using BufferT        = sdf_detail::BufferT<T>;
public:
    using HandleT        = GridHandle<BufferT<std::byte>>;

    /// @brief Construct from device-resident mesh data. Processing starts in getHandle().
    /// @param devicePoints   device vertex list in world space
    /// @param pointCount     vertex count
    /// @param deviceTriangles device triangle vertex-index list
    /// @param triangleCount  triangle count
    /// @param map            world-to-index transform
    /// @param stream         CUDA stream
    MeshToSDF(const PointT* devicePoints, uint32_t pointCount,
              const TriangleIndexT* deviceTriangles, uint32_t triangleCount,
              const nanovdb::Map& map = nanovdb::Map(), cudaStream_t stream = 0)
        : mDevicePoints(devicePoints), mPointCount(pointCount)
        , mDeviceTriangles(deviceTriangles), mTriangleCount(triangleCount)
        , mMap(map), mStream(stream), mTimer(stream)
        , mGridHandle(sdf_detail::allocate<std::byte>(0, stream))
        , mUDF(sdf_detail::allocate<std::byte>(0, stream))
        , mTriangleIndex(sdf_detail::allocate<std::byte>(0, stream))
        , mSurfaceLabels(sdf_detail::allocate<SurfaceLabelT>(0, stream))
        , mSurfaceRepresentatives(sdf_detail::allocate<unsigned long long>(0, stream))
        , mNestingDepth(sdf_detail::allocate<uint32_t>(0, stream))
        , mSign(sdf_detail::allocate<int8_t>(0, stream))
        , mLeafInvertMask(sdf_detail::allocate<nanovdb::Mask<3>>(0, stream))
        , mLowerInvertMask(sdf_detail::allocate<nanovdb::Mask<4>>(0, stream))
        , mUpperInvertMask(sdf_detail::allocate<nanovdb::Mask<5>>(0, stream))
        , mRootInterior(sdf_detail::allocate<uint8_t>(0, stream)) {}

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
    HandleT getHandle();

    const GridT* deviceGrid() const { return mGridHandle.template deviceGrid<BuildT>(); }
    const HandleT& gridHandle() const { return mGridHandle; }
    const float* deviceUDF() const { return reinterpret_cast<const float*>(mUDF.data()); }
    const int8_t* deviceSign() const { return mSign.data(); }
    const nanovdb::Map& map() const { return mMap; }
    float narrowBandWidth() const { return mBandWidth; }

    const nanovdb::Mask<3>* deviceLeafInvertMask() const {return mLeafInvertMask.data();}
    const nanovdb::Mask<4>* deviceLowerInvertMask() const {return mLowerInvertMask.data();}
    const nanovdb::Mask<5>* deviceUpperInvertMask() const {return mUpperInvertMask.data();}
    const uint8_t* deviceRootInterior() const {return mRootInterior.data();}
    nanovdb::Coord rootTileMin() const { return mRootTileMin; }
    nanovdb::Coord rootTileDims() const { return mRootDims; }

    /// @brief Phase times for rasterization, partitioning, surface processing, sign resolution,
    ///        and output finalization.
    const float* phaseMs() const { return mPhaseMs; }

private:
    void rasterize();
    void partitionSurfaces();
    void initializeSurfaceComposition();
    void processSurface(uint32_t surfaceID);
    void resolveSigns();
    void fillInvertMasks();
    HandleT finalizeOutput();
    void finalizeDistances();
    HandleT bakeBlindData();
    void releaseIntermediates();

    const PointT*         mDevicePoints{nullptr};
    uint32_t              mPointCount{0};
    const TriangleIndexT* mDeviceTriangles{nullptr};
    uint32_t              mTriangleCount{0};
    nanovdb::Map          mMap{};
    cudaStream_t          mStream{0};
    util::cuda::Timer     mTimer;
    int                   mVerbose{0};
    float                 mBandWidth{3.f};
    BarrierSigning        mBarrierSigning{BarrierSigning::Interior};
    float                 mIsoValue{0.f};  // world units, >= 0; see setIsoValue()
    int                   mBallStencilRadius{1};  // see setBallStencilRadius()
    float                 mPhaseMs[5]{};   // per-phase time from the last getHandle(), see phaseMs()
    // Per-voxel sidecars include slot 0 for the background.
    HandleT                     mGridHandle;
    BufferT<std::byte>          mUDF;           // float per voxel slot, as returned by MeshToGrid
    BufferT<std::byte>          mTriangleIndex; // uint32_t per voxel slot, as returned by MeshToGrid
    BufferT<SurfaceLabelT>      mSurfaceLabels;
    SurfaceLabelT               mSurfaceCount{0};
    BufferT<unsigned long long> mSurfaceRepresentatives;
    BufferT<uint32_t>           mNestingDepth;

    NestingRule                 mNestingRule{NestingRule::EvenOdd};
    BufferT<int8_t>             mSign;
    BufferT<nanovdb::Mask<3>>   mLeafInvertMask;
    BufferT<nanovdb::Mask<4>>   mLowerInvertMask;
    BufferT<nanovdb::Mask<5>>   mUpperInvertMask;
    BufferT<uint8_t>            mRootInterior;
    Coord       mRootTileMin{0};
    Coord       mRootDims{0};

}; // tools::cuda::MeshToSDF<BuildT>

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace sdf_detail {

template <typename BuildT>
uint32_t leafCount(const NanoGrid<BuildT>* grid) {return util::cuda::DeviceGridTraits<BuildT>::getTreeData(grid).mNodeCount[0];}

template <typename BuildT>
uint64_t activeVoxelCount(const NanoGrid<BuildT>* grid) {return util::cuda::DeviceGridTraits<BuildT>::getActiveVoxelCount(grid);}

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

static constexpr int LEAF_SIZE = 512;  // 8^3 voxels per leaf

/// @brief Combine magnitude and sign into a contiguous SDF channel.
struct SignedDistanceFunctor
{
    __device__ void operator()(const uint64_t slot, const float* dUDF, const int8_t* dSign,
                               float* dOut) const
    {
        dOut[slot] = float(dSign[slot]) * dUDF[slot];
    }
};

/// @brief Convert the mesh UDF to distance from the requested isosurface.
/// @note Interior values away from a sign change are floored at half a voxel diagonal.
template <typename BuildT>
struct IsoMagnitudeFunctor
{
    static constexpr int MaxThreadsPerBlock         = LEAF_SIZE;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(const NanoGrid<BuildT>* dGrid, float* dUDF, const int8_t* dSign,
                               float isoValue, float interiorFloor)
    {
        const int leafID = blockIdx.x, n = threadIdx.x;
        const auto& leaf = dGrid->tree().template getFirstNode<0>()[leafID];
        if (!leaf.isActive(uint32_t(n))) return;

        const uint64_t v = leaf.getValue(uint32_t(n));
        const float    m = fabsf(dUDF[v] - isoValue);
        dUDF[v] = m;
        if (dSign[v] >= int8_t(0) || m >= interiorFloor) return;   // nothing to floor

        // Preserve voxels that can carry a marching-cubes crossing.
        const nanovdb::Coord local = nanovdb::NanoLeaf<BuildT>::OffsetToLocalCoord(uint32_t(n));
        const int lx = local[0], ly = local[1], lz = local[2];
        const int off[6][3] = {{-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1}};

        bool touchesExterior = false, needsAccessor = false;
        for (int k = 0; k < 6 && !touchesExterior; ++k) {
            const int nx = lx + off[k][0], ny = ly + off[k][1], nz = lz + off[k][2];
            if (nx < 0 || nx > 7 || ny < 0 || ny > 7 || nz < 0 || nz > 7) { needsAccessor = true; continue; }
            const uint32_t nOff = (uint32_t(nx) << 6) | (uint32_t(ny) << 3) | uint32_t(nz);
            if (leaf.isActive(nOff) && dSign[leaf.getValue(nOff)] > int8_t(0)) touchesExterior = true;
        }
        if (!touchesExterior && needsAccessor) {
            const nanovdb::Coord origin = leaf.origin();
            auto acc = dGrid->getAccessor();
            for (int k = 0; k < 6 && !touchesExterior; ++k) {
                const int nx = lx + off[k][0], ny = ly + off[k][1], nz = lz + off[k][2];
                if (nx >= 0 && nx <= 7 && ny >= 0 && ny <= 7 && nz >= 0 && nz <= 7) continue;
                const nanovdb::Coord nijk(origin[0] + nx, origin[1] + ny, origin[2] + nz);
                if (acc.isActive(nijk) && dSign[acc.getValue(nijk)] > int8_t(0)) touchesExterior = true;
            }
        }
        if (!touchesExterior) dUDF[v] = interiorFloor;
    }
};

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
        const nanovdb::NanoGrid<BuildT>* dGrid,
        const float*                     dUDF,           // UDF sidecar, WORLD units
        float                            isoValue,        // signed surface = { udf == isoValue }, world
        float                            barrierSqWorld,  // (√3/2 · voxelSize)^2, world^2 units
        nanovdb::Mask<3>*                dDstLeafMasks)
    {
        const int leafID   = blockIdx.x;
        const int threadID = threadIdx.x;

        const auto& leaf       = dGrid->tree().template getFirstNode<0>()[leafID];
        auto&       resultMask = dDstLeafMasks[leafID];

        // Clear the leaf's mask words in parallel, then fill the retain bits.
        if (threadID < nanovdb::Mask<3>::WORD_COUNT)
            resultMask.words()[threadID] = 0UL;
        __syncthreads();

        if (auto n = leaf.data()->getValue(threadID)) {  // n != 0 => active voxel
            const float d = dUDF[n] - isoValue;
            if (d * d >= barrierSqWorld)                 // retain non-barrier voxels
                resultMask.setOnAtomic(threadID);
        }
    }
};

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Non-barrier signing.

/// @brief Find the component containing the minimum-x voxel, which is exterior.
template <typename BuildT>
struct FindExteriorRepFunctor
{
    static constexpr int MaxThreadsPerBlock         = LEAF_SIZE;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(const NanoGrid<BuildT>* dGrid, const uint32_t* dVoxelLabel,
                               unsigned long long* dMinKey)
    {
        const int leafID = blockIdx.x, n = threadIdx.x;
        const auto& leaf = dGrid->tree().template getFirstNode<0>()[leafID];
        if (!leaf.isActive(uint32_t(n))) return;
        const uint64_t v = leaf.getValue(uint32_t(n));
        const nanovdb::Coord ijk = leaf.origin() + nanovdb::NanoLeaf<BuildT>::OffsetToLocalCoord(uint32_t(n));
        const uint32_t rep = dVoxelLabel[v];                              // component representative slot
        const uint32_t ux  = uint32_t(int64_t(ijk[0]) + (int64_t(1) << 31));  // x, shifted to unsigned-comparable
        atomicMin(dMinKey, (static_cast<unsigned long long>(ux) << 32) | rep);
    }
};

/// @brief Sign voxels by their connected-component representative.
template <typename BuildT>
struct SignNonBarrierFunctor
{
    static constexpr int MaxThreadsPerBlock         = LEAF_SIZE;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(const NanoGrid<BuildT>* dGrid, const uint32_t* dVoxelLabel,
                               uint32_t exteriorRep, int8_t* dSign)
    {
        const int leafID = blockIdx.x, n = threadIdx.x;
        const auto& leaf = dGrid->tree().template getFirstNode<0>()[leafID];
        if (!leaf.isActive(uint32_t(n))) return;
        const uint64_t v = leaf.getValue(uint32_t(n));
        dSign[v] = (dVoxelLabel[v] == exteriorRep) ? int8_t(1) : int8_t(-1);
    }
};

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Barrier signing.

static constexpr uint32_t INVALID_TRIANGLE = 0xFFFFFFFFu;

/// @brief Test whether an exterior neighbour's nearest triangle also places @a q outside.
/// @note Double precision prevents large-coordinate cancellation from changing the sign test.
__hostdev__ inline bool
barrierExteriorProof(uint64_t nv, const nanovdb::Coord& nijk, const nanovdb::Vec3d& q_xyz,
                     const int8_t* dSign, const uint32_t* dTriangleIndex,
                     const nanovdb::Vec3f* dPoints, const nanovdb::Vec3i* dTriangles,
                     const nanovdb::Map& map, double isoValueIndex)
{
    if (dSign[nv] != int8_t(1)) return false;
    const uint32_t tid = dTriangleIndex[nv];
    if (tid == INVALID_TRIANGLE) return false;

    const nanovdb::Vec3i& T  = dTriangles[tid];
    const nanovdb::Vec3f& p0 = dPoints[T[0]];
    const nanovdb::Vec3f& p1 = dPoints[T[1]];
    const nanovdb::Vec3f& p2 = dPoints[T[2]];
    const nanovdb::Vec3d  v0 = map.applyInverseMap(nanovdb::Vec3d(p0[0], p0[1], p0[2]));  // world -> index
    const nanovdb::Vec3d  v1 = map.applyInverseMap(nanovdb::Vec3d(p1[0], p1[1], p1[2]));
    const nanovdb::Vec3d  v2 = map.applyInverseMap(nanovdb::Vec3d(p2[0], p2[1], p2[2]));
    const nanovdb::Vec3d  n_xyz(static_cast<double>(nijk[0]), static_cast<double>(nijk[1]), static_cast<double>(nijk[2]));

    double t0, t1;
    const nanovdb::Vec3d cp = nanovdb::math::closestPointOnTriangleToPoint(v0, v1, v2, n_xyz, t0, t1);
    nanovdb::Vec3d dn = n_xyz - cp; dn.normalize();   // surface -> neighbor (its confident side)

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
        const NanoGrid<BuildT>* dGrid,
        const int8_t*           dLabelIn,   // +1 ext / -1 int / 0 undecided
        int8_t*                 dLabelOut,
        const float*            dUDF,       // unsigned distance sidecar, WORLD units
        float                   voxelSize,
        uint32_t*               dChanged,       // incremented once per newly decided voxel
        int                     radius,          // stencil half-width in voxels; 1 = the 26 neighbours
        float                   isoValue)        // surface signed = { udf == isoValue }, world units
    {
        const int leafID = blockIdx.x, n = threadIdx.x;
        const auto& leaf = dGrid->tree().template getFirstNode<0>()[leafID];
        if (!leaf.isActive(uint32_t(n))) return;

        const uint64_t qv = leaf.getValue(uint32_t(n));
        const int8_t   ql = dLabelIn[qv];
        if (ql != int8_t(0)) { dLabelOut[qv] = ql; return; }   // already certain: carry through

        const nanovdb::Coord local  = nanovdb::NanoLeaf<BuildT>::OffsetToLocalCoord(uint32_t(n));
        const int            lx = local[0], ly = local[1], lz = local[2];
        const nanovdb::Coord origin = leaf.origin();
        // Radii are distances to the surface being signed, not to the mesh. |udf - isoValue| is a
        // 1-Lipschitz under-estimate of that near the medial axis, which is the safe direction: it
        // shrinks the balls, so it can only certify fewer voxels, never wrongly.
        const float          dq  = fabsf(dUDF[qv] - isoValue);
        const float          eps = 1e-5f * voxelSize;

        bool ext = false, inr = false;

        auto consider = [&] __device__ (uint64_t nv, int dx, int dy, int dz) {
            const int8_t ln = dLabelIn[nv];
            if (ln == int8_t(0)) return;                                   // neighbour not certain yet
            const float len = sqrtf(float(dx*dx + dy*dy + dz*dz)) * voxelSize;
            if (fabsf(dUDF[nv] - isoValue) + dq <= len + eps) return;      // balls do not overlap
            if (ln > 0) ext = true; else inr = true;
        };

        // Avoid accessor lookups for neighbours in the same leaf.
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

        if (lx < radius || lx > 7 - radius || ly < radius || ly > 7 - radius ||
            lz < radius || lz > 7 - radius) {
            auto acc = dGrid->getAccessor();
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
            dLabelOut[qv] = int8_t(0);
            return;
        }
        if (!ext && !inr) { dLabelOut[qv] = int8_t(0); return; }
        dLabelOut[qv] = ext ? int8_t(1) : int8_t(-1);
        atomicAdd(dChanged, 1u);
    }
};

/// @brief Complete the ball-certified sign field, defaulting unresolved voxels to interior.
struct BallFinalizeFunctor
{
    __device__ void operator()(size_t v, const int8_t* dBall, int8_t* dSignOut) const
    {
        if (v == 0) { dSignOut[0] = int8_t(1); return; }   // slot 0 = background = exterior
        const int8_t l = dBall[v];
        if (l != int8_t(0)) { dSignOut[v] = l; return; }
        dSignOut[v] = int8_t(-1);
    }
};

/// @brief Preserve non-barrier signs and classify every barrier voxel as interior.
struct BarrierToInteriorFunctor
{
    __device__ void operator()(const uint64_t slot, const int8_t* dSignIn, int8_t* dSignOut) const
    {
        const int8_t s = dSignIn[slot];
        dSignOut[slot] = (s != int8_t(0)) ? s : int8_t(-1);
    }
};

/// @brief Sign barrier voxels using exterior neighbours and their nearest triangles.
/// @note Input anchors are immutable; unresolved voxels default to interior.
template <typename BuildT>
struct SignBarrierFunctor
{
    static constexpr int MaxThreadsPerBlock         = LEAF_SIZE;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(
        const NanoGrid<BuildT>* dGrid,
        const int8_t*           dSignIn,     // immutable anchors: +1 ext / -1 int / 0 barrier
        int8_t*                 dSignOut,    // result: every active voxel ±1 (slot 0 set by host)
        const uint32_t*         dTriangleIndex,      // nearest-triangle index sidecar (original grid)
        const nanovdb::Vec3f*   dPoints,     // mesh vertices, WORLD space
        const nanovdb::Vec3i*   dTriangles,  // triangle vertex indices
        nanovdb::Map            map,          // world<->index transform (by value)
        double                  isoValueIndex)// surface signed = { udf == isoValue }, INDEX units
    {
        const int leafID = blockIdx.x, n = threadIdx.x;
        const auto& leaf = dGrid->tree().template getFirstNode<0>()[leafID];
        if (!leaf.isActive(uint32_t(n))) return;

        const uint64_t qv = leaf.getValue(uint32_t(n));
        const int8_t   qs = dSignIn[qv];
        if (qs != int8_t(0)) { dSignOut[qv] = qs; return; }  // non-barrier: carry sign through

        const nanovdb::Coord local  = nanovdb::NanoLeaf<BuildT>::OffsetToLocalCoord(uint32_t(n));
        const int            lx = local[0], ly = local[1], lz = local[2];
        const nanovdb::Coord origin = leaf.origin();
        const nanovdb::Vec3d q_xyz(double(origin[0] + lx), double(origin[1] + ly), double(origin[2] + lz));

        bool exterior = false;

        // Avoid accessor lookups for neighbours in the same leaf.
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
                    if (barrierExteriorProof(leaf.getValue(nOff), nijk, q_xyz, dSignIn, dTriangleIndex,
                                             dPoints, dTriangles, map, isoValueIndex)) {
                        exterior = true; break;
                    }
                }
            }
        }

        if (!exterior && (lx == 0 || lx == 7 || ly == 0 || ly == 7 || lz == 0 || lz == 7)) {
            auto acc = dGrid->getAccessor();
            for (int dx = -1; dx <= 1 && !exterior; ++dx)
                for (int dy = -1; dy <= 1 && !exterior; ++dy)
                    for (int dz = -1; dz <= 1; ++dz) {
                        if (dx == 0 && dy == 0 && dz == 0) continue;
                        const int nx = lx + dx, ny = ly + dy, nz = lz + dz;
                        if (nx >= 0 && nx <= 7 && ny >= 0 && ny <= 7 && nz >= 0 && nz <= 7)
                            continue;  // in-leaf neighbor already handled by pass 1
                        const nanovdb::Coord nijk(origin[0] + nx, origin[1] + ny, origin[2] + nz);
                        if (!acc.isActive(nijk)) continue;
                        if (barrierExteriorProof(acc.getValue(nijk), nijk, q_xyz, dSignIn, dTriangleIndex,
                                                 dPoints, dTriangles, map, isoValueIndex)) {
                            exterior = true; break;
                        }
                    }
        }

        dSignOut[qv] = exterior ? int8_t(1) : int8_t(-1);
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

    __device__ void operator()(const NanoGrid<BuildT>* dGrid,
                               const int8_t*           dSign,        // completed sign sidecar
                               nanovdb::Mask<3>*       dInvertMasks) // one Mask<3> per leaf, output
    {
        const int leafID = blockIdx.x, n = threadIdx.x;
        const auto& leaf = dGrid->tree().template getFirstNode<0>()[leafID];

        __shared__ uint8_t sAct[LEAF_SIZE];  // active voxel (wall)
        __shared__ uint8_t sInj[LEAF_SIZE];  // interior active voxel (flood source)
        __shared__ uint8_t sInv[LEAF_SIZE];  // result: inactive voxel marked interior
        __shared__ int     sChanged;

        const bool act = leaf.isActive(uint32_t(n));
        sAct[n] = act ? 1 : 0;
        sInj[n] = (act && dSign[leaf.getValue(uint32_t(n))] == int8_t(-1)) ? 1 : 0;
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
            dInvertMasks[leafID].words()[n] = w;
        }
    }
};

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Childless lower (8^3) and upper (128^3) tiles, filled bottom-up.

/// @brief Find the finest childless tile containing the leaf region at @a c.
/// @return 0 for a root value tile or leaf, 1 for lower, 2 for upper, and 3 for an absent root region.
/// On 1 or 2, @a nodeIdx and @a slot identify the childless tile.
template<typename BuildT>
__hostdev__ inline int
probeChildlessSlot(const NanoGrid<BuildT> &grid, const nanovdb::Coord &c,
                   uint64_t &nodeIdx, uint32_t &slot) {
    using UpperT = NanoUpper<BuildT>;
    using LowerT = NanoLower<BuildT>;
    const auto &tree = grid.tree();
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

    __device__ void operator()(const NanoGrid<BuildT>*  dGrid,
                               const int8_t*            dSign,        // completed signs
                               const nanovdb::Mask<3>*  dLeafInvert,  // leaf invert masks
                               nanovdb::Mask<4>* dLowerSawInt, nanovdb::Mask<4>* dLowerSawExt,
                               nanovdb::Mask<5>* dUpperSawInt, nanovdb::Mask<5>* dUpperSawExt)
    {
        const int leafID = blockIdx.x, t = threadIdx.x;
        const auto& leaf = dGrid->tree().template getFirstNode<0>()[leafID];

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
        const bool interior = act ? (dSign[leaf.getValue(uint32_t(n))] == int8_t(-1))
                                  : dLeafInvert[leafID].isOn(uint32_t(n));
        if (interior) sInt[face] = 1; else sExt[face] = 1;  // benign race: all writers store 1
        __syncthreads();

        if (t < 6) {  // one probe per face
            const int off[6][3] = {{-8,0,0},{8,0,0},{0,-8,0},{0,8,0},{0,0,-8},{0,0,8}};
            const nanovdb::Coord c = leaf.origin().offsetBy(off[t][0], off[t][1], off[t][2]);
            uint64_t nodeIdx; uint32_t slot;
            const int level = probeChildlessSlot(*dGrid, c, nodeIdx, slot);
            if (level == 1) {
                if (sInt[t]) dLowerSawInt[nodeIdx].setOnAtomic(slot);
                if (sExt[t]) dLowerSawExt[nodeIdx].setOnAtomic(slot);
            } else if (level == 2) {
                if (sInt[t]) dUpperSawInt[nodeIdx].setOnAtomic(slot);
                if (sExt[t]) dUpperSawExt[nodeIdx].setOnAtomic(slot);
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

    __device__ void operator()(const NanoGrid<BuildT>*  dGrid,
                               const nanovdb::Mask<4>*  dLowerInvert,  // flooded lower invert masks
                               nanovdb::Mask<5>* dUpperSawInt, nanovdb::Mask<5>* dUpperSawExt)
    {
        const int nodeID = blockIdx.x, t = threadIdx.x;
        const auto& node = dGrid->tree().template getFirstNode<1>()[nodeID];

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
            if (dLowerInvert[nodeID].isOn(uint32_t(n))) sInt[face] = 1; else sExt[face] = 1;
        }
        __syncthreads();

        if (t < 6) {  // whole 128×128 face abuts exactly one 128^3 region across
            const int off[6][3] = {{-128,0,0},{128,0,0},{0,-128,0},{0,128,0},{0,0,-128},{0,0,128}};
            const nanovdb::Coord c = node.origin().offsetBy(off[t][0], off[t][1], off[t][2]);
            uint64_t nodeIdx; uint32_t slot;
            if (probeChildlessSlot(*dGrid, c, nodeIdx, slot) == 2) {
                if (sInt[t]) dUpperSawInt[nodeIdx].setOnAtomic(slot);
                if (sExt[t]) dUpperSawExt[nodeIdx].setOnAtomic(slot);
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

    __device__ void operator()(const NanoGrid<BuildT>* dGrid,
                               const MaskT* dSawInt, const MaskT* dSawExt, MaskT* dInvert)
    {
        const int nodeID = blockIdx.x, t = threadIdx.x;
        const auto& node = dGrid->tree().template getFirstNode<LEVEL>()[nodeID];

        __shared__ uint64_t sWall[WORDS];  // refined slots (childMask)
        __shared__ uint64_t sInv [WORDS];  // result bits (seeded, then flooded)
        __shared__ int      sChanged;

        for (int w = t; w < WORDS; w += blockDim.x) {
            const uint64_t wall = node.childMask().words()[w];
            sWall[w] = wall;
            sInv[w]  = dSawInt[nodeID].words()[w] & ~dSawExt[nodeID].words()[w] & ~wall;
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
            dInvert[nodeID].words()[w] = sInv[w];
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

    __device__ void operator()(const NanoGrid<BuildT>*  dGrid,
                               const int8_t*            dSign,
                               const nanovdb::Mask<3>*  dLeafInvert,
                               const nanovdb::Mask<4>*  dLowerInvert,
                               const nanovdb::Mask<5>*  dUpperInvert,
                               uint8_t* dSawInt, uint8_t* dSawExt,
                               nanovdb::Coord tileMin, nanovdb::Coord dims)
    {
        const int nodeID = blockIdx.x, t = threadIdx.x;
        const auto& node = dGrid->tree().template getFirstNode<LEVEL>()[nodeID];

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
            const bool interior = act ? (dSign[node.getValue(uint32_t(n))] == int8_t(-1))
                                      : dLeafInvert[nodeID].isOn(uint32_t(n));
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
                if (dLowerInvert[nodeID].isOn(uint32_t(n))) sInt[face] = 1; else sExt[face] = 1;
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
                if (dUpperInvert[nodeID].isOn(uint32_t(n))) sInt[face] = 1; else sExt[face] = 1;
            }
        }
        __syncthreads();

        if (t < 6) {  // one probe per face: the whole face abuts exactly one root cell
            const int off[6][3] = {{-NODE_DIM,0,0},{NODE_DIM,0,0},{0,-NODE_DIM,0},{0,NODE_DIM,0},{0,0,-NODE_DIM},{0,0,NODE_DIM}};
            const nanovdb::Coord c = node.origin().offsetBy(off[t][0], off[t][1], off[t][2]);
            uint64_t nodeIdx; uint32_t slot;
            if (probeChildlessSlot(*dGrid, c, nodeIdx, slot) == 3) {  // absent root region
                const int i = (c[0] >> 12) - tileMin[0], j = (c[1] >> 12) - tileMin[1], k = (c[2] >> 12) - tileMin[2];
                if (i >= 0 && i < dims[0] && j >= 0 && j < dims[1] && k >= 0 && k < dims[2]) {
                    const int idx = (i * dims[1] + j) * dims[2] + k;
                    if (sInt[t]) dSawInt[idx] = 1;  // benign race: all writers store 1
                    if (sExt[t]) dSawExt[idx] = 1;
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

    __device__ void operator()(const uint8_t* dWall, const uint8_t* dSawInt, const uint8_t* dSawExt,
                               uint8_t* dOn, nanovdb::Coord dims)
    {
        const int t = threadIdx.x;
        const int P = dims[0], Q = dims[1], R = dims[2], total = P * Q * R;
        __shared__ int sChanged;

        for (int c = t; c < total; c += blockDim.x)
            dOn[c] = (!dWall[c] && dSawInt[c] && !dSawExt[c]) ? 1 : 0;
        __syncthreads();

        for (int it = 0; it < total + 2; ++it) {  // cap: any path length < total cells
            if (t == 0) sChanged = 0;
            __syncthreads();
            bool any = false;
            for (int c = t; c < total; c += blockDim.x) {
                if (dWall[c] || dOn[c]) continue;
                const int k = c % R, j = (c / R) % Q, i = c / (Q * R);
                const bool on =
                    (i > 0     && dOn[c - Q * R]) || (i < P - 1 && dOn[c + Q * R]) ||
                    (j > 0     && dOn[c - R])     || (j < Q - 1 && dOn[c + R])     ||
                    (k > 0     && dOn[c - 1])     || (k < R - 1 && dOn[c + 1]);
                if (on) { dOn[c] = 1; any = true; }  // monotone: racy same-sweep reads only accelerate
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
signAt(const NanoGrid<BuildT>& grid, const nanovdb::Coord& ijk,
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
// Surface partition and composition.

template <typename BuildT>
struct SurfaceMaskFunctor
{
    static constexpr int MaxThreadsPerBlock         = 512;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(const NanoGrid<BuildT>* dGrid, const uint32_t* dSurfaceLabel,
                               uint32_t target, Mask<3>* dMasks)
    {
        const int leafID = blockIdx.x, n = threadIdx.x;
        const auto& leaf = dGrid->tree().template getFirstNode<0>()[leafID];
        auto&       mask = dMasks[leafID];
        if (n < int(Mask<3>::WORD_COUNT)) mask.words()[n] = 0UL;
        __syncthreads();
        if (auto v = leaf.data()->getValue(uint32_t(n)))          // v != 0 => active voxel
            if (dSurfaceLabel[v] == target) mask.setOnAtomic(uint32_t(n));
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
template <typename BuildT>
struct SurfaceRepFunctor
{
    static constexpr int MaxThreadsPerBlock         = 512;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(const NanoGrid<BuildT>* dGrid, const uint32_t* dSurfaceLabel,
                               uint32_t surfaceCount, unsigned long long* dRepKey)
    {
        const int leafID = blockIdx.x, n = threadIdx.x;
        const auto& leaf = dGrid->tree().template getFirstNode<0>()[leafID];
        if (!leaf.isActive(uint32_t(n))) return;
        const uint32_t s = dSurfaceLabel[leaf.getValue(uint32_t(n))];
        if (s >= surfaceCount) return;
        const Coord ijk = leaf.origin() + NanoLeaf<BuildT>::OffsetToLocalCoord(uint32_t(n));
        atomicMin(&dRepKey[s], packCoord(ijk));
    }
};

// Accumulate how many other surfaces enclose each representative voxel.
template <typename BuildT>
struct AccumulateNestingFunctor
{
    __device__ void operator()(size_t j, const NanoGrid<BuildT>* dGrid,
                               const unsigned long long* dRepresentatives, const int8_t* dSign,
                               const Mask<3>* dLeaf, const nanovdb::Mask<4>* dLower,
                               const nanovdb::Mask<5>* dUpper, const uint8_t* dRoot,
                               Coord rootMin, Coord rootDims, uint32_t surfaceID,
                               uint32_t* dDepth) const
    {
        if (j == surfaceID) return;
        const unsigned long long k = dRepresentatives[j];
        const Coord ijk(int((k >> 42) & 0x1FFFFF) - (1 << 20),
                                 int((k >> 21) & 0x1FFFFF) - (1 << 20),
                                 int( k        & 0x1FFFFF) - (1 << 20));
        if (signAt<BuildT>(*dGrid, ijk, dSign, dLeaf, dLower, dUpper,
                                 dRoot, rootMin, rootDims) < 0)
            ++dDepth[j];
    }
};

/// @brief Compose per-surface signs under the selected nesting rule.
/// @note The enclosing count is the surface depth plus its local inside state.
struct ResolveNestingFunctor
{
    __device__ void operator()(size_t v, const uint32_t* dSurfaceLabel, const uint32_t* dDepth,
                               uint32_t surfaceCount, bool evenOdd, int8_t* dSign) const
    {
        if (v == 0) return;                                  // slot 0 is the background
        const uint32_t s = dSurfaceLabel[v];
        if (s >= surfaceCount) return;
        const uint32_t enclosing = dDepth[s] + (dSign[v] < int8_t(0) ? 1u : 0u);
        const bool interior = evenOdd ? ((enclosing & 1u) != 0u) : (enclosing != 0u);
        dSign[v] = interior ? int8_t(-1) : int8_t(1);
    }
};

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
GridHandle<BufferT<std::byte>>
pruneBarrier(const NanoGrid<BuildT>* dGrid, const float* dUDF, float voxelSize,
             float isoValue, cudaStream_t stream)
{
    using FunctorT = UDFBarrierPruneMaskFunctor<BuildT>;

    const float    barrierSqWorld = 0.75f * voxelSize * voxelSize;
    const uint32_t leaves         = leafCount(dGrid);

    auto  retainMask  = allocate<nanovdb::Mask<3>>(leaves, stream);
    auto* dRetainMask = retainMask.data();

    util::cuda::operatorKernel<FunctorT><<<leaves, FunctorT::MaxThreadsPerBlock, 0, stream>>>(
        dGrid, dUDF, isoValue, barrierSqWorld, dRetainMask);
    cudaCheckError();

    PruneGrid<BuildT> pruner(dGrid, dRetainMask, stream);
    return pruner.getHandle(allocate<std::byte>(0, stream));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
BufferT<int8_t> signNonBarrierComponents(const NanoGrid<BuildT>* dGrid, const uint32_t* dVoxelLabel,
                                         cudaStream_t stream)
{
    const uint32_t leaves = leafCount(dGrid);
    if (leaves == 0) return allocate<int8_t>(0, stream);

    const uint64_t activeCount = activeVoxelCount(dGrid);

    auto minKeyBuffer = allocate<unsigned long long>(1, stream);
    auto* dMinKey = minKeyBuffer.data();
    cudaCheck(cudaMemsetAsync(dMinKey, 0xFF, sizeof(unsigned long long), stream));
    using FindExteriorFunctorT = FindExteriorRepFunctor<BuildT>;
    util::cuda::operatorKernel<FindExteriorFunctorT><<<leaves, FindExteriorFunctorT::MaxThreadsPerBlock, 0, stream>>>(
        dGrid, dVoxelLabel, dMinKey);
    cudaCheckError();

    unsigned long long minKey = 0;
    cudaCheck(cudaMemcpyAsync(&minKey, dMinKey, sizeof(minKey), cudaMemcpyDeviceToHost, stream));
    cudaCheck(cudaStreamSynchronize(stream));
    const uint32_t exteriorRep = uint32_t(minKey & 0xFFFFFFFFull);

    auto sign = allocate<int8_t>(activeCount + 1, stream);
    cudaCheck(cudaMemsetAsync(sign.data(), 1, sign.size_bytes(), stream));
    using SignFunctorT = SignNonBarrierFunctor<BuildT>;
    util::cuda::operatorKernel<SignFunctorT><<<leaves, SignFunctorT::MaxThreadsPerBlock, 0, stream>>>(
        dGrid, dVoxelLabel, exteriorRep, sign.data());
    cudaCheckError();
    return sign;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
BufferT<int8_t> injectNonBarrierSigns(const NanoGrid<BuildT>* dGrid, const NanoGrid<BuildT>* dDerivedGrid,
                                      const BufferT<int8_t>& derivedSign, cudaStream_t stream)
{
    const uint64_t activeCount      = activeVoxelCount(dGrid);
    const uint32_t derivedLeafCount = leafCount(dDerivedGrid);

    auto partialSign = allocate<int8_t>(activeCount + 1, stream);
    cudaCheck(cudaMemsetAsync(partialSign.data(), 0, partialSign.size_bytes(), stream));
    cudaCheck(cudaMemsetAsync(partialSign.data(), 1, sizeof(int8_t), stream));

    if (derivedLeafCount == 0) return partialSign;

    using FunctorT = util::cuda::InjectGridDataFunctor<BuildT, int8_t>;
    util::cuda::operatorKernel<FunctorT><<<derivedLeafCount, FunctorT::MaxThreadsPerBlock, 0, stream>>>(
        dDerivedGrid, dGrid, derivedSign.data(), partialSign.data());
    cudaCheckError();
    return partialSign;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
BufferT<int8_t> signBarrierAsInterior(const NanoGrid<BuildT>* dGrid, const BufferT<int8_t>& partialSign,
                                      cudaStream_t stream)
{
    const uint64_t activeCount = activeVoxelCount(dGrid);
    const uint64_t slots       = activeCount + 1;

    auto sign = allocate<int8_t>(slots, stream);
    if (leafCount(dGrid) == 0) {
        cudaCheck(cudaMemsetAsync(sign.data(), 1, sizeof(int8_t), stream));
        return sign;
    }

    util::cuda::lambdaKernel<<<(unsigned int)((slots + 255) / 256), 256, 0, stream>>>(
        slots, BarrierToInteriorFunctor{}, partialSign.data(), sign.data());
    cudaCheckError();
    return sign;
}

template <typename BuildT>
BufferT<int8_t> signBarrierHeuristic(const NanoGrid<BuildT>* dGrid, const BufferT<int8_t>& partialSign,
                                     const uint32_t* dTriangleIndex, const nanovdb::Vec3f* dPoints,
                                     const nanovdb::Vec3i* dTriangles, const nanovdb::Map& map,
                                     float isoValue, float voxelSize, cudaStream_t stream)
{
    const uint64_t activeCount = activeVoxelCount(dGrid);
    const uint32_t leaves      = leafCount(dGrid);

    auto sign = allocate<int8_t>(activeCount + 1, stream);
    cudaCheck(cudaMemsetAsync(sign.data(), 0, sign.size_bytes(), stream));
    cudaCheck(cudaMemsetAsync(sign.data(), 1, sizeof(int8_t), stream));
    if (leaves == 0) return sign;

    using FunctorT = SignBarrierFunctor<BuildT>;
    util::cuda::operatorKernel<FunctorT><<<leaves, FunctorT::MaxThreadsPerBlock, 0, stream>>>(
        dGrid, partialSign.data(), sign.data(),
        dTriangleIndex, dPoints, dTriangles, map,
        (voxelSize > 0.f) ? double(isoValue) / double(voxelSize) : 0.0);
    cudaCheckError();
    return sign;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
BufferT<int8_t> signBarrierWithBalls(const NanoGrid<BuildT>* dGrid, const BufferT<int8_t>& partialSign,
                                     const float* dUDF, float voxelSize, int maxRounds, int radius,
                                     float isoValue, cudaStream_t stream)
{
    const uint64_t activeCount = activeVoxelCount(dGrid);
    const uint32_t leaves      = leafCount(dGrid);
    const std::size_t bytes    = std::size_t(activeCount + 1) * sizeof(int8_t);

    auto sign = allocate<int8_t>(activeCount + 1, stream);
    if (leaves == 0) {
        cudaCheck(cudaMemsetAsync(sign.data(), 1, sizeof(int8_t), stream));
        return sign;
    }

    auto labels  = allocate<int8_t>(activeCount + 1, stream);
    auto scratch = allocate<int8_t>(activeCount + 1, stream);
    cudaCheck(cudaMemcpyAsync(labels.data(), partialSign.data(), bytes,
                              cudaMemcpyDeviceToDevice, stream));
    cudaCheck(cudaMemcpyAsync(scratch.data(), partialSign.data(), bytes,
                              cudaMemcpyDeviceToDevice, stream));

    auto  counter   = allocate<uint32_t>(1, stream);
    auto* dChanged = counter.data();
    int8_t* labelIn  = labels.data();
    int8_t* labelOut = scratch.data();

    using FunctorT = BallCertifyFunctor<BuildT>;
    uint32_t changed = 0;
    for (int r = 0; r < maxRounds; ++r) {
        cudaCheck(cudaMemsetAsync(dChanged, 0, sizeof(uint32_t), stream));
        util::cuda::operatorKernel<FunctorT><<<leaves, FunctorT::MaxThreadsPerBlock, 0, stream>>>(
            dGrid, labelIn, labelOut, dUDF, voxelSize, dChanged, radius, isoValue);
        cudaCheckError();
        cudaCheck(cudaMemcpyAsync(&changed, dChanged, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
        cudaCheck(cudaStreamSynchronize(stream));
        std::swap(labelIn, labelOut);
        if (changed == 0) break;
    }
    util::cuda::lambdaKernel<<<(unsigned int)((activeCount + 256) / 256), 256, 0, stream>>>(
        activeCount + 1, BallFinalizeFunctor{}, labelIn, sign.data());
    cudaCheckError();
    cudaCheck(cudaStreamSynchronize(stream));
    return sign;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
BufferT<nanovdb::Mask<3>> buildLeafInvertMask(const NanoGrid<BuildT>* dGrid, const int8_t* dSign,
                                              cudaStream_t stream)
{
    const uint32_t leaves = leafCount(dGrid);
    if (leaves == 0) return allocate<nanovdb::Mask<3>>(0, stream);

    auto mask = allocate<nanovdb::Mask<3>>(leaves, stream);
    using FunctorT = FillLeafInvertMaskFunctor<BuildT>;
    util::cuda::operatorKernel<FunctorT><<<leaves, FunctorT::MaxThreadsPerBlock, 0, stream>>>(
        dGrid, dSign, mask.data());
    cudaCheckError();
    return mask;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
void buildCoarseInvertMasks(const NanoGrid<BuildT>* dGrid, const int8_t* dSign,
                            const BufferT<nanovdb::Mask<3>>& leafInvertMask,
                            BufferT<nanovdb::Mask<4>>& lowerInvertMask,
                            BufferT<nanovdb::Mask<5>>& upperInvertMask, cudaStream_t stream)
{
    const auto     treeData   = util::cuda::DeviceGridTraits<BuildT>::getTreeData(dGrid);
    const uint32_t leafCount  = treeData.mNodeCount[0];
    const uint32_t lowerCount = treeData.mNodeCount[1];
    const uint32_t upperCount = treeData.mNodeCount[2];

    const std::size_t lowerBytes = std::size_t(lowerCount) * sizeof(nanovdb::Mask<4>);
    const std::size_t upperBytes = std::size_t(upperCount) * sizeof(nanovdb::Mask<5>);
    lowerInvertMask = allocate<nanovdb::Mask<4>>(lowerCount, stream);
    upperInvertMask = allocate<nanovdb::Mask<5>>(upperCount, stream);
    if (lowerCount) cudaCheck(cudaMemsetAsync(lowerInvertMask.data(), 0, lowerBytes, stream));
    if (upperCount) cudaCheck(cudaMemsetAsync(upperInvertMask.data(), 0, upperBytes, stream));
    if (leafCount == 0 || lowerCount == 0) return;  // nothing to seed from

    auto lowSawIntBuffer = allocate<nanovdb::Mask<4>>(lowerCount, stream);
    auto lowSawExtBuffer = allocate<nanovdb::Mask<4>>(lowerCount, stream);
    auto upSawIntBuffer  = allocate<nanovdb::Mask<5>>(upperCount, stream);
    auto upSawExtBuffer  = allocate<nanovdb::Mask<5>>(upperCount, stream);
    auto* dLowSawInt = lowSawIntBuffer.data();
    auto* dLowSawExt = lowSawExtBuffer.data();
    auto* dUpSawInt  = upSawIntBuffer.data();
    auto* dUpSawExt  = upSawExtBuffer.data();
    cudaCheck(cudaMemsetAsync(dLowSawInt, 0, lowerBytes, stream));
    cudaCheck(cudaMemsetAsync(dLowSawExt, 0, lowerBytes, stream));
    cudaCheck(cudaMemsetAsync(dUpSawInt,  0, upperBytes, stream));
    cudaCheck(cudaMemsetAsync(dUpSawExt,  0, upperBytes, stream));

    auto* dLeafInvert  = leafInvertMask.data();
    auto* dLowerInvert = lowerInvertMask.data();
    auto* dUpperInvert = upperInvertMask.data();

    using LeafSeedFunctorT = LeafFaceSeedFunctor<BuildT>;
    util::cuda::operatorKernel<LeafSeedFunctorT><<<leafCount, LeafSeedFunctorT::MaxThreadsPerBlock, 0, stream>>>(
        dGrid, dSign, dLeafInvert,
        dLowSawInt, dLowSawExt, dUpSawInt, dUpSawExt);
    cudaCheckError();

    using LowerFloodFunctorT = CoarseInvertFloodFunctor<BuildT, 1>;
    util::cuda::operatorKernel<LowerFloodFunctorT><<<lowerCount, LowerFloodFunctorT::MaxThreadsPerBlock, 0, stream>>>(
        dGrid, dLowSawInt, dLowSawExt, dLowerInvert);
    cudaCheckError();

    using LowerSeedFunctorT = LowerFaceSeedFunctor<BuildT>;
    util::cuda::operatorKernel<LowerSeedFunctorT><<<lowerCount, LowerSeedFunctorT::MaxThreadsPerBlock, 0, stream>>>(
        dGrid, dLowerInvert, dUpSawInt, dUpSawExt);
    cudaCheckError();

    if (upperCount) {
        using UpperFloodFunctorT = CoarseInvertFloodFunctor<BuildT, 2>;
        util::cuda::operatorKernel<UpperFloodFunctorT><<<upperCount, UpperFloodFunctorT::MaxThreadsPerBlock, 0, stream>>>(
            dGrid, dUpSawInt, dUpSawExt, dUpperInvert);
        cudaCheckError();
    }

    cudaCheck(cudaStreamSynchronize(stream));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
void buildRootInteriorMask(const NanoGrid<BuildT>* dGrid, const int8_t* dSign,
                           const BufferT<nanovdb::Mask<3>>& leafInvertMask,
                           const BufferT<nanovdb::Mask<4>>& lowerInvertMask,
                           const BufferT<nanovdb::Mask<5>>& upperInvertMask, BufferT<uint8_t>& rootInterior,
                           nanovdb::Coord& rootTileMin, nanovdb::Coord& rootDims,
                           cudaStream_t stream)
{
    const auto     treeData   = util::cuda::DeviceGridTraits<BuildT>::getTreeData(dGrid);
    const uint32_t leafCount  = treeData.mNodeCount[0];
    const uint32_t lowerCount = treeData.mNodeCount[1];
    const uint32_t upperCount = treeData.mNodeCount[2];
    if (leafCount == 0) {
        rootInterior = allocate<uint8_t>(0, stream);
        rootDims = nanovdb::Coord(0);
        return;
    }

    const auto bbox = util::cuda::DeviceGridTraits<BuildT>::getIndexBBox(dGrid, treeData);
    rootTileMin = nanovdb::Coord(bbox.min()[0] >> 12, bbox.min()[1] >> 12, bbox.min()[2] >> 12);
    const nanovdb::Coord tileMax(bbox.max()[0] >> 12, bbox.max()[1] >> 12, bbox.max()[2] >> 12);
    rootDims = nanovdb::Coord(tileMax[0] - rootTileMin[0] + 1,
                              tileMax[1] - rootTileMin[1] + 1,
                              tileMax[2] - rootTileMin[2] + 1);
    const std::size_t total = std::size_t(rootDims[0]) * rootDims[1] * rootDims[2];

    rootInterior    = allocate<uint8_t>(total, stream);
    auto wallBuffer     = allocate<uint8_t>(total, stream);
    auto sawIntBuffer   = allocate<uint8_t>(total, stream);
    auto sawExtBuffer   = allocate<uint8_t>(total, stream);
    auto* dWall     = wallBuffer.data();
    auto* dSawInt   = sawIntBuffer.data();
    auto* dSawExt   = sawExtBuffer.data();
    cudaCheck(cudaMemsetAsync(rootInterior.data(), 0, total, stream));
    cudaCheck(cudaMemsetAsync(dWall,   0, total, stream));
    cudaCheck(cudaMemsetAsync(dSawInt, 0, total, stream));
    cudaCheck(cudaMemsetAsync(dSawExt, 0, total, stream));

    const auto* dLeafInvert  = leafInvertMask.data();
    const auto* dLowerInvert = lowerInvertMask.data();
    const auto* dUpperInvert = upperInvertMask.data();
    auto*       dRootInterior = rootInterior.data();

    constexpr unsigned int kWallThreads = 128;
    util::cuda::lambdaKernel<<<unsigned((total + kWallThreads - 1) / kWallThreads), kWallThreads, 0, stream>>>(
        total, RootWallMarkFunctor<BuildT>{ dGrid, dWall, rootTileMin, rootDims });
    cudaCheckError();

    using LeafSeedFunctorT  = RootFaceSeedFunctor<BuildT, 0>;
    using LowerSeedFunctorT = RootFaceSeedFunctor<BuildT, 1>;
    using UpperSeedFunctorT = RootFaceSeedFunctor<BuildT, 2>;
    util::cuda::operatorKernel<LeafSeedFunctorT><<<leafCount, LeafSeedFunctorT::MaxThreadsPerBlock, 0, stream>>>(
        dGrid, dSign, dLeafInvert, dLowerInvert, dUpperInvert,
        dSawInt, dSawExt, rootTileMin, rootDims);
    cudaCheckError();
    if (lowerCount) {
        util::cuda::operatorKernel<LowerSeedFunctorT><<<lowerCount, LowerSeedFunctorT::MaxThreadsPerBlock, 0, stream>>>(
            dGrid, dSign, dLeafInvert, dLowerInvert, dUpperInvert,
            dSawInt, dSawExt, rootTileMin, rootDims);
        cudaCheckError();
    }
    if (upperCount) {
        util::cuda::operatorKernel<UpperSeedFunctorT><<<upperCount, UpperSeedFunctorT::MaxThreadsPerBlock, 0, stream>>>(
            dGrid, dSign, dLeafInvert, dLowerInvert, dUpperInvert,
            dSawInt, dSawExt, rootTileMin, rootDims);
        cudaCheckError();
    }

    using FloodFunctorT = RootInteriorFloodFunctor;
    util::cuda::operatorKernel<FloodFunctorT><<<1, FloodFunctorT::MaxThreadsPerBlock, 0, stream>>>(
        dWall, dSawInt, dSawExt, dRootInterior, rootDims);
    cudaCheckError();

    cudaCheck(cudaStreamSynchronize(stream));
}

struct InvertMasks
{
    BufferT<nanovdb::Mask<3>> leaf;
    BufferT<nanovdb::Mask<4>> lower;
    BufferT<nanovdb::Mask<5>> upper;
    BufferT<uint8_t>          rootInterior;
    nanovdb::Coord            rootTileMin{0};
    nanovdb::Coord            rootDims{0};
};

template <typename BuildT>
InvertMasks buildInvertMasks(const NanoGrid<BuildT>* dGrid, const int8_t* dSign,
                             cudaStream_t stream)
{
    InvertMasks masks{buildLeafInvertMask(dGrid, dSign, stream), allocate<nanovdb::Mask<4>>(0, stream),
                      allocate<nanovdb::Mask<5>>(0, stream), allocate<uint8_t>(0, stream)};
    buildCoarseInvertMasks(dGrid, dSign, masks.leaf, masks.lower, masks.upper, stream);
    buildRootInteriorMask(
        dGrid, dSign, masks.leaf, masks.lower, masks.upper,
        masks.rootInterior, masks.rootTileMin, masks.rootDims, stream);
    return masks;
}

} // namespace sdf_detail

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
typename MeshToSDF<BuildT>::HandleT MeshToSDF<BuildT>::getHandle()
{
    auto stopTimer = [&](int phase) {
        mTimer.record();
        mPhaseMs[phase] = mTimer.milliseconds();
        if (mVerbose == 1) mTimer.print();
    };

    if (mIsoValue < 0.f)
        throw std::runtime_error("MeshToSDF: setIsoValue() must be >= 0");

    if (mVerbose == 1) mTimer.start("\nRasterizing mesh");
    else mTimer.start();
    rasterize();
    stopTimer(0);

    if (mVerbose == 1) mTimer.start("Partitioning surfaces");
    else mTimer.start();
    partitionSurfaces();
    initializeSurfaceComposition();
    stopTimer(1);

    if (mVerbose == 1) mTimer.start("Processing surfaces");
    else mTimer.start();
    for (SurfaceLabelT surfaceID = 0; surfaceID < mSurfaceCount; ++surfaceID)
        processSurface(surfaceID);
    stopTimer(2);

    if (mVerbose == 1) mTimer.start("Resolving signs on the original grid");
    else mTimer.start();
    resolveSigns();
    stopTimer(3);

    if (mVerbose == 1) mTimer.start("Filling invert masks and finalizing output");
    else mTimer.start();
    fillInvertMasks();
    HandleT handle = finalizeOutput();
    stopTimer(4);

    return handle;
} // MeshToSDF<BuildT>::getHandle

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
void MeshToSDF<BuildT>::rasterize()
{
    const float voxelSize = float(mMap.getVoxelSize()[0]);

    // Preserve the requested band width beyond the offset surface.
    const float extra = (voxelSize > 0.f) ? mIsoValue / voxelSize : 0.f;

    MeshToGrid<BuildT> converter(mDevicePoints, mPointCount, mDeviceTriangles, mTriangleCount, mMap, mStream);
    converter.setNarrowBandWidth(mBandWidth + extra);
    std::tie(mGridHandle, mUDF, mTriangleIndex) =
        converter.template getHandleAndUDFAndIndex<BufferT<std::byte>, BufferT<std::byte>>(mGridHandle.buffer(), mUDF);
} // MeshToSDF<BuildT>::rasterize

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
void MeshToSDF<BuildT>::partitionSurfaces()
{
    ConnectedComponents<BuildT> components(deviceGrid(), mStream);
    auto result = components.getVoxelLabelsAndCount();
    mSurfaceLabels = std::move(result.first);
    mSurfaceCount = result.second;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
void MeshToSDF<BuildT>::initializeSurfaceComposition()
{
    if (mSurfaceCount <= 1) return;

    const auto* dGrid = deviceGrid();
    const auto& tree = TraitsT::getTreeData(dGrid);
    const uint64_t activeCount = TraitsT::getActiveVoxelCount(dGrid);

    mSurfaceRepresentatives = sdf_detail::allocate<unsigned long long>(mSurfaceCount, mStream);
    mNestingDepth = sdf_detail::allocate<uint32_t>(mSurfaceCount, mStream);
    mSign = sdf_detail::allocate<int8_t>(activeCount + 1, mStream);
    cudaCheck(cudaMemsetAsync(mSurfaceRepresentatives.data(), 0xFF,
                              mSurfaceRepresentatives.size_bytes(), mStream));
    cudaCheck(cudaMemsetAsync(mNestingDepth.data(), 0, mNestingDepth.size_bytes(), mStream));
    cudaCheck(cudaMemsetAsync(mSign.data(), 1, mSign.size_bytes(), mStream));

    using FunctorT = sdf_detail::SurfaceRepFunctor<BuildT>;
    util::cuda::operatorKernel<FunctorT><<<tree.mNodeCount[0], FunctorT::MaxThreadsPerBlock, 0, mStream>>>(
        dGrid, mSurfaceLabels.data(), mSurfaceCount, mSurfaceRepresentatives.data());
    cudaCheckError();
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
void MeshToSDF<BuildT>::processSurface(uint32_t surfaceID)
{
    const auto*    dSurfaceLabels    = mSurfaceLabels.data();
    const auto*    dOriginalGrid     = this->deviceGrid();
    const uint32_t originalLeafCount = TraitsT::getTreeData(dOriginalGrid).mNodeCount[0];
    const float    voxelSize         = float(mMap.getVoxelSize()[0]);

    HandleT surfaceHandle(sdf_detail::allocate<std::byte>(0, mStream));
    auto surfaceUDF           = sdf_detail::allocate<float>(0, mStream);
    auto surfaceTriangleIndex = sdf_detail::allocate<uint32_t>(0, mStream);
    const GridT*    dGrid          = dOriginalGrid;
    const float*    dUDF           = this->deviceUDF();
    const uint32_t* dTriangleIndex = reinterpret_cast<const uint32_t*>(mTriangleIndex.data());

    if (mSurfaceCount > 1) {
        auto maskBuffer = sdf_detail::allocate<Mask<3>>(originalLeafCount, mStream);
        auto* dSurfaceMask = maskBuffer.data();
        using SurfaceMaskFunctorT = sdf_detail::SurfaceMaskFunctor<BuildT>;
        util::cuda::operatorKernel<SurfaceMaskFunctorT><<<originalLeafCount, SurfaceMaskFunctorT::MaxThreadsPerBlock, 0, mStream>>>(
            dOriginalGrid, dSurfaceLabels, surfaceID, dSurfaceMask);
        cudaCheckError();

        PruneGrid<BuildT> pruner(dOriginalGrid, dSurfaceMask, mStream);
        surfaceHandle = pruner.getHandle(mGridHandle.buffer());
        dGrid = surfaceHandle.template deviceGrid<BuildT>();

        const uint64_t activeCount = TraitsT::getActiveVoxelCount(dGrid);
        surfaceUDF = sdf_detail::allocate<float>(activeCount + 1, mStream);
        surfaceTriangleIndex = sdf_detail::allocate<uint32_t>(activeCount + 1, mStream);
        cudaCheck(cudaMemsetAsync(surfaceUDF.data(), 0, surfaceUDF.size_bytes(), mStream));
        cudaCheck(cudaMemsetAsync(surfaceTriangleIndex.data(), 0xFF, surfaceTriangleIndex.size_bytes(), mStream));

        using InjectUDFFunctorT = util::cuda::InjectGridDataFunctor<BuildT, float>;
        using InjectIndexFunctorT = util::cuda::InjectGridDataFunctor<BuildT, uint32_t>;
        util::cuda::operatorKernel<InjectUDFFunctorT><<<originalLeafCount, InjectUDFFunctorT::MaxThreadsPerBlock, 0, mStream>>>(
            dOriginalGrid, dGrid, this->deviceUDF(), surfaceUDF.data());
        cudaCheckError();
        util::cuda::operatorKernel<InjectIndexFunctorT><<<originalLeafCount, InjectIndexFunctorT::MaxThreadsPerBlock, 0, mStream>>>(
            dOriginalGrid, dGrid, reinterpret_cast<const uint32_t*>(mTriangleIndex.data()),
            surfaceTriangleIndex.data());
        cudaCheckError();
        cudaCheck(cudaStreamSynchronize(mStream));

        dUDF = surfaceUDF.data();
        dTriangleIndex = surfaceTriangleIndex.data();
    }

    auto derived = sdf_detail::pruneBarrier(
        dGrid, dUDF, voxelSize, mIsoValue, mStream);
    const auto* dDerivedGrid = derived.template deviceGrid<BuildT>();

    ConnectedComponents<BuildT> components(dDerivedGrid, mStream);
    const auto componentLabels = components.getVoxelLabelsAndCount();

    auto derivedSign = sdf_detail::signNonBarrierComponents(
        dDerivedGrid, componentLabels.first.data(), mStream);
    auto partialSign = sdf_detail::injectNonBarrierSigns(
        dGrid, dDerivedGrid, derivedSign, mStream);

    auto sign = sdf_detail::allocate<int8_t>(0, mStream);
    switch (mBarrierSigning) {
    case BarrierSigning::Ball:
        sign = sdf_detail::signBarrierWithBalls(
            dGrid, partialSign, dUDF, voxelSize, 32, mBallStencilRadius,
            mIsoValue, mStream);
        break;
    case BarrierSigning::Heuristic:
        sign = sdf_detail::signBarrierHeuristic(
            dGrid, partialSign, dTriangleIndex, mDevicePoints, mDeviceTriangles, mMap,
            mIsoValue, voxelSize, mStream);
        break;
    case BarrierSigning::Interior:
    default:
        sign = sdf_detail::signBarrierAsInterior(dGrid, partialSign, mStream);
        break;
    }

    const int8_t* dSign = sign.data();
    if (mSurfaceCount == 1) {
        mSign = std::move(sign);
        return;
    }

    const auto invertMasks = sdf_detail::buildInvertMasks(dGrid, dSign, mStream);
    const unsigned long long* dRepresentatives = mSurfaceRepresentatives.data();
    uint32_t* dNestingDepth = mNestingDepth.data();
    using NestingFunctorT = sdf_detail::AccumulateNestingFunctor<BuildT>;
    util::cuda::lambdaKernel<<<1, mSurfaceCount, 0, mStream>>>(
        mSurfaceCount, NestingFunctorT{}, dGrid, dRepresentatives, dSign,
        invertMasks.leaf.data(), invertMasks.lower.data(), invertMasks.upper.data(),
        invertMasks.rootInterior.data(),
        invertMasks.rootTileMin, invertMasks.rootDims, surfaceID, dNestingDepth);
    cudaCheckError();

    using InjectSignFunctorT = util::cuda::InjectGridDataFunctor<BuildT, int8_t>;
    const uint32_t leafCount = TraitsT::getTreeData(dGrid).mNodeCount[0];
    util::cuda::operatorKernel<InjectSignFunctorT><<<leafCount, InjectSignFunctorT::MaxThreadsPerBlock, 0, mStream>>>(
        dGrid, dOriginalGrid, dSign, mSign.data());
    cudaCheckError();
    cudaCheck(cudaStreamSynchronize(mStream));
} // MeshToSDF<BuildT>::processSurface

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
void MeshToSDF<BuildT>::resolveSigns()
{
    if (mSurfaceCount <= 1) return;

    const uint64_t activeCount =
        TraitsT::getActiveVoxelCount(this->deviceGrid());
    util::cuda::lambdaKernel<<<(unsigned int)((activeCount + 256) / 256), 256, 0, mStream>>>(
        activeCount + 1, sdf_detail::ResolveNestingFunctor{}, mSurfaceLabels.data(),
        mNestingDepth.data(), mSurfaceCount, mNestingRule == NestingRule::EvenOdd, mSign.data());
    cudaCheckError();
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
void MeshToSDF<BuildT>::fillInvertMasks()
{
    auto masks = sdf_detail::buildInvertMasks(deviceGrid(), deviceSign(), mStream);
    mLeafInvertMask = std::move(masks.leaf);
    mLowerInvertMask = std::move(masks.lower);
    mUpperInvertMask = std::move(masks.upper);
    mRootInterior = std::move(masks.rootInterior);
    mRootTileMin = masks.rootTileMin;
    mRootDims = masks.rootDims;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
typename MeshToSDF<BuildT>::HandleT MeshToSDF<BuildT>::finalizeOutput()
{
    finalizeDistances();
    HandleT handle = bakeBlindData();
    releaseIntermediates();
    return handle;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
void MeshToSDF<BuildT>::finalizeDistances()
{
    if (mIsoValue == 0.f || !mSign.size()) return;

    const float voxelSize = float(mMap.getVoxelSize()[0]);

    // Match the half-diagonal barrier used during signing.
    const float interiorFloor = 0.8660254f * voxelSize;

    using FunctorT = sdf_detail::IsoMagnitudeFunctor<BuildT>;
    const uint32_t leaves = TraitsT::getTreeData(this->deviceGrid()).mNodeCount[0];
    if (leaves)
        util::cuda::operatorKernel<FunctorT><<<leaves, FunctorT::MaxThreadsPerBlock, 0, mStream>>>(
            this->deviceGrid(), reinterpret_cast<float*>(mUDF.data()), mSign.data(), mIsoValue, interiorFloor);
    cudaCheckError();

    // The voxel kernel does not visit the background slot.
    const float bg = mBandWidth * voxelSize;
    cudaCheck(cudaMemcpyAsync(mUDF.data(), &bg, sizeof(float), cudaMemcpyHostToDevice, mStream));
    cudaCheck(cudaStreamSynchronize(mStream));
} // MeshToSDF<BuildT>::finalizeDistances

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
typename MeshToSDF<BuildT>::HandleT MeshToSDF<BuildT>::bakeBlindData()
{
    namespace tc = nanovdb::tools::cuda;

    const auto*    dGrid = this->deviceGrid();
    const uint64_t slots  = TraitsT::getActiveVoxelCount(dGrid) + 1;
    const auto&    tree   = TraitsT::getTreeData(dGrid);
    const uint32_t leaves = tree.mNodeCount[0], lowers = tree.mNodeCount[1], uppers = tree.mNodeCount[2];

    auto  sdfBuffer = sdf_detail::allocate<float>(slots, mStream);
    auto* dSDF  = sdfBuffer.data();
    util::cuda::lambdaKernel<<<(unsigned int)((slots + 255) / 256), 256, 0, mStream>>>(
        slots, sdf_detail::SignedDistanceFunctor{}, this->deviceUDF(), this->deviceSign(), dSDF);
    cudaCheckError();

    // mGridHandle.buffer() is only the prototype addBlindData takes the allocation resource from.
    HandleT h = tc::addBlindData<BuildT, float>(dGrid, dSDF, slots,
                   GridBlindDataClass::ChannelArray, GridBlindDataSemantic::LevelSet, "sdf",
                   mGridHandle.buffer(), mStream);

    auto addMask = [&](const void* dSrc, uint64_t words, const char* name) {
        if (!words) return;
        h = tc::addBlindData<BuildT, uint64_t>(h.template deviceGrid<BuildT>(),
                static_cast<const uint64_t*>(dSrc), words,
                GridBlindDataClass::ChannelArray, GridBlindDataSemantic::Unknown, name,
                mGridHandle.buffer(), mStream);
    };
    addMask(this->deviceLeafInvertMask(),  uint64_t(leaves) * (sizeof(nanovdb::Mask<3>) / 8), "leaf_invert");
    addMask(this->deviceLowerInvertMask(), uint64_t(lowers) * (sizeof(nanovdb::Mask<4>) / 8), "lower_invert");
    addMask(this->deviceUpperInvertMask(), uint64_t(uppers) * (sizeof(nanovdb::Mask<5>) / 8), "upper_invert");

    const nanovdb::Coord tileMin = this->rootTileMin(), dims = this->rootTileDims();
    const uint64_t       cells   = uint64_t(dims[0]) * dims[1] * dims[2];
    if (cells) {
        h = tc::addBlindData<BuildT, uint8_t>(h.template deviceGrid<BuildT>(),
                this->deviceRootInterior(), cells,
                GridBlindDataClass::ChannelArray, GridBlindDataSemantic::Unknown, "root_interior",
                mGridHandle.buffer(), mStream);

        const int32_t extent[6] = {tileMin[0], tileMin[1], tileMin[2], dims[0], dims[1], dims[2]};
        auto dExtent = sdf_detail::allocate<int32_t>(6, mStream);
        cudaCheck(cudaMemcpyAsync(dExtent.data(), extent, sizeof(extent), cudaMemcpyHostToDevice, mStream));
        cudaCheck(cudaStreamSynchronize(mStream));
        h = tc::addBlindData<BuildT, int32_t>(h.template deviceGrid<BuildT>(), dExtent.data(), 6,
                GridBlindDataClass::ChannelArray, GridBlindDataSemantic::Unknown, "root_extent",
                mGridHandle.buffer(), mStream);
    }
    return h;
} // MeshToSDF<BuildT>::bakeBlindData

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
void MeshToSDF<BuildT>::releaseIntermediates()
{
    mSurfaceLabels.destroy();
    mSurfaceCount = 0;
    mSurfaceRepresentatives.destroy();
    mNestingDepth.destroy();
    mTriangleIndex.destroy();
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

} // namespace tools::cuda

} // namespace nanovdb

#endif // NVIDIA_TOOLS_CUDA_MESHTOSDF_CUH_HAS_BEEN_INCLUDED
