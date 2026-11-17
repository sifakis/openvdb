// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/tools/cuda/LevelSetTracker.cuh

    \authors JaeHyun Lee and Efty Sifakis

    \brief Tracking (dilation, renormalization and pruning) of a narrow-band level set stored as a NanoVDB
           indexGrid plus a value sidecar.

    \note Comments marked "Provisional" record choices made without a final decision; IDs such as C1
          refer to the open questions in nanovdb/examples/ex_renormalize_levelset_cuda/RENORMALIZATION_DESIGN.md,
          and "the reference" to the CUDA implementation imported under that directory's Benchmark/.

    \warning The header file contains cuda device code so be sure
             to only include it in .cu files (or other .cuh files)
*/

#ifndef NVIDIA_TOOLS_CUDA_LEVELSETTRACKER_CUH_HAS_BEEN_INCLUDED
#define NVIDIA_TOOLS_CUDA_LEVELSETTRACKER_CUH_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h>
#include <nanovdb/GridHandle.h>
#include <nanovdb/cuda/Buffer.h>
#include <nanovdb/cuda/DeviceResource.h>
#include <nanovdb/math/FiniteDifference.h>
#include <nanovdb/math/Stencils.h>
#include <nanovdb/tools/cuda/DilateGrid.cuh>
#include <nanovdb/tools/cuda/PropagateSign.cuh>
#include <nanovdb/tools/cuda/PruneGrid.cuh>
#include <nanovdb/tools/cuda/VoxelBlockManager.cuh>
#include <nanovdb/util/cuda/DeviceGridTraits.cuh>
#include <nanovdb/util/cuda/Injection.cuh>
#include <nanovdb/util/cuda/Timer.h>
#include <nanovdb/util/cuda/Util.h> // for operatorKernel

#include <cuda/std/algorithm> // cuda::std::binary_search

#include <cstddef>      // std::byte
#include <stdexcept>    // std::runtime_error
#include <type_traits>  // std::is_same, std::is_floating_point
#include <utility>      // std::pair, std::move

namespace nanovdb {

namespace tools::cuda {

/// @brief Maintains a narrow-band level set: track() dilates the band by one voxel, renormalizes it and prunes
///        it back to the background, the counterpart of openvdb::tools::LevelSetTracker::track().
/// @details The level set is a ValueOnIndex grid and a sidecar phi holding one value per grid index.
///          Slot 0 of the sidecar holds the background (the band's half-width in world units). A voxel outside
///          the band takes the background or its negation, as told by the invert masks of PropagateSign.
///          The tracker owns the level set (State), the time-stepping scratch, and the VoxelBlockManager,
///          which all depend on the grid's numbering of its active voxels and nodes.
/// @tparam BuildT Grid build type; must be ValueOnIndex.
/// @tparam ValueT Sidecar element type.
/// @tparam ResourceT Stream-ordered resource of the grid and the sidecar, used for all other allocations too.
template<typename BuildT, typename ValueT, typename ResourceT = nanovdb::cuda::DeviceResource>
class LevelSetTracker
{
    static_assert(std::is_same<BuildT, ValueOnIndex>::value, "LevelSetTracker requires a ValueOnIndex grid");
    static_assert(std::is_floating_point<ValueT>::value, "LevelSetTracker requires floating-point values");
    static_assert(nanovdb::cuda::is_async_resource<ResourceT>::value,
                  "LevelSetTracker allocates stream-ordered storage and requires an AsyncResource");

public:
    using GridT = NanoGrid<BuildT>;

    template <typename T>
    using BufferT = nanovdb::cuda::Buffer<T, nanovdb::cuda::ResourceRef<ResourceT>>;

    using HandleT = GridHandle<BufferT<std::byte>>;

    /// @brief The level set: the grid, the values of its active voxels, and the sign of the background everywhere
    ///        else. The sidecar and the masks are indexed by the grid's numbering, so the six change together.
    struct State
    {
        HandleT                   grid;
        BufferT<ValueT>           phi;         // slot 0 = background; slot k = value of the active voxel with index k
        BufferT<nanovdb::Mask<3>> leafMasks;   // PropagateSign's invert masks: one per leaf, lower and upper node
        BufferT<nanovdb::Mask<4>> lowerMasks;
        BufferT<nanovdb::Mask<5>> upperMasks;
        BufferT<nanovdb::Coord>   rootOrigins; // PropagateSign's root-level result, sorted by (x, y, z)
    };

    /// @param grid Device handle of the index grid.
    /// @param phi Device sidecar with one value per grid index; slot 0 is overwritten with @a background.
    /// @param background Magnitude assigned to voxels outside the band.
    /// @param stream Stream all work of this tracker is ordered on.
    /// @param resource Allocation resource; must outlive the tracker and the State it releases.
    /// @throw std::runtime_error if the voxels are not isotropic or the leaf nodes are not sequential, or if
    ///        PropagateSign rejects the level set.
    /// @warning The level set must meet the preconditions of PropagateSign, which computes the invert masks.
    LevelSetTracker(HandleT&& grid, BufferT<ValueT>&& phi, ValueT background, cudaStream_t stream = 0,
                    ResourceT& resource = nanovdb::cuda::default_resource<ResourceT>())
        : mStream(stream)
        , mResource(&resource)
        , mState{std::move(grid), std::move(phi),
                 BufferT<nanovdb::Mask<3>>(stream, this->ref(), 0, nanovdb::cuda::noInit),
                 BufferT<nanovdb::Mask<4>>(stream, this->ref(), 0, nanovdb::cuda::noInit),
                 BufferT<nanovdb::Mask<5>>(stream, this->ref(), 0, nanovdb::cuda::noInit),
                 BufferT<nanovdb::Coord>(stream, this->ref(), 0, nanovdb::cuda::noInit)}
        , mScratch(stream, this->ref(), 0, nanovdb::cuda::noInit)
        , mBackground(background)
        , mDx(validate(mState.grid, stream))
        , mVBM(buildVoxelBlockManager<Log2BlockWidth, BufferT<std::byte>>(
              mState.grid.template deviceGrid<BuildT>(), 0, 0, 0, stream, &mState.grid.buffer()))
    {
        cudaCheck(cudaMemcpyAsync(mState.phi.data(), &mBackground, sizeof(ValueT), cudaMemcpyHostToDevice, mStream));
        propagateSign();
    }

    LevelSetTracker(const LevelSetTracker&) = delete;
    LevelSetTracker& operator=(const LevelSetTracker&) = delete;
    LevelSetTracker(LevelSetTracker&&) = default;
    LevelSetTracker& operator=(LevelSetTracker&&) = default;

    /// @brief Toggle on and off verbose mode
    /// @param level Verbose level: 0=quiet, 1=timing
    void setVerbose(int level = 1) { mVerbose = level; }

    /// @brief Moves the level set out; the tracker cannot be used afterwards.
    State release()
    {
        mVBM.reset();
        return std::move(mState);
    }

    // Provisional (A2): schemes are runtime enums checked in normalize(), rather than compile-time tags.
    math::BiasedGradientScheme getSpatialScheme() const { return mSpatialScheme; }
    void setSpatialScheme(math::BiasedGradientScheme scheme) { mSpatialScheme = scheme; }

    math::TemporalIntegrationScheme getTemporalScheme() const { return mTemporalScheme; }
    void setTemporalScheme(math::TemporalIntegrationScheme scheme) { mTemporalScheme = scheme; }

    /// @brief Number of pseudo-time steps per normalize(); each step moves information about one voxel.
    int getNormCount() const { return mNormCount; }
    void setNormCount(int n) { mNormCount = n; }

    /// @brief One maintenance cycle of the narrow band, to be called after the active values have moved by less
    ///        than one voxel: dilate, renormalize, and prune the voxels that reach the background.
    void track()
    {
        dilate();
        normalize();
        prune(mBackground);
    }

    /// @brief Grows the band by one layer of face neighbors. A new voxel takes the value it had outside the band:
    ///        the background or its negation, as told by the invert masks.
    void dilate();

    /// @brief Removes the active voxels with |phi| >= @a threshold, and the nodes this empties.
    void prune(ValueT threshold);

    /// @brief Restores |grad phi| = 1 without moving the zero crossing, by integrating
    ///        d(phi)/d(tau) = S(phi) (1 - |grad phi|) in pseudo-time: normCount TVD-RK2 steps on the active values,
    ///        in place. The topology is unchanged.
    /// @note The stream is synchronized before returning.
    void normalize()
    {
        if (mSpatialScheme != math::HJWENO5_BIAS || mTemporalScheme != math::TVD_RK2)
            throw std::runtime_error("LevelSetTracker::normalize: only HJWENO5_BIAS with TVD_RK2 is implemented");
        if (mState.phi.empty()) throw std::runtime_error("LevelSetTracker::normalize: the grid has been released");
        if (!mVBM.blockCount()) return;

        allocateScratch(scratchCount(math::TVD_RK2));
        // CFL number 0.9 / sqrt(3), as in openvdb::tools::LevelSetAdvection. This differs from
        // openvdb::tools::LevelSetTracker, whose 0.9 was measured to be unstable off the grid axes.
        const ValueT dt = ValueT(0.9) / math::Sqrt(ValueT(3)) * mDx;
        for (int n = 0; n < mNormCount; ++n) {
            eulerStep<0, 1, Phi, Phi, Scratch>(dt); // scratch = phi - dt * L(phi)
            eulerStep<1, 2, Scratch, Phi, Phi>(dt); // phi = (phi + scratch - dt * L(scratch)) / 2
        }
        cudaCheck(cudaStreamSynchronize(mStream));
    }

private:

    // --- Stages (validate and propagateSign run in the constructor; replaceTopology ends dilate and prune;
    //     the others run in normalize). ---

    // Throw unless the voxels are isotropic and the leaf nodes are sequential; return the voxel size.
    static ValueT validate(const HandleT& grid, cudaStream_t stream);

    // Compute the invert masks and the root origins of mState from its grid and sidecar, with PropagateSign.
    void propagateSign();

    // Replace the grid and the sidecar of mState, and recompute what depends on the grid's numbering: the invert
    // masks, the root origins and the VoxelBlockManager.
    void replaceTopology(HandleT&& grid, BufferT<ValueT>&& phi);

    // Allocate count RK temporaries in mScratch. Skipped when the size already matches.
    void allocateScratch(int count);

    // One Euler stage over all active voxels:
    // buffer(ResultID) = alpha * buffer(BlendID) + (1 - alpha) * (s - dt * L(s)), s = buffer(StencilID),
    // alpha = Numerator / Denominator.
    template<int Numerator, int Denominator, int StencilID, int BlendID, int ResultID>
    void eulerStep(ValueT dt);

    // Buffer IDs bound by eulerStep: phi, then the RK temporaries in mScratch.
    enum BufferID : int { Phi = 0, Scratch = 1 };
    ValueT* buffer(int id) { return id ? mScratch.data() + (id - 1) * mState.phi.size() : mState.phi.data(); }

    // Number of RK temporaries a temporal scheme needs.
    static constexpr int scratchCount(math::TemporalIntegrationScheme scheme) { return scheme == math::TVD_RK2 ? 1 : 0; }

    static constexpr int Log2BlockWidth = 7; // 128 active voxels per CUDA block. Provisional (B2): fixed here

    /// @brief Borrowed handle to the injected resource, for the buffers allocated by the tracker.
    nanovdb::cuda::ResourceRef<ResourceT> ref() { return nanovdb::cuda::ResourceRef<ResourceT>(*mResource); }

    cudaStream_t    mStream{0};
    int             mVerbose{0};
    ResourceT*      mResource{nullptr}; // declared before mState and mScratch, which allocate through ref()
    State           mState;
    BufferT<ValueT> mScratch;    // scratchCount() x mState.phi.size(); contents undefined between calls
    ValueT          mBackground; // Provisional: no separate half-width member (design note 3.5)
    ValueT          mDx;         // declared before mVBM: validate() must run before the VBM is built
    VoxelBlockManagerHandle<BufferT<std::byte>> mVBM;

    math::BiasedGradientScheme      mSpatialScheme{math::HJWENO5_BIAS};
    math::TemporalIntegrationScheme mTemporalScheme{math::TVD_RK2};
    int                             mNormCount{3};
}; // tools::cuda::LevelSetTracker<BuildT, ValueT, ResourceT>

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename ValueT, typename ResourceT>
ValueT LevelSetTracker<BuildT, ValueT, ResourceT>::validate(const HandleT& grid, cudaStream_t stream)
{
    GridData data;
    cudaCheck(cudaMemcpyAsync(&data, grid.template deviceGrid<BuildT>(), sizeof(GridData),
                              cudaMemcpyDeviceToHost, stream));
    cudaCheck(cudaStreamSynchronize(stream));
    const auto* header = reinterpret_cast<const NanoGrid<BuildT>*>(&data); // reads GridData members only

    const Vec3d dx = header->voxelSize();
    if (dx[0] != dx[1] || dx[0] != dx[2]) // Provisional: exact; OpenVDB's uniform-scale test uses a tolerance
        throw std::runtime_error("LevelSetTracker: voxels must be isotropic");
    if (!header->template isSequential<0>())
        throw std::runtime_error("LevelSetTracker: leaf nodes must be sequential");
    return ValueT(dx[0]);
}// LevelSetTracker<BuildT, ValueT, ResourceT>::validate

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename ValueT, typename ResourceT>
void LevelSetTracker<BuildT, ValueT, ResourceT>::propagateSign()
{
    util::cuda::Timer timer(mStream); // local: Timer is not safely movable, and the tracker is
    if (mVerbose==1) timer.start("Computing the invert masks");
    const GridT* dGrid = mState.grid.template deviceGrid<BuildT>();
    const auto treeData = util::cuda::DeviceGridTraits<BuildT>::getTreeData(dGrid, mStream);
    mState.leafMasks  = BufferT<nanovdb::Mask<3>>(mStream, this->ref(), treeData.mNodeCount[0], nanovdb::cuda::noInit);
    mState.lowerMasks = BufferT<nanovdb::Mask<4>>(mStream, this->ref(), treeData.mNodeCount[1], nanovdb::cuda::noInit);
    mState.upperMasks = BufferT<nanovdb::Mask<5>>(mStream, this->ref(), treeData.mNodeCount[2], nanovdb::cuda::noInit);

    PropagateSign<BuildT, ValueT, ResourceT> propagator(dGrid, mState.phi.data(), mStream, *mResource);
    mState.rootOrigins = propagator.propagate(mState.leafMasks.data(), mState.lowerMasks.data(),
                                              mState.upperMasks.data()).first;
    if (mVerbose==1) timer.stop();
}// LevelSetTracker<BuildT, ValueT, ResourceT>::propagateSign

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename ValueT, typename ResourceT>
void LevelSetTracker<BuildT, ValueT, ResourceT>::allocateScratch(int count)
{
    const size_t size = size_t(count) * mState.phi.size();
    if (mScratch.size() == size) return;

    util::cuda::Timer timer(mStream); // local: Timer is not safely movable, and the tracker is
    if (mVerbose==1) timer.start("Allocating RK scratch");
    mScratch = BufferT<ValueT>(mStream, this->ref(), size, nanovdb::cuda::noInit);
    if (mVerbose==1) timer.stop();
}// LevelSetTracker<BuildT, ValueT, ResourceT>::allocateScratch

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace levelset::detail {

/// @brief Return the invert bit of @a ijk: true if the inactive element of @a grid that holds @a ijk takes the
///        negative background. The element is the first level of a tree descent with no child at @a ijk: an
///        upper-node-sized region without a root-table entry, a childless upper or lower slot, or a leaf voxel.
/// @details The bit of an active voxel is off, so this returns false for an active voxel: its sign is that of its
///          own value.
/// @param grid The grid PropagateSign ran on.
/// @param leafMasks, lowerMasks, upperMasks The invert masks PropagateSign filled.
/// @param rootOrigins, rootOriginCount The origins PropagateSign returned, sorted by (x, y, z).
template <typename BuildT>
__hostdev__ inline bool isInverted(const NanoGrid<BuildT>& grid, const nanovdb::Coord& ijk,
                                   const nanovdb::Mask<3>* leafMasks, const nanovdb::Mask<4>* lowerMasks,
                                   const nanovdb::Mask<5>* upperMasks, const nanovdb::Coord* rootOrigins,
                                   uint64_t rootOriginCount)
{
    using UpperT = NanoUpper<BuildT>;
    using LowerT = NanoLower<BuildT>;
    using LeafT  = NanoLeaf<BuildT>;
    const auto& tree = grid.tree();

    const auto* tile = tree.root().probeTile(ijk);
    if (!tile) return ::cuda::std::binary_search(rootOrigins, rootOrigins + rootOriginCount,
                                                 ijk & ~int(UpperT::DIM - 1));
    NANOVDB_ASSERT(tile->isChild()); // PropagateSign rejects root value tiles

    const auto*    upper = tree.root().getChild(tile);
    const uint32_t u     = UpperT::CoordToOffset(ijk);
    if (!upper->isChild(u)) return upperMasks[upper - tree.getFirstUpper()].isOn(u);

    const auto*    lower = upper->getChild(u);
    const uint32_t l     = LowerT::CoordToOffset(ijk);
    if (!lower->isChild(l)) return lowerMasks[lower - tree.getFirstLower()].isOn(l);

    return leafMasks[lower->getChild(l) - tree.getFirstLeaf()].isOn(LeafT::CoordToOffset(ijk));
} // isInverted

// One thread per voxel of a leaf of the dilated grid: copy the value of a voxel that was active in the old grid,
// and give a new voxel the value it had outside the old band, the background or its negation.
template<typename BuildT, typename ValueT>
struct DilateValuesFunctor
{
    static constexpr int MaxThreadsPerBlock = NanoLeaf<BuildT>::SIZE;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(const NanoGrid<BuildT>* oldGrid,
                               const NanoGrid<BuildT>* newGrid,
                               const ValueT* oldPhi,
                               const nanovdb::Mask<3>* oldLeafMasks,
                               const nanovdb::Mask<4>* oldLowerMasks,
                               const nanovdb::Mask<5>* oldUpperMasks,
                               const nanovdb::Coord* oldRootOrigins,
                               uint64_t oldRootOriginCount,
                               ValueT background,
                               ValueT* newPhi) const
    {
        const auto&    newLeaf = newGrid->tree().template getFirstNode<0>()[blockIdx.x];
        const uint32_t n       = threadIdx.x;
        if (!newLeaf.isActive(n)) return;

        const Coord ijk     = newLeaf.offsetToGlobalCoord(n);
        const auto* oldLeaf = oldGrid->tree().root().probeLeaf(ijk);
        if (oldLeaf && oldLeaf->isActive(n))
            newPhi[newLeaf.getValue(n)] = oldPhi[oldLeaf->getValue(n)];
        else
            newPhi[newLeaf.getValue(n)] = isInverted(*oldGrid, ijk, oldLeafMasks, oldLowerMasks, oldUpperMasks,
                                                     oldRootOrigins, oldRootOriginCount) ? -background : background;
    }
}; // DilateValuesFunctor

// One thread per voxel of a leaf: turn on the bit of every active voxel with |phi| < threshold.
template<typename BuildT, typename ValueT>
struct RetainMaskFunctor
{
    static constexpr int MaxThreadsPerBlock = NanoLeaf<BuildT>::SIZE;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(const NanoGrid<BuildT>* grid, const ValueT* phi, ValueT threshold,
                               nanovdb::Mask<3>* retainMasks) const
    {
        const auto&    leaf = grid->tree().template getFirstNode<0>()[blockIdx.x];
        const uint32_t n    = threadIdx.x;
        auto&          mask = retainMasks[blockIdx.x];
        if (n < nanovdb::Mask<3>::WORD_COUNT) mask.words()[n] = 0;
        __syncthreads();
        if (leaf.isActive(n) && math::Abs(phi[leaf.getValue(n)]) < threshold) mask.setOnAtomic(n);
    }
}; // RetainMaskFunctor

// One thread per active voxel (decoded through the VBM): gather the 19-point WENO stencil, give the taps that
// fall outside the band the background with the sign told by the invert masks, and apply one Euler step of
// d(phi)/d(tau) = S(phi) (1 - |grad phi|), blended with phi by alpha = Numerator / Denominator.
template<typename BuildT, typename ValueT, int Log2BlockWidth, int Numerator, int Denominator>
struct NormalizeEulerFunctor
{
    using VBM = VoxelBlockManager<Log2BlockWidth>;
    // Provisional (B4): WenoStencil serves only for SIZE, WenoPt and normSqGrad; the gather is written here.
    using StencilT = math::WenoStencil<NanoGrid<ValueT>>;

    static constexpr int MaxThreadsPerBlock = VBM::BlockWidth;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    /// @brief Position in the WenoPt layout of the tap at offset @a d (|d| in 1..3) along @a axis.
    __hostdev__ static constexpr int tap(int axis, int d) { return 1 + 6 * axis + (d < 0 ? d + 3 : d + 2); }

    __device__ void operator()(const NanoGrid<BuildT>* grid,
                               const uint32_t* firstLeafID,
                               const uint64_t* jumpMap,
                               uint64_t firstOffset,
                               const ValueT* stencilBuffer,
                               const ValueT* blendBuffer,
                               ValueT* resultBuffer,
                               ValueT dt,
                               ValueT invDx,
                               ValueT dx2,
                               const nanovdb::Mask<3>* leafMasks,
                               const nanovdb::Mask<4>* lowerMasks,
                               const nanovdb::Mask<5>* upperMasks,
                               const nanovdb::Coord* rootOrigins,
                               uint64_t rootOriginCount,
                               ValueT background) const
    {
        static_assert(tap(0, -3) == math::WenoPt<-3, 0, 0>::idx && tap(1, 1) == math::WenoPt<0, 1, 0>::idx &&
                      tap(2, 3) == math::WenoPt<0, 0, 3>::idx, "tap() must match the WenoPt layout");

        uint32_t leafIndex;
        uint16_t voxelOffset;
        VBM::decodeInverseMap(grid, firstLeafID[blockIdx.x], jumpMap + VBM::JumpMapLength * blockIdx.x,
                              firstOffset + uint64_t(blockIdx.x) * VBM::BlockWidth, threadIdx.x,
                              leafIndex, voxelOffset);
        if (leafIndex == VBM::UnusedLeafIndex) return;

        const auto& tree = grid->tree();
        const auto& leaf = tree.template getFirstNode<0>()[leafIndex];
        const Coord ijk      = leaf.offsetToGlobalCoord(voxelOffset);
        const Coord localIjk = NanoLeaf<BuildT>::OffsetToLocalCoord(voxelOffset);

        // Gather the sidecar index of every tap; index 0 means "no value": the tap lies outside the band.
        uint64_t idx[StencilT::SIZE] = {};
        idx[0] = leaf.getValue(voxelOffset);
        #pragma unroll
        for (int axis = 0; axis < 3; ++axis) {
            // Taps reach at most 3 voxels, so only the neighbor leaf on the voxel's side of its own leaf is needed.
            const bool upper = localIjk[axis] & 4;
            const NanoLeaf<BuildT>* leaves[3] = {nullptr, &leaf, nullptr};
            Coord neighborIjk = ijk;
            neighborIjk[axis] += upper ? 4 : -4;
            leaves[upper ? 2 : 0] = tree.root().probeLeaf(neighborIjk);
            const int stride = 1 << (3 * (2 - axis));
            #pragma unroll
            for (int d = -3; d <= 3; ++d) {
                if (d == 0) continue;
                const int tapLocal = localIjk[axis] + d; // in [-3, 10]: leaves[0] below 0, leaves[2] above 7
                if (const NanoLeaf<BuildT>* tapLeaf = leaves[(tapLocal + 8) >> 3])
                    idx[tap(axis, d)] = tapLeaf->getValue(uint32_t(voxelOffset + ((tapLocal & 7) - localIjk[axis]) * stride));
            }
        }

        // A tap in the band reads its value; a tap outside it takes the background or its negation.
        ValueT v[StencilT::SIZE];
        v[0] = stencilBuffer[idx[0]];
        #pragma unroll
        for (int axis = 0; axis < 3; ++axis) {
            #pragma unroll
            for (int d = -3; d <= 3; ++d) {
                if (d == 0) continue;
                const int i = tap(axis, d);
                if (idx[i]) {
                    v[i] = stencilBuffer[idx[i]];
                } else {
                    Coord tapIjk = ijk;
                    tapIjk[axis] += d;
                    v[i] = isInverted(*grid, tapIjk, leafMasks, lowerMasks, upperMasks, rootOrigins, rootOriginCount)
                               ? -background : background;
                }
            }
        }

        // Index-space gradient (invDx2 = 1); the physical scale enters through invDx below.
        // The third argument is WENO5's scale2 (epsilon = 1e-6 * scale2). It is dx^2, the scale of the squared
        // differences of v that epsilon is compared with, so that the result does not depend on the unit of
        // length; OpenVDB uses 0.01 for every dx. Provisional (C1): RealT is ValueT.
        const ValueT normSqGradPhi = StencilT::normSqGrad(v, ValueT(1), dx2);
        const ValueT phi0 = v[0];
        ValueT s = phi0 / (math::Sqrt(math::Pow2(phi0) + normSqGradPhi) + math::Tolerance<ValueT>::value());
        s = phi0 - dt * s * (math::Sqrt(normSqGradPhi) * invDx - ValueT(1));

        constexpr ValueT alpha = ValueT(Numerator) / ValueT(Denominator);
        resultBuffer[idx[0]] = Numerator ? alpha * blendBuffer[idx[0]] + (ValueT(1) - alpha) * s : s;
    }
}; // NormalizeEulerFunctor

} // namespace levelset::detail

template<typename BuildT, typename ValueT, typename ResourceT>
template<int Numerator, int Denominator, int StencilID, int BlendID, int ResultID>
void LevelSetTracker<BuildT, ValueT, ResourceT>::eulerStep(ValueT dt)
{
    static_assert(ResultID != StencilID, "a stage must not overwrite the buffer its stencil gathers from");

    using Op = levelset::detail::NormalizeEulerFunctor<BuildT, ValueT, Log2BlockWidth, Numerator, Denominator>;
    util::cuda::Timer timer(mStream); // local: Timer is not safely movable, and the tracker is
    if (mVerbose==1) timer.start("Euler stage of renormalization");
    util::cuda::operatorKernel<Op>
        <<<unsigned(mVBM.blockCount()), Op::MaxThreadsPerBlock, 0, mStream>>>(
            mState.grid.template deviceGrid<BuildT>(), mVBM.deviceFirstLeafID(), mVBM.deviceJumpMap(), mVBM.firstOffset(),
            buffer(StencilID), buffer(BlendID), buffer(ResultID), dt, ValueT(1) / mDx, mDx * mDx,
            mState.leafMasks.data(), mState.lowerMasks.data(), mState.upperMasks.data(),
            mState.rootOrigins.data(), mState.rootOrigins.size(), mBackground);
    cudaCheckError();
    if (mVerbose==1) timer.stop();
}// LevelSetTracker<BuildT, ValueT, ResourceT>::eulerStep

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename ValueT, typename ResourceT>
void LevelSetTracker<BuildT, ValueT, ResourceT>::dilate()
{
    if (mState.phi.empty()) throw std::runtime_error("LevelSetTracker::dilate: the grid has been released");
    util::cuda::Timer timer(mStream); // local: Timer is not safely movable, and the tracker is

    if (mVerbose==1) timer.start("Dilating the grid");
    const GridT* oldGrid = mState.grid.template deviceGrid<BuildT>();
    DilateGrid<BuildT, ResourceT> dilator(oldGrid, mStream, *mResource);
    dilator.setOperation(morphology::NN_FACE);
    HandleT newHandle = dilator.build();

    if (mVerbose==1) timer.restart("Filling the values of the dilated grid");
    const GridT* newGrid = newHandle.template deviceGrid<BuildT>();
    const auto newTreeData = util::cuda::DeviceGridTraits<BuildT>::getTreeData(newGrid, mStream);
    BufferT<ValueT> newPhi(mStream, this->ref(), newTreeData.mVoxelCount + 1, nanovdb::cuda::noInit);
    cudaCheck(cudaMemcpyAsync(newPhi.data(), &mBackground, sizeof(ValueT), cudaMemcpyHostToDevice, mStream));
    if (newTreeData.mNodeCount[0]) {
        using Op = levelset::detail::DilateValuesFunctor<BuildT, ValueT>;
        util::cuda::operatorKernel<Op><<<newTreeData.mNodeCount[0], Op::MaxThreadsPerBlock, 0, mStream>>>(
            oldGrid, newGrid, mState.phi.data(), mState.leafMasks.data(), mState.lowerMasks.data(),
            mState.upperMasks.data(), mState.rootOrigins.data(), mState.rootOrigins.size(), mBackground,
            newPhi.data());
        cudaCheckError();
    }
    if (mVerbose==1) timer.stop();

    replaceTopology(std::move(newHandle), std::move(newPhi));
}// LevelSetTracker<BuildT, ValueT, ResourceT>::dilate

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename ValueT, typename ResourceT>
void LevelSetTracker<BuildT, ValueT, ResourceT>::prune(ValueT threshold)
{
    if (mState.phi.empty()) throw std::runtime_error("LevelSetTracker::prune: the grid has been released");
    const GridT* oldGrid = mState.grid.template deviceGrid<BuildT>();
    const uint32_t oldLeafCount = util::cuda::DeviceGridTraits<BuildT>::getTreeData(oldGrid, mStream).mNodeCount[0];
    if (!oldLeafCount) return;
    util::cuda::Timer timer(mStream); // local: Timer is not safely movable, and the tracker is

    if (mVerbose==1) timer.start("Computing the retain masks");
    BufferT<nanovdb::Mask<3>> retainMasks(mStream, this->ref(), oldLeafCount, nanovdb::cuda::noInit);
    using RetainOp = levelset::detail::RetainMaskFunctor<BuildT, ValueT>;
    util::cuda::operatorKernel<RetainOp><<<oldLeafCount, RetainOp::MaxThreadsPerBlock, 0, mStream>>>(
        oldGrid, mState.phi.data(), threshold, retainMasks.data());
    cudaCheckError();

    if (mVerbose==1) timer.restart("Pruning the grid");
    PruneGrid<BuildT, ResourceT> pruner(oldGrid, retainMasks.data(), mStream, *mResource);
    HandleT newHandle = pruner.build();

    if (mVerbose==1) timer.restart("Injecting the values into the pruned grid");
    const GridT* newGrid = newHandle.template deviceGrid<BuildT>();
    const auto newTreeData = util::cuda::DeviceGridTraits<BuildT>::getTreeData(newGrid, mStream);
    BufferT<ValueT> newPhi(mStream, this->ref(), newTreeData.mVoxelCount + 1, nanovdb::cuda::noInit);
    cudaCheck(cudaMemcpyAsync(newPhi.data(), &mBackground, sizeof(ValueT), cudaMemcpyHostToDevice, mStream));
    using InjectOp = util::cuda::InjectGridDataFunctor<BuildT, ValueT>;
    util::cuda::operatorKernel<InjectOp><<<oldLeafCount, InjectOp::MaxThreadsPerBlock, 0, mStream>>>(
        oldGrid, newGrid, mState.phi.data(), newPhi.data());
    cudaCheckError();
    if (mVerbose==1) timer.stop();

    replaceTopology(std::move(newHandle), std::move(newPhi));
}// LevelSetTracker<BuildT, ValueT, ResourceT>::prune

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename ValueT, typename ResourceT>
void LevelSetTracker<BuildT, ValueT, ResourceT>::replaceTopology(HandleT&& grid, BufferT<ValueT>&& phi)
{
    mVBM.reset(); // built on the grid being replaced
    mState.grid = std::move(grid);
    mState.phi  = std::move(phi);
    propagateSign();
    mVBM = buildVoxelBlockManager<Log2BlockWidth, BufferT<std::byte>>(
        mState.grid.template deviceGrid<BuildT>(), 0, 0, 0, mStream, &mState.grid.buffer());
}// LevelSetTracker<BuildT, ValueT, ResourceT>::replaceTopology

} // namespace tools::cuda

} // namespace nanovdb

#endif // NVIDIA_TOOLS_CUDA_LEVELSETTRACKER_CUH_HAS_BEEN_INCLUDED
