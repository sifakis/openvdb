// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/tools/cuda/MeshToSDF.cuh

    \authors Efty Sifakis and JaeHyun Lee

    \brief SDF-domain stages of the GPU mesh-to-signed-distance-field pipeline on NanoVDB index
           grids. This is the layer that knows about meshes / UDF / barriers / signs; it builds on
           two domain-agnostic primitives:
             - nanovdb::tools::cuda::MeshToGrid       (mesh -> narrow-band ValueOnIndex grid + UDF),
             - nanovdb::tools::cuda::ConnectedComponents (generic CC labeling of a ValueOnIndex grid).

           Stages implemented here:
             - computeDerivedTopology(): prune the surface/barrier shell (UDF^2 < 0.75*voxelSize^2)
               into a clean ValueOnIndex grid that CC then runs on.
             - signNonBarrier(): given a CC-labeled grid, sign every non-barrier active voxel
               (+ exterior / - interior; convention +outside / -inside).

    \warning The header file contains cuda device code so be sure
             to only include it in .cu files (or other .cuh files)
*/

#ifndef NVIDIA_TOOLS_CUDA_MESHTOSDF_CUH_HAS_BEEN_INCLUDED
#define NVIDIA_TOOLS_CUDA_MESHTOSDF_CUH_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h>
#include <nanovdb/GridHandle.h>
#include <nanovdb/cuda/DeviceBuffer.h>
#include <nanovdb/tools/cuda/PruneGrid.cuh>
#include <nanovdb/tools/cuda/ConnectedComponents.cuh>  // ConnectedComponents<> + ccVoxelComponentSlot()
#include <nanovdb/util/cuda/Injection.cuh>             // InjectGridDataFunctor
#include <nanovdb/util/cuda/DeviceGridTraits.cuh>
#include <nanovdb/util/cuda/Timer.h>
#include <nanovdb/util/cuda/Util.h>                    // operatorKernel, cudaCheck

namespace nanovdb {

namespace tools::cuda {

namespace sdf_detail {

static constexpr int LEAF_SIZE = 512;  // 8^3 voxels per leaf

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

/// @brief Reduce over active voxels to the global minimum x, carrying that voxel's component
///        representative into d_minKey = (unsigned(x) << 32) | uint32(rep). All min-x voxels are
///        exterior (and share one representative), so the low 32 bits resolve to the exterior rep.
template <typename BuildT>
struct FindExteriorRepFunctor
{
    static constexpr int MaxThreadsPerBlock         = LEAF_SIZE;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(const NanoGrid<BuildT>* d_grid, const nanovdb::Mask<3>* d_masks,
                               const uint64_t* d_offsets, const uint64_t* d_parent,
                               unsigned long long* d_minKey)
    {
        const int leafID = blockIdx.x, n = threadIdx.x;
        const auto& leaf = d_grid->tree().template getFirstNode<0>()[leafID];
        if (!leaf.isActive(uint32_t(n))) return;
        const nanovdb::Coord ijk = leaf.origin() + nanovdb::NanoLeaf<BuildT>::OffsetToLocalCoord(uint32_t(n));
        const uint64_t s   = ccVoxelComponentSlot(d_masks, d_offsets, uint32_t(leafID), uint32_t(n));
        const uint32_t rep = uint32_t(d_parent[s]);
        const uint32_t ux  = uint32_t(int64_t(ijk[0]) + (int64_t(1) << 31));  // x, shifted to unsigned-comparable
        const unsigned long long key = (static_cast<unsigned long long>(ux) << 32) | rep;
        atomicMin(d_minKey, key);
    }
};

/// @brief Write per-active-voxel signs (+1 exterior / -1 interior), indexed by leaf.getValue(n),
///        into d_sign (length activeVoxelCount+1; slot 0 = background, pre-filled +1).
template <typename BuildT>
struct SignNonBarrierFunctor
{
    static constexpr int MaxThreadsPerBlock         = LEAF_SIZE;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(const NanoGrid<BuildT>* d_grid, const nanovdb::Mask<3>* d_masks,
                               const uint64_t* d_offsets, const uint64_t* d_parent,
                               uint32_t exteriorRep, int8_t* d_sign)
    {
        const int leafID = blockIdx.x, n = threadIdx.x;
        const auto& leaf = d_grid->tree().template getFirstNode<0>()[leafID];
        if (!leaf.isActive(uint32_t(n))) return;
        const uint64_t s = ccVoxelComponentSlot(d_masks, d_offsets, uint32_t(leafID), uint32_t(n));
        d_sign[leaf.getValue(uint32_t(n))] = (uint32_t(d_parent[s]) == exteriorRep) ? int8_t(1) : int8_t(-1);
    }
};

} // namespace sdf_detail

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

/// @brief SDF-domain operations over a narrow-band ValueOnIndex grid + its UDF sidecar.
/// @tparam BuildT Build type of the index grid (e.g. nanovdb::ValueOnIndex).
template <typename BuildT>
class MeshToSDF
{
    using GridT = NanoGrid<BuildT>;

public:

    /// @brief Constructor
    /// @param stream optional CUDA stream (defaults to CUDA stream 0)
    MeshToSDF(cudaStream_t stream = 0) : mStream(stream), mTimer(stream) {}

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

    /// @brief Sign the non-barrier voxels of a CC-labeled grid: the component containing the global
    ///        minimum-x active voxel is the exterior (+); every other component is interior (-).
    ///        Convention: +outside / -inside. Requires @c cc to have completed
    ///        processLeafConnectedComponents() → processCrossLeafEdges() → processComponentLabels().
    /// @param d_grid the CC-labeled (derived) device grid
    /// @param cc     a ConnectedComponents instance holding that grid's labeling
    void signNonBarrier(const GridT* d_grid, ConnectedComponents<BuildT>& cc);

    /// @brief Carry the derived-grid signs (from signNonBarrier) back onto the original grid. The
    ///        derived grid is the barrier-pruned subset of the original, so the injection covers all
    ///        non-barrier voxels; barrier voxels (present only in the original) keep the sentinel 0
    ///        ("unsigned barrier") for step 5 to fill. Requires signNonBarrier() first.
    /// @param d_origGrid    the original (pre-prune) grid — injection target.
    /// @param d_derivedGrid the barrier-pruned grid that was signed.
    void injectSignsToOriginal(const GridT* d_origGrid, const GridT* d_derivedGrid);

    /// @brief Device pointer to the per-active-voxel sign array (+1 exterior / -1 interior),
    ///        valid after signNonBarrier(). Length activeVoxelCount+1, indexed by leaf.getValue(n);
    ///        slot 0 is the background (+1).
    int8_t* deviceVoxelSign() { return static_cast<int8_t*>(mVoxelSign.deviceData()); }

    /// @brief Global representative (slot) of the exterior component, valid after signNonBarrier().
    uint64_t exteriorRepresentative() const { return mExteriorRep; }

    /// @brief Device pointer to the per-active-voxel sign array on the ORIGINAL grid, valid after
    ///        injectSignsToOriginal(). Length origActiveVoxelCount+1, indexed by leaf.getValue(n);
    ///        slot 0 = background (+1); non-barrier voxels carry +1/-1; barrier voxels carry the
    ///        sentinel 0 (still unsigned, awaiting step 5).
    int8_t* deviceOriginalVoxelSign() { return static_cast<int8_t*>(mOriginalVoxelSign.deviceData()); }

private:

    cudaStream_t                 mStream{0};
    util::cuda::Timer            mTimer;
    int                          mVerbose{0};

    uint64_t                     mExteriorRep{0};  // representative slot of the exterior component
    nanovdb::cuda::DeviceBuffer  mVoxelSign;       // (derived activeVoxelCount+1) × int8_t: +1 ext / -1 int
    nanovdb::cuda::DeviceBuffer  mOriginalVoxelSign; // (orig activeVoxelCount+1) × int8_t: +1/-1 non-barrier, 0 barrier

}; // tools::cuda::MeshToSDF<BuildT>

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
template <typename BufferT>
GridHandle<BufferT>
MeshToSDF<BuildT>::computeDerivedTopology(const GridT* d_srcGrid, const float* d_udf,
                                          float voxelSize, const BufferT& buffer)
{
    using PruneOp = sdf_detail::UDFBarrierPruneMaskFunctor<BuildT>;

    // Barrier threshold √3/2 voxels expressed in the sidecar's WORLD units, squared.
    const float    barrierSqWorld = 0.75f * voxelSize * voxelSize;
    const uint32_t srcLeafCount =
        util::cuda::DeviceGridTraits<BuildT>::getTreeData(d_srcGrid).mNodeCount[0];

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
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
void MeshToSDF<BuildT>::signNonBarrier(const GridT* d_grid, ConnectedComponents<BuildT>& cc)
{
    const uint32_t leafCount =
        util::cuda::DeviceGridTraits<BuildT>::getTreeData(d_grid).mNodeCount[0];
    mExteriorRep = 0;
    if (leafCount == 0) return;  // every materialized leaf has >=1 component, so leafCount==0 => K==0

    const uint64_t activeCount =
        util::cuda::DeviceGridTraits<BuildT>::getActiveVoxelCount(d_grid);

    // (1) Exterior representative = parent[ component of the global min-x active voxel ].
    if (mVerbose==1) mTimer.start("Sign: find exterior component");
    auto minKeyBuf = nanovdb::cuda::DeviceBuffer::create(sizeof(unsigned long long), nullptr, false);
    auto* d_minKey = static_cast<unsigned long long*>(minKeyBuf.deviceData());
    const unsigned long long initKey = ~0ull;
    cudaCheck(cudaMemcpyAsync(d_minKey, &initKey, sizeof(initKey), cudaMemcpyHostToDevice, mStream));
    using FindOp = sdf_detail::FindExteriorRepFunctor<BuildT>;
    util::cuda::operatorKernel<FindOp><<<leafCount, FindOp::MaxThreadsPerBlock, 0, mStream>>>(
        d_grid, cc.deviceLeafComponentMasks(), cc.deviceLeafComponentOffsets(),
        cc.deviceComponentParent(), d_minKey);
    cudaCheckError();
    unsigned long long minKey = 0;
    cudaCheck(cudaMemcpyAsync(&minKey, d_minKey, sizeof(minKey), cudaMemcpyDeviceToHost, mStream));
    cudaCheck(cudaStreamSynchronize(mStream));
    mExteriorRep = uint64_t(uint32_t(minKey & 0xFFFFFFFFull));
    if (mVerbose==1) mTimer.stop();

    // (2) Per-voxel sign: +1 exterior / -1 interior (slot 0 = background +1).
    mVoxelSign = nanovdb::cuda::DeviceBuffer::create((activeCount + 1) * sizeof(int8_t), nullptr, false);
    cudaCheck(cudaMemsetAsync(mVoxelSign.deviceData(), 1, (activeCount + 1) * sizeof(int8_t), mStream));// all +1
    using SignOp = sdf_detail::SignNonBarrierFunctor<BuildT>;
    if (mVerbose==1) mTimer.start("Sign: write per-voxel signs");
    util::cuda::operatorKernel<SignOp><<<leafCount, SignOp::MaxThreadsPerBlock, 0, mStream>>>(
        d_grid, cc.deviceLeafComponentMasks(), cc.deviceLeafComponentOffsets(),
        cc.deviceComponentParent(), uint32_t(mExteriorRep), deviceVoxelSign());
    cudaCheckError();
    if (mVerbose==1) mTimer.stop();
}// MeshToSDF<BuildT>::signNonBarrier

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
void MeshToSDF<BuildT>::injectSignsToOriginal(const GridT* d_origGrid, const GridT* d_derivedGrid)
{
    const uint64_t origActive =
        util::cuda::DeviceGridTraits<BuildT>::getActiveVoxelCount(d_origGrid);
    const uint32_t derivedLeafCount =
        util::cuda::DeviceGridTraits<BuildT>::getTreeData(d_derivedGrid).mNodeCount[0];

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
}// MeshToSDF<BuildT>::injectSignsToOriginal

} // namespace tools::cuda

} // namespace nanovdb

#endif // NVIDIA_TOOLS_CUDA_MESHTOSDF_CUH_HAS_BEEN_INCLUDED
