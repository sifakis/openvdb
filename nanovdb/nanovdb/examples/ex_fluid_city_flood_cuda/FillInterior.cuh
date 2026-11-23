// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

// FillInterior.cuh: the grid of a narrow-band level set with its interior filled in.
//
// Input: an index grid (the band) and the invert masks of PropagateSign, which mark every inactive element that
// lies inside the surface: single voxels (leaf masks), 8^3 regions (lower masks) and 128^3 regions (upper masks).
// Output: an index grid whose active voxels are the band's active voxels plus every voxel of a marked element.
//
// Counterpart: openvdb::tools::sdfInteriorMask (tools/LevelSetUtil.h) united with the level set's own active voxels.
// Built on tools::cuda::TopologyBuilder, following the stages of tools::cuda::PruneGrid.
#pragma once
#include <nanovdb/NanoVDB.h>
#include <nanovdb/GridHandle.h>
#include <nanovdb/HostBuffer.h>
#include <nanovdb/cuda/Buffer.h>
#include <nanovdb/tools/cuda/TopologyBuilder.cuh>
#include <nanovdb/util/cuda/DeviceGridTraits.cuh>
#include <nanovdb/util/cuda/Injection.cuh>
#include <nanovdb/util/cuda/Util.h>
#include <cstddef>
#include <stdexcept>

namespace fill {

using nanovdb::Mask;

// One thread per lower node of the source: its output node keeps the source's leaves and gains a full leaf for
// every 8^3 region marked inside.
template <typename BuildT>
struct FillLowerNodesFunctor
{
    __device__ void operator()(size_t srcLowerID, const nanovdb::NanoGrid<BuildT>* srcGrid, const Mask<4>* invertLowerMasks,
                               const nanovdb::NanoRoot<BuildT>* filledRoot, Mask<5>* upperMasks,
                               Mask<4> (*lowerMasks)[Mask<5>::SIZE]) const
    {
        const auto&    lower     = srcGrid->tree().template getFirstNode<1>()[srcLowerID];
        const uint64_t tileIndex = nanovdb::util::PtrDiff(filledRoot->probeTile(lower.origin()), filledRoot->tile(0)) /
                                   sizeof(typename nanovdb::NanoRoot<BuildT>::Tile);
        const uint32_t upperChildIndex = nanovdb::NanoUpper<BuildT>::CoordToOffset(lower.origin());
        upperMasks[tileIndex].setOnAtomic(upperChildIndex);
        uint64_t* out = const_cast<uint64_t*>(lowerMasks[tileIndex][upperChildIndex].words());
        for (uint32_t w = 0; w < Mask<4>::WORD_COUNT; ++w)
            out[w] = lower.childMask().words()[w] | invertLowerMasks[srcLowerID].words()[w];
    }
};

// One thread per 64-bit word of an upper node's invert mask: every 128^3 region marked inside becomes a lower node
// with all of its leaves.
template <typename BuildT>
struct FillUpperNodesFunctor
{
    __device__ void operator()(size_t tid, const nanovdb::NanoGrid<BuildT>* srcGrid, const Mask<5>* invertUpperMasks,
                               const nanovdb::NanoRoot<BuildT>* filledRoot, Mask<5>* upperMasks,
                               Mask<4> (*lowerMasks)[Mask<5>::SIZE]) const
    {
        const size_t   srcUpperID = tid / Mask<5>::WORD_COUNT;
        const uint32_t w          = uint32_t(tid % Mask<5>::WORD_COUNT);
        const uint64_t word       = invertUpperMasks[srcUpperID].words()[w];
        if (!word) return;
        const auto&    upper     = srcGrid->tree().template getFirstNode<2>()[srcUpperID];
        const uint64_t tileIndex = nanovdb::util::PtrDiff(filledRoot->probeTile(upper.origin()), filledRoot->tile(0)) /
                                   sizeof(typename nanovdb::NanoRoot<BuildT>::Tile);
        nanovdb::util::atomicOr(const_cast<uint64_t*>(&upperMasks[tileIndex].words()[w]), word);
        for (uint32_t bit = 0; bit < 64; ++bit)
            if (word & (uint64_t(1) << bit)) lowerMasks[tileIndex][w * 64 + bit].setOn();
    }
};

// One thread per leaf of the output: a leaf of the source keeps its active voxels and gains the voxels marked
// inside; a leaf the source does not have comes from a region marked inside, so all of its voxels are active.
template <typename BuildT>
struct FillLeafMasksFunctor
{
    __device__ void operator()(size_t dstLeafID, const nanovdb::NanoGrid<BuildT>* srcGrid, const Mask<3>* invertLeafMasks,
                               const nanovdb::tools::cuda::TopologyBuilderData<BuildT>* dstData) const
    {
        auto&       dstLeaf = dstData->getLeaf(int(dstLeafID));
        auto&       mask    = const_cast<Mask<3>&>(dstLeaf.valueMask());
        const auto* srcLeaf = srcGrid->tree().root().probeLeaf(dstLeaf.origin());
        if (!srcLeaf) { mask.setOn(); return; }
        const Mask<3>& invert = invertLeafMasks[srcLeaf - srcGrid->tree().template getFirstNode<0>()];
        uint64_t*      out    = const_cast<uint64_t*>(mask.words());
        for (uint32_t w = 0; w < Mask<3>::WORD_COUNT; ++w) out[w] = srcLeaf->valueMask().words()[w] | invert.words()[w];
    }
};

template <typename BuildT, typename ResourceT = nanovdb::cuda::DeviceResource>
class FillInterior
{
    using GridT = nanovdb::NanoGrid<BuildT>;
    using RootT = nanovdb::NanoRoot<BuildT>;

public:
    using BufferT = nanovdb::cuda::Buffer<std::byte, nanovdb::cuda::ResourceRef<ResourceT>>;

    /// @param d_srcGrid device grid of the band
    /// @param d_leafMasks, d_lowerMasks, d_upperMasks the invert masks PropagateSign filled for @a d_srcGrid
    /// @param rootOriginCount number of root-level origins PropagateSign returned; must be 0
    FillInterior(const GridT* d_srcGrid, const Mask<3>* d_leafMasks, const Mask<4>* d_lowerMasks, const Mask<5>* d_upperMasks,
                 uint64_t rootOriginCount, cudaStream_t stream = 0,
                 ResourceT& resource = nanovdb::cuda::default_resource<ResourceT>())
        : mBuilder(stream, resource), mStream(stream), mSrcGrid(d_srcGrid), mLeafMasks(d_leafMasks)
        , mLowerMasks(d_lowerMasks), mUpperMasks(d_upperMasks), mRootOriginCount(rootOriginCount) {}

    /// @brief Creates the filled grid.
    /// @throw std::runtime_error if the interior reaches a 4096^3 region without a root-table entry: filling it
    ///        would take 6.9e10 voxels.
    nanovdb::GridHandle<BufferT> build()
    {
        if (mRootOriginCount) throw std::runtime_error("FillInterior: interior regions at the root level are not supported");
        mSrcTreeData = nanovdb::util::cuda::DeviceGridTraits<BuildT>::getTreeData(mSrcGrid, mStream);
        if (mSrcTreeData.mTileCount[2] || mSrcTreeData.mTileCount[1] || mSrcTreeData.mTileCount[0])
            throw std::runtime_error("Topological operations not supported on grids with value tiles");

        this->fillRoot();
        mBuilder.allocateInternalMaskBuffers(mStream);
        this->fillInternalNodes();
        mBuilder.countNodes(mStream);
        auto buffer = mBuilder.getBuffer(BufferT(mStream, mBuilder.ref(), 0, nanovdb::cuda::noInit), mStream);
        this->processGridTreeRoot();
        mBuilder.processUpperNodes(mStream);
        mBuilder.processLowerNodes(mStream);
        this->fillLeafNodes();
        mBuilder.processBBox(mStream);
        mBuilder.postProcessGridTree(mStream);
        cudaCheck(cudaStreamSynchronize(mStream));
        return nanovdb::GridHandle<BufferT>(std::move(buffer));
    }

private:
    static constexpr unsigned int mNumThreads = 128;
    static unsigned int numBlocks(size_t n) { return unsigned((n + mNumThreads - 1) / mNumThreads); }

    // The output has the root tiles of the source: an interior region is always inside an existing upper node.
    void fillRoot()
    {
        nanovdb::tools::cuda::topology::detail::ProcessedTileMap<RootT> tiles;
        if (mSrcTreeData.mVoxelCount) {
            auto deviceSrcRoot = static_cast<const RootT*>(nanovdb::util::PtrAdd(mSrcGrid, GridT::memUsage() + mSrcTreeData.mNodeOffset[3]));
            const uint64_t rootSize = mSrcTreeData.mNodeOffset[2] - mSrcTreeData.mNodeOffset[3];
            auto srcRootBuffer = nanovdb::HostBuffer::create(rootSize);
            cudaCheck(cudaMemcpyAsync(srcRootBuffer.data(), deviceSrcRoot, rootSize, cudaMemcpyDeviceToHost, mStream));
            cudaCheck(cudaStreamSynchronize(mStream));
            auto srcRoot = static_cast<RootT*>(srcRootBuffer.data());
            for (uint32_t t = 0; t < srcRoot->tileCount(); t++)
                nanovdb::tools::cuda::topology::detail::insertProcessedTile<RootT>(tiles, srcRoot->tile(t)->origin());
        }
        auto rootPtr = mBuilder.allocateProcessedRootTiles(static_cast<uint32_t>(tiles.size()));
        nanovdb::tools::cuda::topology::detail::packProcessedRoot(tiles, rootPtr);
        mBuilder.uploadProcessedRoot(mStream);
    }

    void fillInternalNodes()
    {
        using nanovdb::util::cuda::lambdaKernel;
        if (const size_t lowerCount = mSrcTreeData.mNodeCount[1]) {
            lambdaKernel<<<numBlocks(lowerCount), mNumThreads, 0, mStream>>>(lowerCount, FillLowerNodesFunctor<BuildT>(),
                mSrcGrid, mLowerMasks, mBuilder.deviceProcessedRoot(), mBuilder.deviceUpperMasks(), mBuilder.deviceLowerMasks());
            cudaCheckError();
        }
        if (const size_t words = size_t(mSrcTreeData.mNodeCount[2]) * Mask<5>::WORD_COUNT) {
            lambdaKernel<<<numBlocks(words), mNumThreads, 0, mStream>>>(words, FillUpperNodesFunctor<BuildT>(),
                mSrcGrid, mUpperMasks, mBuilder.deviceProcessedRoot(), mBuilder.deviceUpperMasks(), mBuilder.deviceLowerMasks());
            cudaCheckError();
        }
    }

    void processGridTreeRoot()
    {
        cudaCheck(cudaMemcpyAsync(&mBuilder.data()->getGrid(), mSrcGrid->data(), GridT::memUsage(), cudaMemcpyDeviceToDevice, mStream));
        nanovdb::util::cuda::lambdaKernel<<<1, 1, 0, mStream>>>(1, nanovdb::tools::cuda::topology::detail::BuildGridTreeRootFunctor<BuildT>(),
                                                               mBuilder.deviceData());
        cudaCheckError();
    }

    void fillLeafNodes()
    {
        if (const size_t dstLeafCount = mBuilder.data()->nodeCount[0]) {
            nanovdb::util::cuda::lambdaKernel<<<numBlocks(dstLeafCount), mNumThreads, 0, mStream>>>(
                dstLeafCount, FillLeafMasksFunctor<BuildT>(), mSrcGrid, mLeafMasks, mBuilder.deviceData());
            cudaCheckError();
        }
        mBuilder.processLeafOffsets(mStream);
    }

    nanovdb::tools::cuda::TopologyBuilder<BuildT, ResourceT> mBuilder;
    cudaStream_t      mStream{0};
    const GridT*      mSrcGrid;
    const Mask<3>*    mLeafMasks;
    const Mask<4>*    mLowerMasks;
    const Mask<5>*    mUpperMasks;
    uint64_t          mRootOriginCount;
    nanovdb::TreeData mSrcTreeData;
};

/// @brief The level set's values on the filled grid: the band's own values, and the negated background on the
///        voxels that were filled in. Slot 0 keeps the (positive) background.
template <typename BuildT, typename ValueT, typename ResourceT = nanovdb::cuda::DeviceResource>
nanovdb::cuda::Buffer<ValueT, nanovdb::cuda::ResourceRef<ResourceT>>
fillValues(const nanovdb::NanoGrid<BuildT>* srcGrid, const ValueT* srcPhi, const nanovdb::NanoGrid<BuildT>* dstGrid,
           ValueT background, cudaStream_t stream = 0, ResourceT& resource = nanovdb::cuda::default_resource<ResourceT>())
{
    using nanovdb::util::cuda::DeviceGridTraits;
    const uint32_t srcLeafCount = DeviceGridTraits<BuildT>::getTreeData(srcGrid, stream).mNodeCount[0];
    const uint64_t dstCount     = DeviceGridTraits<BuildT>::getTreeData(dstGrid, stream).mVoxelCount + 1;
    nanovdb::cuda::Buffer<ValueT, nanovdb::cuda::ResourceRef<ResourceT>> dstPhi(
        stream, nanovdb::cuda::ResourceRef<ResourceT>(resource), dstCount, nanovdb::cuda::noInit);
    ValueT* out = dstPhi.data();
    nanovdb::util::cuda::lambdaKernel<<<nanovdb::util::cuda::blocksPerGrid(dstCount, 256), 256, 0, stream>>>(
        dstCount, [=] __device__(size_t i) { out[i] = i ? -background : background; });
    cudaCheckError();
    if (srcLeafCount) {
        using InjectOp = nanovdb::util::cuda::InjectGridDataFunctor<BuildT, ValueT>;
        nanovdb::util::cuda::operatorKernel<InjectOp><<<srcLeafCount, InjectOp::MaxThreadsPerBlock, 0, stream>>>(
            srcGrid, dstGrid, srcPhi, out);
        cudaCheckError();
    }
    return dstPhi;
}

} // namespace fill
