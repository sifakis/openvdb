// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/tools/cuda/CSG.cuh

    \author JaeHyun Lee

    \brief CSG operations on narrow-band level sets, each stored as a NanoVDB indexGrid plus a value sidecar and the
           invert masks of PropagateSign.

    \note DRAFT, after nanovdb/examples/ex_csg_nanovdb_cuda/CSG-design.md:
          - csgUnion, csgIntersection, csgDifference: route B. One topology build, and the invert masks of the
            result derived from those of the inputs at every tree level. The inputs may have a negative background.
          - csgComplement: negates a level set in place.
          - csgUnionPropagateSign: route A, step 5(a). MergeGrids, PruneGrid, and PropagateSign rerun on the result.
          - csgUnionLeafResolved: route A, step 5(c). The invert masks resolved to the leaf level.
          The last two are comparison baselines for csgUnion on positive backgrounds, and are to be removed once it
          is settled.

    \warning The header file contains cuda device code so be sure
             to only include it in .cu files (or other .cuh files)
*/

#ifndef NANOVDB_TOOLS_CUDA_CSG_CUH_HAS_BEEN_INCLUDED
#define NANOVDB_TOOLS_CUDA_CSG_CUH_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h>
#include <nanovdb/GridHandle.h>
#include <nanovdb/cuda/Buffer.h>
#include <nanovdb/cuda/DeviceResource.h>
#include <nanovdb/tools/cuda/LevelSetTracker.cuh> // for LevelSetTracker::State and levelset::detail::isInverted
#include <nanovdb/tools/cuda/MergeGrids.cuh>
#include <nanovdb/tools/cuda/PropagateSign.cuh>
#include <nanovdb/tools/cuda/PruneGrid.cuh>
#include <nanovdb/tools/cuda/TopologyBuilder.cuh>
#include <nanovdb/util/cuda/DeviceGridTraits.cuh>
#include <nanovdb/util/cuda/Morphology.cuh> // for PruneInternalNodesFunctor
#include <nanovdb/util/cuda/Util.h> // for operatorKernel

#include <cstddef>   // std::byte
#include <stdexcept> // std::runtime_error
#include <utility>   // std::move

namespace nanovdb {

namespace tools::cuda {

namespace csg::detail {

template<typename T, typename ResourceT>
using BufferT = nanovdb::cuda::Buffer<T, nanovdb::cuda::ResourceRef<ResourceT>>;

/// @brief Device pointers of one input level set: its grid, the values of its active voxels, and its invert masks.
template<typename BuildT, typename ValueT>
struct LevelSetView
{
    const NanoGrid<BuildT>* grid;
    const ValueT*           phi;
    const nanovdb::Mask<3>* leafMasks;
    const nanovdb::Mask<4>* lowerMasks;
    const nanovdb::Mask<5>* upperMasks;
    const nanovdb::Coord*   rootOrigins;
    uint64_t                rootOriginCount;

    /// @brief True if @a ijk is inactive in this level set and takes the negative background.
    __hostdev__ bool isInverted(const nanovdb::Coord& ijk) const
    {
        return levelset::detail::isInverted(*grid, ijk, leafMasks, lowerMasks, upperMasks, rootOrigins, rootOriginCount);
    }
}; // LevelSetView

// One thread per voxel of a leaf of the merged grid: turn on the bit of every active voxel that stays active in
// the union, i.e. that neither input holds as an inactive voxel with the negative background.
template<typename BuildT, typename ValueT>
struct PointQueryRetainMaskFunctor
{
    static constexpr int MaxThreadsPerBlock = NanoLeaf<BuildT>::SIZE;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(const NanoGrid<BuildT>* mergedGrid,
                               LevelSetView<BuildT, ValueT> input1,
                               LevelSetView<BuildT, ValueT> input2,
                               nanovdb::Mask<3>* retainMasks) const
    {
        const auto&    leaf = mergedGrid->tree().template getFirstNode<0>()[blockIdx.x];
        const uint32_t n    = threadIdx.x;
        auto&          mask = retainMasks[blockIdx.x];
        if (n < nanovdb::Mask<3>::WORD_COUNT) mask.words()[n] = 0;
        __syncthreads();
        if (!leaf.isActive(n)) return;

        const Coord ijk = leaf.offsetToGlobalCoord(n);
        if (!input1.isInverted(ijk) && !input2.isInverted(ijk)) mask.setOnAtomic(n);
    }
}; // PointQueryRetainMaskFunctor

// One thread per voxel of a leaf of the union grid: an active voxel takes the minimum of the values of the inputs
// in which it is active. An input in which it is inactive holds the positive background there (the retain masks
// removed the other case), which is above every active value.
template<typename BuildT, typename ValueT>
struct UnionValuesFunctor
{
    static constexpr int MaxThreadsPerBlock = NanoLeaf<BuildT>::SIZE;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(const NanoGrid<BuildT>* unionGrid,
                               LevelSetView<BuildT, ValueT> input1,
                               LevelSetView<BuildT, ValueT> input2,
                               ValueT background,
                               ValueT* unionPhi) const
    {
        const auto&    leaf = unionGrid->tree().template getFirstNode<0>()[blockIdx.x];
        const uint32_t n    = threadIdx.x;
        if (!leaf.isActive(n)) return;

        const Coord ijk = leaf.offsetToGlobalCoord(n);
        ValueT value = background;
        const auto* leaf1 = input1.grid->tree().root().probeLeaf(ijk);
        if (leaf1 && leaf1->isActive(n)) value = math::Min(value, input1.phi[leaf1->getValue(n)]);
        const auto* leaf2 = input2.grid->tree().root().probeLeaf(ijk);
        if (leaf2 && leaf2->isActive(n)) value = math::Min(value, input2.phi[leaf2->getValue(n)]);
        unionPhi[leaf.getValue(n)] = value;
    }
}; // UnionValuesFunctor

// One thread per leaf of the merged grid: retain the active voxels that are not active in the interior grid, word
// by word: C_out = (C1 | C2) & ~IN_out, with the merged leaf holding C1 | C2 and the interior leaf IN_out.
template<typename BuildT>
struct LeafResolvedRetainMaskFunctor
{
    __device__ void operator()(size_t leafID, const NanoGrid<BuildT>* mergedGrid, const NanoGrid<BuildT>* interiorGrid,
                               nanovdb::Mask<3>* retainMasks) const
    {
        const auto& leaf         = mergedGrid->tree().template getFirstNode<0>()[leafID];
        const auto* interiorLeaf = interiorGrid->tree().root().probeLeaf(leaf.origin());
        for (uint32_t w = 0; w < nanovdb::Mask<3>::WORD_COUNT; ++w)
            retainMasks[leafID].words()[w] = leaf.valueMask().words()[w] &
                                             ~(interiorLeaf ? interiorLeaf->valueMask().words()[w] : uint64_t(0));
    }
}; // LeafResolvedRetainMaskFunctor

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// csgUnion, csgIntersection and csgDifference share the kernels below. Every operation is the intersection of its
// two operands, each read as is or complemented, with the result stored as is or complemented:
//     intersection(1, 2) = 1 & 2        difference(1, 2) = 1 & ~2        union(1, 2) = ~(~1 & ~2)
// A voxel of a leaf, or a slot of a lower or upper node, is in one of three states in an operand:
//     stored   an active voxel, or a slot with a child node
//     inside   inactive, where the operand is negative
//     outside  inactive, where the operand is positive
// In the intersection a point is inside if it is inside both operands, outside if it is outside either, and stored
// otherwise.

/// @brief One operand of the intersection: a level set and how to read it.
template<typename BuildT, typename ValueT>
struct Operand : LevelSetView<BuildT, ValueT>
{
    ValueT sign;             // +1 to read the level set as is, -1 to read its complement
    bool   insideAtInfinity; // the operand, as read, is negative at infinity: its invert bits then mark the outside
};

/// @brief What a level set holds at the node of a given level (0 leaf, 1 lower, 2 upper) with a given origin:
///        that node (exists, with its index among the nodes of its level), or no node, in which case the node's
///        whole extent lies in one inactive element of a coarser level and inverted is that element's invert bit.
struct Cover
{
    bool     exists;
    uint32_t index;
    bool     inverted;
};

/// @brief Descend @a levelSet toward the node of @a level at @a origin (see Cover): one tree descent per node.
template<typename BuildT, typename ValueT>
__hostdev__ inline Cover cover(const LevelSetView<BuildT, ValueT>& levelSet, const nanovdb::Coord& origin, int level)
{
    using UpperT = NanoUpper<BuildT>;
    using LowerT = NanoLower<BuildT>;
    const auto& tree = levelSet.grid->tree();

    const auto* tile = tree.root().probeTile(origin);
    if (!tile) return {false, 0, ::cuda::std::binary_search(levelSet.rootOrigins, levelSet.rootOrigins + levelSet.rootOriginCount,
                                                            origin & ~int(UpperT::DIM - 1))};
    NANOVDB_ASSERT(tile->isChild()); // PropagateSign rejects root value tiles
    const auto* upper = tree.root().getChild(tile);
    if (level == 2) return {true, uint32_t(upper - tree.getFirstUpper()), false};

    const uint32_t u = UpperT::CoordToOffset(origin);
    if (!upper->isChild(u)) return {false, 0, levelSet.upperMasks[upper - tree.getFirstUpper()].isOn(u)};
    const auto* lower = upper->getChild(u);
    if (level == 1) return {true, uint32_t(lower - tree.getFirstLower()), false};

    const uint32_t l = LowerT::CoordToOffset(origin);
    if (!lower->isChild(l)) return {false, 0, levelSet.lowerMasks[lower - tree.getFirstLower()].isOn(l)};
    return {true, uint32_t(lower->getChild(l) - tree.getFirstLeaf()), false};
} // cover

/// @brief The mask of what a node of @a LEVEL stores: the value mask of a leaf, the child mask of a lower or upper
///        node.
template<int LEVEL, typename NodeT>
__hostdev__ inline const nanovdb::Mask<LEVEL + 3>& storedMask(const NodeT& node)
{
    if constexpr (LEVEL == 0) return node.valueMask();
    else return node.childMask();
}

/// @brief The invert mask of node @a index of @a LEVEL of @a levelSet.
template<int LEVEL, typename BuildT, typename ValueT>
__hostdev__ inline const nanovdb::Mask<LEVEL + 3>& invertMask(const LevelSetView<BuildT, ValueT>& levelSet, uint32_t index)
{
    if constexpr (LEVEL == 0) return levelSet.leafMasks[index];
    else if constexpr (LEVEL == 1) return levelSet.lowerMasks[index];
    else return levelSet.upperMasks[index];
}

/// @brief One mask word of the three states of an operand over the extent of a node.
struct States
{
    uint64_t stored, inside, outside;
};

/// @brief Word @a w of the states of @a operand over the node of @a LEVEL that @a c was computed for (see cover).
template<int LEVEL, typename BuildT, typename ValueT>
__hostdev__ inline States states(const Operand<BuildT, ValueT>& operand, const Cover& c, uint32_t w)
{
    uint64_t stored = 0, inverted = c.inverted ? ~uint64_t(0) : uint64_t(0);
    if (c.exists) {
        stored   = storedMask<LEVEL>(operand.grid->tree().template getFirstNode<LEVEL>()[c.index]).words()[w];
        inverted = invertMask<LEVEL>(operand, c.index).words()[w];
    }
    const uint64_t plain = ~(stored | inverted); // inactive, on the side that the operand is on at infinity
    return operand.insideAtInfinity ? States{stored, plain, inverted} : States{stored, inverted, plain};
}

// One thread per leaf of an operand: retain its active voxels that are not outside the other operand.
template<typename BuildT, typename ValueT>
struct RetainMaskFunctor
{
    __device__ void operator()(size_t leafID, Operand<BuildT, ValueT> self, Operand<BuildT, ValueT> other,
                               nanovdb::Mask<3>* retainMasks) const
    {
        const auto& leaf = self.grid->tree().template getFirstNode<0>()[leafID];
        const Cover c = cover(other, leaf.origin(), 0);
        for (uint32_t w = 0; w < nanovdb::Mask<3>::WORD_COUNT; ++w)
            retainMasks[leafID].words()[w] = leaf.valueMask().words()[w] & ~states<0>(other, c, w).outside;
    }
}; // RetainMaskFunctor

// One thread per leaf of an operand: OR its retained active voxels into the leaf of the result grid at the same
// origin, if the result has one. The counterpart of PruneLeafMasksFunctor for several inputs.
template<typename BuildT>
struct MergePrunedLeafMasksFunctor
{
    __device__ void operator()(size_t leafID, const NanoGrid<BuildT>* srcGrid, NanoGrid<BuildT>* dstGrid,
                               const nanovdb::Mask<3>* retainMasks) const
    {
        const auto& srcLeaf = srcGrid->tree().template getFirstNode<0>()[leafID];
        if (const auto* dstLeaf = dstGrid->tree().root().probeLeaf(srcLeaf.origin())) {
            auto& dstMask = const_cast<nanovdb::Mask<3>&>(dstLeaf->valueMask());
            for (uint32_t w = 0; w < nanovdb::Mask<3>::WORD_COUNT; ++w)
                dstMask.words()[w] |= srcLeaf.valueMask().words()[w] & retainMasks[leafID].words()[w];
        }
    }
}; // MergePrunedLeafMasksFunctor

// One thread per voxel of a leaf of the result: an active voxel takes the maximum of the two operands. An operand
// in which it is inactive is inside there (the retain masks removed the other case) and holds -magnitude, the
// negative background. @a resultSign is -1 to store the complement of the intersection.
template<typename BuildT, typename ValueT>
struct IntersectionValuesFunctor
{
    static constexpr int MaxThreadsPerBlock = NanoLeaf<BuildT>::SIZE;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(const NanoGrid<BuildT>* resultGrid,
                               Operand<BuildT, ValueT> input1,
                               Operand<BuildT, ValueT> input2,
                               ValueT magnitude,
                               ValueT resultSign,
                               ValueT* resultPhi) const
    {
        const auto&    leaf = resultGrid->tree().template getFirstNode<0>()[blockIdx.x];
        const uint32_t n    = threadIdx.x;
        if (!leaf.isActive(n)) return;

        const Coord ijk = leaf.offsetToGlobalCoord(n);
        ValueT value = -magnitude;
        const auto* leaf1 = input1.grid->tree().root().probeLeaf(ijk);
        if (leaf1 && leaf1->isActive(n)) value = math::Max(value, input1.sign * input1.phi[leaf1->getValue(n)]);
        const auto* leaf2 = input2.grid->tree().root().probeLeaf(ijk);
        if (leaf2 && leaf2->isActive(n)) value = math::Max(value, input2.sign * input2.phi[leaf2->getValue(n)]);
        resultPhi[leaf.getValue(n)] = resultSign * value;
    }
}; // IntersectionValuesFunctor

// One thread per node of @a LEVEL of the result (0 leaf, 1 lower, 2 upper): its invert mask. A voxel or slot that
// the result does not store is inside the intersection if it is inside both operands, and outside it otherwise.
// The invert bits mark whichever of the two the intersection is not at infinity.
template<typename BuildT, typename ValueT, int LEVEL>
struct InvertMaskFunctor
{
    using MaskT = nanovdb::Mask<LEVEL + 3>;

    __device__ void operator()(size_t nodeID, const NanoGrid<BuildT>* resultGrid, Operand<BuildT, ValueT> input1,
                               Operand<BuildT, ValueT> input2, bool insideAtInfinity, MaskT* invertMasks) const
    {
        const auto& node = resultGrid->tree().template getFirstNode<LEVEL>()[nodeID];
        const Cover c1 = cover(input1, node.origin(), LEVEL), c2 = cover(input2, node.origin(), LEVEL);
        for (uint32_t w = 0; w < MaskT::WORD_COUNT; ++w) {
            const uint64_t stored  = storedMask<LEVEL>(node).words()[w];
            const uint64_t inside  = states<LEVEL>(input1, c1, w).inside & states<LEVEL>(input2, c2, w).inside;
            const uint64_t outside = ~(stored | inside);
            invertMasks[nodeID].words()[w] = insideAtInfinity ? outside : inside;
        }
    }
}; // InvertMaskFunctor

// One thread per sidecar value: negate it.
template<typename ValueT>
struct NegateFunctor
{
    __device__ void operator()(size_t i, ValueT* phi) const { phi[i] = -phi[i]; }
}; // NegateFunctor

/// @brief The grid whose active voxels are the retained active voxels of the two inputs: MergeGrids and PruneGrid
///        in one build, so that a node the result does not need is never created. A leaf of an input is stamped
///        into the densified child masks only if it retains a voxel (PruneInternalNodesFunctor), and
///        TopologyBuilder drops the tiles left without a child.
template<typename BuildT, typename ResourceT>
GridHandle<BufferT<std::byte, ResourceT>>
mergePrunedGrids(const NanoGrid<BuildT>* grid1, const nanovdb::Mask<3>* retainMasks1,
                 const NanoGrid<BuildT>* grid2, const nanovdb::Mask<3>* retainMasks2,
                 cudaStream_t stream, ResourceT& resource)
{
    using GridT = NanoGrid<BuildT>;
    using RootT = NanoRoot<BuildT>;
    static constexpr unsigned int threadsPerBlock = 128;
    const GridT*            grids[2]  = {grid1, grid2};
    const nanovdb::Mask<3>* retain[2] = {retainMasks1, retainMasks2};
    const typename GridT::TreeType::DataType treeData[2] = {
        util::cuda::DeviceGridTraits<BuildT>::getTreeData(grid1, stream),
        util::cuda::DeviceGridTraits<BuildT>::getTreeData(grid2, stream)};

    TopologyBuilder<BuildT, ResourceT> builder(stream, resource);

    // Root: the tiles of both inputs, as MergeGrids does; the tiles left empty are dropped by the builder.
    topology::detail::ProcessedTileMap<RootT> tiles;
    for (int i = 0; i < 2; ++i) {
        if (!treeData[i].mVoxelCount) continue;
        auto deviceSrcRoot = static_cast<const RootT*>(util::PtrAdd(grids[i], GridT::memUsage() + treeData[i].mNodeOffset[3]));
        const uint64_t rootSize = treeData[i].mNodeOffset[2] - treeData[i].mNodeOffset[3];
        auto srcRootBuffer = nanovdb::HostBuffer::create(rootSize);
        cudaCheck(cudaMemcpyAsync(srcRootBuffer.data(), deviceSrcRoot, rootSize, cudaMemcpyDeviceToHost, stream));
        auto srcRoot = static_cast<RootT*>(srcRootBuffer.data());
        for (uint32_t t = 0; t < srcRoot->tileCount(); t++)
            topology::detail::insertProcessedTile<RootT>(tiles, srcRoot->tile(t)->origin());
    }
    auto rootPtr = builder.allocateProcessedRootTiles(static_cast<uint32_t>(tiles.size()));
    topology::detail::packProcessedRoot(tiles, rootPtr);
    builder.uploadProcessedRoot(stream);

    // Internal nodes: each input stamps the leaves that retain a voxel.
    builder.allocateInternalMaskBuffers(stream);
    for (int i = 0; i < 2; ++i) {
        const uint32_t leafCount = treeData[i].mNodeCount[0];
        if (!leafCount) continue;
        util::cuda::lambdaKernel<<<util::cuda::blocksPerGrid(leafCount, threadsPerBlock), threadsPerBlock, 0, stream>>>(
            leafCount, util::morphology::cuda::PruneInternalNodesFunctor<BuildT>(), grids[i], builder.deviceProcessedRoot(),
            retain[i], builder.deviceUpperMasks(), builder.deviceLowerMasks());
        cudaCheckError();
    }
    builder.countNodes(stream);
    auto buffer = builder.getBuffer(BufferT<std::byte, ResourceT>(stream, builder.ref(), 0, nanovdb::cuda::noInit), stream);

    cudaCheck(cudaMemcpyAsync(&builder.data()->getGrid(), grid1->data(), GridT::memUsage(), cudaMemcpyDeviceToDevice, stream));
    util::cuda::lambdaKernel<<<1, 1, 0, stream>>>(1, topology::detail::BuildGridTreeRootFunctor<BuildT>(), builder.deviceData());
    cudaCheckError();
    builder.processUpperNodes(stream);
    builder.processLowerNodes(stream);

    // Leaves: each input ORs its retained voxels into the leaf at the same origin.
    for (int i = 0; i < 2; ++i) {
        const uint32_t leafCount = treeData[i].mNodeCount[0];
        if (!leafCount) continue;
        util::cuda::lambdaKernel<<<util::cuda::blocksPerGrid(leafCount, threadsPerBlock), threadsPerBlock, 0, stream>>>(
            leafCount, MergePrunedLeafMasksFunctor<BuildT>(), grids[i], &builder.data()->getGrid(), retain[i]);
        cudaCheckError();
    }
    builder.processLeafOffsets(stream);
    builder.processBBox(stream);
    builder.postProcessGridTree(stream);
    cudaCheck(cudaStreamSynchronize(stream));

    return GridHandle<BufferT<std::byte, ResourceT>>(std::move(buffer));
} // mergePrunedGrids

/// @brief The intersection of two operands, each read as is or complemented, stored as is or complemented: the
///        implementation of csgUnion, csgIntersection and csgDifference (see their documentation).
/// @details 1. Retain masks, per leaf of each operand: its active voxels that are not outside the other operand.
///          2. One topology build: only the nodes that hold a retained voxel exist.
///          3. Values: the maximum over the operands in which the voxel is active.
///          4. Invert masks, per node of the result, from the states of the operands over that node.
///          Where an operand has no node at the origin of a node being processed, its state there is read by one
///          tree descent per node, from the coarser level that holds it (see cover).
template<typename BuildT, typename ValueT, typename ResourceT>
typename LevelSetTracker<BuildT, ValueT, ResourceT>::State
intersect(const typename LevelSetTracker<BuildT, ValueT, ResourceT>::State& levelSet1, bool complement1,
          const typename LevelSetTracker<BuildT, ValueT, ResourceT>::State& levelSet2, bool complement2,
          bool complementResult, cudaStream_t stream, ResourceT& resource)
{
    using GridT    = NanoGrid<BuildT>;
    using StateT   = typename LevelSetTracker<BuildT, ValueT, ResourceT>::State;
    using OperandT = Operand<BuildT, ValueT>;
    using TraitsT  = util::cuda::DeviceGridTraits<BuildT>;
    const nanovdb::cuda::ResourceRef<ResourceT> ref(resource);
    static constexpr unsigned int threadsPerBlock = 128;

    if (levelSet1.phi.empty() || levelSet2.phi.empty())
        throw std::runtime_error("nanovdb::tools::cuda::csg: a level set has no sidecar");
    if (levelSet1.rootOrigins.size() || levelSet2.rootOrigins.size())
        throw std::runtime_error("nanovdb::tools::cuda::csg: root-level inverted regions are not supported yet");

    // The backgrounds: slot 0 of the sidecars.
    ValueT background1, background2;
    cudaCheck(cudaMemcpyAsync(&background1, levelSet1.phi.data(), sizeof(ValueT), cudaMemcpyDeviceToHost, stream));
    cudaCheck(cudaMemcpyAsync(&background2, levelSet2.phi.data(), sizeof(ValueT), cudaMemcpyDeviceToHost, stream));
    cudaCheck(cudaStreamSynchronize(stream));
    const ValueT magnitude = math::Abs(background1);
    if (magnitude == ValueT(0) || magnitude != math::Abs(background2))
        throw std::runtime_error("nanovdb::tools::cuda::csg: the backgrounds must be nonzero and equal in magnitude");

    // The operands as the intersection reads them, and the side the intersection is on at infinity: it is negative
    // there only if both operands are.
    const auto operand = [](const StateT& s, ValueT background, bool complement) {
        const ValueT sign = complement ? ValueT(-1) : ValueT(1);
        return OperandT{{s.grid.template deviceGrid<BuildT>(), s.phi.data(), s.leafMasks.data(), s.lowerMasks.data(),
                         s.upperMasks.data(), s.rootOrigins.data(), s.rootOrigins.size()},
                        sign, sign * background < ValueT(0)};
    };
    const OperandT input1 = operand(levelSet1, background1, complement1);
    const OperandT input2 = operand(levelSet2, background2, complement2);
    const bool   insideAtInfinity = input1.insideAtInfinity && input2.insideAtInfinity;
    const ValueT resultSign       = complementResult ? ValueT(-1) : ValueT(1);
    const ValueT resultBackground = resultSign * (insideAtInfinity ? -magnitude : magnitude);

    const uint32_t leafCount1 = TraitsT::getTreeData(input1.grid, stream).mNodeCount[0];
    const uint32_t leafCount2 = TraitsT::getTreeData(input2.grid, stream).mNodeCount[0];

    // 1. The active voxels of each operand that stay active in the result.
    BufferT<nanovdb::Mask<3>, ResourceT> retainMasks1(stream, ref, leafCount1, nanovdb::cuda::noInit);
    BufferT<nanovdb::Mask<3>, ResourceT> retainMasks2(stream, ref, leafCount2, nanovdb::cuda::noInit);
    using RetainOp = RetainMaskFunctor<BuildT, ValueT>;
    if (leafCount1) {
        util::cuda::lambdaKernel<<<util::cuda::blocksPerGrid(leafCount1, threadsPerBlock), threadsPerBlock, 0, stream>>>(
            leafCount1, RetainOp(), input1, input2, retainMasks1.data());
        cudaCheckError();
    }
    if (leafCount2) {
        util::cuda::lambdaKernel<<<util::cuda::blocksPerGrid(leafCount2, threadsPerBlock), threadsPerBlock, 0, stream>>>(
            leafCount2, RetainOp(), input2, input1, retainMasks2.data());
        cudaCheckError();
    }

    // 2. The topology of the result.
    auto resultHandle = mergePrunedGrids<BuildT, ResourceT>(input1.grid, retainMasks1.data(),
                                                           input2.grid, retainMasks2.data(), stream, resource);
    const GridT* resultGrid = resultHandle.template deviceGrid<BuildT>();
    const auto treeData = TraitsT::getTreeData(resultGrid, stream);
    const uint32_t leafCount = treeData.mNodeCount[0], lowerCount = treeData.mNodeCount[1], upperCount = treeData.mNodeCount[2];

    // 3. The values of its active voxels; slot 0 holds the background.
    BufferT<ValueT, ResourceT> resultPhi(stream, ref, treeData.mVoxelCount + 1, nanovdb::cuda::noInit);
    cudaCheck(cudaMemcpyAsync(resultPhi.data(), &resultBackground, sizeof(ValueT), cudaMemcpyHostToDevice, stream));
    if (leafCount) {
        using ValuesOp = IntersectionValuesFunctor<BuildT, ValueT>;
        util::cuda::operatorKernel<ValuesOp><<<leafCount, ValuesOp::MaxThreadsPerBlock, 0, stream>>>(
            resultGrid, input1, input2, magnitude, resultSign, resultPhi.data());
        cudaCheckError();
    }

    // 4. The invert masks of the result, level by level. Complementing the result leaves them as they are.
    BufferT<nanovdb::Mask<3>, ResourceT> leafMasks(stream, ref, leafCount, nanovdb::cuda::noInit);
    BufferT<nanovdb::Mask<4>, ResourceT> lowerMasks(stream, ref, lowerCount, nanovdb::cuda::noInit);
    BufferT<nanovdb::Mask<5>, ResourceT> upperMasks(stream, ref, upperCount, nanovdb::cuda::noInit);
    if (leafCount) {
        util::cuda::lambdaKernel<<<util::cuda::blocksPerGrid(leafCount, threadsPerBlock), threadsPerBlock, 0, stream>>>(
            leafCount, InvertMaskFunctor<BuildT, ValueT, 0>(), resultGrid, input1, input2, insideAtInfinity, leafMasks.data());
        cudaCheckError();
        util::cuda::lambdaKernel<<<util::cuda::blocksPerGrid(lowerCount, threadsPerBlock), threadsPerBlock, 0, stream>>>(
            lowerCount, InvertMaskFunctor<BuildT, ValueT, 1>(), resultGrid, input1, input2, insideAtInfinity, lowerMasks.data());
        cudaCheckError();
        util::cuda::lambdaKernel<<<util::cuda::blocksPerGrid(upperCount, threadsPerBlock), threadsPerBlock, 0, stream>>>(
            upperCount, InvertMaskFunctor<BuildT, ValueT, 2>(), resultGrid, input1, input2, insideAtInfinity, upperMasks.data());
        cudaCheckError();
    }
    cudaCheck(cudaStreamSynchronize(stream));

    // todo: the root origins of the result (section 8.3 of CSG-design.md); empty here.
    return StateT{std::move(resultHandle), std::move(resultPhi), std::move(leafMasks), std::move(lowerMasks),
                  std::move(upperMasks), decltype(levelSet1.rootOrigins)(stream, ref, 0, nanovdb::cuda::noInit)};
} // intersect

} // namespace csg::detail

/// @brief Union of two level sets: the level set whose value is the minimum of the two at every point of space.
/// @details Each level set is a LevelSetTracker::State: a ValueOnIndex grid, the sidecar of its active values and
///          the invert masks of PropagateSign. The result is built with the existing operators:
///          1. MergeGrids: the voxels active in either input.
///          2. Retain masks: of those, the voxels that neither input holds as inactive with the negative background.
///          3. PruneGrid: the topology of the union.
///          4. Values: the minimum over the inputs in which the voxel is active.
///          5. PropagateSign: the invert masks of the union.
/// @tparam BuildT Grid build type; must be ValueOnIndex.
/// @tparam ValueT Sidecar element type.
/// @tparam ResourceT Stream-ordered resource of the result, used for all scratch allocations too.
/// @param levelSet1, levelSet2 The inputs; not modified.
/// @param background Positive magnitude that both inputs assign to the voxels outside their bands (slot 0 of their
///        sidecars).
/// @param stream Stream all work is ordered on.
/// @param resource Allocation resource; must outlive the result.
/// @throw std::runtime_error if neither input has an active voxel, or if PropagateSign rejects the union.
/// @warning The inputs must share one index space (the same transform) and one positive background, and each must
///          take the positive background at infinity. The union must meet the preconditions of PropagateSign.
/// @note The stream is synchronized before returning.
template<typename BuildT, typename ValueT, typename ResourceT = nanovdb::cuda::DeviceResource>
typename LevelSetTracker<BuildT, ValueT, ResourceT>::State
csgUnionPropagateSign(const typename LevelSetTracker<BuildT, ValueT, ResourceT>::State& levelSet1,
         const typename LevelSetTracker<BuildT, ValueT, ResourceT>::State& levelSet2,
         ValueT background,
         cudaStream_t stream = 0,
         ResourceT& resource = nanovdb::cuda::default_resource<ResourceT>())
{
    using GridT   = NanoGrid<BuildT>;
    using StateT  = typename LevelSetTracker<BuildT, ValueT, ResourceT>::State;
    using ViewT   = csg::detail::LevelSetView<BuildT, ValueT>;
    using TraitsT = util::cuda::DeviceGridTraits<BuildT>;
    const nanovdb::cuda::ResourceRef<ResourceT> ref(resource);

    const auto view = [](const StateT& s) {
        return ViewT{s.grid.template deviceGrid<BuildT>(), s.phi.data(), s.leafMasks.data(), s.lowerMasks.data(),
                     s.upperMasks.data(), s.rootOrigins.data(), s.rootOrigins.size()};
    };
    const ViewT input1 = view(levelSet1), input2 = view(levelSet2);

    // 1. The voxels active in either input.
    MergeGrids<BuildT, ResourceT> merger(input1.grid, input2.grid, stream, resource);
    auto mergedHandle = merger.build();
    const GridT* mergedGrid = mergedHandle.template deviceGrid<BuildT>();
    const uint32_t mergedLeafCount = TraitsT::getTreeData(mergedGrid, stream).mNodeCount[0];
    if (!mergedLeafCount) throw std::runtime_error("nanovdb::tools::cuda::csgUnionPropagateSign: both level sets are empty");

    // 2. Of those, the voxels that stay active in the union.
    csg::detail::BufferT<nanovdb::Mask<3>, ResourceT> retainMasks(stream, ref, mergedLeafCount, nanovdb::cuda::noInit);
    using RetainOp = csg::detail::PointQueryRetainMaskFunctor<BuildT, ValueT>;
    util::cuda::operatorKernel<RetainOp><<<mergedLeafCount, RetainOp::MaxThreadsPerBlock, 0, stream>>>(
        mergedGrid, input1, input2, retainMasks.data());
    cudaCheckError();

    // 3. The topology of the union.
    PruneGrid<BuildT, ResourceT> pruner(mergedGrid, retainMasks.data(), stream, resource);
    auto unionHandle = pruner.build();
    const GridT* unionGrid = unionHandle.template deviceGrid<BuildT>();
    const auto treeData = TraitsT::getTreeData(unionGrid, stream);

    // 4. The values of its active voxels; slot 0 holds the background.
    csg::detail::BufferT<ValueT, ResourceT> unionPhi(stream, ref, treeData.mVoxelCount + 1, nanovdb::cuda::noInit);
    cudaCheck(cudaMemcpyAsync(unionPhi.data(), &background, sizeof(ValueT), cudaMemcpyHostToDevice, stream));
    if (treeData.mNodeCount[0]) {
        using ValuesOp = csg::detail::UnionValuesFunctor<BuildT, ValueT>;
        util::cuda::operatorKernel<ValuesOp><<<treeData.mNodeCount[0], ValuesOp::MaxThreadsPerBlock, 0, stream>>>(
            unionGrid, input1, input2, background, unionPhi.data());
        cudaCheckError();
    }

    // 5. The invert masks of the union.
    csg::detail::BufferT<nanovdb::Mask<3>, ResourceT> leafMasks(stream, ref, treeData.mNodeCount[0], nanovdb::cuda::noInit);
    csg::detail::BufferT<nanovdb::Mask<4>, ResourceT> lowerMasks(stream, ref, treeData.mNodeCount[1], nanovdb::cuda::noInit);
    csg::detail::BufferT<nanovdb::Mask<5>, ResourceT> upperMasks(stream, ref, treeData.mNodeCount[2], nanovdb::cuda::noInit);
    PropagateSign<BuildT, ValueT, ResourceT> propagator(unionGrid, unionPhi.data(), stream, resource);
    auto rootOrigins = propagator.propagate(leafMasks.data(), lowerMasks.data(), upperMasks.data()).first;

    return StateT{std::move(unionHandle), std::move(unionPhi), std::move(leafMasks), std::move(lowerMasks),
                  std::move(upperMasks), std::move(rootOrigins)};
}// csgUnionPropagateSign

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

/// @brief The result of csgUnionLeafResolved.
template<typename BuildT, typename ValueT, typename ResourceT = nanovdb::cuda::DeviceResource>
struct LeafResolvedLevelSet
{
    using HandleT = GridHandle<csg::detail::BufferT<std::byte, ResourceT>>;
    HandleT                                 grid;     // the narrow band
    csg::detail::BufferT<ValueT, ResourceT> phi;      // slot 0 = background; slot k = value of the active voxel with index k
    HandleT                                 interior; // active voxels = the inactive voxels of grid that take the negative background
};

/// @brief Union of two level sets whose invert masks are resolved to the leaf level (DRAFT, comparison only: section
///        7.3 and step 5(c) of route A in CSG-design.md).
/// @details A level set is given as its grid, its sidecar, and an interior grid: a grid whose active voxels are
///          exactly the inactive voxels of the level set that take the negative background, so that its leaf value
///          masks are the invert masks resolved to the leaf level. The interior grid costs memory in proportion to
///          the enclosed volume. The union is then mask arithmetic, with no query of the inputs' sign:
///          1. MergeGrids on the interior grids: IN_out = IN1 | IN2.
///          2. MergeGrids on the grids: C1 | C2.
///          3. Retain masks, per leaf: C_out = (C1 | C2) & ~IN_out.
///          4. PruneGrid: the topology of the union.
///          5. Values: the minimum over the inputs in which the voxel is active.
/// @param grid1, phi1, interior1 First input, on the device.
/// @param grid2, phi2, interior2 Second input, on the device.
/// @param background Positive magnitude that both inputs assign to the voxels outside their bands.
/// @param stream Stream all work is ordered on.
/// @param resource Allocation resource; must outlive the result.
/// @throw std::runtime_error if neither input has an active voxel.
/// @warning The inputs must share one index space (the same transform) and one positive background, and each must
///          take the positive background at infinity.
/// @note The stream is synchronized before returning.
template<typename BuildT, typename ValueT, typename ResourceT = nanovdb::cuda::DeviceResource>
LeafResolvedLevelSet<BuildT, ValueT, ResourceT>
csgUnionLeafResolved(const NanoGrid<BuildT>* grid1, const ValueT* phi1, const NanoGrid<BuildT>* interior1,
                     const NanoGrid<BuildT>* grid2, const ValueT* phi2, const NanoGrid<BuildT>* interior2,
                     ValueT background,
                     cudaStream_t stream = 0,
                     ResourceT& resource = nanovdb::cuda::default_resource<ResourceT>())
{
    using GridT   = NanoGrid<BuildT>;
    using ViewT   = csg::detail::LevelSetView<BuildT, ValueT>;
    using TraitsT = util::cuda::DeviceGridTraits<BuildT>;
    const nanovdb::cuda::ResourceRef<ResourceT> ref(resource);
    static constexpr unsigned int threadsPerBlock = 128;

    // 1. IN_out = IN1 | IN2.
    MergeGrids<BuildT, ResourceT> interiorMerger(interior1, interior2, stream, resource);
    auto interiorHandle = interiorMerger.build();
    const GridT* interiorGrid = interiorHandle.template deviceGrid<BuildT>();

    // 2. C1 | C2.
    MergeGrids<BuildT, ResourceT> merger(grid1, grid2, stream, resource);
    auto mergedHandle = merger.build();
    const GridT* mergedGrid = mergedHandle.template deviceGrid<BuildT>();
    const uint32_t mergedLeafCount = TraitsT::getTreeData(mergedGrid, stream).mNodeCount[0];
    if (!mergedLeafCount) throw std::runtime_error("nanovdb::tools::cuda::csgUnionLeafResolved: both level sets are empty");

    // 3. C_out = (C1 | C2) & ~IN_out.
    csg::detail::BufferT<nanovdb::Mask<3>, ResourceT> retainMasks(stream, ref, mergedLeafCount, nanovdb::cuda::noInit);
    util::cuda::lambdaKernel<<<util::cuda::blocksPerGrid(mergedLeafCount, threadsPerBlock), threadsPerBlock, 0, stream>>>(
        mergedLeafCount, csg::detail::LeafResolvedRetainMaskFunctor<BuildT>(), mergedGrid, interiorGrid, retainMasks.data());
    cudaCheckError();

    // 4. The topology of the union.
    PruneGrid<BuildT, ResourceT> pruner(mergedGrid, retainMasks.data(), stream, resource);
    auto unionHandle = pruner.build();
    const GridT* unionGrid = unionHandle.template deviceGrid<BuildT>();
    const auto treeData = TraitsT::getTreeData(unionGrid, stream);

    // 5. The values of its active voxels; slot 0 holds the background.
    csg::detail::BufferT<ValueT, ResourceT> unionPhi(stream, ref, treeData.mVoxelCount + 1, nanovdb::cuda::noInit);
    cudaCheck(cudaMemcpyAsync(unionPhi.data(), &background, sizeof(ValueT), cudaMemcpyHostToDevice, stream));
    if (treeData.mNodeCount[0]) {
        using ValuesOp = csg::detail::UnionValuesFunctor<BuildT, ValueT>;
        util::cuda::operatorKernel<ValuesOp><<<treeData.mNodeCount[0], ValuesOp::MaxThreadsPerBlock, 0, stream>>>(
            unionGrid, ViewT{grid1, phi1}, ViewT{grid2, phi2}, background, unionPhi.data());
        cudaCheckError();
    }
    cudaCheck(cudaStreamSynchronize(stream));

    return {std::move(unionHandle), std::move(unionPhi), std::move(interiorHandle)};
}// csgUnionLeafResolved

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// csgUnion, csgIntersection, csgDifference and csgComplement (DRAFT: route B of CSG-design.md, sections 3, 4, 6,
// 8 and 9.2). A level set is a LevelSetTracker::State: a ValueOnIndex grid, the sidecar of its active values and the
// invert masks of PropagateSign. Slot 0 of the sidecar holds the background, the value of the level set at infinity.
// It is positive for a bounded solid and negative for the complement of one. An inactive voxel takes the background,
// or its negation where its invert bit is on.
//
// The three binary operations share these conditions:
// - The inputs are not modified. They must share one index space (the same transform), and their backgrounds must
//   be equal in magnitude; the signs may differ.
// - The result is built in one pass from the masks of the inputs: no merged-then-pruned grid, no PropagateSign, no
//   per-voxel query of the invert state (see csg::detail::intersect).
// - std::runtime_error is thrown if the backgrounds differ in magnitude, or if an input has root-level inverted
//   regions (root origins), which this draft does not write for the result.
// - The stream is synchronized before returning.

/// @brief Union of two level sets: the minimum of the two at every point of space.
/// @tparam BuildT Grid build type; must be ValueOnIndex.
/// @tparam ValueT Sidecar element type.
/// @tparam ResourceT Stream-ordered resource of the result, used for all scratch allocations too.
/// @param levelSet1, levelSet2 The inputs.
/// @param stream Stream all work is ordered on.
/// @param resource Allocation resource; must outlive the result.
template<typename BuildT, typename ValueT, typename ResourceT = nanovdb::cuda::DeviceResource>
typename LevelSetTracker<BuildT, ValueT, ResourceT>::State
csgUnion(const typename LevelSetTracker<BuildT, ValueT, ResourceT>::State& levelSet1,
         const typename LevelSetTracker<BuildT, ValueT, ResourceT>::State& levelSet2,
         cudaStream_t stream = 0,
         ResourceT& resource = nanovdb::cuda::default_resource<ResourceT>())
{
    return csg::detail::intersect<BuildT, ValueT, ResourceT>(levelSet1, true, levelSet2, true, true, stream, resource);
}// csgUnion

/// @brief Intersection of two level sets: the maximum of the two at every point of space.
/// @param levelSet1, levelSet2, stream, resource As for csgUnion.
template<typename BuildT, typename ValueT, typename ResourceT = nanovdb::cuda::DeviceResource>
typename LevelSetTracker<BuildT, ValueT, ResourceT>::State
csgIntersection(const typename LevelSetTracker<BuildT, ValueT, ResourceT>::State& levelSet1,
                const typename LevelSetTracker<BuildT, ValueT, ResourceT>::State& levelSet2,
                cudaStream_t stream = 0,
                ResourceT& resource = nanovdb::cuda::default_resource<ResourceT>())
{
    return csg::detail::intersect<BuildT, ValueT, ResourceT>(levelSet1, false, levelSet2, false, false, stream, resource);
}// csgIntersection

/// @brief Difference of two level sets, @a levelSet1 minus @a levelSet2: the maximum of the first and the negation
///        of the second at every point of space.
/// @param levelSet1, levelSet2, stream, resource As for csgUnion.
template<typename BuildT, typename ValueT, typename ResourceT = nanovdb::cuda::DeviceResource>
typename LevelSetTracker<BuildT, ValueT, ResourceT>::State
csgDifference(const typename LevelSetTracker<BuildT, ValueT, ResourceT>::State& levelSet1,
              const typename LevelSetTracker<BuildT, ValueT, ResourceT>::State& levelSet2,
              cudaStream_t stream = 0,
              ResourceT& resource = nanovdb::cuda::default_resource<ResourceT>())
{
    return csg::detail::intersect<BuildT, ValueT, ResourceT>(levelSet1, false, levelSet2, true, false, stream, resource);
}// csgDifference

/// @brief Complement of a level set, in place: every sidecar value is negated, the background in slot 0 included.
///        The grid and the invert masks are unchanged.
/// @param levelSet The level set to complement.
/// @param stream Stream all work is ordered on.
/// @warning The complement of a bounded solid has a negative background. csgUnion, csgIntersection and
///          csgDifference accept it; LevelSetTracker, csgUnionPropagateSign and csgUnionLeafResolved expect a
///          positive one.
/// @note The stream is not synchronized.
template<typename BuildT, typename ValueT, typename ResourceT = nanovdb::cuda::DeviceResource>
void csgComplement(typename LevelSetTracker<BuildT, ValueT, ResourceT>::State& levelSet, cudaStream_t stream = 0)
{
    static constexpr unsigned int threadsPerBlock = 128;
    const size_t valueCount = levelSet.phi.size();
    if (!valueCount) throw std::runtime_error("nanovdb::tools::cuda::csgComplement: the level set has no sidecar");
    util::cuda::lambdaKernel<<<util::cuda::blocksPerGrid(valueCount, threadsPerBlock), threadsPerBlock, 0, stream>>>(
        valueCount, csg::detail::NegateFunctor<ValueT>(), levelSet.phi.data());
    cudaCheckError();
}// csgComplement

} // namespace tools::cuda

} // namespace nanovdb

#endif // NANOVDB_TOOLS_CUDA_CSG_CUH_HAS_BEEN_INCLUDED
