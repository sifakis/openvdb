// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/tools/cuda/CSG.cuh

    \author JaeHyun Lee

    \brief CSG union of two narrow-band level sets, each stored as a NanoVDB indexGrid plus a value sidecar and the
           invert masks of PropagateSign.

    \note DRAFT. Three implementations of the union of nanovdb/examples/ex_csg_nanovdb_cuda/CSG-design.md:
          - csgUnion: route B. One topology build, and the invert masks of the result derived from those of the
            inputs at every tree level.
          - csgUnionPropagateSign: route A, step 5(a). MergeGrids, PruneGrid, and PropagateSign rerun on the result.
          - csgUnionLeafResolved: route A, step 5(c). The invert masks resolved to the leaf level.
          The last two are comparison baselines for the first, and are to be removed once it is settled.

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

// csgUnion (route B): the union built in one pass from the masks of the inputs.

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

/// @brief Word @a w of the leaf-level invert mask of @a levelSet over the leaf at @a origin: the leaf's own invert
///        mask, or all ones / all zeros where the level set has no leaf there.
template<typename BuildT, typename ValueT>
__hostdev__ inline uint64_t leafInvertWord(const LevelSetView<BuildT, ValueT>& levelSet, const Cover& c, uint32_t w)
{
    return c.exists ? levelSet.leafMasks[c.index].words()[w] : (c.inverted ? ~uint64_t(0) : uint64_t(0));
}

// One thread per leaf of an input: retain its active voxels that the other input does not hold as inactive with
// the negative background, C_self & ~IN_other. Over both inputs this is C_out = (C1 | C2) & ~(IN1 | IN2), since
// an active voxel is never inverted in its own level set.
template<typename BuildT, typename ValueT>
struct NativeRetainMaskFunctor
{
    __device__ void operator()(size_t leafID, LevelSetView<BuildT, ValueT> self, LevelSetView<BuildT, ValueT> other,
                               nanovdb::Mask<3>* retainMasks) const
    {
        const auto& leaf = self.grid->tree().template getFirstNode<0>()[leafID];
        const Cover c = cover(other, leaf.origin(), 0);
        for (uint32_t w = 0; w < nanovdb::Mask<3>::WORD_COUNT; ++w)
            retainMasks[leafID].words()[w] = leaf.valueMask().words()[w] & ~leafInvertWord(other, c, w);
    }
}; // NativeRetainMaskFunctor

// One thread per leaf of an input: OR its retained active voxels into the leaf of the union grid at the same
// origin, if the union has one. The counterpart of PruneLeafMasksFunctor for several inputs.
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

// One thread per leaf of the union: IN_out = IN1 | IN2, each input's invert mask resolved to this leaf.
template<typename BuildT, typename ValueT>
struct NativeLeafInvertMaskFunctor
{
    __device__ void operator()(size_t leafID, const NanoGrid<BuildT>* unionGrid, LevelSetView<BuildT, ValueT> input1,
                               LevelSetView<BuildT, ValueT> input2, nanovdb::Mask<3>* leafMasks) const
    {
        const auto& leaf = unionGrid->tree().template getFirstNode<0>()[leafID];
        const Cover c1 = cover(input1, leaf.origin(), 0), c2 = cover(input2, leaf.origin(), 0);
        for (uint32_t w = 0; w < nanovdb::Mask<3>::WORD_COUNT; ++w)
            leafMasks[leafID].words()[w] = leafInvertWord(input1, c1, w) | leafInvertWord(input2, c2, w);
    }
}; // NativeLeafInvertMaskFunctor

// One block per lower (LEVEL 1) or upper (LEVEL 2) node of the union, one thread per mask word:
// IN_out = (IN1 | IN2 | C1 | C2) & ~C_out. A childless slot of the union is inverted unless both inputs hold it
// as not inverted: a slot that had a child in an input and has none in the union lost every active voxel to the
// other input's inverted region. An input with no node here contributes C = 0 and IN = all ones or all zeros.
template<typename BuildT, typename ValueT, int LEVEL>
struct NativeInternalInvertMaskFunctor
{
    using MaskT = nanovdb::Mask<LEVEL + 3>;
    static constexpr int MaxThreadsPerBlock = MaskT::WORD_COUNT;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ static uint64_t word(const LevelSetView<BuildT, ValueT>& input, const nanovdb::Coord& origin, uint32_t w)
    {
        const Cover c = cover(input, origin, LEVEL);
        if (!c.exists) return c.inverted ? ~uint64_t(0) : uint64_t(0);
        const auto& node = input.grid->tree().template getFirstNode<LEVEL>()[c.index];
        const MaskT& invert = LEVEL == 1 ? reinterpret_cast<const MaskT*>(input.lowerMasks)[c.index]
                                         : reinterpret_cast<const MaskT*>(input.upperMasks)[c.index];
        return invert.words()[w] | node.childMask().words()[w];
    }

    __device__ void operator()(const NanoGrid<BuildT>* unionGrid, LevelSetView<BuildT, ValueT> input1,
                               LevelSetView<BuildT, ValueT> input2, MaskT* masks) const
    {
        const auto&    node = unionGrid->tree().template getFirstNode<LEVEL>()[blockIdx.x];
        const uint32_t w    = threadIdx.x;
        masks[blockIdx.x].words()[w] = (word(input1, node.origin(), w) | word(input2, node.origin(), w)) &
                                       ~node.childMask().words()[w];
    }
}; // NativeInternalInvertMaskFunctor

/// @brief The grid whose active voxels are the retained active voxels of the two inputs: MergeGrids and PruneGrid
///        in one build, so that a node the union does not need is never created. A leaf of an input is stamped
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

/// @brief Union of two level sets, built in one pass from the masks of the inputs (DRAFT: route B of
///        CSG-design.md, sections 8 and 9.2). No merged-then-pruned grid, no PropagateSign, no per-voxel query.
/// @details 1. Retain masks, per leaf of each input: its active voxels outside the other input's inverted region.
///          2. One topology build: only the nodes that hold a retained voxel exist.
///          3. Values: the minimum over the inputs in which the voxel is active.
///          4. Invert masks, per node of the union: IN1 | IN2 at the leaves, (IN1 | IN2 | C1 | C2) & ~C_out at
///             the lower and upper nodes.
///          Where an input has no node at the origin of a node being processed, its state there (inverted or
///          not) is read by one tree descent per node, from the coarser level that holds it.
/// @param levelSet1, levelSet2 The inputs; not modified.
/// @param background Positive magnitude that both inputs assign to the voxels outside their bands.
/// @param stream Stream all work is ordered on.
/// @param resource Allocation resource; must outlive the result.
/// @throw std::runtime_error if neither input has an active voxel, or if an input has root-level inverted regions
///        (root origins), which this draft does not write for the union.
/// @warning The inputs must share one index space (the same transform) and one positive background, and each must
///          take the positive background at infinity.
/// @note The stream is synchronized before returning.
template<typename BuildT, typename ValueT, typename ResourceT = nanovdb::cuda::DeviceResource>
typename LevelSetTracker<BuildT, ValueT, ResourceT>::State
csgUnion(const typename LevelSetTracker<BuildT, ValueT, ResourceT>::State& levelSet1,
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
    static constexpr unsigned int threadsPerBlock = 128;

    if (levelSet1.rootOrigins.size() || levelSet2.rootOrigins.size())
        throw std::runtime_error("nanovdb::tools::cuda::csgUnion: root-level inverted regions are not supported yet");

    const auto view = [](const StateT& s) {
        return ViewT{s.grid.template deviceGrid<BuildT>(), s.phi.data(), s.leafMasks.data(), s.lowerMasks.data(),
                     s.upperMasks.data(), s.rootOrigins.data(), s.rootOrigins.size()};
    };
    const ViewT input1 = view(levelSet1), input2 = view(levelSet2);
    const uint32_t leafCount1 = TraitsT::getTreeData(input1.grid, stream).mNodeCount[0];
    const uint32_t leafCount2 = TraitsT::getTreeData(input2.grid, stream).mNodeCount[0];
    if (!leafCount1 && !leafCount2) throw std::runtime_error("nanovdb::tools::cuda::csgUnion: both level sets are empty");

    // 1. The active voxels of each input that stay active in the union.
    csg::detail::BufferT<nanovdb::Mask<3>, ResourceT> retainMasks1(stream, ref, leafCount1, nanovdb::cuda::noInit);
    csg::detail::BufferT<nanovdb::Mask<3>, ResourceT> retainMasks2(stream, ref, leafCount2, nanovdb::cuda::noInit);
    using RetainOp = csg::detail::NativeRetainMaskFunctor<BuildT, ValueT>;
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

    // 2. The topology of the union.
    auto unionHandle = csg::detail::mergePrunedGrids<BuildT, ResourceT>(input1.grid, retainMasks1.data(),
                                                                       input2.grid, retainMasks2.data(), stream, resource);
    const GridT* unionGrid = unionHandle.template deviceGrid<BuildT>();
    const auto treeData = TraitsT::getTreeData(unionGrid, stream);

    // 3. The values of its active voxels; slot 0 holds the background.
    csg::detail::BufferT<ValueT, ResourceT> unionPhi(stream, ref, treeData.mVoxelCount + 1, nanovdb::cuda::noInit);
    cudaCheck(cudaMemcpyAsync(unionPhi.data(), &background, sizeof(ValueT), cudaMemcpyHostToDevice, stream));
    if (treeData.mNodeCount[0]) {
        using ValuesOp = csg::detail::UnionValuesFunctor<BuildT, ValueT>;
        util::cuda::operatorKernel<ValuesOp><<<treeData.mNodeCount[0], ValuesOp::MaxThreadsPerBlock, 0, stream>>>(
            unionGrid, input1, input2, background, unionPhi.data());
        cudaCheckError();
    }

    // 4. The invert masks of the union, level by level.
    csg::detail::BufferT<nanovdb::Mask<3>, ResourceT> leafMasks(stream, ref, treeData.mNodeCount[0], nanovdb::cuda::noInit);
    csg::detail::BufferT<nanovdb::Mask<4>, ResourceT> lowerMasks(stream, ref, treeData.mNodeCount[1], nanovdb::cuda::noInit);
    csg::detail::BufferT<nanovdb::Mask<5>, ResourceT> upperMasks(stream, ref, treeData.mNodeCount[2], nanovdb::cuda::noInit);
    if (treeData.mNodeCount[0]) {
        util::cuda::lambdaKernel<<<util::cuda::blocksPerGrid(treeData.mNodeCount[0], threadsPerBlock), threadsPerBlock, 0, stream>>>(
            treeData.mNodeCount[0], csg::detail::NativeLeafInvertMaskFunctor<BuildT, ValueT>(), unionGrid, input1, input2,
            leafMasks.data());
        cudaCheckError();
        using LowerOp = csg::detail::NativeInternalInvertMaskFunctor<BuildT, ValueT, 1>;
        util::cuda::operatorKernel<LowerOp><<<treeData.mNodeCount[1], LowerOp::MaxThreadsPerBlock, 0, stream>>>(
            unionGrid, input1, input2, lowerMasks.data());
        cudaCheckError();
        using UpperOp = csg::detail::NativeInternalInvertMaskFunctor<BuildT, ValueT, 2>;
        util::cuda::operatorKernel<UpperOp><<<treeData.mNodeCount[2], UpperOp::MaxThreadsPerBlock, 0, stream>>>(
            unionGrid, input1, input2, upperMasks.data());
        cudaCheckError();
    }
    cudaCheck(cudaStreamSynchronize(stream));

    // todo: the root origins of the union (section 8.3 of CSG-design.md); empty here, see @throw.
    return StateT{std::move(unionHandle), std::move(unionPhi), std::move(leafMasks), std::move(lowerMasks),
                  std::move(upperMasks), decltype(levelSet1.rootOrigins)(stream, ref, 0, nanovdb::cuda::noInit)};
}// csgUnion

} // namespace tools::cuda

} // namespace nanovdb

#endif // NANOVDB_TOOLS_CUDA_CSG_CUH_HAS_BEEN_INCLUDED
