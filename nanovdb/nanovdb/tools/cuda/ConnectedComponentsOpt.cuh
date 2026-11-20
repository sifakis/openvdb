// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/tools/cuda/ConnectedComponentsOpt.cuh

    \authors JaeHyun Lee

    \brief Experimental, faster variant of tools::cuda::ConnectedComponents, with the same labels.

    \details ConnectedComponentsOpt has the interface and output of ConnectedComponents, and reuses its shared pieces
             (ConnectedComponents.cuh: the component vocabulary, the cross-leaf enumeration, the lock-free union-find
             and the component ranks). Three stages differ:
             1. Leaf stage: the union-find runs over the z-runs of a leaf instead of its voxels. A z-run is a maximal
                stretch of active voxels along z within one row (x, y); two runs are adjacent iff they lie in
                neighbouring rows and their bits overlap. The schedule is LeafUnionFind's (Liu and Tarjan, P then R),
                with one warp per leaf instead of one block.
             2. Cross-leaf stage: each pair of touching leaf components is united where it is found, so there is
                no edge count, prefix sum, host synchronization, edge list or separate unite pass.
             3. Labels: one warp per leaf reads each component's label once and writes it at every voxel of the
                component; only slot 0 of the label sidecar is initialized, since every other slot is an active
                voxel.
             See examples/ex_connected_components_opt/OPTIMIZATION_PLAN.md for measurements and open items.

    \warning The header file contains cuda device code so be sure
             to only include it in .cu files (or other .cuh files)
*/

#ifndef NANOVDB_TOOLS_CUDA_CONNECTEDCOMPONENTSOPT_CUH_HAS_BEEN_INCLUDED
#define NANOVDB_TOOLS_CUDA_CONNECTEDCOMPONENTSOPT_CUH_HAS_BEEN_INCLUDED

#include <nanovdb/tools/cuda/ConnectedComponents.cuh>

#include <utility>   // std::pair
#include <stdexcept> // std::runtime_error

// Define utility macro used to call cub functions that use dynamic temporary storage
#ifndef CALL_CUBS
#ifdef _WIN32
#define CALL_CUBS(func, ...) \
    cudaCheck(cub::func(nullptr, mTempDevicePool.requestedSize(), __VA_ARGS__, mStream)); \
    mTempDevicePool.reallocate(mStream); \
    cudaCheck(cub::func(mTempDevicePool.data(), mTempDevicePool.size(), __VA_ARGS__, mStream));
#else
#define CALL_CUBS(func, args...) \
    cudaCheck(cub::func(nullptr, mTempDevicePool.requestedSize(), args, mStream)); \
    mTempDevicePool.reallocate(mStream); \
    cudaCheck(cub::func(mTempDevicePool.data(), mTempDevicePool.size(), args, mStream));
#endif
#endif // ifndef CALL_CUBS

namespace nanovdb {

namespace tools::cuda {

/// @brief Same interface and labels as ConnectedComponents, with the leaf, cross-leaf and label stages of the file
///        comment.
template <typename BuildT, typename ResourceT = nanovdb::cuda::DeviceResource>
class ConnectedComponentsOpt : public ConnectedComponentsBase
{
    static_assert(nanovdb::BuildTraits<BuildT>::is_onindex,
                  "ConnectedComponentsOpt labels the active voxels of a ValueOnIndex grid");
    static_assert(nanovdb::cuda::is_async_resource<ResourceT>::value,
                  "ConnectedComponentsOpt allocates stream-ordered scratch and requires an AsyncResource");
    static_assert(ResourceT::DEFAULT_ALIGNMENT >= alignof(uint64_t),
                  "ConnectedComponentsOpt stores scratch up to uint64_t width and requires word-aligned allocations");

    using GridT = NanoGrid<BuildT>;
    using TreeT = NanoTree<BuildT>;

    /// @brief Stream-ordered, device-only scratch storage borrowed from the injected resource.
    template <typename T>
    using BufT = nanovdb::cuda::Buffer<T, nanovdb::cuda::ResourceRef<ResourceT>>;

public:

    ConnectedComponentsOpt(const GridT* d_srcGrid, cudaStream_t stream = 0,
                           ResourceT& resource = nanovdb::cuda::default_resource<ResourceT>())
        : mStream(stream), mTimer(stream), mDeviceSrcGrid(d_srcGrid)
        , mResource(&resource)
        , mTempDevicePool(resource)
        , mLeafComponentCounts(stream, resource, 0, nanovdb::cuda::noInit)
        , mLeafComponentOffsets(stream, resource, 0, nanovdb::cuda::noInit)
        , mLeafComponentMasks(stream, resource, 0, nanovdb::cuda::noInit)
        , mLeafComponentFaceMasks(stream, resource, 0, nanovdb::cuda::noInit)
        , mComponentParent(stream, resource, 0, nanovdb::cuda::noInit)
        , mVoxelLabel(stream, resource, 0, nanovdb::cuda::noInit) {}

    /// @brief Toggle on and off verbose mode
    /// @param level Verbose level: 0=quiet, 1=timing
    void setVerbose(int level = 1) { mVerbose = level; }

    /// @brief Run the pipeline and return { labels, componentCount }, exactly as ConnectedComponents does.
    std::pair<BufT<ComponentLabelT>, ComponentLabelT> getVoxelLabelsAndCount()
    {
        mSrcTreeData = util::cuda::DeviceGridTraits<BuildT>::getTreeData(mDeviceSrcGrid, mStream);
        if (mSrcTreeData.mTileCount[0] || mSrcTreeData.mTileCount[1] || mSrcTreeData.mTileCount[2])
            throw std::runtime_error("Topological operations not supported on grids with value tiles");

        processLeafConnectedComponents();
        mLeafComponentCounts.destroy();
        processGlobalConnectedComponents();
        mLeafComponentFaceMasks.destroy();
        processVoxelLabels();
        mLeafComponentOffsets.destroy();
        mLeafComponentMasks.destroy();
        mComponentParent.destroy();
        cudaCheck(cudaStreamSynchronize(mStream));
        return { std::move(mVoxelLabel), mGlobalComponentCount };
    }

private:

    // Stage 1: per-leaf components (union-find over z-runs) -> per-leaf count, prefix-sum offset, component masks
    // and face masks.
    void processLeafConnectedComponents();

    // Stage 2: global union-find, uniting touching components across leaf faces where they are found.
    void processGlobalConnectedComponents();

    // Stage 3: dense component ids, written per component into the per-voxel label sidecar.
    void processVoxelLabels();

    /// @brief Borrowed handle to the injected resource, for the scratch allocated after construction.
    nanovdb::cuda::ResourceRef<ResourceT> ref() { return nanovdb::cuda::ResourceRef<ResourceT>(*mResource); }

    cudaStream_t                       mStream{0};
    util::cuda::Timer                  mTimer;
    int                                mVerbose{0};
    const GridT                       *mDeviceSrcGrid;
    typename TreeT::DataType           mSrcTreeData{};
    ResourceT                         *mResource;
    nanovdb::cuda::TempPool<ResourceT> mTempDevicePool;

    uint64_t                           mLeafComponentAggregateCount{0}; // K: leaf components over all leaves
    BufT<uint16_t>                     mLeafComponentCounts;     // leafCount x uint16_t
    BufT<uint64_t>                     mLeafComponentOffsets;    // (leafCount+1) x uint64_t
    BufT<nanovdb::Mask<3>>             mLeafComponentMasks;      // K x Mask<3>
    BufT<uint64_t[6]>                  mLeafComponentFaceMasks;  // K x uint64_t[6]
    BufT<uint64_t>                     mComponentParent;         // K x uint64_t
    BufT<ComponentLabelT>              mVoxelLabel;              // valueCount x ComponentLabelT
    ComponentLabelT                    mGlobalComponentCount{0};
}; // tools::cuda::ConnectedComponentsOpt<BuildT, ResourceT>

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace components::detail {

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Stage 1: leaf-local connected components, by union-find over z-runs.
//
// A leaf's 8^3 voxels are 64 rows (x, y) of 8 bits (z): row r = 8x + y is byte r of the value mask. A z-run is a
// maximal stretch of set bits in one row; two runs are adjacent iff their rows are face neighbours (x +- 1 or y +- 1)
// and their bits overlap, since voxels adjacent along z already lie in one run. One warp handles a leaf, with the
// runs in shared memory; four leaves share a block.

constexpr unsigned WarpFullMask = 0xFFFFFFFFu;

/// @brief The z-runs of one leaf, built by one warp. Runs are numbered in ascending order of their first voxel offset
///        (row by row, then by z), so the smallest run number in a component holds the component's smallest voxel
///        offset: the root LeafUnionFind picks.
struct LeafRuns
{
    static constexpr int MaxRuns = 256;  // 64 rows, at most 4 runs in an 8-bit row
    uint16_t rowBase[65];                // first run number of each row; rowBase[64] = run count
    uint8_t  runRow[MaxRuns];
    uint8_t  runBits[MaxRuns];           // the run's voxels, as bits z of its row
    int      parentA[MaxRuns], parentB[MaxRuns];

    __device__ int count() const { return rowBase[64]; }

    // Lane l handles rows l and l + 32.
    __device__ void build(const uint64_t* leafWords, int lane)
    {
        uint32_t bytes[2], starts[2], counts[2], before[2];
        for (int h = 0; h < 2; ++h) {
            const int r = lane + 32 * h;
            bytes[h]  = uint32_t(leafWords[r >> 3] >> (8 * (r & 7))) & 0xFF;
            starts[h] = bytes[h] & ~(bytes[h] << 1) & 0xFF;   // first voxel of each run
            counts[h] = __popc(starts[h]);
        }
        // Exclusive prefix over rows 0..63: lanes hold rows 0..31 first, then 32..63.
        for (int h = 0; h < 2; ++h) {
            uint32_t inclusive = counts[h];
            for (int d = 1; d < 32; d *= 2) {
                const uint32_t v = __shfl_up_sync(WarpFullMask, inclusive, d);
                if (lane >= d) inclusive += v;
            }
            before[h] = inclusive - counts[h];
            if (h == 1) before[1] += __shfl_sync(WarpFullMask, before[0] + counts[0], 31);
        }
        // Run numbers and counts fit uint16_t: a leaf has at most MaxRuns runs.
        rowBase[lane]      = uint16_t(before[0]);
        rowBase[lane + 32] = uint16_t(before[1]);
        if (lane == 31) rowBase[64] = uint16_t(before[1] + counts[1]);
        for (int h = 0; h < 2; ++h) {
            int k = int(before[h]);
            for (uint32_t s = starts[h]; s; s &= s - 1) {
                const int z0     = __ffs(s) - 1;
                const int length = __ffs(~(bytes[h] >> z0)) - 1;
                runRow[k]  = uint8_t(lane + 32 * h);                  // a row index, < 64
                runBits[k] = uint8_t(((1u << length) - 1) << z0);     // bits of an 8-bit row
                ++k;
            }
        }
        __syncwarp();
    }

    /// @brief Smallest parent among run i and the overlapping runs of the four neighbouring rows.
    __device__ int neighborMin(const int* parent, int i) const
    {
        const int r = runRow[i], x = r >> 3, y = r & 7;
        int m = parent[i];
        if (x > 0) m = rowMin(parent, r - 8, runBits[i], m);
        if (x < 7) m = rowMin(parent, r + 8, runBits[i], m);
        if (y > 0) m = rowMin(parent, r - 1, runBits[i], m);
        if (y < 7) m = rowMin(parent, r + 1, runBits[i], m);
        return m;
    }

    /// @brief Smallest of @a m and the parents of the runs of row @a row that overlap @a bits.
    __device__ int rowMin(const int* parent, int row, uint32_t bits, int m) const
    {
        for (int k = rowBase[row]; k < rowBase[row + 1]; ++k)
            if (runBits[k] & bits) m = ::min(m, parent[k]);
        return m;
    }

    /// @brief Index of the component whose root is run @a root, among the components of the leaf in root order:
    ///        the number of roots before it, from the bit set @a isRoot of MaxRuns bits.
    __device__ static int componentIndex(const uint32_t* isRoot, int root)
    {
        int index = __popc(isRoot[root >> 5] & ((1u << (root & 31)) - 1));
        for (int w = 0; w < (root >> 5); ++w) index += __popc(isRoot[w]);
        return index;
    }
};

/// @brief LeafUnionFind's hook/compress/solve, with one warp instead of one block and runs instead of voxels.
struct RunUnionFind
{
    __device__ static bool hook(const LeafRuns& L, int*& cur, int*& nxt, int n, int lane, bool rootsOnly)
    {
        for (int i = lane; i < n; i += 32) nxt[i] = cur[i];
        __syncwarp();
        bool changed = false;
        for (int i = lane; i < n; i += 32) {
            const int pi = cur[i];
            if (rootsOnly && cur[pi] != pi) continue;
            const int m = L.neighborMin(cur, i);
            if (m < pi && atomicMin_block(&nxt[pi], m) > m) changed = true;
        }
        __syncwarp();
        int* t = cur; cur = nxt; nxt = t;
        return __any_sync(WarpFullMask, changed);
    }

    __device__ static bool compress(int*& cur, int*& nxt, int n, int lane)
    {
        bool changed = false;
        for (int i = lane; i < n; i += 32) {
            const int v = cur[cur[i]];
            changed |= v != cur[i];
            nxt[i] = v;
        }
        __syncwarp();
        int* t = cur; cur = nxt; nxt = t;
        return __any_sync(WarpFullMask, changed);
    }

    // On return cur[i] is the smallest run number of run i's component.
    __device__ static void solve(const LeafRuns& L, int*& cur, int*& nxt, int lane)
    {
        const int n = L.count();
        for (int i = lane; i < n; i += 32) cur[i] = i;
        __syncwarp();
        hook(L, cur, nxt, n, lane, false);
        compress(cur, nxt, n, lane);
        compress(cur, nxt, n, lane);
        compress(cur, nxt, n, lane);
        compress(cur, nxt, n, lane);
        constexpr int SwitchToRootAfter = 8, MaxConvergenceIters = 64;
        for (int it = 0; it < MaxConvergenceIters; ++it) {
            bool changed = hook(L, cur, nxt, n, lane, it >= SwitchToRootAfter);
            changed |= compress(cur, nxt, n, lane);
            if (!changed) return;
        }
        NANOVDB_ASSERT(false);  // exhausted the backstop: algorithm R's bound was not met
    }
};

/// @brief The six face masks of one component's Mask<3>, laid out as LeafComponentMaskFunctor does.
__device__ inline void writeLeafFaces(const uint64_t (&w)[8], uint64_t* face)
{
    using LeafNeighborTap = ConnectedComponentsBase::LeafNeighborTap;
    face[LeafNeighborTap::minusX] = w[0];
    face[LeafNeighborTap::plusX]  = w[7];
    uint64_t mY = w[7] & 0xFF, pY = (w[7] >> 56) & 0xFF;
    for (int x = 6; x >= 0; --x) {
        mY = (mY << 8) | (w[x] & 0xFF);
        pY = (pY << 8) | ((w[x] >> 56) & 0xFF);
    }
    face[LeafNeighborTap::minusY] = mY;
    face[LeafNeighborTap::plusY]  = pY;
    uint64_t mZ = w[7] & UINT64_C(0x0101010101010101);
    uint64_t pZ = (w[7] >> 7) & UINT64_C(0x0101010101010101);
    for (int x = 6; x >= 0; --x) {
        mZ = (mZ << 1) | (w[x] & UINT64_C(0x0101010101010101));
        pZ = (pZ << 1) | ((w[x] >> 7) & UINT64_C(0x0101010101010101));
    }
    face[LeafNeighborTap::minusZ] = mZ;
    face[LeafNeighborTap::plusZ]  = pZ;
}

/// @brief Counts the components of each leaf, which size the per-leaf slices that RunLeafComponentMaskFunctor fills.
template <typename BuildT>
struct RunLeafComponentCountFunctor
{
    static constexpr int LeavesPerBlock             = 4;
    static constexpr int MaxThreadsPerBlock         = 32 * LeavesPerBlock;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(const NanoGrid<BuildT>* d_grid, uint16_t* d_counts, uint32_t leafCount)
    {
        __shared__ LeafRuns sRuns[LeavesPerBlock];
        const int      lane   = threadIdx.x & 31;
        const uint32_t leafID = blockIdx.x * LeavesPerBlock + (threadIdx.x >> 5);
        if (leafID >= leafCount) return;
        LeafRuns& L = sRuns[threadIdx.x >> 5];
        L.build(d_grid->tree().template getFirstNode<0>()[leafID].valueMask().words(), lane);
        int* cur = L.parentA;
        int* nxt = L.parentB;
        RunUnionFind::solve(L, cur, nxt, lane);
        int roots = 0;
        for (int i = lane; i < L.count(); i += 32) roots += cur[i] == i;
        roots = __reduce_add_sync(WarpFullMask, roots);
        if (lane == 0) d_counts[leafID] = uint16_t(roots);  // at most MaxRuns components per leaf
    }
};

/// @brief Re-solves each leaf and fills its slice [d_offsets[leaf], d_offsets[leaf+1]), in ascending root order: each
///        component's voxel Mask<3>, and its 6 face masks.
template <typename BuildT>
struct RunLeafComponentMaskFunctor
{
    static constexpr int LeavesPerBlock             = 4;
    static constexpr int MaxThreadsPerBlock         = 32 * LeavesPerBlock;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(const NanoGrid<BuildT>* d_grid, const uint64_t* d_offsets,
                               nanovdb::Mask<3>* d_masks, uint64_t (*d_faces)[6], uint32_t leafCount)
    {
        __shared__ LeafRuns sRuns[LeavesPerBlock];
        __shared__ uint32_t sIsRoot[LeavesPerBlock][LeafRuns::MaxRuns / 32];
        const int      lane   = threadIdx.x & 31;
        const int      warp   = threadIdx.x >> 5;
        const uint32_t leafID = blockIdx.x * LeavesPerBlock + warp;
        if (leafID >= leafCount) return;
        LeafRuns& L = sRuns[warp];
        L.build(d_grid->tree().template getFirstNode<0>()[leafID].valueMask().words(), lane);
        int* cur = L.parentA;
        int* nxt = L.parentB;
        RunUnionFind::solve(L, cur, nxt, lane);
        const int n = L.count();

        // Roots ascend with the smallest offset, so a component's index is the number of roots before its root.
        uint32_t* isRoot = sIsRoot[warp];
        if (lane < LeafRuns::MaxRuns / 32) isRoot[lane] = 0;
        __syncwarp();
        for (int i = lane; i < n; i += 32)
            if (cur[i] == i) atomicOr_block(&isRoot[i >> 5], 1u << (i & 31));
        __syncwarp();
        int components = 0;
        for (int w = 0; w < LeafRuns::MaxRuns / 32; ++w) components += __popc(isRoot[w]);

        // Each run ORs its voxels into its component's mask, then lane k derives component k's faces.
        nanovdb::Mask<3>* masks = d_masks + d_offsets[leafID];
        for (int t = lane; t < 8 * components; t += 32) masks[t >> 3].words()[t & 7] = 0;
        __syncwarp();
        for (int i = lane; i < n; i += 32) {
            const int r = L.runRow[i];
            atomicOr(reinterpret_cast<unsigned long long*>(&masks[LeafRuns::componentIndex(isRoot, cur[i])].words()[r >> 3]),
                     (unsigned long long)L.runBits[i] << (8 * (r & 7)));
        }
        __threadfence_block();
        __syncwarp();
        for (int k = lane; k < components; k += 32) {
            uint64_t w[8];
            for (int x = 0; x < 8; ++x) w[x] = masks[k].words()[x];
            writeLeafFaces(w, d_faces[d_offsets[leafID] + k]);
        }
    }
};

} // namespace components::detail

template <typename BuildT, typename ResourceT>
void ConnectedComponentsOpt<BuildT, ResourceT>::processLeafConnectedComponents()
{
    const uint32_t leafCount = mSrcTreeData.mNodeCount[0];
    mLeafComponentCounts = BufT<uint16_t>(mStream, this->ref(), leafCount, nanovdb::cuda::noInit);
    if (leafCount == 0) return;

    using CountOp = components::detail::RunLeafComponentCountFunctor<BuildT>;
    if (mVerbose==1) mTimer.start("Per-leaf connected-component counting");
    util::cuda::operatorKernel<CountOp>
        <<<util::cuda::blocksPerGrid(leafCount, CountOp::LeavesPerBlock), CountOp::MaxThreadsPerBlock, 0, mStream>>>(
            mDeviceSrcGrid, mLeafComponentCounts.data(), leafCount);
    cudaCheckError();
    if (mVerbose==1) mTimer.stop();

    // Offsets: [0] = 0, [1..leafCount] = inclusive sum of the counts, so [leafCount] = K.
    mLeafComponentOffsets = BufT<uint64_t>(mStream, this->ref(), std::size_t(leafCount) + 1, nanovdb::cuda::noInit);
    uint64_t* d_offsets = mLeafComponentOffsets.data();
    cudaCheck(cudaMemsetAsync(d_offsets, 0, sizeof(uint64_t), mStream));
    static constexpr unsigned int threadsPerBlock = 256;
    util::cuda::lambdaKernel<<<util::cuda::blocksPerGrid(leafCount, threadsPerBlock), threadsPerBlock, 0, mStream>>>(
        leafCount, components::detail::UpcastCountsFunctor{}, mLeafComponentCounts.data(), d_offsets);
    cudaCheckError();
    if (mVerbose==1) mTimer.start("Per-leaf component offset prefix sum");
    CALL_CUBS(DeviceScan::InclusiveSum, d_offsets + 1, d_offsets + 1, leafCount);
    cudaCheckError();
    if (mVerbose==1) mTimer.stop();
    cudaCheck(cudaMemcpyAsync(&mLeafComponentAggregateCount, d_offsets + leafCount, sizeof(uint64_t),
                              cudaMemcpyDeviceToHost, mStream));
    cudaCheck(cudaStreamSynchronize(mStream));
    if (mLeafComponentAggregateCount > MaxComponentCount)
        throw std::runtime_error("nanovdb::tools::cuda::ConnectedComponentsOpt: leaf-local component "
                                 "count exceeds the indexable range");

    mLeafComponentMasks = BufT<nanovdb::Mask<3>>(mStream, this->ref(), mLeafComponentAggregateCount, nanovdb::cuda::noInit);
    mLeafComponentFaceMasks = BufT<uint64_t[6]>(mStream, this->ref(), mLeafComponentAggregateCount, nanovdb::cuda::noInit);
    using MaskOp = components::detail::RunLeafComponentMaskFunctor<BuildT>;
    if (mVerbose==1) mTimer.start("Per-leaf component mask fill");
    util::cuda::operatorKernel<MaskOp>
        <<<util::cuda::blocksPerGrid(leafCount, MaskOp::LeavesPerBlock), MaskOp::MaxThreadsPerBlock, 0, mStream>>>(
            mDeviceSrcGrid, d_offsets, mLeafComponentMasks.data(), mLeafComponentFaceMasks.data(), leafCount);
    cudaCheckError();
    if (mVerbose==1) mTimer.stop();
}// ConnectedComponentsOpt<BuildT, ResourceT>::processLeafConnectedComponents

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace components::detail {

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Stage 2: global union-find across leaf faces.
//
// forEachCrossLeafEdge enumerates the touching component pairs of a leaf and its +X/+Y/+Z neighbours; each pair is
// united on the spot with the lock-free unite of ConnectedComponents. The union is order-independent (a class's
// minimum slot is never linked under another), so the representatives are those of uniting a collected edge list.

/// @brief Unites the components of a leaf with those of its +X/+Y/+Z neighbours whose touching faces intersect.
///        One block per leaf.
template <typename BuildT>
struct CrossLeafUniteFunctor
{
    static constexpr int MaxThreadsPerBlock         = 128;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(const NanoGrid<BuildT>* d_grid, const uint64_t* d_offsets, const uint64_t (*d_faces)[6],
                               uint64_t* d_parent)
    {
        using ComponentLabelT = ConnectedComponentsBase::ComponentLabelT;
        forEachCrossLeafEdge<BuildT>(d_grid, d_offsets, d_faces, blockIdx.x, threadIdx.x, blockDim.x,
            [&] __device__ (ComponentLabelT a, ComponentLabelT b) { unite(d_parent, uint64_t(a), uint64_t(b)); });
    }
};

} // namespace components::detail

template <typename BuildT, typename ResourceT>
void ConnectedComponentsOpt<BuildT, ResourceT>::processGlobalConnectedComponents()
{
    const uint64_t K = mLeafComponentAggregateCount;
    mComponentParent = BufT<uint64_t>(mStream, this->ref(), K, nanovdb::cuda::noInit);
    if (K == 0) return;
    uint64_t* d_parent = mComponentParent.data();
    static constexpr unsigned int threadsPerBlock = 256;

    if (mVerbose==1) mTimer.start("Global union-find init");
    util::cuda::lambdaKernel<<<util::cuda::blocksPerGrid(K, threadsPerBlock), threadsPerBlock, 0, mStream>>>(
        K, components::detail::LabelInitFunctor{}, d_parent);
    cudaCheckError();
    if (mVerbose==1) mTimer.stop();

    using UniteOp = components::detail::CrossLeafUniteFunctor<BuildT>;
    if (mVerbose==1) mTimer.start("Cross-leaf unite");
    util::cuda::operatorKernel<UniteOp><<<mSrcTreeData.mNodeCount[0], UniteOp::MaxThreadsPerBlock, 0, mStream>>>(
        mDeviceSrcGrid, mLeafComponentOffsets.data(), mLeafComponentFaceMasks.data(), d_parent);
    cudaCheckError();
    if (mVerbose==1) mTimer.stop();

    if (mVerbose==1) mTimer.start("Global union-find flatten");
    util::cuda::lambdaKernel<<<util::cuda::blocksPerGrid(K, threadsPerBlock), threadsPerBlock, 0, mStream>>>(
        K, components::detail::LabelFlattenFunctor{}, d_parent);
    cudaCheckError();
    if (mVerbose==1) mTimer.stop();
}// ConnectedComponentsOpt<BuildT, ResourceT>::processGlobalConnectedComponents

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace components::detail {

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Stage 3: dense labels, written per component.

/// @brief One warp per leaf: each component's dense label is read once and written at the value index of every
///        voxel of its mask.
template <typename BuildT>
struct WarpVoxelLabelScatterFunctor
{
    static constexpr int LeavesPerBlock             = 4;
    static constexpr int MaxThreadsPerBlock         = 32 * LeavesPerBlock;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    template <typename LabelT>
    __device__ void operator()(const NanoGrid<BuildT>* d_grid, const uint64_t* d_offsets, const nanovdb::Mask<3>* d_masks,
                               const uint64_t* d_parent, const LabelT* d_rank, LabelT* d_voxelLabel, uint32_t leafCount)
    {
        const int      lane   = threadIdx.x & 31;
        const uint32_t leafID = blockIdx.x * LeavesPerBlock + (threadIdx.x >> 5);
        if (leafID >= leafCount) return;
        const auto& leaf = d_grid->tree().template getFirstNode<0>()[leafID];
        const uint64_t base = d_offsets[leafID], end = d_offsets[leafID + 1];
        for (uint64_t s = base; s < end; ++s) {
            const LabelT label = LabelT(d_rank[d_parent[s]] - 1);  // dense id in [0, N)
            const uint32_t* words = reinterpret_cast<const uint32_t*>(d_masks[s].words());
            for (int h = 0; h < 16; ++h)
                if ((words[h] >> lane) & 1u) d_voxelLabel[leaf.getValue(uint32_t(h * 32 + lane))] = label;
        }
    }
};

} // namespace components::detail

template <typename BuildT, typename ResourceT>
void ConnectedComponentsOpt<BuildT, ResourceT>::processVoxelLabels()
{
    const uint32_t leafCount = mSrcTreeData.mNodeCount[0];
    const uint64_t valueCount =
        util::cuda::DeviceGridTraits<BuildT>::getValueCount(mDeviceSrcGrid, mStream);
    // Every slot but 0 is an active voxel, and is written below, only if the grid indexes exactly its active voxels
    // (no stats), as NanoVDB's CUDA builders produce.
    NANOVDB_ASSERT(valueCount == mSrcTreeData.mVoxelCount + 1);

    mVoxelLabel = BufT<ComponentLabelT>(mStream, this->ref(), valueCount, nanovdb::cuda::noInit);
    cudaCheck(cudaMemsetAsync(mVoxelLabel.data(), 0xFF, sizeof(ComponentLabelT), mStream)); // background = 0xFFFFFFFF

    mGlobalComponentCount = 0;
    const uint64_t K = mLeafComponentAggregateCount;
    if (leafCount == 0 || K == 0) return;

    // rank[s] = inclusive count of self-roots in [0, s], so a root's dense id is rank[root] - 1 and rank[K-1] = N.
    BufT<ComponentLabelT> rankBuf(mStream, this->ref(), K, nanovdb::cuda::noInit);
    auto* d_rank = rankBuf.data();
    static constexpr unsigned int threadsPerBlock = 256;
    util::cuda::lambdaKernel<<<util::cuda::blocksPerGrid(K, threadsPerBlock), threadsPerBlock, 0, mStream>>>(
        K, components::detail::RootFlagFunctor{}, mComponentParent.data(), d_rank);
    cudaCheckError();
    if (mVerbose==1) mTimer.start("Component-rank prefix sum");
    CALL_CUBS(DeviceScan::InclusiveSum, d_rank, d_rank, K);
    cudaCheckError();
    if (mVerbose==1) mTimer.stop();
    cudaCheck(cudaMemcpyAsync(&mGlobalComponentCount, d_rank + (K - 1), sizeof(ComponentLabelT),
                              cudaMemcpyDeviceToHost, mStream));

    using ScatterOp = components::detail::WarpVoxelLabelScatterFunctor<BuildT>;
    if (mVerbose==1) mTimer.start("Voxel-label scatter");
    util::cuda::operatorKernel<ScatterOp>
        <<<util::cuda::blocksPerGrid(leafCount, ScatterOp::LeavesPerBlock), ScatterOp::MaxThreadsPerBlock, 0, mStream>>>(
            mDeviceSrcGrid, mLeafComponentOffsets.data(), mLeafComponentMasks.data(), mComponentParent.data(), d_rank,
            mVoxelLabel.data(), leafCount);
    cudaCheckError();
    if (mVerbose==1) mTimer.stop();
}// ConnectedComponentsOpt<BuildT, ResourceT>::processVoxelLabels

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

} // namespace tools::cuda

} // namespace nanovdb

#ifdef CALL_CUBS
#undef CALL_CUBS
#endif

#endif // NANOVDB_TOOLS_CUDA_CONNECTEDCOMPONENTSOPT_CUH_HAS_BEEN_INCLUDED
