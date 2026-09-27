// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/tools/cuda/PropagateSign.cuh

    \authors JaeHyun Lee and Efty Sifakis

    \brief Propagates active-voxel signs into the inactive regions of a NanoVDB index grid.

    \warning This header contains CUDA device code and must be included from a .cu or .cuh file.
*/

#ifndef NVIDIA_TOOLS_CUDA_PROPAGATESIGN_CUH_HAS_BEEN_INCLUDED
#define NVIDIA_TOOLS_CUDA_PROPAGATESIGN_CUH_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h>
#include <nanovdb/cuda/Buffer.h>
#include <nanovdb/cuda/DeviceResource.h>
#include <nanovdb/util/cuda/DeviceGridTraits.cuh>
#include <nanovdb/util/cuda/Timer.h>
#include <nanovdb/util/cuda/Util.h>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace nanovdb {

namespace tools::cuda {

/// @brief Propagate the signs of active voxels through every inactive level of an index grid.
///
/// TODO: Explain the per-level interior masks.
/// TODO:  explain the difference between SingedFloodFill.cuh and why this operator is needed.
/// todo: 6 face seeds, jacobi fill
/// todo: why we are using jacobi style double buffered flood fill? not scanline?
/// @tparam BuildT NanoVDB index-grid build type.
/// @tparam ResourceT Stream-ordered resource used for all output and scratch allocations.
template <typename BuildT, typename ResourceT = nanovdb::cuda::DeviceResource>
class PropagateSign
{
    static_assert(nanovdb::cuda::is_async_resource<ResourceT>::value,
                  "PropagateSign allocates stream-ordered storage and requires an AsyncResource");
    static_assert(ResourceT::DEFAULT_ALIGNMENT >= alignof(uint64_t),
                  "PropagateSign stores uint64_t masks and requires word-aligned allocations");

    using GridT = NanoGrid<BuildT>;

    template <typename T>
    using BufferT = nanovdb::cuda::Buffer<T, nanovdb::cuda::ResourceRef<ResourceT>>;

public:
    /// @param deviceGrid Device index grid whose active values index @a deviceSigns.
    /// @param deviceSigns Per-active-value signs: -1 inside and +1 outside.
    /// @param resource Allocation resource; must outlive the returned buffers.
    /// @note @a deviceGrid and @a deviceSigns must remain valid until propagate() returns.
    PropagateSign(const GridT* deviceGrid, const int8_t* deviceSigns,
                  cudaStream_t stream = 0,
                  ResourceT& resource = nanovdb::cuda::default_resource<ResourceT>())
        : mStream(stream)
        , mTimer(stream)
        , mDeviceGrid(deviceGrid)
        , mDeviceSigns(deviceSigns)
        , mResource(&resource)
        , mLeafInteriorMasks(stream, resource, 0, nanovdb::cuda::noInit)
        , mLowerInteriorMasks(stream, resource, 0, nanovdb::cuda::noInit)
        , mUpperInteriorMasks(stream, resource, 0, nanovdb::cuda::noInit)
        , mRootInteriorTileOrigins(stream, resource, 0, nanovdb::cuda::noInit)
        , mRootMask(stream, resource, 0, nanovdb::cuda::noInit)
    {}

    /// @brief Toggle timing output.
    void setVerbose(int level = 1) { mVerbose = level; }

    /// @brief Propagate interior (-1) sign beyond the active voxels
    /// @note The stream is synchronized before this method returns.
    void propagate();

    /// @brief Transfer ownership of an output buffer. Valid after propagate().
    /// @note Leaf masks include active interior voxels.
    BufferT<nanovdb::Mask<3>> getLeafInteriorMasks() { return std::move(mLeafInteriorMasks); }
    BufferT<nanovdb::Mask<4>> getLowerInteriorMasks() { return std::move(mLowerInteriorMasks); }
    BufferT<nanovdb::Mask<5>> getUpperInteriorMasks() { return std::move(mUpperInteriorMasks); }
    BufferT<nanovdb::Coord> getRootInteriorTileOrigins() { return std::move(mRootInteriorTileOrigins); }

private:
    bool initializeMasks();

    void fillLeafInteriorMasks();

    template <int Level>
    void emitSeedsToCoarserLevels();

    void fillLowerInteriorMasks();

    void fillUpperInteriorMasks();

    void fillRootInteriorMasks();

    nanovdb::cuda::ResourceRef<ResourceT> ref() { return nanovdb::cuda::ResourceRef<ResourceT>(*mResource); }

    cudaStream_t       mStream{0};
    util::cuda::Timer  mTimer;
    int                mVerbose{0};
    const GridT*       mDeviceGrid{nullptr};
    const int8_t*      mDeviceSigns{nullptr};
    ResourceT*         mResource{nullptr};

    BufferT<nanovdb::Mask<3>> mLeafInteriorMasks;
    BufferT<nanovdb::Mask<4>> mLowerInteriorMasks;
    BufferT<nanovdb::Mask<5>> mUpperInteriorMasks;
    BufferT<nanovdb::Coord>   mRootInteriorTileOrigins;
    BufferT<uint8_t>           mRootMask; // One byte per cell because root bounds vary.
    nanovdb::Coord             mRootTileMin{0};
    nanovdb::Coord             mRootTileDims{0};
}; // tools::cuda::PropagateSign<BuildT, ResourceT>

//----------------------------------------------------------------------------------------------------------------------

template <typename BuildT, typename ResourceT>
void PropagateSign<BuildT, ResourceT>::propagate()
{
    if (!initializeMasks()) { // no active voxels
        cudaCheck(cudaStreamSynchronize(mStream));
        return;
    }

    // Root propagation is needed when the active bounds contain positions without an Upper child.
    const bool needsRootPropagation = mRootMask.size() > mUpperInteriorMasks.size();

    if (mVerbose == 1) mTimer.start("Filling leaf interior masks");
    fillLeafInteriorMasks();
    if (mVerbose == 1) mTimer.stop();

    if (mVerbose == 1) mTimer.start("Emitting seeds from leaf to lower level");
    emitSeedsToCoarserLevels<0>();
    if (mVerbose == 1) mTimer.stop();

    if (mVerbose == 1) mTimer.start("Filling lower interior masks");
    fillLowerInteriorMasks();
    if (mVerbose == 1) mTimer.stop();

    if (mVerbose == 1) mTimer.start("Emitting seeds from lower to upper level");
    emitSeedsToCoarserLevels<1>();
    if (mVerbose == 1) mTimer.stop();

    if (mVerbose == 1) mTimer.start("Filling upper interior masks");
    fillUpperInteriorMasks();
    if (mVerbose == 1) mTimer.stop();

    if (needsRootPropagation) {
        if (mVerbose == 1) mTimer.start("Emitting seed from upper to root level");
        emitSeedsToCoarserLevels<2>();
        if (mVerbose == 1) mTimer.stop();

        if (mVerbose == 1) mTimer.start("Filling root interior masks");
        fillRootInteriorMasks();
        if (mVerbose == 1) mTimer.stop();
    }

    cudaCheck(cudaStreamSynchronize(mStream));
}

template <typename BuildT, typename ResourceT>
bool PropagateSign<BuildT, ResourceT>::initializeMasks()
{
    const auto treeData = util::cuda::DeviceGridTraits<BuildT>::getTreeData(mDeviceGrid);
    const auto& [leafCount, lowerCount, upperCount] = treeData.mNodeCount;
    if (leafCount == 0) return false;

    mLeafInteriorMasks = BufferT<nanovdb::Mask<3>>(mStream, this->ref(), leafCount, nanovdb::cuda::noInit);
    mLowerInteriorMasks = BufferT<nanovdb::Mask<4>>(mStream, this->ref(), lowerCount, nanovdb::cuda::noInit);
    mUpperInteriorMasks = BufferT<nanovdb::Mask<5>>(mStream, this->ref(), upperCount, nanovdb::cuda::noInit);
    cudaCheck(cudaMemsetAsync(mLowerInteriorMasks.data(), 0, mLowerInteriorMasks.size_bytes(), mStream));
    cudaCheck(cudaMemsetAsync(mUpperInteriorMasks.data(), 0, mUpperInteriorMasks.size_bytes(), mStream));

    // Convert active index bounds to root-tile coordinates and zero one byte per position, including absent tiles.
    const auto activeIndexBBox = util::cuda::DeviceGridTraits<BuildT>::getIndexBBox(mDeviceGrid, treeData);
    constexpr int RootTileShift = NanoUpper<BuildT>::TOTAL;
    const nanovdb::CoordBBox rootTileBounds(activeIndexBBox.min() >> RootTileShift,
                                           activeIndexBBox.max() >> RootTileShift);
    mRootTileMin = rootTileBounds.min();
    mRootTileDims = rootTileBounds.dim();
    const std::size_t rootCellCount = rootTileBounds.volume();
    mRootMask = BufferT<uint8_t>(mStream, this->ref(), rootCellCount, nanovdb::cuda::noInit);
    cudaCheck(cudaMemsetAsync(mRootMask.data(), 0, mRootMask.size_bytes(), mStream));
    return true;
}

//----------------------------------------------------------------------------------------------------------------------

namespace signing::detail {

template <typename BuildT>
struct LeafPropagationFunctor
{
    static constexpr int MaxThreadsPerBlock         = NanoLeaf<BuildT>::SIZE;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    // face neighbor
    __device__ static bool hasInteriorNeighbor(const uint8_t* interior, const nanovdb::Coord& local)
    {
        const auto& inside = reinterpret_cast<const uint8_t(&)[8][8][8]>(*interior);
        const int x = local[0], y = local[1], z = local[2];
        return (x > 0 && inside[x-1][y][z]) || (x < 7 && inside[x+1][y][z]) ||
               (y > 0 && inside[x][y-1][z]) || (y < 7 && inside[x][y+1][z]) ||
               (z > 0 && inside[x][y][z-1]) || (z < 7 && inside[x][y][z+1]);
    }

    __device__ void operator()(const NanoGrid<BuildT>* grid, const int8_t* signs,
                               nanovdb::Mask<3>* masks)
    {
        const int leafID = blockIdx.x;
        const int tID = threadIdx.x;
        const auto& leaf = grid->tree().template getFirstNode<0>()[leafID];

        __shared__ uint8_t interior[NanoLeaf<BuildT>::SIZE];
        __shared__ int changed;

        const bool isActive = leaf.isActive(uint32_t(tID));

        // initial seeds
        interior[tID] = isActive && (signs[leaf.getValue(uint32_t(tID))] == int8_t(-1));
        __syncthreads();

        const nanovdb::Coord local = NanoLeaf<BuildT>::OffsetToLocalCoord(tID);
        //todo: why 64?
        for (int iteration = 0; iteration < 64; ++iteration) {
            if (tID == 0) changed = 0;
            __syncthreads();

            bool turnOn = false;
            if (!isActive && !interior[tID]) turnOn = hasInteriorNeighbor(interior, local);

            __syncthreads(); // finish this sweep's reads before updating the shared mask
            if (turnOn) {
                interior[tID] = 1;
                changed = 1; // benign race: every writer stores 1
            }
            __syncthreads();
            const bool done = changed == 0;
            __syncthreads(); // all threads read changed before the next reset
            if (done) break;
        }

        // Thread tID owns bit tID of the leaf mask, so each warp's ballot is exactly the mask's
        // 32-bit word (tID >> 5); warps write disjoint words, so no atomics are needed.
        const uint32_t ballot = __ballot_sync(0xFFFFFFFFu, interior[tID]);
        if ((tID & 31) == 0) reinterpret_cast<uint32_t*>(masks[leafID].words())[tID >> 5] = ballot;
    }
}; // LeafPropagationFunctor

} // namespace signing::detail

template <typename BuildT, typename ResourceT>
void PropagateSign<BuildT, ResourceT>::fillLeafInteriorMasks()
{
    using Op = signing::detail::LeafPropagationFunctor<BuildT>;
    util::cuda::operatorKernel<Op><<<unsigned(mLeafInteriorMasks.size()), Op::MaxThreadsPerBlock, 0, mStream>>>(
        mDeviceGrid, mDeviceSigns, mLeafInteriorMasks.data());
    cudaCheckError();
}

//----------------------------------------------------------------------------------------------------------------------

namespace signing::detail {

template <typename BuildT, int Level>
struct FaceSeedFunctor
{   //todo: scatter (push) vs gather (pull)?
    static_assert(Level >= 0 && Level <= 2, "FaceSeedFunctor requires a tree node level");

    using NodeT = typename NanoNode<BuildT, Level>::type;
    static constexpr int Log2Dim = NodeT::LOG2DIM;
    static constexpr int NodeDim = NodeT::DIM;
    static constexpr int FaceSize = 1 << (2 * Log2Dim);
    //6 faces × (8x8) face voxels = 384 threads for a leaf, each thread handles each face voxel
    static constexpr int MaxThreadsPerBlock = Level == 0 ? 6 * FaceSize : 512;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ static uint32_t faceOffset(int face, int a, int b)
    {
        constexpr int Max = (1 << Log2Dim) - 1;
        //TODO: consider change this face to enum.
        // todo: lets consider make an alternative helper such as LeafFaceVoxelIndices which enumerates all the face voxels
        switch (face) {
        case 0:  return (0   << (2 * Log2Dim)) | (a << Log2Dim) | b;
        case 1:  return (Max << (2 * Log2Dim)) | (a << Log2Dim) | b;
        case 2:  return (a   << (2 * Log2Dim)) | (0 << Log2Dim) | b;
        case 3:  return (a   << (2 * Log2Dim)) | (Max << Log2Dim) | b;
        case 4:  return (a   << (2 * Log2Dim)) | (b << Log2Dim);
        default: return (a   << (2 * Log2Dim)) | (b << Log2Dim) | Max;
        }
    }

    __device__ static nanovdb::Coord faceNeighbor(nanovdb::Coord origin, int face, int distance)
    {
        switch (face) {
        case 0:  return origin.offsetBy(-distance, 0, 0);
        case 1:  return origin.offsetBy( distance, 0, 0);
        case 2:  return origin.offsetBy(0, -distance, 0);
        case 3:  return origin.offsetBy(0,  distance, 0);
        case 4:  return origin.offsetBy(0, 0, -distance);
        default: return origin.offsetBy(0, 0,  distance);
        }
    }

    /// @return 0 for a root value tile or leaf, 1 for lower, 2 for upper, and 3 for absent root space.
    __hostdev__ static int probeChildlessSlot(const NanoGrid<BuildT>& grid, const nanovdb::Coord& ijk,
                                              uint64_t& nodeID, uint32_t& slot) //todo: is there any better name for this?
    {
        using UpperT = NanoUpper<BuildT>;
        using LowerT = NanoLower<BuildT>;

        const auto& tree = grid.tree();
        const auto* tile = tree.root().probeTile(ijk);
        if (!tile) return 3;
        if (!tile->isChild()) return 0;

        const UpperT* upper = tree.root().getChild(tile);
        const LowerT* lower = upper->probeChild(ijk);
        if (!lower) {
            nodeID = util::PtrDiff(upper, tree.template getFirstNode<2>()) / sizeof(UpperT);
            slot = UpperT::CoordToOffset(ijk);
            return 2;
        }

        slot = LowerT::CoordToOffset(ijk);
        if (!lower->childMask().isOn(slot)) {
            nodeID = util::PtrDiff(lower, tree.template getFirstNode<1>()) / sizeof(LowerT);
            return 1;
        }
        return 0;
    }

    __device__ void operator()(const NanoGrid<BuildT>* grid, const nanovdb::Mask<3>* leafMasks,
                               nanovdb::Mask<4>* lowerMasks,
                               nanovdb::Mask<5>* upperMasks,
                               uint8_t* rootMask,
                               nanovdb::Coord tileMin, nanovdb::Coord tileDims)
    {
        const int nodeID = blockIdx.x;
        const int tID = threadIdx.x;
        const auto& node = grid->tree().template getFirstNode<Level>()[nodeID];

        __shared__ int faceHasInside[6];
        if (tID < 6) faceHasInside[tID] = 0;
        __syncthreads();

        for (int i = tID; i < 6 * FaceSize; i += blockDim.x) {
            const int face = i >> (2 * Log2Dim);
            const int a = (i >> Log2Dim) & ((1 << Log2Dim) - 1);
            const int b = i & ((1 << Log2Dim) - 1);
            const uint32_t n = faceOffset(face, a, b);

            bool isInside;
            if constexpr (Level == 0) {
                isInside = leafMasks[nodeID].isOn(n);
            } else if constexpr (Level == 1) {
                if (node.childMask().isOn(n)) continue;
                isInside = lowerMasks[nodeID].isOn(n);
            } else {
                if (node.childMask().isOn(n)) continue;
                isInside = upperMasks[nodeID].isOn(n);
            }

            if (isInside) faceHasInside[face] = 1; // all writers store 1
        }
        __syncthreads();

        if (tID < 6) {
            if (!faceHasInside[tID]) return;
            const nanovdb::Coord ijk = faceNeighbor(node.origin(), tID, NodeDim);
            uint64_t targetNodeID;
            uint32_t slot;
            const int level = probeChildlessSlot(*grid, ijk, targetNodeID, slot);
            if (level <= Level) return; // only a coarser slot receives a seed

            if (level == 1) {
                lowerMasks[targetNodeID].setOnAtomic(slot);
            } else if (level == 2) {
                upperMasks[targetNodeID].setOnAtomic(slot);
            } else {
                const int i = (ijk[0] >> NanoUpper<BuildT>::TOTAL) - tileMin[0];
                const int j = (ijk[1] >> NanoUpper<BuildT>::TOTAL) - tileMin[1];
                const int k = (ijk[2] >> NanoUpper<BuildT>::TOTAL) - tileMin[2];
                if (i >= 0 && i < tileDims[0] &&
                    j >= 0 && j < tileDims[1] &&
                    k >= 0 && k < tileDims[2]) {
                    const int index = (i * tileDims[1] + j) * tileDims[2] + k;
                    rootMask[index] = 1;
                }
            }
        }
    }

}; // FaceSeedFunctor

} // namespace signing::detail

template <typename BuildT, typename ResourceT>
template <int Level>
void PropagateSign<BuildT, ResourceT>::emitSeedsToCoarserLevels()
{
    const std::size_t nodeCount = Level == 0 ? mLeafInteriorMasks.size()
                                : Level == 1 ? mLowerInteriorMasks.size()
                                             : mUpperInteriorMasks.size();
    using Op = signing::detail::FaceSeedFunctor<BuildT, Level>;
    util::cuda::operatorKernel<Op><<<unsigned(nodeCount), Op::MaxThreadsPerBlock, 0, mStream>>>(
        mDeviceGrid, mLeafInteriorMasks.data(), mLowerInteriorMasks.data(), mUpperInteriorMasks.data(),
        mRootMask.data(), mRootTileMin, mRootTileDims);
    cudaCheckError();
}

//----------------------------------------------------------------------------------------------------------------------

namespace signing::detail {

template <typename BuildT, int Level>
struct InternalPropagationFunctor
{
    static_assert(Level == 1 || Level == 2, "InternalPropagationFunctor requires a lower or upper node");

    static constexpr int Log2Dim = Level == 1 ? NanoLower<BuildT>::LOG2DIM : NanoUpper<BuildT>::LOG2DIM;
    static constexpr int Dim     = 1 << Log2Dim;
    static constexpr int SlotCount = 1 << (3 * Log2Dim);
    static constexpr int WordCount = SlotCount >> 6;
    using MaskT = nanovdb::Mask<Log2Dim>;

    static constexpr int MaxThreadsPerBlock         = 512;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(const NanoGrid<BuildT>* grid, MaskT* masks)
    {
        const int nodeID = blockIdx.x;
        const int tID = threadIdx.x;
        const auto& node = grid->tree().template getFirstNode<Level>()[nodeID];

        // A __shared__ variable cannot run Mask's zero-filling constructor, so the masks live in
        // raw shared words that are viewed as MaskT.
        __shared__ uint64_t wallWords[WordCount];
        __shared__ uint64_t interiorWords[WordCount];
        __shared__ int changed;
        const auto& walls = reinterpret_cast<const MaskT&>(wallWords);
        auto& interior = reinterpret_cast<MaskT&>(interiorWords);

        for (int word = tID; word < WordCount; word += blockDim.x) {
            wallWords[word] = node.childMask().words()[word];
            interiorWords[word] = masks[nodeID].words()[word] & ~wallWords[word];
        }
        __syncthreads();

        // Once a bit turns on it is final, so same-sweep reads may only accelerate the flood.
        for (int iteration = 0; iteration < 3 * Dim + 16; ++iteration) {
            if (tID == 0) changed = 0;
            __syncthreads();

            bool any = false;
            for (int n = tID; n < SlotCount; n += blockDim.x) {
                if (walls.isOn(n) || interior.isOn(n)) continue;
                const nanovdb::Coord local = node.OffsetToLocalCoord(uint32_t(n));
                constexpr int dx = 1 << (2 * Log2Dim);
                constexpr int dy = 1 << Log2Dim;
                if ((local[0] > 0       && interior.isOn(n - dx)) ||
                    (local[0] < Dim - 1 && interior.isOn(n + dx)) ||
                    (local[1] > 0       && interior.isOn(n - dy)) ||
                    (local[1] < Dim - 1 && interior.isOn(n + dy)) ||
                    (local[2] > 0       && interior.isOn(n - 1))  ||
                    (local[2] < Dim - 1 && interior.isOn(n + 1))) {
                    interior.setOnAtomic(n);
                    any = true;
                }
            }
            if (any) changed = 1;
            __syncthreads();
            const bool done = changed == 0;
            __syncthreads(); // all threads read changed before the next reset
            if (done) break;
        }

        for (int word = tID; word < WordCount; word += blockDim.x)
            masks[nodeID].words()[word] = interiorWords[word];
    }
}; // InternalPropagationFunctor

} // namespace signing::detail

template <typename BuildT, typename ResourceT>
void PropagateSign<BuildT, ResourceT>::fillLowerInteriorMasks()
{
    using Op = signing::detail::InternalPropagationFunctor<BuildT, 1>;
    util::cuda::operatorKernel<Op><<<unsigned(mLowerInteriorMasks.size()), Op::MaxThreadsPerBlock, 0, mStream>>>(
        mDeviceGrid, mLowerInteriorMasks.data());
    cudaCheckError();
}

template <typename BuildT, typename ResourceT>
void PropagateSign<BuildT, ResourceT>::fillUpperInteriorMasks()
{
    using Op = signing::detail::InternalPropagationFunctor<BuildT, 2>;
    util::cuda::operatorKernel<Op><<<unsigned(mUpperInteriorMasks.size()), Op::MaxThreadsPerBlock, 0, mStream>>>(
        mDeviceGrid, mUpperInteriorMasks.data());
    cudaCheckError();
}

//----------------------------------------------------------------------------------------------------------------------

namespace signing::detail {

template <typename BuildT>
struct RootWallFunctor
{
    const NanoGrid<BuildT>* grid;
    uint8_t* walls;
    nanovdb::Coord tileMin;
    nanovdb::Coord tileDims;

    __device__ void operator()(size_t n) const
    {
        const int k = int(n) % tileDims[2];
        const int j = (int(n) / tileDims[2]) % tileDims[1];
        const int i = int(n) / (tileDims[1] * tileDims[2]);
        const nanovdb::Coord ijk((tileMin[0] + i) << NanoUpper<BuildT>::TOTAL,
                                 (tileMin[1] + j) << NanoUpper<BuildT>::TOTAL,
                                 (tileMin[2] + k) << NanoUpper<BuildT>::TOTAL);
        const auto* tile = grid->tree().root().probeTile(ijk);
        walls[n] = tile && tile->isChild();
    }
}; // RootWallFunctor

struct RootPropagationFunctor
{
    static constexpr int MaxThreadsPerBlock         = 256;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(const uint8_t* walls, uint8_t* mask,
                               nanovdb::Coord tileDims)
    {
        const int tID = threadIdx.x;
        const int width = tileDims[0];
        const int height = tileDims[1];
        const int depth = tileDims[2];
        const int total = width * height * depth;
        __shared__ int changed;

        for (int n = tID; n < total; n += blockDim.x)
            mask[n] = !walls[n] && mask[n];
        __syncthreads();

        // Every simple path contains fewer than total cells; the bound is a convergence backstop.
        for (int iteration = 0; iteration < total + 2; ++iteration) {
            if (tID == 0) changed = 0;
            __syncthreads();

            bool any = false;
            for (int n = tID; n < total; n += blockDim.x) {
                if (walls[n] || mask[n]) continue;
                const int k = n % depth;
                const int j = (n / depth) % height;
                const int i = n / (height * depth);
                const bool turnOn =
                    (i > 0          && mask[n - height * depth]) ||
                    (i < width - 1  && mask[n + height * depth]) ||
                    (j > 0          && mask[n - depth]) ||
                    (j < height - 1 && mask[n + depth]) ||
                    (k > 0          && mask[n - 1]) ||
                    (k < depth - 1  && mask[n + 1]);
                if (turnOn) {
                    mask[n] = 1;
                    any = true;
                }
            }
            if (any) changed = 1; // benign race: every writer stores 1
            __syncthreads();
            const bool done = changed == 0;
            __syncthreads(); // all threads read changed before the next reset
            if (done) return;
        }
        NANOVDB_ASSERT(false);
    }
}; // RootPropagationFunctor

} // namespace signing::detail

template <typename BuildT, typename ResourceT>
void PropagateSign<BuildT, ResourceT>::fillRootInteriorMasks()
{
    const std::size_t rootCellCount = mRootMask.size();
    BufferT<uint8_t> walls(mStream, this->ref(), rootCellCount, nanovdb::cuda::noInit);
    cudaCheck(cudaMemsetAsync(walls.data(), 0, walls.size_bytes(), mStream));

    constexpr unsigned int WallThreads = 128;
    util::cuda::lambdaKernel
        <<<unsigned((rootCellCount + WallThreads - 1) / WallThreads), WallThreads, 0, mStream>>>(
            rootCellCount,
            signing::detail::RootWallFunctor<BuildT>{mDeviceGrid, walls.data(),
                                                      mRootTileMin, mRootTileDims});
    cudaCheckError();

    using RootPropagationOp = signing::detail::RootPropagationFunctor;
    util::cuda::operatorKernel<RootPropagationOp>
        <<<1, RootPropagationOp::MaxThreadsPerBlock, 0, mStream>>>(
        walls.data(), mRootMask.data(), mRootTileDims);
    cudaCheckError();

    std::vector<uint8_t> interior(rootCellCount);
    cudaCheck(cudaMemcpyAsync(interior.data(), mRootMask.data(), rootCellCount, cudaMemcpyDeviceToHost, mStream));
    cudaCheck(cudaStreamSynchronize(mStream));

    // Root tile happens rarely, so this could be host side logic
    std::vector<nanovdb::Coord> origins;
    for (std::size_t n = 0; n < rootCellCount; ++n) {
        if (!interior[n]) continue;
        const int k = int(n % mRootTileDims[2]);
        const int j = int(n / mRootTileDims[2] % mRootTileDims[1]);
        const int i = int(n / (std::size_t(mRootTileDims[1]) * mRootTileDims[2]));
        origins.push_back((mRootTileMin + nanovdb::Coord(i, j, k)) << NanoUpper<BuildT>::TOTAL);
    }
    mRootInteriorTileOrigins = BufferT<nanovdb::Coord>(mStream, this->ref(), origins.size(), nanovdb::cuda::noInit);
    cudaCheck(cudaMemcpyAsync(mRootInteriorTileOrigins.data(), origins.data(), mRootInteriorTileOrigins.size_bytes(),
                              cudaMemcpyHostToDevice, mStream));
}

} // namespace tools::cuda

} // namespace nanovdb

#endif // NVIDIA_TOOLS_CUDA_PROPAGATESIGN_CUH_HAS_BEEN_INCLUDED
