// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/tools/cuda/ConnectedComponents.cuh

    \authors Efty Sifakis and JaeHyun Lee

    \brief Connected-components labeling of NanoVDB indexGrids on the device.

           Identifies connected components of active voxels in a ValueOnIndex grid: two
           active voxels share a component label iff they are connected through a path of
           adjacent active voxels.

    \warning The header file contains cuda device code so be sure
             to only include it in .cu files (or other .cuh files)
*/

#ifndef NVIDIA_TOOLS_CUDA_CONNECTEDCOMPONENTS_CUH_HAS_BEEN_INCLUDED
#define NVIDIA_TOOLS_CUDA_CONNECTEDCOMPONENTS_CUH_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h>
#include <nanovdb/GridHandle.h>
#include <nanovdb/cuda/DeviceBuffer.h>
#include <nanovdb/cuda/TempPool.h>
#include <nanovdb/util/cuda/DeviceGridTraits.cuh>
#include <nanovdb/util/cuda/Timer.h>
#include <nanovdb/util/cuda/Util.h> // for operatorKernel

#include <cub/cub.cuh>

// Define utility macro used to call CUB functions that use dynamic temporary storage.
// Uses mTempDevicePool (a TempDevicePool member) and mStream from the enclosing class.
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

/// @brief Indices into the per-component face-mask array [6] for the six axis-aligned
///        leaf-neighbor directions. Used to unambiguously identify which face of a leaf's
///        connected component is being referenced when testing cross-leaf adjacency.
///
///        Each face is stored as a uint64_t bitmask over the 8×8 boundary plane.
///        The bit-ordering conventions differ by axis (they fall out naturally from the
///        x-major voxel offset layout n = (x<<6)|(y<<3)|z and the extraction algorithm):
///
///          ±X faces (free coords y, z):  bit index = y*8 + z   (y major, z minor)
///          ±Y faces (free coords x, z):  bit index = x*8 + z   (x major, z minor)
///          ±Z faces (free coords x, y):  bit index = y*8 + x   (y major, x minor)
///
///        Cross-leaf connectivity between component I of one leaf and component J of a
///        face-adjacent leaf is detected by a single bitwise AND of the touching faces,
///        e.g. faceMasks[I][plusX] & faceMasks[J][minusX] != 0. The AND is meaningful
///        as long as both sides use the same encoding — which they do, since all masks
///        are produced by the same extraction code.
enum LeafNeighborTap : int {
    minusX = 0,
    plusX  = 1,
    minusY = 2,
    plusY  = 3,
    minusZ = 4,
    plusZ  = 5
};

template <typename BuildT>
class ConnectedComponents
{
    using GridT = NanoGrid<BuildT>;
    using TreeT = NanoTree<BuildT>;
    using RootT = NanoRoot<BuildT>;

public:

    /// @brief Constructor
    /// @param d_srcGrid source device indexGrid whose active voxels are to be labeled
    /// @param stream optional CUDA stream (defaults to CUDA stream 0)
    ConnectedComponents(const GridT* d_srcGrid, cudaStream_t stream = 0)
        : mStream(stream), mTimer(stream), mDeviceSrcGrid(d_srcGrid) {}

    /// @brief Toggle on and off verbose mode
    /// @param level Verbose level: 0=quiet, 1=timing, 2=benchmarking
    void setVerbose(int level = 1) { mVerbose = level; }

    /// @brief Compute per-leaf connected components.
    ///
    ///        Initially this only enumerates, for every leaf node, the number of distinct
    ///        connected components formed by that leaf's active voxels (treating each leaf
    ///        in isolation), storing the result in the per-leaf array mLeafComponentCounts.
    void processLeafConnectedComponents();

    /// @brief Device pointer to the per-leaf component-count array (one uint16_t per leaf),
    ///        valid after processLeafConnectedComponents(). A leaf has at most 256 components
    ///        (8^3 voxels, 6-connected worst case = 3D checkerboard), so uint16_t suffices.
    uint16_t* deviceLeafComponentCounts() { return static_cast<uint16_t*>(mLeafComponentCounts.deviceData()); }

    /// @brief Device pointer to the per-leaf component-offset array (one uint64_t per leaf plus
    ///        one sentinel), valid after processLeafConnectedComponents().
    ///        offsets[i]   = exclusive prefix = start of leaf i's component block.
    ///        offsets[i+1] = inclusive prefix = end   of leaf i's component block.
    ///        offsets[leafCount] = K, the total number of leaf-local components.
    uint64_t* deviceLeafComponentOffsets() { return static_cast<uint64_t*>(mLeafComponentOffsets.deviceData()); }

    /// @brief Device pointer to the flat array of per-component active-voxel masks,
    ///        valid after processLeafConnectedComponents(). Leaf i's components occupy
    ///        slots [offsets[i], offsets[i+1]), so mLeafComponentMasks[offsets[i] + k]
    ///        is the Mask<3> for leaf i's k-th local component.
    nanovdb::Mask<3>* deviceLeafComponentMasks() { return static_cast<nanovdb::Mask<3>*>(mLeafComponentMasks.deviceData()); }

    /// @brief Device pointer to the per-component face masks, valid after
    ///        processLeafConnectedComponents(). Indexed as [compIdx][LeafNeighborTap].
    ///        Cross-leaf connectivity between component I of one leaf and component J of
    ///        a face-adjacent leaf reduces to a single bitwise AND of the touching faces.
    uint64_t (*deviceLeafComponentFaceMasks())[6] {
        return reinterpret_cast<uint64_t(*)[6]>(mLeafComponentFaceMasks.deviceData());
    }

private:

    cudaStream_t                 mStream{0};
    util::cuda::Timer            mTimer;
    int                          mVerbose{0};
    const GridT                 *mDeviceSrcGrid;
    nanovdb::cuda::TempDevicePool mTempDevicePool;

    uint64_t                     mLeafComponentAggregateCount{0}; // total leaf-local components across all leaves (= K = offsets[leafCount])

    nanovdb::cuda::DeviceBuffer  mLeafComponentCounts;      // leafCount                    × uint16_t:      per-leaf component count
    nanovdb::cuda::DeviceBuffer  mLeafComponentOffsets;     // (leafCount+1)                × uint64_t:      exclusive+inclusive prefix sums
    nanovdb::cuda::DeviceBuffer  mLeafComponentMasks;       // mLeafComponentAggregateCount × Mask<3>:       per-component active-voxel footprint
    nanovdb::cuda::DeviceBuffer  mLeafComponentFaceMasks;   // mLeafComponentAggregateCount × uint64_t[6]:   per-component face bitmasks (0=-X,1=+X,2=-Y,3=+Y,4=-Z,5=+Z)

}; // tools::cuda::ConnectedComponents<BuildT>

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace cc_detail {

// Per-leaf connected-components via a Shiloach-Vishkin union-find run in shared memory,
// one CUDA block per leaf, one thread per voxel offset n in [0, 512). The forest is stored
// as a parent array of leaf-local voxel offsets: parent[n] = n for active roots, a smaller
// active offset for non-roots, and -1 for inactive voxels. Connectivity is 6-connected and
// strictly intra-leaf (cross-leaf edges are ignored at this stage).
//
// The three primitives are double-buffered (Jacobi): they read the "cur" buffer and write
// the "nxt" buffer, then swap. Inactive entries (-1) are carried through unchanged. The
// pointer swap is performed identically by every thread, so the register copies stay in sync.

constexpr int LEAF_DIM  = 8;            // NanoLeaf DIM
constexpr int LEAF_SIZE = 512;          // 8^3
constexpr int CC_INACTIVE = -1;         // parent sentinel for inactive voxels

// 3D view over a 512-entry parent buffer. The voxel offset is x-major
// (n = (x<<6)|(y<<3)|z), so a row-major int[8][8][8] indexed [x][y][z] aliases the flat
// buffer exactly: element [x][y][z] sits at linear offset x*64 + y*8 + z == n. Accessing
// individual int elements through this view is well-defined (the storage really is int).
using ParentsT = int[LEAF_DIM][LEAF_DIM][LEAF_DIM];

// Smallest parent among the (up to 6) active in-leaf face neighbors of offset n, floored at
// the supplied current value. Offset layout is x-major: n = (x<<6)|(y<<3)|z.
__device__ inline int ccNeighborMin(const int* parentsPtr, int n, int current)
{
    const auto& p = reinterpret_cast<const ParentsT&>(*parentsPtr);
    const int x =  n >> 6       ;
    const int y = (n >> 3) & 0x7;
    const int z =       n  & 0x7;
    int m = current;
    if (x > 0 && p[x-1][y][z] != CC_INACTIVE) m = ::min(m, p[x-1][y][z]);   // -X
    if (x < 7 && p[x+1][y][z] != CC_INACTIVE) m = ::min(m, p[x+1][y][z]);   // +X
    if (y > 0 && p[x][y-1][z] != CC_INACTIVE) m = ::min(m, p[x][y-1][z]);   // -Y
    if (y < 7 && p[x][y+1][z] != CC_INACTIVE) m = ::min(m, p[x][y+1][z]);   // +Y
    if (z > 0 && p[x][y][z-1] != CC_INACTIVE) m = ::min(m, p[x][y][z-1]);   // -Z
    if (z < 7 && p[x][y][z+1] != CC_INACTIVE) m = ::min(m, p[x][y][z+1]);   // +Z
    return m;
}

// SV root hook: every vertex v whose smallest active neighbor label m is below parent[v]
// lowers the slot of v's *parent* (its tree root, once flattened) toward m, via atomicMin.
// Sets *changed (when non-null) iff some root slot was actually lowered.
__device__ inline void ccHook(int*& cur, int*& nxt, int n, int* changed)
{
    const int pn = cur[n];
    nxt[n] = pn;                                  // Phase A: seed nxt = cur (own slot, no race)
    __syncthreads();
    if (pn != CC_INACTIVE) {                      // active voxel
        const int m = ccNeighborMin(cur, n, pn);
        if (m < pn) {                             // root slot is data-dependent -> atomicMin
            const int old = atomicMin_block(&nxt[pn], m);  // block scope: nxt[] is shared
            if (changed && old > m) *changed = 1;
        }
    }
    __syncthreads();
    int* t = cur; cur = nxt; nxt = t;             // swap (identical on every thread)
}

// Pointer-jumping compress: parent[v] <- parent[parent[v]]. Halves tree depth per call.
// Sets *changed (when non-null) iff some entry actually moved.
__device__ inline void ccCompress(int*& cur, int*& nxt, int n, int* changed)
{
    const int pn = cur[n];
    int v = CC_INACTIVE;
    if (pn != CC_INACTIVE) {                      // active: grandparent (cur[pn] is valid)
        v = cur[pn];
        if (changed && v != pn) *changed = 1;
    }
    nxt[n] = v;                                   // own slot, no race
    __syncthreads();
    int* t = cur; cur = nxt; nxt = t;             // swap
}

template <typename BuildT>
struct LeafComponentCountFunctor
{
    static constexpr int MaxThreadsPerBlock         = LEAF_SIZE;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    // Safety cap on the convergence loop, well above the worst case for an 8^3 leaf
    // (~log2(depth) + log2(#local minima) <= ~18); guards against a non-terminating bug
    // rather than limiting any legitimate input.
    static constexpr int MaxConvergenceIters = 64;

    __device__ void operator()(const NanoGrid<BuildT>* d_grid, uint16_t* d_counts)
    {
        __shared__ int bufA[LEAF_SIZE];
        __shared__ int bufB[LEAF_SIZE];
        __shared__ int changed;
        __shared__ int compCount;

        const int   leafID = blockIdx.x;
        const int   tID    = threadIdx.x;
        const auto& leaf   = d_grid->tree().template getFirstNode<0>()[leafID];

        int* cur = bufA;
        int* nxt = bufB;

        // Init: active voxels label themselves, inactive get the sentinel.
        cur[tID] = leaf.isActive(uint32_t(tID)) ? tID : CC_INACTIVE;
        __syncthreads();

        // Unconditional warm-up: 1 hook + log2(DIM)=3 compresses.
        ccHook    (cur, nxt, tID, nullptr);
        ccCompress(cur, nxt, tID, nullptr);
        ccCompress(cur, nxt, tID, nullptr);
        ccCompress(cur, nxt, tID, nullptr);

        // Then alternate (hook, compress) until a full iteration changes nothing.
        for (int it = 0; it < MaxConvergenceIters; ++it) {
            if (tID == 0) changed = 0;
            __syncthreads();
            ccHook    (cur, nxt, tID, &changed);
            ccCompress(cur, nxt, tID, &changed);
            __syncthreads();
            if (changed == 0) break;
            __syncthreads();  // all threads have read `changed`; safe for thread 0 to reset it next iteration
        }

        // Component count = number of surviving roots (cur[tID] == tID; inactive entries are -1).
        if (tID == 0) compCount = 0;
        __syncthreads();
        if (cur[tID] == tID) atomicAdd_block(&compCount, 1);  // block scope: compCount is shared
        __syncthreads();
        if (tID == 0) d_counts[leafID] = uint16_t(compCount);
    }
}; // LeafComponentCountFunctor

template <typename BuildT>
struct LeafComponentMaskFunctor
{
    static constexpr int MaxThreadsPerBlock         = LEAF_SIZE;
    static constexpr int MinBlocksPerMultiprocessor = 1;
    static constexpr int MaxConvergenceIters        = 64;

    __device__ void operator()(const NanoGrid<BuildT>* d_grid,
                                const uint64_t*         d_offsets,
                                nanovdb::Mask<3>*       d_masks,
                                uint64_t              (*d_faces)[6])
    {
        __shared__ int   bufA[LEAF_SIZE];
        __shared__ int   bufB[LEAF_SIZE];
        __shared__ typename cub::BlockReduce<uint32_t, LEAF_SIZE>::TempStorage reduceTmp;
        __shared__ int      changed;
        __shared__ uint32_t sMinLabel;
        __shared__ union {
            uint32_t sMaskWords_u32[16];  // ballot granularity: one u32 per warp, written by laneID==0
            uint64_t sMaskWords[8];       // Mask<3>::words() granularity: used for face extraction + GMEM write
        };

        const int   leafID = blockIdx.x;
        const int   tID    = threadIdx.x;
        const auto& leaf   = d_grid->tree().template getFirstNode<0>()[leafID];

        int* cur = bufA;
        int* nxt = bufB;

        // Init + SV convergence: identical schedule to LeafComponentCountFunctor.
        cur[tID] = leaf.isActive(uint32_t(tID)) ? tID : CC_INACTIVE;
        __syncthreads();

        ccHook    (cur, nxt, tID, nullptr);
        ccCompress(cur, nxt, tID, nullptr);
        ccCompress(cur, nxt, tID, nullptr);
        ccCompress(cur, nxt, tID, nullptr);

        for (int it = 0; it < MaxConvergenceIters; ++it) {
            if (tID == 0) changed = 0;
            __syncthreads();
            ccHook    (cur, nxt, tID, &changed);
            ccCompress(cur, nxt, tID, &changed);
            __syncthreads();
            if (changed == 0) break;
            __syncthreads();
        }

        // Mask-fill: iterate over leaf-local components in ascending root-label order.
        //
        // Each iteration finds the smallest unprocessed root label via a block-wide unsigned
        // min (CC_INACTIVE = -1 recasts to 0xFFFFFFFF and thus never wins), then collects
        // matching voxels via __ballot_sync and writes the 32-bit result for each warp
        // directly into the Mask<3> word array recast as uint32_t* (each of the 16 warps
        // covers exactly one 32-bit word, so all words are written unconditionally).
        // Processed entries are erased to CC_INACTIVE so they don't win a future min.
        // localCompIdx tracks the dense component index in lockstep across all threads.

        const uint64_t baseOffset = d_offsets[leafID];
        const int warpID = tID >> 5;   // tID / 32
        const int laneID = tID & 31;   // tID % 32

        int localCompIdx = 0;

        while (true) {
            uint32_t minLabel = cub::BlockReduce<uint32_t, LEAF_SIZE>(reduceTmp)
                                    .Reduce(uint32_t(cur[tID]), ::cuda::minimum<uint32_t>{});
            if (tID == 0) sMinLabel = minLabel;
            __syncthreads();
            if (sMinLabel == uint32_t(CC_INACTIVE)) break;

            const bool     match  = (uint32_t(cur[tID]) == sMinLabel);
            const uint32_t ballot = __ballot_sync(0xFFFFFFFF, match);
            if (laneID == 0) sMaskWords_u32[warpID] = ballot;

            if (match) cur[tID] = CC_INACTIVE;
            __syncthreads();  // [SYNC1]: sMaskWords_u32 fully written + cur erases visible

            // Coalesced write: first 8 threads store sMaskWords → mask.words(), one uint64_t each.
            if (tID < 8) {
                d_masks[baseOffset + localCompIdx].words()[tID] = sMaskWords[tID];
            }

            // Extract 6 face masks from sMaskWords (uint64_t view) and store them.
            if (tID == 0) {
                uint64_t* face = d_faces[baseOffset + localCompIdx];

                // ±X: whole word 0 / word 7 are exactly the minusX / plusX face planes.
                face[minusX] = sMaskWords[0];
                face[plusX]  = sMaskWords[7];

                // ±Y: bottom byte (y=0) and top byte (y=7) of each word x;
                //     shift-accumulate from x=7 down; result bit index = x*8 + z (x major, z minor).
                uint64_t mY = sMaskWords[7] & 0xFF,  pY = (sMaskWords[7] >> 56) & 0xFF;
                for (int x = 6; x >= 0; --x) {
                    mY = (mY << 8) | (sMaskWords[x] & 0xFF);
                    pY = (pY << 8) | ((sMaskWords[x] >> 56) & 0xFF);
                }
                face[minusY] = mY;
                face[plusY]  = pY;

                // ±Z: bit 0 (z=0) and bit 7 (z=7) of each byte in each word;
                //     shift-accumulate from x=7 down; result bit index = y*8 + x (y major, x minor).
                uint64_t mZ = sMaskWords[7] & UINT64_C(0x0101010101010101);
                uint64_t pZ = (sMaskWords[7] >> 7) & UINT64_C(0x0101010101010101);
                for (int x = 6; x >= 0; --x) {
                    mZ = (mZ << 1) | (sMaskWords[x] & UINT64_C(0x0101010101010101));
                    pZ = (pZ << 1) | ((sMaskWords[x] >> 7) & UINT64_C(0x0101010101010101));
                }
                face[minusZ] = mZ;
                face[plusZ]  = pZ;
            }
            __syncthreads();  // [SYNC2]: mask + face writes done; all threads safe for next BlockReduce

            ++localCompIdx;
        }
    }
}; // LeafComponentMaskFunctor

} // namespace cc_detail

template <typename BuildT>
void ConnectedComponents<BuildT>::processLeafConnectedComponents()
{
    const uint32_t leafCount =
        util::cuda::DeviceGridTraits<BuildT>::getTreeData(mDeviceSrcGrid).mNodeCount[0];

    // Allocate one component-count per leaf (device-only). At most 256 components per 8^3
    // leaf (6-connected worst case), so uint16_t is sufficient.
    if (mVerbose==1) mTimer.start("Allocating per-leaf component counts");
    mLeafComponentCounts = nanovdb::cuda::DeviceBuffer::create(
        std::size_t(leafCount) * sizeof(uint16_t), nullptr, false);
    if (mVerbose==1) mTimer.stop();

    if (leafCount == 0) return;

    // One block per leaf, one thread per voxel offset; counts the distinct 6-connected
    // components of each leaf's active voxels (in isolation) into mLeafComponentCounts.
    using Op = cc_detail::LeafComponentCountFunctor<BuildT>;
    if (mVerbose==1) mTimer.start("Per-leaf connected-component counting");
    util::cuda::operatorKernel<Op>
        <<<leafCount, Op::MaxThreadsPerBlock, 0, mStream>>>(mDeviceSrcGrid, deviceLeafComponentCounts());
    cudaCheckError();
    if (mVerbose==1) mTimer.stop();

    // Prefix sum: mLeafComponentOffsets[0]=0, mLeafComponentOffsets[1..leafCount] = inclusive sum
    // of mLeafComponentCounts. mLeafComponentOffsets[leafCount] = K (total leaf-local components).
    if (mVerbose==1) mTimer.start("Allocating per-leaf component offsets");
    mLeafComponentOffsets = nanovdb::cuda::DeviceBuffer::create(
        (std::size_t(leafCount) + 1) * sizeof(uint64_t), nullptr, false);
    if (mVerbose==1) mTimer.stop();

    cudaCheck(cudaMemsetAsync(mLeafComponentOffsets.deviceData(), 0, sizeof(uint64_t), mStream));

    // Upcast per-leaf uint16_t counts into offsets[1..leafCount] as uint64_t.
    uint16_t* d_counts  = deviceLeafComponentCounts();
    uint64_t* d_offsets = deviceLeafComponentOffsets();
    util::cuda::lambdaKernel<<<(leafCount + 255) / 256, 256, 0, mStream>>>(
        leafCount,
        [] __device__(size_t i, const uint16_t* counts, uint64_t* offsets) {
            offsets[i + 1] = uint64_t(counts[i]);
        },
        d_counts, d_offsets);
    cudaCheckError();

    // In-place inclusive sum over offsets[1..leafCount]; offsets[leafCount] = K (total components).
    if (mVerbose==1) mTimer.start("Per-leaf component offset prefix sum");
    CALL_CUBS(DeviceScan::InclusiveSum, d_offsets + 1, d_offsets + 1, int(leafCount));
    cudaCheckError();
    if (mVerbose==1) mTimer.stop();

    // Read the aggregate count from the sentinel at offsets[leafCount].
    // Use async copy on mStream (ordered after the scan) + stream sync to avoid
    // inadvertently synchronizing other streams.
    cudaCheck(cudaMemcpyAsync(&mLeafComponentAggregateCount, d_offsets + leafCount,
                              sizeof(uint64_t), cudaMemcpyDeviceToHost, mStream));
    cudaCheck(cudaStreamSynchronize(mStream));

    // Allocate mLeafComponentAggregateCount Mask<3> objects — one per leaf-local component.
    // No zero-init needed: the mask-fill kernel writes all 16 uint32_t words of every mask
    // unconditionally via warp ballot (zero ballot = no component voxels in that warp).
    if (mVerbose==1) mTimer.start("Allocating per-component leaf masks");
    mLeafComponentMasks = nanovdb::cuda::DeviceBuffer::create(
        mLeafComponentAggregateCount * sizeof(nanovdb::Mask<3>), nullptr, false);
    if (mVerbose==1) mTimer.stop();

    // Allocate 6 uint64_t face masks per component (one per LeafNeighborTap entry).
    // The face-extraction kernel fills these; no zero-init needed for the same reason.
    if (mVerbose==1) mTimer.start("Allocating per-component face masks");
    mLeafComponentFaceMasks = nanovdb::cuda::DeviceBuffer::create(
        mLeafComponentAggregateCount * 6 * sizeof(uint64_t), nullptr, false);
    if (mVerbose==1) mTimer.stop();

    // Re-run SV per leaf and scatter each active voxel's bit into its component's Mask<3>.
    using MaskOp = cc_detail::LeafComponentMaskFunctor<BuildT>;
    if (mVerbose==1) mTimer.start("Per-leaf component mask fill");
    util::cuda::operatorKernel<MaskOp>
        <<<leafCount, MaskOp::MaxThreadsPerBlock, 0, mStream>>>(
            mDeviceSrcGrid, deviceLeafComponentOffsets(),
            deviceLeafComponentMasks(), deviceLeafComponentFaceMasks());
    cudaCheckError();
    if (mVerbose==1) mTimer.stop();
}// ConnectedComponents<BuildT>::processLeafConnectedComponents

} // namespace tools::cuda

} // namespace nanovdb

#endif // NVIDIA_TOOLS_CUDA_CONNECTEDCOMPONENTS_CUH_HAS_BEEN_INCLUDED
