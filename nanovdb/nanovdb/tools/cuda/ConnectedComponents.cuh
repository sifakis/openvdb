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
#include <nanovdb/util/cuda/DeviceGridTraits.cuh>
#include <nanovdb/util/cuda/Timer.h>
#include <nanovdb/util/cuda/Util.h> // for operatorKernel

namespace nanovdb {

namespace tools::cuda {

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

private:

    cudaStream_t                 mStream{0};
    util::cuda::Timer            mTimer;
    int                          mVerbose{0};
    const GridT                 *mDeviceSrcGrid;

    nanovdb::cuda::DeviceBuffer  mLeafComponentCounts; // one uint16_t component count per leaf

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
        const int   n      = threadIdx.x;
        const auto& leaf   = d_grid->tree().template getFirstNode<0>()[leafID];

        int* cur = bufA;
        int* nxt = bufB;

        // Init: active voxels label themselves, inactive get the sentinel.
        cur[n] = leaf.isActive(uint32_t(n)) ? n : CC_INACTIVE;
        __syncthreads();

        // Unconditional warm-up: 1 hook + log2(DIM)=3 compresses.
        ccHook    (cur, nxt, n, nullptr);
        ccCompress(cur, nxt, n, nullptr);
        ccCompress(cur, nxt, n, nullptr);
        ccCompress(cur, nxt, n, nullptr);

        // Then alternate (hook, compress) until a full iteration changes nothing.
        for (int it = 0; it < MaxConvergenceIters; ++it) {
            if (n == 0) changed = 0;
            __syncthreads();
            ccHook    (cur, nxt, n, &changed);
            ccCompress(cur, nxt, n, &changed);
            __syncthreads();
            if (changed == 0) break;
            __syncthreads();  // all threads have read `changed`; safe for thread 0 to reset it next iteration
        }

        // Component count = number of surviving roots (cur[n] == n; inactive entries are -1).
        if (n == 0) compCount = 0;
        __syncthreads();
        if (cur[n] == n) atomicAdd_block(&compCount, 1);  // block scope: compCount is shared
        __syncthreads();
        if (n == 0) d_counts[leafID] = uint16_t(compCount);
    }
}; // LeafComponentCountFunctor

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
}// ConnectedComponents<BuildT>::processLeafConnectedComponents

} // namespace tools::cuda

} // namespace nanovdb

#endif // NVIDIA_TOOLS_CUDA_CONNECTEDCOMPONENTS_CUH_HAS_BEEN_INCLUDED
