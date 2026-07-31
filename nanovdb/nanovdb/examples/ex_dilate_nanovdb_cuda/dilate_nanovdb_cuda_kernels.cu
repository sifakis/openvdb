// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <nanovdb/tools/cuda/DilateGrid.cuh>
#include <nanovdb/tools/cuda/PruneGrid.cuh>
#include <nanovdb/tools/cuda/PointsToGrid.cuh>
#include <nanovdb/tools/cuda/VoxelBlockManager.cuh>
#include <nanovdb/util/cuda/Injection.cuh>

#include <cub/device/device_scan.cuh>

#include <cstdlib>
#include <vector>

template<typename T>
bool bufferCheck(const T* deviceBuffer, const T* hostBuffer, size_t elem_count) {
    T* tmpBuffer = new T[elem_count];
    cudaCheck(cudaMemcpy(tmpBuffer, deviceBuffer, elem_count * sizeof(T), cudaMemcpyDeviceToHost));
    bool same = true;
    for (int i=0; same && i< elem_count; ++i) { same = (tmpBuffer[i] == hostBuffer[i]); }
    delete [] tmpBuffer;
    return same;
}

/// @brief Report where a computed grid diverges from a reference grid. Only invoked when the
///        byte-wise comparison has already failed, to make the nature of the mismatch apparent
///        (topological difference vs. metadata/padding difference).
template<typename GridT>
void reportGridDiff(const GridT* reference, const GridT* computed)
{
    std::cout << "  [diff] gridSize   ref=" << reference->gridSize()         << " got=" << computed->gridSize()         << std::endl;
    std::cout << "  [diff] activeVox  ref=" << reference->activeVoxelCount() << " got=" << computed->activeVoxelCount() << std::endl;
    std::cout << "  [diff] valueCount ref=" << reference->valueCount()       << " got=" << computed->valueCount()       << std::endl;
    for (int l = 0; l < 3; ++l)
        std::cout << "  [diff] nodeCount[" << l << "] ref=" << reference->tree().nodeCount(l)
                  << " got=" << computed->tree().nodeCount(l) << std::endl;
    auto rbb = reference->indexBBox(), gbb = computed->indexBBox();
    std::cout << "  [diff] indexBBox  ref=[" << rbb.min()[0] << "," << rbb.min()[1] << "," << rbb.min()[2] << "]->["
              << rbb.max()[0] << "," << rbb.max()[1] << "," << rbb.max()[2] << "]  got=["
              << gbb.min()[0] << "," << gbb.min()[1] << "," << gbb.min()[2] << "]->["
              << gbb.max()[0] << "," << gbb.max()[1] << "," << gbb.max()[2] << "]" << std::endl;

    // Contiguous runs of differing bytes, annotated with the GridData field they land in
    auto fieldName = [](size_t off) -> const char* {
        if (off <   8) return "mMagic";
        if (off <  16) return "mChecksum";
        if (off <  20) return "mVersion";
        if (off <  24) return "mFlags";
        if (off <  28) return "mGridIndex";
        if (off <  32) return "mGridCount";
        if (off <  40) return "mGridSize";
        if (off < 296) return "mGridName";
        if (off < 560) return "mMap";
        if (off < 608) return "mWorldBBox";
        if (off < 632) return "mVoxelSize";
        if (off < 636) return "mGridClass";
        if (off < 640) return "mGridType";
        if (off < 648) return "mBlindMetadataOffset";
        if (off < 652) return "mBlindMetadataCount";
        if (off < 664) return "mData0/mData1";
        if (off < 672) return "mData2";
        if (off < 736) return "TreeData";
        return "<node data>";
    };
    const char*  ref = reinterpret_cast<const char*>(reference);
    const char*  got = reinterpret_cast<const char*>(computed);
    const size_t n   = std::min(reference->gridSize(), computed->gridSize());
    size_t nDiff = 0, nRuns = 0;
    for (size_t i = 0; i < n; ) {
        if (ref[i] != got[i]) {
            size_t j = i; while (j < n && ref[j] != got[j]) ++j;
            nDiff += j - i;
            if (++nRuns <= 24) std::cout << "  [diff]   bytes [" << i << "," << j << ") in " << fieldName(i) << std::endl;
            i = j;
        } else ++i;
    }
    std::cout << "  [diff] " << nDiff << " of " << n << " bytes differ, in " << nRuns << " runs" << std::endl;
}

/// @brief Log2 of the number of active voxels handled by each VBM block (and each CUDA
///        thread block) in the voxelsToGrid dilation baselines.
static constexpr int BaselineLog2BlockWidth = 7;

/// @brief Decide whether a voxelsToGrid rebuild of @c coordCount coordinates can be attempted.
///
/// PointsToGrid holds, concurrently, the coordinate list itself (12 bytes each), 64-bit keys and
/// 32-bit indices in two copies (24 bytes each), and CUB radix-sort scratch of comparable size to
/// one key/value pair (~12 bytes each) -- roughly 50 bytes per coordinate at peak, plus whatever
/// the caller keeps resident.
///
/// This has to be predicted rather than attempted: NanoVDB routes allocation failures through
/// cudaCheck, which calls exit() rather than throwing, so an OOM would terminate the process and
/// take the remaining benchmarks with it.
///
/// @warning The 50-bytes-per-coordinate figure is an estimate derived from reading the
///          allocations in PointsToGrid::countNodes; CUB's exact temporary storage was not
///          measured. A run reported as skipped here might in principle have fit.
inline bool rebuildWouldFit(uint64_t coordCount, uint64_t residentBytes, const char* label)
{
    size_t freeBytes = 0, totalBytes = 0;
    cudaCheck(cudaMemGetInfo(&freeBytes, &totalBytes));
    const uint64_t needed = coordCount * 50ull + residentBytes;
    const double   toGB   = 1. / (1024. * 1024. * 1024.);
    if (double(needed) < 0.92 * double(freeBytes)) return true;
    std::cout << label << " SKIPPED: estimated peak " << double(needed) * toGB
              << " GB exceeds the " << double(freeBytes) * toGB << " GB currently free"
              << " (prediction, not an attempt -- see rebuildWouldFit)" << std::endl;
    return false;
}

/// @brief Build the list of index-space offsets of a dilation stencil, with the center first.
///
/// The NearestNeighbors enumerants are literally the neighbor counts (6/18/26), which correspond
/// to L1 radii of 1/2/3 respectively. The center is placed at index 0 so that the deduplicating
/// baseline can iterate the strictly-off-center spokes as [1, stencilSize).
inline std::vector<nanovdb::Coord> buildDilationStencil(uint32_t nnType)
{
    const int maxL1 = (nnType == nanovdb::tools::morphology::NN_FACE)      ? 1 :
                      (nnType == nanovdb::tools::morphology::NN_FACE_EDGE) ? 2 : 3;
    std::vector<nanovdb::Coord> stencil;
    stencil.push_back(nanovdb::Coord(0, 0, 0));
    for (int di = -1; di <= 1; ++di)
    for (int dj = -1; dj <= 1; ++dj)
    for (int dk = -1; dk <= 1; ++dk)
        if ((di || dj || dk) && std::abs(di) + std::abs(dj) + std::abs(dk) <= maxL1)
            stencil.push_back(nanovdb::Coord(di, dj, dk));
    return stencil;// 7, 19 or 27 entries, including the center
}

/// @brief Shared preamble of the VBM-driven baseline kernels: decode this thread block's inverse
///        maps and recover the calling thread's active voxel.
///
/// @warning Must be reached by every thread of the block, before any divergence, because
///          decodeInverseMaps synchronizes internally. Hence the "did this thread get a voxel"
///          answer is returned rather than acted on here.
///
/// @param coord  Index-space coordinate of this thread's active voxel (only written if true)
/// @param idx    Dense 0-based sequential index of that voxel (only written if true)
/// @return true if this thread maps to an active voxel; false for the threads of the final block
///         that run past the last active voxel of the grid
template<typename BuildT, int Log2BlockWidth>
__device__ inline bool decodeThreadVoxel(
    const nanovdb::NanoGrid<BuildT>* grid,
    const uint32_t*                  firstLeafIDArray,
    const uint64_t*                  jumpMapArray,
    const uint64_t                   firstOffset,
    uint32_t*                        smemLeafIndex,
    uint16_t*                        smemVoxelOffset,
    nanovdb::Coord&                  coord,
    uint64_t&                        idx)
{
    using VBM = nanovdb::tools::cuda::VoxelBlockManager<Log2BlockWidth>;

    const int      tID              = threadIdx.x;
    const uint64_t blockFirstOffset = firstOffset + uint64_t(blockIdx.x) * VBM::BlockWidth;

    VBM::decodeInverseMaps(grid, firstLeafIDArray[blockIdx.x],
                           jumpMapArray + VBM::JumpMapLength * blockIdx.x,
                           blockFirstOffset, smemLeafIndex, smemVoxelOffset);

    if (smemLeafIndex[tID] == VBM::UnusedLeafIndex) return false;

    const auto& leaf = grid->tree().template getFirstNode<0>()[ smemLeafIndex[tID] ];
    coord = leaf.offsetToGlobalCoord( smemVoxelOffset[tID] );
    // Sequential active-voxel indices are dense, so this lands in [0, activeVoxelCount)
    idx = blockFirstOffset + tID - firstOffset;
    return true;
}

/// @brief Declare the shared-memory scratch that decodeThreadVoxel writes into.
#define DECLARE_VBM_SMEM(Log2BlockWidth)                                                       \
    __shared__ uint32_t smemLeafIndex[nanovdb::tools::cuda::VoxelBlockManager<Log2BlockWidth>::BlockWidth]; \
    __shared__ uint16_t smemVoxelOffset[nanovdb::tools::cuda::VoxelBlockManager<Log2BlockWidth>::BlockWidth]

/// @brief Serialize the active voxels of @c grid into an explicit coordinate list, expanded
///        by the dilation stencil (the "v0" baseline).
///
/// One CUDA thread block per VBM block, one thread per active voxel; each thread emits
/// @c stencilSize copies of its voxel, offset by each spoke of the stencil. Duplicates are left
/// in place; voxelsToGrid dedups them during the rebuild.
///
/// The output is written spoke-major (@c dstCoords[s * voxelCount + idx]) so that the writes of
/// each spoke are fully coalesced across the thread block.
template<typename BuildT, int Log2BlockWidth>
__global__ void serializeDilatedCoordsKernel(
    const nanovdb::NanoGrid<BuildT>* grid,
    const uint32_t*                  firstLeafIDArray,
    const uint64_t*                  jumpMapArray,
    const uint64_t                   firstOffset,
    const uint64_t                   voxelCount,
    const nanovdb::Coord* __restrict stencil,
    const int                        stencilSize,
    nanovdb::Coord* __restrict       dstCoords)
{
    DECLARE_VBM_SMEM(Log2BlockWidth);
    nanovdb::Coord coord;
    uint64_t       idx;
    if (!decodeThreadVoxel<BuildT, Log2BlockWidth>(grid, firstLeafIDArray, jumpMapArray, firstOffset,
                                                   smemLeafIndex, smemVoxelOffset, coord, idx)) return;

    for (int s = 0; s < stencilSize; ++s)
        dstCoords[uint64_t(s) * voxelCount + idx] = coord + stencil[s];
}

/// @brief Count, for each active voxel, how many stencil taps it must contribute to the
///        coordinate list of the partially-deduplicated ("v1") baseline.
///
/// A voxel contributes (a) itself, always, and (b) each strictly-off-center stencil neighbor that
/// is currently *inactive*. Active neighbors are skipped because they necessarily contribute
/// themselves under rule (a), so the union of all contributions is still exactly the dilated
/// active set -- this deduplication is exact, not approximate.
///
/// Activity is probed with a leaf-caching ReadAccessor, i.e. the access pattern a NanoVDB user
/// would write today, rather than any of the bit-parallel machinery of DilateGrid.
template<typename BuildT, int Log2BlockWidth>
__global__ void countDilationTapsKernel(
    const nanovdb::NanoGrid<BuildT>* grid,
    const uint32_t*                  firstLeafIDArray,
    const uint64_t*                  jumpMapArray,
    const uint64_t                   firstOffset,
    const nanovdb::Coord* __restrict stencil,
    const int                        stencilSize,
    uint64_t* __restrict             dstCounts)
{
    DECLARE_VBM_SMEM(Log2BlockWidth);
    nanovdb::Coord coord;
    uint64_t       idx;
    if (!decodeThreadVoxel<BuildT, Log2BlockWidth>(grid, firstLeafIDArray, jumpMapArray, firstOffset,
                                                   smemLeafIndex, smemVoxelOffset, coord, idx)) return;

    nanovdb::ReadAccessor<BuildT, 0, -1, -1> acc(*grid);// cache leaf nodes only
    uint64_t count = 1;// the voxel itself
    for (int s = 1; s < stencilSize; ++s)
        if (!acc.isActive(coord + stencil[s])) ++count;

    dstCounts[idx] = count;
}

/// @brief Emit the partially-deduplicated coordinate list, using the offsets produced by a
///        zero-prefixed inclusive scan of the counts from countDilationTapsKernel.
///
/// Each thread writes a contiguous run of coordinates starting at @c offsets[idx], applying the
/// same inclusion rule as the counting pass.
template<typename BuildT, int Log2BlockWidth>
__global__ void emitDilationTapsKernel(
    const nanovdb::NanoGrid<BuildT>* grid,
    const uint32_t*                  firstLeafIDArray,
    const uint64_t*                  jumpMapArray,
    const uint64_t                   firstOffset,
    const nanovdb::Coord* __restrict stencil,
    const int                        stencilSize,
    const uint64_t* __restrict       offsets,
    nanovdb::Coord* __restrict       dstCoords)
{
    DECLARE_VBM_SMEM(Log2BlockWidth);
    nanovdb::Coord coord;
    uint64_t       idx;
    if (!decodeThreadVoxel<BuildT, Log2BlockWidth>(grid, firstLeafIDArray, jumpMapArray, firstOffset,
                                                   smemLeafIndex, smemVoxelOffset, coord, idx)) return;

    nanovdb::ReadAccessor<BuildT, 0, -1, -1> acc(*grid);// cache leaf nodes only
    uint64_t w = offsets[idx];
    dstCoords[w++] = coord;// the voxel itself
    for (int s = 1; s < stencilSize; ++s) {
        const auto neighbor = coord + stencil[s];
        if (!acc.isActive(neighbor)) dstCoords[w++] = neighbor;
    }
}

/// @brief Benchmark dilation emulated by rebuilding the grid from scratch with voxelsToGrid.
///
/// Serves as a baseline against the dedicated DilateGrid topological operator. The cost of
/// building the VoxelBlockManager is excluded (it is an amortizable structure that a real
/// application would keep around for other purposes), as is the allocation of the coordinate
/// list. The timed region covers the serialization of the stencil-expanded coordinate list plus
/// the voxelsToGrid rebuild -- i.e. everything that is specific to performing this one dilation.
template<typename BuildT>
void benchmarkVoxelsToGridDilation(
    nanovdb::NanoGrid<BuildT>* deviceGridOriginal,
    nanovdb::NanoGrid<BuildT>* indexGridOriginal,
    nanovdb::NanoGrid<BuildT>* indexGridDilated,
    uint32_t                   nnType,
    uint32_t                   benchmark_iters)
{
    using VBM = nanovdb::tools::cuda::VoxelBlockManager<BaselineLog2BlockWidth>;
    static constexpr int BlockWidth = VBM::BlockWidth;

    nanovdb::util::cuda::Timer gpuTimer;

    std::cout << "======== voxelsToGrid dilation baseline (v0: no dedup) ========" << std::endl;

    const std::vector<nanovdb::Coord> hostStencil = buildDilationStencil(nnType);
    const int stencilSize = static_cast<int>(hostStencil.size()); // 7, 19 or 27 (includes the center)

    const uint64_t voxelCount = indexGridOriginal->activeVoxelCount();
    const uint64_t coordCount = voxelCount * uint64_t(stencilSize);

    std::cout << "Stencil size (incl. center)           : " << stencilSize << std::endl;
    std::cout << "Coordinate list entries               : " << coordCount << std::endl;
    std::cout << "Coordinate list size                  : "
              << (coordCount * sizeof(nanovdb::Coord)) / (1024. * 1024.) << " MB" << std::endl;

    if (!rebuildWouldFit(coordCount, 0, "v0 baseline")) return;

    // Untimed: upload the stencil and allocate the (large) coordinate list once, then reuse it
    // across all benchmark iterations so that the timings do not measure cudaMalloc.
    auto stencilBuffer = nanovdb::cuda::DeviceBuffer::create(
        hostStencil.size() * sizeof(nanovdb::Coord), nullptr, false);
    auto* deviceStencil = static_cast<nanovdb::Coord*>(stencilBuffer.deviceData());
    if (!deviceStencil) throw std::runtime_error("No GPU buffer for the dilation stencil");
    cudaCheck(cudaMemcpy(deviceStencil, hostStencil.data(),
                         hostStencil.size() * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice));

    auto coordBuffer = nanovdb::cuda::DeviceBuffer::create(
        coordCount * sizeof(nanovdb::Coord), nullptr, false);
    auto* deviceCoords = static_cast<nanovdb::Coord*>(coordBuffer.deviceData());
    if (!deviceCoords) throw std::runtime_error("No GPU buffer for the expanded coordinate list");

    // Untimed: the VoxelBlockManager is an amortizable acceleration structure
    gpuTimer.start("Building the VoxelBlockManager (excluded from the benchmark)");
    auto vbmHandle = nanovdb::tools::cuda::buildVoxelBlockManager<BaselineLog2BlockWidth>(deviceGridOriginal);
    gpuTimer.stop();

    const uint64_t     nBlocks  = vbmHandle.blockCount();
    const nanovdb::Map srcMap   = indexGridDilated->map();
    const std::string  gridName = indexGridDilated->gridName();

    // Phase 1: serialize the stencil-expanded active voxels into the explicit coordinate list
    auto runSerialize = [&]() {
        serializeDilatedCoordsKernel<BuildT, BaselineLog2BlockWidth><<<nBlocks, BlockWidth>>>(
            deviceGridOriginal, vbmHandle.deviceFirstLeafID(), vbmHandle.deviceJumpMap(),
            vbmHandle.firstOffset(), voxelCount, deviceStencil, stencilSize, deviceCoords);
        cudaCheckError();
    };

    // Phase 2: rebuild the grid from that coordinate list. This performs exactly the work that
    // voxelsToGrid(deviceCoords, coordCount, voxelSize) would; we drive PointsToGrid directly
    // only to match metadata that the convenience wrapper cannot express, so that the result can
    // be compared byte-wise against the reference:
    //   - the source transform. voxelsToGrid hardcodes Map(voxelSize, Vec3d(0)), i.e. a uniform
    //     scale about the origin, which silently drops any translation the input VDB carries.
    //   - the grid name, which the wrapper leaves empty.
    //   - the checksum mode, which the wrapper disables. DilateGrid computes a CheckMode::Default
    //     checksum, so matching it here also keeps the timing comparison apples-to-apples.
    // None of these affect the sort/dedup/build work being measured.
    auto runBuild = [&]() {
        nanovdb::tools::cuda::PointsToGrid<BuildT> converter(srcMap);
        converter.setGridName(gridName);
        converter.setChecksum(nanovdb::CheckMode::Default);
        return converter.getHandle(deviceCoords, coordCount);
    };

    auto runOnce = [&]() { runSerialize(); return runBuild(); };

    // One warm-up run, whose result is checked against the reference
    auto  handle  = runOnce();
    auto* dstGrid = handle.template deviceGrid<BuildT>();

    if (bufferCheck((char*)dstGrid, (char*)indexGridDilated->data(), indexGridDilated->gridSize()))
        std::cout << "Result of voxelsToGrid baseline check out CORRECT against reference" << std::endl;
    else {
        std::cout << "Result of voxelsToGrid baseline compares INCORRECT against reference" << std::endl;
        handle.deviceDownload();
        reportGridDiff(indexGridDilated, handle.template grid<BuildT>());
    }

    // Re-run warm-started iterations of the complete baseline (serialization + rebuild)
    for (uint32_t i = 0; i < benchmark_iters; i++) {
        gpuTimer.start("Re-running entire voxelsToGrid dilation baseline after warmstart");
        auto dummyHandle = runOnce();
        gpuTimer.stop();
    }

    // Isolate the cost of the rebuild alone, i.e. what the baseline would cost if the serialized
    // coordinate list were somehow available for free. The list left behind by the loop above is
    // identical on every iteration, so it can simply be re-consumed without re-serializing it.
    for (uint32_t i = 0; i < benchmark_iters; i++) {
        gpuTimer.start("Re-running only the voxelsToGrid rebuild (serialization excluded)");
        auto dummyHandle = runBuild();
        gpuTimer.stop();
    }

    // ... and the serialization alone, for completeness (the two should account for the total)
    for (uint32_t i = 0; i < benchmark_iters; i++) {
        gpuTimer.start("Re-running only the coordinate-list serialization");
        runSerialize();
        cudaCheck(cudaStreamSynchronize(0));
        gpuTimer.stop();
    }
}

/// @brief Benchmark dilation by rebuild-from-scratch, from a partially deduplicated coordinate
///        list (the "v1" baseline).
///
/// Identical in spirit to benchmarkVoxelsToGridDilation, except that the coordinate list omits
/// every stencil tap that lands on an already-active voxel. Such taps are redundant: an active
/// voxel always contributes itself, so dropping them leaves the emitted set exactly equal to the
/// dilated active set while shrinking the list substantially.
///
/// The list is built in the canonical two-pass fashion: a counting sweep, a zero-prefixed
/// inclusive scan yielding both the per-voxel write offsets and the exact total, an allocation of
/// precisely that size, and an emitting sweep.
///
/// The headline number reported here is the cost of the voxelsToGrid rebuild *alone*, excluding
/// the neighbor probing, the scan and the emission. That is deliberately generous to the
/// alternative: it charges it only for work intrinsic to rebuilding, having handed it a
/// coordinate list that no straightforward implementation would obtain for free. It therefore
/// serves as a lower bound on what any rebuild-from-scratch approach could achieve. The honest
/// end-to-end cost is reported alongside it.
template<typename BuildT>
void benchmarkDedupedVoxelsToGridDilation(
    nanovdb::NanoGrid<BuildT>* deviceGridOriginal,
    nanovdb::NanoGrid<BuildT>* indexGridOriginal,
    nanovdb::NanoGrid<BuildT>* indexGridDilated,
    uint32_t                   nnType,
    uint32_t                   benchmark_iters)
{
    using VBM = nanovdb::tools::cuda::VoxelBlockManager<BaselineLog2BlockWidth>;
    static constexpr int BlockWidth = VBM::BlockWidth;

    nanovdb::util::cuda::Timer gpuTimer;

    std::cout << "======== voxelsToGrid dilation baseline (v1: partial dedup) ========" << std::endl;

    const std::vector<nanovdb::Coord> hostStencil = buildDilationStencil(nnType);
    const int      stencilSize = static_cast<int>(hostStencil.size());
    const uint64_t voxelCount  = indexGridOriginal->activeVoxelCount();

    int device = 0;
    cudaCheck(cudaGetDevice(&device));

    // Untimed: stencil upload
    auto stencilBuffer = nanovdb::cuda::DeviceBuffer::create(
        hostStencil.size() * sizeof(nanovdb::Coord), nullptr, false);
    auto* deviceStencil = static_cast<nanovdb::Coord*>(stencilBuffer.deviceData());
    if (!deviceStencil) throw std::runtime_error("No GPU buffer for the dilation stencil");
    cudaCheck(cudaMemcpy(deviceStencil, hostStencil.data(),
                         hostStencil.size() * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice));

    // Untimed: the VoxelBlockManager is an amortizable acceleration structure
    auto vbmHandle = nanovdb::tools::cuda::buildVoxelBlockManager<BaselineLog2BlockWidth>(deviceGridOriginal);
    const uint64_t nBlocks = vbmHandle.blockCount();

    // Scratch for the counting pass and the scan. Counts are accumulated in 64 bits because at
    // 27 taps per voxel the total can exceed 2^32 for large inputs.
    auto countsBuffer  = nanovdb::cuda::DeviceBuffer::create(voxelCount * sizeof(uint64_t), nullptr, false);
    auto offsetsBuffer = nanovdb::cuda::DeviceBuffer::create((voxelCount + 1) * sizeof(uint64_t), nullptr, false);
    auto* dCounts  = static_cast<uint64_t*>(countsBuffer.deviceData());
    auto* dOffsets = static_cast<uint64_t*>(offsetsBuffer.deviceData());
    if (!dCounts || !dOffsets) throw std::runtime_error("No GPU buffer for the tap counts/offsets");

    // CUB scan temporary storage, sized once and reused
    size_t tempBytes = 0;
    cub::DeviceScan::InclusiveSum(nullptr, tempBytes, dCounts, dOffsets + 1, voxelCount);
    auto tempBuffer = nanovdb::cuda::DeviceBuffer::create(tempBytes, nullptr, false);
    auto* dTemp = tempBuffer.deviceData();

    // Phase 1: count the taps contributed by each active voxel, then scan to obtain both the
    // per-voxel write offsets (element [i]) and the exact total (element [voxelCount]).
    auto runCountAndScan = [&]() {
        countDilationTapsKernel<BuildT, BaselineLog2BlockWidth><<<nBlocks, BlockWidth>>>(
            deviceGridOriginal, vbmHandle.deviceFirstLeafID(), vbmHandle.deviceJumpMap(),
            vbmHandle.firstOffset(), deviceStencil, stencilSize, dCounts);
        cudaCheckError();
        cudaCheck(cudaMemsetAsync(dOffsets, 0, sizeof(uint64_t), 0));
        size_t bytes = tempBytes;
        cub::DeviceScan::InclusiveSum(dTemp, bytes, dCounts, dOffsets + 1, voxelCount);
        cudaCheckError();
    };

    // Determine the exact coordinate-list length (untimed; the timed loops reuse the allocation)
    runCountAndScan();
    uint64_t coordCount = 0;
    cudaCheck(cudaMemcpy(&coordCount, dOffsets + voxelCount, sizeof(uint64_t), cudaMemcpyDeviceToHost));

    const uint64_t undedupedCount = voxelCount * uint64_t(stencilSize);
    std::cout << "Stencil size (incl. center)           : " << stencilSize << std::endl;
    std::cout << "Coordinate list entries               : " << coordCount
              << "  (vs " << undedupedCount << " undeduplicated)" << std::endl;
    std::cout << "Coordinate list size                  : "
              << (coordCount * sizeof(nanovdb::Coord)) / (1024. * 1024.) << " MB" << std::endl;
    std::cout << "Reduction vs v0                       : "
              << double(undedupedCount) / double(coordCount) << "x" << std::endl;

    // The counts and offsets arrays stay resident across the rebuild, so charge them too
    if (!rebuildWouldFit(coordCount, voxelCount * 2ull * sizeof(uint64_t), "v1 baseline")) return;

    auto coordBuffer = nanovdb::cuda::DeviceBuffer::create(coordCount * sizeof(nanovdb::Coord), nullptr, false);
    auto* deviceCoords = static_cast<nanovdb::Coord*>(coordBuffer.deviceData());
    if (!deviceCoords) throw std::runtime_error("No GPU buffer for the deduplicated coordinate list");

    // Phase 2: emit the coordinate list at the scanned offsets
    auto runEmit = [&]() {
        emitDilationTapsKernel<BuildT, BaselineLog2BlockWidth><<<nBlocks, BlockWidth>>>(
            deviceGridOriginal, vbmHandle.deviceFirstLeafID(), vbmHandle.deviceJumpMap(),
            vbmHandle.firstOffset(), deviceStencil, stencilSize, dOffsets, deviceCoords);
        cudaCheckError();
    };

    // Phase 3: rebuild, matching the reference metadata exactly as in the v0 baseline
    const nanovdb::Map srcMap   = indexGridDilated->map();
    const std::string  gridName = indexGridDilated->gridName();
    auto runBuild = [&]() {
        nanovdb::tools::cuda::PointsToGrid<BuildT> converter(srcMap);
        converter.setGridName(gridName);
        converter.setChecksum(nanovdb::CheckMode::Default);
        return converter.getHandle(deviceCoords, coordCount);
    };

    auto runOnce = [&]() { runCountAndScan(); runEmit(); return runBuild(); };

    // One warm-up run, whose result is checked against the reference
    auto  handle  = runOnce();
    auto* dstGrid = handle.template deviceGrid<BuildT>();

    if (bufferCheck((char*)dstGrid, (char*)indexGridDilated->data(), indexGridDilated->gridSize()))
        std::cout << "Result of deduplicated baseline check out CORRECT against reference" << std::endl;
    else {
        std::cout << "Result of deduplicated baseline compares INCORRECT against reference" << std::endl;
        handle.deviceDownload();
        reportGridDiff(indexGridDilated, handle.template grid<BuildT>());
    }

    // Honest end-to-end cost: probing + scan + emission + rebuild
    for (uint32_t i = 0; i < benchmark_iters; i++) {
        gpuTimer.start("Re-running entire deduplicated baseline after warmstart");
        auto dummyHandle = runOnce();
        gpuTimer.stop();
    }

    // Headline: the rebuild alone, as if the deduplicated list were available for free
    for (uint32_t i = 0; i < benchmark_iters; i++) {
        gpuTimer.start("Re-running only the voxelsToGrid rebuild of the deduplicated list");
        auto dummyHandle = runBuild();
        gpuTimer.stop();
    }

    // The probing/scan and emission phases, for completeness
    for (uint32_t i = 0; i < benchmark_iters; i++) {
        gpuTimer.start("Re-running only the tap counting and prefix sum");
        runCountAndScan();
        cudaCheck(cudaStreamSynchronize(0));
        gpuTimer.stop();
    }
    for (uint32_t i = 0; i < benchmark_iters; i++) {
        gpuTimer.start("Re-running only the deduplicated coordinate-list emission");
        runEmit();
        cudaCheck(cudaStreamSynchronize(0));
        gpuTimer.stop();
    }
}

template<typename BuildT>
void mainDilateGrid(
    nanovdb::NanoGrid<BuildT> *deviceGridOriginal,
    nanovdb::NanoGrid<BuildT> *deviceGridDilated,
    nanovdb::NanoGrid<BuildT> *indexGridOriginal,
    nanovdb::NanoGrid<BuildT> *indexGridDilated,
    uint32_t nnType,
    uint32_t benchmark_iters)
{
    nanovdb::util::cuda::Timer gpuTimer;

    // Initialize dilator
    nanovdb::tools::cuda::DilateGrid<BuildT> dilator( deviceGridOriginal );
    dilator.setOperation(nanovdb::tools::morphology::NearestNeighbors(nnType));
    dilator.setChecksum(nanovdb::CheckMode::Default);
    dilator.setVerbose(1);

    auto handle = dilator.getHandle();
    auto dstGrid = handle.template deviceGrid<BuildT>();

    // Check for correctness
    if (bufferCheck((char*)dstGrid, (char*)indexGridDilated->data(), indexGridDilated->gridSize()))
        std::cout << "Result of DilateGrid check out CORRECT against reference" << std::endl;
    else
        std::cout << "Result of DilateGrid compares INCORRECT against reference" << std::endl;

    // Re-run warm-started iterations
    dilator.setVerbose(0);
    for (int i = 0; i < benchmark_iters; i++) {
        gpuTimer.start("Re-running entire dilation after warmstart");
        auto dummyHandle = dilator.getHandle();
        gpuTimer.stop();
    }

    // Additional points of comparison for the dilation operation: emulate it by serializing the
    // stencil-expanded active voxels and rebuilding the grid from scratch with voxelsToGrid,
    // both without (v0) and with (v1) the elision of taps landing on already-active voxels
    // Either baseline may exhaust device memory on large inputs -- which is itself a reportable
    // result -- so a failure of one must not prevent the others from running.
    try {
        benchmarkVoxelsToGridDilation( deviceGridOriginal, indexGridOriginal, indexGridDilated, nnType, benchmark_iters );
    } catch (const std::exception& e) {
        std::cout << "v0 baseline FAILED: " << e.what() << std::endl;
    }
    try {
        benchmarkDedupedVoxelsToGridDilation( deviceGridOriginal, indexGridOriginal, indexGridDilated, nnType, benchmark_iters );
    } catch (const std::exception& e) {
        std::cout << "v1 baseline FAILED: " << e.what() << std::endl;
    }

    uint32_t dstLeafCount = nanovdb::util::cuda::DeviceGridTraits<BuildT>::getTreeData(dstGrid).mNodeCount[0];
    nanovdb::cuda::DeviceBuffer dstLeafMaskBuffer;
    nanovdb::Mask<3>* dstLeafMasks = nullptr;
    if (dstLeafCount) {
        dstLeafMaskBuffer = nanovdb::cuda::DeviceBuffer::create( std::size_t(dstLeafCount) * sizeof(nanovdb::Mask<3>), nullptr, false );
        dstLeafMasks = static_cast<nanovdb::Mask<3>*>(dstLeafMaskBuffer.deviceData());
        if (!dstLeafMasks) throw std::runtime_error("No GPU buffer for dstLeafMask");
    }

    const unsigned int numThreads = 128;
    auto numBlocks = [numThreads] (unsigned int n) {return (n + numThreads - 1) / numThreads;};
    gpuTimer.start("Injecting un-dilated topology as a pruning mask");
    if (dstLeafCount)
        nanovdb::util::cuda::lambdaKernel<<<numBlocks(dstLeafCount), numThreads>>>(dstLeafCount,
            nanovdb::util::cuda::InjectGridMaskFunctor<BuildT>(),
            deviceGridOriginal, dstGrid, dstLeafMasks );
    gpuTimer.stop();

    // Initialize pruner
    nanovdb::tools::cuda::PruneGrid<BuildT> pruner( dstGrid, dstLeafMasks );
    pruner.setChecksum(nanovdb::CheckMode::Default);
    pruner.setVerbose(1);

    auto prunedHandle = pruner.getHandle();
    auto prunedGrid = prunedHandle.template deviceGrid<BuildT>();

    // Check for correctness
    if (bufferCheck((char*)prunedGrid, (char*)indexGridOriginal->data(), indexGridOriginal->gridSize()))
        std::cout << "Result of PruneGrid check out CORRECT against reference" << std::endl;
    else
        std::cout << "Result of PruneGrid compares INCORRECT against reference" << std::endl;

    // Re-run warm-started iterations
    pruner.setVerbose(0);
    for (int i = 0; i < benchmark_iters; i++) {
        gpuTimer.start("Re-running entire pruning after warmstart");
        auto dummyHandle = pruner.getHandle();
        gpuTimer.stop();
    }

}

template
void mainDilateGrid(
    nanovdb::NanoGrid<nanovdb::ValueOnIndex> *deviceGridOriginal,
    nanovdb::NanoGrid<nanovdb::ValueOnIndex> *deviceGridDilated,
    nanovdb::NanoGrid<nanovdb::ValueOnIndex> *indexGridOriginal,
    nanovdb::NanoGrid<nanovdb::ValueOnIndex> *indexGridDilated,
    uint32_t nnType,
    uint32_t benchmark_iters
);
