// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file  connected_components_cuda_kernels.cu
///
/// @brief CUDA / NanoVDB side of the connected-components example (no OpenVDB).
///
///        computeUDF(): uploads the triangle mesh and voxelizes it into a narrow-band
///        ValueOnIndex grid plus a per-active-voxel unsigned-distance-field (UDF)
///        sidecar, via nanovdb::tools::cuda::MeshToGrid.
///
///        The connected-components labeling (a union-find with hierarchical insights)
///        and the derivative-grid step it operates on will be added here as follow-ups.
///        See MeshToSDFDevelopmentPlan.md in this directory for the design notes and roadmap.

#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/cuda/MeshToGrid.cuh>
#include <nanovdb/tools/cuda/PruneGrid.cuh>
#include <nanovdb/tools/cuda/ConnectedComponents.cuh>
#include <nanovdb/util/cuda/DeviceGridTraits.cuh>
#include <nanovdb/util/cuda/Util.h>

#include <thrust/universal_vector.h>

#include <iostream>
#include <string>
#include <utility>
#include <vector>

// Must match the aliases in connected_components_cuda.cpp.
using GridHandleT = nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>;
using UDFSidecarT = nanovdb::cuda::DeviceBuffer;

std::pair<GridHandleT, UDFSidecarT> computeUDF(
    const std::vector<nanovdb::Vec3f>& points,
    const std::vector<nanovdb::Vec3i>& triangles,
    const nanovdb::Map&                map,
    float                              bandWidth)
{
    using BuildT = nanovdb::ValueOnIndex;

    // Upload mesh to the device (managed memory keeps the example simple).
    thrust::universal_vector<nanovdb::Vec3f> dPoints(points.begin(), points.end());
    thrust::universal_vector<nanovdb::Vec3i> dTriangles(triangles.begin(), triangles.end());

    nanovdb::tools::cuda::MeshToGrid<BuildT> converter(
        dPoints.data().get(),    uint32_t(dPoints.size()),
        dTriangles.data().get(), uint32_t(dTriangles.size()),
        map);
    converter.setVerbose(1);
    converter.setNarrowBandWidth(bandWidth);

    // { index-grid handle, UDF sidecar }, both backed by DeviceBuffer.
    return converter.getHandleAndUDF();
}

/// @brief Print topology diagnostics for a device-resident ValueOnIndex grid, in the
///        style of ex_dilate_nanovdb_cuda. Reads the grid's header/tree fields directly
///        off the device via DeviceGridTraits (small targeted cudaMemcpy's) - no full
///        deviceDownload of the grid buffer is required.
void printGridDiagnostics(const GridHandleT& handle, const std::string& title)
{
    using BuildT = nanovdb::ValueOnIndex;
    using Traits = nanovdb::util::cuda::DeviceGridTraits<BuildT>;

    const auto* d_grid = handle.deviceGrid<BuildT>();

    const auto     treeData     = Traits::getTreeData(d_grid);
    const uint64_t valueCount   = Traits::getValueCount(d_grid);
    const uint64_t activeVoxels = Traits::getActiveVoxelCount(d_grid);
    const uint64_t gridSize     = Traits::getGridSize(d_grid);
    const auto     bbox         = Traits::getIndexBBox(d_grid, treeData);

    const uint64_t leafNodes  = treeData.mNodeCount[0];
    const uint64_t lowerNodes = treeData.mNodeCount[1];
    const uint64_t upperNodes = treeData.mNodeCount[2];

    std::cout << "============ " << title << " ============\n";
    std::cout << "Allocated values [valueCount()]       : " << valueCount << "\n";
    std::cout << "Active voxels    [activeVoxelCount()] : " << activeVoxels << "\n";
    std::cout << "Index-space bounding box              : ["
              << bbox.min().x() << "," << bbox.min().y() << "," << bbox.min().z() << "] -> ["
              << bbox.max().x() << "," << bbox.max().y() << "," << bbox.max().z() << "]\n";
    std::cout << "Leaf nodes                            : " << leafNodes  << "\n";
    std::cout << "Lower internal nodes                  : " << lowerNodes << "\n";
    std::cout << "Upper internal nodes                  : " << upperNodes << "\n";
    std::cout << "Leaf-level occupancy                  : "
              << (leafNodes ? 100.f * float(activeVoxels) / float(leafNodes * 512) : 0.f)
              << "%\n";
    std::cout << "Memory usage                          : " << gridSize << " bytes\n";
}

namespace {

/// @brief CUDA functor: build a per-leaf retain bitmask that drops the surface/barrier
///        shell. A voxel is PRUNED iff it is within √3/2 voxels of the surface (half a
///        voxel space-diagonal - the same barrier OpenVDB's MeshToVolume uses); every
///        other active voxel is RETAINED. Because the UDF sidecar is in WORLD units, the
///        test is udf^2 < (√3/2 · voxelSize)^2 = 0.75 · voxelSize^2, passed in precomputed.
///
///        Mirrors Benchmark.cu's PruneNarrowBandFunctor: launched via operatorKernel, one
///        block per leaf, 512 threads per block (one thread per voxel in the 8^3 leaf).
template <typename BuildT>
struct UDFBarrierPruneMaskFunctor
{
    static constexpr int MaxThreadsPerBlock         = 512;
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
            if (udf * udf >= barrierSqWorld)            // retain non-barrier voxels
                resultMask.setOnAtomic(threadID);
        }
    }
};

} // anonymous namespace

GridHandleT computeDerivedTopology(const GridHandleT& srcHandle, const UDFSidecarT& udfSidecar,
                                   float voxelSize)
{
    using BuildT  = nanovdb::ValueOnIndex;
    using PruneOp = UDFBarrierPruneMaskFunctor<BuildT>;

    // Barrier threshold √3/2 voxels expressed in the sidecar's WORLD units, squared.
    const float barrierSqWorld = 0.75f * voxelSize * voxelSize;

    const auto*  d_srcGrid = srcHandle.deviceGrid<BuildT>();
    const float* d_udf     = static_cast<const float*>(udfSidecar.deviceData());

    const uint32_t srcLeafCount =
        nanovdb::util::cuda::DeviceGridTraits<BuildT>::getTreeData(d_srcGrid).mNodeCount[0];

    // Leaf-indexed retain mask: one Mask<3> (512 bits) per source leaf (device-only).
    auto retainMask = nanovdb::cuda::DeviceBuffer::create(
        std::size_t(srcLeafCount) * sizeof(nanovdb::Mask<3>), nullptr, false);
    auto* d_retainMask = static_cast<nanovdb::Mask<3>*>(retainMask.deviceData());

    // Build the retain mask (drop voxels within √3/2 voxels of the surface).
    nanovdb::util::cuda::operatorKernel<PruneOp>
        <<<srcLeafCount, PruneOp::MaxThreadsPerBlock>>>(d_srcGrid, d_udf, barrierSqWorld, d_retainMask);
    cudaCheck(cudaGetLastError());

    // Topological pruning -> clean, topology-only derived index grid (UDF no longer needed).
    nanovdb::tools::cuda::PruneGrid<BuildT> pruner(d_srcGrid, d_retainMask);
    pruner.setVerbose(1);
    return pruner.getHandle();
}

namespace {

/// @brief CPU reference for per-leaf 6-connected component counts on a host-resident
///        ValueOnIndex grid. Harvested from TestNanoVDB.cu's LeafConnectedComponents
///        oracle: a union-find local to each 8^3 leaf (parent[n]=n for active voxels,
///        -1 for inactive; union only active +X/+Y/+Z in-leaf neighbors so each undirected
///        face edge is visited once; larger root attaches under the smaller; the component
///        count is the number of surviving roots). Counts are in leaf storage order, to
///        match the device-side deviceLeafComponentCounts().
template <typename GridT>
std::vector<uint16_t> cpuLeafComponentCounts(const GridT* grid)
{
    const auto&    tree      = grid->tree();
    const uint32_t leafCount = tree.nodeCount(0);
    const auto*    leaves    = tree.getFirstLeaf();

    int parent[512] = {};// re-initialized per leaf below
    auto find = [&](int i) {
        while (parent[i] != i) { parent[i] = parent[parent[i]]; i = parent[i]; }
        return i;
    };
    auto unite = [&](int a, int b) {
        const int ra = find(a), rb = find(b);
        if (ra == rb) return;
        if (ra < rb) parent[rb] = ra;// attach larger root under smaller root
        else         parent[ra] = rb;
    };

    std::vector<uint16_t> counts(leafCount);
    for (uint32_t li = 0; li < leafCount; ++li) {
        const auto& leaf = leaves[li];
        for (int n = 0; n < 512; ++n) parent[n] = leaf.isActive(uint32_t(n)) ? n : -1;
        // x-major offset n = (x<<6)|(y<<3)|z, so +X/+Y/+Z in-leaf neighbors are n+64/n+8/n+1.
        for (int n = 0; n < 512; ++n) {
            if (parent[n] < 0) continue;// inactive
            const int x = n >> 6, y = (n >> 3) & 7, z = n & 7;
            if (x < 7 && parent[n + 64] >= 0) unite(n, n + 64);
            if (y < 7 && parent[n +  8] >= 0) unite(n, n +  8);
            if (z < 7 && parent[n +  1] >= 0) unite(n, n +  1);
        }
        uint16_t c = 0;
        for (int n = 0; n < 512; ++n) if (parent[n] == n) ++c;// surviving roots == components
        counts[li] = c;
    }
    return counts;
}

/// @brief CPU reference for the per-component leaf masks and 6 face bitmasks, mirroring
///        cpuLeafComponentCounts(): the same per-leaf 6-connected union-find, then components
///        enumerated in ascending root-label order (which matches the GPU kernel's repeated
///        BlockReduce-min enumeration, so global component slots line up). For each component
///        it builds the Mask<3> footprint and the 6 face bitmasks directly from voxel
///        coordinates -- intentionally NOT mirroring the kernel's shift/0x0101... extraction,
///        so this is an independent check. Bit conventions follow the LeafNeighborTap enum in
///        tools/cuda/ConnectedComponents.cuh:
///          +/-X face: bit y*8+z ;  +/-Y face: bit x*8+z ;  +/-Z face: bit y*8+x.
struct CpuMasksFaces {
    std::vector<nanovdb::Mask<3>> masks;  // K masks, one per leaf-local component (global slot order)
    std::vector<uint64_t>         faces;  // 6*K: face t of component c at faces[6*c + t]
    uint64_t                      K{0};   // total leaf-local components (== deviceLeafComponentOffsets()[leafCount])
};

template <typename GridT>
CpuMasksFaces cpuMasksFaces(const GridT* grid)
{
    namespace cc = nanovdb::tools::cuda;  // for the LeafNeighborTap enumerators (minusX..plusZ)
    const auto&    tree      = grid->tree();
    const uint32_t leafCount = tree.nodeCount(0);
    const auto*    leaves    = tree.getFirstLeaf();

    int parent[512] = {};// re-initialized per leaf below
    auto find = [&](int i) {
        while (parent[i] != i) { parent[i] = parent[parent[i]]; i = parent[i]; }
        return i;
    };
    auto unite = [&](int a, int b) {
        const int ra = find(a), rb = find(b);
        if (ra == rb) return;
        if (ra < rb) parent[rb] = ra;// attach larger root under smaller root
        else         parent[ra] = rb;
    };

    CpuMasksFaces out;
    int rootIdx[512];
    for (uint32_t li = 0; li < leafCount; ++li) {
        const auto& leaf = leaves[li];
        for (int n = 0; n < 512; ++n) parent[n] = leaf.isActive(uint32_t(n)) ? n : -1;
        for (int n = 0; n < 512; ++n) {
            if (parent[n] < 0) continue;// inactive
            const int x = n >> 6, y = (n >> 3) & 7, z = n & 7;
            if (x < 7 && parent[n + 64] >= 0) unite(n, n + 64);
            if (y < 7 && parent[n +  8] >= 0) unite(n, n +  8);
            if (z < 7 && parent[n +  1] >= 0) unite(n, n +  1);
        }
        // Dense local component indices in ascending root-label (== ascending offset) order.
        int localCount = 0;
        for (int n = 0; n < 512; ++n) rootIdx[n] = -1;
        for (int n = 0; n < 512; ++n) if (parent[n] == n) rootIdx[n] = localCount++;

        std::vector<nanovdb::Mask<3>> lmask(localCount);
        for (auto& m : lmask) m.setOff();
        std::vector<uint64_t> lface(std::size_t(6) * localCount, 0);

        for (int n = 0; n < 512; ++n) {
            if (parent[n] < 0) continue;
            const int k = rootIdx[find(n)];
            const int x = n >> 6, y = (n >> 3) & 7, z = n & 7;
            lmask[k].setOn(uint32_t(n));
            uint64_t* f = &lface[6 * k];
            if (x == 0) f[cc::minusX] |= uint64_t(1) << (y * 8 + z);
            if (x == 7) f[cc::plusX ] |= uint64_t(1) << (y * 8 + z);
            if (y == 0) f[cc::minusY] |= uint64_t(1) << (x * 8 + z);
            if (y == 7) f[cc::plusY ] |= uint64_t(1) << (x * 8 + z);
            if (z == 0) f[cc::minusZ] |= uint64_t(1) << (y * 8 + x);
            if (z == 7) f[cc::plusZ ] |= uint64_t(1) << (y * 8 + x);
        }
        out.masks.insert(out.masks.end(), lmask.begin(), lmask.end());
        out.faces.insert(out.faces.end(), lface.begin(), lface.end());
        out.K += uint64_t(localCount);
    }
    return out;
}

} // anonymous namespace

void computeCC(const GridHandleT& gridHandle)
{
    using BuildT = nanovdb::ValueOnIndex;
    using Traits = nanovdb::util::cuda::DeviceGridTraits<BuildT>;

    const auto* d_grid = gridHandle.deviceGrid<BuildT>();

    // Device: per-leaf connected-component counts.
    nanovdb::tools::cuda::ConnectedComponents<BuildT> cc(d_grid);
    cc.setVerbose(1);
    cc.processLeafConnectedComponents();
    cudaCheck(cudaDeviceSynchronize());

    const uint32_t leafCount = Traits::getTreeData(d_grid).mNodeCount[0];
    if (leafCount == 0) { std::cout << "CC validation: empty grid, nothing to check\n"; return; }

    // GPU result -> host.
    std::vector<uint16_t> gpuCounts(leafCount);
    cudaCheck(cudaMemcpy(gpuCounts.data(), cc.deviceLeafComponentCounts(),
                         std::size_t(leafCount) * sizeof(uint16_t), cudaMemcpyDeviceToHost));

    // CPU validation. NanoVDB grids are position-independent (relative offsets), so a raw
    // byte copy of the device blob into a (32B-aligned) scratch host buffer is itself a valid
    // host grid. We don't touch the handle (no deviceDownload residue); the scratch is freed
    // immediately after the check.
    const uint64_t gridBytes = gridHandle.bufferSize();
    void* hostBlob = nullptr;
    cudaCheck(cudaMallocHost(&hostBlob, gridBytes));   // pinned, >=32B aligned
    cudaCheck(cudaMemcpy(hostBlob, gridHandle.deviceData(), gridBytes, cudaMemcpyDeviceToHost));
    const auto* h_grid = reinterpret_cast<const nanovdb::NanoGrid<BuildT>*>(hostBlob);

    const std::vector<uint16_t> cpuCounts = cpuLeafComponentCounts(h_grid);

    std::size_t mismatches = 0;
    uint64_t    gpuTotal = 0, cpuTotal = 0;
    for (uint32_t li = 0; li < leafCount; ++li) {
        gpuTotal += gpuCounts[li];
        cpuTotal += cpuCounts[li];
        if (gpuCounts[li] != cpuCounts[li]) {
            if (mismatches < 10) {
                // host active popcount for this leaf
                const auto& hleaf = h_grid->tree().getFirstLeaf()[li];
                int act = 0; for (int q = 0; q < 512; ++q) if (hleaf.isActive(uint32_t(q))) ++act;
                std::cerr << "  CC mismatch @ leaf " << li << ": gpu=" << gpuCounts[li]
                          << " cpu=" << cpuCounts[li] << " (host active voxels=" << act << ")\n";
            }
            ++mismatches;
        }
    }

    std::cout << "CC per-leaf component-count validation: "
              << (mismatches == 0 ? "PASS" : "FAIL") << " ("
              << leafCount << " leaves, " << mismatches << " mismatches; "
              << "total components gpu=" << gpuTotal << " cpu=" << cpuTotal << ")\n";

    // ---- Validate per-component leaf masks and face flags (pipeline step 4) ----
    // Offsets give K (total components) and each leaf's slot range; masks/faces are flat,
    // indexed by global component slot = offsets[leafID] + localIdx.
    std::vector<uint64_t> gpuOffsets(std::size_t(leafCount) + 1);
    cudaCheck(cudaMemcpy(gpuOffsets.data(), cc.deviceLeafComponentOffsets(),
                         (std::size_t(leafCount) + 1) * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    const uint64_t K = gpuOffsets[leafCount];

    std::vector<nanovdb::Mask<3>> gpuMasks(K);
    cudaCheck(cudaMemcpy(gpuMasks.data(), cc.deviceLeafComponentMasks(),
                         std::size_t(K) * sizeof(nanovdb::Mask<3>), cudaMemcpyDeviceToHost));
    std::vector<uint64_t> gpuFaces(std::size_t(6) * K);
    cudaCheck(cudaMemcpy(gpuFaces.data(), cc.deviceLeafComponentFaceMasks(),
                         std::size_t(6) * K * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    const CpuMasksFaces cmf = cpuMasksFaces(h_grid);

    std::size_t maskMismatch = 0, faceMismatch = 0, shownMF = 0;
    for (uint64_t c = 0; c < K; ++c) {
        bool mbad = false;
        for (int w = 0; w < 8; ++w)
            if (gpuMasks[c].words()[w] != cmf.masks[c].words()[w]) { mbad = true; break; }
        bool fbad = false;
        for (int t = 0; t < 6; ++t)
            if (gpuFaces[6 * c + t] != cmf.faces[6 * c + t]) { fbad = true; break; }
        if (mbad) ++maskMismatch;
        if (fbad) ++faceMismatch;
        if ((mbad || fbad) && shownMF++ < 10) {
            uint32_t li = 0; while (li < leafCount && gpuOffsets[li + 1] <= c) ++li;
            std::cerr << "  CC mask/face mismatch @ component " << c << " (leaf " << li
                      << ", local " << (c - gpuOffsets[li]) << ")"
                      << (mbad ? " [mask]" : "") << (fbad ? " [face]" : "") << "\n";
        }
    }

    std::cout << "CC per-component mask/face validation:  "
              << ((maskMismatch == 0 && faceMismatch == 0) ? "PASS" : "FAIL") << " ("
              << K << " components, " << maskMismatch << " mask mismatches, "
              << faceMismatch << " face mismatches)\n";

    cudaCheck(cudaFreeHost(hostBlob));
}
