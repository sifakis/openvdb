// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0


#include <nanovdb/NanoVDB.h>

#include <nanovdb/tools/cuda/MeshToGrid.cuh>

#include <openvdb/openvdb.h>

#include <algorithm>
#include <limits>
#include <array>
#include <vector>

#if 0
#include <nanovdb/tools/cuda/DilateGrid.cuh>
#include <nanovdb/tools/cuda/PruneGrid.cuh>
#include <nanovdb/util/cuda/Injection.cuh>

template<typename T>
bool bufferCheck(const T* deviceBuffer, const T* hostBuffer, size_t elem_count) {
    T* tmpBuffer = new T[elem_count];
    cudaCheck(cudaMemcpy(tmpBuffer, deviceBuffer, elem_count * sizeof(T), cudaMemcpyDeviceToHost));
    bool same = true;
    for (int i=0; same && i< elem_count; ++i) { same = (tmpBuffer[i] == hostBuffer[i]); }
    delete [] tmpBuffer;
    return same;
}
#endif

template<typename BuildT>
void mainMeshToGrid(
    const nanovdb::Vec3f *devicePoints,
    const int pointCount,
    const nanovdb::Vec3i *deviceTriangles,
    const int triangleCount,
    const nanovdb::Map map,
    const openvdb::FloatGrid::Ptr refGrid)
{
    nanovdb::util::cuda::Timer gpuTimer;

    // Initialize mesh-to-grid converter
    nanovdb::tools::cuda::MeshToGrid<BuildT> converter( devicePoints, pointCount, deviceTriangles, triangleCount, map );
    converter.setVerbose(1);
    auto handle = converter.getHandle();


#if 0
    // --- DIAGNOSTIC CHECK: Root-level Modulus & Bounds Test (only valid before processLeafTrianglePairs) ---
    uint64_t pairCount = converter.getPairCount();
    if (pairCount > 0) {
        std::cout << "\n--- Running GPU Diagnostics ---" << std::endl;

        std::vector<typename nanovdb::tools::cuda::MeshToGrid<BuildT>::BoxTrianglePair> hostPairs(pairCount);
        cudaMemcpy(hostPairs.data(), converter.getDevicePairs(),
                   pairCount * sizeof(typename nanovdb::tools::cuda::MeshToGrid<BuildT>::BoxTrianglePair),
                   cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        bool passed = true;
        for (uint64_t i = 0; i < pairCount; ++i) {
            const auto& pair = hostPairs[i];
            if (pair.origin[0] % 4096 != 0 ||
                pair.origin[1] % 4096 != 0 ||
                pair.origin[2] % 4096 != 0) {
                std::cerr << "FAIL: Misaligned Root Origin at index " << i
                          << " (" << pair.origin[0] << ", " << pair.origin[1] << ", " << pair.origin[2] << ")\n";
                passed = false;
                break;
            }
            if (pair.triangleID >= (uint32_t)triangleCount) {
                std::cerr << "FAIL: Out-of-bounds TriangleID " << pair.triangleID << " at index " << i << "\n";
                passed = false;
                break;
            }
        }
        if (passed)
            std::cout << "SUCCESS: All " << pairCount << " pairs are perfectly 4096-aligned and bounded!" << std::endl;
    }
#endif

    // --- Voxel-level UDF correctness check ---
    // Download NanoVDB grid to host for CPU-side analysis
    std::vector<char> hostBuf(handle.bufferSize());
    cudaCheck(cudaMemcpy(hostBuf.data(), handle.deviceData(), handle.bufferSize(), cudaMemcpyDeviceToHost));
    const auto *nanoGrid = reinterpret_cast<const nanovdb::NanoGrid<BuildT>*>(hostBuf.data());

    std::cout << "\n--- Voxel-level UDF correctness check ---" << std::endl;

    // Encode a voxel coord as a sortable int64_t key.
    // Coords fit well within 20 bits per axis for typical meshes.
    auto encodeCoord = [](int x, int y, int z) -> int64_t {
        return (int64_t(x) + (1<<20))
             | ((int64_t(y) + (1<<20)) << 21)
             | ((int64_t(z) + (1<<20)) << 42);
    };

    // Single pass over our NanoVDB active voxels:
    //   - build sorted coord list (for false-negative lookup)
    //   - count false positives (active in ours, background in OpenVDB)
    auto ovdbAcc = refGrid->getConstAccessor();
    uint64_t falsePositives = 0;
    std::vector<int64_t> ourCoords;
    ourCoords.reserve(12000000);

    const uint32_t nanoLeafCount = nanoGrid->tree().nodeCount(0);
    const auto *leaves = nanoGrid->tree().getFirstLeaf();
    for (uint32_t li = 0; li < nanoLeafCount; ++li) {
        const auto &leaf = leaves[li];
        const auto origin = leaf.origin();
        for (uint32_t vi = 0; vi < 512; ++vi) {
            if (leaf.isActive(vi)) {
                const int lx = vi & 7, ly = (vi >> 3) & 7, lz = (vi >> 6) & 7;
                const int x = origin[0]+lx, y = origin[1]+ly, z = origin[2]+lz;
                ourCoords.push_back(encodeCoord(x, y, z));
                if (!ovdbAcc.isValueOn(openvdb::Coord(x, y, z)))
                    ++falsePositives;
            }
        }
    }
    std::sort(ourCoords.begin(), ourCoords.end());

    // False negatives: active in OpenVDB UDF but absent from our sorted set.
    uint64_t falseNegatives = 0;
    float minMissedUDF = std::numeric_limits<float>::max();
    for (auto it = refGrid->tree().cbeginValueOn(); it; ++it) {
        const auto c = it.getCoord();
        if (!std::binary_search(ourCoords.begin(), ourCoords.end(),
                                encodeCoord(c[0], c[1], c[2]))) {
            ++falseNegatives;
            minMissedUDF = std::min(minMissedUDF, *it);
        }
    }

    const uint64_t ourActiveCount = ourCoords.size();
    std::cout << "Our active voxels:     " << ourActiveCount << "\n";
    std::cout << "OpenVDB active voxels: " << refGrid->tree().activeVoxelCount() << "\n";
    if (falseNegatives == 0)
        std::cout << "False negatives: 0 -- all OpenVDB active voxels present in our output\n";
    else
        std::cerr << "False negatives: " << falseNegatives
                  << " (smallest missed UDF value: " << minMissedUDF << ")\n";
    std::cout << "False positives: " << falsePositives
              << " (" << (100.0 * falsePositives / ourActiveCount) << "% of our active voxels)\n";



#if 0
    dilator.setOperation(nanovdb::tools::morphology::NearestNeighbors(nnType));
    dilator.setChecksum(nanovdb::CheckMode::Default);

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
#endif
}

template
void mainMeshToGrid<nanovdb::ValueOnIndex>(
    const nanovdb::Vec3f *devicePoints,
    const int pointCount,
    const nanovdb::Vec3i *deviceTriangles,
    const int triangleCount,
    const nanovdb::Map map,
    const openvdb::FloatGrid::Ptr refGrid);
