// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0


#include <nanovdb/NanoVDB.h>

#include <nanovdb/tools/cuda/MeshToGrid.cuh>

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
    const nanovdb::Map map)
{
    nanovdb::util::cuda::Timer gpuTimer;

    // Initialize mesh-to-grid converter
    nanovdb::tools::cuda::MeshToGrid<BuildT> converter( devicePoints, pointCount, deviceTriangles, triangleCount, map );
    converter.setVerbose(1);
    converter.getHandle();


    // --- DIAGNOSTIC CHECK 1: The Modulus & Bounds Test ---
    uint64_t pairCount = converter.getPairCount();
    if (pairCount > 0) {
        std::cout << "\n--- Running GPU Diagnostics ---" << std::endl;
        
        // Allocate host memory and download the pairs
        std::vector<typename nanovdb::tools::cuda::MeshToGrid<BuildT>::BoxTrianglePair> hostPairs(pairCount);
        cudaMemcpy(hostPairs.data(), converter.getDevicePairs(), 
                   pairCount * sizeof(typename nanovdb::tools::cuda::MeshToGrid<BuildT>::BoxTrianglePair), 
                   cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        bool passed = true;
        for (uint64_t i = 0; i < pairCount; ++i) {
            const auto& pair = hostPairs[i];
            
            // 1. Verify Modulo 4096 (Strict Root Tile Alignment)
            if (pair.origin[0] % 4096 != 0 || 
                pair.origin[1] % 4096 != 0 || 
                pair.origin[2] % 4096 != 0) {
                std::cerr << "FAIL: Misaligned Root Origin at index " << i 
                          << " (" << pair.origin[0] << ", " << pair.origin[1] << ", " << pair.origin[2] << ")\n";
                passed = false;
                break;
            }

            // 2. Verify Triangle ID bounds
            if (pair.triangleID >= triangleCount) {
                std::cerr << "FAIL: Out-of-bounds TriangleID " << pair.triangleID << " at index " << i << "\n";
                passed = false;
                break;
            }
        }
        
        if (passed) {
            std::cout << "SUCCESS: All " << pairCount << " pairs are perfectly 4096-aligned and bounded!" << std::endl;
            // Print a sample to visually inspect
            std::cout << "Sample Pair [0]: Origin(" << hostPairs[0].origin[0] << ", " 
                      << hostPairs[0].origin[1] << ", " << hostPairs[0].origin[2] << ") - TriID: " 
                      << hostPairs[0].triangleID << std::endl;
        }
    }



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
    const nanovdb::Map map);
