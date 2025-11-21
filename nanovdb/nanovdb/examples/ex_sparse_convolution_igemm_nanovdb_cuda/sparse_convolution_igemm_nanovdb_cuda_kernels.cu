// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <nanovdb/tools/cuda/PointsToGrid.cuh>
#include <nanovdb/tools/cuda/MergeGrids.cuh>

template<typename T>
bool bufferCheck(const T* deviceBuffer, const T* hostBuffer, size_t elem_count) {
    T* tmpBuffer = new T[elem_count];
    cudaCheck(cudaMemcpy(tmpBuffer, deviceBuffer, elem_count * sizeof(T), cudaMemcpyDeviceToHost));
    bool same = true;
    for (int i=0; same && i< elem_count; ++i) { same = (tmpBuffer[i] == hostBuffer[i]); }
    delete [] tmpBuffer;
    return same;
}

template<class BufferT>
void printGridDiagnostics(nanovdb::GridHandle<BufferT>& handle)
{
    using BuildT = nanovdb::ValueOnIndex;

    auto deviceGrid = handle.template deviceGrid<BuildT>();
    if (!deviceGrid) throw std::logic_error("No GPU grid found in printGridDiagnostics()");

    auto valueCount = nanovdb::util::cuda::DeviceGridTraits<BuildT>::getValueCount(deviceGrid);
    auto treeData = nanovdb::util::cuda::DeviceGridTraits<BuildT>::getTreeData(deviceGrid);
    auto gridSize = nanovdb::util::cuda::DeviceGridTraits<BuildT>::getGridSize(deviceGrid);
    auto indexBBox = nanovdb::util::cuda::DeviceGridTraits<BuildT>::getIndexBBox(deviceGrid, treeData);

    std::cout << "======= Grid info =======" << std::endl;
    std::cout << "Allocated values         : " << valueCount << std::endl;
    std::cout << "Active voxels            : " << treeData.mVoxelCount << std::endl;
    auto minCorner = indexBBox.min(), maxCorner = indexBBox.max();
    std::cout << "Index-space bounding box : [" << minCorner.x() << "," << minCorner.y() << "," << minCorner.z()
              << "] -> [" << maxCorner.x() << "," << maxCorner.y() << "," << maxCorner.z() << "]" << std::endl;
    std::cout << "Leaf nodes               : " << treeData.mNodeCount[0] << std::endl;
    std::cout << "Lower internal nodes     : " << treeData.mNodeCount[1] << std::endl;
    std::cout << "Upper internal nodes     : " << treeData.mNodeCount[2] << std::endl;
    std::cout << "Leaf-level occupancy     : "
              << 100.f * (float)(treeData.mVoxelCount)/(float)(treeData.mNodeCount[0] * 512)
              << "%" << std::endl;
    std::cout << "Memory usage             : " << gridSize << " bytes" << std::endl;
}

void mainSparseConvolutionIGEMM(
    const std::vector<nanovdb::Coord>& inputPoints,
    const std::vector<nanovdb::Coord>& outputPoints,
    uint32_t benchmark_iters)
{
    using BuildT = nanovdb::ValueOnIndex;

    nanovdb::util::cuda::Timer gpuTimer;

    gpuTimer.start("Building input grid");
    auto inputBuffer = nanovdb::cuda::DeviceBuffer::create( inputPoints.size() * sizeof(nanovdb::Coord), nullptr, false);
    cudaCheck(cudaMemcpy(inputBuffer.deviceData(), inputPoints.data(), inputPoints.size() * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice));
    nanovdb::tools::cuda::PointsToGrid<BuildT> converter;
    converter.setChecksum(nanovdb::CheckMode::Default);
    auto inputHandle = converter.getHandle(static_cast<nanovdb::Coord*>(inputBuffer.deviceData()), inputPoints.size());
    auto inputGrid = inputHandle.deviceGrid<BuildT>();
    gpuTimer.stop();

    std::cout << "Input Grid Diagnostics:" << std::endl;
    printGridDiagnostics(inputHandle);

    gpuTimer.start("Building output grid");
    auto outputBuffer = nanovdb::cuda::DeviceBuffer::create( outputPoints.size() * sizeof(nanovdb::Coord), nullptr, false);
    cudaCheck(cudaMemcpy(outputBuffer.deviceData(), outputPoints.data(), outputPoints.size() * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice));
    converter.setChecksum(nanovdb::CheckMode::Default);
    auto outputHandle = converter.getHandle(static_cast<nanovdb::Coord*>(outputBuffer.deviceData()), outputPoints.size());
    auto outputGrid = outputHandle.deviceGrid<BuildT>();
    gpuTimer.stop();

    std::cout << "Output Grid Diagnostics:" << std::endl;
    printGridDiagnostics(outputHandle);

    // Initialize merger
    nanovdb::tools::cuda::MergeGrids<BuildT> merger( inputGrid, outputGrid );
    merger.setChecksum(nanovdb::CheckMode::Default);
    merger.setVerbose(0);
    auto mergedHandle = merger.getHandle();
    printGridDiagnostics(mergedHandle);


#if 0
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
