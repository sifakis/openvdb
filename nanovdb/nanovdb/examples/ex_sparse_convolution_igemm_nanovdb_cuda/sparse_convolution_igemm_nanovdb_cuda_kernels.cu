// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <nanovdb/cuda/UnifiedBuffer.h>
#include <nanovdb/tools/cuda/PointsToGrid.cuh>
#include <nanovdb/tools/cuda/MergeGrids.cuh>
#include <nanovdb/util/cuda/DeviceGridTraits.cuh>

#include <thrust/universal_vector.h>
#include <random>

#include "ampere_conv_kernel.h"

#define USE_HIERARCHICAL_BLOCK_TRAVERSAL

template<typename T>
bool bufferCheck(const T* deviceBuffer, const T* hostBuffer, size_t elem_count) {
    T* tmpBuffer = new T[elem_count];
    cudaCheck(cudaMemcpy(tmpBuffer, deviceBuffer, elem_count * sizeof(T), cudaMemcpyDeviceToHost));
    bool same = true;
    for (int i=0; same && i< elem_count; ++i) { same = (tmpBuffer[i] == hostBuffer[i]); }
    delete [] tmpBuffer;
    return same;
}

struct IGEMM_Geometry
{
    //
    // Convolution geometry
    //

    static constexpr int T = 3;     // X-dimension of convolution filter
    static constexpr int R = 3;     // Y-dimension of convolution filter
    static constexpr int S = 3;     // Z-dimension of convolution filter
    
    static constexpr int Z = 4;     // X-dimension of output block
    static constexpr int P = 2;     // Y-dimension of output block
    static constexpr int Q = 2;     // Z-dimension of output block

    static constexpr int D = Z+T-1; // X-dimension of input block (inluding halo)
    static constexpr int H = P+R-1; // Y-dimension of input block (inluding halo)
    static constexpr int W = Q+S-1; // Z-dimension of input block (inluding halo)

    static_assert(D==6, "Only convolution geometry supported is 4x2x2 block and 3x3x3 filter size");
    static_assert(H==4, "Only convolution geometry supported is 4x2x2 block and 3x3x3 filter size");
    static_assert(W==4, "Only convolution geometry supported is 4x2x2 block and 3x3x3 filter size");

    static constexpr int C = 64;    // Input feature dimension
    static constexpr int K = 128;   // Output feature dimension
    
    //
    // Leaf node geometry
    //

    static constexpr int ZZ = 1;         // Blocks of size (Z,P,Q) are grouped into "clusters" in a (ZZ,PP,QQ) arrangement
    static constexpr int PP = 2;         // I.e. ZZ blocks are grouped along the X-dimension, PP along the Y- and QQ along the Z-dimension
    static constexpr int QQ = 2;         // The total voxel size of a cluster will be (ZZ*Z,PP*P,QQ*Q)

    static constexpr int Bx = 8/Z;       // Block count along X-dimension of leaf node
    static constexpr int By = 8/P;       // Block count along Y-dimension of leaf node
    static constexpr int Bz = 8/Q;       // Block count along Z-dimension of leaf node

    static constexpr int Cx = 8/(ZZ*Z);  // Cluster count along X-dimension of leaf node
    static constexpr int Cy = 8/(PP*P);  // Cluster count along Y-dimension of leaf node
    static constexpr int Cz = 8/(QQ*Q);  // Cluster count along Z-dimension of leaf node

    static constexpr int Hx = T+7;       // X-dimension of leaf node domain, enlarged by the necessary halo for convolution
    static constexpr int Hy = R+7;       // Y-dimension of leaf node domain, enlarged by the necessary halo for convolution
    static constexpr int Hz = S+7;       // Z-dimension of leaf node domain, enlarged by the necessary halo for convolution

    static constexpr int CHx = ZZ*Z+T-1; // Cluster halo (voxel width, plus halo of one cluster) count along X-dimension
    static constexpr int CHy = PP*P+R-1; // Cluster halo (voxel width, plus halo of one cluster) count along X-dimension
    static constexpr int CHz = QQ*Q+S-1; // Cluster halo (voxel width, plus halo of one cluster) count along X-dimension

    static constexpr int VoxelsPerLeafnodeNoHalo() { return 512; }
    static constexpr int VoxelsPerLeafnodeWithHalo() { return Hx*Hy*Hz; }

    static constexpr int VoxelsPerClusterNoHalo() { return Z*P*Q*ZZ*PP*QQ; }
    static constexpr int VoxelsPerClusterWithHalo() { return CHx*CHy*CHz; }

    //
    // Filter offset (coordinate offset in the input domain that the [0,0,0] filter spoke corresponds to)
    //

    static constexpr int Dx = -1; // X-coordinate offset of the minimum corner of the convolution filter
    static constexpr int Dy = -1; // Y-coordinate offset of the minimum corner of the convolution filter
    static constexpr int Dz = -1; // Z-coordinate offset of the minimum corner of the convolution filter

};


template<class GeometryT, int Di, int Do, class ValueType>
void SparseConvolveCPUReference(
    nanovdb::NanoGrid<nanovdb::ValueOnIndex> *srcGrid,
    nanovdb::NanoGrid<nanovdb::ValueOnIndex> *dstGrid,
    const ValueType (*filter)[GeometryT::R][GeometryT::S][Do][Di],
    const ValueType (*inputArray)[Di],
    ValueType (*outputArray)[Do])
{
    auto dstLeafCount = dstGrid->nodeCount<0>();
    auto srcAcc = srcGrid->getAccessor();
#pragma omp parallel for firstPrivate(srcAcc)
    for ( int dstLeafID = 0; dstLeafID < dstLeafCount; ++dstLeafID )
    {
        auto& dstLeaf = dstGrid->tree().getFirstLeaf()[dstLeafID];
        for ( auto dstLeafIt = dstLeaf.cbeginValueOn(); dstLeafIt; ++dstLeafIt ) {
            const auto dstIndex = *dstLeafIt;
            const auto dstCoord = dstLeafIt.getCoord();
            for ( int i = 0; i < Do; ++i )
                outputArray[dstIndex][i] = 0.f;
            for ( int di = 0; di < GeometryT::T; ++di )
            for ( int dj = 0; dj < GeometryT::R; ++dj )
            for ( int dk = 0; dk < GeometryT::S; ++dk )
            {
                const auto srcCoord = dstCoord.offsetBy(di+GeometryT::Dx, dj+GeometryT::Dy, dk+GeometryT::Dz);
                const auto srcIndex = srcAcc.getValue(srcCoord);
                if (srcIndex)
                    for ( int out = 0; out < Do; ++out )
                    for ( int in  = 0; in  < Di; ++in  )
                        outputArray[dstIndex][out] += filter[di][dj][dk][out][in] * inputArray[srcIndex][in];
            }
        }
    }
}

template <typename Functor>
__global__
void lambda_kernel_wrapper(Functor func) { func(); }

template<class GeometryT, int Di, int Do, class ValueType>
void SparseConvolveCudaReference(
    nanovdb::NanoGrid<nanovdb::ValueOnIndex> *srcGrid,
    nanovdb::NanoGrid<nanovdb::ValueOnIndex> *dstGrid,
    const ValueType (*filter)[GeometryT::R][GeometryT::S][Do][Di],
    const ValueType (*inputArray)[Di],
    ValueType (*outputArray)[Do])
{
    auto dstLeafCount = dstGrid->nodeCount<0>();

    auto convolver = [=] __device__ () {
        int dstLeafID = blockIdx.x;
        int out = threadIdx.x;
        auto& dstLeaf = dstGrid->tree().getFirstLeaf()[dstLeafID];
        for ( auto dstLeafIt = dstLeaf.cbeginValueOn(); dstLeafIt; ++dstLeafIt ) {
            const auto dstIndex = *dstLeafIt;
            const auto dstCoord = dstLeafIt.getCoord();
            outputArray[dstIndex][out] = 0.f;
            for ( int di = 0; di < GeometryT::T; ++di )
            for ( int dj = 0; dj < GeometryT::R; ++dj )
            for ( int dk = 0; dk < GeometryT::S; ++dk )
            {
                const auto srcCoord = dstCoord.offsetBy(di+GeometryT::Dx, dj+GeometryT::Dy, dk+GeometryT::Dz);
                const auto srcIndex = srcGrid->tree().getValue(srcCoord);
                if (srcIndex)
                    for ( int in = 0; in < Di; ++in )
                        outputArray[dstIndex][out] += filter[di][dj][dk][out][in] * inputArray[srcIndex][in];
            }
        }
    };

    lambda_kernel_wrapper<<<dstLeafCount,Do>>>(convolver);
}

template<class GeometryT, int Di, int Do, class ValueType>
void SparseConvolveScatterGatherMapsReference(
    uint64_t (*gather_idx_buf) [GeometryT::D][GeometryT::H][GeometryT::W],
    uint64_t (*scatter_idx_buf)[GeometryT::Z][GeometryT::P][GeometryT::Q],
    const std::size_t blockCount,
    const ValueType (*filter)[GeometryT::R][GeometryT::S][Do][Di],
    const ValueType (*inputArray)[Di],
    ValueType (*outputArray)[Do])
{
    auto convolver = [=] __device__ () {
        int blockID = blockIdx.x;
        auto gatherIndices = gather_idx_buf[blockID];
        auto scatterIndices = scatter_idx_buf[blockID];
        int out = threadIdx.x;

        for ( int i = 0; i < GeometryT::Z; ++i )
        for ( int j = 0; j < GeometryT::P; ++j )
        for ( int k = 0; k < GeometryT::Q; ++k ) {
            const auto dstIndex = scatterIndices[i][j][k];
            if (dstIndex) {
                outputArray[dstIndex][out] = 0.f;
                for ( int di = 0; di < GeometryT::T; ++di )
                for ( int dj = 0; dj < GeometryT::R; ++dj )
                for ( int dk = 0; dk < GeometryT::S; ++dk ) {
                    const auto srcIndex = gatherIndices[i+di][j+dj][k+dk];
                    if (srcIndex)
                        for ( int in = 0; in < Di; ++in )
                            outputArray[dstIndex][out] += filter[di][dj][dk][out][in] * inputArray[srcIndex][in];
                }
            }
        }
    };

    lambda_kernel_wrapper<<<blockCount,Do>>>(convolver);
}

template<int Do, class ValueType>
void ResultCompare(
    const std::size_t size,
    const ValueType (*outputArray1)[Do],
    const ValueType (*outputArray2)[Do]
)
{
    ValueType result = 0.f;
#pragma omp parallel for reduction(max:result)
    for (int i = 0; i < size; i++)
        for (int j = 0; j < Do; j++)
            result = std::max(result, std::abs(outputArray1[i][j]-outputArray2[i][j]));
    std::cout << "Discrepancy = " << result << std::endl;
}

template<class Operator, class BuildT, class FilterTensor>
__global__
__launch_bounds__(Operator::MaxThreadsPerBlock, Operator::MinBlocksPerMultiprocessor)
  void kernel_entrypoint_custom(
      FilterTensor mFlt,
      const nanovdb::NanoGrid<BuildT>* inputGrid,
      const nanovdb::NanoGrid<BuildT>* outputGrid,
      const float *inputData,
      float *outputData
  ) {
  extern __shared__ char smem_buf[];
  Operator op;
  op(
      mFlt,
      inputGrid,
      outputGrid,
      inputData,
      outputData,
      smem_buf);
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
    using BufferT = nanovdb::cuda::UnifiedBuffer;
    static constexpr int Di = 64;
    static constexpr int Do = 128;
    using inputArrayT = float (&) [][Di];
    using outputArrayT = float (&) [][Do];
    using filterT = float (&) [3][3][3][Do][Di];
    
    nanovdb::util::cuda::Timer gpuTimer;

    gpuTimer.start("Building input grid");
    auto inputBuffer = BufferT::create( inputPoints.size() * sizeof(nanovdb::Coord), nullptr, false);
    cudaCheck(cudaMemcpy(inputBuffer.deviceData(), inputPoints.data(), inputPoints.size() * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice));
    nanovdb::tools::cuda::PointsToGrid<BuildT> converter;
    converter.setChecksum(nanovdb::CheckMode::Default);
    auto inputHandle = converter.getHandle<nanovdb::Coord*, BufferT>(static_cast<nanovdb::Coord*>(inputBuffer.deviceData()), inputPoints.size());
    auto inputGrid = inputHandle.deviceGrid<BuildT>();
    gpuTimer.stop();

    std::cout << "Input Grid Diagnostics:" << std::endl;
    printGridDiagnostics(inputHandle);

    gpuTimer.start("Building output grid");
    auto outputBuffer = BufferT::create( outputPoints.size() * sizeof(nanovdb::Coord), nullptr, false);
    cudaCheck(cudaMemcpy(outputBuffer.deviceData(), outputPoints.data(), outputPoints.size() * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice));
    converter.setChecksum(nanovdb::CheckMode::Default);
    auto outputHandle = converter.getHandle<nanovdb::Coord*, BufferT>(static_cast<nanovdb::Coord*>(outputBuffer.deviceData()), outputPoints.size());
    auto outputGrid = outputHandle.deviceGrid<BuildT>();
    gpuTimer.stop();

    std::cout << "Output Grid Diagnostics:" << std::endl;
    printGridDiagnostics(outputHandle);

    // Initialize merger
    gpuTimer.start("Merging input/output grids (for testing)");
    nanovdb::tools::cuda::MergeGrids<BuildT> merger( inputGrid, outputGrid );
    merger.setChecksum(nanovdb::CheckMode::Default);
    merger.setVerbose(0);
    auto mergedHandle = merger.getHandle();
    gpuTimer.stop();

    std::cout << "Merged Grid Diagnostics:" << std::endl;
    printGridDiagnostics(mergedHandle);

    // Allocate and initialize benchmark data

    std::random_device rd;
    // std::mt19937 generator(rd());
    std::mt19937 generator(23456);
    std::uniform_int_distribution<int> distribution(-256, 256);

    gpuTimer.start("Initializing input (activation) data");
    auto inputValueCount = nanovdb::util::cuda::DeviceGridTraits<BuildT>::getValueCount(inputGrid);
    auto inputVoxelCount = nanovdb::util::cuda::DeviceGridTraits<BuildT>::getActiveVoxelCount(inputGrid);
    auto inputData = thrust::universal_vector<float>(inputValueCount*Di);
    auto inputArray = reinterpret_cast<inputArrayT>(*inputData.data().get());
    for (int i = 0; i < Di; i++) inputArray[0][i] = 0.f;
#pragma omp parallel for
    for (int v = 0; v <= inputVoxelCount; v++)
        for (int i = 0; i < Di; i++)
            inputArray[v][i] = ((float)distribution(generator))/256.0f; // Use only up to 7 bits in the mantissa
    gpuTimer.stop();
    
    gpuTimer.start("Initializing output (including reference) data");
    auto outputValueCount = nanovdb::util::cuda::DeviceGridTraits<BuildT>::getValueCount(outputGrid);
    auto outputVoxelCount = nanovdb::util::cuda::DeviceGridTraits<BuildT>::getActiveVoxelCount(outputGrid);
    auto outputData = thrust::universal_vector<float>(outputValueCount*Do);
    auto outputArray = reinterpret_cast<outputArrayT>(*outputData.data().get());
    auto outputReferenceData = thrust::universal_vector<float>(outputValueCount*Do);
    auto outputReferenceArray = reinterpret_cast<outputArrayT>(*outputReferenceData.data().get());
#pragma omp parallel for
    for (int v = 1; v <= inputValueCount; v++)
        for (int i = 0; i < Di; i++)
            outputArray[v][i] = outputReferenceArray[v][i] = 0.f;
    for (int i = 0; i < Di; i++)
        outputArray[0][i] = outputReferenceArray[0][i] = ((float)distribution(generator))/256.0f; // Use only up to 7 bits in the mantissa   
    gpuTimer.stop();

    gpuTimer.start("Initializing filter data");
    auto filterData = thrust::universal_vector<float>(3*3*3*Do*Di);
    auto filter = reinterpret_cast<filterT>(*filterData.data().get());
#pragma omp parallel for
    for (int i = 0; i < filterData.size(); i++)
        filterData[i] = ((float)distribution(generator))/256.0f; // Use only up to 7 bits in the mantissa
    gpuTimer.stop();

    gpuTimer.start("Initializing scatter indices");
    auto outputLeafCount = outputGrid->tree().nodeCount(0);
    auto blockCount = outputLeafCount
        * IGEMM_Geometry::Bx * IGEMM_Geometry::By * IGEMM_Geometry::Bz;

    using ConvOp = AmperePredicatedFprop<IGEMM_Geometry>;
#ifdef USE_HIERARCHICAL_BLOCK_TRAVERSAL
    auto leafShape = make_shape(Int<IGEMM_Geometry::Bx>{},  Int<IGEMM_Geometry::By>{},  Int<IGEMM_Geometry::Bz>{});
    auto blockedLeafShape = shape(zipped_divide(make_layout(leafShape), ConvOp::Tiler_N{}));
    auto blockedLeafLayout = make_ordered_layout(
        blockedLeafShape,
        make_tuple(make_tuple(_2{},_1{},_0{}),make_tuple(_5{},_4{},_3{})));
#endif

    auto outputVoxelsPerBlock = IGEMM_Geometry::Z * IGEMM_Geometry::P * IGEMM_Geometry::Q;
    using ScatterIndexLegacyT = uint64_t [IGEMM_Geometry::Z][IGEMM_Geometry::P][IGEMM_Geometry::Q];
#ifndef USE_HIERARCHICAL_BLOCK_TRAVERSAL
    using ScatterIndexArrayLegacyT = ScatterIndexLegacyT [IGEMM_Geometry::Bx][IGEMM_Geometry::By][IGEMM_Geometry::Bz];
#else
    using ScatterIndexArrayLegacyT = ScatterIndexLegacyT [IGEMM_Geometry::Bx*IGEMM_Geometry::By*IGEMM_Geometry::Bz];
#endif
    auto scatterIndexDataLegacy = thrust::universal_vector<uint64_t>(blockCount*outputVoxelsPerBlock);
    auto scatterIndexArrayLegacy = reinterpret_cast<ScatterIndexArrayLegacyT*>(scatterIndexDataLegacy.data().get());

    // Per-leaf non-halo index map
    using ScatterIndexArrayT = uint64_t [8][8][8];
    auto scatterIndexData = thrust::universal_vector<uint64_t>(outputLeafCount*512);
    auto scatterIndexArray = reinterpret_cast<ScatterIndexArrayT*>(scatterIndexData.data().get());

#pragma omp parallel for
    for (int l = 0; l < outputLeafCount; l++) {
        auto &leaf = outputGrid->tree().getFirstLeaf()[l];
#ifndef USE_HIERARCHICAL_BLOCK_TRAVERSAL
        for (int bi = 0; bi < IGEMM_Geometry::Bx; bi++)
        for (int bj = 0; bj < IGEMM_Geometry::By; bj++)
        for (int bk = 0; bk < IGEMM_Geometry::Bz; bk++) {
#else
    for (int bbi = 0; bbi < shape<1,0>(blockedLeafLayout); ++bbi)
    for (int bbj = 0; bbj < shape<1,1>(blockedLeafLayout); ++bbj)
    for (int bbk = 0; bbk < shape<1,2>(blockedLeafLayout); ++bbk)
    for (int bii = 0; bii < shape<0,0>(blockedLeafLayout); ++bii)
    for (int bjj = 0; bjj < shape<0,1>(blockedLeafLayout); ++bjj)
    for (int bkk = 0; bkk < shape<0,2>(blockedLeafLayout); ++bkk)
    {
        int bi = bbi * shape<0,0>(blockedLeafLayout) + bii;
        int bj = bbj * shape<0,1>(blockedLeafLayout) + bjj;
        int bk = bbk * shape<0,2>(blockedLeafLayout) + bkk;
#endif
            nanovdb::Coord blockOffset(bi*IGEMM_Geometry::Z, bj*IGEMM_Geometry::P, bk*IGEMM_Geometry::Q);
            for (int i = 0; i < IGEMM_Geometry::Z; i++)
            for (int j = 0; j < IGEMM_Geometry::P; j++)
            for (int k = 0; k < IGEMM_Geometry::Q; k++) {
                auto localCoord = blockOffset.offsetBy(i,j,k);
#ifndef USE_HIERARCHICAL_BLOCK_TRAVERSAL
                scatterIndexArrayLegacy[l][bi][bj][bk][i][j][k] = leaf.getValue(localCoord);
#else
                scatterIndexArrayLegacy[l]
                    [blockedLeafLayout(make_tuple(bii, bjj, bkk), make_tuple(bbi, bbj, bbk))]
                    [i][j][k] = leaf.getValue(localCoord);
#endif
            }
        }
        for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
        for (int k = 0; k < 8; k++) {
            scatterIndexArray[l][i][j][k] = leaf.getValue(nanovdb::Coord(i,j,k));
        }
    }
    gpuTimer.stop();

    gpuTimer.start("Initializing gather indices");
    auto inputVoxelsPerBlock = IGEMM_Geometry::D * IGEMM_Geometry::H * IGEMM_Geometry::W;
    using GatherIndexLegacyT = uint64_t [IGEMM_Geometry::D][IGEMM_Geometry::H][IGEMM_Geometry::W];
#ifndef USE_HIERARCHICAL_BLOCK_TRAVERSAL
    using GatherIndexArrayLegacyT = GatherIndexLegacyT [IGEMM_Geometry::Bx][IGEMM_Geometry::By][IGEMM_Geometry::Bz];
#else
    using GatherIndexArrayLegacyT = GatherIndexLegacyT [IGEMM_Geometry::Bx*IGEMM_Geometry::By*IGEMM_Geometry::Bz];
#endif
    auto gatherIndexDataLegacy = thrust::universal_vector<uint64_t>(blockCount*inputVoxelsPerBlock);
    auto gatherIndexArrayLegacy = reinterpret_cast<GatherIndexArrayLegacyT*>(gatherIndexDataLegacy.data().get());

    // Per-leaf halo index map
    using GatherIndexArrayT = uint64_t [IGEMM_Geometry::Hx][IGEMM_Geometry::Hy][IGEMM_Geometry::Hz];
    auto gatherIndexData = thrust::universal_vector<uint64_t>(outputLeafCount*IGEMM_Geometry::Hx*IGEMM_Geometry::Hy*IGEMM_Geometry::Hz);
    auto gatherIndexArray = reinterpret_cast<GatherIndexArrayT*>(gatherIndexData.data().get());

#pragma omp parallel for
    for (int l = 0; l < outputLeafCount; l++) {
        auto &outputLeaf = outputGrid->tree().getFirstLeaf()[l];
        const auto origin = outputLeaf.origin();
#ifndef USE_HIERARCHICAL_BLOCK_TRAVERSAL
        for (int bi = 0; bi < IGEMM_Geometry::Bx; bi++)
        for (int bj = 0; bj < IGEMM_Geometry::By; bj++)
        for (int bk = 0; bk < IGEMM_Geometry::Bz; bk++) {
#else
    for (int bbi = 0; bbi < shape<1,0>(blockedLeafLayout); ++bbi)
    for (int bbj = 0; bbj < shape<1,1>(blockedLeafLayout); ++bbj)
    for (int bbk = 0; bbk < shape<1,2>(blockedLeafLayout); ++bbk)
    for (int bii = 0; bii < shape<0,0>(blockedLeafLayout); ++bii)
    for (int bjj = 0; bjj < shape<0,1>(blockedLeafLayout); ++bjj)
    for (int bkk = 0; bkk < shape<0,2>(blockedLeafLayout); ++bkk)
    {
        int bi = bbi * shape<0,0>(blockedLeafLayout) + bii;
        int bj = bbj * shape<0,1>(blockedLeafLayout) + bjj;
        int bk = bbk * shape<0,2>(blockedLeafLayout) + bkk;
#endif
            nanovdb::Coord blockOffset(bi*IGEMM_Geometry::Z, bj*IGEMM_Geometry::P, bk*IGEMM_Geometry::Q);
            for (int i = 0; i < IGEMM_Geometry::D; i++)
            for (int j = 0; j < IGEMM_Geometry::H; j++)
            for (int k = 0; k < IGEMM_Geometry::W; k++) {
                auto localCoord = blockOffset.offsetBy(i+IGEMM_Geometry::Dx,j+IGEMM_Geometry::Dy,k+IGEMM_Geometry::Dz);
                auto globalCoord = origin+localCoord;
#ifndef USE_HIERARCHICAL_BLOCK_TRAVERSAL
                gatherIndexArrayLegacy[l][bi][bj][bk][i][j][k] = inputGrid->tree().getValue(globalCoord);
#else
                gatherIndexArrayLegacy[l]
                    [blockedLeafLayout(make_tuple(bii, bjj, bkk), make_tuple(bbi, bbj, bbk))]
                    [i][j][k] = inputGrid->tree().getValue(globalCoord);
#endif
            }
        }

        auto offsetOrigin = origin.offsetBy(IGEMM_Geometry::Dx, IGEMM_Geometry::Dy, IGEMM_Geometry::Dz);
        for (int i = 0; i < IGEMM_Geometry::Hx; ++i)
        for (int j = 0; j < IGEMM_Geometry::Hy; ++j)
        for (int k = 0; k < IGEMM_Geometry::Hz; ++k)
            gatherIndexArray[l][i][j][k] = inputGrid->tree().getValue(offsetOrigin+nanovdb::Coord(i,j,k));
    }
    gpuTimer.stop();
    
    gpuTimer.start("Reference (GPU) execution");
    SparseConvolveCudaReference<IGEMM_Geometry, Di, Do>(
        inputGrid,
        outputGrid,
        filter,
        inputArray,
        outputReferenceArray
    );
    gpuTimer.stop();


#if 0
    // CPU version; may be extremely slow for all but the smallest resolutions
    gpuTimer.start("Reference (CPU) execution");
    SparseConvolveCPUReference<IGEMM_Geometry, Di, Do>(
        inputGrid,
        outputGrid,
        filter,
        inputArray,
        outputArray
    );
    gpuTimer.stop();
#endif

#if 0
    gpuTimer.start("Reference (Gather-Scatter) execution");
    SparseConvolveScatterGatherMapsReference<IGEMM_Geometry, Di, Do>(
        reinterpret_cast<GatherIndexLegacyT*>(gatherIndexArrayLegacy),
        reinterpret_cast<ScatterIndexLegacyT*>(scatterIndexArrayLegacy),
        blockCount,
        filter,
        inputArray,
        outputArray
    );
    gpuTimer.stop();

    ResultCompare<Do>(
        outputValueCount,
        outputArray,
        outputReferenceArray
    );
#endif

    IGEMM_Layouts<IGEMM_Geometry> layouts;

    Tensor tFilter = make_tensor(
        make_gmem_ptr(filterData.data().get()),
        layouts.filterLayout()
    );

    constexpr size_t smem_size = sizeof(typename AmperePredicatedFprop<IGEMM_Geometry>::SharedStorage);
    std::cout << "smem_size = " << smem_size << std::endl;

    cudaCheck(
        cudaFuncSetAttribute(
            kernel_entrypoint_custom<AmperePredicatedFprop<IGEMM_Geometry>, BuildT, decltype(tFilter)>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size
        ));

    int num_iterations = 10;
    for (int i = 0; i < num_iterations; ++i) {
        gpuTimer.start("Scatter-Gather Cutlass IGEMM (GPU) execution");
        kernel_entrypoint_custom<AmperePredicatedFprop<IGEMM_Geometry>, BuildT, decltype(tFilter)>
            <<<outputLeafCount, AmperePredicatedFprop<IGEMM_Geometry>::MaxThreadsPerBlock, smem_size>>>(
                tFilter,
                inputGrid,
                outputGrid,
                inputData.data().get(),
                outputData.data().get()
            );
        gpuTimer.stop();
    }

    // Potentially needed due to unified memory synchronization
    // cudaDeviceSynchronize();

    ResultCompare<Do>(
        outputValueCount,
        outputArray,
        outputReferenceArray
    );

}
