// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <nanovdb/cuda/UnifiedBuffer.h>
#include <nanovdb/tools/cuda/PointsToGrid.cuh>
#include <nanovdb/tools/cuda/MergeGrids.cuh>
#include <nanovdb/util/cuda/DeviceGridTraits.cuh>

#include <thrust/universal_vector.h>
#include <random>

#include "ampere_conv_kernel.h"
#include "gather_tensor.hpp"

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

    static constexpr int ZZ = 1;        // Blocks of size (Z,P,Q) are grouped into "clusters" in a (ZZ,PP,QQ) arrangement
    static constexpr int PP = 2;        // I.e. ZZ blocks are grouped along the X-dimension, PP along the Y- and QQ along the Z-dimension
    static constexpr int QQ = 2;        // The total voxel size of a cluster will be (ZZ*Z,PP*P,QQ*Q)

    static constexpr int Bx = 8/Z;      // Block count along X-dimension of leaf node
    static constexpr int By = 8/P;      // Block count along Y-dimension of leaf node
    static constexpr int Bz = 8/Q;      // Block count along Z-dimension of leaf node

    static constexpr int Cx = 8/(ZZ*Z); // Cluster count along X-dimension of leaf node
    static constexpr int Cy = 8/(PP*P); // Cluster count along Y-dimension of leaf node
    static constexpr int Cz = 8/(QQ*Q); // Cluster count along Z-dimension of leaf node

    static constexpr int Hx = T+7;      // X-dimension of leaf node domain, enlarged by the necessary halo for convolution
    static constexpr int Hy = R+7;      // Y-dimension of leaf node domain, enlarged by the necessary halo for convolution
    static constexpr int Hz = S+7;      // Z-dimension of leaf node domain, enlarged by the necessary halo for convolution

    static constexpr int VoxelsPerLeafnodeNoHalo() { return 512; }
    static constexpr int VoxelsPerLeafnodeWithHalo() { return Hx*Hy*Hz; }

    //
    // Filter offset (coordinate offset in the input domain that the [0,0,0] filter spoke corresponds to)
    //

    static constexpr int Dx = -1; // X-coordinate offset of the minimum corner of the convolution filter
    static constexpr int Dy = -1; // Y-coordinate offset of the minimum corner of the convolution filter
    static constexpr int Dz = -1; // Z-coordinate offset of the minimum corner of the convolution filter

};


template<class GeometryT>
struct IGEMM_Layouts
{
    static constexpr auto T = Int<GeometryT::T>{};
    static constexpr auto R = Int<GeometryT::R>{};
    static constexpr auto S = Int<GeometryT::S>{};
    static constexpr auto Z = Int<GeometryT::Z>{};
    static constexpr auto P = Int<GeometryT::P>{};
    static constexpr auto Q = Int<GeometryT::Q>{};
    static constexpr auto D = Int<GeometryT::D>{};
    static constexpr auto H = Int<GeometryT::H>{};
    static constexpr auto W = Int<GeometryT::W>{};
    static constexpr auto C = Int<GeometryT::C>{};
    static constexpr auto K = Int<GeometryT::K>{};
    static constexpr auto Bx = Int<GeometryT::Bx>{};
    static constexpr auto By = Int<GeometryT::By>{};
    static constexpr auto Bz = Int<GeometryT::Bz>{};
    static constexpr auto Hx = Int<GeometryT::Hx>{};
    static constexpr auto Hy = Int<GeometryT::Hy>{};
    static constexpr auto Hz = Int<GeometryT::Hz>{};

    static auto xformedActivationComposedLayoutLegacy(const int N, const uint64_t* gather_idx_buf)
    {
        // Input gather layout
        // inner_layout(make_coord((nzpq), (csrt))) => (idx_buffer_idx, dense_c_idx)
        auto EG = E<0>{};  // Gather basis     (1,0) (idx_buffer_idx) 
        auto EC = E<1>{};  // Contiguous basis (0,1) (dense_offset)    
        auto xformed_act_logical_inner = make_layout(
            make_shape (make_shape (       N,      Z,    P,  Q), make_shape ( C,      T,    R,  S)),
            make_stride(make_stride(D*H*W*EG, H*W*EG, W*EG, EG), make_stride(EC, H*W*EG, W*EG, EG)));

        // outer_layout(make_coord(idx_buffer_idx, dense_c_idx)) => idx
        // IndexedGather obtains idx by applying (gmem_base_ptr + gather_idx_buf[idx_buffer_idx] + dense_offset)
        auto xformed_act_gather_outer = make_layout(
            make_shape(_1{},_1{}),
            make_stride(example::CustomStride{example::IndexedGather{gather_idx_buf}, C}, _1{}));

        // Compose the inner and outer layouts
        // gather_composed(make_coord((nzpq), (csrt))) => idx
        return composition(
            xformed_act_gather_outer,
            make_arithmetic_tuple(_0{}, _0{}),
            xformed_act_logical_inner);
    }

    static auto xformedActivationComposedLayout(const int N, const uint64_t* gather_idx_buf)
    {
        // Input gather layout
        // inner_layout(make_coord((nzpq), (csrt))) => (idx_buffer_idx, dense_c_idx)
        auto EG = E<0>{};  // Gather basis     (1,0) (idx_buffer_idx) 
        auto EC = E<1>{};  // Contiguous basis (0,1) (dense_offset)    
        auto xformed_act_logical_inner = make_layout(
            make_shape (make_shape (make_shape (          N,         Bx,      By,   Bz),        Z,     P,  Q), make_shape ( C,        T,     R,  S)),
            make_stride(make_stride(make_stride(Hx*Hy*Hz*EG, Hy*Hz*Z*EG, Hz*P*EG, Q*EG), Hy*Hz*EG, Hz*EG, EG), make_stride(EC, Hy*Hz*EG, Hz*EG, EG)));

        // outer_layout(make_coord(idx_buffer_idx, dense_c_idx)) => idx
        // IndexedGather obtains idx by applying (gmem_base_ptr + gather_idx_buf[idx_buffer_idx] + dense_offset)
        auto xformed_act_gather_outer = make_layout(
            make_shape(_1{},_1{}),
            make_stride(example::CustomStride{example::IndexedGather{gather_idx_buf}, C}, _1{}));

        // Compose the inner and outer layouts
        // gather_composed(make_coord((nzpq), (csrt))) => idx
        return composition(
            xformed_act_gather_outer,
            make_arithmetic_tuple(_0{}, _0{}),
            xformed_act_logical_inner);
    }

    static auto gatherIndexLayoutLegacy(const int N)
    {
        // Input gather index layout
        // gather_layout_index(make_coord((ndhw), c)) => buffer_idx
        return make_layout(
            make_shape (make_shape (    N,   D, H,  W  ), make_shape ( C  , _1{}, _1{}, _1{})),
            make_stride(make_stride(D*H*W, H*W, W, _1{}), make_stride(_0{}, _0{}, _0{}, _0{})));
    }

    static auto gatherIndexLayout(const int N)
    {
        // Input gather index layout
        // gather_layout_index(make_coord((ndhw), c)) => buffer_idx
        return make_layout(
            make_shape (make_shape (make_shape (       N,      Bx,   By, Bz),     Z,  P,    Q), make_shape (   C,     T,  R,    S)),
            make_stride(make_stride(make_stride(Hx*Hy*Hz, Hy*Hz*Z, Hz*P,  Q), Hy*Hz, Hz, _1{}), make_stride(_0{}, Hy*Hz, Hz, _1{})));
    }

    static auto filterLayout()
    {
        return make_ordered_layout(
            make_shape(K, make_shape(C, T, R, S)),
            tuple<_1, tuple<_0,_4,_3,_2>>{}
        );
    }

    static auto xformedOutputComposedLayoutLegacy(const int N, const uint64_t* scatter_idx_buf)
    {
        // Output scatter layout
        // scatter_layout_index(k, make_coord((nzpq))) => buffer_idx
        auto ES = E<0>{};  // Scatter basis    (1,0) (idx_buffer_idx)
        auto EC = E<1>{};  // Contiguous basis (0,1) (dense_offset)
        auto xformed_out_logical_inner = make_layout(
            make_shape ( K, make_shape(        N,      Z,    P,  Q)),
            make_stride(EC, make_stride(Z*P*Q*ES, P*Q*ES, Q*ES, ES)));
        auto xformed_out_scatter_outer = make_layout(
            make_shape(_1{},_1{}),
            make_stride(example::CustomStride{example::IndexedGather{scatter_idx_buf}, K}, _1{}));
        return composition(
            xformed_out_scatter_outer,
            make_arithmetic_tuple(_0{},_0{}),
            xformed_out_logical_inner);
    }

    static auto xformedOutputComposedLayout(const int N, const uint64_t* scatter_idx_buf)
    {
        // Output scatter layout
        // scatter_layout_index(k, make_coord((nzpq))) => buffer_idx
        auto ES = E<0>{};  // Scatter basis    (1,0) (idx_buffer_idx)
        auto EC = E<1>{};  // Contiguous basis (0,1) (dense_offset)
        auto xformed_out_logical_inner = make_layout(
            make_shape ( K, make_shape (make_shape (        N,         Bx,        By,   Bz),        Z,       P,  Q)),
            make_stride(EC, make_stride(make_stride(_512{}*ES, _64{}*Z*ES, _8()*P*ES, Q*ES), _64{}*ES, _8{}*ES, ES)));
        auto xformed_out_scatter_outer = make_layout(
            make_shape(_1{},_1{}),
            make_stride(example::CustomStride{example::IndexedGather{scatter_idx_buf}, K}, _1{}));
        return composition(
            xformed_out_scatter_outer,
            make_arithmetic_tuple(_0{},_0{}),
            xformed_out_logical_inner);
    }

    static auto scatterIndexLayoutLegacy(const int N)
    {
        // Output scatter index layout
        // scatter_layout_index(k, make_coord((nzpq))) => buffer_idx
        return make_layout(
            make_shape (   K, make_shape (    N,   Z, P,  Q  )),
            make_stride(_0{}, make_stride(Z*P*Q, P*Q, Q, _1{})));
    }

    static auto scatterIndexLayout(const int N)
    {
        // Output scatter index layout
        // scatter_layout_index(k, make_coord((nzpq))) => buffer_idx
        return make_layout(
            make_shape (   K, make_shape (make_shape (     N,      Bx,     By, Bz),     Z,    P,    Q)),
            make_stride(_0{}, make_stride(make_stride(_512{}, _64{}*Z, _8{}*P,  Q), _64{}, _8{}, _1{})));
    }

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

template<class Operator, class FilterTensor,
    class ActivationTensor,
    class ActivationTensorIndex,
    class OutputTensor,
    class OutputTensorIndex
    >
__global__
__launch_bounds__(Operator::MaxThreadsPerBlock, Operator::MinBlocksPerMultiprocessor)
  void kernel_entrypoint_custom(
      FilterTensor mFlt,
      ActivationTensor mAct,
      ActivationTensorIndex mActI,
      OutputTensor mOut,
      OutputTensorIndex mOutI,
      const float *inputData,
      float *outputData
  ) {
  extern __shared__ char smem_buf[];
  Operator op;
  op(
      mFlt,
      mAct,
      mActI,
      mOut,
      mOutI,
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
    auto blockedLeafShape = shape(zipped_divide(make_layout(leafShape), take<1,4>(ConvOp::Tiler_N{})));
    auto blockedLeafLayout = make_ordered_layout(
        blockedLeafShape,
        make_tuple(make_tuple(_2{},_1{},_0{}),make_tuple(_5{},_4{},_3{})));

#if 0
    print("\n");
    print("leafShape=");print(leafShape);print("\n");
    print("blockedLeafShape=");print(blockedLeafShape);print("\n");
    print("blockedLeafLayout=");print(blockedLeafLayout);print("\n");
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
        printf("((%d,%d,%d),(%d,%d,%d)) -> %d (%d,%d,%d)\n", bii, bjj, bkk, bbi, bbj, bbk,
            blockedLeafLayout(make_tuple(bii, bjj, bkk), make_tuple(bbi, bbj, bbk)),
            bi, bj, bk
        );
    }
#endif        
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

    Tensor tXformedActGatherLegacy = make_tensor(
        make_gmem_ptr(inputData.data().get()),
        layouts.xformedActivationComposedLayoutLegacy(blockCount, gatherIndexDataLegacy.data().get())
    );

    Tensor tXformedActGather = make_tensor(
        make_gmem_ptr(inputData.data().get()),
        layouts.xformedActivationComposedLayout(outputLeafCount, gatherIndexData.data().get())
    );

    Tensor tGatherIndexLegacy = make_tensor(
        make_gmem_ptr(gatherIndexDataLegacy.data().get()),
        layouts.gatherIndexLayoutLegacy(blockCount)
    );

    Tensor tGatherIndex = make_tensor(
        make_gmem_ptr(gatherIndexData.data().get()),
        layouts.gatherIndexLayout(outputLeafCount)
    );


#if 0
    auto tXformedActGatherTiled = local_tile(tXformedActGather, ConvOp::TilerAct{}, make_coord(_,_));
    auto tGatherIndexTiled = local_tile(tGatherIndex, ConvOp::TilerAct{}, make_coord(_,_));
    // print("tXformedActGather.layout() = ");print(tXformedActGather.layout());print("\n");
    // print("tXformedActGatherTiled.layout() = ");print(tXformedActGatherTiled.layout());print("\n");
    // print("tGatherIndex.layout() = ");print(tGatherIndex.layout());print("\n");
    // print("tGatherIndexTiled.layout() = ");print(tGatherIndexTiled.layout());print("\n");

    for (int l = 0; l < outputLeafCount; ++l)
        for (int bbi = 0; bbi < size<2,0,1>(tGatherIndexTiled); ++bbi)
        for (int bbj = 0; bbj < size<2,0,2>(tGatherIndexTiled); ++bbj)
        for (int bbk = 0; bbk < size<2,0,3>(tGatherIndexTiled); ++bbk)
            for (int bii = 0; bii < size<0,0,1>(tGatherIndexTiled); ++bii)
            for (int bjj = 0; bjj < size<0,0,2>(tGatherIndexTiled); ++bjj)
            for (int bkk = 0; bkk < size<0,0,3>(tGatherIndexTiled); ++bkk)
                for (int iii = 0; iii < size<0,1>(tGatherIndexTiled); ++iii)
                for (int jjj = 0; jjj < size<0,2>(tGatherIndexTiled); ++jjj)
                for (int kkk = 0; kkk < size<0,3>(tGatherIndexTiled); ++kkk)
                    for (int t = 0; t < size<3,1>(tGatherIndexTiled); ++t)
                    for (int r = 0; r < size<3,2>(tGatherIndexTiled); ++r)
                    for (int s = 0; s < size<3,3>(tGatherIndexTiled); ++s)
                    {
                        //                     (((0,bii,bjj,bkk),iii,jjj,kkk),(cc,0,0,0),(( l,bbi,bbj,bbk),0,0,0),(bc,t,r,s))
                        auto coord = 
                            make_tuple         (
                                make_tuple      (
                                    make_tuple   (0,bii,bjj,bkk),iii,jjj,kkk),
                                make_tuple                                    ( 0,0,0,0),
                                make_tuple                                               (
                                    make_tuple                                            ( l,bbi,bbj,bbk),0,0,0),
                                make_tuple                                                                        ( 0,t,r,s));

                        for (int bc = 0; bc < size<3,0>(tGatherIndexTiled); ++bc)
                        for (int cc = 0; cc < size<1,0>(tGatherIndexTiled); ++cc)
                        {
                            //                     (((0,bii,bjj,bkk),iii,jjj,kkk),(cc,0,0,0),(( l,bbi,bbj,bbk),0,0,0),(bc,t,r,s))
                            auto component_coord = 
                                make_tuple         (
                                    make_tuple      (
                                        make_tuple   (0,bii,bjj,bkk),iii,jjj,kkk),
                                    make_tuple                                    (cc,0,0,0),
                                    make_tuple                                               (
                                        make_tuple                                            ( l,bbi,bbj,bbk),0,0,0),
                                    make_tuple                                                                        (bc,t,r,s));
                            int c = bc * size<1,0>(tGatherIndexTiled) + cc;
                            if(&tXformedActGatherTiled(component_coord) !=
                                inputData.data().get() + tGatherIndexTiled(coord) * IGEMM_Geometry::C + c)
                                throw std::runtime_error("Inconsistency detected between input activations and gather indices");
                        }
                }
#endif

#if 0
    for (int n = 0; n < blockCount; ++n)
        for (int z = 0; z < IGEMM_Geometry::Z; ++z)
        for (int p = 0; p < IGEMM_Geometry::P; ++p)
        for (int q = 0; q < IGEMM_Geometry::Q; ++q) 
            for (int t = 0; t < IGEMM_Geometry::T; ++t)
            for (int r = 0; r < IGEMM_Geometry::R; ++r)
            for (int s = 0; s < IGEMM_Geometry::S; ++s)
                for (int c = 0; c < IGEMM_Geometry::C; ++c) {
                    if (&tXformedActGatherLegacy(make_tuple(n,z,p,q),make_tuple(c,t,r,s)) !=
                        inputData.data().get()+IGEMM_Geometry::C*tGatherIndexLegacy(make_tuple(n,z+t,p+r,q+s),make_tuple(c,0,0,0))+c)
                    {
                        std::cout << "tXformedActGatherLegacy("
                                  << "(" << std::setw(6) << n << "," << z << "," << p << "," << q << "),"
                                  << "(" << std::setw(2) << c << "," << t << "," << r << "," << s << ")) = "
                                  << tXformedActGatherLegacy(make_tuple(n,z,p,q),make_tuple(c,t,r,s))
                                  << ", tGatherIndexLegacy(" 
                                  << "(" << std::setw(6) << n << "," << z+t << "," << p+r << "," << q+s << "),"
                                  << "(" << std::setw(2) << c << ",0,0,0)) = "
                                  << tGatherIndexLegacy(make_tuple(n,z+t,p+r,q+s),make_tuple(c,0,0,0))
                                  << std::endl;
                    }
                    if (tGatherIndexLegacy(make_tuple(n,z+t,p+r,q+s),make_tuple(c,0,0,0)) == 0)
                        if (tXformedActGatherLegacy(make_tuple(n,z,p,q),make_tuple(c,t,r,s)) != 0.f)
                        {
                            std::cout << "tXformedActGatherLegacy("
                                      << "(" << std::setw(6) << n << "," << z << "," << p << "," << q << "),"
                                      << "(" << std::setw(2) << c << "," << t << "," << r << "," << s << ")) = "
                                      << tXformedActGatherLegacy(make_tuple(n,z,p,q),make_tuple(c,t,r,s))
                                      << ", tGatherIndexLegacy(" 
                                      << "(" << std::setw(6) << n << "," << z+t << "," << p+r << "," << q+s << "),"
                                      << "(" << std::setw(2) << c << ",0,0,0)) = "
                                      << tGatherIndexLegacy(make_tuple(n,z+t,p+r,q+s),make_tuple(c,0,0,0))
                                      << std::endl;
                        }   
                }
#endif

    Tensor tFilter = make_tensor(
        make_gmem_ptr(filterData.data().get()),
        layouts.filterLayout()
    );

    Tensor tXformedOutScatterLegacy = make_tensor(
        make_gmem_ptr(outputData.data().get()),
        layouts.xformedOutputComposedLayoutLegacy(blockCount, scatterIndexDataLegacy.data().get())
    );

    Tensor tXformedOutScatter = make_tensor(
        make_gmem_ptr(outputData.data().get()),
        layouts.xformedOutputComposedLayout(outputLeafCount, scatterIndexData.data().get())
    );

    Tensor tScatterIndexLegacy = make_tensor(
        make_gmem_ptr(scatterIndexDataLegacy.data().get()),
        layouts.scatterIndexLayoutLegacy(blockCount)
    );
 
    Tensor tScatterIndex = make_tensor(
        make_gmem_ptr(scatterIndexData.data().get()),
        layouts.scatterIndexLayout(outputLeafCount)
    );
    
#if 0
    auto tXformedOutScatterTiled = local_tile(tXformedOutScatter, ConvOp::TilerOut{}, make_coord(_,_));
    auto tScatterIndexTiled = local_tile(tScatterIndex, ConvOp::TilerOut{}, make_coord(_,_));
    // print("tXformedOutScatter.layout() = ");print(tXformedOutScatter.layout());print("\n");
    // print("tXformedOutScatterTiled.layout() = ");print(tXformedOutScatterTiled.layout());print("\n");
    // print("tScatterIndex.layout() = ");print(tScatterIndex.layout());print("\n");
    // print("tScatterIndexTiled.layout() = ");print(tScatterIndexTiled.layout());print("\n");

    for (int l = 0; l < outputLeafCount; ++l)
        for (int bbi = 0; bbi < size<3,0,1>(tScatterIndexTiled); ++bbi)
        for (int bbj = 0; bbj < size<3,0,2>(tScatterIndexTiled); ++bbj)
        for (int bbk = 0; bbk < size<3,0,3>(tScatterIndexTiled); ++bbk)
            for (int bii = 0; bii < size<1,0,1>(tScatterIndexTiled); ++bii)
            for (int bjj = 0; bjj < size<1,0,2>(tScatterIndexTiled); ++bjj)
            for (int bkk = 0; bkk < size<1,0,3>(tScatterIndexTiled); ++bkk)
                for (int iii = 0; iii < size<1,1>(tScatterIndexTiled); ++iii)
                for (int jjj = 0; jjj < size<1,2>(tScatterIndexTiled); ++jjj)
                for (int kkk = 0; kkk < size<1,3>(tScatterIndexTiled); ++kkk)
                {
                    auto coord =
                        make_tuple
                                           (0,
                            make_tuple
                                              (
                                make_tuple
                                               (0,bii,bjj,bkk),iii,jjj,kkk),0,
                            make_tuple
                                                                              (
                                make_tuple
                                                                               ( l,bbi,bbj,bbk),0,0,0));

                    for (int bk = 0; bk < size<2>(tScatterIndexTiled); ++bk)
                    for (int kk = 0; kk < size<0>(tScatterIndexTiled); ++kk)
                    {
                        int k = bk * size<0>(tScatterIndexTiled) + kk;
                        auto component_coord = 
                            make_tuple
                                                 (kk,
                                make_tuple
                                                     (
                                    make_tuple
                                                      (0,bii,bjj,bkk),iii,jjj,kkk),bk,
                                make_tuple
                                                                                     (
                                    make_tuple
                                                                                      ( l,bbi,bbj,bbk),0,0,0));
                        if(&tXformedOutScatterTiled(component_coord) !=
                            outputData.data().get() + tScatterIndexTiled(coord) * IGEMM_Geometry::K + k)
                            throw std::runtime_error("Inconsistent addresses");
                    }
                }
#endif

    // ((BLK_M, BLK_N), (m', n'))
    Tensor gOutput_mn = zipped_divide(tXformedOutScatter, typename AmperePredicatedFprop<IGEMM_Geometry>::TilerOut{});
    print("\n");print("shape(gOutput_mn)=");print(shape(gOutput_mn));print("\n");
    dim3 launch_grid {static_cast<uint32_t>(size<1,1>(gOutput_mn)), static_cast<uint32_t>(size<1,0>(gOutput_mn)), 1};
    constexpr size_t smem_size = sizeof(typename AmperePredicatedFprop<IGEMM_Geometry>::SharedStorage);
    std::cout << "smem_size = " << smem_size << std::endl;

    cudaCheck(
        cudaFuncSetAttribute(
            kernel_entrypoint_custom<
                AmperePredicatedFprop<IGEMM_Geometry>, decltype(tFilter), decltype(tXformedActGather),
                decltype(tGatherIndex), decltype(tXformedOutScatter), decltype(tScatterIndex)>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size
        ));

    int num_iterations = 10;
    for (int i = 0; i < num_iterations; ++i) {
        gpuTimer.start("Scatter-Gather Cutlass IGEMM (GPU) execution");
        kernel_entrypoint_custom<
            AmperePredicatedFprop<IGEMM_Geometry>, decltype(tFilter), decltype(tXformedActGather),
            decltype(tGatherIndex), decltype(tXformedOutScatter), decltype(tScatterIndex)>
            <<<launch_grid, AmperePredicatedFprop<IGEMM_Geometry>::MaxThreadsPerBlock, smem_size>>>(
                tFilter,
                tXformedActGather,
                tGatherIndex,
                tXformedOutScatter,
                tScatterIndex,
                inputData.data().get(),
                outputData.data().get()
            );
        gpuTimer.stop();
    }

    ResultCompare<Do>(
        outputValueCount,
        outputArray,
        outputReferenceArray
    );

}
