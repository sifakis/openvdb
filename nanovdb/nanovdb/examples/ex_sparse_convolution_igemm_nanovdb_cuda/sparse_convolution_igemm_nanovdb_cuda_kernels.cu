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

    static constexpr int Bx = 8/Z;  // Block count along X-dimension of leaf node
    static constexpr int By = 8/P;  // Block count along Y-dimension of leaf node
    static constexpr int Bz = 8/Q;  // Block count along Z-dimension of leaf node

    //
    // Filter offset (coordinate offset in the input domain that the [0,0,0] filter spoke corresponds to)
    //

    static constexpr int Dx = -1;  // Filter centered at (0,0,0), thus (0,0,0) spoke corresponds
    static constexpr int Dy = -1;  // to a (-1,-1,-1) grid offset
    static constexpr int Dz = -1;

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

    static auto xformedActivationComposedLayout(const int N, const uint64_t* gather_idx_buf)
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

    static auto gatherIndexLayout(const int N)
    {
        // Input gather index layout
        // gather_layout_index(make_coord((ndhw), c)) => buffer_idx
        return make_layout(
            make_shape (make_shape (    N,   D, H,  W  ), make_shape ( C  , _1{}, _1{}, _1{})),
            make_stride(make_stride(D*H*W, H*W, W, _1{}), make_stride(_0{}, _0{}, _0{}, _0{})));
    }

    static auto filterLayout()
    {
        return make_ordered_layout(
            make_shape(K, make_shape(C, T, R, S)),
            tuple<_1, tuple<_0,_4,_3,_2>>{}
        );
    }

    static auto xformedOutputComposedLayout(const int N, const uint64_t* scatter_idx_buf)
    {
        // TODO: Simplify these, no need to create the dense strides just to replace them

        // Tensor Output
        auto output_layout = make_ordered_layout(
            make_shape( K,   make_shape( N,   Z,   P,   Q)),
            make_tuple(_0{}, make_tuple(_4{},_3{},_2{},_1{})));

        // Output scatter layout
        auto ES = E<0>{};  // Scatter basis    (1,0) (idx_buffer_idx) 
        auto EC = E<1>{};  // Contiguous basis (0,1) (dense_offset)    
        auto out_basis_stride = make_stride( EC, make_stride(Z*P*Q*ES, P*Q*ES, Q*ES, _1{}*ES)); // -> (crd0, crd1)
        auto out_basis_layout = make_layout(shape(output_layout), out_basis_stride);
        auto out_scatter_layout = make_layout(
            make_shape(_1{},_1{}),
            make_stride(example::CustomStride{example::IndexedGather{scatter_idx_buf}, K}, _1{}));
        return composition(
            out_scatter_layout,
            make_arithmetic_tuple(_0{},_0{}),
            out_basis_layout);
    }

    static auto scatterIndexLayout(const int N)
    {
        // Output scatter index layout
        // scatter_layout_index(k, make_coord((nzpq))) => buffer_idx
        return make_layout(
            make_shape (   K, make_shape (    N,   Z, P,  Q  )),
            make_stride(_0{}, make_stride(Z*P*Q, P*Q, Q, _1{})));
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

template<class Operator, class FilterTensor, class ActivationTensor, class ActivationTensorIndex, class OutputTensor>
__global__
__launch_bounds__(Operator::MaxThreadsPerBlock, Operator::MinBlocksPerMultiprocessor)
  void kernel_entrypoint_custom(FilterTensor mFlt, ActivationTensor mAct, ActivationTensorIndex mActI, OutputTensor mOut) {
  extern __shared__ char smem_buf[];
  Operator op;
  op(mFlt, mAct, mActI, mOut, smem_buf);
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
    std::mt19937 generator(rd());
    std::uniform_int_distribution<int> distribution(-256, 256);

    gpuTimer.start("Initializing input (activation) data");
    auto inputValueCount = nanovdb::util::cuda::DeviceGridTraits<BuildT>::getValueCount(inputGrid);
    auto inputVoxelCount = nanovdb::util::cuda::DeviceGridTraits<BuildT>::getActiveVoxelCount(inputGrid);
    auto inputData = thrust::universal_vector<float>(inputValueCount*Di);
    auto inputArray = reinterpret_cast<inputArrayT>(*inputData.data().get());
    for (int i = 0; i < Di; i++) inputArray[0][i] = 0.f;
#pragma omp parallel for
    for (int v = 1; v <= inputVoxelCount; v++)
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
    for (int v = 0; v <= inputValueCount; v++)
        for (int i = 0; i < Di; i++)
            outputArray[v][i] = outputReferenceArray[v][i] = 0.f;
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

    auto outputVoxelsPerBlock = IGEMM_Geometry::Z * IGEMM_Geometry::P * IGEMM_Geometry::Q;
    using ScatterIndexT = uint64_t [IGEMM_Geometry::Z][IGEMM_Geometry::P][IGEMM_Geometry::Q];
    using ScatterIndexArrayT = ScatterIndexT [IGEMM_Geometry::Bx][IGEMM_Geometry::By][IGEMM_Geometry::Bz];
    auto scatterIndexData = thrust::universal_vector<uint64_t>(blockCount*outputVoxelsPerBlock);
    auto scatterIndexArray = reinterpret_cast<ScatterIndexArrayT*>(scatterIndexData.data().get());
    
#pragma omp parallel for
    for (int l = 0; l < outputLeafCount; l++) {
        auto &leaf = outputGrid->tree().getFirstLeaf()[l];
        for (int bi = 0; bi < IGEMM_Geometry::Bx; bi++)
        for (int bj = 0; bj < IGEMM_Geometry::By; bj++)
        for (int bk = 0; bk < IGEMM_Geometry::Bz; bk++) {
            nanovdb::Coord blockOffset(bi*IGEMM_Geometry::Z, bj*IGEMM_Geometry::P, bk*IGEMM_Geometry::Q);
            for (int i = 0; i < IGEMM_Geometry::Z; i++)
            for (int j = 0; j < IGEMM_Geometry::P; j++)
            for (int k = 0; k < IGEMM_Geometry::Q; k++) {
                auto localCoord = blockOffset.offsetBy(i,j,k);
                scatterIndexArray[l][bi][bj][bk][i][j][k] = leaf.getValue(localCoord);
            }
        }
    }
    gpuTimer.stop();

    gpuTimer.start("Initializing gather indices");
    auto inputVoxelsPerBlock = IGEMM_Geometry::D * IGEMM_Geometry::H * IGEMM_Geometry::W;
    using GatherIndexT = uint64_t [IGEMM_Geometry::D][IGEMM_Geometry::H][IGEMM_Geometry::W];
    using GatherIndexArrayT = GatherIndexT [IGEMM_Geometry::Bx][IGEMM_Geometry::By][IGEMM_Geometry::Bz];
    auto gatherIndexData = thrust::universal_vector<uint64_t>(blockCount*inputVoxelsPerBlock);
    auto gatherIndexArray = reinterpret_cast<GatherIndexArrayT*>(gatherIndexData.data().get());
#pragma omp parallel for
    for (int l = 0; l < outputLeafCount; l++) {
        auto &outputLeaf = outputGrid->tree().getFirstLeaf()[l];
        const auto origin = outputLeaf.origin();
        for (int bi = 0; bi < IGEMM_Geometry::Bx; bi++)
        for (int bj = 0; bj < IGEMM_Geometry::By; bj++)
        for (int bk = 0; bk < IGEMM_Geometry::Bz; bk++) {
            nanovdb::Coord blockOffset(bi*IGEMM_Geometry::Z, bj*IGEMM_Geometry::P, bk*IGEMM_Geometry::Q);
            for (int i = 0; i < IGEMM_Geometry::D; i++)
            for (int j = 0; j < IGEMM_Geometry::H; j++)
            for (int k = 0; k < IGEMM_Geometry::W; k++) {
                auto localCoord = blockOffset.offsetBy(i+IGEMM_Geometry::Dx,j+IGEMM_Geometry::Dy,k+IGEMM_Geometry::Dz);
                auto globalCoord = origin+localCoord;
                gatherIndexArray[l][bi][bj][bk][i][j][k] = inputGrid->tree().getValue(globalCoord);
            }
        }
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
        reinterpret_cast<GatherIndexT*>(gatherIndexArray),
        reinterpret_cast<ScatterIndexT*>(scatterIndexArray),
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

    Tensor tXformedActGather = make_tensor(
        make_gmem_ptr(inputData.data().get()),
        layouts.xformedActivationComposedLayout(blockCount, gatherIndexData.data().get())
    );

    Tensor tGatherIndex = make_tensor(
        make_gmem_ptr(gatherIndexData.data().get()),
        layouts.gatherIndexLayout(blockCount)
    );

#if 0
    for (int n = 0; n < blockCount; ++n)
        for (int z = 0; z < IGEMM_Geometry::Z; ++z)
        for (int p = 0; p < IGEMM_Geometry::P; ++p)
        for (int q = 0; q < IGEMM_Geometry::Q; ++q) 
            for (int t = 0; t < IGEMM_Geometry::T; ++t)
            for (int r = 0; r < IGEMM_Geometry::R; ++r)
            for (int s = 0; s < IGEMM_Geometry::S; ++s)
                for (int c = 0; c < IGEMM_Geometry::C; ++c) {
                    if (&tXformedActGather(make_tuple(n,z,p,q),make_tuple(c,t,r,s)) !=
                        inputData.data().get()+IGEMM_Geometry::C*tGatherIndex(make_tuple(n,z+t,p+r,q+s),make_tuple(c,0,0,0))+c)
                    {
                        std::cout << "tXformedActGather("
                                  << "(" << std::setw(6) << n << "," << z << "," << p << "," << q << "),"
                                  << "(" << std::setw(2) << c << "," << t << "," << r << "," << s << ")) = "
                                  << tXformedActGather(make_tuple(n,z,p,q),make_tuple(c,t,r,s))
                                  << ", tGatherIndex(" 
                                  << "(" << std::setw(6) << n << "," << z+t << "," << p+r << "," << q+s << "),"
                                  << "(" << std::setw(2) << c << ",0,0,0)) = "
                                  << tGatherIndex(make_tuple(n,z+t,p+r,q+s),make_tuple(c,0,0,0))
                                  << std::endl;
                    }
                    if (tGatherIndex(make_tuple(n,z+t,p+r,q+s),make_tuple(c,0,0,0)) == 0)
                        if (tXformedActGather(make_tuple(n,z,p,q),make_tuple(c,t,r,s)) != 0.f)
                        {
                            std::cout << "tXformedActGather("
                                      << "(" << std::setw(6) << n << "," << z << "," << p << "," << q << "),"
                                      << "(" << std::setw(2) << c << "," << t << "," << r << "," << s << ")) = "
                                      << tXformedActGather(make_tuple(n,z,p,q),make_tuple(c,t,r,s))
                                      << ", tGatherIndex(" 
                                      << "(" << std::setw(6) << n << "," << z+t << "," << p+r << "," << q+s << "),"
                                      << "(" << std::setw(2) << c << ",0,0,0)) = "
                                      << tGatherIndex(make_tuple(n,z+t,p+r,q+s),make_tuple(c,0,0,0))
                                      << std::endl;
                        }   
                }
#endif

    Tensor tFilter = make_tensor(
        make_gmem_ptr(filterData.data().get()),
        layouts.filterLayout()
    );

    Tensor tXformedOutScatter = make_tensor(
        make_gmem_ptr(outputData.data().get()),
        layouts.xformedOutputComposedLayout(blockCount, scatterIndexData.data().get())
    );

    Tensor tScatterrIndex = make_tensor(
        make_gmem_ptr(scatterIndexData.data().get()),
        layouts.scatterIndexLayout(blockCount)
    );

    // ((BLK_M, BLK_N), (m', n'))
    Tensor gOutput_mn = zipped_divide(tXformedOutScatter, typename AmperePredicatedFprop<IGEMM_Geometry>::TilerOut{});
    dim3 launch_grid {static_cast<uint32_t>(size<1,1>(gOutput_mn)), static_cast<uint32_t>(size<1,0>(gOutput_mn)), 1};
    constexpr size_t smem_size = sizeof(typename AmperePredicatedFprop<IGEMM_Geometry>::SharedStorage);

    cudaCheck(cudaFuncSetAttribute(
            kernel_entrypoint_custom<AmperePredicatedFprop<IGEMM_Geometry>, decltype(tFilter), decltype(tXformedActGather), decltype(tGatherIndex), decltype(tXformedOutScatter)>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size));

    int num_iterations = 10;
    for (int i = 0; i < num_iterations; ++i) {
        gpuTimer.start("Scatter-Gather Cutlass IGEMM (GPU) execution");
        kernel_entrypoint_custom<AmperePredicatedFprop<IGEMM_Geometry>, decltype(tFilter), decltype(tXformedActGather), decltype(tGatherIndex), decltype(tXformedOutScatter)>
            <<<launch_grid, AmperePredicatedFprop<IGEMM_Geometry>::MaxThreadsPerBlock, smem_size>>>(
                tFilter, tXformedActGather, tGatherIndex, tXformedOutScatter);
        gpuTimer.stop();
    }

    ResultCompare<Do>(
        outputValueCount,
        outputArray,
        outputReferenceArray
    );

}
