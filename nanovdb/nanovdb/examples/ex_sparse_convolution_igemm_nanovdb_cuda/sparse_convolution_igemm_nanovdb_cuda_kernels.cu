// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <nanovdb/cuda/UnifiedBuffer.h>
#include <nanovdb/tools/cuda/PointsToGrid.cuh>
#include <nanovdb/tools/cuda/MergeGrids.cuh>
#include <nanovdb/util/cuda/DeviceGridTraits.cuh>

#include <thrust/universal_vector.h>
#include <random>

#include "SparseFpropSm80.h"


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

    static constexpr int STx = 2;   // Convolution stride along X
    static constexpr int STy = 2;   // Convolution stride along Y
    static constexpr int STz = 2;   // Convolution stride along Z

    static constexpr int Z = 4;     // X-dimension of output block
    static constexpr int P = 2;     // Y-dimension of output block
    static constexpr int Q = 2;     // Z-dimension of output block

    int c, k;                       // Input/output feature dimensions (runtime)
    int dx, dy, dz;                 // Filter offset (runtime)
    __hostdev__ int C()  const { return c;  }
    __hostdev__ int K()  const { return k;  }
    __hostdev__ int Dx() const { return dx; }
    __hostdev__ int Dy() const { return dy; }
    __hostdev__ int Dz() const { return dz; }

    static constexpr int TC = 32;   // Tile size along C (input feature) dimension
    static constexpr int TK = 128;  // Tile size along K (output feature) dimension
    
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

    static constexpr int CHx = (ZZ*Z-1)*STx+T; // Cluster halo count along X-dimension
    static constexpr int CHy = (PP*P-1)*STy+R; // Cluster halo count along Y-dimension
    static constexpr int CHz = (QQ*Q-1)*STz+S; // Cluster halo count along Z-dimension

    static constexpr int CVx = ZZ*Z;     // Voxel count per cluster along X-dimension
    static constexpr int CVy = PP*P;     // Voxel count per cluster along Y-dimension
    static constexpr int CVz = QQ*Q;     // Voxel count per cluster along Z-dimension

    static constexpr int VoxelsPerLeafnodeNoHalo() { return 512; }

    static constexpr int VoxelsPerClusterNoHalo() { return CVx*CVy*CVz; }
    static constexpr int VoxelsPerClusterWithHalo() { return CHx*CHy*CHz; }

};

// Derived geometry with compile-time default C/K values, used for test setup
// and legacy reference implementations.
struct Test_Geometry : IGEMM_Geometry {
    static constexpr int C_ = 64;
    static constexpr int K_ = 128;
    static constexpr int Di = C_;
    static constexpr int Do = K_;
    static constexpr int Hx = (Bx*Z-1)*STx+T; // X-dimension of leaf node domain, enlarged by the necessary halo for convolution
    static constexpr int Hy = (By*P-1)*STy+R; // Y-dimension of leaf node domain, enlarged by the necessary halo for convolution
    static constexpr int Hz = (Bz*Q-1)*STz+S; // Z-dimension of leaf node domain, enlarged by the necessary halo for convolution
    static constexpr int VoxelsPerLeafnodeWithHalo() { return Hx*Hy*Hz; }
    Test_Geometry() : IGEMM_Geometry{C_, K_, -1, -1, -1} {}
};


template<class GeometryT, int Di, int Do, class ValueType>
void SparseConvolveCPUReference(
    nanovdb::NanoGrid<nanovdb::ValueOnIndex> *srcGrid,
    nanovdb::NanoGrid<nanovdb::ValueOnIndex> *dstGrid,
    const ValueType (*filter)[GeometryT::R][GeometryT::S][Do][Di],
    const ValueType (*inputArray)[Di],
    ValueType (*outputArray)[Do],
    GeometryT geometry = {})
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
                const auto srcCoord = nanovdb::Coord(
                    dstCoord[0]*GeometryT::STx + di + geometry.Dx(),
                    dstCoord[1]*GeometryT::STy + dj + geometry.Dy(),
                    dstCoord[2]*GeometryT::STz + dk + geometry.Dz());
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
    ValueType (*outputArray)[Do],
    GeometryT geometry = {})
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
                const auto srcCoord = nanovdb::Coord(
                    dstCoord[0]*GeometryT::STx + di + geometry.Dx(),
                    dstCoord[1]*GeometryT::STy + dj + geometry.Dy(),
                    dstCoord[2]*GeometryT::STz + dk + geometry.Dz());
                const auto srcIndex = srcGrid->tree().getValue(srcCoord);
                if (srcIndex)
                    for ( int in = 0; in < Di; ++in )
                        outputArray[dstIndex][out] += filter[di][dj][dk][out][in] * inputArray[srcIndex][in];
            }
        }
    };

    lambda_kernel_wrapper<<<dstLeafCount,Do>>>(convolver);
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
      float *outputData,
      Operator op
  ) {
  extern __shared__ char smem_buf[];
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

void validateClusterActivationLayout()
{
    using G = IGEMM_Geometry;
    Test_Geometry geometry;
    IGEMM_Layouts<G> layouts(geometry);
    auto indexLayout = layouts.clusterActivationIndexLayout();

    int errors = 0, checks = 0;
    for (int zz = 0; zz < G::ZZ; zz++)
    for (int pp = 0; pp < G::PP; pp++)
    for (int qq = 0; qq < G::QQ; qq++)
    for (int z  = 0; z  < G::Z;  z++ )
    for (int p  = 0; p  < G::P;  p++ )
    for (int q  = 0; q  < G::Q;  q++ )
    for (int t  = 0; t  < G::T;  t++ )
    for (int r  = 0; r  < G::R;  r++ )
    for (int s  = 0; s  < G::S;  s++ )
    {
        auto coord = make_coord(
            make_coord(make_coord(zz, pp, qq), z, p, q),
            make_coord(0, t, r, s));

        int got      = indexLayout(coord);
        int expected = ((zz*G::Z+z)*G::STx + t) * G::CHy * G::CHz
                     + ((pp*G::P+p)*G::STy + r) * G::CHz
                     + ((qq*G::Q+q)*G::STz + s);
        if (got != expected) {
            std::cout << "MISMATCH at (zz,pp,qq,z,p,q,t,r,s)=("
                      << zz<<","<<pp<<","<<qq<<","<<z<<","<<p<<","<<q<<","
                      << t<<","<<r<<","<<s<<"): got "<<got<<" expected "<<expected<<"\n";
            ++errors;
        }
        ++checks;
    }
    std::cout << "clusterActivationIndexLayout: "
              << checks << " coords checked, " << errors << " errors\n";
}

void mainSparseConvolutionIGEMM(
    const std::vector<nanovdb::Coord>& inputPoints,
    const std::vector<nanovdb::Coord>& outputPoints,
    uint32_t benchmark_iters)
{
    validateClusterActivationLayout();

    Test_Geometry geometry;

    using BuildT = nanovdb::ValueOnIndex;
    using BufferT = nanovdb::cuda::UnifiedBuffer;
    static constexpr int Di = Test_Geometry::Di;
    static constexpr int Do = Test_Geometry::Do;
    using inputArrayT = float (&) [][Di];
    using outputArrayT = float (&) [][Do];
    using filterT = float (&) [IGEMM_Geometry::T][IGEMM_Geometry::R][IGEMM_Geometry::S][Do][Di];
    
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

    auto outputLeafCount = outputGrid->tree().nodeCount(0);
    
    gpuTimer.start("Reference (GPU) execution");
    SparseConvolveCudaReference<IGEMM_Geometry, Di, Do>(
        inputGrid,
        outputGrid,
        filter,
        inputArray,
        outputReferenceArray,
        geometry
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

    IGEMM_Layouts<IGEMM_Geometry> layouts(geometry);

    Tensor tFilter = make_tensor(
        make_gmem_ptr(filterData.data().get()),
        layouts.filterLayout()
    );

    SparseFpropSm80Strided<IGEMM_Geometry> op(geometry);

    constexpr size_t smem_size = sizeof(typename SparseFpropSm80Strided<IGEMM_Geometry>::SharedStorage);
    std::cout << "smem_size = " << smem_size << std::endl;

    cudaCheck(
        cudaFuncSetAttribute(
            kernel_entrypoint_custom<SparseFpropSm80Strided<IGEMM_Geometry>, BuildT, decltype(tFilter)>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size
        ));

    int num_iterations = 10;
    for (int i = 0; i < num_iterations; ++i) {
        gpuTimer.start("Scatter-Gather Cutlass IGEMM (GPU) execution");
        kernel_entrypoint_custom<SparseFpropSm80Strided<IGEMM_Geometry>, BuildT, decltype(tFilter)>
            <<<outputLeafCount, SparseFpropSm80Strided<IGEMM_Geometry>::MaxThreadsPerBlock, smem_size>>>(
                tFilter,
                inputGrid,
                outputGrid,
                inputData.data().get(),
                outputData.data().get(),
                op
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
