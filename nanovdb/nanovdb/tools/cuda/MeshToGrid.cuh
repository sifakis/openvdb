// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/tools/cuda/MeshToGrid.cuh

    \authors Efty Sifakis

    \brief Rasterization of triangle mesh into a sparse NanoVDB indexGrid on the device

    \warning The header file contains cuda device code so be sure
             to only include it in .cu files (or other .cuh files)
*/

#ifndef NVIDIA_TOOLS_CUDA_MESHTOGRID_CUH_HAS_BEEN_INCLUDED
#define NVIDIA_TOOLS_CUDA_MESHTOGRID_CUH_HAS_BEEN_INCLUDED

#include <cub/cub.cuh>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/GridHandle.h>
#include <nanovdb/cuda/TempPool.h>
#include <nanovdb/util/cuda/Timer.h>

namespace nanovdb {

namespace tools::cuda {

template <typename BuildT>
class MeshToGrid
{
    using PointT = nanovdb::Vec3f;
    using TriangleIndexT = nanovdb::Vec3i;
    using TriangleT = std::array<PointT,3>;

public:
    struct alignas(16) BoxTrianglePair { // sizeof(BoxTrianglePair) = 16B
        nanovdb::Coord origin; // 12B
        uint32_t triangleID;   // 4B
    };

    /// @brief Constructor
    /// @param devicePoints Vertex list for input triangle surface
    /// @param pointCount Vertex count for input triangle surface
    /// @param deviceTriangles Triangle index list
    /// @param triangleCount Triangle count
    /// @param map Affine map to be used in the conversion
    MeshToGrid(
        const nanovdb::Vec3f *devicePoints,
        const uint32_t pointCount,
        const nanovdb::Vec3i *deviceTriangles,
        const uint32_t triangleCount,
        const nanovdb::Map map = nanovdb::Map(),
        cudaStream_t stream = 0
    )
        : mStream(stream), mTimer(stream), mDevicePoints(devicePoints), mPointCount(pointCount),
         mDeviceTriangles(deviceTriangles), mTriangleCount(triangleCount), mMap(map)
    {}

    /// @brief Toggle on and off verbose mode
    /// @param level Verbose level: 0=quiet, 1=timing, 2=benchmarking
    void setVerbose(int level = 1) { mVerbose = level; }

    /// @brief Set desired width of narrow band
    /// @param bandWidth Narrow band width in cell units
    void setNarrowBandWidth(float bandWidth = 3.f) { mBandWidth = bandWidth; }

    /// @brief Set the mode for checksum computation, which is disabled by default
    /// @param mode Mode of checksum computation
    // void setChecksum(CheckMode mode = CheckMode::Disable){mBuilder.mChecksum = mode;}

    /// @brief Creates a handle to the output grid
    /// @tparam BufferT Buffer type used for allocation of the grid handle
    /// @param buffer optional buffer (currently ignored)
    /// @return returns a handle with a grid of type NanoGrid<BuildT>
    template<typename BufferT = nanovdb::cuda::DeviceBuffer>
    // GridHandle<BufferT>
    void
    getHandle(const BufferT &buffer = BufferT());

private:
    void transformTriangles();

    void processRootTrianglePairs();

    void processLeafTrianglePairs();


    static constexpr unsigned int mNumThreads = 128;// for kernels spawned via lambdaKernel (others may specialize)
    static unsigned int numBlocks(unsigned int n) {return (n + mNumThreads - 1) / mNumThreads;}

    // TopologyBuilder<BuildT>      mBuilder;
    cudaStream_t                 mStream{0};
    util::cuda::Timer            mTimer;
    int                          mVerbose{0};
    float                        mBandWidth{3.f};
    const nanovdb::Vec3f         *mDevicePoints;
    const uint32_t               mPointCount;
    const nanovdb::Vec3i         *mDeviceTriangles;
    const uint32_t               mTriangleCount;
    const nanovdb::Map           mMap;

    nanovdb::cuda::DeviceBuffer  mXformedTriangles;
    nanovdb::cuda::DeviceBuffer  mBoxTrianglePairsBuffer;
    uint64_t                     mBoxTrianglePairCount{0};

    auto deviceXformedTriangles() { return static_cast<TriangleT*>(mXformedTriangles.deviceData()); }
    // auto hostXformedTriangles() { return static_cast<TriangleT*>(mXformedTriangles.data()); }
    auto deviceBoxTrianglePairs() { return static_cast<BoxTrianglePair*>(mBoxTrianglePairsBuffer.deviceData()); }

    nanovdb::cuda::TempDevicePool mTempDevicePool;

    // For diagnostic purposes
public:
    uint64_t getPairCount() const { return mBoxTrianglePairCount; }
    const BoxTrianglePair* getDevicePairs() const { return static_cast<const BoxTrianglePair*>(mBoxTrianglePairsBuffer.deviceData()); }
}; // tools::cuda::MeshToGrid<BuildT>

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// Define utility macro used to call cub functions that use dynamic temporary storage
#ifndef CALL_CUBS
#ifdef _WIN32
#define CALL_CUBS(func, ...) \
    cudaCheck(cub::func(nullptr, mTempDevicePool.requestedSize(), __VA_ARGS__, mStream)); \
    mTempDevicePool.reallocate(mStream); \
    cudaCheck(cub::func(mTempDevicePool.data(), mTempDevicePool.size(), __VA_ARGS__, mStream));
#else// ndef _WIN32
#define CALL_CUBS(func, args...) \
    cudaCheck(cub::func(nullptr, mTempDevicePool.requestedSize(), args, mStream)); \
    mTempDevicePool.reallocate(mStream); \
    cudaCheck(cub::func(mTempDevicePool.data(), mTempDevicePool.size(), args, mStream));
#endif// ifdef _WIN32
#endif// ifndef CALL_CUBS

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT>
template<typename BufferT>
// GridHandle<BufferT>
void
MeshToGrid<BuildT>::getHandle(const BufferT &pool)
{
    cudaStreamSynchronize(mStream);
    
    // Transform triangle data to (floating-point) index space
    if (mVerbose==1) mTimer.start("Transforming triangles to grid index space");
    transformTriangles();
    if (mVerbose==1) mTimer.stop();

    // Process Root-Triangle pairs
    if (mVerbose==1) mTimer.start("Computing candidate RootTile-Triangle intersection pairs");
    processRootTrianglePairs();
    if (mVerbose==1) mTimer.stop();

    // Process Leaf-Triangle pairs
    if (mVerbose==1) mTimer.start("Computing candidate LeafNode-Triangle intersection pairs");
    processLeafTrianglePairs();
    if (mVerbose==1) mTimer.stop();

    // int device = 0;
    // cudaGetDevice(&device);
    // mXformedTriangles.deviceDownload(device, mStream, false);
    // while(1) {
    //     int tri;
    //     std::cin >> tri;
    //     for (int v = 0; v < 3; v++) {
    //         for (int w = 0; w < 3; w++)
    //             std::cout << hostXformedTriangles()[tri][v][w] << " ";
    //         std::cout << std::endl;
    //     }
    // }
    
#if 0
    // Copy TreeData from GPU -> CPU
    mSrcTreeData = util::cuda::DeviceGridTraits<BuildT>::getTreeData(mDeviceSrcGrid);

    // Ensure that the input grid contains no tile values
    if (mSrcTreeData.mTileCount[2] || mSrcTreeData.mTileCount[1] || mSrcTreeData.mTileCount[0])
        throw std::runtime_error("Topological operations not supported on grids with value tiles");

    // Speculatively dilate root node
    if (mVerbose==1) mTimer.start("\nDilating root node");
    dilateRoot();

    // Allocate memory for dilated upper/lower masks
    if (mVerbose==1) mTimer.restart("Allocating internal node mask buffers");
    mBuilder.allocateInternalMaskBuffers(mStream);

    // Dilate masks of upper/lower nodes
    if (mVerbose==1) mTimer.restart("Dilate internal nodes");
    dilateInternalNodes();

    // Enumerate tree nodes
    if (mVerbose==1) mTimer.restart("Count dilated tree nodes");
    mBuilder.countNodes(mStream);

    cudaStreamSynchronize(mStream);

    // Allocate new device grid buffer for dilated result
    if (mVerbose==1) mTimer.restart("Allocating dilated grid buffer");
    auto buffer = mBuilder.getBuffer(pool, mStream);

    // Process GridData/TreeData/RootData of dilated result
    if (mVerbose==1) mTimer.restart("Processing grid/tree/root");
    processGridTreeRoot();

    // Process upper nodes of dilated result
    if (mVerbose==1) mTimer.restart("Processing upper nodes");
    mBuilder.processUpperNodes(mStream);

    // Process lower nodes of dilated result
    if (mVerbose==1) mTimer.restart("Processing lower nodes");
    mBuilder.processLowerNodes(mStream);

    // Dilate leaf node active masks into new topology
    if (mVerbose==1) mTimer.restart("Dilating leaf nodes");
    dilateLeafNodes();

    // Process bounding boxes
    if (mVerbose==1) mTimer.restart("Processing bounding boxes");
    mBuilder.processBBox(mStream);

    // Post-process Grid/Tree data
    if (mVerbose==1) mTimer.restart("Post-processing grid/tree data");
    mBuilder.postProcessGridTree(mStream);
    if (mVerbose==1) mTimer.stop();

    cudaStreamSynchronize(mStream);

    return GridHandle<BufferT>(std::move(buffer));
#endif
} // MeshToGrid<BuildT>::getHandle

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace topology::detail {

template <typename BuildT>
struct TransformTrianglesFunctor
{
    const nanovdb::Vec3f* dPoints;
    const nanovdb::Vec3i* dTriangleIndices;
    std::array<nanovdb::Vec3f, 3>* dXformedTriangles;
    nanovdb::Map map; 

    __device__
    void operator()(size_t triangleID) const
    {
        for (int v = 0; v < 3; ++v) {
            dXformedTriangles[triangleID][v] = map.applyInverseMap(dPoints[dTriangleIndices[triangleID][v]]);
        }
    }
};

} // namespace topology::detail

template<typename BuildT>
void MeshToGrid<BuildT>::transformTriangles()
{
    // TODO: Handle null input case
    if (mTriangleCount == 0)
        throw std::runtime_error("MeshToGrid currently requires mTriangleCount > 0 (Holistic zero-handling pending).");

    int device = 0;
    cudaGetDevice(&device);

    mXformedTriangles = nanovdb::cuda::DeviceBuffer::create(mTriangleCount*sizeof(TriangleT), nullptr, device, mStream);
    if (mXformedTriangles.deviceData() == nullptr) throw std::runtime_error("Failed to allocate transofmed upper mask buffer on device");

    if (mVerbose == 1) mTimer.restart("[In MeshToGrid::transformTriangles()] Launching TransformTrianglesFunctor");
    util::cuda::lambdaKernel<<<numBlocks(mTriangleCount), mNumThreads, 0, mStream>>>(
        mTriangleCount,
        topology::detail::TransformTrianglesFunctor<BuildT>{
            mDevicePoints, mDeviceTriangles, deviceXformedTriangles(), mMap
        }
    );

    cudaCheckError();

} // MeshToGrid<BuildT>::transformTriangles

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace topology::detail {

template <typename BuildT>
struct CountRootBoxesFunctor
{
    const std::array<nanovdb::Vec3f, 3>* dXformedTriangles;
    uint64_t* dCounts;
    float mPadding; // typically 2.5f for a 3-cell-width narrowband

    __device__
    void operator()(size_t triangleID) const
    {
        const auto& tri = dXformedTriangles[triangleID];

        // Compute the strict AABB of the triangle in (cell-centered) index space
        float min_x = fminf(tri[0][0], fminf(tri[1][0], tri[2][0]));
        float min_y = fminf(tri[0][1], fminf(tri[1][1], tri[2][1]));
        float min_z = fminf(tri[0][2], fminf(tri[1][2], tri[2][2]));

        float max_x = fmaxf(tri[0][0], fmaxf(tri[1][0], tri[2][0]));
        float max_y = fmaxf(tri[0][1], fmaxf(tri[1][1], tri[2][1]));
        float max_z = fmaxf(tri[0][2], fmaxf(tri[1][2], tri[2][2]));

        // Apply geometric expansion (mPadding) and cell-center alignment shift (+0.5f)
        float adj_min_x = min_x - mPadding + 0.5f;
        float adj_min_y = min_y - mPadding + 0.5f;
        float adj_min_z = min_z - mPadding + 0.5f;
        
        float adj_max_x = max_x + mPadding + 0.5f;
        float adj_max_y = max_y + mPadding + 0.5f;
        float adj_max_z = max_z + mPadding + 0.5f;

        // Convert padded continuous bounds to discrete Root Tile index space
        // A NanoVDB Upper node (Root tile) spans exactly 4096^3 voxels
        const float invRootDim = 1.0f / 4096.0f;

        // floorf safely maps negative/positive coordinates to the correct integer index
        int min_i = static_cast<int>(floorf(adj_min_x * invRootDim));
        int min_j = static_cast<int>(floorf(adj_min_y * invRootDim));
        int min_k = static_cast<int>(floorf(adj_min_z * invRootDim));

        int max_i = static_cast<int>(floorf(adj_max_x * invRootDim));
        int max_j = static_cast<int>(floorf(adj_max_y * invRootDim));
        int max_k = static_cast<int>(floorf(adj_max_z * invRootDim));

        // Compute the 3D grid dimensions of overlapping root boxes
        uint64_t count_x = max_i - min_i + 1;
        uint64_t count_y = max_j - min_j + 1;
        uint64_t count_z = max_k - min_k + 1;

        // Write the total count of root boxes this triangle touches
        dCounts[triangleID] = count_x * count_y * count_z;
    }
};

template <typename BuildT>
struct ScatterRootTrianglePairsFunctor
{
    using PairT = typename MeshToGrid<BuildT>::BoxTrianglePair;

    const std::array<nanovdb::Vec3f, 3>* dXformedTriangles;
    const uint64_t* dOffsets;
    PairT* dPairs;
    float mPadding;

    __device__
    void operator()(size_t triangleID) const
    {
        const auto& tri = dXformedTriangles[triangleID];

        // Recompute the strict AABB
        float min_x = fminf(tri[0][0], fminf(tri[1][0], tri[2][0]));
        float min_y = fminf(tri[0][1], fminf(tri[1][1], tri[2][1]));
        float min_z = fminf(tri[0][2], fminf(tri[1][2], tri[2][2]));

        float max_x = fmaxf(tri[0][0], fmaxf(tri[1][0], tri[2][0]));
        float max_y = fmaxf(tri[0][1], fmaxf(tri[1][1], tri[2][1]));
        float max_z = fmaxf(tri[0][2], fmaxf(tri[1][2], tri[2][2]));

        // Apply geometric expansion and cell-center alignment shift
        float adj_min_x = min_x - mPadding + 0.5f;
        float adj_min_y = min_y - mPadding + 0.5f;
        float adj_min_z = min_z - mPadding + 0.5f;
        
        float adj_max_x = max_x + mPadding + 0.5f;
        float adj_max_y = max_y + mPadding + 0.5f;
        float adj_max_z = max_z + mPadding + 0.5f;

        // Convert padded continuous bounds to discrete root index space
        const float invRootDim = 1.0f / 4096.0f;
        int min_i = static_cast<int>(floorf(adj_min_x * invRootDim));
        int min_j = static_cast<int>(floorf(adj_min_y * invRootDim));
        int min_k = static_cast<int>(floorf(adj_min_z * invRootDim));

        int max_i = static_cast<int>(floorf(adj_max_x * invRootDim));
        int max_j = static_cast<int>(floorf(adj_max_y * invRootDim));
        int max_k = static_cast<int>(floorf(adj_max_z * invRootDim));

        // Scatter the pairs into the global array
        uint64_t write_idx = dOffsets[triangleID];
        for (int k = min_k; k <= max_k; ++k)
        for (int j = min_j; j <= max_j; ++j)
        for (int i = min_i; i <= max_i; ++i) {
            // Multiply back by 4096 to get the true NanoVDB index-space origin
            dPairs[write_idx].origin = nanovdb::Coord(i * 4096, j * 4096, k * 4096);
            dPairs[write_idx].triangleID = static_cast<uint32_t>(triangleID);
            write_idx++;
        }
    }
};

} // namespace topology::detail

template<typename BuildT>
void MeshToGrid<BuildT>::processRootTrianglePairs()
{
    // TODO: Handle null input case
    if (mTriangleCount == 0)
        throw std::runtime_error("MeshToGrid currently requires mTriangleCount > 0 (Holistic zero-handling pending).");

    int device = 0;
    cudaGetDevice(&device);

    // Pass 1: Count intersecting root boxes per triangle
    
    if (mVerbose == 1) mTimer.restart("[In MeshToGrid::processRootTrianglePairs()] Launching processRootTrianglePairs");

    nanovdb::cuda::DeviceBuffer
        rootBoxCounts = nanovdb::cuda::DeviceBuffer::create(mTriangleCount * sizeof(uint64_t), nullptr, device, mStream);
    if (rootBoxCounts.deviceData() == nullptr) throw std::runtime_error("Failed to allocate root box counts buffer");

    util::cuda::lambdaKernel<<<numBlocks(mTriangleCount), mNumThreads, 0, mStream>>>(
        mTriangleCount, 
        topology::detail::CountRootBoxesFunctor<BuildT>{
            deviceXformedTriangles(),
            static_cast<uint64_t*>(rootBoxCounts.deviceData()),
            mBandWidth-.5f // Due to level set values being cell-centered
        }
    );
    cudaCheckError();

    // Pass 2: InclusiveSum Scan to compute offsets and total allocations
    
    if (mVerbose == 1) mTimer.restart("[In MeshToGrid::processRootTrianglePairs()] Prefix sum");
    
    nanovdb::cuda::DeviceBuffer rootBoxOffsets =
        nanovdb::cuda::DeviceBuffer::create((mTriangleCount+1)*sizeof(uint64_t), nullptr, device, mStream);
    if (rootBoxOffsets.deviceData() == nullptr) throw std::runtime_error("Failed to allocate root box offsets buffer");

    cudaCheck(cudaMemsetAsync(rootBoxOffsets.deviceData(), 0, sizeof(uint64_t), mStream));
    CALL_CUBS(DeviceScan::InclusiveSum,
        static_cast<uint64_t*>(rootBoxCounts.deviceData()),
        static_cast<uint64_t*>(rootBoxOffsets.deviceData())+1,
        mTriangleCount);
    cudaCheck(cudaMemcpyAsync(&mBoxTrianglePairCount, static_cast<uint64_t*>(rootBoxOffsets.deviceData())+mTriangleCount, sizeof(uint64_t), cudaMemcpyDeviceToHost, mStream));
    cudaStreamSynchronize(mStream);

    if (mVerbose == 1) printf("Total Root/Triangle pairs to generate: %d\n", (int)mBoxTrianglePairCount);

    // Pass 3: Re-enumerate intersections of (padded) root boxes and triangles, and scatter to allocated list

    if (mVerbose == 1) mTimer.restart("[In MeshToGrid::processRootTrianglePairs()] Scatter pairs");

    mBoxTrianglePairsBuffer = nanovdb::cuda::DeviceBuffer::create(
        mBoxTrianglePairCount * sizeof(typename MeshToGrid<BuildT>::BoxTrianglePair), nullptr, device, mStream);
    if (mBoxTrianglePairsBuffer.deviceData() == nullptr) throw std::runtime_error("Failed to allocate pairs buffer");

    util::cuda::lambdaKernel<<<numBlocks(mTriangleCount), mNumThreads, 0, mStream>>>(
        mTriangleCount, 
        topology::detail::ScatterRootTrianglePairsFunctor<BuildT>{
            deviceXformedTriangles(),
            static_cast<uint64_t*>(rootBoxOffsets.deviceData()),
            deviceBoxTrianglePairs(),
            mBandWidth - 0.5f
        }
    );

} // MeshToGrid<BuildT>::processRootTrianglePairs

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace topology::detail {

/// @brief Tests if a Triangle intersects an Axis-Aligned Bounding Box.
template <bool OnlyUseAABB>
__device__ inline bool testTriangleAABB(
    const nanovdb::Vec3f& boxCenter,
    const nanovdb::Vec3f& boxHalfExtents,
    const nanovdb::Vec3f& V0,
    const nanovdb::Vec3f& V1,
    const nanovdb::Vec3f& V2)
{
    // Translate the triangle as if the AABB is centered at the origin
    nanovdb::Vec3f v0 = V0 - boxCenter;
    nanovdb::Vec3f v1 = V1 - boxCenter;
    nanovdb::Vec3f v2 = V2 - boxCenter;

    // --- PHASE 1: AABB OVERLAP (3 Axes) ---
    float minX = fminf(v0[0], fminf(v1[0], v2[0]));
    float maxX = fmaxf(v0[0], fmaxf(v1[0], v2[0]));
    if (minX > boxHalfExtents[0] || maxX < -boxHalfExtents[0]) return false;

    float minY = fminf(v0[1], fminf(v1[1], v2[1]));
    float maxY = fmaxf(v0[1], fmaxf(v1[1], v2[1]));
    if (minY > boxHalfExtents[1] || maxY < -boxHalfExtents[1]) return false;

    float minZ = fminf(v0[2], fminf(v1[2], v2[2]));
    float maxZ = fmaxf(v0[2], fmaxf(v1[2], v2[2]));
    if (minZ > boxHalfExtents[2] || maxZ < -boxHalfExtents[2]) return false;

    if constexpr (OnlyUseAABB) return true; 

    // --- PHASE 2: SEPARATING AXIS THEOREM (SAT) (10 additional axes) ---
    nanovdb::Vec3f f0 = v1 - v0, f1 = v2 - v1, f2 = v0 - v2;
    float r, p0, p1, p2;

    // Axis 00, 01, 02 (X-axis cross products)
    p0 = v0[2]*f0[1] - v0[1]*f0[2]; p2 = v2[2]*f0[1] - v2[1]*f0[2];
    r = boxHalfExtents[1]*fabsf(f0[2]) + boxHalfExtents[2]*fabsf(f0[1]);
    if (fminf(p0, p2) > r || fmaxf(p0, p2) < -r) return false;

    p0 = v0[2]*f1[1] - v0[1]*f1[2]; p1 = v1[2]*f1[1] - v1[1]*f1[2];
    r = boxHalfExtents[1]*fabsf(f1[2]) + boxHalfExtents[2]*fabsf(f1[1]);
    if (fminf(p0, p1) > r || fmaxf(p0, p1) < -r) return false;

    p0 = v0[2]*f2[1] - v0[1]*f2[2]; p1 = v1[2]*f2[1] - v1[1]*f2[2];
    r = boxHalfExtents[1]*fabsf(f2[2]) + boxHalfExtents[2]*fabsf(f2[1]);
    if (fminf(p0, p1) > r || fmaxf(p0, p1) < -r) return false;

    // Axis 10, 11, 12 (Y-axis cross products)
    p0 = v0[0]*f0[2] - v0[2]*f0[0]; p2 = v2[0]*f0[2] - v2[2]*f0[0];
    r = boxHalfExtents[0]*fabsf(f0[2]) + boxHalfExtents[2]*fabsf(f0[0]);
    if (fminf(p0, p2) > r || fmaxf(p0, p2) < -r) return false;

    p0 = v0[0]*f1[2] - v0[2]*f1[0]; p1 = v1[0]*f1[2] - v1[2]*f1[0];
    r = boxHalfExtents[0]*fabsf(f1[2]) + boxHalfExtents[2]*fabsf(f1[0]);
    if (fminf(p0, p1) > r || fmaxf(p0, p1) < -r) return false;

    p0 = v0[0]*f2[2] - v0[2]*f2[0]; p1 = v1[0]*f2[2] - v1[2]*f2[0];
    r = boxHalfExtents[0]*fabsf(f2[2]) + boxHalfExtents[2]*fabsf(f2[0]);
    if (fminf(p0, p1) > r || fmaxf(p0, p1) < -r) return false;

    // Axis 20, 21, 22 (Z-axis cross products)
    p0 = v0[1]*f0[0] - v0[0]*f0[1]; p2 = v2[1]*f0[0] - v2[0]*f0[1];
    r = boxHalfExtents[0]*fabsf(f0[1]) + boxHalfExtents[1]*fabsf(f0[0]);
    if (fminf(p0, p2) > r || fmaxf(p0, p2) < -r) return false;

    p0 = v0[1]*f1[0] - v0[0]*f1[1]; p1 = v1[1]*f1[0] - v1[0]*f1[1];
    r = boxHalfExtents[0]*fabsf(f1[1]) + boxHalfExtents[1]*fabsf(f1[0]);
    if (fminf(p0, p1) > r || fmaxf(p0, p1) < -r) return false;

    p0 = v0[1]*f2[0] - v0[0]*f2[1]; p1 = v1[1]*f2[0] - v1[0]*f2[1];
    r = boxHalfExtents[0]*fabsf(f2[1]) + boxHalfExtents[1]*fabsf(f2[0]);
    if (fminf(p0, p1) > r || fmaxf(p0, p1) < -r) return false;

    // Face normal test
    nanovdb::Vec3f n(f0[1]*f1[2] - f0[2]*f1[1], f0[2]*f1[0] - f0[0]*f1[2], f0[0]*f1[1] - f0[1]*f1[0]);
    float d = n[0]*v0[0] + n[1]*v0[1] + n[2]*v0[2];
    r = boxHalfExtents[0]*fabsf(n[0]) + boxHalfExtents[1]*fabsf(n[1]) + boxHalfExtents[2]*fabsf(n[2]);
    if (fabsf(d) > r) return false;

    return true;
}

template <typename BuildT, bool OnlyUseAABB>
__global__ void evaluateAndCountSubBoxesKernel(
    const typename MeshToGrid<BuildT>::BoxTrianglePair* dParents,
    const std::array<nanovdb::Vec3f, 3>* dXformedTriangles,
    nanovdb::Mask<3>* dMasks,
    uint64_t* dCounts,
    int parentScale,
    float padding)
{
    // 1 CTA perfectly evaluates 1 Parent Pair
    uint64_t parentID = blockIdx.x;
    int threadID = threadIdx.x; // 0 to 511

    const auto& parentPair = dParents[parentID];
    const auto& tri = dXformedTriangles[parentPair.triangleID];

    int subScale = parentScale / 8;

    // 1. Thread to 3D sub-box index mapping
    int i = threadID & 7;           // % 8
    int j = (threadID >> 3) & 7;    // (/ 8) % 8
    int k = (threadID >> 6) & 7;    // / 64

    // 2. Mathematically exact bounding box for this sub-domain
    // Voxel bounds are [origin - 0.5, origin + subScale - 0.5]
    float centerX = parentPair.origin[0] + i * subScale + (subScale * 0.5f) - 0.5f;
    float centerY = parentPair.origin[1] + j * subScale + (subScale * 0.5f) - 0.5f;
    float centerZ = parentPair.origin[2] + k * subScale + (subScale * 0.5f) - 0.5f;
    
    nanovdb::Vec3f boxCenter(centerX, centerY, centerZ);
    float halfExt = (subScale * 0.5f) + padding;
    nanovdb::Vec3f boxHalfExtents(halfExt, halfExt, halfExt);

    // 3. Evaluate intersection
    bool hit = testTriangleAABB<OnlyUseAABB>(boxCenter, boxHalfExtents, tri[0], tri[1], tri[2]);

    // 4. ZERO-ATOMIC Mask Building using Warp Voting
    // Each warp (32 threads) creates a 32-bit mask of its hits.
    __shared__ uint32_t s_words_32[16]; // 16 warps * 32 bits = 512 bits total
    
    unsigned int ballot = __ballot_sync(0xFFFFFFFF, hit);
    if ((threadID & 31) == 0) {
        // The first thread of each warp writes the warp's result into shared memory
        s_words_32[threadID >> 5] = ballot;
    }

    // 5. CUB Block-wide Reduction to get the total hit count
    using BlockReduce = cub::BlockReduce<int, 512>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    int aggregate_hits = BlockReduce(temp_storage).Sum(hit ? 1 : 0);

    __syncthreads();

    // 6. Thread 0 safely formats and flushes the data to global memory
    if (threadID == 0) {
        nanovdb::Mask<3> outMask;
        // Stitch the 32-bit words into NanoVDB's expected 64-bit words
        for (int w = 0; w < 8; ++w) {
            uint64_t low = s_words_32[w * 2];
            uint64_t high = s_words_32[w * 2 + 1];
            outMask.words()[w] = low | (high << 32);
        }
        
        dMasks[parentID] = outMask;
        dCounts[parentID] = aggregate_hits;
    }
}

} // namespace topology::detail

template<typename BuildT>
void MeshToGrid<BuildT>::processLeafTrianglePairs()
{
    // TODO: Handle null input case
    if (mTriangleCount == 0)
        throw std::runtime_error("MeshToGrid currently requires mTriangleCount > 0 (Holistic zero-handling pending).");

    int device = 0;
    cudaGetDevice(&device);

    int scale = 4096; // Start at Root node scale

    for (int pass = 0; pass < 3; ++pass) {
        if (mVerbose == 1) {
            printf("\n--- Subdivision Pass %d (Scale: %d -> %d) ---\n", 
                   pass, scale, scale / 8);
        }

         // 1. Allocate Mask<3> buffer for the CTA hit results
        //    Size: mBoxTrianglePairCount * sizeof(nanovdb::Mask<3>)
        nanovdb::cuda::DeviceBuffer maskBuffer = nanovdb::cuda::DeviceBuffer::create(
            mBoxTrianglePairCount * sizeof(nanovdb::Mask<3>), nullptr, device, mStream);
        if (maskBuffer.deviceData() == nullptr) {
            throw std::runtime_error("Failed to allocate mask buffer for subdivision pass");
        }
        auto* dMasks = static_cast<nanovdb::Mask<3>*>(maskBuffer.deviceData());

        // 2. Allocate Counts buffer for Prefix Sum
        //    Size: mBoxTrianglePairCount * sizeof(uint64_t)
        nanovdb::cuda::DeviceBuffer countsBuffer = nanovdb::cuda::DeviceBuffer::create(
            mBoxTrianglePairCount * sizeof(uint64_t), nullptr, device, mStream);
        if (countsBuffer.deviceData() == nullptr) {
            throw std::runtime_error("Failed to allocate counts buffer for subdivision pass");
        }
        auto* dCounts = static_cast<uint64_t*>(countsBuffer.deviceData());

        // 3. Evaluate & Count Kernel (Grid: mBoxTrianglePairCount blocks, Block: 512 threads)
        //    Action: Threads do AABB/SAT tests, write to Mask, block reduces count.

        // 4. Prefix Sum (CUB InclusiveScan)
        //    Action: Computes global offsets and newTotalChildPairs

        // 5. Allocate New Child Pair Buffer
        //    Size: newTotalChildPairs * sizeof(BoxTrianglePair)

        // 6. Scatter Kernel
        //    Action: Reads the Mask<3> and writes surviving sub-boxes to the new buffer

        // 7. The std::move Ping-Pong!
        // mBoxTrianglePairsBuffer = std::move(newChildPairsBuffer);
        // mBoxTrianglePairCount = newTotalChildPairs; 
        
        // Prepare for the next subdivision tier
        scale /= 8; 
    }

} // MeshToGrid<BuildT>::processLeafTrianglePairs

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

} // namespace tools::cuda

} // namespace nanovdb

#endif // NVIDIA_TOOLS_CUDA_MESHTOGRID_CUH_HAS_BEEN_INCLUDED
