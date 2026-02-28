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
// #include <nanovdb/tools/cuda/TopologyBuilder.cuh>
// #include <nanovdb/util/cuda/DeviceGridTraits.cuh>
// #include <nanovdb/util/cuda/Morphology.cuh>
#include <nanovdb/util/cuda/Timer.h>
// #include <nanovdb/util/cuda/Util.h>


namespace nanovdb {

namespace tools::cuda {

template <typename BuildT>
class MeshToGrid
{
    // using GridT  = NanoGrid<BuildT>;
    // using TreeT  = NanoTree<BuildT>;
    // using RootT  = NanoRoot<BuildT>;
    // using UpperT = NanoUpper<BuildT>;

    using PointT = nanovdb::Vec3f;
    using TriangleIndexT = nanovdb::Vec3i;
    using TriangleT = std::array<PointT,3>;

public:


    /// @brief Constructor
    /// @param devicePoints Vertex list for input triangle surface
    /// @param pointCount Vertex count for input triangle surface
    /// @param deviceTriangles Triangle index list
    /// @param triangleCount Triangle count
    /// @param map Affine map to be used in the conversion
    MeshToGrid(
        const nanovdb::Vec3f *devicePoints,
        const int pointCount,
        const nanovdb::Vec3i *deviceTriangles,
        const int triangleCount,
        const nanovdb::Map map = nanovdb::Map(),
        cudaStream_t stream = 0
    )
        : mStream(stream), mTimer(stream), mDevicePoints(devicePoints), mPointCount(pointCount),
         mDeviceTriangles(deviceTriangles), mTriangleCount(triangleCount), mMap(map)
    {}

    /// @brief Toggle on and off verbose mode
    /// @param level Verbose level: 0=quiet, 1=timing, 2=benchmarking
    void setVerbose(int level = 1) { mVerbose = level; }

    /// @brief Set the mode for checksum computation, which is disabled by default
    /// @param mode Mode of checksum computation
    // void setChecksum(CheckMode mode = CheckMode::Disable){mBuilder.mChecksum = mode;}

    /// @brief Set type of dilation operation
    /// @param op: NN_FACE=face neighbors, NN_FACE_EDGE=face and edge neibhros, NN_FACE_EDGE_VERTEX=26-connected neighbors
    // void setOperation(morphology::NearestNeighbors op) { mOp = op; }

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

    // void dilateRoot();

    // void dilateInternalNodes();

    // void processGridTreeRoot();

    // void dilateLeafNodes();

    static constexpr unsigned int mNumThreads = 128;// for kernels spawned via lambdaKernel (others may specialize)
    static unsigned int numBlocks(unsigned int n) {return (n + mNumThreads - 1) / mNumThreads;}

    // TopologyBuilder<BuildT>      mBuilder;
    cudaStream_t                 mStream{0};
    util::cuda::Timer            mTimer;
    int                          mVerbose{0};
    const nanovdb::Vec3f         *mDevicePoints;
    const int                    mPointCount;
    const nanovdb::Vec3i         *mDeviceTriangles;
    const int                    mTriangleCount;
    const nanovdb::Map           mMap;

    nanovdb::cuda::DeviceBuffer  mXformedTriangles;
    // const GridT                  *mDeviceSrcGrid;
    // morphology::NearestNeighbors mOp{morphology::NN_FACE_EDGE_VERTEX};
    // TreeData                     mSrcTreeData;

    auto deviceXformedTriangles() { return static_cast<TriangleT*>(mXformedTriangles.deviceData()); }
    auto hostXformedTriangles() { return static_cast<TriangleT*>(mXformedTriangles.data()); }

    nanovdb::cuda::TempDevicePool mTempDevicePool;
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
    float mPadding{0.5f}; // sqrt(3)/2-0.5 should suffice for including the circumsphere of each voxel, but being conservative for safety

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

        // 5. Write the total count of root boxes this triangle touches
        dCounts[triangleID] = count_x * count_y * count_z;
    }
};

} // namespace topology::detail

template<typename BuildT>
void MeshToGrid<BuildT>::processRootTrianglePairs()
{
    int device = 0;
    cudaGetDevice(&device);

    // Allocate the counts array
    nanovdb::cuda::DeviceBuffer
        rootBoxCounts = nanovdb::cuda::DeviceBuffer::create(mTriangleCount * sizeof(uint64_t), nullptr, device, mStream);
    if (rootBoxCounts.deviceData() == nullptr) throw std::runtime_error("Failed to allocate root box counts buffer");
    
    // Pass 1: Count intersecting root boxes per triangle
    if (mVerbose == 1) mTimer.restart("[In MeshToGrid::processRootTrianglePairs()] Launching processRootTrianglePairs");
    
    util::cuda::lambdaKernel<<<numBlocks(mTriangleCount), mNumThreads, 0, mStream>>>(
        mTriangleCount, 
        topology::detail::CountRootBoxesFunctor<BuildT>{
            deviceXformedTriangles(),
            static_cast<uint64_t*>(rootBoxCounts.deviceData())
        }
    );

    cudaCheckError();

    // Pass 2: Inclusive Scan to compute offsets and total allocations
    
    if (mVerbose == 1) mTimer.restart("[In MeshToGrid::processRootTrianglePairs()] Prefix sum");
    
    // Allocate offsets array of size mTriangleCount + 1
    nanovdb::cuda::DeviceBuffer rootBoxOffsets = nanovdb::cuda::DeviceBuffer::create(
        (mTriangleCount + 1) * sizeof(uint64_t), nullptr, device, mStream);
    if (rootBoxOffsets.deviceData() == nullptr) throw std::runtime_error("Failed to allocate root box offsets buffer");

    uint64_t* dCounts = static_cast<uint64_t*>(rootBoxCounts.deviceData());
    uint64_t* dOffsets = static_cast<uint64_t*>(rootBoxOffsets.deviceData());

    // Explicitly set the very first offset to 0
    cudaMemsetAsync(dOffsets, 0, sizeof(uint64_t), mStream);

    // Run inclusive prefix sum using the newly integrated CALL_CUBS macro
    CALL_CUBS(DeviceScan::InclusiveSum, dCounts, dOffsets + 1, mTriangleCount);

    // Read back the grand total of pairs from the very last element
    uint64_t totalPairs = 0;
    cudaMemcpyAsync(&totalPairs, dOffsets + mTriangleCount, sizeof(uint64_t), 
                    cudaMemcpyDeviceToHost, mStream);

    // We MUST synchronize here because the CPU needs 'totalPairs' to allocate memory for Pass 3
    cudaStreamSynchronize(mStream);

    if (mVerbose == 1) {
        printf("Total Root/Triangle pairs to generate: %d\n", totalPairs);
    }


} // MeshToGrid<BuildT>::transformTriangles


//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#if 0
template<typename BuildT>
void MeshToGrid<BuildT>::dilateRoot()
{
    // This method conservatively and speculatively dilates the root tiles, to accommodate
    // any new root nodes that might be introduced by the dilation operation.
    // The index-space bounding box of each tile is examined, and if it is within a 1-pixel of
    // intersecting any of the 26-connected neighboring root tiles, those are preemptively
    // introduced into the root topology.
    // (As of the present implementation this presumes a maximum of 1-voxel radius in dilation)
    // Root tiles that were preemptively introduced, but end up having no active contents will
    // be pruned in later stages of processing.

    int device = 0;
    cudaGetDevice(&device);

    std::map<uint64_t, typename RootT::DataType::Tile> dilatedTiles;

    // This encoding scheme mirrors the one used in PointsToGrid; note that it is different from Tile::key
    auto coordToKey = [](const Coord &ijk)->uint64_t{
        // Note: int32_t has a range of -2^31 to 2^31 - 1 whereas uint32_t has a range of 0 to 2^32 - 1
        static constexpr int64_t kOffset = 1 << 31;
        return (uint64_t(uint32_t(int64_t(ijk[2]) + kOffset) >> 12)      ) | // z is the lower 21 bits
            (uint64_t(uint32_t(int64_t(ijk[1]) + kOffset) >> 12) << 21) | // y is the middle 21 bits
            (uint64_t(uint32_t(int64_t(ijk[0]) + kOffset) >> 12) << 42); //  x is the upper 21 bits
    };// coordToKey lambda functor

    if (mSrcTreeData.mVoxelCount) { // If the input grid is not empty
        // Make a host copy of the source topology RootNode *and* the Upper Nodes (needed for BBox'es)
        // TODO: Consider avoiding to copy the entire set of upper nodes
        auto deviceSrcRoot = static_cast<const RootT*>(util::PtrAdd(mDeviceSrcGrid, GridT::memUsage() + mSrcTreeData.mNodeOffset[3]));
        uint64_t rootAndUpperSize = mSrcTreeData.mNodeOffset[1] - mSrcTreeData.mNodeOffset[3];
        auto srcRootAndUpperBuffer = nanovdb::HostBuffer::create(rootAndUpperSize);
        cudaCheck(cudaMemcpyAsync(srcRootAndUpperBuffer.data(), deviceSrcRoot, rootAndUpperSize, cudaMemcpyDeviceToHost, mStream));
        auto srcRootAndUpper = static_cast<RootT*>(srcRootAndUpperBuffer.data());

        // For each original root tile, consider adding those tiles in its 26-connected neighborhood
        for (uint32_t t = 0; t < srcRootAndUpper->tileCount(); t++) {
            auto srcUpper = srcRootAndUpper->getChild(srcRootAndUpper->tile(t));
            const auto dilatedBBox = srcUpper->bbox().expandBy(1); // TODO: update/specialize if larger dilation neighborhoods are used

            static constexpr int32_t rootTileDim = UpperT::DIM; // 4096
            for (int di = -rootTileDim; di <= rootTileDim; di += rootTileDim)
            for (int dj = -rootTileDim; dj <= rootTileDim; dj += rootTileDim)
            for (int dk = -rootTileDim; dk <= rootTileDim; dk += rootTileDim) {
                auto testBBox = nanovdb::CoordBBox::createCube(srcUpper->origin().offsetBy(di,dj,dk), rootTileDim);
                auto sortKey = coordToKey(testBBox.min()); // key used in the radix sort, in accordance with PointsToGrid
                auto tileKey = RootT::CoordToKey(testBBox.min()); // encoding used in the NanoVDB tile
                if (testBBox.hasOverlap(dilatedBBox) & (dilatedTiles.count(sortKey) == 0)) {
                    typename RootT::Tile neighborTile{tileKey}; // Only the key value is needed; child pointer & value will be unused
                    dilatedTiles.emplace(sortKey, neighborTile);
                }
            }
        }
    }

    // Package the new root topology into a RootNode plus Tile list; upload to the GPU
    uint64_t rootSize = RootT::memUsage(dilatedTiles.size());
    mBuilder.mProcessedRoot = nanovdb::cuda::DeviceBuffer::create(rootSize);
    auto dilatedRootPtr = static_cast<RootT*>(mBuilder.mProcessedRoot.data());
    dilatedRootPtr->mTableSize = dilatedTiles.size();
    uint32_t t = 0;
    for (const auto& [key, tile] : dilatedTiles)
        *dilatedRootPtr->tile(t++) = tile;
    mBuilder.mProcessedRoot.deviceUpload(device, mStream, false);
} // MeshToGrid<BuildT>::dilateRoot

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT>
void MeshToGrid<BuildT>::dilateInternalNodes()
{
    // Computes the masks of upper and (densified) lower internal nodes, as a result of the dilation operation
    // Masks of lower internal nodes are densified in the sense that a serialized array of them is allocated,
    // as if every upper node had a full set of 32^3 lower children
    if (mSrcTreeData.mNodeCount[1]) { // Unless it's an empty grid
        if (mOp == morphology::NN_FACE) {
            using Op = util::morphology::cuda::DilateInternalNodesFunctor<BuildT, morphology::NN_FACE>;
            util::cuda::operatorKernel<Op>
                <<<dim3(mSrcTreeData.mNodeCount[1],Op::SlicesPerLowerNode,1), Op::MaxThreadsPerBlock, 0, mStream>>>
                (mDeviceSrcGrid, mBuilder.deviceProcessedRoot(), mBuilder.deviceUpperMasks(), mBuilder.deviceLowerMasks()); }
        else if (mOp == morphology::NN_FACE_EDGE) {
            using Op = util::morphology::cuda::DilateInternalNodesFunctor<BuildT, morphology::NN_FACE_EDGE>;
            util::cuda::operatorKernel<Op>
                <<<dim3(mSrcTreeData.mNodeCount[1],Op::SlicesPerLowerNode,1), Op::MaxThreadsPerBlock, 0, mStream>>>
                (mDeviceSrcGrid, mBuilder.deviceProcessedRoot(), mBuilder.deviceUpperMasks(), mBuilder.deviceLowerMasks()); }
        else if (mOp == morphology::NN_FACE_EDGE_VERTEX) {
            using Op = util::morphology::cuda::DilateInternalNodesFunctor<BuildT, morphology::NN_FACE_EDGE_VERTEX>;
            util::cuda::operatorKernel<Op>
                <<<dim3(mSrcTreeData.mNodeCount[1],Op::SlicesPerLowerNode,1), Op::MaxThreadsPerBlock, 0, mStream>>>
                (mDeviceSrcGrid, mBuilder.deviceProcessedRoot(), mBuilder.deviceUpperMasks(), mBuilder.deviceLowerMasks()); }
    }
} // MeshToGrid<BuildT>::dilateInternalNodes

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
void MeshToGrid<BuildT>::processGridTreeRoot()
{
    // Copy GridData from source grid
    // By convention: this will duplicate grid name and map. Others will be reset later
    cudaCheck(cudaMemcpyAsync(&mBuilder.data()->getGrid(), mDeviceSrcGrid->data(), GridT::memUsage(), cudaMemcpyDeviceToDevice, mStream));
    util::cuda::lambdaKernel<<<1, 1, 0, mStream>>>(1, topology::detail::BuildGridTreeRootFunctor<BuildT>(), mBuilder.deviceData());
    cudaCheckError();
} // MeshToGrid<BuildT>::processGridTreeRoot

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT>
void MeshToGrid<BuildT>::dilateLeafNodes()
{
    // Dilates the active masks of the source grid (as indicated at the leaf level), into a new grid that
    // has been already topologically dilated to include all necessary leaf nodes.
    if (mBuilder.data()->nodeCount[1]) { // Unless output grid is empty
        if (mOp == morphology::NN_FACE) {
            using Op = util::morphology::cuda::DilateLeafNodesFunctor<BuildT, morphology::NN_FACE>;
            util::cuda::operatorKernel<Op>
                <<<dim3(mBuilder.data()->nodeCount[1],Op::SlicesPerLowerNode,1), Op::MaxThreadsPerBlock, 0, mStream>>>
                (mDeviceSrcGrid, static_cast<GridT*>(mBuilder.data()->d_bufferPtr)); }
        else if (mOp == morphology::NN_FACE_EDGE)
            throw std::runtime_error("dilateLeafNodes() not implemented for NN_FACE_EDGE stencil");
        else if (mOp == morphology::NN_FACE_EDGE_VERTEX) {
            using Op = util::morphology::cuda::DilateLeafNodesFunctor<BuildT, morphology::NN_FACE_EDGE_VERTEX>;
            util::cuda::operatorKernel<Op>
                <<<dim3(mBuilder.data()->nodeCount[1],Op::SlicesPerLowerNode,1), Op::MaxThreadsPerBlock>>>
                (mDeviceSrcGrid, static_cast<GridT*>(mBuilder.data()->d_bufferPtr)); }
    }

    // Update leaf offsets and prefix sums
    mBuilder.processLeafOffsets(mStream);
} // MeshToGrid<BuildT>::dilateLeafNodes

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#endif

} // namespace tools::cuda

} // namespace nanovdb

#endif // NVIDIA_TOOLS_CUDA_MESHTOGRID_CUH_HAS_BEEN_INCLUDED
