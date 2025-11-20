// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

// the following files are from OpenVDB
#include <openvdb/tools/Morphology.h>
#include <openvdb/util/CpuTimer.h>

// the following files are from NanoVDB
#include <nanovdb/NanoVDB.h>
#include <nanovdb/cuda/DeviceBuffer.h>
#include <nanovdb/tools/CreateNanoGrid.h>

#include <random>

void mainSparseConvolutionIGEMM(
    const std::vector<nanovdb::Coord>& inputPoints,
    const std::vector<nanovdb::Coord>& outputPoints,
    uint32_t benchmark_iters);

uint32_t coordinate_bitpack(uint32_t x) {
    x &= 0x49249249; // keep only one every 3 bits
    x |= (x >> 2);
    x &= 0xc30c30c3; // Pack into pairs of bits
    x |= (x >> 4);
    x &= 0x0f00f00f; // Pack into quadruples of bits
    x |= (x >> 8);
    x &= 0xff0000ff; // Pack into quadruples of bits
    x |= (x >> 16);
    x &= 0x0000ffff; // Pack into 16-tuples (actually, 11 max) of bits
    return x;
}
/// @brief This example depends on OpenVDB, NanoVDB, and CUDA
int main(int argc, char *argv[])
{
    int benchmark_iters = 10;
    if (argc > 2) sscanf(argv[2], "%d", &benchmark_iters);

    std::random_device rd;
    std::mt19937 generator(rd());

    static const int ambient_voxels = 1024*1024*2;
    static const float occupancy = .5f;
    static const float overlap = .45f;
    nanovdb::Coord offset(-8,-16,-24);    

    // Mark input voxels at requested occupancy
    int target_active_voxels = (int) (occupancy*(float)ambient_voxels);
    std::vector<bool> voxmap(ambient_voxels);
    int active_voxels = 0;
    std::uniform_int_distribution<int> distribution(0, ambient_voxels-1);
    while (active_voxels < target_active_voxels) {
        int i = distribution(generator);
        if (!voxmap[i]) {
            voxmap[i] = true;
            active_voxels++;
        }
    }

    // Convert to coordinates
    std::vector<nanovdb::Coord> inputPoints;
    for (int i = 0; i < ambient_voxels; i++)
        if(voxmap[i]) {
        int x = coordinate_bitpack(     i & 0x49249249);
        int y = coordinate_bitpack((i>>1) & 0x49249249);
        int z = coordinate_bitpack((i>>2) & 0x49249249);
        inputPoints.emplace_back(nanovdb::Coord(x,y,z)+offset);
    }
    std::cout << inputPoints.size() << " input voxels generated" << std::endl;

    // Discard voxels until desired level of overlap
    int target_overlap_voxels = (int) (overlap*(float)ambient_voxels);
    while (active_voxels > target_overlap_voxels) {
        int i = distribution(generator);
        if (voxmap[i]) {
            voxmap[i] = false;
            active_voxels--;
        }
    }
    // Then sample more voxels until desired occupancy is met
    while (active_voxels < target_active_voxels) {
        int i = distribution(generator);
        if (!voxmap[i]) {
            voxmap[i] = true;
            active_voxels++;
        }
    }
    // Convert to coordinates
    std::vector<nanovdb::Coord> outputPoints;
    for (int i = 0; i < ambient_voxels; i++)
        if(voxmap[i]) {
        int x = coordinate_bitpack(     i & 0x49249249);
        int y = coordinate_bitpack((i>>1) & 0x49249249);
        int z = coordinate_bitpack((i>>2) & 0x49249249);
        outputPoints.emplace_back(nanovdb::Coord(x,y,z)+offset);
    }
    std::cout << outputPoints.size() << " output voxels generated" << std::endl;

    mainSparseConvolutionIGEMM(inputPoints, outputPoints, benchmark_iters);

#if 0
    using GridT = openvdb::FloatGrid;
    using BuildT = nanovdb::ValueOnIndex;

    // Select the type of dilation here. The NN_EDGE case supports leaf dilation too (currently)
    // openvdb::tools::NearestNeighbors nnType = openvdb::tools::NN_FACE_EDGE_VERTEX;
    openvdb::tools::NearestNeighbors nnType = openvdb::tools::NN_FACE;

    openvdb::util::CpuTimer cpuTimer;
    const bool printGridDiagnostics = true;

    try {

        if (argc<2) OPENVDB_THROW(openvdb::ValueError, "usage: "+std::string(argv[0])+" input.vdb [<iterations>]\n");
        int benchmark_iters = 10;
        if (argc > 2) sscanf(argv[2], "%d", &benchmark_iters);

        // Read the initial level set from file

        cpuTimer.start("Read input VDB file");
        openvdb::initialize();
        openvdb::io::File inFile(argv[1]);
        inFile.open(false); // disable delayed loading
        auto baseGrids = inFile.getGrids();
        inFile.close();
        auto grid = openvdb::gridPtrCast<GridT>(baseGrids->at(0));
        openvdb::FloatGrid* ptr = grid.get(); // raw pointer
        if (!grid) OPENVDB_THROW(openvdb::ValueError, "First grid is not a FloatGrid\n");
        cpuTimer.stop();

        // Convert to indexGrid (original, un-dilated)
        cpuTimer.start("Converting openVDB input to indexGrid (original version)");
        auto handleOriginal = nanovdb::tools::openToIndexVDB<BuildT, nanovdb::cuda::DeviceBuffer>(
            grid,
            0u,    // Don't copy data channel
            false, // No stats
            false, // No tiles
            1      // Verbose mode
        );
        auto *indexGridOriginal = handleOriginal.grid<BuildT>();
        cpuTimer.stop();

        if (printGridDiagnostics) {
            std::cout << "============ Original Grid ===========" << std::endl;
            std::cout << "Allocated values [valueCount()]       : " << indexGridOriginal->valueCount() << std::endl;
            std::cout << "Active voxels    [activeVoxelCount()] : " << indexGridOriginal->activeVoxelCount() << std::endl;
            auto minCorner = indexGridOriginal->indexBBox().min(), maxCorner = indexGridOriginal->indexBBox().max();
            std::cout << "Index-space bounding box              : [" << minCorner.x() << "," << minCorner.y() << "," << minCorner.z()
                      << "] -> [" << maxCorner.x() << "," << maxCorner.y() << "," << maxCorner.z() << "]" << std::endl;
            std::cout << "Leaf nodes                            : " << indexGridOriginal->tree().nodeCount(0) << std::endl;
            std::cout << "Lower internal nodes                  : " << indexGridOriginal->tree().nodeCount(1) << std::endl;
            std::cout << "Upper internal nodes                  : " << indexGridOriginal->tree().nodeCount(2) << std::endl;
            std::cout << "Leaf-level occupancy                  : "
                      << 100.f * (float)(indexGridOriginal->activeVoxelCount())/(float)(indexGridOriginal->tree().nodeCount(0) * 512)
                      << "%" << std::endl;
            std::cout << "Memory usage                          : " << indexGridOriginal->gridSize() << " bytes" << std::endl;
        }

        // Dilation (CPU/OpenVDB version)
        cpuTimer.start("Dilating openVDB (on CPU)");
        openvdb::tools::dilateActiveValues(grid->tree(), 1, nnType);
        cpuTimer.stop();

        // Convert to indexGrid (dilated)
        cpuTimer.start("Converting openVDB input to indexGrid (dilated version)");
        auto handleDilated = nanovdb::tools::openToIndexVDB<BuildT, nanovdb::cuda::DeviceBuffer>(
            grid,
            0u,    // Don't copy data channel
            false, // No stats
            false, // No tiles
            1      // Verbose mode
        );
        cpuTimer.stop();

        auto *indexGridDilated = handleDilated.grid<BuildT>();

        if (printGridDiagnostics) {
            std::cout << "============ Dilated Grid ============" << std::endl;
            std::cout << "Allocated values [valueCount()]       : " << indexGridDilated->valueCount() << std::endl;
            std::cout << "Active voxels    [activeVoxelCount()] : " << indexGridDilated->activeVoxelCount() << std::endl;
            auto minCorner = indexGridDilated->indexBBox().min(), maxCorner = indexGridDilated->indexBBox().max();
            std::cout << "Index-space bounding box              : [" << minCorner.x() << "," << minCorner.y() << "," << minCorner.z()
                      << "] -> [" << maxCorner.x() << "," << maxCorner.y() << "," << maxCorner.z() << "]" << std::endl;
            std::cout << "Leaf nodes                            : " << indexGridDilated->tree().nodeCount(0) << std::endl;
            std::cout << "Lower internal nodes                  : " << indexGridDilated->tree().nodeCount(1) << std::endl;
            std::cout << "Upper internal nodes                  : " << indexGridDilated->tree().nodeCount(2) << std::endl;
            std::cout << "Leaf-level occupancy                  : "
                      << 100.f * (float)(indexGridDilated->activeVoxelCount())/(float)(indexGridDilated->tree().nodeCount(0) * 512)
                      << "%" << std::endl;
            std::cout << "Memory usage                          : " << indexGridDilated->gridSize() << " bytes" << std::endl;
        }

        // Copy both NanoVDB grids to GPU
        handleOriginal.deviceUpload();
        handleDilated.deviceUpload();
        auto* deviceGridOriginal = handleOriginal.deviceGrid<BuildT>();
        auto* deviceGridDilated = handleDilated.deviceGrid<BuildT>();
        if (!deviceGridOriginal || !deviceGridDilated)
            OPENVDB_THROW(openvdb::RuntimeError, "Failure while uploading indexGrids to GPU");

    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
    }
#endif
    return 0;
}
