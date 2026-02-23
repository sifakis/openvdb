// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

// the following files are from OpenVDB
#include <openvdb/tools/Morphology.h>
#include <openvdb/util/CpuTimer.h>
#include <openvdb/tools/MeshToVolume.h>

// the following files are from NanoVDB
#include <nanovdb/NanoVDB.h>
#include <nanovdb/cuda/DeviceBuffer.h>
#include <nanovdb/tools/CreateNanoGrid.h>

template<typename BuildT>
void mainDilateGrid(
    nanovdb::NanoGrid<BuildT> *deviceGridOriginal,
    nanovdb::NanoGrid<BuildT> *deviceGridDilated,
    nanovdb::NanoGrid<BuildT> *indexGridOriginal,
    nanovdb::NanoGrid<BuildT> *indexGridDilated,
    uint32_t nnType,
    uint32_t benchmark_iters
);

// A simple OBJ reader to extract vertices and faces (triangles & quads)
bool readOBJ(const std::string& filename,
    std::vector<openvdb::Vec3s>& points,
    std::vector<openvdb::Vec3I>& triangles,
    std::vector<openvdb::Vec4I>& quads) {
             
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string type;
        iss >> type;
        
        if (type == "v") {
            // Vertex definitions
            float x, y, z;
            iss >> x >> y >> z;
            points.push_back(openvdb::Vec3s(x, y, z));
        } else if (type == "f") {
            // Face definitions
            std::vector<int> faceIndices;
            std::string vertexData;
            
            while (iss >> vertexData) {
                std::istringstream viss(vertexData);
                std::string indexStr;
                // Get the first item before the slash (position index)
                std::getline(viss, indexStr, '/'); 
                // OBJ uses 1-based indexing, so subtract 1 for 0-based arrays
                faceIndices.push_back(std::stoi(indexStr) - 1); 
            }
            
            // OpenVDB meshToLevelSet supports triangles and quads natively
            if (faceIndices.size() == 3) {
                triangles.push_back(openvdb::Vec3I(faceIndices[0], faceIndices[1], faceIndices[2]));
            } else if (faceIndices.size() == 4) {
                quads.push_back(openvdb::Vec4I(faceIndices[0], faceIndices[1], faceIndices[2], faceIndices[3]));
            } else {
                std::cerr << "Warning: Skipping face with " << faceIndices.size() << " vertices. Triangulate your mesh!" << std::endl;
            }
        }
    }
    return true;
}

/// @brief This example depends on OpenVDB, NanoVDB, and CUDA
int main(int argc, char *argv[])
{
    using GridT = openvdb::FloatGrid;
    using BuildT = nanovdb::ValueOnIndex;

    // Select the type of dilation here. The NN_EDGE case supports leaf dilation too (currently)
    // openvdb::tools::NearestNeighbors nnType = openvdb::tools::NN_FACE_EDGE_VERTEX;
    openvdb::tools::NearestNeighbors nnType = openvdb::tools::NN_FACE;

    openvdb::util::CpuTimer cpuTimer;

    try {

        if (argc<2) OPENVDB_THROW(openvdb::ValueError, "usage: "+std::string(argv[0])+" input.obj [output.vdb]\n");
        std::string inputFile = argv[1];
        std::string outputFile = "output.vdb";
        if (argc > 2)
            outputFile = argv[2];

        std::vector<openvdb::Vec3s> points;
        std::vector<openvdb::Vec3I> triangles;
        std::vector<openvdb::Vec4I> quads;

        // Read the OBJ file
        std::cout << "Reading " << inputFile << "..." << std::endl;
        if (!readOBJ(inputFile, points, triangles, quads))
            OPENVDB_THROW(openvdb::IoError, "Failed to open or read OBJ file: " + inputFile);
        std::cout << "Loaded " << points.size() << " vertices, " 
                  << triangles.size() << " triangles, and " 
                  << quads.size() << " quads." << std::endl;

        // Initialize OpenVDB
        openvdb::initialize();


    // Setup Grid Transform (Voxel Size)
    float voxelSize = 0.05f; 
    openvdb::math::Transform::Ptr transform =
        openvdb::math::Transform::createLinearTransform(voxelSize);

    // Convert Mesh to Level Set (SDF)
    // halfband specifies the half-width of the narrow band in voxel units
    float halfband = 3.0f; 
    std::cout << "Converting mesh to OpenVDB level set..." << std::endl;
    
    openvdb::FloatGrid::Ptr grid = openvdb::tools::meshToLevelSet<openvdb::FloatGrid>(
        *transform, points, triangles, quads, halfband);

    // Give the grid a recognizable name before saving
    grid->setName("LevelSet");

    // 5. Write the Grid to a VDB File
    std::cout << "Writing to " << outputFile << "..." << std::endl;
    openvdb::GridPtrVec grids;
    grids.push_back(grid);

    openvdb::io::File file(outputFile);
    file.write(grids);
    file.close();

    std::cout << "Done!" << std::endl;
    return 0;

#if 0

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

        // Launch benchmark
        mainDilateGrid( deviceGridOriginal, deviceGridDilated, indexGridOriginal, indexGridDilated, nnType, benchmark_iters );

#endif
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
    }
    return 0;
}
