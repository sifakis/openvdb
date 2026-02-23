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

#include <thrust/universal_vector.h>

void readOBJ(const std::string& filename,
             std::vector<openvdb::Vec3s>& points,
             std::vector<openvdb::Vec3I>& triangles,
             std::vector<openvdb::Vec4I>& quads) {
             
    std::ifstream file(filename);
    if (!file.is_open()) {
        OPENVDB_THROW(openvdb::IoError, "Failed to open OBJ file: " + filename);
    }

    std::string line;
    int lineNumber = 0;

    while (std::getline(file, line)) {
        lineNumber++;
        std::istringstream iss(line);
        std::string type;
        iss >> type;
        
        if (type == "v") {
            float x, y, z;
            iss >> x >> y >> z;
            points.push_back(openvdb::Vec3s(x, y, z));
        } else if (type == "f") {
            std::vector<int> faceIndices;
            std::string vertexData;
            
            while (iss >> vertexData) {
                // Isolate the vertex index (everything before the first slash)
                size_t slashPos = vertexData.find('/');
                std::string indexStr = vertexData.substr(0, slashPos);
                
                if (indexStr.empty()) continue;

                int raw_idx = std::stoi(indexStr);
                int actual_idx = 0;
                
                // Handle negative indices: relative to the number of points parsed so far
                if (raw_idx < 0) {
                    actual_idx = points.size() + raw_idx; 
                } else {
                    // Standard positive indices: OBJ is 1-based, convert to 0-based for C++
                    actual_idx = raw_idx - 1; 
                }

                // Strict bounds checking to prevent segfaults
                if (actual_idx < 0 || actual_idx >= points.size()) {
                    OPENVDB_THROW(openvdb::ValueError, 
                        "OBJ parse error on line " + std::to_string(lineNumber) + 
                        ": Face index out of bounds (Raw: " + std::to_string(raw_idx) + 
                        ", Computed: " + std::to_string(actual_idx) + ", Total Points: " + 
                        std::to_string(points.size()) + ")");
                }

                faceIndices.push_back(actual_idx); 
            }
            
            // Add to the appropriate OpenVDB list
            if (faceIndices.size() == 3) {
                triangles.push_back(openvdb::Vec3I(faceIndices[0], faceIndices[1], faceIndices[2]));
            } else if (faceIndices.size() == 4) {
                quads.push_back(openvdb::Vec4I(faceIndices[0], faceIndices[1], faceIndices[2], faceIndices[3]));
            } else if (faceIndices.size() > 4) {
                std::cerr << "Warning on line " << lineNumber << ": Skipping face with " 
                          << faceIndices.size() << " vertices. Triangulate your mesh!" << std::endl;
            }
        }
    }
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
        float voxelSize = 0.001f;
        if (argc > 3)
            voxelSize = atof(argv[3]);

        std::vector<openvdb::Vec3s> points;
        std::vector<openvdb::Vec3I> triangles;
        std::vector<openvdb::Vec4I> quads;

        // Read the OBJ file
        std::cout << "Reading " << inputFile << "..." << std::endl;
        readOBJ(inputFile, points, triangles, quads);
        std::cout << "Loaded " << points.size() << " vertices, " 
                  << triangles.size() << " triangles, and " 
                  << quads.size() << " quads." << std::endl;

        // Initialize OpenVDB
        openvdb::initialize();

        // Setup Grid Transform (Voxel Size)
        openvdb::math::Transform::Ptr transform =
            openvdb::math::Transform::createLinearTransform(voxelSize);

        // Convert Mesh to Level Set (SDF)
        // halfband specifies the half-width of the narrow band in voxel units
        float halfband = 3.0f; 
        cpuTimer.start("Converting mesh to OpenVDB level set");
        openvdb::FloatGrid::Ptr grid = openvdb::tools::meshToLevelSet<openvdb::FloatGrid>(
        *transform, points, triangles, quads, halfband);
        cpuTimer.stop();


        // Write the Grid to a VDB File
        grid->setName("LevelSet");
        std::cout << "Writing to " << outputFile << "..." << std::endl;
        openvdb::GridPtrVec grids;
        grids.push_back(grid);
        openvdb::io::File file(outputFile);
        file.write(grids);
        file.close();

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
    }
    return 0;
}
