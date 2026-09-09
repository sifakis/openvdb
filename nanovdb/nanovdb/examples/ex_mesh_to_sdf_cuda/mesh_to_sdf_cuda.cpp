// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file mesh_to_sdf_cuda.cpp
/// @brief Convert a Wavefront OBJ mesh to a NanoVDB signed distance field on the GPU.

#include <nanovdb/NanoVDB.h>
#include <nanovdb/GridHandle.h>
#include <nanovdb/cuda/DeviceBuffer.h>
#include <nanovdb/io/IO.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using GridHandleT = nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>;

GridHandleT meshToSdf(const std::vector<nanovdb::Vec3f>& points,
                      const std::vector<nanovdb::Vec3i>& triangles,
                      const nanovdb::Map& map,
                      float bandWidth,
                      float isoValue);

namespace {

void readObj(const std::string& filename,
             std::vector<nanovdb::Vec3f>& points,
             std::vector<nanovdb::Vec3i>& triangles)
{
    std::ifstream file(filename);
    if (!file) throw std::runtime_error("Failed to open OBJ file: " + filename);

    std::string line;
    int lineNumber = 0;
    while (std::getline(file, line)) {
        ++lineNumber;
        std::istringstream input(line);
        std::string type;
        input >> type;

        if (type == "v") {
            float x, y, z;
            if (input >> x >> y >> z) points.emplace_back(x, y, z);
        } else if (type == "f") {
            std::vector<int> face;
            std::string vertex;
            while (input >> vertex) {
                const std::string index = vertex.substr(0, vertex.find('/'));
                if (index.empty()) continue;
                const int raw = std::stoi(index);
                const int i = raw < 0 ? int(points.size()) + raw : raw - 1;
                if (i < 0 || i >= int(points.size())) {
                    throw std::runtime_error(
                        "OBJ face index out of bounds on line " + std::to_string(lineNumber));
                }
                face.push_back(i);
            }
            for (std::size_t i = 2; i < face.size(); ++i)
                triangles.emplace_back(face[0], face[i - 1], face[i]);
        }
    }
}

void printUsage(const char* executable)
{
    std::cerr << "Usage: " << executable
              << " mesh.obj [voxel-size=0.01] [band-width=3] [iso-value=0]"
                 " [output=mesh_to_sdf.nvdb]\n";
}

} // namespace

int main(int argc, char** argv)
{
    try {
        if (argc < 2) {
            printUsage(argv[0]);
            return 1;
        }

        const float voxelSize = argc > 2 ? std::stof(argv[2]) : 0.01f;
        const float bandWidth = argc > 3 ? std::stof(argv[3]) : 3.0f;
        const float isoValue  = argc > 4 ? std::stof(argv[4]) : 0.0f;
        const std::string output = argc > 5 ? argv[5] : "mesh_to_sdf.nvdb";
        if (!(voxelSize > 0.0f)) throw std::runtime_error("voxel size must be positive");
        if (!(bandWidth > 0.0f)) throw std::runtime_error("band width must be positive");
        if (isoValue < 0.0f) throw std::runtime_error("iso value must be non-negative");

        std::vector<nanovdb::Vec3f> points;
        std::vector<nanovdb::Vec3i> triangles;
        readObj(argv[1], points, triangles);
        if (points.empty() || triangles.empty())
            throw std::runtime_error("mesh has no triangles");

        nanovdb::Map map;
        map.set(double(voxelSize), nanovdb::Vec3d(0.0), 1.0);

        auto handle = meshToSdf(points, triangles, map, bandWidth, isoValue);
        handle.deviceDownload(nullptr, true);
        const auto* grid = handle.grid<nanovdb::ValueOnIndex>();
        if (!grid) throw std::runtime_error("MeshToSDF returned no grid");

        nanovdb::io::writeGrid(output, handle);
        std::cout << "Wrote " << output << " with " << grid->activeVoxelCount()
                  << " active voxels and " << grid->blindDataCount()
                  << " blind-data channels\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
