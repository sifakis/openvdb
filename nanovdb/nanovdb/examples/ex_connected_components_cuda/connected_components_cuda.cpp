// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file  connected_components_cuda.cpp
///
/// @brief Host driver for the connected-components example (NanoVDB / CUDA only,
///        no OpenVDB). Reads a triangle mesh from a Wavefront .obj file, builds the
///        index<->world transform, and hands the mesh off to the CUDA side, which
///        voxelizes it into a ValueOnIndex grid + UDF sidecar. Connected-components
///        labeling on top of that grid will be added as a subsequent step.
///
///        See MeshToSDFDevelopmentPlan.md in this directory for the design notes and roadmap.

#include <nanovdb/NanoVDB.h>          // host-usable: Vec3f, Vec3i, Vec3d, Map
#include <nanovdb/GridHandle.h>       // GridHandle (header-only, host-usable)
#include <nanovdb/cuda/DeviceBuffer.h>// nanovdb::cuda::DeviceBuffer

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

// Types of the rasterization result handed across the host/device seam.
using GridHandleT = nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>;
using UDFSidecarT = nanovdb::cuda::DeviceBuffer;

/// @brief Implemented on the CUDA side (connected_components_cuda_kernels.cu):
///        uploads the mesh and voxelizes it into a ValueOnIndex grid plus a per-active-
///        voxel unsigned-distance-field sidecar.
/// @return { index-grid handle, UDF sidecar } (device buffers)
std::pair<GridHandleT, UDFSidecarT> computeUDF(
    const std::vector<nanovdb::Vec3f>& points,
    const std::vector<nanovdb::Vec3i>& triangles,
    const nanovdb::Map&                map,
    float                              bandWidth);

/// @brief Implemented on the CUDA side: prints topology diagnostics for the device-
///        resident index grid (active voxels, node counts, bbox, occupancy, memory).
void printGridDiagnostics(const GridHandleT& handle, const std::string& title);

/// @brief Implemented on the CUDA side: derive the connected-components input topology by
///        pruning the surface/barrier shell (voxels within √3/2 voxels of the surface) from
///        the rasterized index grid via PruneGrid. Returns a clean, topology-only index grid.
///        voxelSize converts the √3/2-voxel barrier into the sidecar's world-space units.
GridHandleT computeDerivedTopology(const GridHandleT& srcHandle, const UDFSidecarT& udfSidecar,
                                   float voxelSize);

/// @brief Minimal Wavefront .obj reader (vertices + faces) using NanoVDB types.
///
///        Polygons with more than 3 vertices are fan-triangulated. Vertex references
///        of the form `v`, `v/vt`, `v//vn`, `v/vt/vn` are accepted, as are negative
///        (relative) indices. Lines that are not `v` or `f` are ignored.
static void readOBJ(const std::string&            filename,
                    std::vector<nanovdb::Vec3f>&  points,
                    std::vector<nanovdb::Vec3i>&  triangles)
{
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Failed to open OBJ file: " + filename);

    std::string line;
    int lineNumber = 0;
    while (std::getline(file, line)) {
        ++lineNumber;
        std::istringstream iss(line);
        std::string type;
        iss >> type;

        if (type == "v") {
            float x, y, z;
            iss >> x >> y >> z;
            points.emplace_back(x, y, z);
        } else if (type == "f") {
            std::vector<int> face;
            std::string vert;
            while (iss >> vert) {
                const size_t slash = vert.find('/');
                const std::string idxStr = vert.substr(0, slash);
                if (idxStr.empty()) continue;
                int raw = std::stoi(idxStr);
                // OBJ indices are 1-based; negatives are relative to points read so far.
                int idx = (raw < 0) ? int(points.size()) + raw : raw - 1;
                if (idx < 0 || idx >= int(points.size()))
                    throw std::runtime_error("OBJ parse error on line " +
                                             std::to_string(lineNumber) +
                                             ": face index out of bounds");
                face.push_back(idx);
            }
            for (size_t i = 2; i < face.size(); ++i)
                triangles.emplace_back(face[0], face[i - 1], face[i]);
        }
    }
}

int main(int argc, char* argv[])
{
    try {
        if (argc < 2)
            throw std::runtime_error("usage: " + std::string(argv[0]) +
                                     " input.obj [voxelSize] [bandWidth]");

        const std::string inputFile = argv[1];
        const float voxelSize = (argc > 2) ? std::stof(argv[2]) : 0.01f;
        const float bandWidth = (argc > 3) ? std::stof(argv[3]) : 3.0f;

        std::vector<nanovdb::Vec3f> points;
        std::vector<nanovdb::Vec3i> triangles;
        std::cout << "Reading " << inputFile << "...\n";
        readOBJ(inputFile, points, triangles);
        std::cout << "Loaded " << points.size() << " vertices, "
                  << triangles.size() << " triangles.\n";
        if (points.empty() || triangles.empty())
            throw std::runtime_error("mesh has no triangles");

        // Index<->world transform: uniform voxel size, no translation. (nanovdb::Map)
        nanovdb::Map map;
        map.set(double(voxelSize), nanovdb::Vec3d(0.0), 1.0);

        // Step 1: rasterize the mesh into an index grid + UDF sidecar.
        auto [handle, sidecar] = computeUDF(points, triangles, map, bandWidth);

        printGridDiagnostics(handle, "Rasterized UDF grid");
        std::cout << "UDF sidecar                           : "
                  << (sidecar.size() / sizeof(float)) << " floats\n";

        // Step 2: derive the CC-input topology by pruning the surface/barrier shell.
        auto derivedHandle = computeDerivedTopology(handle, sidecar, voxelSize);
        printGridDiagnostics(derivedHandle, "Derived CC-input grid");

        // Step 3 (TODO): run the CUDA connected-components labeling on derivedHandle.

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"\n";
        return 1;
    }
}
