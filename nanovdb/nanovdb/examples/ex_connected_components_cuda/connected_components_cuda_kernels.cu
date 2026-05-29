// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file  connected_components_cuda_kernels.cu
///
/// @brief CUDA / NanoVDB side of the connected-components example (no OpenVDB).
///
///        computeUDF(): uploads the triangle mesh and voxelizes it into a narrow-band
///        ValueOnIndex grid plus a per-active-voxel unsigned-distance-field (UDF)
///        sidecar, via nanovdb::tools::cuda::MeshToGrid.
///
///        The connected-components labeling (a union-find with hierarchical insights)
///        and the derivative-grid step it operates on will be added here as follow-ups.
///        See MeshToSDFDevelopmentPlan.md in this directory for the design notes and roadmap.

#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/cuda/MeshToGrid.cuh>
#include <nanovdb/util/cuda/DeviceGridTraits.cuh>

#include <thrust/universal_vector.h>

#include <iostream>
#include <string>
#include <utility>
#include <vector>

// Must match the aliases in connected_components_cuda.cpp.
using GridHandleT = nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>;
using UDFSidecarT = nanovdb::cuda::DeviceBuffer;

std::pair<GridHandleT, UDFSidecarT> computeUDF(
    const std::vector<nanovdb::Vec3f>& points,
    const std::vector<nanovdb::Vec3i>& triangles,
    const nanovdb::Map&                map,
    float                              bandWidth)
{
    using BuildT = nanovdb::ValueOnIndex;

    // Upload mesh to the device (managed memory keeps the example simple).
    thrust::universal_vector<nanovdb::Vec3f> dPoints(points.begin(), points.end());
    thrust::universal_vector<nanovdb::Vec3i> dTriangles(triangles.begin(), triangles.end());

    nanovdb::tools::cuda::MeshToGrid<BuildT> converter(
        dPoints.data().get(),    uint32_t(dPoints.size()),
        dTriangles.data().get(), uint32_t(dTriangles.size()),
        map);
    converter.setVerbose(1);
    converter.setNarrowBandWidth(bandWidth);

    // { index-grid handle, UDF sidecar }, both backed by DeviceBuffer.
    return converter.getHandleAndUDF();
}

/// @brief Print topology diagnostics for a device-resident ValueOnIndex grid, in the
///        style of ex_dilate_nanovdb_cuda. Reads the grid's header/tree fields directly
///        off the device via DeviceGridTraits (small targeted cudaMemcpy's) - no full
///        deviceDownload of the grid buffer is required.
void printGridDiagnostics(const GridHandleT& handle, const std::string& title)
{
    using BuildT = nanovdb::ValueOnIndex;
    using Traits = nanovdb::util::cuda::DeviceGridTraits<BuildT>;

    const auto* d_grid = reinterpret_cast<const nanovdb::NanoGrid<BuildT>*>(handle.deviceData());

    const auto     treeData     = Traits::getTreeData(d_grid);
    const uint64_t valueCount   = Traits::getValueCount(d_grid);
    const uint64_t activeVoxels = Traits::getActiveVoxelCount(d_grid);
    const uint64_t gridSize     = Traits::getGridSize(d_grid);
    const auto     bbox         = Traits::getIndexBBox(d_grid, treeData);

    const uint64_t leafNodes  = treeData.mNodeCount[0];
    const uint64_t lowerNodes = treeData.mNodeCount[1];
    const uint64_t upperNodes = treeData.mNodeCount[2];

    std::cout << "============ " << title << " ============\n";
    std::cout << "Allocated values [valueCount()]       : " << valueCount << "\n";
    std::cout << "Active voxels    [activeVoxelCount()] : " << activeVoxels << "\n";
    std::cout << "Index-space bounding box              : ["
              << bbox.min().x() << "," << bbox.min().y() << "," << bbox.min().z() << "] -> ["
              << bbox.max().x() << "," << bbox.max().y() << "," << bbox.max().z() << "]\n";
    std::cout << "Leaf nodes                            : " << leafNodes  << "\n";
    std::cout << "Lower internal nodes                  : " << lowerNodes << "\n";
    std::cout << "Upper internal nodes                  : " << upperNodes << "\n";
    std::cout << "Leaf-level occupancy                  : "
              << (leafNodes ? 100.f * float(activeVoxels) / float(leafNodes * 512) : 0.f)
              << "%\n";
    std::cout << "Memory usage                          : " << gridSize << " bytes\n";
}
