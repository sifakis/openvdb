// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file mesh_to_sdf_cuda_kernels.cu
/// @brief CUDA wrapper for the MeshToSDF example.

#include <nanovdb/GridHandle.h>
#include <nanovdb/HostBuffer.h>
#include <nanovdb/cuda/Buffer.h>
#include <nanovdb/cuda/HandleStorage.h> // cuda::copyTo
#include <nanovdb/tools/cuda/MeshToSDF.cuh>
#include <nanovdb/util/cuda/Util.h>

#include <cstdint>
#include <vector>

using GridHandleT = nanovdb::GridHandle<nanovdb::HostBuffer>;

GridHandleT meshToSdf(const std::vector<nanovdb::Vec3f>& points,
                      const std::vector<nanovdb::Vec3i>& triangles,
                      const nanovdb::Map& map,
                      float bandWidth,
                      float isoValue)
{
    using Converter = nanovdb::tools::cuda::MeshToSDF<nanovdb::ValueOnIndex>;

    nanovdb::cuda::Buffer<nanovdb::Vec3f> pointBuffer(cudaStream_t(0), points.size(), nanovdb::cuda::noInit);
    nanovdb::cuda::Buffer<nanovdb::Vec3i> triangleBuffer(cudaStream_t(0), triangles.size(), nanovdb::cuda::noInit);
    cudaCheck(cudaMemcpy(pointBuffer.data(), points.data(), pointBuffer.size_bytes(), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(
        triangleBuffer.data(), triangles.data(), triangleBuffer.size_bytes(), cudaMemcpyHostToDevice));

    Converter converter(
        pointBuffer.data(), uint32_t(points.size()),
        triangleBuffer.data(), uint32_t(triangles.size()),
        map);
    converter.setNarrowBandWidth(bandWidth);
    converter.setIsoValue(isoValue);

    // The host driver reads and writes the grid, so it receives a host copy of the device result.
    return nanovdb::cuda::copyTo<nanovdb::HostBuffer>(converter.getHandle());
}
