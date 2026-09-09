// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file mesh_to_sdf_cuda_kernels.cu
/// @brief CUDA wrapper for the MeshToSDF example.

#include <nanovdb/GridHandle.h>
#include <nanovdb/cuda/DeviceBuffer.h>
#include <nanovdb/tools/cuda/MeshToSDF.cuh>
#include <nanovdb/util/cuda/Util.h>

#include <cstdint>
#include <vector>

using GridHandleT = nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>;

GridHandleT meshToSdf(const std::vector<nanovdb::Vec3f>& points,
                      const std::vector<nanovdb::Vec3i>& triangles,
                      const nanovdb::Map& map,
                      float bandWidth,
                      float isoValue)
{
    using Buffer = nanovdb::cuda::DeviceBuffer;
    using Converter = nanovdb::tools::cuda::MeshToSDF<nanovdb::ValueOnIndex>;

    const std::size_t pointBytes = points.size() * sizeof(nanovdb::Vec3f);
    const std::size_t triangleBytes = triangles.size() * sizeof(nanovdb::Vec3i);
    auto pointBuffer = Buffer::create(pointBytes, nullptr, false);
    auto triangleBuffer = Buffer::create(triangleBytes, nullptr, false);
    cudaCheck(cudaMemcpy(pointBuffer.deviceData(), points.data(), pointBytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(
        triangleBuffer.deviceData(), triangles.data(), triangleBytes, cudaMemcpyHostToDevice));

    Converter converter(
        static_cast<const nanovdb::Vec3f*>(pointBuffer.deviceData()), uint32_t(points.size()),
        static_cast<const nanovdb::Vec3i*>(triangleBuffer.deviceData()), uint32_t(triangles.size()),
        map);
    converter.setNarrowBandWidth(bandWidth);
    converter.setIsoValue(isoValue);

    auto handle = converter.build();
    cudaCheck(cudaDeviceSynchronize());
    return handle;
}
