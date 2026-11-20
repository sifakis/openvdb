// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

// probe_sphere.cu: where the values MeshToSDF stores for a sphere differ from |x - c| - R.
// Prints, per bin of the exact distance (voxels), the mean and largest signed error (voxels).
#include "SdfField.cuh"
#include <cmath>
#include <cstdio>
#include <vector>
int main(int argc, char** argv)
{
    const int   resolution = argc > 1 ? std::atoi(argv[1]) : 30;
    const float band = 4.0f, radius = 0.5f, h = radius / float(resolution);
    const double c[3] = {0.7137, 0.6891, 0.7313};
    // a UV sphere, fine enough that its facets are within 1e-5 voxels of the sphere
    const int rings = 2000, sectors = 4000;
    std::vector<nanovdb::Vec3f> points;
    std::vector<nanovdb::Vec3i> triangles;
    for (int i = 0; i <= rings; ++i)
        for (int j = 0; j < sectors; ++j) {
            const double theta = M_PI * i / rings, phi = 2 * M_PI * j / sectors;
            points.emplace_back(float(c[0] + radius * std::sin(theta) * std::cos(phi)), float(c[1] + radius * std::cos(theta)), float(c[2] + radius * std::sin(theta) * std::sin(phi)));
        }
    for (int i = 0; i < rings; ++i)
        for (int j = 0; j < sectors; ++j) {
            const int a = i * sectors + j, b = i * sectors + (j + 1) % sectors, d = a + sectors, e = b + sectors;
            if (i > 0) triangles.emplace_back(a, b, e);
            if (i < rings - 1) triangles.emplace_back(a, e, d);
        }
    nanovdb::cuda::Buffer<nanovdb::Vec3f> dPoints(cudaStream_t(0), points.size(), nanovdb::cuda::noInit);
    nanovdb::cuda::Buffer<nanovdb::Vec3i> dTriangles(cudaStream_t(0), triangles.size(), nanovdb::cuda::noInit);
    cudaCheck(cudaMemcpy(dPoints.data(), points.data(), points.size() * 12, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dTriangles.data(), triangles.data(), triangles.size() * 12, cudaMemcpyHostToDevice));
    const cloth::SdfField field(dPoints.data(), uint32_t(points.size()), dTriangles.data(), uint32_t(triangles.size()), h, band, false);
    std::vector<char>  bytes(field.handle.buffer().size());
    std::vector<float> sdf(field.valueCount);
    cudaCheck(cudaMemcpy(bytes.data(), field.handle.buffer().data(), bytes.size(), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(sdf.data(), field.sdf.data(), sdf.size() * 4, cudaMemcpyDeviceToHost));
    const auto& tree = reinterpret_cast<const cloth::GridT*>(bytes.data())->tree();
    double sum[10] = {}, worst[10] = {};
    size_t count[10] = {};
    for (uint32_t l = 0; l < tree.nodeCount(0); ++l) {
        const auto& leaf = tree.getFirstLeaf()[l];
        for (uint32_t n = 0; n < 512; ++n) {
            if (!leaf.isActive(n)) continue;
            const auto ijk = leaf.offsetToGlobalCoord(n);
            const double x = ijk[0] * double(h) - c[0], y = ijk[1] * double(h) - c[1], z = ijk[2] * double(h) - c[2];
            const double exact = (std::sqrt(x * x + y * y + z * z) - radius) / h, error = sdf[leaf.getValue(n)] / h - exact;
            const int bin = std::min(9, std::max(0, int(std::floor(exact)) + 5));
            sum[bin] += error, ++count[bin];
            if (std::fabs(error) > std::fabs(worst[bin])) worst[bin] = error;
        }
    }
    std::printf("R/voxel %d, %zu triangles, background %.4f voxels\n exact distance (voxels) | voxels | mean error | largest error (voxels)\n", resolution, triangles.size(), sdf[0] / h);
    for (int b = 0; b < 10; ++b) std::printf("  [%+d, %+d) | %8zu | %+.5f | %+.5f\n", b - 5, b - 4, count[b], count[b] ? sum[b] / count[b] : 0.0, worst[b]);
    return 0;
}
