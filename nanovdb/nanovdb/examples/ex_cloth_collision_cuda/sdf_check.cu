// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

// sdf_check.cu: checks of SdfField.cuh, the collision field of the cloth scene.
//   sdf_check sphere [band in voxels]
//       A triangulated sphere at radius 30, 60 and 120 voxels. The exact distance is |x - c| - R, so every number is
//       an error against a known answer: the values MeshToSDF stores, the value and gradient SdfView::sample()
//       returns at random points within (band - 1.75) voxels of the surface, and the sign far from the surface.
//   sdf_check character <dir> <frame> <voxel size> [band in voxels]
//       One pose of the animated character (<dir>/sequence.txt, vertices.bin, triangles.bin, as exported by
//       ex_fluid_apic_cuda/scene/export_character.py), against openvdb::tools::meshToLevelSet of the same mesh.
// Both modes also compare SdfView::sample() with openvdb::tools::BoxSampler on an OpenVDB grid that holds the same
// values (signs outside the band from openvdb::tools::signedFloodFill).
#include "SdfField.cuh"
#include <openvdb/openvdb.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/math/Proximity.h>
#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/tools/SignedFloodFill.h>
#include <tbb/parallel_for.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <vector>

using nanovdb::Vec3f;
using nanovdb::Vec3i;
template <typename T>
using DeviceArray = nanovdb::cuda::Buffer<T>;

template <typename T>
static DeviceArray<T> upload(const std::vector<T>& host)
{
    DeviceArray<T> device(cudaStream_t(0), host.size(), nanovdb::cuda::noInit);
    cudaCheck(cudaMemcpy(device.data(), host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    return device;
}

static double milliseconds(std::chrono::steady_clock::time_point since)
{
    cudaCheck(cudaDeviceSynchronize());
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - since).count();
}

// The host copy of a field: the grid's bytes and the distances.
struct HostField
{
    std::vector<char>  bytes;
    std::vector<float> sdf;
    const cloth::GridT* grid() const { return reinterpret_cast<const cloth::GridT*>(bytes.data()); }
    HostField(const cloth::SdfField& field) : bytes(field.handle.buffer().size()), sdf(field.valueCount)
    {
        cudaCheck(cudaMemcpy(bytes.data(), field.handle.buffer().data(), bytes.size(), cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(sdf.data(), field.sdf.data(), sdf.size() * sizeof(float), cudaMemcpyDeviceToHost));
    }
    // f(coordinate, value) for every active voxel.
    template <typename F>
    void forEachVoxel(F f) const
    {
        const auto& tree = grid()->tree();
        for (uint32_t l = 0; l < tree.nodeCount(0); ++l) {
            const auto& leaf = tree.getFirstLeaf()[l];
            for (uint32_t n = 0; n < 512; ++n)
                if (leaf.isActive(n)) f(leaf.offsetToGlobalCoord(n), sdf[leaf.getValue(n)]);
        }
    }
    // An OpenVDB grid with the same active values; the sign of its inactive values comes from signedFloodFill.
    openvdb::FloatGrid::Ptr toOpenVDB() const
    {
        auto out = openvdb::FloatGrid::create(sdf[0]);
        auto acc = out->getAccessor();
        forEachVoxel([&](const nanovdb::Coord& c, float value) { acc.setValue(openvdb::Coord(c[0], c[1], c[2]), value); });
        openvdb::tools::signedFloodFill(out->tree());
        return out;
    }
};

// SdfView::sample() at every point, on the device.
static void sample(const cloth::SdfView& view, const std::vector<Vec3f>& points, std::vector<float>& phi, std::vector<Vec3f>& gradient)
{
    const size_t       count = points.size();
    auto               dPoints = upload(points);
    DeviceArray<float> dPhi(cudaStream_t(0), count, nanovdb::cuda::noInit);
    DeviceArray<Vec3f> dGradient(cudaStream_t(0), count, nanovdb::cuda::noInit);
    auto kernel = [=] __device__(size_t i, const Vec3f* x, float* phi, Vec3f* g) { phi[i] = view.sample(x[i], g[i]); };
    nanovdb::util::cuda::lambdaKernel<<<nanovdb::util::cuda::blocksPerGrid(count, 256), 256>>>(count, kernel, dPoints.data(), dPhi.data(), dGradient.data());
    cudaCheck(cudaGetLastError());
    phi.resize(count), gradient.resize(count);
    cudaCheck(cudaMemcpy(phi.data(), dPhi.data(), count * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(gradient.data(), dGradient.data(), count * sizeof(Vec3f), cudaMemcpyDeviceToHost));
}

// SdfView::sample() against openvdb::tools::BoxSampler on the same values: the largest difference of the value, and
// of the gradient against a central difference of BoxSampler inside the point's cell (both in voxels).
static void compareWithBoxSampler(const HostField& host, float voxelSize, const std::vector<Vec3f>& points,
                                  const std::vector<float>& phi, const std::vector<Vec3f>& gradient)
{
    const auto   grid = host.toOpenVDB();
    const auto   acc  = grid->getConstAccessor();
    const double step = 1.0e-2; // voxels
    double       worstValue = 0, worstGradient = 0;
    size_t       gradientCount = 0;
    for (size_t i = 0; i < points.size(); ++i) {
        const openvdb::Vec3R p(points[i][0] / voxelSize, points[i][1] / voxelSize, points[i][2] / voxelSize);
        worstValue = std::max(worstValue, std::fabs(double(openvdb::tools::BoxSampler::sample(acc, p)) - phi[i]) / voxelSize);
        bool insideCell = true; // the central difference must not leave the cell: the interpolant has kinks there
        for (int a = 0; a < 3; ++a) insideCell &= p[a] - std::floor(p[a]) > 2 * step && p[a] - std::floor(p[a]) < 1 - 2 * step;
        if (!insideCell) continue;
        ++gradientCount;
        for (int a = 0; a < 3; ++a) {
            openvdb::Vec3R plus = p, minus = p;
            plus[a] += step, minus[a] -= step;
            const double difference = (double(openvdb::tools::BoxSampler::sample(acc, plus)) - double(openvdb::tools::BoxSampler::sample(acc, minus))) / (2 * step * voxelSize);
            worstGradient = std::max(worstGradient, std::fabs(difference - gradient[i][a]));
        }
    }
    std::printf("  against OpenVDB BoxSampler on the same values, %zu points: value differs by at most %.2e voxels; "
                "gradient differs from BoxSampler's central difference by at most %.2e (%zu points)\n",
                points.size(), worstValue, worstGradient, gradientCount);
}

// A sphere of triangles: an icosahedron subdivided `levels` times, vertices on the sphere.
static void makeSphere(Vec3f center, float radius, int levels, std::vector<Vec3f>& points, std::vector<Vec3i>& triangles)
{
    const float t = (1.0f + std::sqrt(5.0f)) / 2.0f;
    std::vector<nanovdb::Vec3d> unit = {{-1, t, 0}, {1, t, 0}, {-1, -t, 0}, {1, -t, 0}, {0, -1, t}, {0, 1, t},
                                        {0, -1, -t}, {0, 1, -t}, {t, 0, -1}, {t, 0, 1}, {-t, 0, -1}, {-t, 0, 1}};
    for (auto& p : unit) p.normalize();
    triangles = {{0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11}, {1, 5, 9}, {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
                 {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8}, {3, 8, 9}, {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}};
    for (int level = 0; level < levels; ++level) {
        std::map<std::pair<int, int>, int> middle;
        auto midpoint = [&](int a, int b) {
            const auto key = std::minmax(a, b);
            const auto it  = middle.find(key);
            if (it != middle.end()) return it->second;
            nanovdb::Vec3d p = unit[a] + unit[b];
            p.normalize();
            unit.push_back(p);
            return middle[key] = int(unit.size()) - 1;
        };
        std::vector<Vec3i> next;
        for (const auto& tri : triangles) {
            const int a = midpoint(tri[0], tri[1]), b = midpoint(tri[1], tri[2]), c = midpoint(tri[2], tri[0]);
            next.insert(next.end(), {Vec3i(tri[0], a, c), Vec3i(tri[1], b, a), Vec3i(tri[2], c, b), Vec3i(a, b, c)});
        }
        triangles.swap(next);
    }
    points.clear();
    for (const auto& p : unit) points.emplace_back(center[0] + radius * float(p[0]), center[1] + radius * float(p[1]), center[2] + radius * float(p[2]));
}

static int sphereCheck(float band)
{
    std::printf("sphere: band %.1f voxels, world radius 0.5; errors against the exact distance |x - c| - R\n", band);
    std::printf("  R/voxel  active voxels | stored: max error of |value|; voxels with the wrong sign (farthest from the surface) | sampled value: mean, max error | gradient: mean, max angle, max | |g| - 1 | | wrong signs far away\n");
    for (int resolution : {30, 60, 120}) {
        const float radius = 0.5f, h = radius / float(resolution);
        const Vec3f center(0.7137f, 0.6891f, 0.7313f); // not aligned with the voxels
        std::vector<Vec3f> points;
        std::vector<Vec3i> triangles;
        makeSphere(center, radius, 7, points, triangles);
        auto dPoints = upload(points);
        auto dTriangles = upload(triangles);
        const cloth::SdfField field(dPoints.data(), uint32_t(points.size()), dTriangles.data(), uint32_t(triangles.size()), h, band, false);
        const HostField       host(field);

        // Stored values: the error of the magnitude, and the voxels whose sign is not the exact one.
        double worstStored = 0, farthestFlip = 0;
        size_t flips = 0;
        host.forEachVoxel([&](const nanovdb::Coord& c, float value) {
            const double exact = (nanovdb::Vec3d(c[0] * double(h), c[1] * double(h), c[2] * double(h)) - nanovdb::Vec3d(center)).length() - radius;
            worstStored = std::max(worstStored, std::fabs(std::fabs(value) - std::fabs(exact)) / h);
            if ((value < 0) != (exact < 0)) ++flips, farthestFlip = std::max(farthestFlip, std::fabs(exact) / h);
        });

        // Random points within (band - 1.75) voxels of the sphere: the 8 voxels around such a point are within
        // sqrt(3) voxels of it, so all of them are in the band (a voxel outside the band holds +-band, not its distance).
        std::mt19937                          random(12345);
        std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
        auto direction = [&] {
            for (;;) {
                const Vec3f d(uniform(random), uniform(random), uniform(random));
                if (d.lengthSqr() > 0.01f && d.lengthSqr() < 1.0f) return d / d.length();
            }
        };
        std::vector<Vec3f> near(200000), far(200000);
        for (auto& x : near) x = center + direction() * (radius + uniform(random) * (band - 1.75f) * h);
        for (size_t i = 0; i < far.size(); ++i) { // beyond the band, inside (even i) or outside (odd i)
            const float beyond = (band + 2.0f + 20.0f * std::fabs(uniform(random))) * h;
            far[i] = center + direction() * (i % 2 == 0 ? std::max(0.0f, radius - beyond) : radius + beyond);
        }
        std::vector<float> phi, farPhi;
        std::vector<Vec3f> gradient, farGradient;
        sample(field.view(), near, phi, gradient);
        sample(field.view(), far, farPhi, farGradient);

        double valueSum = 0, valueWorst = 0, angleSum = 0, angleWorst = 0, lengthWorst = 0;
        for (size_t i = 0; i < near.size(); ++i) {
            const nanovdb::Vec3d r = nanovdb::Vec3d(near[i]) - nanovdb::Vec3d(center);
            const double error = std::fabs(phi[i] - (r.length() - radius)) / h;
            const nanovdb::Vec3d g(gradient[i]);
            const double angle = std::acos(std::min(1.0, g.dot(r) / (g.length() * r.length()))) * 180.0 / M_PI;
            valueSum += error, valueWorst = std::max(valueWorst, error);
            angleSum += angle, angleWorst = std::max(angleWorst, angle);
            lengthWorst = std::max(lengthWorst, std::fabs(g.length() - 1.0));
        }
        size_t wrongSigns = 0;
        for (size_t i = 0; i < far.size(); ++i) wrongSigns += farPhi[i] != (i % 2 == 0 ? -host.sdf[0] : host.sdf[0]);
        std::printf("  %6d  %12zu | %.5f voxels; %zu (%.4f voxels) | %.4f, %.4f voxels | %.3f, %.3f degrees, %.4f | %zu of %zu\n", resolution,
                    host.sdf.size() - 1, worstStored, flips, farthestFlip, valueSum / near.size(), valueWorst, angleSum / near.size(),
                    angleWorst, lengthWorst, wrongSigns, far.size());
        near.insert(near.end(), far.begin(), far.end());
        phi.insert(phi.end(), farPhi.begin(), farPhi.end());
        gradient.insert(gradient.end(), farGradient.begin(), farGradient.end());
        compareWithBoxSampler(host, h, near, phi, gradient);
    }
    return 0;
}

static int characterCheck(const std::string& directory, int frame, float h, float band)
{
    int   frameCount = 0, vertexCount = 0, triangleCount = 0;
    float fps = 0;
    std::ifstream(directory + "/sequence.txt") >> frameCount >> vertexCount >> triangleCount >> fps;
    if (frame < 0 || frame >= frameCount) { std::printf("frame %d is not in 0 .. %d\n", frame, frameCount - 1); return 1; }
    std::vector<Vec3f> points(vertexCount);
    std::vector<Vec3i> triangles(triangleCount);
    std::ifstream vertices(directory + "/vertices.bin", std::ios::binary), indices(directory + "/triangles.bin", std::ios::binary);
    vertices.seekg(std::streamoff(frame) * vertexCount * sizeof(Vec3f));
    vertices.read(reinterpret_cast<char*>(points.data()), points.size() * sizeof(Vec3f));
    indices.read(reinterpret_cast<char*>(triangles.data()), triangles.size() * sizeof(Vec3i));
    if (!vertices || !indices) { std::printf("cannot read the character in %s\n", directory.c_str()); return 1; }
    std::printf("character %s frame %d: %d vertices, %d triangles; voxel %.5f, band %.1f voxels\n", directory.c_str(), frame,
                vertexCount, triangleCount, h, band);

    auto dPoints = upload(points);
    auto dTriangles = upload(triangles);
    auto build = [&] { return cloth::SdfField(dPoints.data(), uint32_t(points.size()), dTriangles.data(), uint32_t(triangles.size()), h, band, true); };
    { const auto warmUp = build(); }
    double total = 0;
    const int runs = 10;
    for (int run = 0; run < runs; ++run) {
        const auto start = std::chrono::steady_clock::now();
        const auto field = build();
        total += milliseconds(start);
    }
    const cloth::SdfField field = build();
    const HostField       host(field);
    std::printf("  MeshToSDF + PropagateSign: %.1f ms per call (mean of %d), %zu active voxels\n", total / runs, runs, host.sdf.size() - 1);

    // OpenVDB on the same mesh, same voxel size and band.
    std::vector<openvdb::Vec3s> vdbPoints;
    std::vector<openvdb::Vec3I> vdbTriangles;
    for (const auto& p : points) vdbPoints.emplace_back(p[0], p[1], p[2]);
    for (const auto& t : triangles) vdbTriangles.emplace_back(t[0], t[1], t[2]);
    const auto transform = openvdb::math::Transform::createLinearTransform(h);
    const auto start     = std::chrono::steady_clock::now();
    const auto reference = openvdb::tools::meshToLevelSet<openvdb::FloatGrid>(*transform, vdbPoints, vdbTriangles, std::vector<openvdb::Vec4I>(), band);
    std::printf("  openvdb::tools::meshToLevelSet: %.1f ms (CPU), %zu active voxels\n",
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count(), size_t(reference->activeVoxelCount()));

    // Voxel by voxel, by the two signs: 0 = both outside, 1 = both inside, 2 = the signs differ.
    const auto  acc = reference->getConstAccessor();
    const char* className[3] = {"both outside", "both inside ", "signs differ"};
    size_t      onlyHere[2] = {0, 0}, count[3] = {0, 0, 0}, histogram[3][5] = {};
    double      worst[3] = {0, 0, 0};
    struct Voxel { nanovdb::Coord c; float here, other; };
    std::vector<Voxel> common;
    host.forEachVoxel([&](const nanovdb::Coord& c, float value) {
        float other;
        if (!acc.probeValue(openvdb::Coord(c[0], c[1], c[2]), other)) { ++onlyHere[value < 0]; return; }
        common.push_back({c, value, other});
        const int    k = (value < 0) != (other < 0) ? 2 : value < 0;
        const double difference = std::fabs(double(value) - double(other)) / h;
        ++count[k], worst[k] = std::max(worst[k], difference);
        ++histogram[k][difference < 1e-4 ? 0 : difference < 1e-2 ? 1 : difference < 0.1 ? 2 : difference < 1 ? 3 : 4];
    });
    std::printf("  voxels active in both: %zu; only in MeshToSDF: %zu outside, %zu inside; only in OpenVDB: %zu\n", common.size(),
                onlyHere[0], onlyHere[1], size_t(reference->activeVoxelCount()) - common.size());
    std::printf("  |MeshToSDF - OpenVDB| in voxels:   voxels |  < 1e-4 |  < 1e-2 |   < 0.1 |     < 1 |    >= 1 | largest\n");
    for (int k = 0; k < 3; ++k)
        std::printf("    %s              %8zu | %7zu | %7zu | %7zu | %7zu | %7zu | %.4f\n", className[k], count[k], histogram[k][0],
                    histogram[k][1], histogram[k][2], histogram[k][3], histogram[k][4], worst[k]);

    // The winding number of the mesh around a voxel (the sum of the solid angles of all triangles over 4 pi; for
    // closed surfaces, the number of surfaces that enclose the voxel) decides the sign under the Solid rule: inside
    // if it is 1 or more. Computed for every voxel whose two signs differ, and for 20000 other voxels.
    {
        std::vector<Voxel> tested;
        for (const auto& v : common) if ((v.here < 0) != (v.other < 0)) tested.push_back(v);
        const size_t differing = tested.size();
        std::vector<Voxel> rest;
        for (const auto& v : common) if ((v.here < 0) == (v.other < 0)) rest.push_back(v);
        std::shuffle(rest.begin(), rest.end(), std::mt19937(4242));
        tested.insert(tested.end(), rest.begin(), rest.begin() + std::min<size_t>(rest.size(), 20000));
        std::vector<double> winding(tested.size());
        tbb::parallel_for(size_t(0), tested.size(), [&](size_t i) {
            const openvdb::Vec3d p(tested[i].c[0] * double(h), tested[i].c[1] * double(h), tested[i].c[2] * double(h));
            double angle = 0;
            for (const auto& t : vdbTriangles) {
                const openvdb::Vec3d a = openvdb::Vec3d(vdbPoints[t[0]]) - p, b = openvdb::Vec3d(vdbPoints[t[1]]) - p, c = openvdb::Vec3d(vdbPoints[t[2]]) - p;
                const double la = a.length(), lb = b.length(), lc = c.length();
                angle += 2.0 * std::atan2(a.dot(b.cross(c)), la * lb * lc + a.dot(b) * lc + a.dot(c) * lb + b.dot(c) * la);
            }
            winding[i] = angle / (4.0 * M_PI);
        });
        // Wrong signs of each tool, by the distance of the voxel from the nearest triangle (MeshToSDF's |value|).
        size_t wrong[2][2][4] = {}, notInteger = 0, negative = 0, levels[4] = {};
        double farthest[2][2] = {};
        for (size_t i = 0; i < tested.size(); ++i) {
            notInteger += std::fabs(winding[i] - std::round(winding[i])) > 0.05;
            negative += winding[i] < -0.5;
            ++levels[std::min(3, int(std::round(std::fabs(winding[i]))))];
            const bool   inside = std::fabs(winding[i]) >= 0.5;
            const double distance = std::fabs(tested[i].here) / h;
            const int    group = i < differing ? 0 : 1, bin = distance < 0.5 ? 0 : distance < 0.87 ? 1 : distance < 1.5 ? 2 : 3;
            if (inside != (tested[i].here < 0)) ++wrong[group][0][bin], farthest[group][0] = std::max(farthest[group][0], distance);
            if (inside != (tested[i].other < 0)) ++wrong[group][1][bin], farthest[group][1] = std::max(farthest[group][1], distance);
        }
        std::printf("  winding number at %zu voxels: not within 0.05 of an integer at %zu, negative at %zu; |winding| = 0, 1, 2, 3+ at %zu, %zu, %zu, %zu\n",
                    tested.size(), notInteger, negative, levels[0], levels[1], levels[2], levels[3]);
        std::printf("  wrong signs against the winding number, by distance from the nearest triangle:  < 0.5 | < 0.87 | < 1.5 | >= 1.5 voxels | farthest\n");
        const char* group[2] = {"voxels whose two signs differ", "20000 voxels whose signs agree"};
        const char* tool[2]  = {"MeshToSDF", "OpenVDB  "};
        for (int g = 0; g < 2; ++g)
            for (int t = 0; t < 2; ++t)
                std::printf("    %s (%zu), %s: %6zu | %6zu | %6zu | %6zu | %.3f\n", group[g], g == 0 ? differing : tested.size() - differing,
                            tool[t], wrong[g][t][0], wrong[g][t][1], wrong[g][t][2], wrong[g][t][3], farthest[g][t]);
    }

    // The exact distance to the nearest triangle (all triangles tried, in double) at 3000 of those voxels.
    std::shuffle(common.begin(), common.end(), std::mt19937(777));
    common.resize(std::min<size_t>(common.size(), 3000));
    double worstHere[3] = {0, 0, 0}, worstOther[3] = {0, 0, 0};
    size_t sampled[3] = {0, 0, 0};
    for (const auto& v : common) {
        const openvdb::Vec3d p(v.c[0] * double(h), v.c[1] * double(h), v.c[2] * double(h));
        double nearest = 1e300;
        for (const auto& t : vdbTriangles) {
            openvdb::Vec3d uvw;
            const auto q = openvdb::math::closestPointOnTriangleToPoint(openvdb::Vec3d(vdbPoints[t[0]]), openvdb::Vec3d(vdbPoints[t[1]]),
                                                                        openvdb::Vec3d(vdbPoints[t[2]]), p, uvw);
            nearest = std::min(nearest, (q - p).lengthSqr());
        }
        nearest = std::sqrt(nearest);
        const int k = (v.here < 0) != (v.other < 0) ? 2 : v.here < 0;
        ++sampled[k];
        worstHere[k]  = std::max(worstHere[k], std::fabs(std::fabs(double(v.here)) - nearest) / h);
        worstOther[k] = std::max(worstOther[k], std::fabs(std::fabs(double(v.other)) - nearest) / h);
    }
    std::printf("  largest error of |value| against the exact distance to the nearest triangle, in voxels (3000 voxels):\n");
    for (int k = 0; k < 3; ++k)
        std::printf("    %s (%4zu voxels): MeshToSDF %.5f, OpenVDB %.5f\n", className[k], sampled[k], worstHere[k], worstOther[k]);

    // The sampler, at random points up to (band - 1.75) voxels from random vertices.
    std::mt19937                          random(12345);
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
    std::vector<Vec3f>                    near(200000);
    for (auto& x : near) x = points[random() % points.size()] + Vec3f(uniform(random), uniform(random), uniform(random)) * ((band - 1.75f) * h / std::sqrt(3.0f));
    std::vector<float> phi;
    std::vector<Vec3f> gradient;
    sample(field.view(), near, phi, gradient);
    compareWithBoxSampler(host, h, near, phi, gradient);
    return 0;
}

int main(int argc, char** argv)
{
    openvdb::initialize();
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "sphere") return sphereCheck(argc > 2 ? float(std::atof(argv[2])) : 4.0f);
    if (mode == "character" && argc >= 5)
        return characterCheck(argv[2], std::atoi(argv[3]), float(std::atof(argv[4])), argc > 5 ? float(std::atof(argv[5])) : 4.0f);
    std::printf("usage: sdf_check sphere [band] | sdf_check character <dir> <frame> <voxel size> [band]\n");
    return 1;
}
