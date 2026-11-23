// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file verify.cu
///
/// @brief Checks MeanCurvature.cuh + tools::cuda::LevelSetTracker::track() against OpenVDB and against exact solutions.
///
/// usage: verify [radius in voxels] [voxel size] [band half-width in voxels]      (defaults 30, 1, 5)
///
///   A. The kernel alone: one update phi += dt H |grad phi| on a sphere and on the union of two spheres, against
///      openvdb::math::CurvatureStencil::meanCurvatureNormGrad() at every active voxel.
///   B. The flow of a sphere until its radius has halved: (update; track()) repeated, against the exact radius
///      sqrt(R0^2 - 2 t) and against openvdb::tools::LevelSetFilter::meanCurvature().
///   C. The flow of the union of two spheres (centers 1.2 radii apart, so with a sharp crease), which has no exact
///      solution: against OpenVDB's filter, at the voxels within one voxel of OpenVDB's surface.
///   D. The volume-preserving flow, which OpenVDB does not have: a sphere must stay as it is; the union of two
///      spheres must keep its volume (openvdb::tools::levelSetVolume) while its area falls, and tend to one sphere.
///
/// The spheres are centered 0.3 * (1, 0.7, 0.4) voxels off a voxel. A sphere centered exactly on a voxel leaves that
/// voxel behind when it vanishes: its value tends to zero from below and, by symmetry, has no gradient to move it.
/// Part B with a band half-width of 3 shows the error that MeanCurvature.cuh's note describes.

#include <openvdb/openvdb.h>
#include <openvdb/math/Stencils.h>
#include <openvdb/tools/Composite.h>
#include <openvdb/tools/LevelSetFilter.h>
#include <openvdb/tools/LevelSetMeasure.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/SignedFloodFill.h>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/GridBuilder.h>
#include <nanovdb/tools/cuda/LevelSetTracker.cuh>
#include "MeanCurvature.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using BuildT    = nanovdb::ValueOnIndex;
using ResourceT = nanovdb::cuda::DeviceResource;
using TrackerT  = nanovdb::tools::cuda::LevelSetTracker<BuildT, float, ResourceT>;
using StateT    = TrackerT::State;
template <typename T>
using BufferT = TrackerT::BufferT<T>;
using HandleT = TrackerT::HandleT;

static const double kCenter[3] = {0.3, 0.21, 0.12}; // of the spheres, in voxels
static const double kPi = 3.14159265358979323846;

static nanovdb::cuda::ResourceRef<ResourceT> ref()
{
    return nanovdb::cuda::ResourceRef<ResourceT>(nanovdb::cuda::default_resource<ResourceT>());
}

struct HostLevelSet
{
    std::vector<std::byte> gridBytes;
    std::vector<float>     phi;
    const nanovdb::NanoGrid<BuildT>* grid() const { return reinterpret_cast<const nanovdb::NanoGrid<BuildT>*>(gridBytes.data()); }
};

static HostLevelSet download(const HandleT& grid, const BufferT<float>& phi)
{
    HostLevelSet h;
    h.gridBytes.resize(grid.buffer().size());
    cudaCheck(cudaMemcpy(h.gridBytes.data(), grid.buffer().data(), h.gridBytes.size(), cudaMemcpyDeviceToHost));
    h.phi.resize(phi.size());
    cudaCheck(cudaMemcpy(h.phi.data(), phi.data(), phi.size_bytes(), cudaMemcpyDeviceToHost));
    return h;
}

// f(coordinate, value) for every active voxel.
template <typename F>
static void forEachActive(const HostLevelSet& h, F&& f)
{
    const auto& tree = h.grid()->tree();
    for (uint32_t l = 0; l < tree.nodeCount(0); ++l) {
        const auto& leaf = tree.template getFirstNode<0>()[l];
        for (uint32_t n = 0; n < 512; ++n)
            if (leaf.isActive(n)) f(leaf.offsetToGlobalCoord(n), h.phi[leaf.getValue(n)]);
    }
}

// The band of an OpenVDB level set as a device level set with its invert masks.
static StateT fromOpenVDB(const openvdb::FloatGrid& grid, float dx)
{
    // CreateNanoGrid cannot read OpenVDB grids under nvcc, so the active voxels go through NanoVDB's host builder.
    nanovdb::tools::build::FloatGrid builder(grid.background());
    builder.setTransform(dx);
    {
        auto builderAcc = builder.getAccessor();
        for (auto it = grid.cbeginValueOn(); it; ++it) {
            const openvdb::Coord c = it.getCoord();
            builderAcc.setValue(nanovdb::Coord(c.x(), c.y(), c.z()), *it);
        }
    }
    auto handle = nanovdb::tools::createNanoGrid<nanovdb::tools::build::FloatGrid, BuildT>(builder, 0u, false, false);
    const auto* hostGrid = handle.grid<BuildT>();
    std::vector<float> hostPhi(hostGrid->tree().activeVoxelCount() + 1, 0.f);
    const auto acc = grid.getConstAccessor();
    for (uint32_t l = 0; l < hostGrid->tree().nodeCount(0); ++l) {
        const auto& leaf = hostGrid->tree().template getFirstNode<0>()[l];
        for (uint32_t n = 0; n < 512; ++n)
            if (leaf.isActive(n)) {
                const auto c = leaf.offsetToGlobalCoord(n);
                hostPhi[leaf.getValue(n)] = acc.getValue(openvdb::Coord(c[0], c[1], c[2]));
            }
    }
    BufferT<std::byte> gridBuffer(cudaStream_t(0), ref(), handle.buffer().size(), nanovdb::cuda::noInit);
    cudaCheck(cudaMemcpy(gridBuffer.data(), handle.buffer().data(), handle.buffer().size(), cudaMemcpyHostToDevice));
    BufferT<float> phi(cudaStream_t(0), ref(), hostPhi.size(), nanovdb::cuda::noInit);
    cudaCheck(cudaMemcpy(phi.data(), hostPhi.data(), phi.size_bytes(), cudaMemcpyHostToDevice));
    return TrackerT(HandleT(std::move(gridBuffer)), std::move(phi), grid.background()).release();
}

// The device level set as an OpenVDB level set, for OpenVDB's measures.
static openvdb::FloatGrid::Ptr toOpenVDB(const StateT& state, float background, float dx)
{
    auto grid = openvdb::FloatGrid::create(background);
    grid->setTransform(openvdb::math::Transform::createLinearTransform(dx));
    grid->setGridClass(openvdb::GRID_LEVEL_SET);
    auto acc = grid->getAccessor();
    forEachActive(download(state.grid, state.phi),
                  [&](const nanovdb::Coord& c, float phi) { acc.setValue(openvdb::Coord(c[0], c[1], c[2]), phi); });
    openvdb::tools::signedFloodFill(grid->tree());
    return grid;
}

// One step on the device: the update of MeanCurvature.cuh with dt = dx^2 / 3, then track().
static void advance(StateT& state, float background, float dx, bool preserveVolume, double* averageCurvature = nullptr)
{
    BufferT<float> moved = curvature::meanCurvatureStep<TrackerT>(state, background, dx, 1.f / 3.f, preserveVolume,
                                                                 cudaStream_t(0), averageCurvature);
    TrackerT tracker(std::move(state.grid), std::move(moved), background);
    tracker.track();
    state = tracker.release();
}

// Distance of a surface from a point, in voxels, read off the level set: |x - center| - phi at the voxels within one
// voxel of the surface (exact where |grad phi| = 1).
struct Radius
{
    double   sum = 0, lo = 1e30, hi = -1e30;
    uint64_t count = 0;
    void add(double i, double j, double k, const double* center, double phiInVoxels)
    {
        if (std::fabs(phiInVoxels) >= 1.0) return;
        const double r = std::sqrt(std::pow(i - center[0], 2) + std::pow(j - center[1], 2) + std::pow(k - center[2], 2)) - phiInVoxels;
        sum += r, lo = std::min(lo, r), hi = std::max(hi, r), ++count;
    }
    double mean() const { return count ? sum / double(count) : 0.0; }
};

static Radius radiusOf(const StateT& state, double dx, const double* center)
{
    Radius r;
    forEachActive(download(state.grid, state.phi),
                  [&](const nanovdb::Coord& c, float phi) { r.add(c[0], c[1], c[2], center, phi / dx); });
    return r;
}

static Radius radiusOf(const openvdb::FloatGrid& grid, double dx, const double* center)
{
    Radius r;
    for (auto it = grid.cbeginValueOn(); it; ++it) {
        const openvdb::Coord c = it.getCoord();
        r.add(c.x(), c.y(), c.z(), center, *it / dx);
    }
    return r;
}

// A. One update of the kernel against OpenVDB's stencil, at every active voxel.
static void checkKernel(const char* label, const openvdb::FloatGrid& vdb, float dx)
{
    const StateT state = fromOpenVDB(vdb, dx);
    const BufferT<float> moved = curvature::meanCurvatureStep<TrackerT>(state, vdb.background(), dx, 1.f / 3.f, false);
    const HostLevelSet after = download(state.grid, moved);

    const float dt = dx * dx / 3.f; // as in LevelSetFilter::Filter::meanCurvatureImpl
    openvdb::math::CurvatureStencil<openvdb::FloatGrid> stencil(vdb, dx);
    uint64_t count = 0;
    double   maxDiff = 0, maxMotion = 0;
    forEachActive(after, [&](const nanovdb::Coord& c, float phi) {
        stencil.moveTo(openvdb::Coord(c[0], c[1], c[2]));
        const float expected = stencil.getValue() + dt * stencil.meanCurvatureNormGrad();
        maxDiff   = std::max(maxDiff, std::fabs(double(phi) - expected) / dx);
        maxMotion = std::max(maxMotion, std::fabs(double(expected) - stencil.getValue()) / dx);
        ++count;
    });
    std::printf("A. %-12s %8llu active voxels (OpenVDB %llu): largest difference to OpenVDB %.3e voxel; the update "
                "moves a value by up to %.4f voxel\n", label, (unsigned long long)count,
                (unsigned long long)vdb.activeVoxelCount(), maxDiff, maxMotion);
}

int main(int argc, char** argv)
{
    const double radius    = argc > 1 ? std::atof(argv[1]) : 30.0; // in voxels
    const float  dx        = argc > 2 ? float(std::atof(argv[2])) : 1.f;
    const float  halfWidth = argc > 3 ? float(std::atof(argv[3])) : 5.f;
    openvdb::initialize();
    auto makeSphere = [&](double shift) { // centered at kCenter + (shift, 0, 0) voxels
        return openvdb::tools::createLevelSetSphere<openvdb::FloatGrid>(
            float(radius * dx), openvdb::Vec3f(float((kCenter[0] + shift) * dx), float(kCenter[1] * dx), float(kCenter[2] * dx)),
            dx, halfWidth);
    };
    auto makeTwoSpheres = [&] {
        auto a = makeSphere(-0.6 * radius), b = makeSphere(0.6 * radius);
        openvdb::tools::csgUnion(*a, *b);
        return a;
    };
    const float background = makeSphere(0)->background();
    std::printf("radius %g voxels, voxel size %g, band half-width %g voxels, time step dx^2 / 3\n", radius, dx, halfWidth);

    checkKernel("sphere", *makeSphere(0), dx);
    checkKernel("two spheres", *makeTwoSpheres(), dx);

    { // B. the flow of a sphere
        auto vdb = makeSphere(0);
        openvdb::tools::LevelSetFilter<openvdb::FloatGrid> filter(*vdb);
        StateT state = fromOpenVDB(*vdb, dx);
        const int steps = int(std::ceil(1.125 * radius * radius)); // the radius halves at t = 0.375 R0^2
        std::printf("B. sphere; radii in voxels\n%6s %10s | %10s %9s %9s %9s | %10s %9s\n", "step", "exact", "GPU", "error",
                    "min-mean", "max-mean", "OpenVDB", "error");
        for (int step = 1; step <= steps; ++step) {
            advance(state, background, dx, false);
            filter.meanCurvature();
            if (step % ((steps + 3) / 4) && step != steps) continue;
            const double exact = std::sqrt(radius * radius - 2.0 * step / 3.0);
            const Radius g = radiusOf(state, dx, kCenter), v = radiusOf(*vdb, dx, kCenter);
            std::printf("%6d %10.4f | %10.4f %+9.4f %+9.4f %+9.4f | %10.4f %+9.4f\n", step, exact, g.mean(),
                        g.mean() - exact, g.lo - g.mean(), g.hi - g.mean(), v.mean(), v.mean() - exact);
        }
    }

    { // C. the flow of two merged spheres
        auto vdb = makeTwoSpheres();
        openvdb::tools::LevelSetFilter<openvdb::FloatGrid> filter(*vdb);
        StateT state = fromOpenVDB(*vdb, dx);
        const int steps = int(std::lround(1.5 * radius * radius));
        std::printf("C. two spheres; difference of the GPU to OpenVDB within one voxel of OpenVDB's surface, in voxels\n"
                    "%6s %10s %10s %10s %10s\n", "step", "largest", "rms", "GPU voxels", "OpenVDB");
        for (int step = 1; step <= steps; ++step) {
            advance(state, background, dx, false);
            filter.meanCurvature();
            if (step % ((steps + 2) / 3) && step != steps) continue;
            const auto acc = vdb->getConstAccessor();
            double   maxDiff = 0, sumSq = 0;
            uint64_t count = 0, active = 0;
            forEachActive(download(state.grid, state.phi), [&](const nanovdb::Coord& c, float phi) {
                ++active;
                const openvdb::Coord ijk(c[0], c[1], c[2]);
                const float reference = acc.getValue(ijk);
                if (!acc.isValueOn(ijk) || std::fabs(reference) >= dx) return;
                const double d = std::fabs(double(phi) - reference) / dx;
                maxDiff = std::max(maxDiff, d), sumSq += d * d, ++count;
            });
            std::printf("%6d %10.4f %10.4f %10llu %10llu\n", step, maxDiff, std::sqrt(sumSq / double(std::max<uint64_t>(count, 1))),
                        (unsigned long long)active, (unsigned long long)vdb->activeVoxelCount());
        }
    }

    { // D. volume-preserving flow of a sphere: nothing moves
        StateT state = fromOpenVDB(*makeSphere(0), dx);
        double lambda = 0;
        for (int step = 0; step < 1000; ++step) advance(state, background, dx, true, &lambda);
        const Radius g = radiusOf(state, dx, kCenter);
        std::printf("D. sphere, 1000 volume-preserving steps: radius %.5f voxels (min %+.4f, max %+.4f), average "
                    "curvature %.6f / voxel (1 / radius = %.6f)\n", g.mean(), g.lo - g.mean(), g.hi - g.mean(), lambda,
                    1.0 / g.mean());
    }

    { // D. volume-preserving flow of two merged spheres
        StateT state = fromOpenVDB(*makeTwoSpheres(), dx);
        const int    steps = int(std::lround(6.0 * radius * radius));
        const double voxel2 = double(dx) * dx, voxel3 = voxel2 * dx;
        double       volume0 = 0, lambda = 0;
        std::printf("D. two spheres, volume-preserving; volume and area by OpenVDB, in voxels; the distance of the surface "
                    "from its centroid\n%6s %12s %9s %12s %11s | %9s %9s %9s\n", "step", "volume", "vs start", "area",
                    "avg curv", "mean", "min", "max");
        for (int step = 0; step <= steps; ++step) {
            if (step) advance(state, background, dx, true, &lambda);
            if (step % ((steps + 5) / 6) && step != steps) continue;
            const auto   grid = toOpenVDB(state, background, dx);
            const double volume = openvdb::tools::levelSetVolume(*grid) / voxel3, area = openvdb::tools::levelSetArea(*grid) / voxel2;
            if (!step) volume0 = volume;
            double   centroid[3] = {0, 0, 0};
            uint64_t count = 0;
            forEachActive(download(state.grid, state.phi), [&](const nanovdb::Coord& c, float phi) {
                if (std::fabs(phi) < dx) centroid[0] += c[0], centroid[1] += c[1], centroid[2] += c[2], ++count;
            });
            for (double& x : centroid) x /= double(std::max<uint64_t>(count, 1));
            const Radius r = radiusOf(state, dx, centroid);
            std::printf("%6d %12.1f %+8.4f%% %12.1f %11.6f | %9.4f %9.4f %9.4f\n", step, volume,
                        100.0 * (volume / volume0 - 1.0), area, lambda, r.mean(), r.lo, r.hi);
        }
        const double sphere = std::cbrt(3.0 * volume0 / (4.0 * kPi));
        std::printf("D. the sphere of the starting volume: radius %.4f voxels, area %.1f\n", sphere, 4.0 * kPi * sphere * sphere);
    }
    return 0;
}
