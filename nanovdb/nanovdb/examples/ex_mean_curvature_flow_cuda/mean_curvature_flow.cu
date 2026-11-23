// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file mean_curvature_flow.cu
///
/// @brief Motion by mean curvature of a level set made by tools::cuda::MeshToSDF. Every step is the update of
///        MeanCurvature.cuh followed by tools::cuda::LevelSetTracker::track().
///
/// usage: mean_curvature_flow <in.sdf> <out dir> <steps> <frames> <preserve volume: 0 | 1> [schedule exponent] [dt / dx^2]
///
/// <in.sdf> is a file of ../ex_level_set_offset_cuda/mesh_to_sdf_file.cu, made with a band of 5 voxels (see
/// MeanCurvature.cuh for why not 3). With <preserve volume> = 1 the flow keeps the enclosed volume and ends in a
/// sphere; with 0 the surface shrinks until it is gone. Frame k of <frames> is written after
/// round(steps * (k / frames)^exponent) steps: exponent 1 (the default) spaces the frames evenly, a larger one puts
/// more of them at the start, where the small features go. The time step is [dt / dx^2] * dx^2 (default 1/3, the
/// one of openvdb::tools::LevelSetFilter::meanCurvature()).
///
/// Output: <out dir>/f_NNN.band, the format ../ex_fluid_city_flood_cuda/mesh_band.cpp reads: int32 count, 0, 0, 0;
/// then per voxel within 2 voxels of the surface int16 i, j, k and float32 phi in voxels. One line per frame is
/// printed: the step, the time in dx^2, the active voxels, the area and the enclosed volume in voxels (from the band
/// alone, see measure()), the volume against the first frame, the average curvature of the last step (0 without
/// volume preservation) and the time per step since the previous frame.

#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/cuda/LevelSetTracker.cuh>
#include "MeanCurvature.cuh"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using BuildT    = nanovdb::ValueOnIndex;
using ResourceT = nanovdb::cuda::DeviceResource;
using TrackerT  = nanovdb::tools::cuda::LevelSetTracker<BuildT, float, ResourceT>;
using StateT    = TrackerT::State;
template <typename T>
using BufferT = TrackerT::BufferT<T>;
using HandleT = TrackerT::HandleT;

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

static HostLevelSet download(const StateT& state)
{
    HostLevelSet h;
    h.gridBytes.resize(state.grid.buffer().size());
    cudaCheck(cudaMemcpy(h.gridBytes.data(), state.grid.buffer().data(), h.gridBytes.size(), cudaMemcpyDeviceToHost));
    h.phi.resize(state.phi.size());
    cudaCheck(cudaMemcpy(h.phi.data(), state.phi.data(), state.phi.size_bytes(), cudaMemcpyDeviceToHost));
    return h;
}

// The file of mesh_to_sdf_file.cu: uint64 grid bytes, uint64 value count, double voxel size, float background, the grid
// buffer, the float sidecar (slot 0 = background).
static HostLevelSet readSdf(const char* path, double& dx, float& background)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) { std::printf("cannot open %s\n", path); std::exit(1); }
    uint64_t gridBytes, valueCount;
    in.read(reinterpret_cast<char*>(&gridBytes), 8).read(reinterpret_cast<char*>(&valueCount), 8);
    in.read(reinterpret_cast<char*>(&dx), 8).read(reinterpret_cast<char*>(&background), 4);
    HostLevelSet h;
    h.gridBytes.resize(gridBytes);
    h.phi.resize(valueCount);
    in.read(reinterpret_cast<char*>(h.gridBytes.data()), gridBytes).read(reinterpret_cast<char*>(h.phi.data()), valueCount * 4);
    return h;
}

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

// Area and enclosed volume, in voxels, from the band alone: with the smeared delta of the surface and the central
// difference gradient g of phi / dx, area = sum of delta |g| and, by the divergence theorem,
// volume = sum of delta (x . g) / 3 (x taken from the middle of the band's bounding box).
static void measure(const HostLevelSet& h, double dx, double& area, double& volume)
{
    nanovdb::Coord lo(INT32_MAX), hi(INT32_MIN);
    forEachActive(h, [&](const nanovdb::Coord& c, float) { lo.minComponent(c), hi.maxComponent(c); });
    const double mid[3] = {0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), 0.5 * (lo[2] + hi[2])};
    const auto acc = h.grid()->getAccessor();
    area = volume = 0;
    forEachActive(h, [&](const nanovdb::Coord& c, float phi) {
        const double delta = curvature::surfaceDelta(phi / dx);
        if (delta == 0.0) return;
        double g[3];
        for (int a = 0; a < 3; ++a) {
            nanovdb::Coord plus = c, minus = c;
            plus[a] += 1, minus[a] -= 1;
            g[a] = (double(h.phi[acc.getValue(plus)]) - h.phi[acc.getValue(minus)]) / (2 * dx); // both lie in the band
        }
        area   += delta * std::sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
        volume += delta * ((c[0] - mid[0]) * g[0] + (c[1] - mid[1]) * g[1] + (c[2] - mid[2]) * g[2]) / 3.0;
    });
}

static void writeBand(const HostLevelSet& h, double dx, const std::string& path)
{
    std::vector<char> records;
    int32_t count = 0;
    forEachActive(h, [&](const nanovdb::Coord& c, float phi) {
        const float p = float(phi / dx);
        if (std::fabs(p) >= 2.f) return;
        const int16_t xyz[3] = {int16_t(c[0]), int16_t(c[1]), int16_t(c[2])};
        records.insert(records.end(), reinterpret_cast<const char*>(xyz), reinterpret_cast<const char*>(xyz) + 6);
        records.insert(records.end(), reinterpret_cast<const char*>(&p), reinterpret_cast<const char*>(&p) + 4);
        ++count;
    });
    const int32_t header[4] = {count, 0, 0, 0};
    std::FILE* out = std::fopen(path.c_str(), "wb");
    std::fwrite(header, sizeof(int32_t), 4, out);
    std::fwrite(records.data(), 1, records.size(), out);
    std::fclose(out);
}

int main(int argc, char** argv)
{
    if (argc < 6) { std::printf("usage: mean_curvature_flow <in.sdf> <out dir> <steps> <frames> <preserve volume: 0 | 1> [schedule exponent] [dt / dx^2]\n"); return 1; }
    const std::string outDir = argv[2];
    const long   steps    = std::atol(argv[3]);
    const int    frames   = std::atoi(argv[4]);
    const bool   preserve = std::atoi(argv[5]) != 0;
    const double exponent = argc > 6 ? std::atof(argv[6]) : 1.0;
    const float  dtFactor = argc > 7 ? float(std::atof(argv[7])) : 1.f / 3.f;

    double dx;
    float  background;
    StateT state = [&] {
        const HostLevelSet input = readSdf(argv[1], dx, background);
        BufferT<std::byte> gridBuffer(cudaStream_t(0), ref(), input.gridBytes.size(), nanovdb::cuda::noInit);
        cudaCheck(cudaMemcpy(gridBuffer.data(), input.gridBytes.data(), input.gridBytes.size(), cudaMemcpyHostToDevice));
        BufferT<float> phi(cudaStream_t(0), ref(), input.phi.size(), nanovdb::cuda::noInit);
        cudaCheck(cudaMemcpy(phi.data(), input.phi.data(), phi.size_bytes(), cudaMemcpyHostToDevice));
        return TrackerT(HandleT(std::move(gridBuffer)), std::move(phi), background).release();
    }();
    std::printf("%s: voxel size %g, band %g voxels, %s, %ld steps of %g dx^2, %d frames, schedule exponent %g\n", argv[1],
                dx, background / dx, preserve ? "volume-preserving" : "plain", steps, dtFactor, frames, exponent);
    std::printf("%5s %8s %10s %10s %12s %14s %9s %11s %9s\n", "frame", "step", "t/dx^2", "voxels", "area", "volume",
                "vs start", "avg curv", "ms/step");

    long   step = 0;
    double volume0 = 0, lambda = 0;
    for (int frame = 0; frame <= frames; ++frame) {
        const long target = std::lround(double(steps) * std::pow(double(frame) / frames, exponent));
        const long first  = step;
        cudaCheck(cudaDeviceSynchronize());
        const auto t0 = std::chrono::steady_clock::now();
        for (; step < target && state.phi.size() > 1; ++step) {
            BufferT<float> moved = curvature::meanCurvatureStep<TrackerT>(state, background, float(dx), dtFactor, preserve,
                                                                   cudaStream_t(0), &lambda);
            TrackerT tracker(std::move(state.grid), std::move(moved), background);
            tracker.track();
            state = tracker.release();
        }
        cudaCheck(cudaDeviceSynchronize());
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

        const HostLevelSet h = download(state);
        double area = 0, volume = 0;
        if (h.phi.size() > 1) measure(h, dx, area, volume);
        if (frame == 0) volume0 = volume;
        char name[32];
        std::snprintf(name, sizeof(name), "/f_%03d.band", frame);
        writeBand(h, dx, outDir + name);
        std::printf("%5d %8ld %10.1f %10zu %12.1f %14.1f %+8.3f%% %11.6f %9.3f\n", frame, step, step * double(dtFactor),
                    h.phi.size() - 1, area, volume, volume0 > 0 ? 100.0 * (volume / volume0 - 1.0) : 0.0, lambda,
                    step > first ? ms / double(step - first) : 0.0);
        std::fflush(stdout);
    }
    return 0;
}
