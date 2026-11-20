// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file CsgExampleUtil.cuh
///
/// @brief Helpers shared by the programs of this example: narrow-band level sets of analytic spheres and of
///        meshes (read from band files, see mesh_to_band.cu), each as a LevelSetTracker::State, host copies of
///        them, and a wall-clock timer.

#ifndef NANOVDB_EXAMPLES_CSG_EXAMPLE_UTIL_CUH_HAS_BEEN_INCLUDED
#define NANOVDB_EXAMPLES_CSG_EXAMPLE_UTIL_CUH_HAS_BEEN_INCLUDED

#include <nanovdb/tools/cuda/CSG.cuh>
#include <nanovdb/tools/cuda/PointsToGrid.cuh>
#include <vector>
#include <string>
#include <functional>
#include <stdexcept>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <algorithm>

using nanovdb::Coord; using nanovdb::ValueOnIndex; using nanovdb::NanoGrid; using nanovdb::NanoLeaf; using nanovdb::Mask;
namespace tools = nanovdb::tools; namespace util = nanovdb::util;
using BuildT = ValueOnIndex;
using ResourceT = nanovdb::cuda::DeviceResource;
using TrackerT = tools::cuda::LevelSetTracker<BuildT, float, ResourceT>;
using StateT = TrackerT::State;
template<typename T> using BufT = nanovdb::cuda::Buffer<T, nanovdb::cuda::ResourceRef<ResourceT>>;

struct Sphere { float cx, cy, cz, r; };
static constexpr float W = 3.0f; // half-width and background, in voxels (voxel size 1)
__hostdev__ inline float dist(const Sphere& s, const Coord& p)
{
    const float x = p[0] - s.cx, y = p[1] - s.cy, z = p[2] - s.cz;
    return sqrtf(x * x + y * y + z * z) - s.r;
}

struct FillFunctor {
    static constexpr int MaxThreadsPerBlock = 512;
    static constexpr int MinBlocksPerMultiprocessor = 1;
    __device__ void operator()(const NanoGrid<BuildT>* grid, Sphere s, float* phi) const
    {
        const auto& leaf = grid->tree().template getFirstNode<0>()[blockIdx.x];
        if (leaf.isActive(threadIdx.x)) phi[leaf.getValue(threadIdx.x)] = dist(s, leaf.offsetToGlobalCoord(threadIdx.x));
    }
};

inline StateT makeSphere(const Sphere& s, ResourceT& res)
{
    std::vector<Coord> coords;
    const int R = int(std::ceil(s.r + W)) + 1;
    for (int i = -R; i <= R; ++i) for (int j = -R; j <= R; ++j) for (int k = -R; k <= R; ++k) {
        const Coord p(int(std::floor(s.cx)) + i, int(std::floor(s.cy)) + j, int(std::floor(s.cz)) + k);
        if (std::fabs(dist(s, p)) < W) coords.push_back(p);
    }
    nanovdb::cuda::ResourceRef<ResourceT> ref(res);
    BufT<Coord> dCoords(cudaStream_t(0), ref, coords.size(), nanovdb::cuda::noInit);
    cudaCheck(cudaMemcpy(dCoords.data(), coords.data(), coords.size() * sizeof(Coord), cudaMemcpyHostToDevice));
    auto handle = tools::cuda::voxelsToGrid<BuildT, Coord*, nanovdb::cuda::ResourceRef<ResourceT>>(dCoords.data(), coords.size(), 1.0, cudaStream_t(0), ref);
    const auto* grid = handle.template deviceGrid<BuildT>();
    const auto td = util::cuda::DeviceGridTraits<BuildT>::getTreeData(grid, 0);
    if (td.mVoxelCount != coords.size()) { printf("voxel count mismatch\n"); exit(1); }
    BufT<float> phi(cudaStream_t(0), ref, td.mVoxelCount + 1, nanovdb::cuda::noInit);
    util::cuda::operatorKernel<FillFunctor><<<td.mNodeCount[0], 512>>>(grid, s, phi.data());
    cudaCheckError();
    TrackerT tracker(std::move(handle), std::move(phi), W, 0, res);
    return tracker.release();
}

struct HostState {
    std::vector<char> grid; std::vector<float> phi; std::vector<Mask<3>> leaf; std::vector<Mask<4>> lower; std::vector<Mask<5>> upper; std::vector<Coord> root;
    const NanoGrid<BuildT>* g() const { return reinterpret_cast<const NanoGrid<BuildT>*>(grid.data()); }
};
template<typename V, typename B> inline void pull(V& v, const B& b)
{
    v.resize(b.size());
    if (b.size()) cudaCheck(cudaMemcpy((void*)v.data(), b.data(), b.size() * sizeof(v[0]), cudaMemcpyDeviceToHost));
}
inline HostState download(const StateT& s)
{
    HostState h;
    h.grid.resize(s.grid.buffer().size());
    cudaCheck(cudaMemcpy(h.grid.data(), s.grid.buffer().data(), h.grid.size(), cudaMemcpyDeviceToHost));
    pull(h.phi, s.phi); pull(h.leaf, s.leafMasks); pull(h.lower, s.lowerMasks); pull(h.upper, s.upperMasks); pull(h.root, s.rootOrigins);
    return h;
}

template<typename M> inline bool same(const std::vector<M>& a, const std::vector<M>& b)
{
    return a.size() == b.size() && (a.empty() || std::memcmp((const void*)a.data(), (const void*)b.data(), a.size() * sizeof(M)) == 0);
}

template<typename F> inline void timeIt(const char* label, F f, int reps = 41)
{
    for (int i = 0; i < 2; ++i) f();
    std::vector<double> t;
    for (int i = 0; i < reps; ++i) {
        cudaCheck(cudaDeviceSynchronize());
        const auto t0 = std::chrono::steady_clock::now();
        f();
        cudaCheck(cudaDeviceSynchronize());
        t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(t.begin(), t.end());
    printf("%-22s   time %-28s min %7.2f ms  median %7.2f ms  (%d runs)\n", "", label, t.front(), t[t.size() / 2], reps);
}

// A narrow band read from a band file: the index coordinates and values of its voxels.
struct Band { std::vector<Coord> coords; std::vector<float> values; };
inline Band loadBand(const char* path)
{
    Band b; FILE* f = fopen(path, "rb"); if (!f) { printf("cannot open %s\n", path); exit(1); }
    uint64_t n = 0; if (fread(&n, 8, 1, f) != 1) exit(1);
    b.coords.resize(n); b.values.resize(n);
    for (uint64_t i = 0; i < n; ++i) { int32_t c[3]; float v; if (fread(c, 4, 3, f) != 3 || fread(&v, 4, 1, f) != 1) exit(1); b.coords[i] = Coord(c[0], c[1], c[2]); b.values[i] = v; }
    fclose(f); return b;
}
struct FillFromListFunctor {
    __device__ void operator()(size_t i, const NanoGrid<BuildT>* grid, const Coord* coords, const float* values, float* phi) const
    {
        const auto* leaf = grid->tree().root().probeLeaf(coords[i]);
        phi[leaf->getValue(NanoLeaf<BuildT>::CoordToOffset(coords[i]))] = values[i];
    }
};
// The level set of a band moved by a whole number of voxels (so that no value is resampled).
inline StateT makeFromBand(const Band& b, const Coord& offset, ResourceT& res)
{
    std::vector<Coord> coords(b.coords.size());
    for (size_t i = 0; i < coords.size(); ++i) coords[i] = b.coords[i] + offset;
    nanovdb::cuda::ResourceRef<ResourceT> ref(res);
    BufT<Coord> dCoords(cudaStream_t(0), ref, coords.size(), nanovdb::cuda::noInit);
    BufT<float> dValues(cudaStream_t(0), ref, coords.size(), nanovdb::cuda::noInit);
    cudaCheck(cudaMemcpy(dCoords.data(), coords.data(), coords.size() * sizeof(Coord), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dValues.data(), b.values.data(), coords.size() * sizeof(float), cudaMemcpyHostToDevice));
    auto handle = tools::cuda::voxelsToGrid<BuildT, Coord*, nanovdb::cuda::ResourceRef<ResourceT>>(dCoords.data(), coords.size(), 1.0, cudaStream_t(0), ref);
    const auto* grid = handle.template deviceGrid<BuildT>();
    const auto td = util::cuda::DeviceGridTraits<BuildT>::getTreeData(grid, 0);
    if (td.mVoxelCount != coords.size()) { printf("voxel count mismatch\n"); exit(1); }
    BufT<float> phi(cudaStream_t(0), ref, td.mVoxelCount + 1, nanovdb::cuda::noInit);
    util::cuda::lambdaKernel<<<(coords.size() + 127) / 128, 128>>>(coords.size(), FillFromListFunctor(), grid, dCoords.data(), dValues.data(), phi.data());
    cudaCheckError();
    TrackerT tracker(std::move(handle), std::move(phi), W, 0, res);
    return tracker.release();
}

// ---- scene parameters of the programs that take mesh inputs ----

/// @brief The index bounding box of the voxels of a band.
inline nanovdb::CoordBBox bandBBox(const Band& b)
{
    nanovdb::CoordBBox box;
    for (const Coord& c : b.coords) box.expand(c);
    return box;
}

/// @brief The translation along +axis that puts @a moving one leaf (8 voxels) beyond @a fixed: the two bands do
///        not touch there, and they come together as the translation shrinks to 0.
inline int separatingOffset(const Band& fixed, const Band& moving, int axis)
{
    return bandBBox(fixed).max()[axis] - bandBBox(moving).min()[axis] + 8;
}

/// @brief The integer after the option @a name on the command line, or @a fallback if the option is not given.
inline int intOption(int argc, char** argv, const char* name, int fallback)
{
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], name) == 0) return std::atoi(argv[i + 1]);
    return fallback;
}

/// @brief The axis (0 = x, 1 = y, 2 = z) after the option @a name on the command line, or @a fallback.
inline int axisOption(int argc, char** argv, const char* name, int fallback)
{
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], name) == 0 && argv[i + 1][0] >= 'x' && argv[i + 1][0] <= 'z') return argv[i + 1][0] - 'x';
    return fallback;
}

#endif // NANOVDB_EXAMPLES_CSG_EXAMPLE_UTIL_CUH_HAS_BEEN_INCLUDED
