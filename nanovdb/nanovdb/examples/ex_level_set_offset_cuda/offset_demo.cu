// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file offset_demo.cu
///
/// @brief Offsets a level set made by MeshToSDF with tools::cuda::LevelSetTracker and writes every frame for the
///        web viewer (viewer/make_viewer.py).
///          grow:   <steps> x (phi -= 0.5 voxel; track()), starting from the input
///          shrink: <steps> x (phi += 0.5 voxel; track()), starting from the input
///        <wide.sdf> is the same mesh at the same resolution with a band of at least steps / 2 + 4 voxels. wide -
///        offset is the exact distance to the offset surface on the side the surface moved to, so the error is
///        taken on that side only, within 1 voxel of the offset surface. Both files are written by
///        mesh_to_sdf_file.
///
///        Output: <prefix>.json (one record per frame) and <prefix>.bin (per frame: surface points, then one
///        slice).
///          point: 3 x uint16 position (index space, mapped from [lo, lo + extent]) and 2 x int8 octahedral normal
///          slice: int8 phi / background * 127 on the plane through the middle of the thinnest axis; -128 = inactive
///
/// usage: offset_demo <input.sdf> <wide.sdf> <steps> <out prefix>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/cuda/LevelSetTracker.cuh>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

using BuildT    = nanovdb::ValueOnIndex;
using ResourceT = nanovdb::cuda::DeviceResource;
using TrackerT  = nanovdb::tools::cuda::LevelSetTracker<BuildT, float, ResourceT>;
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

static HostLevelSet download(const TrackerT::State& state)
{
    HostLevelSet h;
    h.gridBytes.resize(state.grid.buffer().size());
    cudaCheck(cudaMemcpy(h.gridBytes.data(), state.grid.buffer().data(), h.gridBytes.size(), cudaMemcpyDeviceToHost));
    h.phi.resize(state.phi.size());
    cudaCheck(cudaMemcpy(h.phi.data(), state.phi.data(), state.phi.size_bytes(), cudaMemcpyDeviceToHost));
    return h;
}

static TrackerT makeTracker(const HostLevelSet& h, float background)
{
    BufferT<std::byte> gridBuffer(cudaStream_t(0), ref(), h.gridBytes.size(), nanovdb::cuda::noInit);
    cudaCheck(cudaMemcpy(gridBuffer.data(), h.gridBytes.data(), h.gridBytes.size(), cudaMemcpyHostToDevice));
    BufferT<float> phi(cudaStream_t(0), ref(), h.phi.size(), nanovdb::cuda::noInit);
    cudaCheck(cudaMemcpy(phi.data(), h.phi.data(), phi.size_bytes(), cudaMemcpyHostToDevice));
    return TrackerT(HandleT(std::move(gridBuffer)), std::move(phi), background);
}

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
static void forEachActive(const HostLevelSet& h, F f)
{
    const auto& tree = h.grid()->tree();
    for (uint32_t l = 0; l < tree.nodeCount(0); ++l) {
        const auto& leaf = tree.template getFirstNode<0>()[l];
        for (uint32_t n = 0; n < 512; ++n)
            if (leaf.isActive(n)) f(leaf.offsetToGlobalCoord(n), h.phi[leaf.getValue(n)]);
    }
}

static double quantile(std::vector<float>& v, double q)
{
    if (v.empty()) return 0;
    const size_t k = std::min(v.size() - 1, size_t(q * double(v.size())));
    std::nth_element(v.begin(), v.begin() + k, v.end());
    return v[k];
}

struct Frame
{
    double   offset;  // voxels; positive = grown
    uint64_t voxels, outOfBand, wrongSign, errorCount, pointCount, pointsAt, sliceAt;
    double   trackMs, errMedian, err99, errMax;
};

int main(int argc, char** argv)
{
    if (argc < 5) { std::printf("usage: offset_demo <input.sdf> <wide.sdf> <steps> <out prefix>\n"); return 1; }
    double dx, wideDx;
    float  background, wideBackground;
    const HostLevelSet input = readSdf(argv[1], dx, background);
    const HostLevelSet wide  = readSdf(argv[2], wideDx, wideBackground);
    const int          steps = std::atoi(argv[3]);
    const std::string  prefix = argv[4];
    if (wideDx != dx) { std::printf("the two files have different voxel sizes\n"); return 1; }
    if (wideBackground / dx < 0.5 * steps + 4) { std::printf("wide band too narrow\n"); return 1; }
    const auto accW = wide.grid()->getAccessor();

    // Index-space box shared by every frame: the input's active voxels, grown by the largest offset plus a margin.
    nanovdb::Coord lo(1 << 30), hi(-(1 << 30));
    forEachActive(input, [&](const nanovdb::Coord& c, float) { lo.minComponent(c); hi.maxComponent(c); });
    const int margin = steps / 2 + 2;
    for (int a = 0; a < 3; ++a) lo[a] -= margin, hi[a] += margin;
    const int extent = std::max({hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]});
    int sliceAxis = 0;
    for (int a = 1; a < 3; ++a) if (hi[a] - lo[a] < hi[sliceAxis] - lo[sliceAxis]) sliceAxis = a;
    const int uAxis = (sliceAxis + 1) % 3, vAxis = (sliceAxis + 2) % 3;
    const int sliceIndex = (lo[sliceAxis] + hi[sliceAxis]) / 2;
    const int sliceW = hi[uAxis] - lo[uAxis] + 1, sliceH = hi[vAxis] - lo[vAxis] + 1;

    std::ofstream bin(prefix + ".bin", std::ios::binary);
    uint64_t binSize = 0;
    std::vector<Frame> frames;

    auto dump = [&](const HostLevelSet& h, double offset, double trackMs) {
        Frame f{};
        f.offset = offset, f.trackMs = trackMs, f.voxels = h.grid()->tree().activeVoxelCount();
        const auto acc = h.grid()->getAccessor();
        const int sign = offset > 0 ? 1 : -1;

        // Error against wide - offset, on the side the surface moved to.
        std::vector<float> err;
        forEachActive(h, [&](const nanovdb::Coord& c, float v) {
            if (std::fabs(v) >= background) ++f.outOfBand;
            if (!accW.isActive(c)) return;
            const double exact = double(wide.phi[accW.getValue(c)]) - offset * dx;
            if ((v < 0) != (exact < 0) && std::fabs(exact) > 0.25 * dx) ++f.wrongSign;
            if (offset != 0 && exact * sign < 0) return;
            if (std::fabs(exact) < 1.0 * dx) err.push_back(float(std::fabs(v - exact) / dx));
        });
        f.errorCount = err.size();
        f.errMax     = err.empty() ? 0.0 : double(*std::max_element(err.begin(), err.end()));
        f.errMedian  = quantile(err, 0.5), f.err99 = quantile(err, 0.99);

        // Surface points: voxels within 0.6 voxel of the surface, moved onto it along the gradient.
        std::vector<char> points;
        forEachActive(h, [&](const nanovdb::Coord& c, float v) {
            if (std::fabs(v) >= 0.6 * dx) return;
            double g[3];
            for (int a = 0; a < 3; ++a) {
                nanovdb::Coord p = c, m = c;
                p[a] += 1, m[a] -= 1;
                if (!acc.isActive(p) || !acc.isActive(m)) return;
                g[a] = (double(h.phi[acc.getValue(p)]) - h.phi[acc.getValue(m)]) / (2 * dx);
            }
            const double len = std::sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
            if (len < 1e-6) return;
            uint16_t q[3];
            for (int a = 0; a < 3; ++a) {
                g[a] /= len;
                const double p = (c[a] - (v / dx) * g[a] - lo[a]) / double(extent);
                q[a] = uint16_t(std::lround(std::min(1.0, std::max(0.0, p)) * 65535.0));
            }
            const double l1 = std::fabs(g[0]) + std::fabs(g[1]) + std::fabs(g[2]);
            double ox = g[0] / l1, oy = g[1] / l1;
            if (g[2] < 0) {
                const double tx = (1 - std::fabs(oy)) * (ox >= 0 ? 1 : -1), ty = (1 - std::fabs(ox)) * (oy >= 0 ? 1 : -1);
                ox = tx, oy = ty;
            }
            const int8_t o[2] = {int8_t(std::lround(ox * 127)), int8_t(std::lround(oy * 127))};
            points.insert(points.end(), reinterpret_cast<const char*>(q), reinterpret_cast<const char*>(q) + 6);
            points.insert(points.end(), reinterpret_cast<const char*>(o), reinterpret_cast<const char*>(o) + 2);
        });
        f.pointCount = points.size() / 8, f.pointsAt = binSize;
        bin.write(points.data(), points.size());
        binSize += points.size();

        std::vector<int8_t> slice(size_t(sliceW) * sliceH, int8_t(-128));
        for (int j = 0; j < sliceH; ++j)
            for (int i = 0; i < sliceW; ++i) {
                nanovdb::Coord c;
                c[sliceAxis] = sliceIndex, c[uAxis] = lo[uAxis] + i, c[vAxis] = lo[vAxis] + j;
                if (!acc.isActive(c)) continue;
                const double t = std::min(1.0, std::max(-1.0, double(h.phi[acc.getValue(c)]) / background));
                slice[size_t(j) * sliceW + i] = int8_t(std::lround(t * 127));
            }
        f.sliceAt = binSize;
        bin.write(reinterpret_cast<const char*>(slice.data()), slice.size());
        binSize += slice.size();
        frames.push_back(f);
        std::printf("offset %+5.1f voxels: %8llu active voxels, track %7.2f ms, |phi| >= background %llu, wrong sign %llu, "
                    "error (n %llu) median %.4f 99%% %.4f max %.4f voxel, %llu points\n", offset, (unsigned long long)f.voxels,
                    trackMs, (unsigned long long)f.outOfBand, (unsigned long long)f.wrongSign, (unsigned long long)f.errorCount,
                    f.errMedian, f.err99, f.errMax, (unsigned long long)f.pointCount);
    };

    dump(input, 0.0, 0.0);
    for (int sign = 1; sign >= -1; sign -= 2) {
        HostLevelSet current = input;
        const float step = float(0.5 * dx) * float(sign); // subtracted: + grows, - shrinks
        for (int i = 1; i <= steps; ++i) {
            for (size_t k = 1; k < current.phi.size(); ++k) current.phi[k] -= step;
            TrackerT tracker = makeTracker(current, background);
            cudaCheck(cudaDeviceSynchronize());
            const auto t0 = std::chrono::steady_clock::now();
            tracker.track();
            cudaCheck(cudaDeviceSynchronize());
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            current = download(tracker.release());
            dump(current, 0.5 * i * sign, ms);
        }
    }

    std::sort(frames.begin(), frames.end(), [](const Frame& a, const Frame& b) { return a.offset < b.offset; });
    std::FILE* js = std::fopen((prefix + ".json").c_str(), "w");
    std::fprintf(js, "{\"dx\": %.9g, \"backgroundVoxels\": %.6g, \"stepVoxels\": 0.5, \"lo\": [%d, %d, %d], \"extent\": %d,\n"
                     " \"sliceAxis\": %d, \"uAxis\": %d, \"vAxis\": %d, \"sliceIndex\": %d, \"sliceW\": %d, \"sliceH\": %d,\n"
                     " \"frames\": [\n", dx, background / dx, lo[0], lo[1], lo[2], extent, sliceAxis, uAxis, vAxis,
                 sliceIndex, sliceW, sliceH);
    for (size_t i = 0; i < frames.size(); ++i) {
        const Frame& f = frames[i];
        std::fprintf(js, "  {\"offset\": %.1f, \"voxels\": %llu, \"trackMs\": %.3f, \"outOfBand\": %llu, \"wrongSign\": %llu, "
                         "\"errorCount\": %llu, \"errMedian\": %.5f, \"err99\": %.5f, \"errMax\": %.5f, \"pointCount\": %llu, "
                         "\"pointsAt\": %llu, \"sliceAt\": %llu}%s\n", f.offset, (unsigned long long)f.voxels, f.trackMs,
                     (unsigned long long)f.outOfBand, (unsigned long long)f.wrongSign, (unsigned long long)f.errorCount,
                     f.errMedian, f.err99, f.errMax, (unsigned long long)f.pointCount, (unsigned long long)f.pointsAt,
                     (unsigned long long)f.sliceAt, i + 1 < frames.size() ? "," : "");
    }
    std::fprintf(js, " ]}\n");
    std::fclose(js);
    std::printf("wrote %s.json and %s.bin (%llu bytes)\n", prefix.c_str(), prefix.c_str(), (unsigned long long)binSize);
    return 0;
}
