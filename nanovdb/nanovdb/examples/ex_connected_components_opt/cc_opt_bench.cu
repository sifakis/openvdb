// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file cc_opt_bench.cu
///
/// @brief Benchmark of tools::cuda::ConnectedComponentsOpt (ConnectedComponentsOpt.cuh) against
///        tools::cuda::ConnectedComponents on the same grid. The two are run alternately; the labels and the
///        component count of every run of the optimized version are compared with those of the current one.
///
/// Usage:
///   cc_opt_bench mesh   <voxelSize> <halfWidth> <runs> a.obj [b.obj ...]  narrow band of the meshes (MeshToGrid)
///   cc_opt_bench random <boxSize> <density> <runs> [seed]               each voxel of a box active with probability
///                                                                      density (many small components per leaf)
/// Prints the median wall time of each over @a runs runs (after a warm-up run), and each one's stage timers.

#include <nanovdb/tools/cuda/ConnectedComponentsOpt.cuh>
#include <nanovdb/tools/cuda/ConnectedComponents.cuh>
#include <nanovdb/tools/cuda/MeshToGrid.cuh>
#include <nanovdb/tools/cuda/PointsToGrid.cuh>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

using BuildT  = nanovdb::ValueOnIndex;
using GridT   = nanovdb::NanoGrid<BuildT>;
using BufferT = nanovdb::cuda::Buffer<std::byte, nanovdb::cuda::DeviceResource>;

static void readObj(const char* path, std::vector<nanovdb::Vec3f>& pts, std::vector<nanovdb::Vec3i>& tris)
{
    const int base = int(pts.size());
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.size() < 2) continue;
        if (line[0] == 'v' && line[1] == ' ') {
            float x, y, z;
            if (std::sscanf(line.c_str() + 2, "%f %f %f", &x, &y, &z) == 3) pts.emplace_back(x, y, z);
        } else if (line[0] == 'f' && line[1] == ' ') {
            std::istringstream iss(line.substr(2));
            std::string tok;
            std::vector<int> idx;
            while (iss >> tok) {
                const int i = std::stoi(tok.substr(0, tok.find('/')));
                idx.push_back(i > 0 ? base + i - 1 : int(pts.size()) + i);
            }
            for (size_t k = 2; k < idx.size(); ++k) tris.emplace_back(idx[0], idx[k - 1], idx[k]);
        }
    }
}

/// @brief One run of a connected-components class: wall time, labels and count, and its stage timers if verbose.
template <typename CC>
static double runOnce(const GridT* grid, bool verbose, std::vector<uint32_t>& labels, uint32_t& count, std::string& timers)
{
    std::ostringstream sink;
    auto* oldErr = std::cerr.rdbuf(sink.rdbuf());
    auto* oldOut = std::cout.rdbuf(sink.rdbuf());
    cudaCheck(cudaDeviceSynchronize());
    const auto t0 = std::chrono::steady_clock::now();
    CC cc(grid);
    cc.setVerbose(verbose ? 1 : 0);
    auto result = cc.getVoxelLabelsAndCount();
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::cerr.rdbuf(oldErr);
    std::cout.rdbuf(oldOut);
    labels.resize(result.first.size());
    cudaCheck(cudaMemcpy(labels.data(), result.first.data(), labels.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    count = result.second;
    timers = sink.str();
    return ms;
}

static void printTimers(const std::string& text)
{
    std::istringstream in(text);
    std::string line;
    std::regex re(R"(^(.*?)\s*\.\.\.\s*completed in ([0-9.]+) milliseconds)");
    while (std::getline(in, line)) {
        std::smatch m;
        if (std::regex_search(line, m, re) && std::stod(m[2].str()) >= 0.05)
            std::printf("      %-52s %9.2f ms\n", m[1].str().c_str(), std::stod(m[2].str()));
    }
}

static int compare(const GridT* grid, int runs)
{
    using Current   = nanovdb::tools::cuda::ConnectedComponents<BuildT>;
    using Optimized = nanovdb::tools::cuda::ConnectedComponentsOpt<BuildT>;
    std::vector<uint32_t> ref, labels;
    uint32_t refCount = 0, count = 0;
    std::string timers;
    std::vector<double> tCur, tOpt;
    bool identical = true;
    for (int r = 0; r <= runs; ++r) {  // run 0 is the warm-up
        const double a = runOnce<Current>(grid, false, ref, refCount, timers);
        const double b = runOnce<Optimized>(grid, false, labels, count, timers);
        identical &= (labels == ref) && (count == refCount);
        if (r > 0) { tCur.push_back(a); tOpt.push_back(b); }
    }
    std::sort(tCur.begin(), tCur.end());
    std::sort(tOpt.begin(), tOpt.end());
    const auto active = nanovdb::util::cuda::DeviceGridTraits<BuildT>::getActiveVoxelCount(grid);
    const auto leaves = nanovdb::util::cuda::DeviceGridTraits<BuildT>::getTreeData(grid).mNodeCount[0];
    std::printf("  active voxels %llu, leaves %u, components %u\n", (unsigned long long)active, leaves, refCount);
    std::printf("  ConnectedComponents     median %9.2f ms\n", tCur[tCur.size() / 2]);
    runOnce<Current>(grid, true, ref, refCount, timers);
    printTimers(timers);
    std::printf("  ConnectedComponentsOpt  median %9.2f ms  (%.2fx)\n", tOpt[tOpt.size() / 2],
                tCur[tCur.size() / 2] / tOpt[tOpt.size() / 2]);
    runOnce<Optimized>(grid, true, labels, count, timers);
    printTimers(timers);
    std::printf("  labels and count identical in every run: %s\n", identical ? "yes" : "NO");
    return identical ? 0 : 1;
}

int main(int argc, char** argv)
{
    if (argc < 5) {
        std::fprintf(stderr, "usage: %s mesh <voxelSize> <halfWidth> <runs> a.obj [b.obj ...]\n"
                             "       %s random <boxSize> <density> <runs> [seed]\n", argv[0], argv[0]);
        return 1;
    }
    const int runs = std::atoi(argv[4]);
    if (std::strcmp(argv[1], "mesh") == 0) {
        std::vector<nanovdb::Vec3f> pts;
        std::vector<nanovdb::Vec3i> tris;
        for (int i = 5; i < argc; ++i) readObj(argv[i], pts, tris);
        nanovdb::cuda::Buffer<nanovdb::Vec3f> dPts(cudaStream_t(0), pts.size(), nanovdb::cuda::noInit);
        nanovdb::cuda::Buffer<nanovdb::Vec3i> dTris(cudaStream_t(0), tris.size(), nanovdb::cuda::noInit);
        cudaCheck(cudaMemcpy(dPts.data(), pts.data(), dPts.size_bytes(), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(dTris.data(), tris.data(), dTris.size_bytes(), cudaMemcpyHostToDevice));
        nanovdb::Map map;
        map.set(std::atof(argv[2]), nanovdb::Vec3d(0.0), 1.0);
        nanovdb::tools::cuda::MeshToGrid<BuildT> converter(dPts.data(), uint32_t(pts.size()), dTris.data(),
                                                           uint32_t(tris.size()), map);
        converter.setNarrowBandWidth(float(std::atof(argv[3])));
        auto handle = converter.getHandle<BufferT>(BufferT());
        dPts.destroy();
        dTris.destroy();
        std::printf("mesh, %zu triangles, voxel size %s, half-width %s\n", tris.size(), argv[2], argv[3]);
        return compare(handle.deviceGrid<BuildT>(), runs);
    }
    const int    box     = std::atoi(argv[2]);
    const double density = std::atof(argv[3]);
    std::mt19937 rng(argc > 5 ? std::atoi(argv[5]) : 1);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<nanovdb::Coord> coords;
    for (int i = 0; i < box; ++i)
        for (int j = 0; j < box; ++j)
            for (int k = 0; k < box; ++k)
                if (u(rng) < density) coords.emplace_back(i, j, k);
    nanovdb::cuda::Buffer<nanovdb::Coord> dCoords(cudaStream_t(0), coords.size(), nanovdb::cuda::noInit);
    cudaCheck(cudaMemcpy(dCoords.data(), coords.data(), dCoords.size_bytes(), cudaMemcpyHostToDevice));
    auto handle = nanovdb::tools::cuda::voxelsToGrid<BuildT>(dCoords.data(), coords.size(), 1.0, BufferT());
    std::printf("random, box %d^3, density %s\n", box, argv[3]);
    return compare(handle.deviceGrid<BuildT>(), runs);
}
