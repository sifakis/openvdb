// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file nanovdb_timing.cu
///
/// @brief Wall time of csgUnion, csgIntersection and csgDifference of tools/cuda/CSG.cuh on two band files (see
///        ../mesh_to_band.cu), for the comparison with openvdb_timing.cpp. The inputs are built on the device once.
///        One timed run is one call: the result is built and destroyed, between two cudaDeviceSynchronize calls.
///        csgUnionPropagateSign, the baseline that merges the bands, prunes them and reruns PropagateSign on the
///        result, is timed next to csgUnion.
///        Before timing, unions run for warmupSeconds so that the GPU reaches its sustained power state.
///        Built with -DCSG_STAGE_TIMERS against the header copy that stage_timers.py writes, it also prints the
///        time of each step of the operation.
///
/// usage: nanovdb_timing a.band b.band axis(0-2) offset runs warmupSeconds [baseline(0|1)]
///        Input 2 is b.band moved by offset voxels along the axis. baseline 0 leaves csgUnionPropagateSign out
///        (default 1); it needs more device memory than the other operations.

#include "CsgExampleUtil.cuh"

template<typename F> static void run(const char* label, F f, int runs, bool steps = true)
{
    for (int i = 0; i < 3; ++i) f();
#ifdef CSG_STAGE_TIMERS
    for (double& ms : gCsgStageMs) ms = 0;
#endif
    std::vector<double> t;
    for (int i = 0; i < runs; ++i) {
        cudaCheck(cudaDeviceSynchronize());
        const auto t0 = std::chrono::steady_clock::now();
        { auto r = f(); cudaCheck(cudaDeviceSynchronize()); }
        cudaCheck(cudaDeviceSynchronize());
        t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(t.begin(), t.end());
    const auto h = download(f());
    printf("nanovdb | %-21s | min %9.3f | median %9.3f | max %9.3f | active %llu\n", label, t.front(), t[t.size() / 2], t.back(), (unsigned long long)h.g()->activeVoxelCount());
#ifdef CSG_STAGE_TIMERS
    if (!steps) return; // the timers are in csgUnion, csgIntersection and csgDifference only
    const double* s = gCsgStageMs;
    printf("        steps, ms per call: setup %.3f | 1 retain masks %.3f | 2 topology %.3f | 3 values %.3f | 4 invert masks %.3f\n",
           s[0] / runs, s[1] / runs, s[2] / runs, s[3] / runs, s[4] / runs);
    printf("        topology, parts: root %.3f | internal mask buffers %.3f | stamp leaves %.3f | count nodes %.3f | buffer %.3f | grid, upper, lower %.3f | leaf masks %.3f | leaf offsets, bbox, finish %.3f\n",
           s[5] / runs, s[6] / runs, s[7] / runs, s[8] / runs, s[9] / runs, s[10] / runs, s[11] / runs, s[12] / runs);
#endif
}

int main(int argc, char** argv)
{
    if (argc < 7) { printf("usage: %s a.band b.band axis(0-2) offset runs warmupSeconds [baseline(0|1)]\n", argv[0]); return 1; }
    ResourceT& res = nanovdb::cuda::default_resource<ResourceT>();
    const Band bandA = loadBand(argv[1]), bandB = loadBand(argv[2]);
    Coord t(0); t[atoi(argv[3])] = atoi(argv[4]);
    const int runs = atoi(argv[5]);
    const double warmup = atof(argv[6]);
    const bool baseline = argc < 8 || atoi(argv[7]) != 0;
    StateT s1 = makeFromBand(bandA, Coord(0), res), s2 = makeFromBand(bandB, t, res);

    const auto w0 = std::chrono::steady_clock::now();
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count() < warmup) {
        auto r = tools::cuda::csgUnion<BuildT, float, ResourceT>(s1, s2, 0, res);
    }

    run("1 | 2", [&] { return tools::cuda::csgUnion<BuildT, float, ResourceT>(s1, s2, 0, res); }, runs);
    if (baseline) run("1 | 2 PropagateSign", [&] { return tools::cuda::csgUnionPropagateSign<BuildT, float, ResourceT>(s1, s2, W, 0, res); }, runs, false);
    run("1 & 2", [&] { return tools::cuda::csgIntersection<BuildT, float, ResourceT>(s1, s2, 0, res); }, runs);
    run("1 - 2", [&] { return tools::cuda::csgDifference<BuildT, float, ResourceT>(s1, s2, 0, res); }, runs);
    run("1 - 2 pruned", [&] { return tools::cuda::csgDifference<BuildT, float, ResourceT>(s1, s2, 0, res, true); }, runs);
    return 0;
}
