// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file openvdb_timing.cpp
///
/// @brief Wall time of OpenVDB's CSG operations on the two band files of nanovdb_timing.cu.
///        "in place": openvdb::tools::csgUnion, csgIntersection, csgDifference. They write the result into input 1
///                    and take nodes from input 2, so both inputs are consumed; each run gets deep copies of the
///                    inputs, and the copies are not timed. Timed with prune = false, with the defaults, and with
///                    pruneCancelledTiles = true.
///        "copy":     openvdb::tools::csgUnionCopy, csgIntersectionCopy, csgDifferenceCopy. They leave the inputs
///                    intact and return a new grid, as the NanoVDB operations do.
///        Two warm-up runs precede the timed runs of every row.
///
/// usage: openvdb_timing a.band b.band axis(0-2) offset runs

#include <openvdb/openvdb.h>
#include <openvdb/tools/Composite.h>
#include <openvdb/tools/SignedFloodFill.h>
#include <tbb/global_control.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

using namespace openvdb;

static FloatGrid::Ptr load(const char* path, Coord t)
{
    FloatGrid::Ptr g = FloatGrid::create(3.0f);
    g->setGridClass(GRID_LEVEL_SET);
    auto acc = g->getAccessor();
    FILE* f = fopen(path, "rb");
    if (!f) { printf("cannot read %s\n", path); exit(1); }
    uint64_t n;
    if (fread(&n, 8, 1, f) != 1) exit(1);
    for (uint64_t i = 0; i < n; ++i) {
        int32_t c[3]; float v;
        if (fread(c, 4, 3, f) != 3 || fread(&v, 4, 1, f) != 1) exit(1);
        acc.setValueOn(Coord(c[0], c[1], c[2]) + t, v);
    }
    fclose(f);
    tools::signedFloodFill(g->tree());
    return g;
}

int main(int argc, char** argv)
{
    if (argc < 6) { printf("usage: %s a.band b.band axis(0-2) offset runs\n", argv[0]); return 1; }
    initialize();
    Coord t(0); t[atoi(argv[3])] = atoi(argv[4]);
    const int runs = atoi(argv[5]);
    const FloatGrid::Ptr A = load(argv[1], Coord(0)), B = load(argv[2], t);
    printf("openvdb threads %zu, input 1 active %llu, input 2 active %llu\n",
           size_t(tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism)),
           (unsigned long long)A->activeVoxelCount(), (unsigned long long)B->activeVoxelCount());

    using Clock = std::chrono::steady_clock;
    const char* names[3] = {"1 | 2", "1 & 2", "1 - 2"};
    const char* modes[4] = {"in place, prune=false", "in place, defaults", "in place, pruneCancelledTiles", "copy"};
    for (int mode = 0; mode < 4; ++mode) for (int o = 0; o < 3; ++o) {
        std::vector<double> ts;
        uint64_t active = 0;
        for (int r = -2; r < runs; ++r) {
            double ms;
            if (mode < 3) {
                FloatGrid::Ptr a = A->deepCopy(), b = B->deepCopy();
                const bool prune = mode > 0, cancelled = mode == 2;
                const auto t0 = Clock::now();
                if (o == 0) tools::csgUnion(*a, *b, prune, cancelled);
                else if (o == 1) tools::csgIntersection(*a, *b, prune, cancelled);
                else tools::csgDifference(*a, *b, prune, cancelled);
                ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
                active = a->activeVoxelCount();
            } else {
                const auto t0 = Clock::now();
                FloatGrid::Ptr c = o == 0 ? tools::csgUnionCopy(*A, *B) : o == 1 ? tools::csgIntersectionCopy(*A, *B) : tools::csgDifferenceCopy(*A, *B);
                ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
                active = c->activeVoxelCount();
            }
            if (r >= 0) ts.push_back(ms);
        }
        std::sort(ts.begin(), ts.end());
        char label[64];
        snprintf(label, 64, "%s %s", names[o], modes[mode]);
        printf("openvdb | %-36s | min %9.3f | median %9.3f | max %9.3f | active %llu\n", label, ts.front(), ts[ts.size() / 2], ts.back(), (unsigned long long)active);
    }
    return 0;
}
