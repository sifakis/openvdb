// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file openvdb_fields.cpp
///
/// @brief Builds the two level sets of nanovdb_fields.cu in OpenVDB (the voxels and values of the band files, the
///        background 3, the signs of the inactive voxels by signedFloodFill), runs openvdb::tools::csgUnion,
///        csgIntersection and csgDifference, and compares the field value and the active state of every voxel
///        with the file that nanovdb_fields wrote. The OpenVDB operations run in three settings:
///            prune = false
///            the defaults (prune = true, pruneCancelledTiles = false)
///            prune = true, pruneCancelledTiles = true; here the difference is compared with the NanoVDB
///            difference that was made with pruneCancelledLeaves.
///        Exits with 1 if any voxel differs.
///
/// usage: openvdb_fields a.band b.band axis(0-2) offset fields.bin   (the arguments given to nanovdb_fields)

#include <openvdb/openvdb.h>
#include <openvdb/tools/Composite.h>
#include <openvdb/tools/SignedFloodFill.h>
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
    if (argc < 6) { printf("usage: %s a.band b.band axis(0-2) offset fields.bin\n", argv[0]); return 1; }
    initialize();
    Coord t(0); t[atoi(argv[3])] = atoi(argv[4]);

    FILE* f = fopen(argv[5], "rb");
    int32_t box[6];
    if (!f || fread(box, 4, 6, f) != 6) { printf("cannot read %s\n", argv[5]); return 1; }
    const size_t N = size_t(box[3] - box[0] + 1) * (box[4] - box[1] + 1) * (box[5] - box[2] + 1);
    std::vector<float> val[6]; std::vector<unsigned char> act[6];
    for (int n = 0; n < 6; ++n) {
        val[n].resize(N); act[n].resize(N);
        if (fread(val[n].data(), 4, N, f) != N || fread(act[n].data(), 1, N, f) != N) return 1;
    }
    fclose(f);

    size_t mismatches = 0;
    const auto compare = [&](const char* label, const FloatGrid& g, int n) {
        auto acc = g.getConstAccessor();
        size_t i = 0, valueBad = 0, signBad = 0, onlyNano = 0, onlyOvdb = 0, nanoOn = 0;
        for (int x = box[0]; x <= box[3]; ++x) for (int y = box[1]; y <= box[4]; ++y) for (int z = box[2]; z <= box[5]; ++z, ++i) {
            float v;
            const bool on = acc.probeValue(Coord(x, y, z), v);
            nanoOn += act[n][i];
            if (v != val[n][i]) ++valueBad;
            if ((v < 0) != (val[n][i] < 0)) ++signBad;
            if (on && !act[n][i]) ++onlyOvdb;
            if (!on && act[n][i]) ++onlyNano;
        }
        mismatches += valueBad + signBad + onlyNano + onlyOvdb;
        printf("%-30s openvdb active %10llu, nanovdb active %10zu | value differs %zu, sign differs %zu | active only in nanovdb %zu, only in openvdb %zu\n",
               label, (unsigned long long)g.activeVoxelCount(), nanoOn, valueBad, signBad, onlyNano, onlyOvdb);
    };

    compare("input 1", *load(argv[1], Coord(0)), 0);
    compare("input 2", *load(argv[2], t), 1);
    const char* names[3] = {"1 | 2", "1 & 2", "1 - 2"};
    for (int mode = 0; mode < 3; ++mode) for (int o = 0; o < 3; ++o) {
        FloatGrid::Ptr a = load(argv[1], Coord(0)), b = load(argv[2], t);
        const bool prune = mode > 0, cancelled = mode == 2;
        if (o == 0) tools::csgUnion(*a, *b, prune, cancelled);
        else if (o == 1) tools::csgIntersection(*a, *b, prune, cancelled);
        else tools::csgDifference(*a, *b, prune, cancelled);
        char label[64];
        snprintf(label, 64, "%s  %s", names[o], mode == 0 ? "prune=false" : mode == 1 ? "defaults" : "pruneCancelledTiles");
        compare(label, *a, (mode == 2 && o == 2) ? 5 : 2 + o);
    }
    printf("%s\n", mismatches ? "DIFFERENT" : "IDENTICAL");
    return mismatches ? 1 : 0;
}
