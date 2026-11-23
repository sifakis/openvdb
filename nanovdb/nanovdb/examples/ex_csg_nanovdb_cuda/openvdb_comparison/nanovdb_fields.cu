// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file nanovdb_fields.cu
///
/// @brief Runs the CSG operations of tools/cuda/CSG.cuh on two band files (see ../mesh_to_band.cu) and writes, for
///        every voxel of a box around both inputs, the full-space field value and the active state of six level
///        sets: input 1, input 2, 1 | 2, 1 & 2, 1 - 2, and 1 - 2 with pruneCancelledLeaves. openvdb_fields.cpp
///        reads the file and compares it with OpenVDB, voxel by voxel.
///
/// usage: nanovdb_fields a.band b.band axis(0-2) offset out.bin [union.band]
///        Input 2 is b.band moved by offset voxels along the axis. If union.band is given, 1 | 2 is also written
///        as a band file, to build inputs that share a part of their surface.

#include "CsgExampleUtil.cuh"

int main(int argc, char** argv)
{
    if (argc < 6) { printf("usage: %s a.band b.band axis(0-2) offset out.bin [union.band]\n", argv[0]); return 1; }
    ResourceT& res = nanovdb::cuda::default_resource<ResourceT>();
    const Band bandA = loadBand(argv[1]), bandB = loadBand(argv[2]);
    Coord t(0); t[atoi(argv[3])] = atoi(argv[4]);
    StateT s1 = makeFromBand(bandA, Coord(0), res), s2 = makeFromBand(bandB, t, res);

    HostState h[6];
    h[0] = download(s1);
    h[1] = download(s2);
    h[2] = download(tools::cuda::csgUnion<BuildT, float, ResourceT>(s1, s2, 0, res));
    h[3] = download(tools::cuda::csgIntersection<BuildT, float, ResourceT>(s1, s2, 0, res));
    h[4] = download(tools::cuda::csgDifference<BuildT, float, ResourceT>(s1, s2, 0, res));
    h[5] = download(tools::cuda::csgDifference<BuildT, float, ResourceT>(s1, s2, 0, res, true));

    if (argc > 6) { // the union as a band file
        FILE* b = fopen(argv[6], "wb");
        if (!b) { printf("cannot write %s\n", argv[6]); return 1; }
        const uint64_t n = h[2].g()->activeVoxelCount();
        fwrite(&n, 8, 1, b);
        const auto& tree = h[2].g()->tree();
        for (uint32_t l = 0; l < tree.nodeCount(0); ++l) {
            const auto& leaf = tree.template getFirstNode<0>()[l];
            for (uint32_t o = 0; o < 512; ++o) {
                if (!leaf.isActive(o)) continue;
                const Coord c = leaf.offsetToGlobalCoord(o);
                const int32_t ijk[3] = {c[0], c[1], c[2]};
                fwrite(ijk, 4, 3, b);
                fwrite(&h[2].phi[leaf.getValue(o)], 4, 1, b);
            }
        }
        fclose(b);
    }

    // The box: both inputs, padded by 4 voxels.
    auto bb = h[0].g()->indexBBox(); bb.expand(h[1].g()->indexBBox());
    const int32_t box[6] = {bb.min()[0] - 4, bb.min()[1] - 4, bb.min()[2] - 4, bb.max()[0] + 4, bb.max()[1] + 4, bb.max()[2] + 4};
    FILE* f = fopen(argv[5], "wb");
    if (!f) { printf("cannot write %s\n", argv[5]); return 1; }
    fwrite(box, 4, 6, f);
    const char* names[6] = {"input 1", "input 2", "1 | 2", "1 & 2", "1 - 2", "1 - 2 pruned"};
    for (int n = 0; n < 6; ++n) {
        std::vector<float> val; std::vector<unsigned char> act;
        for (int i = box[0]; i <= box[3]; ++i) for (int j = box[1]; j <= box[4]; ++j) for (int k = box[2]; k <= box[5]; ++k) {
            const Coord p(i, j, k);
            const auto* leaf = h[n].g()->tree().root().probeLeaf(p);
            const uint32_t o = NanoLeaf<BuildT>::CoordToOffset(p);
            const bool on = leaf && leaf->isActive(o);
            act.push_back(on);
            // An inactive voxel holds the background, negated where the invert masks say so.
            val.push_back(on ? h[n].phi[leaf->getValue(o)]
                : tools::cuda::levelset::detail::isInverted(*h[n].g(), p, h[n].leaf.data(), h[n].lower.data(), h[n].upper.data(), h[n].root.data(), h[n].root.size()) ? -h[n].phi[0] : h[n].phi[0]);
        }
        fwrite(val.data(), 4, val.size(), f);
        fwrite(act.data(), 1, act.size(), f);
        printf("nanovdb %-12s active %llu leaves %u\n", names[n], (unsigned long long)h[n].g()->activeVoxelCount(), h[n].g()->tree().nodeCount(0));
    }
    fclose(f);
    return 0;
}
