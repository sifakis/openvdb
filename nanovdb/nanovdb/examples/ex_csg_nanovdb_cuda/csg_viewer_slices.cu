// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file csg_viewer_slices.cu
///
/// @brief Data for the slice view of the CSG viewer (viewer/) and for figures: one mesh level set stays in place,
///        a second one moves toward it in steps, and for every step csgUnion, csgIntersection and csgDifference
///        run for each of the four sign pairs of the backgrounds (an input is negated with csgComplement). The
///        program writes, as JSON, one slice of the two inputs and of each of the 12 results per step, and checks
///        every result against the per-voxel rule. viewer/README.md describes the file.
///
/// usage: csg_viewer_slices fixed.band moving.band out.json [options]   (band files: see mesh_to_band.cu)
///   --frames N        number of steps (default 30)
///   --move-axis A     x, y or z: the moving input is translated along +A (default z)
///   --start S         translation in the first step, in voxels; it shrinks linearly to 0 in the last step
///                     (default: the moving input starts one leaf beyond the fixed one)
///   --slice-axis A    x, y or z: the slice is a plane normal to A (default x)
///   --slice-index I   coordinate of the slice plane along its normal (default 0)
/// The slice covers the bounding box of both inputs over all steps, padded by 8 voxels.

#include "CsgExampleUtil.cuh"

template<typename M> static uint64_t bitsOn(const std::vector<M>& v) { uint64_t n = 0; for (const auto& m : v) n += m.countOn(); return n; }

// Where the slice voxel p stands in a level set held as (grid, phi, invert masks):
// 0 out, 1 active negative, 2 active positive, 3 inverted by a leaf mask bit, 4 by a lower mask bit, 5 by an upper
// mask bit, 6 by a root origin; +8 if a leaf exists at p.
static int codeOf(const HostState& h, const Coord& p)
{
    using UpperT = nanovdb::NanoUpper<BuildT>; using LowerT = nanovdb::NanoLower<BuildT>;
    const auto& tree = h.g()->tree();
    const auto* tile = tree.root().probeTile(p);
    if (!tile) return std::binary_search(h.root.begin(), h.root.end(), p & ~int(UpperT::DIM - 1)) ? 6 : 0;
    const auto* upper = tree.root().getChild(tile);
    const uint32_t u = UpperT::CoordToOffset(p);
    if (!upper->isChild(u)) return h.upper[upper - tree.getFirstUpper()].isOn(u) ? 5 : 0;
    const auto* lower = upper->getChild(u);
    const uint32_t l = LowerT::CoordToOffset(p);
    if (!lower->isChild(l)) return h.lower[lower - tree.getFirstLower()].isOn(l) ? 4 : 0;
    const auto* leaf = lower->getChild(l);
    const uint32_t n = NanoLeaf<BuildT>::CoordToOffset(p);
    if (leaf->isActive(n)) return 8 + (h.phi[leaf->getValue(n)] < 0 ? 1 : 2);
    return 8 + (h.leaf[leaf - tree.getFirstLeaf()].isOn(n) ? 3 : 0);
}

static void rle(FILE* f, const std::vector<unsigned char>& v)
{
    fputc('"', f);
    for (size_t i = 0; i < v.size();) { size_t j = i; while (j < v.size() && v[j] == v[i]) ++j; fprintf(f, "%c%zu", 'A' + v[i], j - i); i = j; }
    fputc('"', f);
}

int main(int argc, char** argv)
{
    if (argc < 4) {
        printf("usage: %s fixed.band moving.band out.json [--frames N] [--move-axis x|y|z] [--start S] [--slice-axis x|y|z] [--slice-index I]\n", argv[0]);
        return 1;
    }
    ResourceT& res = nanovdb::cuda::default_resource<ResourceT>();
    const Band bandA = loadBand(argv[1]), bandB = loadBand(argv[2]);

    // The scene: how the moving input travels, and which plane is written.
    const int FRAMES     = std::max(1, intOption(argc, argv, "--frames", 30));
    const int moveAxis   = axisOption(argc, argv, "--move-axis", 2);
    const int START      = intOption(argc, argv, "--start", separatingOffset(bandA, bandB, moveAxis));
    const int sliceAxis  = axisOption(argc, argv, "--slice-axis", 0);
    const int sliceIndex = intOption(argc, argv, "--slice-index", 0);
    // Picture axes: horizontal u is the direction of motion if it lies in the plane, vertical v is the other one.
    const int uAxis = moveAxis != sliceAxis ? moveAxis : (sliceAxis + 1) % 3;
    const int vAxis = 3 - sliceAxis - uAxis;
    // The window: both inputs over all steps, padded by one leaf.
    nanovdb::CoordBBox window = bandBBox(bandA), boxB = bandBBox(bandB);
    window.expand(boxB);
    Coord farthest = boxB.max(); farthest[moveAxis] += START;
    window.expand(farthest);
    const int U0 = window.min()[uAxis] - 8, V0 = window.min()[vAxis] - 8;
    const int WU = window.max()[uAxis] + 8 - U0 + 1, WV = window.max()[vAxis] + 8 - V0 + 1;
    // The voxel at column i of the row at height j above the bottom row.
    const auto at = [&](int i, int j) { Coord c; c[sliceAxis] = sliceIndex; c[uAxis] = U0 + i; c[vAxis] = V0 + j; return c; };
    const auto translation = [&](int off) { Coord t(0); t[moveAxis] = off; return t; };

    const auto slice = [&](const HostState& h) { // row 0 is the top row
        std::vector<unsigned char> v(size_t(WU) * WV);
        for (int j = 0; j < WV; ++j) for (int i = 0; i < WU; ++i) v[size_t(j) * WU + i] = codeOf(h, at(i, WV - 1 - j));
        return v;
    };
    // the full-space field: the stored value, or the background with the sign told by the invert bit
    const auto fieldOf = [](const HostState& h, const Coord& p) {
        const auto* leaf = h.g()->tree().root().probeLeaf(p);
        const uint32_t n = NanoLeaf<BuildT>::CoordToOffset(p);
        if (leaf && leaf->isActive(n)) return h.phi[leaf->getValue(n)];
        return tools::cuda::levelset::detail::isInverted(*h.g(), p, h.leaf.data(), h.lower.data(), h.upper.data(), h.root.data(), h.root.size()) ? -h.phi[0] : h.phi[0];
    };
    const auto complement = [](StateT& s) { tools::cuda::csgComplement<BuildT, float, ResourceT>(s, 0); cudaCheck(cudaDeviceSynchronize()); };

    FILE* f = fopen(argv[3], "w");
    {
        StateT s1 = makeFromBand(bandA, Coord(0), res);
        const HostState h1 = download(s1);
        fprintf(f, "{\"band\":%g,\"moveAxis\":%d,\"sliceAxis\":%d,\"sliceIndex\":%d,\"uAxis\":%d,\"vAxis\":%d,\"x0\":%d,\"y0\":%d,\"w\":%d,\"h\":%d,"
                   "\"flipY\":true,\"in1\":{\"leaves\":%u,\"active\":%llu},\"i1\":", W, moveAxis, sliceAxis, sliceIndex, uAxis, vAxis, U0, V0, WU, WV,
                h1.g()->tree().nodeCount(0), (unsigned long long)h1.g()->activeVoxelCount());
        rle(f, slice(h1));
        fprintf(f, ",\"frames\":[\n");
    }
    bool allOk = true;
    for (int fr = 0; fr < FRAMES; ++fr) {
        const int off = FRAMES > 1 ? int(std::lround(double(START) * (FRAMES - 1 - fr) / (FRAMES - 1))) : START;
        uint64_t checked = 0, ruleBad = 0;
        fprintf(f, "%s{\"off\":%d,", fr ? ",\n" : "", off);
        for (int neg1 = 0; neg1 < 2; ++neg1) for (int neg2 = 0; neg2 < 2; ++neg2) {
            StateT s1 = makeFromBand(bandA, Coord(0), res), s2 = makeFromBand(bandB, translation(off), res);
            if (!neg1 && !neg2) {
                const HostState h2 = download(s2);
                fprintf(f, "\"in2\":{\"leaves\":%u,\"active\":%llu},\"i2\":", h2.g()->tree().nodeCount(0), (unsigned long long)h2.g()->activeVoxelCount());
                rle(f, slice(h2)); fprintf(f, ",");
            }
            if (neg1) complement(s1);
            if (neg2) complement(s2);
            const HostState h1 = download(s1), h2 = download(s2);
            auto bb = h1.g()->indexBBox(); bb.expand(h2.g()->indexBBox());
            for (int o = 0; o < 3; ++o) {
                StateT r = o == 0 ? tools::cuda::csgUnion<BuildT, float, ResourceT>(s1, s2, 0, res)
                         : o == 1 ? tools::cuda::csgIntersection<BuildT, float, ResourceT>(s1, s2, 0, res)
                         : tools::cuda::csgDifference<BuildT, float, ResourceT>(s1, s2, 0, res);
                const HostState h = download(r);
                const auto rule = [&](const Coord& p) {
                    const float f1 = fieldOf(h1, p), f2 = fieldOf(h2, p);
                    const float e = o == 0 ? std::fmin(f1, f2) : o == 1 ? std::fmax(f1, f2) : std::fmax(f1, -f2);
                    ++checked; if (fieldOf(h, p) != e) ++ruleBad;
                };
                // the rule on every second voxel (in each direction) of a box around both inputs, and on the whole slice
                for (int i = bb.min()[0] - 4; i <= bb.max()[0] + 4; i += 2) for (int j = bb.min()[1] - 4; j <= bb.max()[1] + 4; j += 2) for (int k = bb.min()[2] - 4; k <= bb.max()[2] + 4; k += 2) rule(Coord(i, j, k));
                for (int j = 0; j < WV; ++j) for (int i = 0; i < WU; ++i) rule(at(i, j));
                fprintf(f, "\"%c%d%d\":{\"bg\":%g,\"leaves\":%u,\"lowers\":%u,\"uppers\":%u,\"active\":%llu,\"bits\":[%llu,%llu,%llu],\"v\":", "uid"[o], neg1, neg2, h.phi[0],
                        h.g()->tree().nodeCount(0), h.g()->tree().nodeCount(1), h.g()->tree().nodeCount(2), (unsigned long long)h.g()->activeVoxelCount(),
                        (unsigned long long)bitsOn(h.leaf), (unsigned long long)bitsOn(h.lower), (unsigned long long)bitsOn(h.upper));
                rle(f, slice(h)); fprintf(f, "},");
            }
        }
        fprintf(f, "\"checked\":%llu,\"ruleBad\":%llu}", (unsigned long long)checked, (unsigned long long)ruleBad);
        printf("frame %2d off %3d | 12 results | checked %llu rule mismatches %llu\n", fr, off, (unsigned long long)checked, (unsigned long long)ruleBad);
        allOk = allOk && !ruleBad;
    }
    fprintf(f, "\n]}\n"); fclose(f);
    printf(allOk ? "ALL FRAMES AGREE\n" : "DISAGREEMENT\n");
    return 0;
}
