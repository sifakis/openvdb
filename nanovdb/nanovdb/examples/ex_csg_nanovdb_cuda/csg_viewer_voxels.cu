// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file csg_viewer_voxels.cu
///
/// @brief Data for the 3D view of the CSG viewer (viewer/) and for figures: one mesh level set stays in place, a
///        second one moves toward it in steps. For every step the program writes, as JSON, every
///        stored voxel and invert bit of the two inputs and of four results on the plain inputs: 1 | 2, 1 & 2,
///        1 - 2 and 2 - 1. Each of the 12 results (three operations, four sign pairs of the backgrounds) is one
///        of these four or its complement; the program computes all 12 and checks that, byte for byte, and
///        checks each against the per-voxel rule. viewer/README.md describes the file.
///
/// usage: csg_viewer_voxels fixed.band moving.band out.json [options]   (band files: see mesh_to_band.cu)
///   --frames N      number of steps (default 15)
///   --move-axis A   x, y or z: the moving input is translated along +A (default z)
///   --start S       translation in the first step, in voxels; it shrinks linearly to 0 in the last step
///                   (default: the moving input starts one leaf beyond the fixed one)
/// Coordinates are written as 16-bit integers, so every voxel must lie within +-32767 of the origin.

#include "CsgExampleUtil.cuh"

template<typename M> static uint64_t bitsOn(const std::vector<M>& v) { uint64_t n = 0; for (const auto& m : v) n += m.countOn(); return n; }

static void b64(FILE* f, const std::vector<unsigned char>& v)
{
    static const char* T = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    fputc('"', f);
    for (size_t i = 0; i < v.size(); i += 3) {
        const unsigned n = (v[i] << 16) | ((i + 1 < v.size() ? v[i + 1] : 0) << 8) | (i + 2 < v.size() ? v[i + 2] : 0);
        fputc(T[(n >> 18) & 63], f); fputc(T[(n >> 12) & 63], f);
        fputc(i + 1 < v.size() ? T[(n >> 6) & 63] : '=', f); fputc(i + 2 < v.size() ? T[n & 63] : '=', f);
    }
    fputc('"', f);
}
static void putOrigin(std::vector<unsigned char>& v, const Coord& c)
{
    for (int k = 0; k < 3; ++k) { const int16_t x = int16_t(c[k]); v.push_back(x & 255); v.push_back((x >> 8) & 255); }
}
static void putMask(std::vector<unsigned char>& v, const uint64_t* words)
{
    for (int w = 0; w < 8; ++w) for (int b = 0; b < 8; ++b) v.push_back((words[w] >> (8 * b)) & 255);
}
// Every leaf of a result held as (grid, phi, invert masks): origin, value mask, mask of the active voxels with a
// negative value, invert mask.
static std::vector<unsigned char> packLeaves(const NanoGrid<BuildT>* g, const std::vector<float>& phi, const Mask<3>* invert)
{
    std::vector<unsigned char> v;
    const auto& tree = g->tree();
    for (uint32_t l = 0; l < tree.nodeCount(0); ++l) {
        const auto& leaf = tree.template getFirstNode<0>()[l];
        uint64_t neg[8] = {}, inv[8] = {};
        for (uint32_t n = 0; n < 512; ++n) if (leaf.isActive(n) && phi[leaf.getValue(n)] < 0) neg[n >> 6] |= uint64_t(1) << (n & 63);
        if (invert) for (int w = 0; w < 8; ++w) inv[w] = invert[l].words()[w];
        putOrigin(v, leaf.origin()); putMask(v, leaf.valueMask().words()); putMask(v, neg); putMask(v, inv);
    }
    return v;
}
// The origins of the childless slots of the LEVEL nodes whose invert bit is on.
template<int LEVEL, typename MaskT> static std::vector<unsigned char> packSlots(const NanoGrid<BuildT>* g, const std::vector<MaskT>& masks)
{
    std::vector<unsigned char> v;
    const auto& tree = g->tree();
    for (uint32_t i = 0; i < tree.nodeCount(LEVEL); ++i) {
        const auto& node = tree.template getFirstNode<LEVEL>()[i];
        for (uint32_t n = 0; n < MaskT::SIZE; ++n) if (masks[i].isOn(n)) putOrigin(v, node.offsetToGlobalCoord(n));
    }
    return v;
}

int main(int argc, char** argv)
{
    if (argc < 4) { printf("usage: %s fixed.band moving.band out.json [--frames N] [--move-axis x|y|z] [--start S]\n", argv[0]); return 1; }
    ResourceT& res = nanovdb::cuda::default_resource<ResourceT>();
    const Band bandA = loadBand(argv[1]), bandB = loadBand(argv[2]);

    // The scene: how the moving input travels, and the box that holds both inputs over all steps.
    const int FRAMES   = std::max(1, intOption(argc, argv, "--frames", 15));
    const int moveAxis = axisOption(argc, argv, "--move-axis", 2);
    const int START    = intOption(argc, argv, "--start", separatingOffset(bandA, bandB, moveAxis));
    const auto translation = [&](int off) { Coord t(0); t[moveAxis] = off; return t; };
    nanovdb::CoordBBox scene = bandBBox(bandA), boxB = bandBBox(bandB);
    scene.expand(boxB);
    Coord farthest = boxB.max(); farthest[moveAxis] += START;
    scene.expand(farthest);
    for (int a = 0; a < 3; ++a)
        if (scene.min()[a] < -32760 || scene.max()[a] > 32760) { printf("the scene does not fit 16-bit coordinates\n"); return 1; }
    const auto fieldOf = [](const HostState& h, const Coord& p) {
        const auto* leaf = h.g()->tree().root().probeLeaf(p);
        const uint32_t n = NanoLeaf<BuildT>::CoordToOffset(p);
        if (leaf && leaf->isActive(n)) return h.phi[leaf->getValue(n)];
        return tools::cuda::levelset::detail::isInverted(*h.g(), p, h.leaf.data(), h.lower.data(), h.upper.data(), h.root.data(), h.root.size()) ? -h.phi[0] : h.phi[0];
    };
    const auto complement = [](StateT& s) { tools::cuda::csgComplement<BuildT, float, ResourceT>(s, 0); cudaCheck(cudaDeviceSynchronize()); };
    const auto apply = [&](int o, const StateT& a, const StateT& b) {
        return o == 0 ? tools::cuda::csgUnion<BuildT, float, ResourceT>(a, b, 0, res)
             : o == 1 ? tools::cuda::csgIntersection<BuildT, float, ResourceT>(a, b, 0, res)
             : tools::cuda::csgDifference<BuildT, float, ResourceT>(a, b, 0, res);
    };
    FILE* f = fopen(argv[3], "w");
    const auto putTree = [&](const char* key, const HostState& h) {
        fprintf(f, "\"%s\":{\"leaves\":%u,\"lowers\":%u,\"uppers\":%u,\"active\":%llu,\"bits\":[%llu,%llu,%llu],\"t\":", key, h.g()->tree().nodeCount(0), h.g()->tree().nodeCount(1),
                h.g()->tree().nodeCount(2), (unsigned long long)h.g()->activeVoxelCount(), (unsigned long long)bitsOn(h.leaf), (unsigned long long)bitsOn(h.lower), (unsigned long long)bitsOn(h.upper));
        b64(f, packLeaves(h.g(), h.phi, h.leaf.data()));
        fprintf(f, ",\"l\":"); b64(f, packSlots<1>(h.g(), h.lower)); fprintf(f, ",\"u\":"); b64(f, packSlots<2>(h.g(), h.upper)); fprintf(f, "}");
    };
    fprintf(f, "{\"band\":%g,\"moveAxis\":%d,\"lo\":[%d,%d,%d],\"hi\":[%d,%d,%d],", W, moveAxis, scene.min()[0], scene.min()[1], scene.min()[2],
            scene.max()[0], scene.max()[1], scene.max()[2]);
    { StateT s1 = makeFromBand(bandA, Coord(0), res); putTree("i1", download(s1)); }
    fprintf(f, ",\"frames\":[\n");

    // Each of the 12 results (operation x sign pair) equals one of four results on the plain inputs, or its complement:
    // base 0 = 1 | 2, 1 = 1 & 2, 2 = 1 - 2, 3 = 2 - 1. Indexed [neg1][neg2][operation].
    static const int BASE[2][2][3] = {{{0, 1, 2}, {3, 2, 1}}, {{2, 3, 0}, {1, 0, 3}}};
    static const bool COMP[2][2][3] = {{{false, false, false}, {true, false, false}}, {{true, false, true}, {true, true, false}}};
    bool allOk = true;
    for (int fr = 0; fr < FRAMES; ++fr) {
        const int off = FRAMES > 1 ? int(std::lround(double(START) * (FRAMES - 1 - fr) / (FRAMES - 1))) : START;
        HostState base[4];
        {
            StateT s1 = makeFromBand(bandA, Coord(0), res), s2 = makeFromBand(bandB, translation(off), res);
            fprintf(f, "%s{\"off\":%d,", fr ? ",\n" : "", off);
            putTree("i2", download(s2)); fprintf(f, ",");
            base[0] = download(apply(0, s1, s2)); base[1] = download(apply(1, s1, s2)); base[2] = download(apply(2, s1, s2)); base[3] = download(apply(2, s2, s1));
        }
        putTree("u", base[0]); fprintf(f, ","); putTree("i", base[1]); fprintf(f, ","); putTree("d", base[2]); fprintf(f, ","); putTree("e", base[3]); fprintf(f, ",");

        uint64_t checked = 0, ruleBad = 0; int mapBad = 0;
        for (int neg1 = 0; neg1 < 2; ++neg1) for (int neg2 = 0; neg2 < 2; ++neg2) {
            StateT s1 = makeFromBand(bandA, Coord(0), res), s2 = makeFromBand(bandB, translation(off), res);
            if (neg1) complement(s1);
            if (neg2) complement(s2);
            const HostState h1 = download(s1), h2 = download(s2);
            auto bb = h1.g()->indexBBox(); bb.expand(h2.g()->indexBBox());
            for (int o = 0; o < 3; ++o) {
                const HostState h = download(apply(o, s1, s2));
                // byte for byte against the stored base result, or its complement (values negated, everything else equal)
                const HostState& b = base[BASE[neg1][neg2][o]];
                std::vector<float> phi = b.phi;
                if (COMP[neg1][neg2][o]) for (float& v : phi) v = -v;
                if (!(same(h.grid, b.grid) && same(h.phi, phi) && same(h.leaf, b.leaf) && same(h.lower, b.lower) && same(h.upper, b.upper) && h.root.empty())) ++mapBad;
                // the per-voxel rule on every voxel of a box around both inputs
                for (int i = bb.min()[0] - 4; i <= bb.max()[0] + 4; ++i) for (int j = bb.min()[1] - 4; j <= bb.max()[1] + 4; ++j) for (int k = bb.min()[2] - 4; k <= bb.max()[2] + 4; ++k) {
                    const Coord p(i, j, k);
                    const float f1 = fieldOf(h1, p), f2 = fieldOf(h2, p);
                    const float e = o == 0 ? std::fmin(f1, f2) : o == 1 ? std::fmax(f1, f2) : std::fmax(f1, -f2);
                    ++checked; if (fieldOf(h, p) != e) ++ruleBad;
                }
            }
        }
        fprintf(f, "\"checked\":%llu,\"ruleBad\":%llu,\"mapBad\":%d}", (unsigned long long)checked, (unsigned long long)ruleBad, mapBad);
        printf("frame %2d off %3d | leaves u %u i %u d %u e %u | 12 results: checked %llu rule mismatches %llu | results not equal to their base %d\n", fr, off,
               base[0].g()->tree().nodeCount(0), base[1].g()->tree().nodeCount(0), base[2].g()->tree().nodeCount(0), base[3].g()->tree().nodeCount(0),
               (unsigned long long)checked, (unsigned long long)ruleBad, mapBad);
        allOk = allOk && !ruleBad && !mapBad;
    }
    fprintf(f, "\n]}\n"); fclose(f);
    printf(allOk ? "ALL FRAMES AGREE\n" : "DISAGREEMENT\n");
    return 0;
}
