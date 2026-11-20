// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file csg_union_baselines.cu
///
/// @brief csgUnion of tools/cuda/CSG.cuh against its two comparison baselines on pairs of analytic spheres:
///        csgUnionPropagateSign (MergeGrids, PruneGrid and PropagateSign rerun on the result; "A-a" below) and
///        csgUnionLeafResolved (the invert masks resolved to the leaf level; "A-c" below). csgUnion ("B") must
///        be byte-identical to A-a, and all three must match the per-voxel rule from the analytic distances.
///
/// usage: csg_union_baselines

#include "CsgExampleUtil.cuh"

using HandleT = decltype(StateT::grid);
// Resolve the invert masks of a level set to the leaf level: one active voxel per inactive voxel that takes the
// negative background, read from the masks through isInverted (brute force on the host, for the test only).
static HandleT makeInterior(const HostState& h, ResourceT& res, uint64_t& count)
{
    std::vector<Coord> coords;
    const auto bb = h.g()->indexBBox();
    for (int i = bb.min()[0]; i <= bb.max()[0]; ++i) for (int j = bb.min()[1]; j <= bb.max()[1]; ++j) for (int k = bb.min()[2]; k <= bb.max()[2]; ++k) {
        const Coord p(i, j, k);
        if (tools::cuda::levelset::detail::isInverted(*h.g(), p, h.leaf.data(), h.lower.data(), h.upper.data(), h.root.data(), h.root.size())) coords.push_back(p);
    }
    count = coords.size();
    nanovdb::cuda::ResourceRef<ResourceT> ref(res);
    BufT<Coord> dCoords(cudaStream_t(0), ref, coords.size(), nanovdb::cuda::noInit);
    cudaCheck(cudaMemcpy(dCoords.data(), coords.data(), coords.size() * sizeof(Coord), cudaMemcpyHostToDevice));
    return tools::cuda::voxelsToGrid<BuildT, Coord*, nanovdb::cuda::ResourceRef<ResourceT>>(dCoords.data(), coords.size(), 1.0, cudaStream_t(0), ref);
}
static std::vector<char> pullGrid(const HandleT& g)
{
    std::vector<char> v(g.buffer().size());
    cudaCheck(cudaMemcpy(v.data(), g.buffer().data(), v.size(), cudaMemcpyDeviceToHost));
    return v;
}
template<typename M> static uint64_t bitsOn(const std::vector<M>& v) { uint64_t n = 0; for (const auto& m : v) n += m.countOn(); return n; }

enum St { OUT, ACT, IN };
static St state(const Sphere& s, const Coord& p) { const float d = dist(s, p); return std::fabs(d) < W ? ACT : d < 0 ? IN : OUT; }

static bool runCase(const char* name, Sphere a, Sphere b, int stride, ResourceT& res, bool leafResolved = true, bool timing = false)
{
    StateT s1 = makeSphere(a, res), s2 = makeSphere(b, res);
    StateT u = tools::cuda::csgUnionPropagateSign<BuildT, float, ResourceT>(s1, s2, W, 0, res);
    const HostState h = download(u), h1 = download(s1), h2 = download(s2);
    const auto& tree = h.g()->tree();
    const auto& bb = h.g()->indexBBox();

    // route A-c: resolve the inputs' masks to the leaf level, then union by mask arithmetic
    std::vector<char> cGrid, cInterior; std::vector<float> cPhi;
    uint64_t in1 = 0, in2 = 0;
    if (leafResolved) {
        HandleT e1 = makeInterior(h1, res, in1), e2 = makeInterior(h2, res, in2);
        auto c = tools::cuda::csgUnionLeafResolved<BuildT, float, ResourceT>(
            s1.grid.template deviceGrid<BuildT>(), s1.phi.data(), e1.template deviceGrid<BuildT>(),
            s2.grid.template deviceGrid<BuildT>(), s2.phi.data(), e2.template deviceGrid<BuildT>(), W, 0, res);
        cGrid = pullGrid(c.grid); cInterior = pullGrid(c.interior); pull(cPhi, c.phi);
    }
    const auto* cg = leafResolved ? reinterpret_cast<const NanoGrid<BuildT>*>(cGrid.data()) : nullptr;
    const auto* ci = leafResolved ? reinterpret_cast<const NanoGrid<BuildT>*>(cInterior.data()) : nullptr;
    uint64_t badC = 0, diffAC = 0;

    // route B: one topology build, masks derived level by level
    bool okB;
    {
        StateT nb = tools::cuda::csgUnion<BuildT, float, ResourceT>(s1, s2, 0, res);
        const HostState hb = download(nb);
        const bool g = same(hb.grid, h.grid), p = same(hb.phi, h.phi), l0 = same(hb.leaf, h.leaf), l1 = same(hb.lower, h.lower), l2 = same(hb.upper, h.upper);
        printf("%-22s   B vs A-a: grid bytes %s | phi %s | leaf masks %s | lower masks %s | upper masks %s\n", name,
               g ? "identical" : "DIFFER", p ? "identical" : "DIFFER", l0 ? "identical" : "DIFFER", l1 ? "identical" : "DIFFER", l2 ? "identical" : "DIFFER");
        okB = g && p && l0 && l1 && l2;
    }
    if (timing) {
        timeIt("csgUnionPropagateSign", [&] { auto r = tools::cuda::csgUnionPropagateSign<BuildT, float, ResourceT>(s1, s2, W, 0, res); });
        timeIt("csgUnion", [&] { auto r = tools::cuda::csgUnion<BuildT, float, ResourceT>(s1, s2, 0, res); });
        if (leafResolved) {
            uint64_t n1, n2;
            HandleT e1 = makeInterior(h1, res, n1), e2 = makeInterior(h2, res, n2);
            timeIt("csgUnionLeafResolved, given the interior grids", [&] { auto r = tools::cuda::csgUnionLeafResolved<BuildT, float, ResourceT>(
                s1.grid.template deviceGrid<BuildT>(), s1.phi.data(), e1.template deviceGrid<BuildT>(),
                s2.grid.template deviceGrid<BuildT>(), s2.phi.data(), e2.template deviceGrid<BuildT>(), W, 0, res); });
        }
    }

    // every voxel of a box around both spheres
    Coord lo, hi;
    for (int c = 0; c < 3; ++c) {
        const float ca[3] = {a.cx, a.cy, a.cz}, cb[3] = {b.cx, b.cy, b.cz};
        lo[c] = int(std::floor(std::fmin(ca[c] - a.r, cb[c] - b.r))) - 12;
        hi[c] = int(std::ceil(std::fmax(ca[c] + a.r, cb[c] + b.r))) + 12;
    }
    uint64_t n = 0, bad = 0, nAct = 0, nIn = 0, nOut = 0, expAct = 0;
    double maxErr = 0;
    for (int i = lo[0]; i <= hi[0]; i += stride) for (int j = lo[1]; j <= hi[1]; j += stride) for (int k = lo[2]; k <= hi[2]; k += stride) {
        const Coord p(i, j, k);
        const St sa = state(a, p), sb = state(b, p);
        const St e = (sa == IN || sb == IN) ? IN : (sa == ACT || sb == ACT) ? ACT : OUT;
        const auto* leaf = tree.root().probeLeaf(p);
        const uint32_t off = NanoLeaf<BuildT>::CoordToOffset(p);
        const bool active = leaf && leaf->isActive(off);
        St got;
        if (active) {
            got = ACT;
            const float v = h.phi[leaf->getValue(off)], ev = std::fmin(sa == ACT ? dist(a, p) : W, sb == ACT ? dist(b, p) : W);
            if (e == ACT) maxErr = std::fmax(maxErr, std::fabs(double(v) - ev));
        } else {
            got = tools::cuda::levelset::detail::isInverted(*h.g(), p, h.leaf.data(), h.lower.data(), h.upper.data(), h.root.data(), h.root.size()) ? IN : OUT;
        }
        ++n; (got == ACT ? nAct : got == IN ? nIn : nOut)++; if (e == ACT) ++expAct;
        if (leafResolved) {
            const auto* cl = cg->tree().root().probeLeaf(p);
            const bool cActive = cl && cl->isActive(off);
            const auto* il = ci->tree().root().probeLeaf(p);
            const St gotC = cActive ? ACT : (il && il->isActive(off)) ? IN : OUT;
            if (gotC != e) { if (badC < 5) printf("  A-c MISMATCH at (%d,%d,%d): expected %d got %d\n", i, j, k, e, gotC); ++badC; }
            if (gotC != got || (cActive && active && cPhi[cl->getValue(off)] != h.phi[leaf->getValue(off)])) ++diffAC;
        }
        if (got != e) { if (bad < 5) printf("  MISMATCH at (%d,%d,%d): expected %d got %d\n", i, j, k, e, got); ++bad; }
    }
    // no empty leaf, and the stored voxel count
    const uint32_t leafCount = tree.nodeCount(0);
    uint32_t emptyLeaves = 0;
    for (uint32_t l = 0; l < leafCount; ++l) if (tree.template getFirstNode<0>()[l].valueMask().isOff()) ++emptyLeaves;
    printf("%-22s leaves in1 %u in2 %u union %u | active voxels in1 %llu in2 %llu union %llu%s | checked %llu (act %llu in %llu out %llu) mismatches %llu maxValueErr %.2e emptyLeaves %u rootOrigins %zu\n",
           name, h1.g()->tree().nodeCount(0), h2.g()->tree().nodeCount(0), leafCount,
           (unsigned long long)h1.g()->activeVoxelCount(), (unsigned long long)h2.g()->activeVoxelCount(), (unsigned long long)h.g()->activeVoxelCount(),
           stride == 1 ? (h.g()->activeVoxelCount() == expAct ? " (= expected)" : " (!= expected)") : "",
           (unsigned long long)n, (unsigned long long)nAct, (unsigned long long)nIn, (unsigned long long)nOut, (unsigned long long)bad, maxErr, emptyLeaves, h.root.size());
    (void)bb;
    printf("%-22s   A-a masks on: leaf %llu lower %llu upper %llu\n", "", (unsigned long long)bitsOn(h.leaf), (unsigned long long)bitsOn(h.lower), (unsigned long long)bitsOn(h.upper));
    if (leafResolved) {
        const bool sameGrid = cGrid.size() == h.grid.size() && std::memcmp(cGrid.data(), h.grid.data(), cGrid.size()) == 0;
        const bool samePhi = cPhi.size() == h.phi.size() && std::memcmp(cPhi.data(), h.phi.data(), cPhi.size() * sizeof(float)) == 0;
        printf("%-22s   A-c: interior voxels in1 %llu in2 %llu union %llu | mismatches vs rule %llu | points where A-a and A-c differ %llu | grid bytes identical %s | phi identical %s\n", "",
               (unsigned long long)in1, (unsigned long long)in2, (unsigned long long)ci->activeVoxelCount(), (unsigned long long)badC, (unsigned long long)diffAC, sameGrid ? "yes" : "NO", samePhi ? "yes" : "NO");
        if (badC || diffAC || !samePhi) return false;
    }
    return okB && bad == 0 && emptyLeaves == 0 && maxErr < 1e-4 && (stride != 1 || h.g()->activeVoxelCount() == expAct);
}

int main()
{
    ResourceT& res = nanovdb::cuda::default_resource<ResourceT>();
    bool ok = true;
    ok &= runCase("overlap",          {0.3f, 0.1f, -0.2f, 40}, {50.2f, 3.4f, 1.1f, 30}, 1, res);
    ok &= runCase("disjoint",         {0, 0, 0, 20},           {70.5f, 0, 0, 25},       1, res);
    ok &= runCase("contained",        {0, 0, 0, 60},           {5.5f, 4.2f, -3.1f, 20}, 1, res);
    ok &= runCase("touching bands",   {0, 0, 0, 30},           {64.0f, 0, 0, 30},       1, res);
    ok &= runCase("equal",            {1.5f, 2.5f, 3.5f, 33},  {1.5f, 2.5f, 3.5f, 33},  1, res);
    ok &= runCase("coarse interior",  {64, 64, 64, 120},       {190.3f, 70.1f, 64, 100}, 1, res, true, true);
    ok &= runCase("large (A-a only)", {10, 20, 30, 300},       {260.5f, 100.2f, 30, 250}, 1, res, false, true);
    printf(ok ? "ALL PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
