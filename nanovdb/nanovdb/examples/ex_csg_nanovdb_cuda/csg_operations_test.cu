// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file csg_operations_test.cu
///
/// @brief Checks csgUnion, csgIntersection, csgDifference and csgComplement of tools/cuda/CSG.cuh. On pairs of
///        analytic spheres, for every sign pair of the backgrounds: the full-space field of the result against the
///        per-voxel rule (min, max, max with the negated second field) and against the analytic distance, the
///        structure of the result, and the identities 1 - 2 = 1 & ~2 and 1 | 2 = ~(~1 & ~2). Then 17 set
///        identities on three inputs, each side built independently and compared byte for byte. With three band
///        files as arguments (see mesh_to_band.cu) the same checks run on mesh level sets.
///
/// usage: csg_operations_test [a.band b.band c.band]

#include "CsgExampleUtil.cuh"

static bool sameState(const HostState& a, const HostState& b)
{
    return same(a.grid, b.grid) && same(a.phi, b.phi) && same(a.leaf, b.leaf) && same(a.lower, b.lower) && same(a.upper, b.upper) && a.root.size() == b.root.size();
}

// The full-space field of a level set at p: the stored value, or the background with the sign told by the invert bit.
static float field(const HostState& h, const Coord& p)
{
    const auto* leaf = h.g()->tree().root().probeLeaf(p);
    const uint32_t off = NanoLeaf<BuildT>::CoordToOffset(p);
    if (leaf && leaf->isActive(off)) return h.phi[leaf->getValue(off)];
    const bool inv = tools::cuda::levelset::detail::isInverted(*h.g(), p, h.leaf.data(), h.lower.data(), h.upper.data(), h.root.data(), h.root.size());
    return inv ? -h.phi[0] : h.phi[0];
}

// The analytic full-space field of a sphere: its distance, saturated at +-W outside the band; negated if complemented.
static float analytic(const Sphere& s, bool neg, const Coord& p)
{
    const float d = dist(s, p), f = std::fabs(d) < W ? d : d < 0 ? -W : W;
    return neg ? -f : f;
}

enum Op { UNION, INTERSECTION, DIFFERENCE };
static const char* opName[] = {"union", "intersection", "difference"};
static float rule(Op op, float f1, float f2) { return op == UNION ? std::fmin(f1, f2) : op == INTERSECTION ? std::fmax(f1, f2) : std::fmax(f1, -f2); }
static StateT apply(Op op, const StateT& a, const StateT& b, ResourceT& res)
{
    return op == UNION ? tools::cuda::csgUnion<BuildT, float, ResourceT>(a, b, 0, res)
         : op == INTERSECTION ? tools::cuda::csgIntersection<BuildT, float, ResourceT>(a, b, 0, res)
         : tools::cuda::csgDifference<BuildT, float, ResourceT>(a, b, 0, res);
}
static void complement(StateT& s) { tools::cuda::csgComplement<BuildT, float, ResourceT>(s, 0); cudaCheck(cudaDeviceSynchronize()); }

// Structure of a stored level set: no node without content, and no invert bit where something is stored.
static uint64_t structureErrors(const HostState& h)
{
    const auto& tree = h.g()->tree();
    uint64_t bad = 0;
    for (uint32_t i = 0; i < tree.nodeCount(0); ++i) {
        const auto& n = tree.template getFirstNode<0>()[i];
        if (n.valueMask().isOff()) ++bad;
        for (uint32_t w = 0; w < Mask<3>::WORD_COUNT; ++w) if (n.valueMask().words()[w] & h.leaf[i].words()[w]) ++bad;
    }
    for (uint32_t i = 0; i < tree.nodeCount(1); ++i) {
        const auto& n = tree.template getFirstNode<1>()[i];
        if (n.childMask().isOff()) ++bad;
        for (uint32_t w = 0; w < Mask<4>::WORD_COUNT; ++w) if (n.childMask().words()[w] & h.lower[i].words()[w]) ++bad;
    }
    for (uint32_t i = 0; i < tree.nodeCount(2); ++i) {
        const auto& n = tree.template getFirstNode<2>()[i];
        if (n.childMask().isOff()) ++bad;
        for (uint32_t w = 0; w < Mask<5>::WORD_COUNT; ++w) if (n.childMask().words()[w] & h.upper[i].words()[w]) ++bad;
    }
    return bad;
}

static bool runCase(const char* name, Sphere a, Sphere b, ResourceT& res)
{
    bool ok = true;
    const Sphere sph[2] = {a, b};
    for (int neg1 = 0; neg1 < 2; ++neg1) for (int neg2 = 0; neg2 < 2; ++neg2) {
        StateT s1 = makeSphere(a, res), s2 = makeSphere(b, res);
        if (neg1) complement(s1);
        if (neg2) complement(s2);
        const HostState h1 = download(s1), h2 = download(s2);
        for (int o = 0; o < 3; ++o) {
            const Op op = Op(o);
            StateT r = apply(op, s1, s2, res);
            const HostState h = download(r);

            // 1. the field of the result against the rule, at every voxel of a box around each sphere and at a far point
            uint64_t n = 0, bad = 0, nAct = 0, nNeg = 0, badAnalytic = 0;
            double maxErr = 0;
            for (int s = 0; s < 2; ++s) {
                const int R = int(std::ceil(sph[s].r)) + 12;
                const Coord c(int(std::floor(sph[s].cx)), int(std::floor(sph[s].cy)), int(std::floor(sph[s].cz)));
                for (int i = -R; i <= R; ++i) for (int j = -R; j <= R; ++j) for (int k = -R; k <= R; ++k) {
                    const Coord p = c + Coord(i, j, k);
                    const float got = field(h, p), exp = rule(op, field(h1, p), field(h2, p));
                    ++n; if (got < 0) ++nNeg;
                    const auto* leaf = h.g()->tree().root().probeLeaf(p);
                    if (leaf && leaf->isActive(NanoLeaf<BuildT>::CoordToOffset(p))) ++nAct;
                    if (got != exp) { if (bad < 3) printf("  MISMATCH at (%d,%d,%d): expected %g got %g\n", p[0], p[1], p[2], exp, got); ++bad; }
                    // against the analytic spheres: the saturated values exactly, the band values to rounding
                    const float ana = rule(op, analytic(a, neg1, p), analytic(b, neg2, p));
                    const bool satGot = std::fabs(got) == W, satAna = std::fabs(ana) == W;
                    if (satGot != satAna || (satAna && got != ana)) { if (badAnalytic < 3) printf("  ANALYTIC MISMATCH at (%d,%d,%d): expected %g got %g\n", p[0], p[1], p[2], ana, got); ++badAnalytic; }
                    else maxErr = std::fmax(maxErr, std::fabs(double(got) - ana));
                }
            }
            const Coord far(100000, -200000, 300000);
            const bool farOk = field(h, far) == rule(op, field(h1, far), field(h2, far)) && h.phi[0] == field(h, far);

            // 2. the structure of the result
            const uint64_t structBad = structureErrors(h);

            // 3. identities, byte for byte: 1 \ 2 = 1 & ~2, and 1 | 2 = ~(~1 & ~2)
            bool identity = true;
            if (op == DIFFERENCE) {
                complement(s2);
                StateT x = apply(INTERSECTION, s1, s2, res);
                complement(s2);
                identity = sameState(h, download(x));
            } else if (op == UNION) {
                complement(s1); complement(s2);
                StateT x = apply(INTERSECTION, s1, s2, res);
                complement(s1); complement(s2);
                complement(x);
                identity = sameState(h, download(x));
            }
            const bool pass = bad == 0 && badAnalytic == 0 && maxErr < 1e-4 && farOk && structBad == 0 && identity && h.root.empty();
            printf("%-16s bg(%c,%c) %-12s result bg %+g | leaves %6u active %8llu | checked %9llu (active %8llu, negative %9llu) mismatches %llu | analytic mismatches %llu maxErr %.1e | far point %s | structure errors %llu | identity %s | %s\n",
                   name, neg1 ? '-' : '+', neg2 ? '-' : '+', opName[o], h.phi[0], h.g()->tree().nodeCount(0), (unsigned long long)h.g()->activeVoxelCount(),
                   (unsigned long long)n, (unsigned long long)nAct, (unsigned long long)nNeg, (unsigned long long)bad, (unsigned long long)badAnalytic, maxErr, farOk ? "ok" : "BAD",
                   (unsigned long long)structBad, op == INTERSECTION ? "n/a" : identity ? "identical" : "DIFFER", pass ? "pass" : "FAIL");
            ok &= pass;
        }
        // the inputs must be unchanged by all of the above
        ok &= sameState(h1, download(s1)) && sameState(h2, download(s2));
    }
    return ok;
}

// ---- set identities, each side built independently and compared byte for byte ----
static std::string diff(const HostState& x, const HostState& y)
{
    std::string d;
    if (!same(x.phi, y.phi)) d += " phi";
    if (!same(x.leaf, y.leaf)) d += " leafMasks";
    if (!same(x.lower, y.lower)) d += " lowerMasks";
    if (!same(x.upper, y.upper)) d += " upperMasks";
    if (x.root.size() != y.root.size()) d += " rootOrigins";
    if (!same(x.grid, y.grid)) d += " gridBytes";
    return d;
}
using Maker = std::function<StateT(int)>; // builds input 0, 1 or 2 with a positive background
struct Identities {
    ResourceT& res; Maker maker; bool neg[3]; bool ok = true; int count = 0;
    StateT make(int i) const { StateT s = maker(i); if (neg[i]) complement(s); return s; }
    StateT U(const StateT& x, const StateT& y) const { return apply(UNION, x, y, res); }
    StateT I(const StateT& x, const StateT& y) const { return apply(INTERSECTION, x, y, res); }
    StateT D(const StateT& x, const StateT& y) const { return apply(DIFFERENCE, x, y, res); }
    static StateT N(StateT x) { complement(x); return x; }
    void check(const char* name, const StateT& lhs, const StateT& rhs)
    {
        const HostState x = download(lhs), y = download(rhs);
        const std::string d = diff(x, y);
        const bool pass = d.empty() || d == " gridBytes"; // the grid header (name, checksum) may come from another input
        printf("  %-34s leaves %6u active %8llu bg %+g : %s%s\n", name, x.g()->tree().nodeCount(0), (unsigned long long)x.g()->activeVoxelCount(), x.phi[0],
               d.empty() ? "identical" : pass ? "identical (grid header bytes differ)" : "DIFFER in", pass ? "" : d.c_str());
        ok &= pass; ++count;
    }
    void run()
    {
        printf("identities, backgrounds (%c,%c,%c)\n", neg[0] ? '-' : '+', neg[1] ? '-' : '+', neg[2] ? '-' : '+');
        const StateT A = make(0), B = make(1), C = make(2);
        check("A | B = B | A", U(A, B), U(B, A));
        check("A & B = B & A", I(A, B), I(B, A));
        check("A | A = A", U(A, A), A);
        check("A & A = A", I(A, A), A);
        check("~~A = A", N(N(make(0))), A);
        check("(A | B) | C = A | (B | C)", U(U(A, B), C), U(A, U(B, C)));
        check("(A & B) & C = A & (B & C)", I(I(A, B), C), I(A, I(B, C)));
        check("A | (A & B) = A", U(A, I(A, B)), A);
        check("A & (A | B) = A", I(A, U(A, B)), A);
        check("A & (B | C) = (A & B) | (A & C)", I(A, U(B, C)), U(I(A, B), I(A, C)));
        check("A | (B & C) = (A | B) & (A | C)", U(A, I(B, C)), I(U(A, B), U(A, C)));
        check("~(A | B) = ~A & ~B", N(U(A, B)), I(N(make(0)), N(make(1))));
        check("~(A & B) = ~A | ~B", N(I(A, B)), U(N(make(0)), N(make(1))));
        check("A - B = A & ~B", D(A, B), I(A, N(make(1))));
        check("(A - B) - C = A - (B | C)", D(D(A, B), C), D(A, U(B, C)));
        check("A - (B & C) = (A - B) | (A - C)", D(A, I(B, C)), U(D(A, B), D(A, C)));
        check("A - (B | C) = (A - B) & (A - C)", D(A, U(B, C)), I(D(A, B), D(A, C)));
    }
};
static bool runIdentities(Maker maker, ResourceT& res)
{
    bool ok = true; int count = 0;
    for (int m = 0; m < 8; ++m) {
        Identities t{res, maker, {bool(m & 1), bool(m & 2), bool(m & 4)}};
        try { t.run(); } catch (const std::exception& e) { printf("  EXCEPTION: %s\n", e.what()); t.ok = false; }
        ok &= t.ok; count += t.count;
    }
    printf("identities checked: %d, %s\n", count, ok ? "all hold" : "SOME FAIL");
    return ok;
}

// ---- meshes ----

// Every operation and sign pair on two mesh level sets: the field of the result against the rule on the fields of
// the inputs, at every voxel of a box around both, and the structure of the result.
static bool runMeshCase(const char* name, const Band& A, const Band& B, const Coord& offB, ResourceT& res)
{
    bool ok = true;
    for (int neg1 = 0; neg1 < 2; ++neg1) for (int neg2 = 0; neg2 < 2; ++neg2) {
        StateT s1 = makeFromBand(A, Coord(0), res), s2 = makeFromBand(B, offB, res);
        if (neg1) complement(s1);
        if (neg2) complement(s2);
        const HostState h1 = download(s1), h2 = download(s2);
        nanovdb::CoordBBox box = h1.g()->indexBBox();
        box.expand(h2.g()->indexBBox());
        box.min() -= Coord(8); box.max() += Coord(8);
        for (int o = 0; o < 3; ++o) {
            const Op op = Op(o);
            StateT r = apply(op, s1, s2, res);
            const HostState h = download(r);
            uint64_t n = 0, bad = 0;
            for (int i = box.min()[0]; i <= box.max()[0]; ++i) for (int j = box.min()[1]; j <= box.max()[1]; ++j) for (int k = box.min()[2]; k <= box.max()[2]; ++k) {
                const Coord p(i, j, k);
                ++n; if (field(h, p) != rule(op, field(h1, p), field(h2, p))) ++bad;
            }
            const Coord far(100000, -200000, 300000);
            const bool farOk = field(h, far) == rule(op, field(h1, far), field(h2, far)) && h.phi[0] == field(h, far);
            const uint64_t structBad = structureErrors(h);
            const bool pass = bad == 0 && farOk && structBad == 0 && h.root.empty();
            printf("%-12s bg(%c,%c) %-12s result bg %+g | leaves %6u active %8llu | checked %9llu mismatches %llu | far point %s | structure errors %llu | %s\n",
                   name, neg1 ? '-' : '+', neg2 ? '-' : '+', opName[o], h.phi[0], h.g()->tree().nodeCount(0), (unsigned long long)h.g()->activeVoxelCount(),
                   (unsigned long long)n, (unsigned long long)bad, farOk ? "ok" : "BAD", (unsigned long long)structBad, pass ? "pass" : "FAIL");
            ok &= pass;
        }
    }
    return ok;
}

int main(int argc, char** argv)
{
    ResourceT& res = nanovdb::cuda::default_resource<ResourceT>();
    bool ok = true;
    ok &= runCase("overlap",         {0.3f, 0.1f, -0.2f, 40}, {50.2f, 3.4f, 1.1f, 30}, res);
    ok &= runCase("disjoint",        {0, 0, 0, 20},           {70.5f, 0, 0, 25},       res);
    ok &= runCase("contained",       {0, 0, 0, 60},           {5.5f, 4.2f, -3.1f, 20}, res);
    ok &= runCase("touching bands",  {0, 0, 0, 30},           {64.0f, 0, 0, 30},       res);
    ok &= runCase("equal",           {1.5f, 2.5f, 3.5f, 33},  {1.5f, 2.5f, 3.5f, 33},  res);
    ok &= runCase("far tiles",       {100, 100, 100, 30},     {5000.5f, 100, 100, 35}, res);
    ok &= runCase("coarse interior", {64, 64, 64, 120},       {190.3f, 70.1f, 64, 100}, res);
    const Sphere small[3] = {{0, 0, 0, 40}, {35.3f, 10.2f, 0, 35}, {15.1f, -30.4f, 5.5f, 30}};
    const Sphere large[3] = {{64, 64, 64, 120}, {190.3f, 70.1f, 64, 100}, {120.2f, 170.5f, 80, 90}};
    ok &= runIdentities([&](int i) { return makeSphere(small[i], res); }, res);
    ok &= runIdentities([&](int i) { return makeSphere(large[i], res); }, res);

    // empty level sets as operands: E = the intersection of two disjoint spheres
    {
        printf("empty operands\n");
        Identities t{res, [&](int i) { return makeSphere(i == 0 ? Sphere{0, 0, 0, 20} : Sphere{70.5f, 0, 0, 25}, res); }, {false, false, false}};
        try {
            const StateT A = t.make(0), B = t.make(1), E = t.I(A, B);
            t.check("E | E = E", t.U(E, E), E);
            t.check("E & E = E", t.I(E, E), E);
            t.check("E - E = E", t.D(E, E), E);
            t.check("A | E = A", t.U(A, E), A);
            t.check("E | A = A", t.U(E, A), A);
            t.check("A & E = E", t.I(A, E), E);
            t.check("A - E = A", t.D(A, E), A);
            t.check("E - A = E", t.D(E, A), E);
            t.check("~E & A = A", t.I(Identities::N(t.I(A, B)), A), A);
            t.check("~E | A = ~E", t.U(Identities::N(t.I(A, B)), A), Identities::N(t.I(A, B)));
        } catch (const std::exception& e) { printf("  EXCEPTION: %s\n", e.what()); t.ok = false; }
        ok &= t.ok;
    }

    // meshes: narrow bands of OpenVDB level sets, written by mesh_to_band; the second one at several offsets
    if (argc >= 4) {
        const Band bands[3] = {loadBand(argv[1]), loadBand(argv[2]), loadBand(argv[3])};
        for (int off : {210, 120, 60, 20, 0}) {
            char name[64]; snprintf(name, sizeof(name), "mesh z+%d", off);
            ok &= runMeshCase(name, bands[0], bands[1], Coord(0, 0, off), res);
        }
        const Coord offsets[3] = {Coord(0), Coord(0, 0, 60), Coord(0, 20, 30)};
        ok &= runIdentities([&](int i) { return makeFromBand(bands[i], offsets[i], res); }, res);
    }
    printf(ok ? "ALL PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
