// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file hydra.cu
///
/// @brief A hydra: one body with the heads of other models attached, modeled from triangle meshes with
///        tools::cuda::MeshToSDF and the CSG operations of tools/cuda/CSG.cuh:
///
///            result = body | (model_1 & cutter_1) | (model_2 & cutter_2) | ...
///
///        body is the level set of the body mesh as it is. A head comes from another mesh: the scene file gives
///        that mesh, a cutter mesh (a closed surface that holds the head and the neck of the model and nothing else
///        of it) and the matrix that moves both to the place of the head on the body. Intersecting the moved model
///        with the moved cutter leaves a head on a neck; the neck ends in a cut that lies inside the body, so that
///        the union hides it. A level set cannot be moved without resampling its values, so every moved solid is
///        converted from its moved mesh: with N heads, 2 N + 1 MeshToSDF conversions, N intersections, N unions.
///
/// usage: hydra scene.txt longestExtentInVoxels outDir [--operands]
///        hydra --band mesh.obj longestExtentInVoxels out.band
///
///        scene.txt (plan_hydra.py writes one) has one line "body <mesh.obj>" and one line per head,
///            head <name> <mesh.obj> <cutter.obj> m00 m01 m02 m03 m10 m11 m12 m13 m20 m21 m22 m23
///        A point p of the head's mesh or cutter file goes to M (p, 1) in the coordinates of the body's file, M
///        being the 3 x 4 matrix of the line. File names are relative to the directory of scene.txt.
///        The body is centered in the grid and scaled so that the longest extent of its bounding box is
///        longestExtentInVoxels; the heads reach beyond that box.
///        Writes outDir/body.band, head_<name>.band per head and hydra.band (the format of ../mesh_to_band.cu),
///        outDir/hydra.inside (where the result is inside away from its active voxels, per tree level; see
///        writeInside) and outDir/scene.json (how the body was placed in the grid, and the heads' names and matrices, for programs
///        that draw the level sets), and prints the voxel counts, the time of the conversions and of the CSG
///        operations and the number of surface voxels of the result that lie on a cut.
///        --operands   also write the level sets that the intersections take: model_<name>.band and
///                     cutter_<name>.band
///        --band       only convert one mesh, centered and scaled like the body, and write its level set

#include "CsgExampleUtil.cuh"
#include <nanovdb/tools/cuda/MeshToSDF.cuh>
#include <array>
#include <fstream>
#include <memory>
#include <sstream>

struct Mesh { std::vector<nanovdb::Vec3f> points; std::vector<nanovdb::Vec3i> triangles; };

/// @brief The triangles of an OBJ file (polygons are split into fans), in the coordinates of the file.
static Mesh loadObj(const std::string& path)
{
    Mesh m; std::ifstream in(path); std::string line;
    if (!in) { printf("cannot open %s\n", path.c_str()); exit(1); }
    while (std::getline(in, line)) {
        std::istringstream ss(line); std::string tag; ss >> tag;
        if (tag == "v") { float x, y, z; ss >> x >> y >> z; m.points.emplace_back(x, y, z); }
        else if (tag == "f") {
            std::vector<int> idx; std::string tok;
            while (ss >> tok) { int i = std::stoi(tok.substr(0, tok.find('/'))); idx.push_back(i < 0 ? int(m.points.size()) + i : i - 1); }
            for (size_t k = 2; k < idx.size(); ++k) m.triangles.emplace_back(idx[0], idx[k - 1], idx[k]);
        }
    }
    return m;
}

using Matrix = std::array<double, 12>; // 3 x 4, row by row
/// @brief The mesh moved by @a m and then placed in the grid: a point p goes to (M (p, 1) - middle) * scale.
static Mesh placed(const Mesh& in, const Matrix& m, const nanovdb::Vec3f& middle, float scale)
{
    Mesh out = in;
    for (auto& p : out.points) {
        const double q[3] = {p[0], p[1], p[2]};
        for (int i = 0; i < 3; ++i) p[i] = float((m[4 * i] * q[0] + m[4 * i + 1] * q[1] + m[4 * i + 2] * q[2] + m[4 * i + 3] - middle[i]) * scale);
    }
    return out;
}

static double gMeshToSdfMs = 0, gCsgMs = 0; // wall time of the MeshToSDF conversions and of the CSG operations so far
template<typename F> static auto timed(double& total, F f)
{
    cudaCheck(cudaDeviceSynchronize());
    const auto t0 = std::chrono::steady_clock::now();
    auto result = f();
    cudaCheck(cudaDeviceSynchronize());
    total += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return result;
}

/// @brief The narrow-band level set of a closed triangle mesh (points in voxels), half-width W. MeshToSDF runs on
///        the device; its voxels closer than W to the surface then go through a host copy into a
///        LevelSetTracker::State, the form the CSG operations take.
static StateT meshToLevelSet(const Mesh& m, ResourceT& res)
{
    nanovdb::cuda::ResourceRef<ResourceT> ref(res);
    BufT<nanovdb::Vec3f> dPts(cudaStream_t(0), ref, m.points.size(), nanovdb::cuda::noInit);
    BufT<nanovdb::Vec3i> dTris(cudaStream_t(0), ref, m.triangles.size(), nanovdb::cuda::noInit);
    cudaCheck(cudaMemcpy(dPts.data(), m.points.data(), m.points.size() * sizeof(nanovdb::Vec3f), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dTris.data(), m.triangles.data(), m.triangles.size() * sizeof(nanovdb::Vec3i), cudaMemcpyHostToDevice));
    auto converted = timed(gMeshToSdfMs, [&] {
        tools::cuda::MeshToSDF<BuildT, ResourceT> converter(dPts.data(), uint32_t(m.points.size()), dTris.data(), uint32_t(m.triangles.size()), nanovdb::Map(), 0, res);
        converter.setNarrowBandWidth(W);
        return converter.getHandleAndSDF();
    });
    auto& [handle, sdfBytes] = converted;
    std::vector<char> hostGrid(handle.buffer().size());
    cudaCheck(cudaMemcpy(hostGrid.data(), handle.buffer().data(), hostGrid.size(), cudaMemcpyDeviceToHost));
    std::vector<float> sdf(sdfBytes.size() / sizeof(float));
    cudaCheck(cudaMemcpy(sdf.data(), sdfBytes.data(), sdf.size() * sizeof(float), cudaMemcpyDeviceToHost));
    const auto& tree = reinterpret_cast<const NanoGrid<BuildT>*>(hostGrid.data())->tree();
    Band band;
    for (uint32_t l = 0; l < tree.nodeCount(0); ++l) {
        const auto& leaf = tree.template getFirstNode<0>()[l];
        for (uint32_t i = 0; i < 512; ++i) {
            if (!leaf.isActive(i)) continue;
            const float v = sdf[leaf.getValue(i)];
            if (std::fabs(v) < W) { band.coords.push_back(leaf.offsetToGlobalCoord(i)); band.values.push_back(v); }
        }
    }
    return makeFromBand(band, Coord(0), res);
}

/// @brief The value of a host level set at an active voxel; false if the voxel is not active.
static bool activeValue(const HostState& h, const Coord& p, float& v)
{
    const auto* leaf = h.g()->tree().root().probeLeaf(p);
    const uint32_t o = NanoLeaf<BuildT>::CoordToOffset(p);
    if (!leaf || !leaf->isActive(o)) return false;
    v = h.phi[leaf->getValue(o)];
    return true;
}

/// @brief Writes the active voxels of a level set as a band file and returns their number.
static uint64_t writeBand(const HostState& h, const std::string& path)
{
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { printf("cannot write %s\n", path.c_str()); exit(1); }
    const uint64_t n = h.g()->activeVoxelCount();
    fwrite(&n, 8, 1, f);
    const auto& tree = h.g()->tree();
    for (uint32_t l = 0; l < tree.nodeCount(0); ++l) {
        const auto& leaf = tree.template getFirstNode<0>()[l];
        for (uint32_t o = 0; o < 512; ++o) {
            if (!leaf.isActive(o)) continue;
            const Coord c = leaf.offsetToGlobalCoord(o);
            const int32_t ijk[3] = {c[0], c[1], c[2]};
            fwrite(ijk, 4, 3, f);
            fwrite(&h.phi[leaf.getValue(o)], 4, 1, f);
        }
    }
    fclose(f);
    return n;
}

/// @brief Writes where a level set is inside away from its active voxels, tree level by tree level, as the level
///        set itself holds it (the invert masks of PropagateSign that a LevelSetTracker::State carries): four
///        lists, each a uint64 count and then int32 x, y, z per entry, of the origins of
///            the inactive voxels of leaves that are inside,
///            the slots without a child of lower nodes that are inside (each 8 voxels wide),
///            the slots without a child of upper nodes that are inside (each 128 voxels wide),
///            the root-level regions that are inside (each 4096 voxels wide).
///        An invert bit, or a root origin, marks the side opposite to the one at infinity, which is the sign of
///        the background. Returns the number of entries of each list.
static std::array<uint64_t, 4> writeInside(const HostState& h, const std::string& path)
{
    const bool insideAtInfinity = h.phi[0] < 0;
    const auto& tree = h.g()->tree();
    std::vector<int32_t> lists[4];
    auto add = [&](int level, const Coord& c) { lists[level].push_back(c[0]); lists[level].push_back(c[1]); lists[level].push_back(c[2]); };
    for (uint32_t n = 0; n < tree.nodeCount(0); ++n) {
        const auto& leaf = tree.template getFirstNode<0>()[n];
        for (uint32_t o = 0; o < 512; ++o) if (!leaf.isActive(o) && h.leaf[n].isOn(o) != insideAtInfinity) add(0, leaf.offsetToGlobalCoord(o));
    }
    for (uint32_t n = 0; n < tree.nodeCount(1); ++n) {
        const auto& lower = tree.template getFirstNode<1>()[n];
        for (uint32_t o = 0; o < 4096; ++o) if (!lower.isChild(o) && h.lower[n].isOn(o) != insideAtInfinity) add(1, lower.offsetToGlobalCoord(o));
    }
    for (uint32_t n = 0; n < tree.nodeCount(2); ++n) {
        const auto& upper = tree.template getFirstNode<2>()[n];
        for (uint32_t o = 0; o < 32768; ++o) if (!upper.isChild(o) && h.upper[n].isOn(o) != insideAtInfinity) add(2, upper.offsetToGlobalCoord(o));
    }
    if (!insideAtInfinity) for (const Coord& c : h.root) add(3, c); // with the inside at infinity the inside root regions are all the others
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { printf("cannot write %s\n", path.c_str()); exit(1); }
    std::array<uint64_t, 4> counts;
    for (int level = 0; level < 4; ++level) {
        counts[level] = lists[level].size() / 3;
        fwrite(&counts[level], 8, 1, f);
        fwrite(lists[level].data(), 4, lists[level].size(), f);
    }
    fclose(f);
    return counts;
}

/// @brief The middle of the bounding box of @a m and the factor that makes its longest extent @a longestExtent.
static void placement(const Mesh& m, float longestExtent, nanovdb::Vec3f& middle, float& scale)
{
    nanovdb::Vec3f lo(1e30f), hi(-1e30f);
    for (auto& p : m.points) for (int c = 0; c < 3; ++c) { lo[c] = std::min(lo[c], p[c]); hi[c] = std::max(hi[c], p[c]); }
    const nanovdb::Vec3f ext = hi - lo;
    middle = (hi + lo) * 0.5f;
    scale = longestExtent / std::max(ext[0], std::max(ext[1], ext[2]));
}

struct Head { std::string name, mesh, cutter; Matrix matrix; };

int main(int argc, char** argv)
{
    const Matrix identity = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    ResourceT& res = nanovdb::cuda::default_resource<ResourceT>();
    nanovdb::Vec3f middle; float scale;
    if (argc == 5 && std::strcmp(argv[1], "--band") == 0) {
        const Mesh mesh = loadObj(argv[2]);
        placement(mesh, std::stof(argv[3]), middle, scale);
        printf("%s: %llu active voxels\n", argv[2], (unsigned long long)writeBand(download(meshToLevelSet(placed(mesh, identity, middle, scale), res)), argv[4]));
        return 0;
    }
    if (argc < 4) { printf("usage: %s scene.txt longestExtentInVoxels outDir [--operands]\n       %s --band mesh.obj longestExtentInVoxels out.band\n", argv[0], argv[0]); return 1; }
    const float extent = std::stof(argv[2]);
    const std::string scenePath = argv[1], outDir = argv[3];
    const std::string sceneDir = scenePath.find('/') == std::string::npos ? std::string(".") : scenePath.substr(0, scenePath.rfind('/'));
    bool writeOperands = false;
    for (int i = 4; i < argc; ++i) if (std::strcmp(argv[i], "--operands") == 0) writeOperands = true;

    std::string bodyPath; std::vector<Head> heads;
    {
        auto resolved = [&](const std::string& p) { return p[0] == '/' ? p : sceneDir + "/" + p; };
        std::ifstream in(scenePath); std::string line;
        if (!in) { printf("cannot open %s\n", scenePath.c_str()); return 1; }
        while (std::getline(in, line)) {
            std::istringstream ss(line); std::string tag; ss >> tag;
            if (tag == "body") { ss >> bodyPath; bodyPath = resolved(bodyPath); }
            else if (tag == "head") {
                Head h; ss >> h.name >> h.mesh >> h.cutter;
                for (double& v : h.matrix) ss >> v;
                if (!ss) { printf("%s: a head line takes a name, two files and 12 numbers\n", scenePath.c_str()); return 1; }
                h.mesh = resolved(h.mesh); h.cutter = resolved(h.cutter);
                heads.push_back(h);
            }
        }
        if (bodyPath.empty() || heads.empty()) { printf("%s: no body or no head\n", scenePath.c_str()); return 1; }
    }

    const Mesh bodyMesh = loadObj(bodyPath);
    placement(bodyMesh, extent, middle, scale);
    {
        FILE* f = fopen((outDir + "/scene.json").c_str(), "w");
        if (!f) { printf("cannot write %s/scene.json\n", outDir.c_str()); return 1; }
        // A point p of the body's file is at (p - middle) * scale in the grid (lengths in voxels); a point p of a
        // head's mesh or cutter file is at (matrix (p, 1) - middle) * scale, matrix being 3 x 4, row by row.
        fprintf(f, "{\"extent\": %.9g, \"middle\": [%.9g, %.9g, %.9g], \"scale\": %.9g, \"halfWidth\": %.9g,\n \"body\": \"%s\",\n \"heads\": [", extent, middle[0], middle[1], middle[2], scale, W, bodyPath.c_str());
        for (size_t k = 0; k < heads.size(); ++k) {
            fprintf(f, "%s\n  {\"name\": \"%s\", \"mesh\": \"%s\", \"cutter\": \"%s\", \"matrix\": [", k ? "," : "", heads[k].name.c_str(), heads[k].mesh.c_str(), heads[k].cutter.c_str());
            for (int i = 0; i < 12; ++i) fprintf(f, "%s%.17g", i ? ", " : "", heads[k].matrix[i]);
            fprintf(f, "]}");
        }
        fprintf(f, "]}\n");
        fclose(f);
    }
    printf("%s: %zu vertices, %zu triangles, longest extent %g voxels, band half-width %g voxels\n", bodyPath.c_str(), bodyMesh.points.size(), bodyMesh.triangles.size(), extent, W);

    const StateT body = meshToLevelSet(placed(bodyMesh, identity, middle, scale), res);
    const HostState hostBody = download(body);
    printf("%-22s %10llu active voxels\n", "body", (unsigned long long)writeBand(hostBody, outDir + "/body.band"));

    // A head: the moved model intersected with the moved cutter. The result grows by one union per head.
    std::unique_ptr<StateT> result;
    std::vector<HostState> hostCutters;
    for (const Head& h : heads) {
        const Mesh headMesh = loadObj(h.mesh);
        const StateT model = meshToLevelSet(placed(headMesh, h.matrix, middle, scale), res), cutter = meshToLevelSet(placed(loadObj(h.cutter), h.matrix, middle, scale), res);
        const StateT head = timed(gCsgMs, [&] { return tools::cuda::csgIntersection<BuildT, float, ResourceT>(model, cutter, 0, res); });
        hostCutters.push_back(download(cutter));
        const HostState hostModel = download(model);
        if (writeOperands) {
            writeBand(hostModel, outDir + "/model_" + h.name + ".band");
            writeBand(hostCutters.back(), outDir + "/cutter_" + h.name + ".band");
        }
        printf("%-22s %10llu active voxels   (moved model %llu, moved cutter %llu; mesh %zu triangles)\n", ("head " + h.name).c_str(),
               (unsigned long long)writeBand(download(head), outDir + "/head_" + h.name + ".band"), (unsigned long long)hostModel.g()->activeVoxelCount(), (unsigned long long)hostCutters.back().g()->activeVoxelCount(), headMesh.triangles.size());
        result = std::make_unique<StateT>(timed(gCsgMs, [&] { return tools::cuda::csgUnion<BuildT, float, ResourceT>(result ? *result : body, head, 0, res); }));
    }
    const HostState hostResult = download(*result);
    printf("%-22s %10llu active voxels\n", "hydra", (unsigned long long)writeBand(hostResult, outDir + "/hydra.band"));
    const std::array<uint64_t, 4> inside = writeInside(hostResult, outDir + "/hydra.inside");
    printf("inside the hydra, off its active voxels: %llu voxels of leaves, %llu slots of lower nodes (8 voxels wide), %llu slots of upper nodes (128 voxels wide), %llu root regions\n",
           (unsigned long long)inside[0], (unsigned long long)inside[1], (unsigned long long)inside[2], (unsigned long long)inside[3]);
    printf("time: %zu MeshToSDF conversions %.1f ms, %zu intersections and %zu unions %.1f ms\n", 2 * heads.size() + 1, gMeshToSdfMs, heads.size(), heads.size(), gCsgMs);

    // A cut that shows: a voxel within one voxel of the result's surface whose value is the value of a cutter
    // there. (Intersection and union copy the value of one input, so the comparison is exact.)
    uint64_t surface = 0, onCut = 0;
    const auto& tree = hostResult.g()->tree();
    for (uint32_t l = 0; l < tree.nodeCount(0); ++l) {
        const auto& leaf = tree.template getFirstNode<0>()[l];
        for (uint32_t o = 0; o < 512; ++o) {
            if (!leaf.isActive(o)) continue;
            const float v = hostResult.phi[leaf.getValue(o)];
            if (!(std::fabs(v) < 1.0f)) continue;
            ++surface;
            const Coord c = leaf.offsetToGlobalCoord(o);
            float d, u;
            if (activeValue(hostBody, c, d) && d == v) continue; // the body's own surface
            for (const HostState& cutter : hostCutters) if (activeValue(cutter, c, u) && u == v) { ++onCut; break; }
        }
    }
    printf("surface voxels (|value| < 1) %llu, of them on a cut %llu\n", (unsigned long long)surface, (unsigned long long)onCut);
    return 0;
}
