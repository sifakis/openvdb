// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file  shrink_wrap.cu
///
/// @brief Compare the offset stage of OpenVDB's shrink wrap against the NanoVDB pipeline.
///
/// @details tools::PolySoupToLevelSet::offset(dx) turns a polygon soup into the signed distance
///          field of a surface standing dx out from it, and the shrink wrap algorithm calls it once
///          per resolution before eroding anything. It is the only stage of shrink wrap this
///          pipeline replaces, so it is the one to compare first: an error here would be invisible
///          under the erosion loop that follows.
///
///          OpenVDB reaches that field through a mesh round trip -- distance field, contour it at
///          dx, then re-sign the contoured mesh -- which is what the NanoVDB path exists to avoid.
///          MeshToSDF signs { udf == dx } in place instead, never building the intermediate mesh.
///          Both are asked for the same surface, so their zero crossings should agree.
///
///          Usage:  ex_shrink_wrap <mesh.obj> [voxelSize] [halfWidth]
///                  The offset is one voxel, which is what shrink wrap uses.
///
///          Everything OpenVDB-side runs on the host and everything NanoVDB-side on the device, so
///          the mesh is uploaded once and the result brought back as a FloatGrid. That hand-off is
///          the "package it back in OpenVDB territory" step, and it is deliberately kept here rather
///          than in the library until the comparison says the field is worth exposing.

#include <nanovdb/NanoVDB.h>
#include <nanovdb/GridHandle.h>
#include <nanovdb/tools/cuda/MeshToSDF.cuh>
#include <nanovdb/util/cuda/Util.h>
#include <nanovdb/util/cuda/DeviceGridTraits.cuh>

#include <openvdb/openvdb.h>
#include <openvdb/tools/PolySoupToLevelSet.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <openvdb/tools/Composite.h>       // csgUnionCopy
#include <openvdb/tools/GridTransformer.h> // resampleToMatch
#include <openvdb/tools/Interpolation.h>  // BoxSampler
#include <openvdb/tools/LevelSetFilter.h>
#include <openvdb/tools/LevelSetMeasure.h> // levelSetVolume

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <functional>
#include <vector>

using MeshToSDFT = nanovdb::tools::cuda::MeshToSDF<nanovdb::ValueOnIndex>;

// ---------------------------------------------------------------------------------------------------
/// @brief Minimal Wavefront .obj reader. Faces of any arity are fan-triangulated, since MeshToSDF
///        takes triangles only while PolySoup carries quads separately.
static void readOBJ(const std::string& path, std::vector<openvdb::Vec3s>& vtx,
                    std::vector<openvdb::Vec3I>& tri)
{
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot open " + path);

    std::string line;
    while (std::getline(file, line)) {
        if (line.size() < 2) continue;
        std::istringstream is(line);
        std::string        tag;
        is >> tag;
        if (tag == "v") {
            float x, y, z;
            is >> x >> y >> z;
            vtx.emplace_back(x, y, z);
        } else if (tag == "f") {
            std::vector<uint32_t> idx;
            std::string           tok;
            while (is >> tok) {                       // "v", "v/vt", "v//vn" or "v/vt/vn"
                idx.push_back(uint32_t(std::stoi(tok.substr(0, tok.find('/'))) - 1));
            }
            for (std::size_t k = 1; k + 1 < idx.size(); ++k)
                tri.emplace_back(idx[0], idx[k], idx[k + 1]);
        }
    }
    if (vtx.empty() || tri.empty()) throw std::runtime_error("no geometry in " + path);
}

// ---------------------------------------------------------------------------------------------------
/// @brief Bake a finished MeshToSDF into an ordinary FloatGrid.
///
/// @details The pipeline returns an index grid plus sidecars: a distance per active voxel, a sign per
///          active voxel, and an invert mask per tree level carrying the sign of everything the band
///          does not reach. A FloatGrid has one value per voxel and nothing else, so each sidecar has
///          to be written out as values. NanoVDB and OpenVDB share the 5-4-3 tree layout, which is
///          what lets the tiles land on the same nodes.
static openvdb::FloatGrid::Ptr toFloatGrid(const MeshToSDFT& sdf, const std::string& name)
{
    using BuildT = nanovdb::ValueOnIndex;
    using Traits = nanovdb::util::cuda::DeviceGridTraits<BuildT>;

    const double voxelSize  = sdf.map().getVoxelSize()[0];
    const float  background = float(sdf.narrowBandWidth() * voxelSize);

    auto grid = openvdb::FloatGrid::create(background);
    grid->setName(name);
    grid->setGridClass(openvdb::GRID_LEVEL_SET);
    grid->setTransform(openvdb::math::Transform::createLinearTransform(voxelSize));

    const auto&    handle    = sdf.gridHandle();
    const uint64_t gridBytes = handle.gridSize();
    void*          blob      = nullptr;
    cudaCheck(cudaMallocHost(&blob, gridBytes));
    cudaCheck(cudaMemcpy(blob, handle.deviceData(), gridBytes, cudaMemcpyDeviceToHost));
    const auto* h_grid = reinterpret_cast<const nanovdb::NanoGrid<BuildT>*>(blob);

    const uint32_t leafCount  = h_grid->tree().nodeCount(0);
    const uint32_t lowerCount = h_grid->tree().nodeCount(1);
    const uint32_t upperCount = h_grid->tree().nodeCount(2);
    const uint64_t active     = Traits::getActiveVoxelCount(sdf.deviceGrid());

    std::vector<float>            udf(active + 1);
    std::vector<int8_t>           sign(active + 1);
    std::vector<nanovdb::Mask<3>> leafInv(leafCount);
    std::vector<nanovdb::Mask<4>> lowInv(lowerCount);
    std::vector<nanovdb::Mask<5>> upInv(upperCount);
    cudaCheck(cudaMemcpy(udf.data(),  sdf.deviceUDF(),  (active + 1) * sizeof(float),  cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(sign.data(), sdf.deviceSign(), (active + 1) * sizeof(int8_t), cudaMemcpyDeviceToHost));
    if (leafCount)  cudaCheck(cudaMemcpy(leafInv.data(), sdf.deviceLeafInvertMask(),
                              std::size_t(leafCount) * sizeof(nanovdb::Mask<3>), cudaMemcpyDeviceToHost));
    if (lowerCount) cudaCheck(cudaMemcpy(lowInv.data(), sdf.deviceLowerInvertMask(),
                              std::size_t(lowerCount) * sizeof(nanovdb::Mask<4>), cudaMemcpyDeviceToHost));
    if (upperCount) cudaCheck(cudaMemcpy(upInv.data(), sdf.deviceUpperInvertMask(),
                              std::size_t(upperCount) * sizeof(nanovdb::Mask<5>), cudaMemcpyDeviceToHost));

    auto acc = grid->getAccessor();

    // Leaves: active voxels carry their signed distance; the inactive ones in the same leaf get a
    // saturated background whose sign comes from the invert bit. Writing those is what stops a
    // contouring pass from seeing a crossing at the band's inner edge.
    const auto* leaves = h_grid->tree().getFirstLeaf();
    for (uint32_t li = 0; li < leafCount; ++li) {
        const auto&          leaf = leaves[li];
        const nanovdb::Coord o    = leaf.origin();
        for (uint32_t n = 0; n < 512; ++n) {
            const nanovdb::Coord ijk = o + nanovdb::NanoLeaf<BuildT>::OffsetToLocalCoord(n);
            const openvdb::Coord c(ijk[0], ijk[1], ijk[2]);
            if (leaf.isActive(n)) {
                const uint64_t slot = leaf.getValue(n);
                acc.setValue(c, float(sign[slot]) * udf[slot]);
            } else {
                acc.setValueOff(c, leafInv[li].isOn(n) ? -background : background);
            }
        }
    }

    // Childless lower and upper slots: one inactive tile each. The surface cannot cross a childless
    // slot -- it would have forced refinement -- so one value describes all of it.
    auto fillTiles = [&](const auto* nodes, uint32_t count, const auto* inv, int slots, int level) {
        for (uint32_t ni = 0; ni < count; ++ni) {
            const auto& node = nodes[ni];
            for (int n = 0; n < slots; ++n) {
                if (node.childMask().isOn(uint32_t(n))) continue;   // refined: a level down handles it
                const nanovdb::Coord g = node.offsetToGlobalCoord(uint32_t(n));
                acc.addTile(level, openvdb::Coord(g[0], g[1], g[2]),
                            inv[ni].isOn(uint32_t(n)) ? -background : background, /*active=*/false);
            }
        }
    };
    fillTiles(h_grid->tree().template getFirstNode<1>(), lowerCount, lowInv.data(), 4096,  1);
    fillTiles(h_grid->tree().template getFirstNode<2>(), upperCount, upInv.data(), 32768, 2);

    // Regions with no node at all. The root sidecar covers a small cell array over the grid's root
    // range; a cell that is ON is deep interior, and everything outside it keeps +background.
    const nanovdb::Coord tileMin = sdf.rootTileMin(), dims = sdf.rootTileDims();
    const std::size_t    cells   = std::size_t(dims[0]) * dims[1] * dims[2];
    if (cells) {
        std::vector<uint8_t> rootInterior(cells);
        cudaCheck(cudaMemcpy(rootInterior.data(), sdf.deviceRootInterior(), cells, cudaMemcpyDeviceToHost));
        for (int i = 0; i < dims[0]; ++i)
        for (int j = 0; j < dims[1]; ++j)
        for (int k = 0; k < dims[2]; ++k) {
            if (!rootInterior[(std::size_t(i) * dims[1] + j) * dims[2] + k]) continue;
            const openvdb::Coord c((tileMin[0] + i) << 12, (tileMin[1] + j) << 12, (tileMin[2] + k) << 12);
            if (grid->tree().probeConstLeaf(c)) continue;           // a node already covers it
            acc.addTile(3, c, -background, /*active=*/false);
        }
    }

    cudaCheck(cudaFreeHost(blob));
    return grid;
}

// ---------------------------------------------------------------------------------------------------
/// @brief Run the NanoVDB pipeline for the same surface offset() is asked for: { udf == isoValue }.
static openvdb::FloatGrid::Ptr nanovdbOffset(const std::vector<openvdb::Vec3s>& vtx,
                                             const std::vector<openvdb::Vec3I>& tri,
                                             float dx, float halfWidth)
{
    // openvdb::Vec3s and nanovdb::Vec3f are both three floats, but reinterpreting one array as the
    // other is a promise about layout that neither header makes. Copy instead; it is once per run.
    std::vector<nanovdb::Vec3f> pts(vtx.size());
    std::vector<nanovdb::Vec3i> tris(tri.size());
    for (std::size_t i = 0; i < vtx.size(); ++i) pts[i]  = nanovdb::Vec3f(vtx[i][0], vtx[i][1], vtx[i][2]);
    for (std::size_t i = 0; i < tri.size(); ++i) tris[i] = nanovdb::Vec3i(int(tri[i][0]), int(tri[i][1]), int(tri[i][2]));

    nanovdb::Vec3f* d_pts  = nullptr;
    nanovdb::Vec3i* d_tris = nullptr;
    cudaCheck(cudaMalloc(&d_pts,  pts.size()  * sizeof(nanovdb::Vec3f)));
    cudaCheck(cudaMalloc(&d_tris, tris.size() * sizeof(nanovdb::Vec3i)));
    cudaCheck(cudaMemcpy(d_pts,  pts.data(),  pts.size()  * sizeof(nanovdb::Vec3f), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_tris, tris.data(), tris.size() * sizeof(nanovdb::Vec3i), cudaMemcpyHostToDevice));

    // openvdb::math::Transform::createLinearTransform(dx) puts voxel CENTRES on the lattice, which is
    // what nanovdb::Map(dx) does too, so the two grids index the same points.
    MeshToSDFT sdf(d_pts, uint32_t(pts.size()), d_tris, uint32_t(tris.size()), nanovdb::Map(dx));
    sdf.setVerbose(0);
    sdf.setNarrowBandWidth(halfWidth);
    sdf.setIsoValue(dx);                 // the surface shrink wrap asks offset() for

    // The barrier shell is roughly a third of a 3-voxel band, so which policy decides it dominates
    // any comparison against OpenVDB. SW_BARRIER makes that explicit instead of leaving the default
    // to explain a disagreement it caused. "heuristic" is the one that mirrors OpenVDB's own rule.
    if (const char* m = std::getenv("SW_BARRIER")) {
        const std::string mode(m);
        if      (mode == "heuristic") sdf.setBarrierSigning(MeshToSDFT::BarrierSigning::Heuristic);
        else if (mode == "ball")      sdf.setBarrierSigning(MeshToSDFT::BarrierSigning::Ball);
        else if (mode == "interior")  sdf.setBarrierSigning(MeshToSDFT::BarrierSigning::Interior);
        else std::cerr << "SW_BARRIER: unknown mode '" << mode << "' (want interior|heuristic|ball)\n";
    }
    if (const char* r = std::getenv("SW_NESTING")) {
        const std::string rule(r);
        if      (rule == "solid")   sdf.setNestingRule(MeshToSDFT::NestingRule::Solid);
        else if (rule == "evenodd") sdf.setNestingRule(MeshToSDFT::NestingRule::EvenOdd);
        else std::cerr << "SW_NESTING: unknown rule '" << rule << "' (want evenodd|solid)\n";
    }
    auto baked = sdf.build();

    // build() hands back one grid with every sidecar folded in as blind data. Whether the field
    // really travels on its own is the whole point of that, so read it back off the handle here
    // rather than assume it.
    baked.deviceDownload();
    if (const auto* g = baked.template grid<nanovdb::ValueOnIndex>()) {
        std::cout << "baked grid: " << g->gridSize() << " bytes, "
                  << g->blindDataCount() << " blind channels\n";
        for (uint32_t i = 0; i < g->blindDataCount(); ++i) {
            const auto& m = g->blindMetaData(i);
            std::cout << "  [" << i << "] " << m.mName << "  count " << m.mValueCount
                      << ", valueSize " << m.mValueSize << "\n";
        }
        const float* ch = g->template getBlindData<float>(0);
        if (ch) {
            float mn = 1e30f, mx = -1e30f;
            for (uint64_t i = 1; i < g->blindMetaData(0).mValueCount; ++i) {
                mn = std::min(mn, ch[i]); mx = std::max(mx, ch[i]);
            }
            std::cout << "  channel 0 read straight off the handle: signed distance "
                      << mn / dx << " .. " << mx / dx << " vox\n";
        } else {
            std::cout << "  channel 0 NOT readable off the handle\n";
        }
    } else {
        std::cout << "baked grid: host copy unavailable\n";
    }

    auto grid = toFloatGrid(sdf, "nanovdb");
    cudaCheck(cudaFree(d_pts));
    cudaCheck(cudaFree(d_tris));
    return grid;
}

// ---------------------------------------------------------------------------------------------------
/// @brief Report where the two fields disagree.
///
/// @details Compared over the union of the two narrow bands, since a voxel one calls band and the
///          other calls background is itself a disagreement worth seeing. The sign is the thing that
///          has to match: it decides which side of the surface a point is on, and everything shrink
///          wrap does afterwards -- eroding, unioning, renormalising -- rebuilds the magnitudes from
///          the zero crossing anyway.
static void compare(const openvdb::FloatGrid& a, const openvdb::FloatGrid& b, float dx)
{
    openvdb::FloatGrid::Ptr mask = a.deepCopy();
    mask->tree().topologyUnion(b.tree());

    auto accA = a.getConstAccessor();
    auto accB = b.getConstAccessor();

    std::size_t both = 0, signDiff = 0, signDiffBeyondShell = 0, onlyA = 0, onlyB = 0;
    double      sumAbs = 0.0;
    float       maxAbs = 0.f, worstDepth = 0.f;
    openvdb::Coord worst;

    for (auto it = mask->cbeginValueOn(); it; ++it) {
        const openvdb::Coord c = it.getCoord();
        const bool  inA = accA.isValueOn(c), inB = accB.isValueOn(c);
        if (!inA) { ++onlyB; continue; }
        if (!inB) { ++onlyA; continue; }
        ++both;
        const float va = accA.getValue(c), vb = accB.getValue(c);
        if ((va < 0.f) != (vb < 0.f)) {
            ++signDiff;
            // How far from the interface the two disagree. Both fields put the surface within half a
            // voxel of the same place, so a disagreement out beyond the barrier shell would be a
            // real one, while everything inside it is the two rules splitting the same shell.
            const float depth = std::min(std::fabs(va), std::fabs(vb)) / dx;
            if (depth > 0.866f) ++signDiffBeyondShell;
            if (depth > worstDepth) worstDepth = depth;
        }
        const float d = std::fabs(va - vb);
        sumAbs += d;
        if (d > maxAbs) { maxAbs = d; worst = c; }
    }

    std::cout << "\n---- OpenVDB offset(dx) vs NanoVDB setIsoValue(dx) ----\n"
              << "  active voxels     openvdb " << a.activeVoxelCount()
              << ", nanovdb " << b.activeVoxelCount() << "\n"
              << "  band overlap      " << both << " shared, " << onlyA
              << " openvdb-only, " << onlyB << " nanovdb-only\n"
              << "  sign disagreement " << signDiff << " of " << both
              << (both ? " (" + std::to_string(100.0 * double(signDiff) / double(both)) + "%)" : "")
              << "\n"
              << "    beyond the shell " << signDiffBeyondShell
              << " (deeper than sqrt(3)/2 vox from both surfaces), deepest "
              << worstDepth << " vox\n";
    if (both) {
        std::cout << "  |value| gap       mean " << sumAbs / double(both) / dx
                  << " vox, max " << maxAbs / dx << " vox at " << worst << "\n";
    }
}

// ---------------------------------------------------------------------------------------------------
/// @brief The shrink wrap loop, with the offset stage left as a parameter.
///
/// @details A transcription of tools::PolySoupToLevelSet::process(). It is repeated here because
///          that method builds its offset grids internally and keeps them private, so there is no
///          seam to pass a different offset through. Running the SAME loop over both offset stages
///          is what makes the comparison mean anything: a difference in the result can then only
///          come from the stage that was swapped.
///
///          Offsets are built fine to coarse, then the wrap runs coarse to fine, upsampling and
///          eroding until the enclosed volume stops moving. The erosion is a level set offset
///          followed by a union with that resolution's target, which is what stops it eroding past
///          the surface it is wrapping.
static openvdb::FloatGrid::Ptr shrinkWrapLoop(
    const std::function<openvdb::FloatGrid::Ptr(float dx)>& makeOffset,
    float minVoxelSize, float maxVoxelSize, float halfWidth,
    const openvdb::tools::ShrinkWrapLimit& D, bool verbose)
{
    using GridT = openvdb::FloatGrid;

    std::vector<GridT::Ptr> grids;
    for (float dx = minVoxelSize; dx <= maxVoxelSize; dx *= 2.0f) {
        grids.push_back(makeOffset(dx));
        if (verbose) std::cout << "  offset dx=" << dx << ": " << grids.back()->activeVoxelCount()
                               << " active voxels\n";
    }
    if (grids.empty()) throw std::runtime_error("no resolutions in the ladder");

    auto grid   = grids.back();
    grids.pop_back();
    bool isSDF  = true;                 // what offset() promises; a CSG union breaks it

    // Declared out here, as process() does: the convergence test compares against the volume the
    // PREVIOUS resolution finished at, not a fresh zero.
    double vol[2] = {0.0, 0.0};

    const float maxDist = 2.0f;         // voxels eroded per step, as in PolySoupToLevelSet
    for (auto iter = grids.rbegin(); iter != grids.rend(); ++iter) {
        // upsample: dx -> dx/2
        auto finer = openvdb::createLevelSet<GridT>(float(grid->voxelSize()[0]) / 2.f, halfWidth);
        openvdb::tools::resampleToMatch<openvdb::tools::BoxSampler>(*grid, *finer);
        grid  = finer;
        isSDF = true;

        const float dx  = float(grid->voxelSize()[0]);
        const float Ddx = D(dx);
        for (float d = 0.f; d < Ddx; vol[0] = vol[1]) {
            openvdb::tools::LevelSetFilter<GridT> filter(*grid);
            filter.setNormCount(3);
            filter.setSpatialScheme(openvdb::math::FIRST_BIAS);
            filter.setTemporalScheme(openvdb::math::TVD_RK1);
            if (!isSDF) { filter.normalize(); filter.prune(); }
            filter.offset(maxDist * dx);
            isSDF = false;
            d    += maxDist;

            grid   = openvdb::tools::csgUnionCopy(*grid, **iter);
            vol[1] = openvdb::tools::levelSetVolume(*grid);
            if (d > 0.f && openvdb::math::isApproxZero(vol[0] - vol[1])) break;
        }
        if (verbose) std::cout << "  wrapped at dx=" << dx << ": " << grid->activeVoxelCount()
                               << " active voxels, volume " << vol[1] << "\n";
        *iter = grid;
    }
    return grid;
}

// ---------------------------------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <mesh.obj> [voxelSize] [halfWidth]\n";
        return 1;
    }
    const std::string path      = argv[1];
    const float       voxelSize = (argc > 2) ? std::stof(argv[2]) : 0.02f;
    const float       halfWidth = (argc > 3) ? std::stof(argv[3]) : 3.f;

    try {
        openvdb::initialize();

        std::vector<openvdb::Vec3s> vtx;
        std::vector<openvdb::Vec3I> tri;
        readOBJ(path, vtx, tri);
        std::cout << path << ": " << vtx.size() << " vertices, " << tri.size()
                  << " triangles, voxelSize " << voxelSize << ", halfWidth " << halfWidth << "\n";

        // ---- OpenVDB: the stage shrink wrap actually calls, mode 0 (the published algorithm) ----
        // offset() reads only the soup and the half width, both set by the constructor, so it can be
        // driven on its own without running the resolution ladder around it.
        openvdb::tools::PolySoup soup;
        soup.vtx = vtx;
        soup.tri = tri;
        openvdb::tools::PolySoupToLevelSet<openvdb::FloatGrid> wrap(std::move(soup), voxelSize, halfWidth);
        auto ovdb = wrap.offset(voxelSize, /*mode=*/0);
        ovdb->setName("openvdb");
        std::cout << "openvdb offset(dx): " << ovdb->activeVoxelCount() << " active voxels\n";

        // ---- NanoVDB: the same surface, signed in place ----
        auto nvdb = nanovdbOffset(vtx, tri, voxelSize, halfWidth);
        std::cout << "nanovdb  iso=dx  : " << nvdb->activeVoxelCount() << " active voxels\n";

        compare(*ovdb, *nvdb, voxelSize);

        // ---- SW_FULL=1: the whole shrink wrap, once per offset stage ----
        if (std::getenv("SW_FULL")) {
            const float maxLength = wrap.getBBox(vtx).extents()[wrap.getBBox(vtx).maxExtent()];
            float maxVoxel  = maxLength / 2.0f;

            // The offset stage rasterizes the WHOLE soup at every rung, and its working set grows
            // with the voxel size, because a band a fixed number of voxels wide covers more world
            // space the coarser the grid is. On a soup of millions of triangles the coarsest rungs
            // can therefore exceed device memory. SW_MAX_VOXEL stops the ladder short of them; it
            // applies to BOTH offset stages, so the comparison stays like for like.
            if (const char* mv = std::getenv("SW_MAX_VOXEL")) {
                maxVoxel = std::min(maxVoxel, float(std::atof(mv)));
                std::cout << "SW_MAX_VOXEL: ladder capped at dx=" << maxVoxel << "\n";
            }
            const openvdb::tools::ShrinkWrapLimit D;

            std::cout << "\n---- full shrink wrap, dx " << voxelSize << " -> " << maxVoxel << " ----\n";

            // case 0 CONTOURS the soup back over itself, so each rung of the ladder starts from the
            // previous rung's mesh -- that accumulation is how it closes holes. The comparison call
            // above already consumed this object's soup, so the loop gets a fresh one.
            openvdb::tools::PolySoup soupA;
            soupA.vtx = vtx; soupA.tri = tri;
            openvdb::tools::PolySoupToLevelSet<openvdb::FloatGrid> wrapA_src(std::move(soupA), voxelSize, halfWidth);

            std::cout << "openvdb offset stage:\n";
            auto wrapA = shrinkWrapLoop([&](float dx) { return wrapA_src.offset(dx, 0); },
                                        voxelSize, maxVoxel, halfWidth, D, true);
            std::cout << "nanovdb offset stage:\n";
            auto wrapB = shrinkWrapLoop([&](float dx) { return nanovdbOffset(vtx, tri, dx, halfWidth); },
                                        voxelSize, maxVoxel, halfWidth, D, true);
            wrapA->setName("wrap_openvdb");
            wrapB->setName("wrap_nanovdb");

            std::cout << "\nfinal wrapped surfaces:\n"
                      << "  openvdb volume " << openvdb::tools::levelSetVolume(*wrapA) << "\n"
                      << "  nanovdb volume " << openvdb::tools::levelSetVolume(*wrapB) << "\n";
            compare(*wrapA, *wrapB, voxelSize);

            // Is the transcription of process() faithful? Run the stock algorithm and check the
            // openvdb-offset path lands on the same surface. Without this the comparison above
            // could be measuring the loop rather than the offset stage.
            if (std::getenv("SW_CHECK_LOOP")) {
                openvdb::tools::PolySoup soup2;
                soup2.vtx = vtx; soup2.tri = tri;
                openvdb::tools::PolySoupToLevelSet<openvdb::FloatGrid> stock(std::move(soup2), voxelSize, halfWidth);
                stock.process<openvdb::tools::ShrinkWrapLimit, void>(D, nullptr, 0);
                std::cout << "\n---- transcription check: our loop vs PolySoupToLevelSet::process() ----\n";
                compare(*stock.grid(0), *wrapA, voxelSize);
            }

            // Does each wrap actually contain what it was asked to wrap? Sample the input surface and
            // read the wrap there. A wrap is only useful if the answer is "all of it": the input is
            // the thing being enclosed, so a sample that lands outside is geometry the wrap lost.
            auto containment = [&](const openvdb::FloatGrid& g, const char* label) {
                auto        acc = g.getConstAccessor();
                const auto& xf  = g.transform();
                std::size_t out = 0, total = 0;
                float       worst = 0.f;
                const int   K = 5;                       // barycentric samples per triangle edge
                for (const auto& t : tri) {
                    const openvdb::Vec3s A = vtx[t[0]], B = vtx[t[1]], C = vtx[t[2]];
                    for (int a = 0; a < K; ++a)
                    for (int b = 0; b + a < K; ++b) {
                        const float wa = float(a) / (K - 1), wb = float(b) / (K - 1);
                        const openvdb::Vec3s P = A * wa + B * wb + C * (1.f - wa - wb);
                        const float v = openvdb::tools::BoxSampler::sample(acc, xf.worldToIndex(P));
                        ++total;
                        if (v > 0.f) { ++out; worst = std::max(worst, v); }
                    }
                }
                std::cout << "  " << label << ": " << out << " of " << total
                          << " input-surface samples outside the wrap ("
                          << (total ? 100.0 * double(out) / double(total) : 0.0) << "%)";
                if (out) std::cout << ", worst " << worst / voxelSize << " vox out";
                std::cout << "\n";
            };
            std::cout << "\ncontainment of the input surface:\n";
            containment(*wrapA, "openvdb");
            containment(*wrapB, "nanovdb");

            // Contour both wraps so they can be looked at next to the input. The wrap is supposed to
            // be a closed surface that encloses a mesh which is neither, so whether it does is a
            // question about where the two surfaces sit, which reads far better than any statistic.
            if (const char* stem = std::getenv("SW_EXPORT_OBJ")) {
                auto writeObj = [](const openvdb::FloatGrid& g, const std::string& path) {
                    std::vector<openvdb::Vec3s> pts;
                    std::vector<openvdb::Vec3I> tris;
                    std::vector<openvdb::Vec4I> quads;
                    openvdb::tools::volumeToMesh(g, pts, tris, quads, 0.0, 0.0);
                    std::ofstream f(path);
                    for (const auto& v : pts)   f << "v " << v[0] << " " << v[1] << " " << v[2] << "\n";
                    for (const auto& t : tris)  f << "f " << t[0]+1 << " " << t[1]+1 << " " << t[2]+1 << "\n";
                    for (const auto& q : quads) f << "f " << q[0]+1 << " " << q[1]+1 << " "
                                                          << q[2]+1 << " " << q[3]+1 << "\n";
                    std::cout << "  wrote " << path << ": " << pts.size() << " verts, "
                              << tris.size() << " tris, " << quads.size() << " quads\n";
                };
                std::cout << "\n";
                writeObj(*wrapA, std::string(stem) + "_openvdb.obj");
                writeObj(*wrapB, std::string(stem) + "_nanovdb.obj");
            }

            if (const char* out = std::getenv("SW_EXPORT_VDB")) {
                openvdb::io::File(out).write({ovdb, nvdb, wrapA, wrapB});
                std::cout << "\nwrote 4 grids to " << out << "\n";
            }
            openvdb::uninitialize();
            return 0;
        }

        if (const char* out = std::getenv("SW_EXPORT_VDB")) {
            openvdb::io::File(out).write({ovdb, nvdb});
            std::cout << "\nwrote both grids to " << out << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
