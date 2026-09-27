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
#include <nanovdb/HostBuffer.h>
#include <nanovdb/cuda/Buffer.h>
#include <nanovdb/cuda/HandleStorage.h>   // cuda::copyTo
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
#include <openvdb/math/Proximity.h>       // closestPointOnTriangleToPoint
#include <tbb/parallel_for.h>

#include <algorithm>
#include <chrono>
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
/// @brief Wall-clock seconds since construction.
///
/// @details Wall clock rather than a CUDA event, because the question is what a caller waits for:
///          the device work, the host work, and the hand-off between them all count. Device work is
///          synchronised before each reading is taken.
struct Stopwatch
{
    std::chrono::steady_clock::time_point t0{std::chrono::steady_clock::now()};
    void   reset() { t0 = std::chrono::steady_clock::now(); }
    double s() const {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
};

/// Where the time inside nanovdbOffset() goes, accumulated over every call.
///
/// Only `build` is the offset algorithm. The other three exist because the mesh arrives in OpenVDB
/// types on the host and the answer has to go back the same way, and they would all disappear if
/// the caller already held its geometry on the device. Reported apart from the algorithm so that
/// plumbing is not read as compute.
struct OffsetProfile
{
    double upload  = 0.0;  ///< converting the soup to NanoVDB types and copying it to the device
    double build   = 0.0;  ///< MeshToSDF::getHandle()
    double verify  = 0.0;  ///< reading the baked grid back to check it travels on its own
    double handOff = 0.0;  ///< toFloatGrid: NanoVDB result -> OpenVDB grid
};
static OffsetProfile sProfile;

// ---------------------------------------------------------------------------------------------------
/// @brief Run the NanoVDB pipeline for the same surface offset() is asked for: { udf == isoValue }.
static openvdb::FloatGrid::Ptr nanovdbOffset(const std::vector<openvdb::Vec3s>& vtx,
                                             const std::vector<openvdb::Vec3I>& tri,
                                             float dx, float halfWidth)
{
    Stopwatch stage;

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
    sProfile.upload += stage.s();

    // openvdb::math::Transform::createLinearTransform(dx) puts voxel CENTRES on the lattice, which is
    // what nanovdb::Map(dx) does too, so the two grids index the same points.
    MeshToSDFT sdf(d_pts, uint32_t(pts.size()), d_tris, uint32_t(tris.size()), nanovdb::Map(dx));
    // SW_PROFILE also turns on the library's own stage timers, which is the only way to see inside
    // rasterization -- the phase that normally dominates -- without a external profiler.
    sdf.setVerbose(std::getenv("SW_PROFILE") ? 1 : 0);
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
    stage.reset();
    auto baked = sdf.getHandle();
    sProfile.build += stage.s();

    stage.reset();
    // getHandle() hands back one grid with every sidecar folded in as blind data. Whether the field
    // really travels on its own is the whole point of that, so read it back off the handle here
    // rather than assume it.
    const auto bakedHost = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(baked);
    if (const auto* g = bakedHost.template grid<nanovdb::ValueOnIndex>()) {
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
    sProfile.verify += stage.s();

    stage.reset();
    auto grid = toFloatGrid(sdf, "nanovdb");
    sProfile.handOff += stage.s();

    // getHandle()'s own phase breakdown, so a slow offset can be pinned on a phase rather than guessed
    // at. Phase 0 is rasterization, which normally dominates.
    if (std::getenv("SW_PROFILE")) {
        static const char* kPhase[5] = {"rasterize", "partition", "sign", "compose", "fill"};
        const float* ms = sdf.phaseMs();
        std::cout << "  [profile dx=" << dx << "] build phases";
        for (int i = 0; i < 5; ++i)
            std::cout << "  " << kPhase[i] << " " << std::fixed << std::setprecision(2)
                      << ms[i] / 1000.0 << " s";
        std::cout << "\n" << std::defaultfloat << std::setprecision(6);
    }

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
/// @brief Break an offset time down into algorithm and plumbing, if any NanoVDB offset ran.
///
/// @details Prints nothing for the OpenVDB stage, which has no such split -- it never leaves the
///          host, so there is nothing to upload, verify or hand back.
static void reportProfile(const OffsetProfile& before, const OffsetProfile& after)
{
    const double up = after.upload - before.upload, bu = after.build - before.build;
    const double ve = after.verify - before.verify, ho = after.handOff - before.handOff;
    if (up + bu + ve + ho <= 0.0) return;
    std::cout << " (build " << bu << ", upload " << up << ", verify " << ve
              << ", hand-off " << ho << ")";
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
    const openvdb::tools::ShrinkWrapLimit& D, bool verbose, const char* rungStem = nullptr)
{
    using GridT = openvdb::FloatGrid;

    std::vector<GridT::Ptr> grids;
    double offsetSeconds = 0.0, wrapSeconds = 0.0;
    const OffsetProfile profileAtEntry = sProfile;
    Stopwatch clock;
    for (float dx = minVoxelSize; dx <= maxVoxelSize; dx *= 2.0f) {
        const OffsetProfile before = sProfile;
        clock.reset();
        grids.push_back(makeOffset(dx));
        const double dt = clock.s();
        offsetSeconds += dt;
        if (verbose) {
            std::cout << "  offset dx=" << dx << ": " << grids.back()->activeVoxelCount()
                      << " active voxels, " << std::fixed << std::setprecision(2) << dt << " s";
            reportProfile(before, sProfile);
            std::cout << "\n" << std::defaultfloat << std::setprecision(6);
        }
    }
    if (grids.empty()) throw std::runtime_error("no resolutions in the ladder");

    auto grid   = grids.back();
    grids.pop_back();
    bool isSDF  = true;                 // what offset() promises; a CSG union breaks it

    // Declared out here, as process() does: the convergence test compares against the volume the
    // PREVIOUS resolution finished at, not a fresh zero.
    double vol[2] = {0.0, 0.0};

    const float maxDist = 2.0f;         // voxels eroded per step, as in PolySoupToLevelSet
    clock.reset();
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
                               << " active voxels, volume " << vol[1]
                               << ", " << std::fixed << std::setprecision(2) << clock.s() - wrapSeconds
                               << " s\n" << std::defaultfloat << std::setprecision(6);
        wrapSeconds = clock.s();
        // Contour the wrap as it stands so the shape can be inspected rung by rung. The loop erodes
        // and then unions the rung's target back in, so it does not simply simplify as it goes; where
        // a feature first appears is a question about which rung introduced it.
        if (rungStem) {
            std::vector<openvdb::Vec3s> pts; std::vector<openvdb::Vec3I> tris; std::vector<openvdb::Vec4I> quads;
            openvdb::tools::volumeToMesh(*grid, pts, tris, quads, 0.0, 0.0);
            std::ostringstream fn; fn << rungStem << "_dx" << dx << ".obj";
            std::ofstream f(fn.str());
            for (const auto& v : pts)   f << "v " << v[0] << " " << v[1] << " " << v[2] << "\n";
            for (const auto& t : tris)  f << "f " << t[0]+1 << " " << t[1]+1 << " " << t[2]+1 << "\n";
            for (const auto& q : quads) f << "f " << q[0]+1 << " " << q[1]+1 << " " << q[2]+1 << " " << q[3]+1 << "\n";
            // Contouring for inspection is not part of the loop, so do not let it show up in the
            // next rung's timing: roll the clock forward past it.
            clock.t0 += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                            std::chrono::duration<double>(clock.s() - wrapSeconds));
        }
        *iter = grid;
    }
    if (verbose) {
        std::cout << "  [time] offset stage " << std::fixed << std::setprecision(2) << offsetSeconds
                  << " s";
        reportProfile(profileAtEntry, sProfile);
        std::cout << ", wrap loop " << wrapSeconds << " s, total " << offsetSeconds + wrapSeconds
                  << " s\n" << std::defaultfloat << std::setprecision(6);
    }
    return grid;
}

// ---------------------------------------------------------------------------------------------------
/// @brief Replace a polygon soup with the zero isosurface of a level set.
///
/// @details This is the coarsening that OpenVDB's case 0 gets as a side effect: volumeToMesh writes
///          its result back over the soup it was given, so the next, coarser resolution reads a mesh
///          whose size is set by the grid it came from rather than by the original input. Doing it
///          explicitly here gives the NanoVDB stage the same input sequence.
///
///          Contouring at zero, not at dx: the field handed in is already signed, so its own zero
///          crossing IS the offset surface. That is also why this cannot fail the way contouring an
///          unsigned field at dx does -- there is one wall to find, not two.
static void contourIntoSoup(const openvdb::FloatGrid& grid,
                            std::vector<openvdb::Vec3s>& vtx,
                            std::vector<openvdb::Vec3I>& tri)
{
    std::vector<openvdb::Vec3s> pts;
    std::vector<openvdb::Vec3I> tris;
    std::vector<openvdb::Vec4I> quads;
    openvdb::tools::volumeToMesh(grid, pts, tris, quads, 0.0, 0.0);

    tris.reserve(tris.size() + 2 * quads.size());
    for (const auto& q : quads) {           // MeshToSDF takes triangles only
        tris.emplace_back(q[0], q[1], q[2]);
        tris.emplace_back(q[0], q[2], q[3]);
    }
    if (tris.empty()) {                     // nothing to hand on; keep what we had
        std::cerr << "  contour produced no polygons; soup left unchanged\n";
        return;
    }
    vtx.swap(pts);
    tri.swap(tris);
}

// ---------------------------------------------------------------------------------------------------
/// @brief Time the two unsigned-distance rasterizers against each other on the same triangles.
///
/// @details The offset stages do more than rasterize -- OpenVDB contours and re-signs afterwards,
///          MeshToSDF runs connected components and signs -- so comparing them end to end does not
///          say which rasterizer is faster. This compares only the step both pipelines start from:
///          triangles in, a narrow band of unsigned distance out.
///
///          Matched as closely as the two APIs allow. Both get the same band width in voxels and
///          the same voxel size, and neither is asked for a sign. The remaining difference is that
///          MeshToGrid also emits an index-space topology it can hand downstream, where OpenVDB
///          returns a finished FloatGrid; getHandleAndUDF is used rather than the variant that also
///          returns nearest-triangle ids, so that extra sidecar is not on the clock.
static void benchmarkUDF(const std::vector<openvdb::Vec3s>& vtx,
                         const std::vector<openvdb::Vec3I>& tri,
                         float dx, float halfWidth)
{
    std::cout << "\n---- unsigned distance rasterizers, " << tri.size() << " triangles, dx " << dx
              << ", band " << halfWidth << " voxels ----\n";

    const std::vector<openvdb::Vec4I> noQuads;
    auto xform = openvdb::math::Transform::createLinearTransform(dx);

    // Each side is run twice. The first NanoVDB run pays for loading and, on a PTX-only build,
    // compiling the kernel module, which is a one-off per process and not what a caller repeating
    // the operation would see. Reporting both keeps that visible instead of buried in an average.
    double tA = 0.0;
    openvdb::FloatGrid::Ptr udfA;
    for (int run = 0; run < 2; ++run) {
        Stopwatch t;
        udfA = openvdb::tools::meshToUnsignedDistanceField<openvdb::FloatGrid>(
                   *xform, vtx, tri, noQuads, halfWidth);
        tA = t.s();
        std::cout << "  openvdb  meshToUnsignedDistanceField  run " << run + 1 << "  "
                  << std::fixed << std::setprecision(2) << tA << " s, "
                  << udfA->activeVoxelCount() << " active voxels\n" << std::defaultfloat << std::setprecision(6);
    }

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
    cudaCheck(cudaDeviceSynchronize());   // the upload is not what is being timed

    // Both sidecar variants, because they take different code paths and MeshToSDF uses the second.
    using ByteBufferT = nanovdb::cuda::Buffer<std::byte>;
    using TraitsT     = nanovdb::util::cuda::DeviceGridTraits<nanovdb::ValueOnIndex>;
    double tB = 0.0;
    for (int run = 0; run < 2; ++run) {
        for (int withIndex = 0; withIndex < 2; ++withIndex) {
            Stopwatch t;
            nanovdb::tools::cuda::MeshToGrid<nanovdb::ValueOnIndex> converter(
                d_pts, uint32_t(pts.size()), d_tris, uint32_t(tris.size()), nanovdb::Map(dx));
            converter.setVerbose(std::getenv("SW_PROFILE") ? 1 : 0);
            converter.setNarrowBandWidth(halfWidth);
            uint64_t active = 0;
            if (withIndex) {
                auto [handle, udf, idx] = converter.getHandleAndUDFAndIndex<ByteBufferT, ByteBufferT>();
                cudaCheck(cudaDeviceSynchronize());
                tB = t.s();
                if (const auto* g = handle.deviceGrid<nanovdb::ValueOnIndex>()) active = TraitsT::getActiveVoxelCount(g);
            } else {
                auto [handle, udf] = converter.getHandleAndUDF<ByteBufferT, ByteBufferT>();
                cudaCheck(cudaDeviceSynchronize());
                tB = t.s();
                if (const auto* g = handle.deviceGrid<nanovdb::ValueOnIndex>()) active = TraitsT::getActiveVoxelCount(g);
            }
            std::cout << "  nanovdb  MeshToGrid " << (withIndex ? "+index          " : "                ")
                      << "  run " << run + 1 << "  " << std::fixed << std::setprecision(2) << tB
                      << " s, " << active << " active voxels";
            if (tA > 0.0) std::cout << "   (" << tB / tA << "x openvdb)";
            std::cout << "\n" << std::defaultfloat << std::setprecision(6);
        }
    }

    // ---- are the two fields the same, or only the same shape? ----
    //
    // Our rasterizer tests every voxel of a leaf against every triangle whose dilated bounding box
    // reaches that leaf, so within the band it finds the true nearest triangle by exhaustion.
    // OpenVDB instead carries a nearest-triangle index outward from voxels it has already solved
    // and re-evaluates against that inherited shortlist. Inheriting a shortlist can only MISS the
    // true nearest triangle, never invent a closer one, so any disagreement is one-sided: OpenVDB
    // over-estimates exactly where the propagation lost the right candidate. Measuring that is what
    // says whether the cheaper traversal is exact or approximate.
    {
        MeshToSDFT sdf(d_pts, uint32_t(pts.size()), d_tris, uint32_t(tris.size()), nanovdb::Map(dx));
        sdf.setVerbose(0);
        sdf.setNarrowBandWidth(halfWidth);
        sdf.setIsoValue(0.f);          // no isovalue: |value| is then the raw unsigned distance
        sdf.getHandle();
        auto exact = toFloatGrid(sdf, "exact");

        auto accE = exact->getConstAccessor();
        std::size_t both = 0, over = 0, under = 0;
        double sumOver = 0.0;
        float  maxOver = 0.f, maxUnder = 0.f;
        openvdb::Coord worst;
        const float eps = 1e-4f * dx;   // float32 noise on a distance of a few voxels

        // Binned by true distance, because where the error sits decides whether it matters. The
        // shrink wrap offset contours this field at one voxel out, so an error at 2-3 voxels is
        // harmless to it while an error at 1 voxel moves the surface the wrap is built on.
        constexpr int kBins = 6;                  // half a voxel each, out to 3
        std::size_t binN[kBins] = {}, binOver[kBins] = {};
        double      binSum[kBins] = {};
        float       binMax[kBins] = {};

        for (auto it = udfA->cbeginValueOn(); it; ++it) {
            const openvdb::Coord c = it.getCoord();
            if (!accE.isValueOn(c)) continue;
            ++both;
            const float truth = std::fabs(accE.getValue(c));
            const float d = *it - truth;                          // openvdb minus exact
            if (d > eps)  { ++over;  sumOver += d; if (d > maxOver)  { maxOver = d; worst = c; } }
            if (d < -eps) { ++under; if (-d > maxUnder) maxUnder = -d; }

            const int b = std::min(kBins - 1, int(truth / dx * 2.f));
            ++binN[b];
            if (d > eps) { ++binOver[b]; binSum[b] += d; if (d > binMax[b]) binMax[b] = d; }
        }
        std::cout << "  distance agreement over " << both << " shared band voxels\n"
                  << "    openvdb over-estimates  " << over
                  << (both ? " (" + std::to_string(100.0 * double(over) / double(both)) + "%)" : "")
                  << ", mean " << std::fixed << std::setprecision(4)
                  << (over ? sumOver / double(over) / dx : 0.0) << " vox, max "
                  << maxOver / dx << " vox at " << worst << "\n"
                  << "    openvdb under-estimates " << under << ", max " << maxUnder / dx << " vox\n"
                  << std::defaultfloat << std::setprecision(6);

        std::cout << "    where the over-estimate sits (true distance -> share of voxels wrong):\n";
        for (int b = 0; b < kBins; ++b) {
            if (!binN[b]) continue;
            std::cout << "      " << std::fixed << std::setprecision(1) << 0.5 * b << "-"
                      << 0.5 * (b + 1) << " vox: " << std::setprecision(2)
                      << 100.0 * double(binOver[b]) / double(binN[b]) << "% of " << binN[b]
                      << ", mean " << std::setprecision(4)
                      << (binOver[b] ? binSum[b] / double(binOver[b]) / dx : 0.0)
                      << " vox, max " << binMax[b] / dx << " vox\n";
        }
        std::cout << std::defaultfloat << std::setprecision(6);

        // ---- SW_BRUTE=N: an independent reference, computed the dumbest way there is ----
        //
        // Every sampled band voxel against EVERY triangle, on the host, with OpenVDB's own
        // point-triangle routine rather than the one the device kernels use. Nothing is culled, so
        // the answer cannot depend on a bounding box being generous enough or a stencil being wide
        // enough -- it is the definition of the distance, evaluated. That makes it the only thing
        // here entitled to be called ground truth, and it is also what a rasterizer would cost if
        // it refused to prune: N_voxels x N_triangles, which is why nobody ships this.
        if (const char* bn = std::getenv("SW_BRUTE")) {
            std::vector<openvdb::Coord> coords;
            for (auto it = udfA->cbeginValueOn(); it; ++it)
                if (accE.isValueOn(it.getCoord())) coords.push_back(it.getCoord());

            const std::size_t want   = std::max<std::size_t>(1, std::size_t(std::atol(bn)));
            const std::size_t stride = std::max<std::size_t>(1, coords.size() / want);
            std::vector<openvdb::Coord> sample;
            for (std::size_t i = 0; i < coords.size(); i += stride) sample.push_back(coords[i]);

            std::vector<double> exactDist(sample.size());
            Stopwatch bt;
            tbb::parallel_for(tbb::blocked_range<std::size_t>(0, sample.size()),
                [&](const tbb::blocked_range<std::size_t>& r) {
                    openvdb::Vec3d uvw;
                    for (std::size_t i = r.begin(); i != r.end(); ++i) {
                        const openvdb::Coord& c = sample[i];
                        const openvdb::Vec3d p(c[0] * dx, c[1] * dx, c[2] * dx);
                        double best = std::numeric_limits<double>::max();
                        for (const auto& t : tri) {
                            const openvdb::Vec3d a(vtx[t[0]]), b(vtx[t[1]]), q(vtx[t[2]]);
                            const double d2 = (p - openvdb::math::closestPointOnTriangleToPoint(
                                                       a, b, q, p, uvw)).lengthSqr();
                            if (d2 < best) best = d2;
                        }
                        exactDist[i] = std::sqrt(best);
                    }
                });
            const double tBrute = bt.s();

            double oursMax = 0.0, ovdbMax = 0.0, oursUnder = 0.0;
            for (std::size_t i = 0; i < sample.size(); ++i) {
                const double ref  = exactDist[i];
                const double mine = std::fabs(accE.getValue(sample[i]));
                const double theirs = udfA->getConstAccessor().getValue(sample[i]);
                oursMax   = std::max(oursMax,   mine   - ref);   // >0 means we over-estimate
                oursUnder = std::max(oursUnder, ref    - mine);  // >0 means we under-estimate
                ovdbMax   = std::max(ovdbMax,   theirs - ref);
            }
            std::cout << "  brute force over " << sample.size() << " of " << coords.size()
                      << " band voxels x " << tri.size() << " triangles: "
                      << std::fixed << std::setprecision(2) << tBrute << " s on the host\n"
                      << std::setprecision(6)
                      << "    ours    vs truth: max over " << oursMax / dx
                      << " vox, max under " << oursUnder / dx << " vox\n"
                      << "    openvdb vs truth: max over " << ovdbMax / dx << " vox\n"
                      << std::defaultfloat << std::setprecision(6);
        }
    }

    cudaCheck(cudaFree(d_pts));
    cudaCheck(cudaFree(d_tris));
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

        // ---- SW_BENCH_UDF=1: just the two rasterizers, nothing downstream of them ----
        if (std::getenv("SW_BENCH_UDF")) {
            benchmarkUDF(vtx, tri, voxelSize, halfWidth);
            openvdb::uninitialize();
            return 0;
        }

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
            const char* rungA = std::getenv("SW_EXPORT_RUNGS");
            auto wrapA = shrinkWrapLoop([&](float dx) { return wrapA_src.offset(dx, 0); },
                                        voxelSize, maxVoxel, halfWidth, D, true,
                                        rungA ? (std::string(rungA) + "_openvdb").c_str() : nullptr);
            std::cout << "nanovdb offset stage:\n";
            const std::string rungBname = rungA ? std::string(rungA) + "_nanovdb" : std::string();

            // SW_ACCUMULATE makes the NanoVDB ladder read its own previous rung, the way case 0's
            // ladder already reads its own. Without it every rung re-rasterizes the original soup,
            // which is why our coarse rungs cost the same as the finest one. Note what the contour
            // is and is not doing here: it no longer decides any sign -- the field it is cutting is
            // already signed -- it only decimates the input for the next resolution.
            const bool accumulate = std::getenv("SW_ACCUMULATE") != nullptr;
            std::vector<openvdb::Vec3s> soupV = vtx;
            std::vector<openvdb::Vec3I> soupT = tri;
            auto wrapB = shrinkWrapLoop([&](float dx) {
                                            auto g = nanovdbOffset(soupV, soupT, dx, halfWidth);
                                            if (accumulate) {
                                                contourIntoSoup(*g, soupV, soupT);
                                                std::cout << "    soup for the next rung: "
                                                          << soupV.size() << " verts, "
                                                          << soupT.size() << " tris\n";
                                            }
                                            return g;
                                        },
                                        voxelSize, maxVoxel, halfWidth, D, true,
                                        rungA ? rungBname.c_str() : nullptr);
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
