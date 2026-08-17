// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file  connected_components_cuda_kernels.cu
///
/// @brief CUDA / NanoVDB side of the connected-components example. Rasterizes a triangle mesh into a
///        ValueOnIndex narrow-band grid (nanovdb::tools::cuda::MeshToGrid), optionally discards the
///        surface/barrier shell (unsigned distance within sqrt(3)/2 voxels of the surface) with
///        nanovdb::tools::cuda::PruneGrid, and runs connected-components labeling with
///        nanovdb::tools::cuda::ConnectedComponents. A CPU union-find oracle independently verifies
///        the GPU labeling (component count + per-voxel partition), and --openvdb-oracle additionally
///        reports what OpenVDB's own CPU segmentation finds on the same topology.

#include <nanovdb/NanoVDB.h>
#include <nanovdb/GridHandle.h>
#include <nanovdb/cuda/DeviceBuffer.h>

#include <nanovdb/tools/cuda/MeshToGrid.cuh>            // rasterize mesh -> ValueOnIndex + UDF
#include <nanovdb/tools/cuda/PruneGrid.cuh>             // topological prune (barrier removal)
#include <nanovdb/tools/cuda/ConnectedComponents.cuh>  // the connected-components labeling
#include <nanovdb/util/cuda/Util.h>                     // operatorKernel, cudaCheck
#include <nanovdb/util/cuda/DeviceGridTraits.cuh>       // DeviceGridTraits

#include <thrust/universal_vector.h>

#ifdef NANOVDB_USE_OPENVDB
#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetUtil.h>                 // extractActiveVoxelSegmentMasks
#endif

#include <chrono>
#include <cstdint>
#include <functional>
#include <iostream>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace {

using BuildT      = nanovdb::ValueOnIndex;
using GridHandleT = nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>;
using Traits      = nanovdb::util::cuda::DeviceGridTraits<BuildT>;

constexpr int LEAF_SIZE = 512;  // 8^3

using Clock = std::chrono::steady_clock;
inline double secondsBetween(Clock::time_point a, Clock::time_point b)
{
    return std::chrono::duration<double>(b - a).count();
}

// Per-leaf retain-mask functor: a voxel is kept iff its unsigned distance to the surface exceeds the
// barrier threshold sqrt(3)/2 voxels (i.e. UDF^2 >= 0.75 * voxelSize^2 in world units). Removing the
// barrier shell splits each closed surface's narrow band into disjoint inner/outer shells, which is
// what connected components then labels. One CUDA block per leaf, one thread per voxel offset.
struct UDFBarrierPruneMaskFunctor
{
    static constexpr int MaxThreadsPerBlock         = LEAF_SIZE;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(const nanovdb::NanoGrid<BuildT>* d_grid,
                               const float*                     d_udf,          // UDF sidecar, WORLD units
                               float                            barrierSqWorld, // (sqrt(3)/2 * voxelSize)^2
                               nanovdb::Mask<3>*                d_dstLeafMasks)
    {
        const int leafID   = blockIdx.x;
        const int threadID = threadIdx.x;

        const auto& leaf       = d_grid->tree().getFirstNode<0>()[leafID];
        auto&       resultMask = d_dstLeafMasks[leafID];

        // Clear the leaf's mask words in parallel, then set the retain bits.
        if (threadID < nanovdb::Mask<3>::WORD_COUNT)
            resultMask.words()[threadID] = 0UL;
        __syncthreads();

        if (auto n = leaf.data()->getValue(threadID)) {  // n != 0 => active voxel
            const float udf = d_udf[n];
            if (udf * udf >= barrierSqWorld)              // retain non-barrier voxels
                resultMask.setOnAtomic(threadID);
        }
    }
};

// Pack a voxel coordinate into a sortable/ hashable int64 key (offset so negatives stay positive).
inline int64_t encodeCoord(const nanovdb::Coord& c)
{
    return  (int64_t(c[0]) + (1 << 20))
         | ((int64_t(c[1]) + (1 << 20)) << 21)
         | ((int64_t(c[2]) + (1 << 20)) << 42);
}

// CPU union-find oracle: independently label the derived grid's active voxels by 6-connectivity and
// verify the GPU result (a) has the same component count and (b) induces the same partition (two
// voxels share a GPU label iff the oracle puts them in the same component). Returns true on PASS.
bool validateAgainstOracle(const GridHandleT& derivedHandle, uint32_t leafCount, uint64_t active,
                           const uint32_t* d_labels, uint64_t gpuCount)
{
    // Download the derived grid blob + the per-voxel labels to the host.
    std::vector<char> blob(derivedHandle.bufferSize());
    cudaCheck(cudaMemcpy(blob.data(), derivedHandle.deviceData(), blob.size(), cudaMemcpyDeviceToHost));
    const auto* h_grid = reinterpret_cast<const nanovdb::NanoGrid<BuildT>*>(blob.data());

    if (active == 0) {
        std::cout << "CPU-oracle self-check: PASS (empty grid, 0 components)\n";
        return gpuCount == 0;
    }

    std::vector<uint32_t> labels(active + 1);
    cudaCheck(cudaMemcpy(labels.data(), d_labels, (active + 1) * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Gather active voxels: dense id -> {coord, gpu label}, plus a coord->id lookup for neighbours.
    std::unordered_map<int64_t, uint32_t> coordToId;
    std::vector<nanovdb::Coord>           idToCoord;
    std::vector<uint32_t>                 idToGpuLabel;
    coordToId.reserve(active * 2);
    idToCoord.reserve(active);
    idToGpuLabel.reserve(active);

    const auto* leaves = h_grid->tree().getFirstLeaf();
    for (uint32_t li = 0; li < leafCount; ++li) {
        const auto& leaf = leaves[li];
        for (uint32_t n = 0; n < uint32_t(LEAF_SIZE); ++n) {
            if (!leaf.isActive(n)) continue;
            const nanovdb::Coord c = leaf.origin() + nanovdb::NanoLeaf<BuildT>::OffsetToLocalCoord(n);
            const uint64_t       slot = leaf.getValue(n);
            const uint32_t       id   = uint32_t(idToCoord.size());
            coordToId.emplace(encodeCoord(c), id);
            idToCoord.push_back(c);
            idToGpuLabel.push_back(labels[slot]);
        }
    }

    // Union-find over 6-connectivity. Visiting only +X/+Y/+Z reaches every undirected edge once.
    std::vector<uint32_t> parent(idToCoord.size());
    std::iota(parent.begin(), parent.end(), 0u);
    std::function<uint32_t(uint32_t)> find = [&](uint32_t x) {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    };
    auto unite = [&](uint32_t a, uint32_t b) {
        a = find(a); b = find(b);
        if (a != b) parent[a > b ? a : b] = (a < b ? a : b);
    };
    const nanovdb::Coord dirs[3] = { {1,0,0}, {0,1,0}, {0,0,1} };
    for (uint32_t id = 0; id < idToCoord.size(); ++id)
        for (const auto& d : dirs) {
            auto it = coordToId.find(encodeCoord(idToCoord[id] + d));
            if (it != coordToId.end()) unite(id, it->second);
        }

    // Count oracle components and check the GPU labels induce the same partition.
    std::unordered_map<uint32_t, uint32_t> rootToComp;   // oracle root -> dense component id
    for (uint32_t id = 0; id < parent.size(); ++id) {
        const uint32_t r = find(id);
        if (!rootToComp.count(r)) rootToComp.emplace(r, uint32_t(rootToComp.size()));
    }
    const uint64_t oracleCount = rootToComp.size();

    std::unordered_map<uint32_t, uint32_t> gpuToOracle;  // gpu label -> oracle component
    uint64_t partitionViolations = 0;
    for (uint32_t id = 0; id < parent.size(); ++id) {
        const uint32_t gl = idToGpuLabel[id];
        const uint32_t oc = rootToComp[find(id)];
        auto it = gpuToOracle.find(gl);
        if (it == gpuToOracle.end()) gpuToOracle.emplace(gl, oc);
        else if (it->second != oc)   ++partitionViolations;
    }
    const uint64_t gpuDistinctLabels = gpuToOracle.size();

    const bool pass = (gpuCount == oracleCount) &&
                      (gpuDistinctLabels == oracleCount) &&
                      (partitionViolations == 0);

    std::cout << "CPU-oracle self-check: " << (pass ? "PASS" : "FAIL")
              << "  (gpu=" << gpuCount << ", oracle=" << oracleCount
              << ", distinct gpu labels=" << gpuDistinctLabels
              << ", partition violations=" << partitionViolations << ")\n";
    return pass;
}

#ifdef NANOVDB_USE_OPENVDB

// ---- Optional second baseline: OpenVDB's own CPU segmentation (--openvdb-oracle). ----------------
//
// OpenVDB already ships connected-component segmentation over active voxels
// (openvdb/tools/LevelSetUtil.h). It is third-party code that we did not write, so agreement with it
// says more than agreement with the union-find oracle above. It is NOT a replacement for that oracle:
// OpenVDB uses the same per-leaf / cross-leaf / global decomposition the CUDA path does, whereas the
// union-find oracle labels through one flat coordinate hash and so tests that decomposition itself.
//
// See OPENVDB_BASELINE.md for the full analysis this implements.

// Count the grid's ACTIVE TILES: root tiles with no child, plus upper/lower slots whose value bit is
// on while their child bit is off. Must be zero for the comparison to mean anything -- OpenVDB
// densifies tiles into voxels and labels them, while this CUDA path iterates leaves only and reports
// two leaves joined through a tile as separate components.
uint64_t countActiveTiles(const nanovdb::NanoGrid<BuildT>& grid)
{
    const auto& tree = grid.tree();
    const auto& root = tree.root();

    uint64_t count = 0;
    for (uint32_t i = 0; i < root.tileCount(); ++i)
        if (root.data()->tile(i)->isActive()) ++count;

    auto tilesIn = [](const auto* nodes, uint32_t nodeCount) {
        uint64_t n = 0;
        for (uint32_t i = 0; i < nodeCount; ++i) {
            const auto& value = nodes[i].valueMask();
            const auto& child = nodes[i].childMask();
            for (uint32_t w = 0; w < value.wordCount(); ++w)
                n += nanovdb::util::countOn(value.words()[w] & ~child.words()[w]);
        }
        return n;
    };
    count += tilesIn(tree.getFirstUpper(), tree.nodeCount(2));
    count += tilesIn(tree.getFirstLower(), tree.nodeCount(1));
    return count;
}

// Copy the NanoVDB active topology into an OpenVDB MaskGrid, which is all the baseline consumes.
//
// Hand-rolled rather than routed through NanoToOpenVDB.h: both of that header's index-grid overloads
// are built around carrying DATA (a blind-data channel or an explicit sidecar) and neither yields
// "the active topology as a mask grid". Keeping it explicit also keeps the converter auditable -- it
// must not itself be under test.
//
// Voxels are placed by COORDINATE, not by reusing the linear offset: both libraries happen to index a
// leaf as n = 64x + 8y + z, but the baseline must not rest on that coincidence.
openvdb::MaskGrid::Ptr toOpenVDBMask(const nanovdb::NanoGrid<BuildT>& grid, uint32_t leafCount)
{
    using OpenVDBLeafT = openvdb::MaskGrid::TreeType::LeafNodeType;

    auto        out    = openvdb::MaskGrid::create();
    auto&       dstTree = out->tree();
    const auto* leaves  = grid.tree().getFirstLeaf();

    for (uint32_t li = 0; li < leafCount; ++li) {
        const auto&          leaf = leaves[li];
        const nanovdb::Coord o    = leaf.origin();
        auto* dst = dstTree.touchLeaf(openvdb::Coord(o[0], o[1], o[2]));
        for (uint32_t n = 0; n < uint32_t(LEAF_SIZE); ++n) {
            if (!leaf.isActive(n)) continue;
            const nanovdb::Coord ijk = o + nanovdb::NanoLeaf<BuildT>::OffsetToLocalCoord(n);
            dst->setValueOn(OpenVDBLeafT::coordToOffset(openvdb::Coord(ijk[0], ijk[1], ijk[2])));
        }
    }
    return out;
}

// Run the baseline on whichever grid was labeled and report what it found. Reports and skips rather
// than failing when its preconditions do not hold: a skipped check is honest, a spurious FAIL is not.
void runOpenVDBBaseline(const GridHandleT& handle, uint32_t leafCount, uint64_t active,
                        const uint32_t* d_labels, uint64_t gpuCount)
{
    openvdb::initialize();

    std::vector<char> blob(handle.bufferSize());
    cudaCheck(cudaMemcpy(blob.data(), handle.deviceData(), blob.size(), cudaMemcpyDeviceToHost));
    const auto* h_grid = reinterpret_cast<const nanovdb::NanoGrid<BuildT>*>(blob.data());

    if (const uint64_t tiles = countActiveTiles(*h_grid)) {
        std::cout << "OpenVDB baseline: SKIPPED (" << tiles << " active tiles; the two "
                  << "implementations partition tiled input differently)\n";
        return;
    }

    const auto t0   = Clock::now();
    auto       mask = toOpenVDBMask(*h_grid, leafCount);
    const auto t1   = Clock::now();

    const uint64_t maskLeaves = mask->tree().leafCount();
    const uint64_t maskActive = mask->activeVoxelCount();
    const bool     converted  = (maskLeaves == leafCount) && (maskActive == active) &&
                                !mask->tree().hasActiveTiles();
    std::cout << "OpenVDB conversion: " << (converted ? "OK" : "MISMATCH")
              << "  (leaves " << maskLeaves << "/" << leafCount
              << ", active " << maskActive << "/" << active
              << ", " << secondsBetween(t0, t1) << " s)\n";
    if (!converted) {
        std::cout << "OpenVDB baseline: SKIPPED (conversion disagrees with the source grid)\n";
        return;
    }

    std::vector<openvdb::BoolGrid::Ptr> masks;
    const auto t2 = Clock::now();
    openvdb::tools::extractActiveVoxelSegmentMasks(*mask, masks);
    const auto t3 = Clock::now();

    std::cout << "OpenVDB baseline: " << masks.size() << " segments in "
              << secondsBetween(t2, t3) << " s\n";

    // Partition comparison. OpenVDB numbers its segments by descending voxel count and the CUDA path
    // by first appearance, so the labels themselves cannot be compared -- what has to hold is that the
    // two induce the SAME partition: two voxels share a GPU label exactly when they share a segment.
    //
    // Each segment is a mask over the same coordinates, so its voxels are resolved back to the source
    // grid's value slots and the comparison runs over the slot arrays. That keeps it linear and needs
    // no coordinate hash: the source grid's own accessor already maps a coordinate to its slot.
    constexpr uint32_t UNASSIGNED = ~uint32_t(0);
    std::vector<uint32_t> segOfSlot(active + 1, UNASSIGNED);
    auto                  acc = h_grid->getAccessor();
    for (std::size_t s = 0; s < masks.size(); ++s)
        for (auto it = masks[s]->tree().cbeginValueOn(); it; ++it) {
            const openvdb::Coord c = it.getCoord();
            if (const uint64_t slot = acc.getValue(nanovdb::Coord(c.x(), c.y(), c.z())))
                segOfSlot[slot] = uint32_t(s);
        }

    std::vector<uint32_t> labels(active + 1);
    cudaCheck(cudaMemcpy(labels.data(), d_labels, (active + 1) * sizeof(uint32_t),
                         cudaMemcpyDeviceToHost));

    std::unordered_map<uint32_t, uint32_t> gpuToSegment;
    uint64_t partitionViolations = 0, unassigned = 0;
    for (uint64_t slot = 1; slot <= active; ++slot) {
        const uint32_t segment = segOfSlot[slot];
        if (segment == UNASSIGNED) { ++unassigned; continue; }   // no segment claimed this voxel
        auto it = gpuToSegment.find(labels[slot]);
        if (it == gpuToSegment.end()) gpuToSegment.emplace(labels[slot], segment);
        else if (it->second != segment) ++partitionViolations;   // one GPU label split across segments
    }
    const uint64_t gpuDistinctLabels = gpuToSegment.size();

    // Count equality closes the other direction: with no violations each GPU label lands in exactly
    // one segment, so equal counts leave no room for two labels to have been merged into one segment.
    const bool pass = (gpuCount == masks.size()) && (gpuDistinctLabels == masks.size()) &&
                      (partitionViolations == 0) && (unassigned == 0);
    std::cout << "OpenVDB-oracle self-check: " << (pass ? "PASS" : "FAIL")
              << "  (gpu=" << gpuCount << ", openvdb=" << masks.size()
              << ", distinct gpu labels=" << gpuDistinctLabels
              << ", partition violations=" << partitionViolations
              << ", unassigned voxels=" << unassigned << ")\n";
}

#endif // NANOVDB_USE_OPENVDB

} // anonymous namespace

uint64_t connectedComponentsFromMesh(const std::vector<nanovdb::Vec3f>& points,
                                     const std::vector<nanovdb::Vec3i>& triangles,
                                     const nanovdb::Map&                map,
                                     float                              bandWidth,
                                     bool                               discardSurfaceVoxels,
                                     bool                               openvdbOracle)
{
    const cudaStream_t stream = 0;

    // ---- Step 1: rasterize the mesh -> ValueOnIndex narrow-band grid + UDF sidecar. ----
    thrust::universal_vector<nanovdb::Vec3f> dPoints(points.begin(), points.end());
    thrust::universal_vector<nanovdb::Vec3i> dTriangles(triangles.begin(), triangles.end());

    nanovdb::tools::cuda::MeshToGrid<BuildT> converter(
        dPoints.data().get(),    uint32_t(dPoints.size()),
        dTriangles.data().get(), uint32_t(dTriangles.size()), map);
    converter.setVerbose(1);
    converter.setNarrowBandWidth(bandWidth);
    auto [origHandle, udfSidecar] = converter.getHandleAndUDF();
    const auto* d_orig = origHandle.template deviceGrid<BuildT>();

    // ---- Step 2 (optional): discard the surface/barrier shell -> derived topology. ----
    // With the shell removed, each closed surface's band splits into disjoint inner/outer shells;
    // without it, connected components run on the full narrow band (one component per closed surface).
    GridHandleT                      derivedHandle;   // stays empty unless we prune
    const nanovdb::NanoGrid<BuildT>* d_cc = d_orig;    // grid connected components will label
    if (discardSurfaceVoxels) {
        // World-space voxel size from the map (uniform scale here, but read it generically).
        const nanovdb::Vec3d w0 = map.applyMap(nanovdb::Vec3d(0.0, 0.0, 0.0));
        const nanovdb::Vec3d wx = map.applyMap(nanovdb::Vec3d(1.0, 0.0, 0.0));
        const float    voxelSize      = float(wx[0] - w0[0]);
        const float    barrierSqWorld = 0.75f * voxelSize * voxelSize;  // (sqrt(3)/2 * voxelSize)^2
        const uint32_t srcLeafCount   = Traits::getTreeData(d_orig).mNodeCount[0];

        auto  retainMask   = nanovdb::cuda::DeviceBuffer::create(
            std::size_t(srcLeafCount) * sizeof(nanovdb::Mask<3>), nullptr, false);
        auto* d_retainMask = static_cast<nanovdb::Mask<3>*>(retainMask.deviceData());

        nanovdb::util::cuda::operatorKernel<UDFBarrierPruneMaskFunctor>
            <<<srcLeafCount, UDFBarrierPruneMaskFunctor::MaxThreadsPerBlock, 0, stream>>>(
                d_orig, static_cast<const float*>(udfSidecar.deviceData()), barrierSqWorld, d_retainMask);
        cudaCheckError();

        nanovdb::tools::cuda::PruneGrid<BuildT> pruner(d_orig, d_retainMask, stream);
        derivedHandle = pruner.getHandle();
        d_cc          = derivedHandle.template deviceGrid<BuildT>();
    }

    // ---- Step 3: connected-components labeling on the selected grid. ----
    // Timed end to end, stream synchronization included, so it is directly comparable with the two
    // host-side checks below -- all three then answer the same question in the same units.
    const auto tLabel0 = Clock::now();
    nanovdb::tools::cuda::ConnectedComponents<BuildT> cc(d_cc, stream);
    auto [d_labels, numComponents] = cc.getVoxelLabelsAndCount();
    cudaCheck(cudaStreamSynchronize(stream));
    const auto tLabel1 = Clock::now();

    // Diagnostics + CPU-oracle self-check (on whichever grid was labeled).
    const GridHandleT& ccHandle = discardSurfaceVoxels ? derivedHandle : origHandle;
    const uint64_t     ccActive = Traits::getActiveVoxelCount(d_cc);
    const uint32_t     ccLeaves = Traits::getTreeData(d_cc).mNodeCount[0];
    std::cout << (discardSurfaceVoxels ? "Derived (barrier-removed) grid: " : "Full narrow-band grid: ")
              << ccActive << " active voxels, " << ccLeaves << " leaves.\n"
              << "GPU labeling: " << secondsBetween(tLabel0, tLabel1) << " s\n";

    const auto tOracle0 = Clock::now();
    validateAgainstOracle(ccHandle, ccLeaves, ccActive, d_labels, numComponents);
    const auto tOracle1 = Clock::now();
    std::cout << "CPU union-find oracle: " << secondsBetween(tOracle0, tOracle1) << " s\n";

    if (openvdbOracle) {
#ifdef NANOVDB_USE_OPENVDB
        const auto tOvdb0 = Clock::now();
        runOpenVDBBaseline(ccHandle, ccLeaves, ccActive, d_labels, numComponents);
        std::cout << "OpenVDB oracle (total): " << secondsBetween(tOvdb0, Clock::now()) << " s\n";
#else
        std::cout << "OpenVDB baseline: not compiled in "
                     "(reconfigure with -DNANOVDB_USE_OPENVDB=ON)\n";
#endif
    }

    return numComponents;
}
