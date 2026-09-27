// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file propagate_sign_test.cu
///
/// @brief Runtime test for tools::cuda::PropagateSign. Each input gets the checks it supports:
///          - equivalence with sdf_detail::buildInvertMasks after adding active-inside leaf bits,
///            for inputs meeting the narrow-band premise (every face that
///            touches an inactive tile has one sign), where the two differ by design otherwise;
///          - agreement with the exact sign of the analytic shape an input samples, at every inactive
///            element: inactive leaf voxels, childless lower and upper slots, and absent root cells;
///          - active leaf bits match the sign sidecar, and child slots stay off;
///          - repeatability.
///
///        Usage: propagate_sign_test   (the exit code is 0 iff every case passes)

#include <nanovdb/NanoVDB.h>
#include <nanovdb/HostBuffer.h>
#include <nanovdb/cuda/HandleStorage.h>         // cuda::copyTo
#include <nanovdb/tools/CreateNanoGrid.h>       // tools::createNanoGrid
#include <nanovdb/tools/CreatePrimitives.h>     // tools::createLevelSetSphere, ...Torus, ...Box
#include <nanovdb/tools/GridBuilder.h>          // tools::build::Grid
#include <nanovdb/tools/cuda/MeshToSDF.cuh>     // reference: MeshToSDF, sdf_detail::buildInvertMasks
#include <nanovdb/tools/cuda/PropagateSign.cuh> // candidate

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace {

using BuildT   = nanovdb::ValueOnIndex;
using GridT    = nanovdb::NanoGrid<BuildT>;
using LeafT    = nanovdb::NanoLeaf<BuildT>;
using LowerT   = nanovdb::NanoLower<BuildT>;
using UpperT   = nanovdb::NanoUpper<BuildT>;
using CoordT   = nanovdb::Coord;
using InsideFn = std::function<bool(const CoordT&)>;

constexpr int LeafDim     = int(LeafT::DIM);
constexpr int RootCellDim = int(UpperT::DIM);

//------------------------------------------------------------------------------------------------
// Running PropagateSign and the reference

/// One implementation's result, copied to the host as raw words so that both implementations are
/// compared by the same code regardless of the buffer types they return.
struct HostMasks
{
    std::vector<uint64_t> leaf, lower, upper; // Mask<3>, Mask<4>, Mask<5> words, node after node
    std::vector<CoordT>   rootTiles;          // origins of interior root tiles

    bool operator==(const HostMasks& other) const
    {
        return leaf == other.leaf && lower == other.lower && upper == other.upper && rootTiles == other.rootTiles;
    }
};

/// Copy a device buffer's elements to the host as raw words of type @a WordT.
template<typename WordT, typename BufferT>
std::vector<WordT> toHost(const BufferT& buffer)
{
    std::vector<WordT> host(buffer.size_bytes() / sizeof(WordT));
    if (!host.empty())
        cudaCheck(cudaMemcpy(host.data(), buffer.data(), buffer.size_bytes(), cudaMemcpyDeviceToHost));
    return host;
}

HostMasks runReference(const GridT* deviceGrid, const int8_t* deviceSigns)
{
    const auto masks = nanovdb::tools::cuda::sdf_detail::buildInvertMasks(deviceGrid, deviceSigns, cudaStream_t(0));
    const auto rootInterior = toHost<uint8_t>(masks.rootInterior);
    std::vector<CoordT> rootTiles;
    for (size_t cell = 0; cell < rootInterior.size(); ++cell) {
        if (!rootInterior[cell]) continue;
        const int k = int(cell % masks.rootDims[2]);
        const int j = int(cell / masks.rootDims[2] % masks.rootDims[1]);
        const int i = int(cell / (size_t(masks.rootDims[1]) * masks.rootDims[2]));
        rootTiles.push_back((masks.rootTileMin + CoordT(i, j, k)) << UpperT::TOTAL);
    }
    return {toHost<uint64_t>(masks.leaf), toHost<uint64_t>(masks.lower), toHost<uint64_t>(masks.upper), rootTiles};
}

HostMasks runCandidate(const GridT* deviceGrid, const int8_t* deviceSigns)
{
    nanovdb::tools::cuda::PropagateSign<BuildT> op(deviceGrid, deviceSigns);
    op.propagate();
    return {toHost<uint64_t>(op.getLeafInteriorMasks()), toHost<uint64_t>(op.getLowerInteriorMasks()),
            toHost<uint64_t>(op.getUpperInteriorMasks()), toHost<CoordT>(op.getRootInteriorTileOrigins())};
}

//------------------------------------------------------------------------------------------------
// Checks

/// Print how many bits the reference sets at one tree level and return how many bits the two
/// implementations disagree on, listing the first few by node index and slot.
uint64_t compareLevel(const char* level, const std::vector<uint64_t>& reference,
                      const std::vector<uint64_t>& candidate, uint32_t wordsPerNode)
{
    if (reference.size() != candidate.size()) {
        std::printf("  %-5s node counts differ: %zu vs %zu\n", level,
                    reference.size() / wordsPerNode, candidate.size() / wordsPerNode);
        return 1;
    }
    uint64_t interior = 0, mismatches = 0;
    for (size_t word = 0; word < reference.size(); ++word) {
        interior += nanovdb::util::countOn(reference[word]);
        for (uint64_t diff = reference[word] ^ candidate[word]; diff; diff &= diff - 1) {
            if (mismatches++ < 5)
                std::printf("  %-5s mismatch: node %zu, slot %zu\n", level, word / wordsPerNode,
                            (word % wordsPerNode) * 64 + nanovdb::util::findLowestOn(diff));
        }
    }
    std::printf("  %-5s %10llu interior, %llu mismatches\n", level,
                (unsigned long long)interior, (unsigned long long)mismatches);
    return mismatches;
}

bool contains(const std::vector<CoordT>& tiles, const CoordT& origin)
{
    return std::find(tiles.begin(), tiles.end(), origin) != tiles.end();
}

/// Return how many root tiles only one of the two implementations lists.
uint64_t compareRoot(const HostMasks& reference, const HostMasks& candidate)
{
    uint64_t mismatches = 0;
    auto countMissing = [&](const std::vector<CoordT>& tiles, const std::vector<CoordT>& others) {
        for (const CoordT& origin : tiles) {
            if (contains(others, origin)) continue;
            if (mismatches++ < 5) std::printf("  root  mismatch: tile (%d, %d, %d)\n", origin[0], origin[1], origin[2]);
        }
    };
    countMissing(reference.rootTiles, candidate.rootTiles);
    countMissing(candidate.rootTiles, reference.rootTiles);
    std::printf("  %-5s %10zu interior, %llu mismatches\n", "root", reference.rootTiles.size(),
                (unsigned long long)mismatches);
    return mismatches;
}

bool maskBit(const std::vector<uint64_t>& words, uint32_t wordsPerNode, uint64_t node, uint32_t n)
{
    return (words[node * wordsPerNode + (n >> 6)] >> (n & 63)) & 1u;
}

std::vector<uint64_t> activeInteriorBits(const GridT& grid, const int8_t* deviceSigns)
{
    std::vector<int8_t> signs(grid.valueCount());
    if (!signs.empty()) cudaCheck(cudaMemcpy(signs.data(), deviceSigns, signs.size(), cudaMemcpyDeviceToHost));

    const auto& tree = grid.tree();
    constexpr uint32_t WordsPerLeaf = nanovdb::Mask<3>::WORD_COUNT;
    std::vector<uint64_t> bits(size_t(tree.nodeCount(0)) * WordsPerLeaf, 0);
    for (uint64_t i = 0; i < tree.nodeCount(0); ++i) {
        const auto& leaf = tree.getFirstLeaf()[i];
        for (uint32_t n = 0; n < LeafT::SIZE; ++n)
            if (leaf.isActive(n) && signs[leaf.getValue(n)] == int8_t(-1))
                bits[i * WordsPerLeaf + (n >> 6)] |= uint64_t(1) << (n & 63);
    }
    return bits;
}

/// Root cells, in units of root cells, spanning the active voxels of @a grid.
nanovdb::CoordBBox rootCellRange(const GridT& grid)
{
    if (grid.tree().nodeCount(0) == 0) return nanovdb::CoordBBox();
    const auto& bbox = grid.tree().root().bbox();
    return nanovdb::CoordBBox(bbox.min() >> UpperT::TOTAL, bbox.max() >> UpperT::TOTAL);
}

bool sizesMatch(const GridT& grid, const HostMasks& masks)
{
    const auto& tree = grid.tree();
    const bool match = masks.leaf.size() == size_t(tree.nodeCount(0)) * nanovdb::Mask<3>::WORD_COUNT &&
                       masks.lower.size() == size_t(tree.nodeCount(1)) * nanovdb::Mask<4>::WORD_COUNT &&
                       masks.upper.size() == size_t(tree.nodeCount(2)) * nanovdb::Mask<5>::WORD_COUNT;
    if (!match) std::printf("  mask sizes do not match the grid\n");
    return match;
}

/// Count incorrect active leaf bits, interior bits on slots holding children, and root tiles that
/// are not absent root cells within the active range.
uint64_t countMisplacedBits(const GridT& grid, const HostMasks& masks,
                            const std::vector<uint64_t>& activeInside)
{
    const auto& tree = grid.tree();
    uint64_t misplaced = 0;
    for (uint64_t i = 0; i < tree.nodeCount(0); ++i)
        for (uint32_t w = 0; w < nanovdb::Mask<3>::WORD_COUNT; ++w)
            misplaced += nanovdb::util::countOn((masks.leaf[i * nanovdb::Mask<3>::WORD_COUNT + w] ^
                                                activeInside[i * nanovdb::Mask<3>::WORD_COUNT + w]) &
                                                tree.getFirstLeaf()[i].valueMask().words()[w]);
    for (uint64_t i = 0; i < tree.nodeCount(1); ++i)
        for (uint32_t w = 0; w < nanovdb::Mask<4>::WORD_COUNT; ++w)
            misplaced += nanovdb::util::countOn(masks.lower[i * nanovdb::Mask<4>::WORD_COUNT + w] &
                                                tree.getFirstLower()[i].childMask().words()[w]);
    for (uint64_t i = 0; i < tree.nodeCount(2); ++i)
        for (uint32_t w = 0; w < nanovdb::Mask<5>::WORD_COUNT; ++w)
            misplaced += nanovdb::util::countOn(masks.upper[i * nanovdb::Mask<5>::WORD_COUNT + w] &
                                                tree.getFirstUpper()[i].childMask().words()[w]);
    const auto cells = rootCellRange(grid);
    for (const CoordT& origin : masks.rootTiles)
        misplaced += (origin & uint32_t(RootCellDim - 1)) != CoordT(0) || !cells.isInside(origin >> UpperT::TOTAL) ||
                     tree.root().probeChild(origin) != nullptr;
    std::printf("  structure/sign: %llu invalid bits\n", (unsigned long long)misplaced);
    return misplaced;
}

/// Count the inactive elements whose computed sign disagrees with @a inside. A tile or an absent root
/// cell holds no active voxel, so under the premise it has one sign, tested at its center voxel.
uint64_t compareWithTruth(const GridT& grid, const HostMasks& masks, const InsideFn& inside)
{
    const auto& tree = grid.tree();
    uint64_t checked = 0, wrong = 0;
    auto test = [&](bool computedInside, const CoordT& ijk) { ++checked; wrong += computedInside != inside(ijk); };

    for (uint64_t i = 0; i < tree.nodeCount(0); ++i) {
        const auto& leaf = tree.getFirstLeaf()[i];
        for (uint32_t n = 0; n < LeafT::SIZE; ++n)
            if (!leaf.isActive(n)) test(maskBit(masks.leaf, nanovdb::Mask<3>::WORD_COUNT, i, n), leaf.offsetToGlobalCoord(n));
    }
    for (uint64_t i = 0; i < tree.nodeCount(1); ++i) {
        const auto& lower = tree.getFirstLower()[i];
        for (uint32_t n = 0; n < LowerT::SIZE; ++n)
            if (!lower.childMask().isOn(n))
                test(maskBit(masks.lower, nanovdb::Mask<4>::WORD_COUNT, i, n), lower.offsetToGlobalCoord(n).offsetBy(LeafDim / 2));
    }
    for (uint64_t i = 0; i < tree.nodeCount(2); ++i) {
        const auto& upper = tree.getFirstUpper()[i];
        for (uint32_t n = 0; n < UpperT::SIZE; ++n)
            if (!upper.childMask().isOn(n))
                test(maskBit(masks.upper, nanovdb::Mask<5>::WORD_COUNT, i, n), upper.offsetToGlobalCoord(n).offsetBy(int(LowerT::DIM) / 2));
    }
    const auto cells = rootCellRange(grid);
    for (int i = cells.min()[0]; i <= cells.max()[0]; ++i)
        for (int j = cells.min()[1]; j <= cells.max()[1]; ++j)
            for (int k = cells.min()[2]; k <= cells.max()[2]; ++k) {
                const CoordT origin = CoordT(i, j, k) << UpperT::TOTAL;
                if (!tree.root().probeChild(origin)) test(contains(masks.rootTiles, origin), origin.offsetBy(RootCellDim / 2));
            }
    std::printf("  truth: %llu of %llu inactive elements wrong\n", (unsigned long long)wrong,
                (unsigned long long)checked);
    return wrong;
}

/// Run PropagateSign on one input and apply the checks it supports: equivalence with the reference
/// if @a compareReference, agreement with @a truth if given, and always structure and repeatability.
bool check(const std::string& name, const GridT* deviceGrid, const int8_t* deviceSigns,
           const GridT& hostGrid, bool compareReference, const InsideFn* truth)
{
    const auto& tree = hostGrid.tree();
    const HostMasks candidate = runCandidate(deviceGrid, deviceSigns);
    const auto activeInside = activeInteriorBits(hostGrid, deviceSigns);
    const CoordT rootDims = rootCellRange(hostGrid).dim();
    std::printf("%s: %u leaf, %u lower, %u upper nodes, %d x %d x %d root cells\n", name.c_str(),
                tree.nodeCount(0), tree.nodeCount(1), tree.nodeCount(2), rootDims[0], rootDims[1], rootDims[2]);

    uint64_t failures = 0;
    if (compareReference) {
        const HostMasks reference = runReference(deviceGrid, deviceSigns);
        auto expectedLeaf = reference.leaf;
        for (size_t w = 0; w < expectedLeaf.size(); ++w) expectedLeaf[w] |= activeInside[w];
        failures += compareLevel("leaf", expectedLeaf, candidate.leaf, nanovdb::Mask<3>::WORD_COUNT);
        failures += compareLevel("lower", reference.lower, candidate.lower, nanovdb::Mask<4>::WORD_COUNT);
        failures += compareLevel("upper", reference.upper, candidate.upper, nanovdb::Mask<5>::WORD_COUNT);
        failures += compareRoot(reference, candidate);
    }
    if (sizesMatch(hostGrid, candidate)) {
        if (truth) failures += compareWithTruth(hostGrid, candidate, *truth);
        failures += countMisplacedBits(hostGrid, candidate, activeInside);
    } else {
        ++failures;
    }
    const bool repeatable = runCandidate(deviceGrid, deviceSigns) == candidate;
    std::printf("  repeatable: %s\n", repeatable ? "yes" : "NO");
    failures += repeatable ? 0 : 1;

    std::printf("  %s\n\n", failures ? "FAIL" : "PASS");
    return failures == 0;
}

//------------------------------------------------------------------------------------------------
// Inputs produced by the MeshToSDF pipeline

struct Mesh
{
    std::vector<nanovdb::Vec3f> points;
    std::vector<nanovdb::Vec3i> triangles;
};

/// Append a closed UV sphere: two poles joined by (stacks - 1) rings of slices vertices each.
void addSphere(Mesh& mesh, const nanovdb::Vec3f& center, float radius, int stacks = 64, int slices = 128)
{
    const float pi    = 3.14159265358979f;
    const int   north = int(mesh.points.size());
    mesh.points.push_back(center + nanovdb::Vec3f(0.f, 0.f, radius));
    for (int i = 1; i < stacks; ++i) {
        const float theta = pi * float(i) / float(stacks);
        for (int j = 0; j < slices; ++j) {
            const float phi = 2.f * pi * float(j) / float(slices);
            mesh.points.push_back(center + radius * nanovdb::Vec3f(std::sin(theta) * std::cos(phi),
                                                                   std::sin(theta) * std::sin(phi),
                                                                   std::cos(theta)));
        }
    }
    const int south = int(mesh.points.size());
    mesh.points.push_back(center - nanovdb::Vec3f(0.f, 0.f, radius));

    auto ring = [&](int i, int j) { return north + 1 + (i - 1) * slices + j % slices; };
    for (int j = 0; j < slices; ++j) {
        mesh.triangles.emplace_back(north, ring(1, j), ring(1, j + 1));
        mesh.triangles.emplace_back(south, ring(stacks - 1, j + 1), ring(stacks - 1, j));
    }
    for (int i = 1; i < stacks - 1; ++i) {
        for (int j = 0; j < slices; ++j) {
            mesh.triangles.emplace_back(ring(i, j), ring(i + 1, j), ring(i + 1, j + 1));
            mesh.triangles.emplace_back(ring(i, j), ring(i + 1, j + 1), ring(i, j + 1));
        }
    }
}

/// Append an axis-aligned box as two triangles per face.
void addBox(Mesh& mesh, const nanovdb::Vec3f& lo, const nanovdb::Vec3f& hi)
{
    const int base = int(mesh.points.size());
    for (int corner = 0; corner < 8; ++corner) // bit a of corner selects hi along axis a
        mesh.points.emplace_back(corner & 1 ? hi[0] : lo[0], corner & 2 ? hi[1] : lo[1], corner & 4 ? hi[2] : lo[2]);
    const int faces[6][4] = {{0, 2, 6, 4}, {1, 5, 7, 3}, {0, 4, 5, 1}, {2, 3, 7, 6}, {0, 1, 3, 2}, {4, 6, 7, 5}};
    for (const auto& face : faces) {
        mesh.triangles.emplace_back(base + face[0], base + face[1], base + face[2]);
        mesh.triangles.emplace_back(base + face[0], base + face[2], base + face[3]);
    }
}

/// Run MeshToSDF on @a mesh and check PropagateSign on the grid and signs it produced; @a truth is
/// the exact sign of the mesh's shape in index space.
bool checkMesh(const std::string& name, const Mesh& mesh, double voxelSize, const InsideFn& truth)
{
    nanovdb::cuda::Buffer<nanovdb::Vec3f> points(cudaStream_t(0), mesh.points.size(), nanovdb::cuda::noInit);
    nanovdb::cuda::Buffer<nanovdb::Vec3i> triangles(cudaStream_t(0), mesh.triangles.size(), nanovdb::cuda::noInit);
    cudaCheck(cudaMemcpy(points.data(), mesh.points.data(), points.size_bytes(), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(triangles.data(), mesh.triangles.data(), triangles.size_bytes(), cudaMemcpyHostToDevice));

    nanovdb::tools::cuda::MeshToSDF<BuildT> sdf(points.data(), uint32_t(mesh.points.size()),
                                                triangles.data(), uint32_t(mesh.triangles.size()),
                                                nanovdb::Map(voxelSize));
    // The returned self-contained grid is not needed: deviceGrid() and deviceSign() keep referring
    // to the pipeline's own grid and signs, the exact inputs of its buildInvertMasks call.
    sdf.getHandle();
    const auto hostHandle = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(sdf.gridHandle());
    return check(name, sdf.deviceGrid(), sdf.deviceSign(), *hostHandle.grid<BuildT>(), true, &truth);
}

//------------------------------------------------------------------------------------------------
// Inputs built on the host

/// Upload a host index grid together with the sign sidecar PropagateSign reads (slot
/// leaf.getValue(n) holds -1 inside and +1 outside; slot 0 is the background), taking the sign of
/// each active voxel from @a activeInside, and check it.
bool checkIndexGrid(const std::string& name, const nanovdb::GridHandle<nanovdb::HostBuffer>& hostHandle,
                    const InsideFn& activeInside, bool compareReference, const InsideFn* truth)
{
    const GridT* grid = hostHandle.grid<BuildT>();
    std::vector<int8_t> signs(grid->valueCount(), int8_t(1));
    const LeafT* leaves = grid->tree().getFirstLeaf();
    for (uint32_t i = 0; i < grid->tree().nodeCount(0); ++i) {
        for (uint32_t n = 0; n < LeafT::SIZE; ++n) {
            if (leaves[i].isActive(n))
                signs[leaves[i].getValue(n)] = activeInside(leaves[i].offsetToGlobalCoord(n)) ? int8_t(-1) : int8_t(1);
        }
    }

    const auto deviceHandle = nanovdb::cuda::copyTo<nanovdb::cuda::Buffer<std::byte>>(hostHandle);
    nanovdb::cuda::Buffer<int8_t> deviceSigns(cudaStream_t(0), signs.size(), nanovdb::cuda::noInit);
    cudaCheck(cudaMemcpy(deviceSigns.data(), signs.data(), signs.size(), cudaMemcpyHostToDevice));
    return check(name, deviceHandle.deviceGrid<BuildT>(), deviceSigns.data(), *grid, compareReference, truth);
}

/// Index a NanoVDB float level set, whose active voxels form a narrow band of half-width 3 around
/// the analytic shape with exact sign @a truth, signing each active voxel by its distance value.
bool checkLevelSet(const std::string& name, const nanovdb::GridHandle<nanovdb::HostBuffer>& levelSet,
                   const InsideFn& truth)
{
    const nanovdb::FloatGrid& distance = *levelSet.grid<float>();
    const auto hostHandle = nanovdb::tools::createNanoGrid<nanovdb::FloatGrid, BuildT>(
        distance, /*channels*/ 0u, /*includeStats*/ false, /*includeTiles*/ false);
    const auto accessor = distance.getAccessor();
    return checkIndexGrid(name, hostHandle, [&](const CoordT& ijk) { return accessor.getValue(ijk) < 0.f; },
                          true, &truth);
}

/// Index the voxels @a active, signing each by @a inside.
bool checkSynthetic(const std::string& name, const std::vector<CoordT>& active, const InsideFn& inside,
                    bool compareReference, const InsideFn* truth = nullptr)
{
    nanovdb::tools::build::Grid<float> source(0.f);
    auto accessor = source.getAccessor();
    for (const CoordT& ijk : active) accessor.setValue(ijk, 0.f); // only the activation matters
    const auto hostHandle = nanovdb::tools::createNanoGrid<nanovdb::tools::build::Grid<float>, BuildT>(
        source, /*channels*/ 0u, /*includeStats*/ false, /*includeTiles*/ false);
    return checkIndexGrid(name, hostHandle, inside, compareReference, truth);
}

/// Activate each voxel of the leaf at @a origin with probability @a density, so that the leaf's
/// inactive voxels form pockets of varying size and connectivity.
void addLeaf(std::mt19937& rng, const CoordT& origin, float density, std::vector<CoordT>& active)
{
    std::bernoulli_distribution on(density);
    for (int i = 0; i < LeafDim; ++i)
        for (int j = 0; j < LeafDim; ++j)
            for (int k = 0; k < LeafDim; ++k)
                if (on(rng)) active.push_back(origin.offsetBy(i, j, k));
}

/// @a clusters centers drawn uniformly from the box [@a lo, @a hi), each surrounded by
/// @a leavesPerCluster randomly filled leaves whose origins scatter normally by @a spread voxels.
std::vector<CoordT> clusteredLeaves(std::mt19937& rng, const CoordT& lo, const CoordT& hi,
                                    int clusters, int leavesPerCluster, float spread)
{
    std::uniform_real_distribution<float> unit(0.f, 1.f);
    std::normal_distribution<float>       scatter(0.f, spread);
    std::vector<CoordT>                   active;
    for (int c = 0; c < clusters; ++c) {
        CoordT center;
        for (int axis = 0; axis < 3; ++axis) center[axis] = lo[axis] + int(unit(rng) * float(hi[axis] - lo[axis]));
        for (int l = 0; l < leavesPerCluster; ++l) {
            CoordT origin;
            for (int axis = 0; axis < 3; ++axis) origin[axis] = (center[axis] + int(scatter(rng))) & ~(LeafDim - 1);
            addLeaf(rng, origin, unit(rng), active);
        }
    }
    return active;
}

/// Leaves in all 26 root cells around the root cell [0, RootCellDim)^3, including leaves that touch
/// each of its faces from outside, so that it is the only absent root cell and nodes enclose it.
std::vector<CoordT> leavesAroundEmptyRootCell(std::mt19937& rng)
{
    std::uniform_int_distribution<int>    leafSlot(0, RootCellDim / LeafDim - 1);
    std::uniform_real_distribution<float> density(0.2f, 0.8f);
    auto randomOrigin = [&](const CoordT& cell) {
        CoordT origin;
        for (int axis = 0; axis < 3; ++axis) origin[axis] = cell[axis] * RootCellDim + leafSlot(rng) * LeafDim;
        return origin;
    };

    std::vector<CoordT> active;
    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            for (int k = -1; k <= 1; ++k) {
                if (i == 0 && j == 0 && k == 0) continue;
                for (int l = 0; l < 20; ++l) addLeaf(rng, randomOrigin(CoordT(i, j, k)), density(rng), active);
            }
        }
    }
    for (int axis = 0; axis < 3; ++axis) {
        for (int l = 0; l < 20; ++l) {
            CoordT origin = randomOrigin(CoordT(0));
            origin[axis] = l % 2 ? RootCellDim : -LeafDim; // alternate between the two opposite faces
            addLeaf(rng, origin, density(rng), active);
        }
    }
    return active;
}

/// A fixed pseudo-random bit per voxel, so that neighboring voxels disagree about half the time.
bool coinFlip(const CoordT& ijk)
{
    uint32_t h = uint32_t(ijk[0]) * 73856093u ^ uint32_t(ijk[1]) * 19349663u ^ uint32_t(ijk[2]) * 83492791u;
    h ^= h >> 13;
    h *= 0x5bd1e995u;
    h ^= h >> 15;
    return h & 1u;
}

//------------------------------------------------------------------------------------------------
// Analytic shapes, in index space

bool insideSphere(const CoordT& p, const nanovdb::Vec3d& center, double radius)
{
    const double x = p[0] - center[0], y = p[1] - center[1], z = p[2] - center[2];
    return x * x + y * y + z * z < radius * radius;
}

bool insideBox(const CoordT& p, const nanovdb::Vec3d& lo, const nanovdb::Vec3d& hi)
{
    return lo[0] < p[0] && p[0] < hi[0] && lo[1] < p[1] && p[1] < hi[1] && lo[2] < p[2] && p[2] < hi[2];
}

/// The torus of createLevelSetTorus: a ring around the y axis.
bool insideTorus(const CoordT& p, const nanovdb::Vec3d& center, double majorRadius, double minorRadius)
{
    const double x = p[0] - center[0], y = p[1] - center[1], z = p[2] - center[2];
    const double ring = std::sqrt(x * x + z * z) - majorRadius;
    return ring * ring + y * y < minorRadius * minorRadius;
}

} // namespace

int main()
{
    int cases = 0, failures = 0;
    auto record = [&](bool passed) { ++cases; failures += passed ? 0 : 1; };
    const nanovdb::Vec3d origin(0.0);

    // Inputs that meet the narrow-band premise and sample an analytic shape: every check applies.
    {
        Mesh mesh;
        addSphere(mesh, nanovdb::Vec3f(0.f), 1.f);
        record(checkMesh("mesh: sphere", mesh, 0.01,
                         [&](const CoordT& p) { return insideSphere(p, origin, 1.0 / 0.01); }));
    }
    {
        Mesh mesh; // axis-aligned faces at voxel-unaligned offsets
        const nanovdb::Vec3f lo(-0.83f, -0.61f, -0.72f), hi(0.91f, 0.55f, 0.78f);
        const double voxelSize = 0.0123;
        addBox(mesh, lo, hi);
        record(checkMesh("mesh: box", mesh, voxelSize, [&](const CoordT& p) {
            return insideBox(p, nanovdb::Vec3d(lo) / voxelSize, nanovdb::Vec3d(hi) / voxelSize);
        }));
    }
    {
        Mesh mesh; // a shell: the inner sphere's inside is outside again
        addSphere(mesh, nanovdb::Vec3f(0.f), 1.f);
        addSphere(mesh, nanovdb::Vec3f(0.f), 0.55f);
        record(checkMesh("mesh: nested spheres", mesh, 0.01, [&](const CoordT& p) {
            return insideSphere(p, origin, 100.0) != insideSphere(p, origin, 55.0);
        }));
    }
    {
        Mesh mesh; // 10000 voxels apart, leaving absent root cells between them
        addSphere(mesh, nanovdb::Vec3f(-50.f, 0.f, 0.f), 0.3f);
        addSphere(mesh, nanovdb::Vec3f(50.f, 0.f, 0.f), 0.3f);
        record(checkMesh("mesh: two distant spheres", mesh, 0.01, [&](const CoordT& p) {
            return insideSphere(p, nanovdb::Vec3d(-5000, 0, 0), 30.0) || insideSphere(p, nanovdb::Vec3d(5000, 0, 0), 30.0);
        }));
    }
    {
        const nanovdb::Vec3d center(37, -21, 11); // large enough to enclose whole upper-node tiles
        record(checkLevelSet("level set: sphere", nanovdb::tools::createLevelSetSphere<float>(300.0, center),
                             [&](const CoordT& p) { return insideSphere(p, center, 300.0); }));
    }
    record(checkLevelSet("level set: torus", nanovdb::tools::createLevelSetTorus<float>(200.0, 60.0),
                         [&](const CoordT& p) { return insideTorus(p, origin, 200.0, 60.0); }));
    {
        const nanovdb::Vec3d center(5, -3, 7), halfSize(150, 90, 120);
        record(checkLevelSet("level set: box",
                             nanovdb::tools::createLevelSetBox<float>(2 * halfSize[0], 2 * halfSize[1], 2 * halfSize[2], center),
                             [&](const CoordT& p) { return insideBox(p, center - halfSize, center + halfSize); }));
    }

    // Inputs that meet the premise trivially, with one sign everywhere or clusters lying wholly on
    // one side of the sign boundary. With one sign everywhere, every inactive element, absent root
    // cells included, must take it.
    const InsideFn allInside = [](const CoordT&) { return true; };
    record(checkSynthetic("synthetic: empty grid", {}, allInside, true, &allInside));

    std::mt19937 rng(20260922);
    const auto enclosed = leavesAroundEmptyRootCell(rng);
    record(checkSynthetic("synthetic: enclosed absent root cell, all inside", enclosed, allInside, true, &allInside));

    std::vector<CoordT> distant;
    for (const CoordT& cell : {CoordT(-2, -1, 0), CoordT(2, 1, 0), CoordT(0, 0, 2)}) {
        const CoordT lo(cell[0] * RootCellDim + 1024, cell[1] * RootCellDim + 1024, cell[2] * RootCellDim + 1024);
        const auto   leaves = clusteredLeaves(rng, lo, lo.offsetBy(2048), 4, 30, 100.f);
        distant.insert(distant.end(), leaves.begin(), leaves.end());
    }
    record(checkSynthetic("synthetic: distant clusters, all inside", distant, allInside, true, &allInside));
    record(checkSynthetic("synthetic: distant clusters, half-space", distant,
                          [](const CoordT& ijk) { return ijk[0] + ijk[1] < 0; }, true));

    // Inputs that break the premise: faces touching tiles see both signs, where PropagateSign and the
    // reference resolve differently by design, so only structure and repeatability are checked.
    record(checkSynthetic("robustness: enclosed absent root cell, inside a ball", enclosed, [](const CoordT& ijk) {
        const double x = ijk[0] - 2048.0, y = ijk[1] - 2048.0, z = ijk[2] - 2048.0;
        return x * x + y * y + z * z < 5000.0 * 5000.0;
    }, false));
    const auto clusters = clusteredLeaves(rng, CoordT(-RootCellDim), CoordT(RootCellDim), 60, 40, 150.f);
    record(checkSynthetic("robustness: clustered leaves, random signs", clusters, coinFlip, false));
    record(checkSynthetic("robustness: clustered leaves, smooth signs", clusters, [](const CoordT& ijk) {
        return std::sin(float(ijk[0]) / 150.f) + std::sin(float(ijk[1]) / 190.f) + std::sin(float(ijk[2]) / 170.f) < 0.4f;
    }, false));

    std::printf("%d of %d cases passed\n", cases - failures, cases);
    return failures ? 1 : 0;
}
