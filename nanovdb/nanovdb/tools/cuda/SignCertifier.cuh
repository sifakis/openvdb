// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/tools/cuda/SignCertifier.cuh

    \authors JaeHyun Lee and Efty Sifakis

    \brief Certifies the exterior of a narrow-band unsigned distance field on a NanoVDB index grid by ball overlap.

    \warning The header file contains cuda device code so be sure
             to only include it in .cu files (or other .cuh files)
*/

#ifndef NVIDIA_TOOLS_CUDA_SIGNCERTIFIER_CUH_HAS_BEEN_INCLUDED
#define NVIDIA_TOOLS_CUDA_SIGNCERTIFIER_CUH_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h>
#include <nanovdb/cuda/Buffer.h>
#include <nanovdb/cuda/DeviceResource.h>
#include <nanovdb/util/cuda/DeviceGridTraits.cuh>
#include <nanovdb/util/cuda/Util.h> // for lambdaKernel, operatorKernel

#include <cub/cub.cuh>    // cub::BlockScan
#include <cuda/atomic>    // cuda::atomic_ref

#include <algorithm> // std::sort
#include <cmath>     // std::sqrt, std::nextafter, INFINITY
#include <cstdint>   // int32_t
#include <limits>    // std::numeric_limits
#include <numeric>   // std::gcd
#include <stdexcept> // std::runtime_error
#include <utility>   // std::move, std::swap
#include <vector>    // std::vector

namespace nanovdb {

namespace tools::cuda {

namespace certification::detail {

// Setup: the tap tables.

/// @brief One offset w = (dx, dy, dz) of a halo tile, stored as the device code uses it: delta, the index
///        difference between a tile cell and the cell w away from it, and length, |w| rounded up to a float.
struct Tap { int32_t delta; float length; };

/// @brief The halo tile of a leaf: its voxels plus Halo voxels beyond each face, in z-fastest order.
template <typename BuildT, int Halo>
struct HaloTile
{
    static constexpr int DIM  = int(NanoLeaf<BuildT>::DIM) + 2 * Halo;
    static constexpr int SIZE = DIM * DIM * DIM;
};

/// @brief The taps of the primitive offsets w in [-Halo, Halo]^3 with minLength <= |w| < maxLength, sorted
///        by length.
/// @details A voxel q is certified from a certified voxel q + w when the balls around them, with radii r their
///          distances to the surface, overlap: r(q) + r(q + w) > |w| + tolerance.
///          Multiples k*w (k > 1) are left out: certifying q from q + k*w can be replaced by k steps of w,
///          through the voxels in between, which are in the band and get certified too, so the final certified
///          set is the same.
///          Sorting puts the shortest offsets first: since r is at most the band half-width W, only
///          |w| < r(q) + W can certify q, so the scan for q can stop at the first longer one.
template <typename BuildT, int Halo>
std::vector<Tap> makeTaps(float minLength, float maxLength)
{
    constexpr int dim = HaloTile<BuildT, Halo>::DIM;
    // The loops below only visit [-Halo, Halo]^3; a primitive offset outside it is at least
    // sqrt((Halo + 1)^2 + 1) long, so this bound keeps every offset shorter than maxLength in the table
    NANOVDB_ASSERT(maxLength * maxLength <= float((Halo + 1) * (Halo + 1) + 1));

    std::vector<Tap> taps;
    for (int dx = -Halo; dx <= Halo; ++dx)
        for (int dy = -Halo; dy <= Halo; ++dy)
            for (int dz = -Halo; dz <= Halo; ++dz) {
                if (std::gcd(std::gcd(dx, dy), dz) != 1) continue; // w = 0, or a multiple of a shorter offset
                const int lengthSqr = dx * dx + dy * dy + dz * dz;
                if (float(lengthSqr) < minLength * minLength || float(lengthSqr) >= maxLength * maxLength) continue;
                // Rounded up, so the overlap test can only get stricter
                const double exact = std::sqrt(double(lengthSqr));
                float length = float(exact);
                if (length < exact) length = std::nextafter(length, std::numeric_limits<float>::infinity());
                taps.push_back({(dx * dim + dy) * dim + dz, length});
            }
    std::sort(taps.begin(), taps.end(), [](const Tap& a, const Tap& b) {
        return a.length < b.length || (a.length == b.length && a.delta < b.delta); });
    return taps;
}

} // namespace certification::detail

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

/// @brief Certifies voxels of a narrow band as outside a surface, starting from seed voxels known to be
///        outside. The ball of a voxel is centred at it, with radius r its distance to the surface, so it
///        contains no surface; a voxel whose ball overlaps the ball of a certified voxel is on the same side,
///        and gets certified. Lengths are in voxels.
template <typename BuildT, typename ResourceT = nanovdb::cuda::DeviceResource>
class SignCertifier
{
    static_assert(nanovdb::cuda::is_async_resource<ResourceT>::value,
                  "SignCertifier allocates stream-ordered scratch and requires an AsyncResource");

    using GridT = NanoGrid<BuildT>;
    using TapT  = certification::detail::Tap;

    /// @brief Stream-ordered, device-only storage borrowed from the injected resource.
    template <typename T>
    using BufT = nanovdb::cuda::Buffer<T, nanovdb::cuda::ResourceRef<ResourceT>>;

public:

    /// @brief Half-width W of the narrow band: no active voxel is farther than W from the surface.
    /// @note Fixed, so the halos below are fixed too. A wider band needs larger halos; halo 8, the largest
    ///       that stays within the 3x3x3 leaves around a leaf, has a 24^3 float tile (55 KB), more than the
    ///       48 KB of static shared memory a block can use.
    static constexpr float BandWidth = 3.f;

    /// @brief Barrier threshold D: voxels closer than D to the surface are barrier voxels. Face neighbours at
    ///        least D from the surface are on the same side of it, since 2D > 1, so a seed grown through face
    ///        neighbours that are not barrier voxels stays outside; D = 0.55 rather than 1/2 keeps the voxels
    ///        of a surface at a half-integer offset (udf = 1/2 exactly) in the barrier.
    static constexpr float BarrierThreshold = 0.55f;

    /// @brief Halo widths: how many voxels a halo tile extends beyond each face of a leaf. The default pass,
    ///        run in both modes, uses the offsets w with |w| < W + D; the pedantic pass, which only pedantic
    ///        mode adds, uses W + D <= |w| < 2W. A primitive w (not a multiple of a shorter offset) has a
    ///        component above a halo width h only if |w| >= sqrt((h + 1)^2 + 1), so each set of offsets fits
    ///        in [-h, h]^3.
    static constexpr int DefaultHalo  = 3;
    static constexpr int PedanticHalo = 5;

    /// @brief How many times a round re-tests the candidates of a leaf over its loaded tile, using the
    ///        voxels certified by the previous test; what is left goes to the next round.
    static constexpr int MaxLocalIterations = 8;

    static_assert(BarrierThreshold > .5f && BarrierThreshold <= 1.f && BandWidth >= 3.f * BarrierThreshold,
                  "Seeding by face neighbours needs D > 1/2, and leaving out multiples of offsets needs D <= 1 and W >= 3D");
    static_assert((BandWidth + BarrierThreshold) * (BandWidth + BarrierThreshold) <= (DefaultHalo + 1) * (DefaultHalo + 1) + 1,
                  "The offsets of the default pass must fit in [-DefaultHalo, DefaultHalo]^3");
    static_assert(4.f * BandWidth * BandWidth <= (PedanticHalo + 1) * (PedanticHalo + 1) + 1,
                  "The offsets of the pedantic pass must fit in [-PedanticHalo, PedanticHalo]^3");

    /// @brief Constructor
    /// @param d_grid device index grid of the narrow band around one connected surface, without active tiles
    /// @param d_udf unsigned distance to the surface of each active voxel, in world units, indexed by
    ///        the grid's value index (slot 0 is the background)
    /// @param d_seedMasks one mask per leaf of the grid, in leaf order, whose on bits are the voxels known
    ///        to be outside the surface; certification starts from them
    /// @param stream optional CUDA stream (defaults to CUDA stream 0)
    /// @param resource resource instance all device memory is allocated from;
    ///        must outlive this operator (defaults to the per-type default resource)
    /// @warning Every seed voxel must be outside the surface: an inside one makes the result wrong.
    SignCertifier(const GridT* d_grid, const float* d_udf, const Mask<3>* d_seedMasks, cudaStream_t stream = 0,
                  ResourceT& resource = nanovdb::cuda::default_resource<ResourceT>())
        : mStream(stream), mDeviceGrid(d_grid), mDeviceUDF(d_udf), mDeviceSeedMasks(d_seedMasks), mResource(&resource)
        , mCertifiedMasks(stream, resource, 0, nanovdb::cuda::noInit)
        , mRoundStartMasks(stream, resource, 0, nanovdb::cuda::noInit)
        , mHasCandidates(stream, resource, 0, nanovdb::cuda::noInit)
        , mDirtyCurr(stream, resource, 0, nanovdb::cuda::noInit)
        , mDirtyNext(stream, resource, 0, nanovdb::cuda::noInit)
        , mChanged(stream, resource, 0, nanovdb::cuda::noInit)
        , mDefaultTaps(stream, resource, 0, nanovdb::cuda::noInit)
        , mPedanticTaps(stream, resource, 0, nanovdb::cuda::noInit) {}

    /// @brief Sets the overlap tolerance tau: a voxel q is certified from a certified voxel q + w only if
    ///        r(q) + r(q + w) > |w| + tau.
    void setTolerance(float tolerance = 1e-3f) { mTolerance = tolerance; }

    /// @brief Toggles pedantic mode (off by default). It adds the pedantic pass: once the default pass
    ///        certifies nothing more, the voxels that are not barrier voxels are tested against the offsets
    ///        W + D <= |w| < 2W, alternating with the default pass until neither certifies anything. Without
    ///        it, a voxel reachable only through such an offset stays uncertified.
    void setPedantic(bool pedantic = true) { mPedantic = pedantic; }

    /// @brief Toggles reference mode (off by default), for testing: every round processes every leaf with
    ///        candidates, tests each candidate once, and reads the certified masks only as they were at the
    ///        start of the round, so a round does not depend on the order its blocks run in. It certifies the
    ///        same voxels as the default mode, more slowly.
    void setReferenceMode(bool reference = true) { mReferenceMode = reference; }

    /// @brief Returns one mask per leaf of the grid, in leaf order, whose on bits are the voxels certified
    ///        to be outside the surface: the seed voxels and every voxel reached from them. Ownership passes
    ///        to the caller.
    BufT<Mask<3>> getCertifiedMasks()
    {
        initialize();
        processDefaultPass(); // W + D pass
        while (mPedantic && processPedanticPass()) processDefaultPass();
        cudaCheck(cudaStreamSynchronize(mStream));
        return std::move(mCertifiedMasks);
    }

private:

    // Stage 1: the certified masks start as the seed masks (active voxels only); every leaf with a
    // candidate, an active voxel not certified yet, is dirty. Uploads the tap tables.
    void initialize();

    // Stage 2: rounds of the default pass until a round certifies nothing.
    void processDefaultPass();

    // Stage 3: one round of the pedantic pass over every leaf; returns whether it certified anything.
    bool processPedanticPass();

    /// @brief One round: one block per leaf, each certifying the candidates of its leaf at least minRadius
    ///        from the surface with the taps; skips the leaves that are not dirty unless allLeaves. Returns
    ///        whether the round certified anything.
    template <int Halo>
    bool sweep(const BufT<TapT>& taps, float minRadius, bool allLeaves);

    /// @brief Copies a tap table to the device.
    BufT<TapT> upload(const std::vector<TapT>& taps)
    {
        BufT<TapT> deviceTaps(mStream, this->ref(), taps.size(), nanovdb::cuda::noInit);
        // The source is pageable, so the copy has been staged by the time the call returns
        cudaCheck(cudaMemcpyAsync(deviceTaps.data(), taps.data(), deviceTaps.size_bytes(), cudaMemcpyHostToDevice, mStream));
        return deviceTaps;
    }

    /// @brief Borrowed handle to the injected resource, for the storage allocated after construction.
    nanovdb::cuda::ResourceRef<ResourceT> ref() { return nanovdb::cuda::ResourceRef<ResourceT>(*mResource); }

    cudaStream_t   mStream{0};
    const GridT   *mDeviceGrid;
    const float   *mDeviceUDF;
    const Mask<3> *mDeviceSeedMasks;
    ResourceT     *mResource;         // non-owning
    float          mTolerance{1e-3f};
    bool           mPedantic{false};
    bool           mReferenceMode{false};
    uint32_t       mLeafCount{0};

    BufT<Mask<3>>  mCertifiedMasks;   // leafCount x Mask<3>: certified voxels of each leaf, written by the leaf's block only
    BufT<Mask<3>>  mRoundStartMasks;  // leafCount x Mask<3>: reference mode only, the certified masks at the start of the round
    BufT<uint32_t> mHasCandidates;    // leafCount x uint32_t: 1 while the leaf has a candidate, written by the leaf's block only
    BufT<uint32_t> mDirtyCurr;        // leafCount x uint32_t: 1 if a voxel of the leaf or of a neighbour leaf was certified in the previous round
    BufT<uint32_t> mDirtyNext;        // leafCount x uint32_t: the same for the current round, for the next one
    BufT<uint32_t> mChanged;          // 1 x uint32_t: 1 if the current round certified anything
    BufT<TapT>     mDefaultTaps;      // taps of the default pass, sorted by length
    BufT<TapT>     mPedanticTaps;     // taps of the pedantic pass (pedantic mode only), sorted by length
};// tools::cuda::SignCertifier<BuildT, ResourceT>

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace certification::detail {

// Stage 1: initialization.

/// @brief Certified mask of a leaf = its seed mask restricted to its active voxels; the leaf has candidates,
///        and is dirty, if an active voxel is left uncertified.
template <typename BuildT>
struct InitializeFunctor
{
    __device__ void operator()(size_t leafID, const NanoGrid<BuildT>* d_grid, const Mask<3>* d_seedMasks,
                               Mask<3>* d_certifiedMasks, uint32_t* d_hasCandidates, uint32_t* d_dirty)
    {
        const auto& leaf = d_grid->tree().template getFirstNode<0>()[leafID];
        uint64_t uncertified = 0;
        for (uint32_t w = 0; w < Mask<3>::WORD_COUNT; ++w) {
            const uint64_t active    = leaf.valueMask().words()[w];
            const uint64_t certified = d_seedMasks[leafID].words()[w] & active;
            d_certifiedMasks[leafID].words()[w] = certified;
            uncertified |= active & ~certified;
        }
        d_hasCandidates[leafID] = d_dirty[leafID] = uint32_t(uncertified != 0);
    }
};

} // namespace certification::detail

template <typename BuildT, typename ResourceT>
void SignCertifier<BuildT, ResourceT>::initialize()
{
    const auto treeData = util::cuda::DeviceGridTraits<BuildT>::getTreeData(mDeviceGrid);
    if (treeData.mTileCount[0] || treeData.mTileCount[1] || treeData.mTileCount[2])
        throw std::runtime_error("nanovdb::tools::cuda::SignCertifier: grids with active tiles are not supported");

    mLeafCount = treeData.mNodeCount[0];
    mCertifiedMasks = BufT<Mask<3>>(mStream, this->ref(), mLeafCount, nanovdb::cuda::noInit);
    if (mReferenceMode) mRoundStartMasks = BufT<Mask<3>>(mStream, this->ref(), mLeafCount, nanovdb::cuda::noInit);
    mHasCandidates = BufT<uint32_t>(mStream, this->ref(), mLeafCount, nanovdb::cuda::noInit);
    mDirtyCurr     = BufT<uint32_t>(mStream, this->ref(), mLeafCount, nanovdb::cuda::noInit);
    mDirtyNext     = BufT<uint32_t>(mStream, this->ref(), mLeafCount, nanovdb::cuda::noInit);
    mChanged       = BufT<uint32_t>(mStream, this->ref(), 1, nanovdb::cuda::noInit);
    if (mLeafCount == 0) return;

    util::cuda::lambdaKernel<<<(mLeafCount + 127) / 128, 128, 0, mStream>>>(
        mLeafCount, certification::detail::InitializeFunctor<BuildT>(), mDeviceGrid, mDeviceSeedMasks,
        mCertifiedMasks.data(), mHasCandidates.data(), mDirtyCurr.data());
    cudaCheckError();

    mDefaultTaps = upload(certification::detail::makeTaps<BuildT, DefaultHalo>(0.f, BandWidth + BarrierThreshold));
    if (mPedantic)
        mPedanticTaps = upload(certification::detail::makeTaps<BuildT, PedanticHalo>(BandWidth + BarrierThreshold, 2.f * BandWidth));
}// SignCertifier<BuildT, ResourceT>::initialize

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace certification::detail {

// Stages 2 and 3: rounds of the default and pedantic passes.

/// @brief Relaxed, device-scope access to a word that blocks of the same launch read while another block
///        writes it: every writer of a word is its only writer, or writes the same value as the others.
template <typename T>
__device__ inline ::cuda::atomic_ref<T, ::cuda::thread_scope_device> atomicRef(const T& word)
{
    return ::cuda::atomic_ref<T, ::cuda::thread_scope_device>(const_cast<T&>(word));
}

/// @brief What one round uses besides the grid and the per-leaf arrays.
struct SweepSettings
{
    const Tap* taps;               // sorted by length
    int        tapCount;
    float      bandWidth;          // W
    float      tolerance;          // tau
    float      minRadius;          // only voxels at least this far from the surface are candidates
    int        maxLocalIterations; // tests of the candidates per round, over the same loaded tile
    bool       allLeaves;          // process the leaves that are not dirty too
};

/// @brief One round over one leaf: certifies the candidates of the leaf (its active voxels not certified
///        yet, at least minRadius from the surface) from the certified voxels within reach, using the taps
///        of a halo tile with Halo voxels beyond each face of the leaf.
/// @details The block skips a leaf with no candidates, or that is not dirty unless allLeaves. Otherwise it
///          lists the candidates, gathers the halo tile, cell c holding r (the distance to the surface) if
///          its voxel is certified and -inf if not, then tests each candidate against its taps, one warp per
///          candidate and one lane per tap, until one certifies it. Certifications are written into the tile
///          after each test of all candidates, so the next test can use them, until a test certifies nothing
///          or maxLocalIterations is reached; what remains is left to the next round.
///          The tile is gathered from d_readMasks and new certifications are written to d_certifiedMasks;
///          they are the same masks except in reference mode.
template <typename BuildT, int Halo>
struct SweepFunctor
{
    static constexpr int MaxThreadsPerBlock         = NanoLeaf<BuildT>::SIZE;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    using LeafT = NanoLeaf<BuildT>;
    using TileT = HaloTile<BuildT, Halo>;

    /// @brief Tile cell of voxel n of the leaf.
    __device__ static int cellOf(uint32_t n)
    {
        const Coord ijk = LeafT::OffsetToLocalCoord(n);
        return ((ijk[0] + Halo) * TileT::DIM + ijk[1] + Halo) * TileT::DIM + ijk[2] + Halo;
    }

    /// @brief Index, from 0 to 2, of the leaf a tile coordinate t falls in along one axis.
    __device__ static int leafSlotOf(int t) { return int(t >= Halo) + int(t >= Halo + int(LeafT::DIM)); }

    __device__ void operator()(const NanoGrid<BuildT>* d_grid, const float* d_udf, SweepSettings settings,
                               const Mask<3>* d_readMasks, Mask<3>* d_certifiedMasks, uint32_t* d_hasCandidates,
                               const uint32_t* d_dirtyCurr, uint32_t* d_dirtyNext, uint32_t* d_changed)
    {
        using BlockScanT = cub::BlockScan<int, MaxThreadsPerBlock>;
        constexpr int WarpCount = MaxThreadsPerBlock / 32;

        __shared__ uint32_t sNeighbourIDs[27];
        __shared__ float    sTile[TileT::SIZE];
        __shared__ uint16_t sCandidates[MaxThreadsPerBlock];  // voxel offsets
        __shared__ float    sRadii[MaxThreadsPerBlock];       // their distances to the surface
        __shared__ uint8_t  sStates[MaxThreadsPerBlock];      // 0: uncertified, 1: certified in this test, 2: before
        __shared__ uint32_t sNewBits[WarpCount];              // voxels certified by this block, 32 per word
        __shared__ typename BlockScanT::TempStorage sScanStorage;

        const uint32_t leafID = blockIdx.x, tID = threadIdx.x, warpID = tID >> 5, laneID = tID & 31;
        // Uniform across the block: neither array is written for this leaf until the block is done with it
        if (!d_hasCandidates[leafID] || (!settings.allLeaves && !d_dirtyCurr[leafID])) return;

        const auto* firstLeaf = d_grid->tree().template getFirstNode<0>();
        const auto& leaf      = firstLeaf[leafID];
        const float voxelSize = float(d_grid->voxelSize()[0]);

        // List the candidates
        bool  isCandidate = leaf.isActive(tID) && !d_readMasks[leafID].isOn(tID);
        float radius      = 0.f;
        if (isCandidate) {
            radius      = d_udf[leaf.getValue(tID)] / voxelSize;
            isCandidate = radius >= settings.minRadius;
        }
        int index, candidateCount;
        BlockScanT(sScanStorage).ExclusiveSum(int(isCandidate), index, candidateCount);
        if (candidateCount == 0) return;
        if (isCandidate) {
            sCandidates[index] = uint16_t(tID);
            sRadii[index]      = radius;
            sStates[index]     = 0;
        }

        // The 27 leaves around this one, itself at 13, as leaf indices, or UINT32_MAX where there is none
        if (tID < 27) {
            const int  dim       = int(LeafT::DIM);
            const auto neighbour = d_grid->tree().root().probeLeaf(
                leaf.origin().offsetBy((int(tID) / 9 - 1) * dim, (int(tID) / 3 % 3 - 1) * dim, (int(tID) % 3 - 1) * dim));
            sNeighbourIDs[tID] = neighbour ? uint32_t(neighbour - firstLeaf) : UINT32_MAX;
        }
        if (tID < WarpCount) sNewBits[tID] = 0;
        __syncthreads();

        // Gather the tile
        for (int c = int(tID); c < TileT::SIZE; c += MaxThreadsPerBlock) {
            const int i = c / (TileT::DIM * TileT::DIM), j = c / TileT::DIM % TileT::DIM, k = c % TileT::DIM;
            const uint32_t neighbourID = sNeighbourIDs[9 * leafSlotOf(i) + 3 * leafSlotOf(j) + leafSlotOf(k)];
            const uint32_t n = LeafT::CoordToOffset(Coord(i - Halo, j - Halo, k - Halo));
            float r = -INFINITY;
            if (neighbourID != UINT32_MAX &&
                (atomicRef(d_readMasks[neighbourID].words()[n >> 6]).load(::cuda::memory_order_relaxed) >> (n & 63) & 1))
                r = d_udf[firstLeaf[neighbourID].getValue(n)] / voxelSize;
            sTile[c] = r;
        }
        __syncthreads();

        for (int iteration = 0; iteration < settings.maxLocalIterations; ++iteration) {
            // Test each uncertified candidate; the tile stays unchanged during the test
            for (int i = int(warpID); i < candidateCount; i += WarpCount) {
                if (sStates[i]) continue;
                const int   centre = cellOf(sCandidates[i]);
                const float r      = sRadii[i];
                // Every voxel is within the band, so taps this long cannot certify
                const float reach  = r + settings.bandWidth;
                bool certified = false;
                for (int t = 0; !certified && t < settings.tapCount && settings.taps[t].length < reach; t += 32) {
                    bool overlap = false;
                    if (t + int(laneID) < settings.tapCount) {
                        const Tap tap = settings.taps[t + laneID];
                        overlap = sTile[centre + tap.delta] + r > tap.length + settings.tolerance;
                    }
                    certified = __any_sync(0xFFFFFFFFu, overlap);
                }
                if (certified && laneID == 0) sStates[i] = 1;
            }
            __syncthreads();

            // Write this test's certifications into the tile
            bool changed = false;
            for (int i = int(tID); i < candidateCount; i += MaxThreadsPerBlock)
                if (sStates[i] == 1) {
                    const uint32_t n = sCandidates[i];
                    sTile[cellOf(n)] = sRadii[i];
                    atomicOr(&sNewBits[n >> 5], 1u << (n & 31));
                    sStates[i] = 2;
                    changed = true;
                }
            if (!__syncthreads_or(changed)) break;
        }

        // Write back; mark this leaf and its neighbours dirty for the next round
        if (!__syncthreads_or(tID < WarpCount && sNewBits[tID] != 0)) return;
        bool uncertifiedLeft = false;
        if (tID < Mask<3>::WORD_COUNT) {
            const uint64_t newBits = uint64_t(sNewBits[2 * tID]) | uint64_t(sNewBits[2 * tID + 1]) << 32;
            const uint64_t word    = d_readMasks[leafID].words()[tID] | newBits;
            if (newBits) atomicRef(d_certifiedMasks[leafID].words()[tID]).store(word, ::cuda::memory_order_relaxed);
            uncertifiedLeft = (leaf.valueMask().words()[tID] & ~word) != 0;
        }
        const bool hasCandidates = __syncthreads_or(uncertifiedLeft);
        if (tID < 27 && sNeighbourIDs[tID] != UINT32_MAX)
            atomicRef(d_dirtyNext[sNeighbourIDs[tID]]).store(1u, ::cuda::memory_order_relaxed);
        if (tID == 0) {
            d_hasCandidates[leafID] = uint32_t(hasCandidates);
            atomicRef(*d_changed).store(1u, ::cuda::memory_order_relaxed);
        }
    }
};

} // namespace certification::detail

template <typename BuildT, typename ResourceT>
template <int Halo>
bool SignCertifier<BuildT, ResourceT>::sweep(const BufT<TapT>& taps, float minRadius, bool allLeaves)
{
    const certification::detail::SweepSettings settings{taps.data(), int(taps.size()), BandWidth, mTolerance, minRadius,
                                                        mReferenceMode ? 1 : MaxLocalIterations, allLeaves || mReferenceMode};
    const Mask<3>* d_readMasks = mCertifiedMasks.data();
    if (mReferenceMode) {
        cudaCheck(cudaMemcpyAsync(mRoundStartMasks.data(), mCertifiedMasks.data(), mCertifiedMasks.size_bytes(),
                                  cudaMemcpyDeviceToDevice, mStream));
        d_readMasks = mRoundStartMasks.data();
    }
    cudaCheck(cudaMemsetAsync(mDirtyNext.data(), 0, mDirtyNext.size_bytes(), mStream));
    cudaCheck(cudaMemsetAsync(mChanged.data(), 0, mChanged.size_bytes(), mStream));

    using SweepOp = certification::detail::SweepFunctor<BuildT, Halo>;
    util::cuda::operatorKernel<SweepOp><<<mLeafCount, SweepOp::MaxThreadsPerBlock, 0, mStream>>>(
        mDeviceGrid, mDeviceUDF, settings, d_readMasks, mCertifiedMasks.data(), mHasCandidates.data(),
        mDirtyCurr.data(), mDirtyNext.data(), mChanged.data());
    cudaCheckError();

    uint32_t changed = 0;
    cudaCheck(cudaMemcpyAsync(&changed, mChanged.data(), sizeof(changed), cudaMemcpyDeviceToHost, mStream));
    cudaCheck(cudaStreamSynchronize(mStream));
    std::swap(mDirtyCurr, mDirtyNext);
    return changed != 0;
}// SignCertifier<BuildT, ResourceT>::sweep

template <typename BuildT, typename ResourceT>
void SignCertifier<BuildT, ResourceT>::processDefaultPass()
{
    if (mLeafCount == 0) return;
    // A round that changes something certifies at least one voxel, so the loop ends
    while (sweep<DefaultHalo>(mDefaultTaps, 0.f, false)) {}
}// SignCertifier<BuildT, ResourceT>::processDefaultPass

template <typename BuildT, typename ResourceT>
bool SignCertifier<BuildT, ResourceT>::processPedanticPass()
{
    if (mLeafCount == 0) return false;
    // A pedantic offset needs r(q) + r(q + w) > W + D, so both voxels are at least D from the surface. The
    // leaves it marks dirty are where the next default pass resumes.
    return sweep<PedanticHalo>(mPedanticTaps, BarrierThreshold, true);
}// SignCertifier<BuildT, ResourceT>::processPedanticPass

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

}// namespace tools::cuda

}// namespace nanovdb

#endif // NVIDIA_TOOLS_CUDA_SIGNCERTIFIER_CUH_HAS_BEEN_INCLUDED
