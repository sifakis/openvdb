// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/tools/cuda/MeshToSDF.cuh

    \authors Efty Sifakis and JaeHyun Lee

    \brief GPU conversion of a triangle mesh into a narrow-band signed distance field on NanoVDB
           index grids. Builds on two domain-agnostic primitives:
             - nanovdb::tools::cuda::MeshToGrid            (mesh -> narrow-band ValueOnIndex + UDF),
             - nanovdb::tools::cuda::ConnectedComponents   (generic CC labeling of such a grid).

           MeshToSDF is the entry point. It partitions the rasterized band into closed surfaces,
           signs each one independently, and composes the results:

             1  rasterize                    MeshToGrid -> band + UDF + nearest-triangle index
             2  partition                    ConnectedComponents on the UN-PRUNED band. The barrier
                                             shell glues a surface's inner and outer sides together,
                                             so each closed surface is exactly one component
             3  per surface: carve it out and sign it alone -> its own complete sign field
             4  compose by nesting parity     a point wrapped by k surfaces is inside iff k is odd
             4b finalize magnitudes           report distance to the signed surface, floor the interior
             5  fill                          extend the composed sign off the band

           The surface signed is { udf == isoValue }, the mesh itself when isoValue is 0
           (setIsoValue()). Everything above works on the DISPLACEMENT udf - isoValue rather than on
           udf, so moving the surface moves the barrier shell, the components and the seed together
           instead of leaving them behind at the mesh. Step 4b is the one place the two are folded
           back into a single magnitude, and it runs last because folding destroys the side
           information the earlier steps depend on.

           Step 3 is sdf_detail::SurfaceSigner, whose stages are, in call order:

             computeDerivedTopology()  drop the surface shell, leaving the grid CC labels
             signNonBarrier()          the min-x component is the exterior; the rest is interior
             injectSignsToOriginal()   carry those signs back onto the un-pruned grid
             signBarrier()             sign the shell the first stage set aside
             fillLeafInvertMask()      \
             fillCoarseInvertMasks()    | extend the sign off the band, level by level
             fillRootInteriorMask()    /

           Partitioning first is what makes signNonBarrier's rule — the leftmost active voxel is
           outside — legitimate: it is a statement about ONE closed surface, and a carved grid is
           where it holds. A mesh with a single closed surface skips the carve, since the rasterized
           band already is that surface's band.

           A single signedSignAt() query composes the resulting sidecars into the sign at any
           coordinate. The grid is never rebuilt: everything the stages produce is a sidecar indexed
           by leaf.getValue(n), or a per-node invert mask.

           Reading order below: MeshToSDF first, then sdf_detail (SurfaceSigner, the CUDA functors
           grouped by the stage that launches them, and the query), then the method definitions.

    \warning The header file contains cuda device code so be sure
             to only include it in .cu files (or other .cuh files)
*/

#ifndef NVIDIA_TOOLS_CUDA_MESHTOSDF_CUH_HAS_BEEN_INCLUDED
#define NVIDIA_TOOLS_CUDA_MESHTOSDF_CUH_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h>
#include <nanovdb/GridHandle.h>
#include <nanovdb/cuda/DeviceBuffer.h>
#include <nanovdb/math/Proximity.h>                    // closestPointOnTriangleToPoint
#include <nanovdb/tools/cuda/ConnectedComponents.cuh>
#include <nanovdb/tools/cuda/MeshToGrid.cuh>
#include <nanovdb/tools/cuda/PruneGrid.cuh>
#include <nanovdb/util/cuda/Injection.cuh>             // InjectGridDataFunctor
#include <nanovdb/util/cuda/DeviceGridTraits.cuh>
#include <nanovdb/util/cuda/Timer.h>
#include <nanovdb/util/cuda/Util.h>                    // operatorKernel, cudaCheck

#include <chrono>
#include <memory>
#include <utility>
#include <vector>

namespace nanovdb {

namespace tools::cuda {

namespace sdf_detail { template <typename BuildT> class SurfaceSigner; }

/// @brief Convert a triangle mesh into a narrow-band signed distance field on a NanoVDB index grid,
///        correctly for any number of closed surfaces — separate objects, cavities and nesting alike.
///        See the file notes above for the pipeline.
/// @tparam BuildT Build type of the index grid (e.g. nanovdb::ValueOnIndex).
template <typename BuildT>
class MeshToSDF
{
    using GridT   = NanoGrid<BuildT>;
    using Handle  = GridHandle<nanovdb::cuda::DeviceBuffer>;
    using Buffer  = nanovdb::cuda::DeviceBuffer;
    using Signer  = sdf_detail::SurfaceSigner<BuildT>;

public:

    /// @brief Constructor. Mirrors MeshToGrid: the mesh lives on the device, the map is the
    ///        world<->index transform, and nothing runs until build().
    /// @param d_points       device vertex list of the input triangle surface (WORLD space)
    /// @param pointCount     vertex count
    /// @param d_triangles    device triangle vertex-index list
    /// @param triangleCount  triangle count
    /// @param map            affine map used for the conversion
    /// @param stream         optional CUDA stream (defaults to CUDA stream 0)
    MeshToSDF(const nanovdb::Vec3f* d_points, uint32_t pointCount,
              const nanovdb::Vec3i* d_triangles, uint32_t triangleCount,
              const nanovdb::Map& map = nanovdb::Map(), cudaStream_t stream = 0)
        : mPoints(d_points), mPointCount(pointCount)
        , mTriangles(d_triangles), mTriangleCount(triangleCount)
        , mMap(map), mStream(stream) {}

    /// @brief Toggle on and off verbose mode
    /// @param level Verbose level: 0=quiet, 1=timing
    void setVerbose(int level = 1) { mVerbose = level; }

    /// @brief Set desired width of the narrow band
    /// @param bandWidth Narrow band width in cell units
    void setNarrowBandWidth(float bandWidth = 3.f) { mBandWidth = bandWidth; }

    /// @brief How the barrier voxels -- the ones the surface passes through, which the connected
    ///        components stage has to remove and therefore leaves unsigned -- get their sign.
    enum class BarrierSigning {
        Interior,   ///< call every barrier voxel interior. No oracle, no neighbour search: the shell
                    ///< is exactly the set that might hold surface, so declaring it interior is the
                    ///< one choice that cannot put surface in the exterior set. Loosest of the three
                    ///< -- the reported surface can stand up to √3/2 voxels out -- and the baseline
                    ///< the other two are measured against.
        Heuristic,  ///< mirror of OpenVDB's ComputeIntersectingVoxelSign: find an exterior neighbour
                    ///< and test whether this voxel lies in the same half space about that
                    ///< neighbour's closest surface point. One-sided and approximate near curvature.
        Ball        ///< certify by ball intersection: the ball of radius udf(V) about a voxel cannot
                    ///< meet the surface, so two overlapping such balls prove their centres share a
                    ///< side. Two-sided and exact, but leaves a few percent of voxels unproven right
                    ///< at the surface, which then default to interior.
    };

    /// @brief Choose the barrier signing method (default Interior).
    void setBarrierSigning(BarrierSigning m) { mBarrierSigning = m; }

    /// @brief How a point enclosed by several of the input's closed surfaces is signed.
    enum class NestingRule {
        EvenOdd,  ///< inside iff an ODD number of surfaces enclose it (default). A shell inside a
                  ///< shell is a cavity, which is what a hollow model means, and it is the rule
                  ///< OpenVDB and most mesh formats follow.
        Solid     ///< inside iff ANY surface encloses it. Only the outermost boundary of each object
                  ///< separates inside from outside; whatever it wraps is filled, however much
                  ///< internal structure the mesh has. Use it when the interior detail is noise --
                  ///< a scan's inner shells, or a shrink wrap that only wants the outer hull.
    };

    /// @brief Choose the nesting rule (default EvenOdd).
    void setNestingRule(NestingRule r) { mNestingRule = r; }

    /// @brief Stencil half-width for BarrierSigning::Ball: 1 = the 26 neighbours (default), 2 = 5x5x5.
    ///        Costs (2r+1)^3 loads per voxel per round. Ball radii are capped at the band width, so
    ///        pairs further apart than 2*narrowBandWidth() never overlap and a wider stencil buys
    ///        nothing.
    void setBallStencilRadius(int radius) { mBallStencilRadius = radius; }

    /// @brief Sign the isosurface { x : udf(x) == isoValue } rather than the mesh itself, so the
    ///        result is the signed distance to a surface standing @a isoValue WORLD units off the
    ///        mesh. Must be >= 0; zero (the default) signs the mesh. This is the dilation a
    ///        shrink-wrap or offset pipeline asks for.
    ///
    /// The whole pipeline runs on udf - isoValue rather than on udf, from the barrier test onwards.
    /// That field is already signed in the only sense the pipeline needs -- negative within isoValue
    /// of the mesh, positive elsewhere -- and it has TWO zero crossings, one on each side of the
    /// mesh. The interior one is not a problem to be avoided but the reason the rest of the pipeline
    /// exists: dropping the barrier splits the band, and the component holding the minimum index-space
    /// voxel is the only one that is outside, so the crossing buried inside the mesh ends up enclosed
    /// by an interior component and is signed away. Subtracting the constant from the FINISHED signed
    /// field instead would leave the barrier shell hugging the mesh, isoValue away from the surface
    /// actually being signed.
    ///
    /// Only the reported magnitude is folded about the isovalue, as |udf - isoValue|, and only after
    /// every sign is settled.
    ///
    /// It also gives an OPEN mesh an interior. A mesh with a hole has none -- inside and outside are
    /// one region -- but { udf == isoValue } closes over any hole narrower than isoValue.
    ///
    /// @warning The band is rasterized out to isoValue + narrowBandWidth(), so everything within
    ///          isoValue of the mesh stays materialized and the cost grows accordingly.
    void setIsoValue(float isoValue = 0.f) { mIsoValue = isoValue; }

    /// @brief Run the whole pipeline. Afterwards the accessors below describe a complete sign field
    ///        over the rasterized band, extended off it by the invert masks.
    void build();//TODO: change the function name to more intuitive one buildSDF?.

    /// @brief The rasterized narrow band (all surfaces together), valid after build().
    const GridT* deviceGrid() const { return mGridHandle.template deviceGrid<BuildT>(); }
    /// @brief Handle owning that grid, valid after build().
    const Handle& gridHandle() const { return mGridHandle; }
    /// @brief Per-active-voxel unsigned distance in WORLD units, valid after build().
    const float* deviceUDF() const { return static_cast<const float*>(mUDF.deviceData()); }
    /// @brief Per-active-voxel nearest-triangle index (0xFFFFFFFF = none), valid after build().
    const uint32_t* deviceIndex() const { return static_cast<const uint32_t*>(mIndex.deviceData()); }
    /// @brief Per-active-voxel sign over the rasterized band (+1 outside / -1 inside), valid after
    ///        build(). Length activeVoxelCount+1, indexed by leaf.getValue(n); slot 0 = +1.
    const int8_t* deviceSign() const { return mSign; }
    /// @brief The world<->index transform the grid was built with.
    const nanovdb::Map& map() const { return mMap; }
    /// @brief The narrow-band width, in cell units, the grid was built with.
    float narrowBandWidth() const { return mBandWidth; }

    /// @brief Number of closed surfaces the band was partitioned into, valid after build().
    uint32_t surfaceCount() const { return uint32_t(mSurfaces.size()); }
    /// @brief Per-active-voxel closed-surface id in [0, surfaceCount()), on the rasterized band.
    const uint32_t* deviceSurfaceLabels() const { return mSurfaceLabels.first; }
    /// @brief Per surface: 1 iff its nesting depth is odd, so its own signs were negated.
    const std::vector<uint32_t>& nestingDepth() const { return mNestingDepth; }
    /// @brief The nesting rule the last build() resolved those depths with.
    NestingRule nestingRule() const { return mNestingRule; }

    /// @name Invert masks — the sign of everything the band does not cover, valid after build().
    ///       Consumed together with deviceGrid() and deviceSign() by sdf_detail::signedSignAt().
    /// @{
    const nanovdb::Mask<3>* deviceLeafInvertMask() const;
    const nanovdb::Mask<4>* deviceLowerInvertMask() const;
    const nanovdb::Mask<5>* deviceUpperInvertMask() const;
    const uint8_t*          deviceRootInterior() const;
    nanovdb::Coord          rootTileMin() const;
    nanovdb::Coord          rootTileDims() const;
    /// @}

    /// @brief The signer that produced surface @a i's own field, for tests and debugging.
    Signer& surfaceSigner(uint32_t i) const { return *mSurfaces[i].signer; }
    /// @brief Surface @a i's carved band, or the rasterized band when there is only one surface.
    const Handle& surfaceGridHandle(uint32_t i) const;
    /// @brief Surface @a i's barrier-pruned grid (the connected-components input).
    const Handle& derivedGridHandle(uint32_t i) const { return mSurfaces[i].derived; }
    /// @brief Surface @a i's per-active-voxel component labels, on its derived grid.
    const uint32_t* deviceComponentLabels(uint32_t i) const { return mSurfaces[i].ccLabels.first; } //todo: since this is i'th surface data we want to consider it as deviceSurfaceComponentsLabel or something like that
    /// @brief How many components surface @a i's derived grid was labeled into.
    uint64_t componentCount(uint32_t i) const { return mSurfaces[i].ccLabels.second; }//todo: surfaceComponentCount or something else
    /// @brief Surface @a i's nearest-triangle index sidecar, re-indexed onto its carved band.
    const uint32_t* surfaceIndex(uint32_t i) const; //todo: more specifically surfaceNearestTriangleIndex
    /// @brief Surface @a i's unsigned-distance sidecar (WORLD units), re-indexed onto its carved band.
    const float*    surfaceUdf(uint32_t i) const;

    /// @brief Wall-clock milliseconds spent in each of build()'s five phases, indexed by the step
    ///        numbers in the private section below (0 = rasterize ... 4 = fillOnOriginal). Valid after build(). Each phase
    ///        is bracketed by a stream sync, so the numbers add up to build()'s own wall time and
    ///        include host-side work, not just kernel time. Useful for separating the cost of
    ///        rasterization -- normally the bulk of the pipeline -- from everything downstream.
    const float* phaseMs() const { return mPhaseMs; }

private:

    /// @brief Everything the pipeline produces for ONE closed surface: that surface's band as a
    ///        stand-alone grid, plus the complete sign field it would have if it were the only object
    ///        in the scene. Nothing here knows that other surfaces exist, which is exactly what makes
    ///        a single global exterior seed valid again.
    struct SurfaceField { //todo: The name SurfaceField is somewhat unclear, let's come up with other name?
        Handle  subGrid;   // this surface's band, carved out of the rasterized one. EMPTY when the
                           // mesh has a single closed surface — that band is then used as-is.
        Buffer  subUdf;    // udf / nearest-triangle index re-indexed onto subGrid (carving renumbers
        Buffer  subIndex;  // the value slots). Both empty in that same single-surface case.
        Handle  derived;   // barrier-pruned copy of the surface grid = connected-components input todo: name to something as barrierPrunedGrid
        std::unique_ptr<ConnectedComponents<BuildT>> cc;
        std::pair<uint32_t*, uint64_t>               ccLabels{nullptr, 0};
        std::unique_ptr<Signer>                      signer;
    };

    void rasterize();                    // step 1
    void partition();                    // step 2, run connected component on un-pruned grid for seperating each surfaces.
    void signSurface(uint32_t surface);  // step 3, once per closed surface
    void composeByInclusion();           // step 4
    void finalizeMagnitudes();           // step 4b, report distance to the isosurface, not to the mesh
    void fillOnOriginal();               // step 5 TODO: let's rename it to more specific name?

    // Surface i's grid: its own carved band, or the rasterized band when uncarved.
    const GridT*    surfaceGrid(uint32_t i) const;

    const nanovdb::Vec3f* mPoints{nullptr};
    uint32_t              mPointCount{0};
    const nanovdb::Vec3i* mTriangles{nullptr};
    uint32_t              mTriangleCount{0};
    nanovdb::Map          mMap{};
    cudaStream_t          mStream{0};
    int                   mVerbose{0};
    float                 mBandWidth{3.f};
    BarrierSigning        mBarrierSigning{BarrierSigning::Interior};
    float                 mIsoValue{0.f};  // world units, >= 0; see setIsoValue()
    int                   mBallStencilRadius{1};  // see setBallStencilRadius()
    float                 mPhaseMs[5]{};   // per-phase wall time from the last build(), see phaseMs()
    // Sizes below use A = the rasterized band's active voxel count and N = the closed-surface count.
    // Every per-voxel sidecar is A+1 long and indexed by leaf.getValue(n), so slot 0 is the background.
    Handle mGridHandle;   // step 1: the rasterized narrow band, all surfaces together
    Buffer mUDF, mIndex;  // (A+1) x float / (A+1) x uint32: unsigned distance, nearest-triangle index

    std::unique_ptr<ConnectedComponents<BuildT>> mSurfaceCC;                   // step 2
    std::pair<uint32_t*, uint64_t>               mSurfaceLabels{nullptr, 0};   // { (A+1) x uint32 surface
                                                                               // id, N }; owned by mSurfaceCC
    std::vector<SurfaceField> mSurfaces;   // step 3: N entries, one per closed surface — the count
                                           // itself is surfaceCount(), not a separate member

    NestingRule          mNestingRule{NestingRule::EvenOdd};
    std::vector<uint32_t> mNestingDepth;   // step 4: N x uint32, how many other surfaces enclose each
    Buffer               mComposedSign;    // (A+1) x int8_t: gathered signs on the rasterized band. EMPTY
                                           // when a lone uncarved surface's own array is used instead —
                                           // which is also how fillOnOriginal knows its fill is already done.
    int8_t*              mSign{nullptr};   // (A+1) x int8_t, NOT owned: the signs every later stage reads.
                                           // Points into mComposedSign, or into surface 0's signer.

    std::unique_ptr<Signer> mOrigSigner;   // step 5; empty when surfaces[0]'s fill is adopted
    Signer*                 mFinalSigner{nullptr};  // NOT owned: mOrigSigner, or surface 0's signer

}; // tools::cuda::MeshToSDF<BuildT>

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace sdf_detail {

/// @brief Signs ONE closed surface: the stages that turn a narrow-band ValueOnIndex grid holding a
///        single closed surface, plus its UDF sidecar, into a complete sign field for that surface —
///        band signs plus the per-level invert masks that extend them off the band. Every rule here
///        assumes exactly one closed surface (see the file notes); MeshToSDF partitions and drives it.
/// @tparam BuildT Build type of the index grid (e.g. nanovdb::ValueOnIndex).
template <typename BuildT>
class SurfaceSigner //todo: rename? or do we really need this class? why this class should keep invert Mask?
{
    using GridT = NanoGrid<BuildT>;

public:

    /// @brief Constructor
    /// @param stream optional CUDA stream (defaults to CUDA stream 0)
    SurfaceSigner(cudaStream_t stream = 0) : mStream(stream), mTimer(stream) {}

    /// @brief Toggle on and off verbose mode
    /// @param level Verbose level: 0=quiet, 1=timing
    void setVerbose(int level = 1) { mVerbose = level; }

    /// @brief Prune the surface/barrier shell of a narrow-band UDF index grid, producing a clean
    ///        ValueOnIndex grid of the non-barrier voxels (the input to connected components).
    ///        A voxel is dropped iff (udf - isoValue)^2 < 0.75·voxelSize^2, i.e. it lies within
    ///        √3/2 voxels of the surface { udf == isoValue } being signed.
    /// @param d_srcGrid device narrow-band ValueOnIndex grid
    /// @param d_udf     device UDF sidecar (WORLD units), indexed by leaf.getValue(n); slot 0 = background
    /// @param voxelSize world-space voxel size (to convert the √3/2-voxel barrier into world units)
    /// @param isoValue  WORLD-unit level set of @a d_udf to sign; 0 signs the mesh itself
    /// @return a handle to the derived (barrier-pruned) ValueOnIndex grid
    template <typename BufferT = nanovdb::cuda::DeviceBuffer>
    GridHandle<BufferT> computeDerivedTopology(const GridT* d_srcGrid, const float* d_udf,
                                               float voxelSize, float isoValue = 0.f,
                                               const BufferT& buffer = BufferT());

    /// @brief Sign the non-barrier voxels of a CC-labeled grid: the component containing the grid's
    ///        minimum-x active voxel is the exterior (+); every other component is interior (-).
    ///        Convention: +outside / -inside.
    ///
    ///        Valid only on a grid carrying ONE closed surface: its leftmost voxel is necessarily
    ///        outside it. Several objects would need a seed each, so the caller partitions the band
    ///        into closed surfaces first and runs this per surface.
    /// @param d_grid        the CC-labeled (derived) device grid
    /// @param d_voxelLabel  per-active-voxel component-label sidecar for @a d_grid (from
    ///                      ConnectedComponents::getVoxelLabelsAndCount()), indexed by leaf.getValue(n).
    void signNonBarrier(const GridT* d_grid, const uint32_t* d_voxelLabel); // todo: signNonBarrierVoxels

    /// @brief Carry the derived-grid signs (from signNonBarrier) back onto the original grid. The
    ///        derived grid is the barrier-pruned subset of the original, so the injection covers all
    ///        non-barrier voxels; barrier voxels (present only in the original) keep the sentinel 0
    ///        ("unsigned barrier") for step 5 to fill. Requires signNonBarrier() first.
    /// @param d_origGrid    the original (pre-prune) grid — injection target.
    /// @param d_derivedGrid the barrier-pruned grid that was signed.
    void injectSignsToOriginal(const GridT* d_origGrid, const GridT* d_derivedGrid); //TODO: injectSignsToOriginalGrid

    /// @brief Sign every barrier voxel (sign == 0) of the original grid, completing the sign field so
    ///        no sentinel-0 voxel remains. Faithful mirror of OpenVDB MeshToVolume.h
    ///        ComputeIntersectingVoxelSign: a barrier voxel is exterior (+1) iff some exterior (+1)
    ///        neighbor's nearest triangle places it on the same side of the surface, else interior (-1).
    ///        Requires injectSignsToOriginal() first (it reads those signs as the anchor snapshot); the
    ///        completed signs land in a new buffer exposed via deviceSignedVoxelSign().
    /// @param d_grid      the original (post-injection) grid being signed.
    /// @param d_index     nearest-triangle-index sidecar of the original grid (uint32; 0xFFFFFFFF = none).
    /// @param d_points    device mesh vertices (WORLD space).
    /// @param d_triangles device triangle vertex-index list.
    /// @param map         the world<->index transform used to build the grid.
    /// @param isoValue  WORLD-unit level set being signed; 0 signs the mesh itself
    /// @param voxelSize world-space voxel size (the test runs in index space)
    void signBarrier(const GridT* d_grid, const uint32_t* d_index,
                     const nanovdb::Vec3f* d_points, const nanovdb::Vec3i* d_triangles,
                     const nanovdb::Map& map, float isoValue = 0.f,
                     float voxelSize = 1.f); //TODO: SignBarrierVoxels

    /// @brief Sign every barrier voxel interior, completing the sign field with no oracle at all.
    ///        The cheapest of the three barrier policies and the only one whose exterior set provably
    ///        holds no surface, at the cost of a surface that stands up to √3/2 voxels out.
    ///        Requires injectSignsToOriginal() first; the result lands in deviceSignedVoxelSign().
    /// @param d_grid the original (post-injection) grid being signed.
    void signBarrierAsInterior(const GridT* d_grid);

    /// @brief EXPERIMENTAL alternative to signBarrier: decide the barrier voxels by ball-intersection
    ///        certification instead of the closest-point heuristic. Iterates to a fixed point from
    ///        both exterior and interior seeds, leaving anything it cannot prove at 0 so the two
    ///        methods stay comparable voxel by voxel. Does not touch deviceSignedVoxelSign().
    /// @param d_grid    the grid whose barrier voxels are to be decided
    /// @param d_udf     unsigned distance sidecar for @a d_grid, WORLD units
    /// @param voxelSize world-space voxel size
    /// @param maxRounds cap on Jacobi rounds
    /// @param radius    stencil half-width: 1 = the 26 neighbours, 2 = 5x5x5. Pairs further apart
    ///                  than 2*bandWidth voxels can never overlap, so nothing is gained past that.
    /// @param isoValue  WORLD-unit level set being signed; 0 signs the mesh itself
    void signBarrierByBalls(const GridT* d_grid, const float* d_udf, float voxelSize,
                            int maxRounds = 32, int radius = 1, float isoValue = 0.f);

    /// @brief Result of signBarrierByBalls(): +1 ext / -1 int / 0 = not proven either way.
    int8_t* deviceBallVoxelSign() { return static_cast<int8_t*>(mBallVoxelSign.deviceData()); }
    /// @brief Voxels signBarrierByBalls() could prove nothing about: no neighbour's ball reached
    ///        them. They fall back to interior, the safe direction.
    uint32_t ballUndecided() const { return mBallUndecided; }

    /// @brief DISTINCT voxels that signBarrierByBalls() proved both ways, counted once each however
    ///        many rounds they persist for.
    ///
    /// The lemma forbids this: overlapping balls are surface-free, so a path exists between the two
    /// neighbours that never meets the surface, and a closed surface would have to separate them.
    /// A non-zero count therefore means the surface does not separate them there -- a hole, or a
    /// sheet thinner than the grid resolves. Tangency has to be excluded strictly for this to mean
    /// anything; see the epsilon in BallCertifyFunctor.
    uint32_t ballContradictions() const { return mBallContradictions; }
    uint32_t ballRounds() const { return mBallRounds; }

    /// @brief Build the per-leaf invert masks that sign the INACTIVE voxels of materialized leaves:
    ///        bit ON => interior (-background), bit OFF => exterior (+background, the default).
    ///        Floods interior signs from the active band through each leaf's inactive voxels, with
    ///        active voxels as walls. Needs completed signs, from signBarrier() or @a d_sign.
    /// @param d_grid the original (fully signed) device grid.
    /// @param d_sign optional per-active-voxel sign array for @a d_grid, overriding the one this
    ///               object computed. Lets the fill run on a grid it did not sign — a per-component
    ///               sub-grid, or the original grid after the signs were recomposed.
    void fillLeafInvertMask(const GridT* d_grid, const int8_t* d_sign = nullptr);

    /// @brief Fill the coarse invert masks that sign the CHILDLESS child slots of lower and upper
    ///        internal nodes (bit ON => that tile is interior). Bottom-up: leaf faces seed the lower
    ///        and upper tiles they abut, lower floods, lower faces seed upper, upper floods.
    ///        Requires fillLeafInvertMask() first.
    /// @param d_grid the original (fully signed) device grid.
    /// @param d_sign optional per-active-voxel sign array for @a d_grid, overriding the one this
    ///               object computed. Lets the fill run on a grid it did not sign — a per-component
    ///               sub-grid, or the original grid after the signs were recomposed.
    void fillCoarseInvertMasks(const GridT* d_grid, const int8_t* d_sign = nullptr);

    /// @brief Build the root-interior SIDECAR that signs the deep interior beyond any upper node: a
    ///        PxQxR cell array over the grid's root-tile range, one uint8 per 4096^3 root region.
    ///        Existing root entries are walls, faces from all three levels seed interior evidence
    ///        into abutting absent cells, and a flood fills what is enclosed. The grid is NOT
    ///        mutated -- queries go through signedSignAt(). Requires fillCoarseInvertMasks() first.
    /// @param d_grid the original (fully signed) device grid.
    /// @param d_sign optional per-active-voxel sign array for @a d_grid, overriding the one this
    ///               object computed. Lets the fill run on a grid it did not sign — a per-component
    ///               sub-grid, or the original grid after the signs were recomposed.
    void fillRootInteriorMask(const GridT* d_grid, const int8_t* d_sign = nullptr);

    /// @brief Device pointer to the per-active-voxel sign array (+1 exterior / -1 interior),
    ///        valid after signNonBarrier(). Length activeVoxelCount+1, indexed by leaf.getValue(n);
    ///        slot 0 is the background (+1).
    int8_t* deviceVoxelSign() { return static_cast<int8_t*>(mVoxelSign.deviceData()); }

    /// @brief Representative (slot) of the exterior component, valid after signNonBarrier().
    uint64_t exteriorRepresentative() const { return mExteriorRep; }

    /// @brief Device pointer to the per-active-voxel sign array on the ORIGINAL grid, valid after
    ///        injectSignsToOriginal(). Length origActiveVoxelCount+1, indexed by leaf.getValue(n);
    ///        slot 0 = background (+1); non-barrier voxels carry +1/-1; barrier voxels carry the
    ///        sentinel 0 (still unsigned, awaiting step 5).
    int8_t* deviceOriginalVoxelSign() { return static_cast<int8_t*>(mOriginalVoxelSign.deviceData()); }

    /// @brief Device pointer to the fully-signed per-active-voxel array on the ORIGINAL grid, valid
    ///        after signBarrier(): every active voxel is +1 (exterior) or -1 (interior), no sentinel-0
    ///        remains. Length origActiveVoxelCount+1, indexed by leaf.getValue(n); slot 0 = +1.
    int8_t* deviceSignedVoxelSign() { return static_cast<int8_t*>(mSignedVoxelSign.deviceData()); }

    /// @brief Device pointer to the per-leaf invert masks (nodeCount[0] × Mask<3>), valid after
    ///        fillLeafInvertMask(). For each materialized leaf, bit n ON means the INACTIVE voxel n
    ///        is interior (-background); OFF means exterior (+background). Bits of active voxels are
    ///        always OFF (their sign lives in the sign sidecar).
    nanovdb::Mask<3>* deviceLeafInvertMask() { return static_cast<nanovdb::Mask<3>*>(mLeafInvertMask.deviceData()); }

    /// @brief Device pointer to the lower-level invert masks (nodeCount[1] × Mask<4>), valid after
    ///        fillCoarseInvertMasks(). Bit n ON means the CHILDLESS lower slot n (an 8^3-voxel tile)
    ///        is interior; OFF means exterior. Refined slots (childMask ON) carry no invert bit.
    nanovdb::Mask<4>* deviceLowerInvertMask() { return static_cast<nanovdb::Mask<4>*>(mLowerInvertMask.deviceData()); }

    /// @brief Device pointer to the upper-level invert masks (nodeCount[2] × Mask<5>), valid after
    ///        fillCoarseInvertMasks(). Bit n ON means the CHILDLESS upper slot n (a 128^3-voxel tile)
    ///        is interior; OFF means exterior. Refined slots carry no invert bit.
    nanovdb::Mask<5>* deviceUpperInvertMask() { return static_cast<nanovdb::Mask<5>*>(mUpperInvertMask.deviceData()); }

    /// @brief Device pointer to the root-interior sidecar (rootTileDims() cells, one uint8 per 4096^3
    ///        root region; 1 = deep interior), valid after fillRootInteriorMask(). Cell (i,j,k) covers
    ///        the root region at (rootTileMin()+(i,j,k))*4096. Cells outside the array are exterior.
    uint8_t* deviceRootInterior() { return static_cast<uint8_t*>(mRootInterior.deviceData()); }

    /// @brief Origin of the root-cell array, in 4096-tile units (valid after fillRootInteriorMask()).
    nanovdb::Coord rootTileMin() const { return mRootTileMin; }

    /// @brief Dimensions P×Q×R of the root-cell array (valid after fillRootInteriorMask()).
    nanovdb::Coord rootTileDims() const { return mRootDims; }

private:

    // Shorthands for the two device-grid queries the stages keep asking for.
    static uint32_t leafCountOf(const GridT* g) {
        return util::cuda::DeviceGridTraits<BuildT>::getTreeData(g).mNodeCount[0];
    }
    static uint64_t activeCountOf(const GridT* g) {
        return util::cuda::DeviceGridTraits<BuildT>::getActiveVoxelCount(g);
    }

    cudaStream_t                 mStream{0};
    util::cuda::Timer            mTimer;
    int                          mVerbose{0};

    uint64_t                     mExteriorRep{0};  // the exterior component (see exteriorRepresentative())
    nanovdb::cuda::DeviceBuffer  mVoxelSign;       // (derived activeVoxelCount+1) × int8_t: +1 ext / -1 int
    nanovdb::cuda::DeviceBuffer  mOriginalVoxelSign; // (orig activeVoxelCount+1) × int8_t: +1/-1 non-barrier, 0 barrier
    nanovdb::cuda::DeviceBuffer  mSignedVoxelSign;   // (orig activeVoxelCount+1) × int8_t: +1/-1 everywhere (barriers signed)
    nanovdb::cuda::DeviceBuffer  mBallVoxelSign;     // (orig activeVoxelCount+1) × int8_t: experimental ball result
    uint32_t                     mBallUndecided{0}, mBallContradictions{0}, mBallRounds{0};
    nanovdb::cuda::DeviceBuffer  mLeafInvertMask;    // nodeCount[0] × Mask<3>: inactive-voxel interior bits
    nanovdb::cuda::DeviceBuffer  mLowerInvertMask;   // nodeCount[1] × Mask<4>: childless-lower-tile interior bits
    nanovdb::cuda::DeviceBuffer  mUpperInvertMask;   // nodeCount[2] × Mask<5>: childless-upper-tile interior bits
    nanovdb::cuda::DeviceBuffer  mRootInterior;      // P×Q×R × uint8: deep-interior bits of absent root regions
    nanovdb::Coord               mRootTileMin{0, 0, 0};  // root-cell array origin (4096-tile units)
    nanovdb::Coord               mRootDims{0, 0, 0};     // root-cell array dims P×Q×R

}; // SurfaceSigner<BuildT>

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

static constexpr int LEAF_SIZE = 512;  // 8^3 voxels per leaf

/// @brief CUDA functor: turn the distance-to-mesh sidecar into the magnitude actually reported,
///        once every sign is settled. One thread per sidecar slot via lambdaKernel.
///        See MeshToSDF::setIsoValue() and MeshToSDF::finalizeMagnitudes().
///
///        Two steps. Folding udf about the isovalue makes the pair (sign, magnitude) describe one
///        surface: the sign came from udf - isoValue, so leaving the magnitude as the distance to the
///        mesh would report a field off by isoValue everywhere. The fold is safe here and not earlier
///        precisely because it destroys the very information -- which side of the mesh -- that the
///        signing stages needed.
///
///        The clamp then pushes interior magnitudes out to at least half a voxel diagonal. Subtracting
///        a constant from a coarsely sampled distance field leaves values like udf 0.99 -> -0.01,
///        which claims an interface that is not there: the neighbouring samples are nowhere near
///        consistent with one, and a contouring pass would put the surface back roughly where the
///        isovalue moved it from. Non-barrier interior voxels are already beyond this threshold by the
///        barrier test itself, so the clamp only ever bites on barrier voxels signed interior.
template <typename BuildT>
struct IsoMagnitudeFunctor
{
    static constexpr int MaxThreadsPerBlock         = LEAF_SIZE;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(const NanoGrid<BuildT>* d_grid, float* d_udf, const int8_t* d_sign,
                               float isoValue, float interiorFloor)
    {
        const int leafID = blockIdx.x, n = threadIdx.x;
        const auto& leaf = d_grid->tree().template getFirstNode<0>()[leafID];
        if (!leaf.isActive(uint32_t(n))) return;

        const uint64_t v = leaf.getValue(uint32_t(n));
        const float    m = fabsf(d_udf[v] - isoValue);
        d_udf[v] = m;
        if (d_sign[v] >= int8_t(0) || m >= interiorFloor) return;   // nothing to floor

        // Only a voxel with an exterior FACE neighbour can carry the interface: a marching-cubes
        // vertex lands on an axis edge, between two face-adjacent samples of opposite sign. Flooring
        // such a voxel would drag that crossing along, so leave it alone and floor the rest.
        const nanovdb::Coord local = nanovdb::NanoLeaf<BuildT>::OffsetToLocalCoord(uint32_t(n));
        const int lx = local[0], ly = local[1], lz = local[2];
        const int off[6][3] = {{-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1}};

        bool touchesExterior = false, needsAccessor = false;
        for (int k = 0; k < 6 && !touchesExterior; ++k) {
            const int nx = lx + off[k][0], ny = ly + off[k][1], nz = lz + off[k][2];
            if (nx < 0 || nx > 7 || ny < 0 || ny > 7 || nz < 0 || nz > 7) { needsAccessor = true; continue; }
            const uint32_t nOff = (uint32_t(nx) << 6) | (uint32_t(ny) << 3) | uint32_t(nz);
            if (leaf.isActive(nOff) && d_sign[leaf.getValue(nOff)] > int8_t(0)) touchesExterior = true;
        }
        if (!touchesExterior && needsAccessor) {
            const nanovdb::Coord origin = leaf.origin();
            auto acc = d_grid->getAccessor();
            for (int k = 0; k < 6 && !touchesExterior; ++k) {
                const int nx = lx + off[k][0], ny = ly + off[k][1], nz = lz + off[k][2];
                if (nx >= 0 && nx <= 7 && ny >= 0 && ny <= 7 && nz >= 0 && nz <= 7) continue;
                const nanovdb::Coord nijk(origin[0] + nx, origin[1] + ny, origin[2] + nz);
                if (acc.isActive(nijk) && d_sign[acc.getValue(nijk)] > int8_t(0)) touchesExterior = true;
            }
        }
        if (!touchesExterior) d_udf[v] = interiorFloor;
    }
};// sdf_detail::IsoMagnitudeFunctor

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Step 2 - prune the surface/barrier shell.
// Voxels within sqrt(3)/2 of the surface being signed straddle it, so which side they are on is not
// yet decided. Dropping them splits the band into shells that connected components can label, and
// each shell then lies wholly on one side.

/// @brief CUDA functor: build a per-leaf retain bitmask that drops the surface/barrier shell. A
///        voxel is PRUNED iff it is within √3/2 voxels of the signed surface (half a voxel
///        space-diagonal — the same barrier OpenVDB's MeshToVolume uses); every other active voxel is
///        RETAINED. Launched via operatorKernel, one block per leaf, 512 threads (one per voxel in
///        the 8^3 leaf).
///
///        The surface being signed is { udf == isoValue }, so the test is on the DISPLACEMENT
///        udf - isoValue, not on udf. Both are in the sidecar's WORLD units and the comparison is
///        made squared: (udf - isoValue)^2 < (√3/2 · voxelSize)^2 = 0.75 · voxelSize^2, passed in
///        precomputed. With isoValue == 0 this is the plain mesh barrier.
///
///        What retaining leaves behind, for isoValue > 0, is three kinds of region: the true exterior
///        (udf > isoValue), the mesh's deep interior (also udf > isoValue, but walled off from the
///        exterior by the shell), and the tube hugging the mesh (udf < isoValue) which is connected
///        THROUGH the mesh surface because udf == 0 is not a barrier here. Only the first holds the
///        minimum index-space voxel, so the other two are signed interior — which is what folds the
///        isosurface's second, buried zero crossing away.
template <typename BuildT>
struct UDFBarrierPruneMaskFunctor
{
    static constexpr int MaxThreadsPerBlock         = LEAF_SIZE;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(
        const nanovdb::NanoGrid<BuildT>* d_grid,
        const float*                     d_udf,           // UDF sidecar, WORLD units
        float                            isoValue,        // signed surface = { udf == isoValue }, world
        float                            barrierSqWorld,  // (√3/2 · voxelSize)^2, world^2 units
        nanovdb::Mask<3>*                d_dstLeafMasks)
    {
        const int leafID   = blockIdx.x;
        const int threadID = threadIdx.x;

        const auto& leaf       = d_grid->tree().template getFirstNode<0>()[leafID];
        auto&       resultMask = d_dstLeafMasks[leafID];

        // Clear the leaf's mask words in parallel, then fill the retain bits.
        if (threadID < nanovdb::Mask<3>::WORD_COUNT)
            resultMask.words()[threadID] = 0UL;
        __syncthreads();

        if (auto n = leaf.data()->getValue(threadID)) {  // n != 0 => active voxel
            const float d = d_udf[n] - isoValue;
            if (d * d >= barrierSqWorld)                 // retain non-barrier voxels
                resultMask.setOnAtomic(threadID);
        }
    }
};

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Step 4 - sign the non-barrier voxels.
// Within one closed surface, the component holding that surface's minimum-x voxel is its exterior and
// every other component of the same surface is interior. Seeding per surface rather than once globally
// is what makes the rule hold for several disconnected objects.

/// @brief Reduce over the active voxels to the minimum x, carrying the voxel's component representative
///        into *d_minKey = (unsigned(x) << 32) | uint32(rep). The min-x voxels are all exterior (and
///        share one representative), so the low 32 bits resolve to the exterior rep.
template <typename BuildT>
struct FindExteriorRepFunctor
{
    static constexpr int MaxThreadsPerBlock         = LEAF_SIZE;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(const NanoGrid<BuildT>* d_grid, const uint32_t* d_voxelLabel,
                               unsigned long long* d_minKey)
    {
        const int leafID = blockIdx.x, n = threadIdx.x;
        const auto& leaf = d_grid->tree().template getFirstNode<0>()[leafID];
        if (!leaf.isActive(uint32_t(n))) return;
        const uint64_t v = leaf.getValue(uint32_t(n));
        const nanovdb::Coord ijk = leaf.origin() + nanovdb::NanoLeaf<BuildT>::OffsetToLocalCoord(uint32_t(n));
        const uint32_t rep = d_voxelLabel[v];                              // component representative slot
        const uint32_t ux  = uint32_t(int64_t(ijk[0]) + (int64_t(1) << 31));  // x, shifted to unsigned-comparable
        atomicMin(d_minKey, (static_cast<unsigned long long>(ux) << 32) | rep);
    }
};

/// @brief Write per-active-voxel signs (+1 exterior / -1 interior), indexed by leaf.getValue(n),
///        into d_sign (length activeVoxelCount+1; slot 0 = background, pre-filled +1). A voxel is
///        exterior iff its component is the exterior representative.
template <typename BuildT>
struct SignNonBarrierFunctor
{
    static constexpr int MaxThreadsPerBlock         = LEAF_SIZE;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(const NanoGrid<BuildT>* d_grid, const uint32_t* d_voxelLabel,
                               uint32_t exteriorRep, int8_t* d_sign)
    {
        const int leafID = blockIdx.x, n = threadIdx.x;
        const auto& leaf = d_grid->tree().template getFirstNode<0>()[leafID];
        if (!leaf.isActive(uint32_t(n))) return;
        const uint64_t v = leaf.getValue(uint32_t(n));
        d_sign[v] = (d_voxelLabel[v] == exteriorRep) ? int8_t(1) : int8_t(-1);
    }
};

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Step 5 - sign the barrier voxels.
// A faithful mirror of OpenVDB MeshToVolume's ComputeIntersectingVoxelSign: a barrier voxel is exterior
// iff some already-signed exterior neighbour's nearest triangle places both of them on the same side of
// the surface. Anchors come from an immutable snapshot, so the result is independent of execution order.

static constexpr uint32_t INVALID_TRIANGLE = 0xFFFFFFFFu;  // nearest-triangle-index sentinel

/// @brief True iff neighbour @a n proves barrier voxel @a q exterior: @a n must itself be exterior
///        (sign == +1), and its nearest triangle must place both on the same side, i.e.
///        normalize(n - cp) . normalize(q - cp) > 0 for cp the closest point on that triangle to n.
///        Interior, barrier and no-hit neighbours prove nothing. Mirrors one neighbour test of
///        OpenVDB MeshToVolume.h ComputeIntersectingVoxelSign; __hostdev__ so the CPU oracle can
///        reuse it.
///
///        Double precision is deliberate: this is a sign-of-dot decision, and at large index
///        coordinates float cancellation can flip it differently on host and device.
__hostdev__ inline bool
barrierExteriorProof(uint64_t nv, const nanovdb::Coord& nijk, const nanovdb::Vec3d& q_xyz,
                     const int8_t* d_sign, const uint32_t* d_index,
                     const nanovdb::Vec3f* d_points, const nanovdb::Vec3i* d_triangles,
                     const nanovdb::Map& map, double isoValueIndex)
{
    if (d_sign[nv] != int8_t(1)) return false;       // only exterior (+1) neighbors anchor
    const uint32_t tid = d_index[nv];
    if (tid == INVALID_TRIANGLE) return false;        // neighbor had no nearest triangle (no-hit)

    const nanovdb::Vec3i& T  = d_triangles[tid];
    const nanovdb::Vec3f& p0 = d_points[T[0]];
    const nanovdb::Vec3f& p1 = d_points[T[1]];
    const nanovdb::Vec3f& p2 = d_points[T[2]];
    const nanovdb::Vec3d  v0 = map.applyInverseMap(nanovdb::Vec3d(p0[0], p0[1], p0[2]));  // world -> index
    const nanovdb::Vec3d  v1 = map.applyInverseMap(nanovdb::Vec3d(p1[0], p1[1], p1[2]));
    const nanovdb::Vec3d  v2 = map.applyInverseMap(nanovdb::Vec3d(p2[0], p2[1], p2[2]));
    const nanovdb::Vec3d  n_xyz(static_cast<double>(nijk[0]), static_cast<double>(nijk[1]), static_cast<double>(nijk[2]));

    double t0, t1;
    const nanovdb::Vec3d cp = nanovdb::math::closestPointOnTriangleToPoint(v0, v1, v2, n_xyz, t0, t1);
    nanovdb::Vec3d dn = n_xyz - cp; dn.normalize();   // surface -> neighbor (its confident side)

    // Slide the reference point out onto the surface actually being signed. cp is the nearest point
    // of the MESH to n, so every point of the segment [cp, n] has cp as ITS nearest point too, and
    // the one isoValue along it therefore sits exactly on { udf == isoValue }. Offset surfaces are
    // parallel, so dn is its normal there as well and only the base point moves.
    const nanovdb::Vec3d base = cp + dn * isoValueIndex;

    nanovdb::Vec3d dq = q_xyz - base; dq.normalize();  // surface -> q
    return dn.dot(dq) > 0.0;                           // same side => q is exterior
}


/// @brief One Jacobi round of ball-intersection certification (EXPERIMENTAL, an alternative to
///        SignBarrierFunctor).
///
/// The ball of radius udf(V) about a voxel centre cannot reach the surface, so two overlapping such
/// balls form a connected surface-free set and their centres share a side. Overlap is
/// |V1 - V0| < d0 + d1, so an unknown voxel may take the sign of any certain neighbour meeting it.
/// This is a proof rather than a heuristic, it reads distances only, and it propagates from interior
/// and exterior seeds alike.
///
/// Tangency is excluded by a tolerance: touching balls prove nothing, and on axis-aligned input
/// exact tangency is common enough that rounding would decide it differently on the two sides.
///
/// Reads @a d_labelIn and writes @a d_labelOut, so the round is order-independent. A voxel certified
/// both ways contradicts the lemma; it is left undecided and counted, and can only happen where the
/// surface is not closed.
template <typename BuildT>
struct BallCertifyFunctor
{
    static constexpr int MaxThreadsPerBlock         = LEAF_SIZE;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(
        const NanoGrid<BuildT>* d_grid,
        const int8_t*           d_labelIn,   // +1 ext / -1 int / 0 undecided
        int8_t*                 d_labelOut,
        const float*            d_udf,       // unsigned distance sidecar, WORLD units
        float                   voxelSize,
        uint32_t*               d_changed,       // incremented once per newly decided voxel
        uint32_t*               d_contradictions,// incremented once per voxel, on its FIRST contradiction
        uint32_t*               d_everContradicted,  // one bit per slot, persistent across rounds
        int                     radius,          // stencil half-width in voxels; 1 = the 26 neighbours
        float                   isoValue)        // surface signed = { udf == isoValue }, world units
    {
        const int leafID = blockIdx.x, n = threadIdx.x;
        const auto& leaf = d_grid->tree().template getFirstNode<0>()[leafID];
        if (!leaf.isActive(uint32_t(n))) return;

        const uint64_t qv = leaf.getValue(uint32_t(n));
        const int8_t   ql = d_labelIn[qv];
        if (ql != int8_t(0)) { d_labelOut[qv] = ql; return; }   // already certain: carry through

        const nanovdb::Coord local  = nanovdb::NanoLeaf<BuildT>::OffsetToLocalCoord(uint32_t(n));
        const int            lx = local[0], ly = local[1], lz = local[2];
        const nanovdb::Coord origin = leaf.origin();
        // Radii are distances to the surface being signed, not to the mesh. |udf - isoValue| is a
        // 1-Lipschitz under-estimate of that near the medial axis, which is the safe direction: it
        // shrinks the balls, so it can only certify fewer voxels, never wrongly.
        const float          dq  = fabsf(d_udf[qv] - isoValue);
        const float          eps = 1e-5f * voxelSize;

        bool ext = false, inr = false;

        // The ball test needs only the neighbour's distance and its label, so both passes reduce to
        // two loads and a comparison against the offset length -- one of three constants.
        auto consider = [&] __device__ (uint64_t nv, int dx, int dy, int dz) {
            const int8_t ln = d_labelIn[nv];
            if (ln == int8_t(0)) return;                                   // neighbour not certain yet
            const float len = sqrtf(float(dx*dx + dy*dy + dz*dz)) * voxelSize;
            if (fabsf(d_udf[nv] - isoValue) + dq <= len + eps) return;      // balls do not overlap
            if (ln > 0) ext = true; else inr = true;
        };

        // Pass 1: neighbours inside this leaf, straight off the leaf buffer.
        for (int dx = -radius; dx <= radius; ++dx) {
            const int nx = lx + dx; if (nx < 0 || nx > 7) continue;
            for (int dy = -radius; dy <= radius; ++dy) {
                const int ny = ly + dy; if (ny < 0 || ny > 7) continue;
                for (int dz = -radius; dz <= radius; ++dz) {
                    const int nz = lz + dz; if (nz < 0 || nz > 7) continue;
                    if (!dx && !dy && !dz) continue;
                    const uint32_t nOff = (uint32_t(nx) << 6) | (uint32_t(ny) << 3) | uint32_t(nz);
                    if (leaf.isActive(nOff)) consider(leaf.getValue(nOff), dx, dy, dz);
                }
            }
        }

        // Pass 2: the rest of the stencil, which crosses the leaf boundary. One reused accessor.
        if (lx < radius || lx > 7 - radius || ly < radius || ly > 7 - radius ||
            lz < radius || lz > 7 - radius) {
            auto acc = d_grid->getAccessor();
            for (int dx = -radius; dx <= radius; ++dx) {
                const int nx = lx + dx;
                for (int dy = -radius; dy <= radius; ++dy) {
                    const int ny = ly + dy;
                    for (int dz = -radius; dz <= radius; ++dz) {
                        const int nz = lz + dz;
                        if (!dx && !dy && !dz) continue;
                        if (nx >= 0 && nx <= 7 && ny >= 0 && ny <= 7 && nz >= 0 && nz <= 7) continue;
                        const nanovdb::Coord nijk(origin[0] + nx, origin[1] + ny, origin[2] + nz);
                        if (acc.isActive(nijk)) consider(acc.getValue(nijk), dx, dy, dz);
                    }
                }
            }
        }

        if (ext && inr) {
            // The lemma forbids this, so it is evidence the surface does not separate the two
            // neighbours -- a hole, or a sheet thinner than the grid resolves. Count the VOXEL, not
            // the event: a voxel that stays contradicted is re-detected every round, and a voxel
            // contradicted only in a middle round would otherwise vanish from the tally entirely.
            const uint32_t word = uint32_t(qv >> 5), bit = 1u << (uint32_t(qv) & 31u);
            if ((atomicOr(d_everContradicted + word, bit) & bit) == 0u)
                atomicAdd(d_contradictions, 1u);
            d_labelOut[qv] = int8_t(0);
            return;
        }
        if (!ext && !inr) { d_labelOut[qv] = int8_t(0); return; }
        d_labelOut[qv] = ext ? int8_t(1) : int8_t(-1);
        atomicAdd(d_changed, 1u);
    }
};

/// @brief Turn ball-certification labels into a complete sign field. Anything the certification
///        could not prove is called interior, matching the heuristic's own fallback; the count is
///        reported so a caller can see how much of the field rests on that default rather than on a
///        proof.
struct BallFinalizeFunctor
{
    __device__ void operator()(size_t v, const int8_t* d_ball, int8_t* d_signOut,
                               uint32_t* d_undecided) const
    {
        if (v == 0) { d_signOut[0] = int8_t(1); return; }   // slot 0 = background = exterior
        const int8_t l = d_ball[v];
        if (l != int8_t(0)) { d_signOut[v] = l; return; }
        d_signOut[v] = int8_t(-1);
        atomicAdd(d_undecided, 1u);
    }
};

/// @brief Sign every barrier voxel (sign == 0), mirroring OpenVDB's ComputeIntersectingVoxelSign.
///        One block per leaf, one thread per voxel. Non-barrier voxels pass through unchanged;
///        a barrier voxel searches its 26 neighbours for an exterior anchor that proves it exterior
///        (barrierExteriorProof) and defaults to interior if none does. Pass 1 stays inside the leaf
///        buffer, pass 2 crosses the leaf boundary through one reused ReadAccessor.
///
///        Anchors come from @a d_signIn and results go to a separate @a d_signOut, so a just-signed
///        barrier voxel never anchors another and the result is order-independent.
///
///        That last property also bounds which voxels this can move. An anchor has to carry +1 in
///        @a d_signIn, and the only +1 entries there are the certified-outside voxels the component
///        stage decided -- every barrier voxel is 0. So a barrier voxel more than one neighbourhood
///        away from certified outside has no anchor to find and falls through to interior, whatever
///        the shell's thickness. The rule "re-sign the layer adjacent to certified outside and call
///        everything beyond it interior" is therefore not an extra pass to add; it is what a single
///        anchored round already computes.
/// @brief CUDA functor: carry the non-barrier signs through and call every barrier voxel interior.
///        One thread per sidecar slot via lambdaKernel. Slot 0 (background) is set by the caller.
///
///        The barrier shell is by construction the only place the surface can be, so declaring all
///        of it interior is the one assignment that cannot leave surface inside the exterior set.
///        It needs no neighbour search and no oracle.
struct BarrierToInteriorFunctor
{
    __device__ void operator()(const uint64_t slot, const int8_t* d_signIn, int8_t* d_signOut) const
    {
        const int8_t s = d_signIn[slot];
        d_signOut[slot] = (s != int8_t(0)) ? s : int8_t(-1);
    }
};// sdf_detail::BarrierToInteriorFunctor

template <typename BuildT>
struct SignBarrierFunctor
{
    static constexpr int MaxThreadsPerBlock         = LEAF_SIZE;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(
        const NanoGrid<BuildT>* d_grid,
        const int8_t*           d_signIn,     // immutable anchors: +1 ext / -1 int / 0 barrier
        int8_t*                 d_signOut,    // result: every active voxel ±1 (slot 0 set by host)
        const uint32_t*         d_index,      // nearest-triangle index sidecar (original grid)
        const nanovdb::Vec3f*   d_points,     // mesh vertices, WORLD space
        const nanovdb::Vec3i*   d_triangles,  // triangle vertex indices
        nanovdb::Map            map,          // world<->index transform (by value)
        double                  isoValueIndex)// surface signed = { udf == isoValue }, INDEX units
    {
        const int leafID = blockIdx.x, n = threadIdx.x;
        const auto& leaf = d_grid->tree().template getFirstNode<0>()[leafID];
        if (!leaf.isActive(uint32_t(n))) return;

        const uint64_t qv = leaf.getValue(uint32_t(n));
        const int8_t   qs = d_signIn[qv];
        if (qs != int8_t(0)) { d_signOut[qv] = qs; return; }  // non-barrier: carry sign through

        const nanovdb::Coord local  = nanovdb::NanoLeaf<BuildT>::OffsetToLocalCoord(uint32_t(n));
        const int            lx = local[0], ly = local[1], lz = local[2];
        const nanovdb::Coord origin = leaf.origin();
        const nanovdb::Vec3d q_xyz(double(origin[0] + lx), double(origin[1] + ly), double(origin[2] + lz));

        bool exterior = false;

        // Pass 1: in-leaf 3×3×3 neighbors (direct leaf buffer), early-out on first proof.
        for (int dx = -1; dx <= 1 && !exterior; ++dx) {
            const int nx = lx + dx; if (nx < 0 || nx > 7) continue;
            for (int dy = -1; dy <= 1 && !exterior; ++dy) {
                const int ny = ly + dy; if (ny < 0 || ny > 7) continue;
                for (int dz = -1; dz <= 1; ++dz) {
                    const int nz = lz + dz; if (nz < 0 || nz > 7) continue;
                    if (dx == 0 && dy == 0 && dz == 0) continue;  // skip q itself
                    const uint32_t nOff = (uint32_t(nx) << 6) | (uint32_t(ny) << 3) | uint32_t(nz);
                    if (!leaf.isActive(nOff)) continue;
                    const nanovdb::Coord nijk(origin[0] + nx, origin[1] + ny, origin[2] + nz);
                    if (barrierExteriorProof(leaf.getValue(nOff), nijk, q_xyz, d_signIn, d_index,
                                             d_points, d_triangles, map, isoValueIndex)) {
                        exterior = true; break;
                    }
                }
            }
        }

        // Pass 2: 26-neighborhood crossing the leaf boundary (only if unresolved and q is on a face).
        if (!exterior && (lx == 0 || lx == 7 || ly == 0 || ly == 7 || lz == 0 || lz == 7)) {
            auto acc = d_grid->getAccessor();
            for (int dx = -1; dx <= 1 && !exterior; ++dx)
                for (int dy = -1; dy <= 1 && !exterior; ++dy)
                    for (int dz = -1; dz <= 1; ++dz) {
                        if (dx == 0 && dy == 0 && dz == 0) continue;
                        const int nx = lx + dx, ny = ly + dy, nz = lz + dz;
                        if (nx >= 0 && nx <= 7 && ny >= 0 && ny <= 7 && nz >= 0 && nz <= 7)
                            continue;  // in-leaf neighbor already handled by pass 1
                        const nanovdb::Coord nijk(origin[0] + nx, origin[1] + ny, origin[2] + nz);
                        if (!acc.isActive(nijk)) continue;
                        if (barrierExteriorProof(acc.getValue(nijk), nijk, q_xyz, d_signIn, d_index,
                                                 d_points, d_triangles, map, isoValueIndex)) {
                            exterior = true; break;
                        }
                    }
        }

        d_signOut[qv] = exterior ? int8_t(1) : int8_t(-1);
    }
};

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Step 6A - inactive voxels inside materialized leaves.
// The band only covers the surface's neighbourhood; everywhere else the sign is carried implicitly by a
// per-level invert mask whose bits mark the interior regions. Each level is flooded outwards from the
// signed band, with the active voxels acting as walls.

/// @brief Leaf-level invert mask: one Mask<3> per leaf signing that leaf's INACTIVE voxels, bit ON
///        meaning interior. One block per leaf, 512 threads, shared-memory Jacobi flood seeded from
///        the active interior voxels and bounded by ALL active voxels, propagating through inactive
///        voxels via the 6 face neighbours. The active band walls the flood in, so it cannot leak
///        from interior to exterior. A leaf with no interior voxels stays all-0.
template <typename BuildT>
struct FillLeafInvertMaskFunctor
{
    static constexpr int MaxThreadsPerBlock         = LEAF_SIZE;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(const NanoGrid<BuildT>* d_grid,
                               const int8_t*           d_sign,        // full sign sidecar (post step 5)
                               nanovdb::Mask<3>*       d_invertMasks) // one Mask<3> per leaf, output
    {
        const int leafID = blockIdx.x, n = threadIdx.x;
        const auto& leaf = d_grid->tree().template getFirstNode<0>()[leafID];

        __shared__ uint8_t sAct[LEAF_SIZE];  // active voxel (wall)
        __shared__ uint8_t sInj[LEAF_SIZE];  // interior active voxel (flood source)
        __shared__ uint8_t sInv[LEAF_SIZE];  // result: inactive voxel marked interior
        __shared__ int     sChanged;

        const bool act = leaf.isActive(uint32_t(n));
        sAct[n] = act ? 1 : 0;
        sInj[n] = (act && d_sign[leaf.getValue(uint32_t(n))] == int8_t(-1)) ? 1 : 0;
        sInv[n] = 0;
        __syncthreads();

        // Voxel offset layout n = (x<<6)|(y<<3)|z; face neighbors are n±64 / n±8 / n±1.
        const int x = n >> 6, y = (n >> 3) & 7, z = n & 7;
        auto feeds = [&](int m) { return sInj[m] || (!sAct[m] && sInv[m]); };

        // Jacobi flood to convergence. Sources are ≤1 band-thickness away through smooth inactive
        // pockets, so convergence takes far fewer than 64 sweeps; 64 is a safety cap.
        for (int it = 0; it < 64; ++it) {
            if (n == 0) sChanged = 0;
            __syncthreads();
            bool turnOn = false;
            if (!act && !sInv[n]) {
                if ((x > 0 && feeds(n - 64)) || (x < 7 && feeds(n + 64)) ||
                    (y > 0 && feeds(n -  8)) || (y < 7 && feeds(n +  8)) ||
                    (z > 0 && feeds(n -  1)) || (z < 7 && feeds(n +  1)))
                    turnOn = true;
            }
            __syncthreads();                          // all reads of sInv precede this sweep's writes
            if (turnOn) { sInv[n] = 1; sChanged = 1; }
            __syncthreads();                          // writes (incl. sChanged) visible to the break test
            const bool done = (sChanged == 0);        // latch into a register...
            __syncthreads();                          // ...so thread 0's next-iter reset can't race the read
            if (done) break;
        }

        // Pack the 512 result bits into the leaf's Mask<3> (bit n lives in word n>>6, bit n&63).
        if (n < int(nanovdb::Mask<3>::WORD_COUNT)) {
            uint64_t w = 0;
            for (int b = 0; b < 64; ++b)
                if (sInv[(n << 6) | b]) w |= (uint64_t(1) << b);
            d_invertMasks[leafID].words()[n] = w;
        }
    }
};

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Step 6B - childless lower (8^3) and upper (128^3) tiles.
// Built bottom-up: leaf faces seed the tiles they abut, those tiles flood among themselves, then lower
// faces seed the upper level and it floods in turn.

/// @brief Descend root -> upper -> lower for the 8^3 leaf-region at coord @a c and report where the
///        finest CHILDLESS tile containing it lives -- the target a face seeds into.
/// @return 0 = skip: root-level value tile or refined all the way to a leaf (handled at the finer
///         level); 1 = childless LOWER slot; 2 = childless UPPER slot; 3 = ABSENT root region (no
///         root tile at all — the chunk-C root-cell target, cell = floorDiv(c,4096) per axis).
///         On 1/2, @a nodeIdx = linear node index at that level and @a slot = child-slot offset.
template <typename BuildT>
__hostdev__ inline int
probeChildlessSlot(const NanoGrid<BuildT>& grid, const nanovdb::Coord& c,
                   uint64_t& nodeIdx, uint32_t& slot)
{
    using UpperT = NanoUpper<BuildT>;
    using LowerT = NanoLower<BuildT>;
    const auto& tree = grid.tree();
    const auto* tile = tree.root().probeTile(c);
    if (!tile) return 3;                                        // absent root region -> root sidecar
    if (!tile->isChild()) return 0;                             // root-level value tile: skip
    const UpperT* upper = tree.root().getChild(tile);
    const LowerT* lower = upper->probeChild(c);
    if (!lower) {                                               // childless upper slot
        nodeIdx = util::PtrDiff(upper, tree.template getFirstNode<2>()) / sizeof(UpperT);
        slot    = UpperT::CoordToOffset(c);
        return 2;
    }
    const uint32_t lOff = LowerT::CoordToOffset(c);
    if (!lower->childMask().isOn(lOff)) {                       // childless lower slot
        nodeIdx = util::PtrDiff(lower, tree.template getFirstNode<1>()) / sizeof(LowerT);
        slot    = lOff;
        return 1;
    }
    return 0;                                                   // refined to a leaf
}

/// @brief Coarse-fill seeding: each leaf classifies its 6 faces and records interior/exterior
///        evidence on the childless tile across each one. One block per leaf, 384 threads
///        (6 faces x 64 voxels). A face voxel counts as interior if it is active with sign -1 or
///        inactive with its leaf invert bit ON, and exterior in the mirror case.
///
///        An 8x8 face abuts exactly one 8^3 region, so one probe per face finds the target: a
///        childless tile gets its sawInterior/sawExterior bit OR-ed in, anything refined is skipped.
///        The flood kernel applies the seed gate (sawInterior && !sawExterior).
template <typename BuildT>
struct LeafFaceSeedFunctor
{
    static constexpr int MaxThreadsPerBlock         = 384;  // 6 faces × 64 cells
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(const NanoGrid<BuildT>*  d_grid,
                               const int8_t*            d_sign,        // completed signs (post step 5)
                               const nanovdb::Mask<3>*  d_leafInvert,  // chunk-A leaf invert masks
                               nanovdb::Mask<4>* d_lowerSawInt, nanovdb::Mask<4>* d_lowerSawExt,
                               nanovdb::Mask<5>* d_upperSawInt, nanovdb::Mask<5>* d_upperSawExt)
    {
        const int leafID = blockIdx.x, t = threadIdx.x;
        const auto& leaf = d_grid->tree().template getFirstNode<0>()[leafID];

        __shared__ int sInt[6], sExt[6];
        if (t < 6) { sInt[t] = 0; sExt[t] = 0; }
        __syncthreads();

        // Face voxel of this thread: face = t/64 (0..5 = -x,+x,-y,+y,-z,+z), (a,b) = 8×8 position.
        const int face = t >> 6, a = (t >> 3) & 7, b = t & 7;
        int n;  // voxel offset n = (x<<6)|(y<<3)|z
        switch (face) {
            case 0:  n = (0 << 6) | (a << 3) | b; break;
            case 1:  n = (7 << 6) | (a << 3) | b; break;
            case 2:  n = (a << 6) | (0 << 3) | b; break;
            case 3:  n = (a << 6) | (7 << 3) | b; break;
            case 4:  n = (a << 6) | (b << 3) | 0; break;
            default: n = (a << 6) | (b << 3) | 7; break;
        }
        const bool act      = leaf.isActive(uint32_t(n));
        const bool interior = act ? (d_sign[leaf.getValue(uint32_t(n))] == int8_t(-1))
                                  : d_leafInvert[leafID].isOn(uint32_t(n));
        if (interior) sInt[face] = 1; else sExt[face] = 1;  // benign race: all writers store 1
        __syncthreads();

        if (t < 6) {  // one probe per face
            const int off[6][3] = {{-8,0,0},{8,0,0},{0,-8,0},{0,8,0},{0,0,-8},{0,0,8}};
            const nanovdb::Coord c = leaf.origin().offsetBy(off[t][0], off[t][1], off[t][2]);
            uint64_t nodeIdx; uint32_t slot;
            const int level = probeChildlessSlot(*d_grid, c, nodeIdx, slot);
            if (level == 1) {
                if (sInt[t]) d_lowerSawInt[nodeIdx].setOnAtomic(slot);
                if (sExt[t]) d_lowerSawExt[nodeIdx].setOnAtomic(slot);
            } else if (level == 2) {
                if (sInt[t]) d_upperSawInt[nodeIdx].setOnAtomic(slot);
                if (sExt[t]) d_upperSawExt[nodeIdx].setOnAtomic(slot);
            }
        }
    }
};

/// @brief The lower-level counterpart of LeafFaceSeedFunctor: each LOWER node classifies the
///        childless slots on its 6 faces, using the already-flooded lower invert mask, and seeds the
///        childless UPPER tile across each face. Refined face slots are skipped -- their finer
///        content was contributed by LeafFaceSeedFunctor. A probe landing on another lower node is a
///        same-level neighbour and is left to the flood. One block per lower node, 512 threads over
///        6 faces x 16x16 cells.
template <typename BuildT>
struct LowerFaceSeedFunctor
{
    static constexpr int MaxThreadsPerBlock         = 512;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(const NanoGrid<BuildT>*  d_grid,
                               const nanovdb::Mask<4>*  d_lowerInvert,  // flooded lower invert masks
                               nanovdb::Mask<5>* d_upperSawInt, nanovdb::Mask<5>* d_upperSawExt)
    {
        const int nodeID = blockIdx.x, t = threadIdx.x;
        const auto& node = d_grid->tree().template getFirstNode<1>()[nodeID];

        __shared__ int sInt[6], sExt[6];
        if (t < 6) { sInt[t] = 0; sExt[t] = 0; }
        __syncthreads();

        for (int u = t; u < 6 * 256; u += blockDim.x) {
            const int face = u >> 8, a = (u >> 4) & 15, b = u & 15;
            int n;  // lower slot offset n = (x<<8)|(y<<4)|z
            switch (face) {
                case 0:  n = ( 0 << 8) | (a << 4) | b; break;
                case 1:  n = (15 << 8) | (a << 4) | b; break;
                case 2:  n = (a << 8) | ( 0 << 4) | b; break;
                case 3:  n = (a << 8) | (15 << 4) | b; break;
                case 4:  n = (a << 8) | (b << 4) |  0; break;
                default: n = (a << 8) | (b << 4) | 15; break;
            }
            if (node.childMask().isOn(uint32_t(n))) continue;  // refined: leaf faces already contributed
            if (d_lowerInvert[nodeID].isOn(uint32_t(n))) sInt[face] = 1; else sExt[face] = 1;
        }
        __syncthreads();

        if (t < 6) {  // whole 128×128 face abuts exactly one 128^3 region across
            const int off[6][3] = {{-128,0,0},{128,0,0},{0,-128,0},{0,128,0},{0,0,-128},{0,0,128}};
            const nanovdb::Coord c = node.origin().offsetBy(off[t][0], off[t][1], off[t][2]);
            uint64_t nodeIdx; uint32_t slot;
            if (probeChildlessSlot(*d_grid, c, nodeIdx, slot) == 2) {
                if (sInt[t]) d_upperSawInt[nodeIdx].setOnAtomic(slot);
                if (sExt[t]) d_upperSawExt[nodeIdx].setOnAtomic(slot);
            }
        }
    }
};

/// @brief Within-node flood at a coarse level (LEVEL 1 = lower/16^3, 2 = upper/32^3): the coarse
///        analogue of FillLeafInvertMaskFunctor. Seeds are childless slots with interior evidence and
///        none to the contrary (mixed evidence errs to exterior, the safe side), then a monotone
///        ON-flood over 6-adjacent childless slots with refined slots as walls.
///
///        The flood cannot leak interior -> exterior because two adjacent CHILDLESS slots can never
///        straddle the surface -- a surface between them would have forced refinement. One block per
///        node, 512 threads striding the slots, slot bits packed in shared memory.
template <typename BuildT, int LEVEL>
struct CoarseInvertFloodFunctor
{
    static constexpr int LOG2DIM = (LEVEL == 1) ? 4 : 5;
    static constexpr int DIM     = 1 << LOG2DIM;             // 16 / 32
    static constexpr int SLOTS   = 1 << (3 * LOG2DIM);       // 4096 / 32768
    static constexpr int WORDS   = SLOTS >> 6;               // 64 / 512
    using MaskT = nanovdb::Mask<LOG2DIM>;

    static constexpr int MaxThreadsPerBlock         = 512;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(const NanoGrid<BuildT>* d_grid,
                               const MaskT* d_sawInt, const MaskT* d_sawExt, MaskT* d_invert)
    {
        const int nodeID = blockIdx.x, t = threadIdx.x;
        const auto& node = d_grid->tree().template getFirstNode<LEVEL>()[nodeID];

        __shared__ uint64_t sWall[WORDS];  // refined slots (childMask)
        __shared__ uint64_t sInv [WORDS];  // result bits (seeded, then flooded)
        __shared__ int      sChanged;

        for (int w = t; w < WORDS; w += blockDim.x) {
            const uint64_t wall = node.childMask().words()[w];
            sWall[w] = wall;
            sInv[w]  = d_sawInt[nodeID].words()[w] & ~d_sawExt[nodeID].words()[w] & ~wall;
        }
        __syncthreads();

        auto isOn = [&](const uint64_t* m, int s) { return (m[s >> 6] >> (s & 63)) & 1ull; };

        // Monotone ON-flood: racy same-sweep reads only accelerate legitimate propagation (a set bit
        // is final truth), so in-place atomicOr is safe. Cap = Manhattan diameter + slack.
        for (int it = 0; it < 3 * DIM + 16; ++it) {
            if (t == 0) sChanged = 0;
            __syncthreads();
            bool any = false;
            for (int s = t; s < SLOTS; s += blockDim.x) {
                if (isOn(sWall, s) || isOn(sInv, s)) continue;
                const int x = s >> (2 * LOG2DIM), y = (s >> LOG2DIM) & (DIM - 1), z = s & (DIM - 1);
                constexpr int dx = 1 << (2 * LOG2DIM), dy = 1 << LOG2DIM;
                // inv bits exist only on childless slots, so a set neighbor bit is a valid feeder
                if ((x > 0       && isOn(sInv, s - dx)) || (x < DIM - 1 && isOn(sInv, s + dx)) ||
                    (y > 0       && isOn(sInv, s - dy)) || (y < DIM - 1 && isOn(sInv, s + dy)) ||
                    (z > 0       && isOn(sInv, s -  1)) || (z < DIM - 1 && isOn(sInv, s +  1))) {
                    ::atomicOr(reinterpret_cast<unsigned long long*>(&sInv[s >> 6]), 1ull << (s & 63));
                    any = true;
                }
            }
            if (any) sChanged = 1;
            __syncthreads();
            const bool done = (sChanged == 0);  // latch, then barrier: next-iter reset can't race the read
            __syncthreads();
            if (done) break;
        }

        for (int w = t; w < WORDS; w += blockDim.x)
            d_invert[nodeID].words()[w] = sInv[w];
    }
};

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Step 6C - the deep interior beyond any upper node, held in a small
// provisional P×Q×R cell array over the grid's root-tile bounding range (one cell per 4096^3 root
// region) — a SIDECAR consulted by signedSignAt(); the grid itself is never mutated.

/// @brief Mark the WALL cells of the root-cell array: cells whose 4096^3 region has a pre-existing
///        root entry (an upper child — the band passes through). One thread per cell via lambdaKernel.
template <typename BuildT>
struct RootWallMarkFunctor
{
    const NanoGrid<BuildT>* dGrid;
    uint8_t*                dWall;
    nanovdb::Coord          tileMin;  // root-cell range origin, in 4096-tile units
    nanovdb::Coord          dims;     // P×Q×R

    __device__ void operator()(size_t idx) const
    {
        const int k = int(idx) % dims[2], j = (int(idx) / dims[2]) % dims[1], i = int(idx) / (dims[1] * dims[2]);
        const nanovdb::Coord c((tileMin[0] + i) << 12, (tileMin[1] + j) << 12, (tileMin[2] + k) << 12);
        const auto* tile = dGrid->tree().root().probeTile(c);
        dWall[idx] = (tile && tile->isChild()) ? 1 : 0;
    }
};

/// @brief Chunk-C seeding: accumulate interior/exterior evidence from node faces that abut an ABSENT
///        root region, from ALL THREE levels (a deep-interior root cell can be abutted by a childless
///        upper tile, a childless lower tile, OR a leaf's inactive-interior voxels — seeding from the
///        upper level alone would miss the finer abutments). One block per node; face cells classified
///        exactly as in the chunk-B seed functors (leaf: sign / leaf-invert; lower/upper: childless
///        slot invert bit, refined slots skipped); each whole face abuts exactly ONE root cell, probed
///        once. Out-of-range cells (beyond the array) are skipped — they stay exterior by default.
///        // classification duplicated from Leaf/LowerFaceSeedFunctor — keep in sync
template <typename BuildT, int LEVEL>  // 0 = leaf, 1 = lower, 2 = upper
struct RootFaceSeedFunctor
{
    static constexpr int MaxThreadsPerBlock         = (LEVEL == 0) ? 384 : 512;
    static constexpr int MinBlocksPerMultiprocessor = 1;
    static constexpr int NODE_DIM = (LEVEL == 0) ? 8 : (LEVEL == 1) ? 128 : 4096;

    __device__ void operator()(const NanoGrid<BuildT>*  d_grid,
                               const int8_t*            d_sign,
                               const nanovdb::Mask<3>*  d_leafInvert,
                               const nanovdb::Mask<4>*  d_lowerInvert,
                               const nanovdb::Mask<5>*  d_upperInvert,
                               uint8_t* d_sawInt, uint8_t* d_sawExt,
                               nanovdb::Coord tileMin, nanovdb::Coord dims)
    {
        const int nodeID = blockIdx.x, t = threadIdx.x;
        const auto& node = d_grid->tree().template getFirstNode<LEVEL>()[nodeID];

        __shared__ int sInt[6], sExt[6];
        if (t < 6) { sInt[t] = 0; sExt[t] = 0; }
        __syncthreads();

        if constexpr (LEVEL == 0) {
            const int face = t >> 6, a = (t >> 3) & 7, b = t & 7;
            int n;
            switch (face) {
                case 0:  n = (0 << 6) | (a << 3) | b; break;
                case 1:  n = (7 << 6) | (a << 3) | b; break;
                case 2:  n = (a << 6) | (0 << 3) | b; break;
                case 3:  n = (a << 6) | (7 << 3) | b; break;
                case 4:  n = (a << 6) | (b << 3) | 0; break;
                default: n = (a << 6) | (b << 3) | 7; break;
            }
            const bool act      = node.isActive(uint32_t(n));
            const bool interior = act ? (d_sign[node.getValue(uint32_t(n))] == int8_t(-1))
                                      : d_leafInvert[nodeID].isOn(uint32_t(n));
            if (interior) sInt[face] = 1; else sExt[face] = 1;
        } else if constexpr (LEVEL == 1) {
            for (int u = t; u < 6 * 256; u += blockDim.x) {
                const int face = u >> 8, a = (u >> 4) & 15, b = u & 15;
                int n;
                switch (face) {
                    case 0:  n = ( 0 << 8) | (a << 4) | b; break;
                    case 1:  n = (15 << 8) | (a << 4) | b; break;
                    case 2:  n = (a << 8) | ( 0 << 4) | b; break;
                    case 3:  n = (a << 8) | (15 << 4) | b; break;
                    case 4:  n = (a << 8) | (b << 4) |  0; break;
                    default: n = (a << 8) | (b << 4) | 15; break;
                }
                if (node.childMask().isOn(uint32_t(n))) continue;  // refined: finer level contributes
                if (d_lowerInvert[nodeID].isOn(uint32_t(n))) sInt[face] = 1; else sExt[face] = 1;
            }
        } else {
            for (int u = t; u < 6 * 1024; u += blockDim.x) {
                const int face = u >> 10, a = (u >> 5) & 31, b = u & 31;
                int n;
                switch (face) {
                    case 0:  n = ( 0 << 10) | (a << 5) | b; break;
                    case 1:  n = (31 << 10) | (a << 5) | b; break;
                    case 2:  n = (a << 10) | ( 0 << 5) | b; break;
                    case 3:  n = (a << 10) | (31 << 5) | b; break;
                    case 4:  n = (a << 10) | (b << 5) |  0; break;
                    default: n = (a << 10) | (b << 5) | 31; break;
                }
                if (node.childMask().isOn(uint32_t(n))) continue;
                if (d_upperInvert[nodeID].isOn(uint32_t(n))) sInt[face] = 1; else sExt[face] = 1;
            }
        }
        __syncthreads();

        if (t < 6) {  // one probe per face: the whole face abuts exactly one root cell
            const int off[6][3] = {{-NODE_DIM,0,0},{NODE_DIM,0,0},{0,-NODE_DIM,0},{0,NODE_DIM,0},{0,0,-NODE_DIM},{0,0,NODE_DIM}};
            const nanovdb::Coord c = node.origin().offsetBy(off[t][0], off[t][1], off[t][2]);
            uint64_t nodeIdx; uint32_t slot;
            if (probeChildlessSlot(*d_grid, c, nodeIdx, slot) == 3) {  // absent root region
                const int i = (c[0] >> 12) - tileMin[0], j = (c[1] >> 12) - tileMin[1], k = (c[2] >> 12) - tileMin[2];
                if (i >= 0 && i < dims[0] && j >= 0 && j < dims[1] && k >= 0 && k < dims[2]) {
                    const int idx = (i * dims[1] + j) * dims[2] + k;
                    if (sInt[t]) d_sawInt[idx] = 1;  // benign race: all writers store 1
                    if (sExt[t]) d_sawExt[idx] = 1;
                }
            }
        }
    }
};

/// @brief Chunk-C multi-seed flood on the P×Q×R root-cell array (single block; the array is a handful
///        of cells). Seed = sawInterior && !sawExterior && !wall (mixed evidence => exterior, the safe
///        side); then flood ON among non-wall cells over 6-adjacency — WALL cells (pre-existing root
///        entries) block. Every interior-abutting cell seeds (multi-seed), so disconnected interiors
///        all fill. The surface at 4096 granularity always lies inside a wall cell, so the flood
///        cannot leak interior -> exterior. Grid-independent (plain arrays) so it is unit-testable.
struct RootInteriorFloodFunctor
{
    static constexpr int MaxThreadsPerBlock         = 256;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(const uint8_t* d_wall, const uint8_t* d_sawInt, const uint8_t* d_sawExt,
                               uint8_t* d_on, nanovdb::Coord dims)
    {
        const int t = threadIdx.x;
        const int P = dims[0], Q = dims[1], R = dims[2], total = P * Q * R;
        __shared__ int sChanged;

        for (int c = t; c < total; c += blockDim.x)
            d_on[c] = (!d_wall[c] && d_sawInt[c] && !d_sawExt[c]) ? 1 : 0;
        __syncthreads();

        for (int it = 0; it < total + 2; ++it) {  // cap: any path length < total cells
            if (t == 0) sChanged = 0;
            __syncthreads();
            bool any = false;
            for (int c = t; c < total; c += blockDim.x) {
                if (d_wall[c] || d_on[c]) continue;
                const int k = c % R, j = (c / R) % Q, i = c / (Q * R);
                const bool on =
                    (i > 0     && d_on[c - Q * R]) || (i < P - 1 && d_on[c + Q * R]) ||
                    (j > 0     && d_on[c - R])     || (j < Q - 1 && d_on[c + R])     ||
                    (k > 0     && d_on[c - 1])     || (k < R - 1 && d_on[c + 1]);
                if (on) { d_on[c] = 1; any = true; }  // monotone: racy same-sweep reads only accelerate
            }
            if (any) sChanged = 1;
            __syncthreads();
            const bool done = (sChanged == 0);  // latch, then barrier: next-iter reset can't race the read
            __syncthreads();
            if (done) break;
        }
    }
};

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Query - compose the per-level sidecars into a sign at any coordinate.
// Descends the tree and answers from whichever level owns the coordinate: the sign sidecar for an active
// voxel, the leaf/lower/upper invert mask for an inactive voxel or a childless tile, and the
// root-interior array beyond the last upper node.

/// @brief Full-domain sign query (host + device): the completed level-set sign at ANY index coord,
///        composed from the grid plus the step-6 sidecars — no grid mutation anywhere. Descends:
///          active voxel            -> sign sidecar (±1)
///          inactive leaf voxel     -> leaf invert bit
///          childless lower slot    -> lower invert bit
///          childless upper slot    -> upper invert bit
///          ABSENT root region      -> root-interior sidecar (in-range && ON => interior, else exterior)
/// @return +1 outside / -1 inside.
template <typename BuildT>
__hostdev__ inline int8_t
signedSignAt(const NanoGrid<BuildT>& grid, const nanovdb::Coord& ijk,
             const int8_t* sign, const nanovdb::Mask<3>* leafInvert,
             const nanovdb::Mask<4>* lowerInvert, const nanovdb::Mask<5>* upperInvert,
             const uint8_t* rootInterior, const nanovdb::Coord& rootTileMin, const nanovdb::Coord& rootDims)
{
    using UpperT = NanoUpper<BuildT>;
    using LowerT = NanoLower<BuildT>;
    using LeafT  = NanoLeaf<BuildT>;
    const auto& tree = grid.tree();
    const auto* tile = tree.root().probeTile(ijk);
    if (!tile || !tile->isChild()) {  // absent root region (or root value tile): consult the sidecar
        const int i = (ijk[0] >> 12) - rootTileMin[0],
                  j = (ijk[1] >> 12) - rootTileMin[1],
                  k = (ijk[2] >> 12) - rootTileMin[2];
        const bool interior = rootInterior &&
            i >= 0 && i < rootDims[0] && j >= 0 && j < rootDims[1] && k >= 0 && k < rootDims[2] &&
            rootInterior[(i * rootDims[1] + j) * rootDims[2] + k];
        return interior ? int8_t(-1) : int8_t(1);
    }
    const UpperT* upper = tree.root().getChild(tile);
    const LowerT* lower = upper->probeChild(ijk);
    if (!lower) {
        const uint64_t u = util::PtrDiff(upper, tree.template getFirstNode<2>()) / sizeof(UpperT);
        return upperInvert[u].isOn(UpperT::CoordToOffset(ijk)) ? int8_t(-1) : int8_t(1);
    }
    const LeafT* leaf = lower->probeChild(ijk);
    if (!leaf) {
        const uint64_t l = util::PtrDiff(lower, tree.template getFirstNode<1>()) / sizeof(LowerT);
        return lowerInvert[l].isOn(LowerT::CoordToOffset(ijk)) ? int8_t(-1) : int8_t(1);
    }
    const uint32_t n = LeafT::CoordToOffset(ijk);
    if (leaf->isActive(n)) return sign[leaf->getValue(n)];
    const uint64_t lf = util::PtrDiff(leaf, tree.template getFirstNode<0>()) / sizeof(LeafT);
    return leafInvert[lf].isOn(n) ? int8_t(-1) : int8_t(1);
}

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Partition and composition - the stages MeshToSDF runs around the per-surface signing: carving one
// surface out of the rasterized band, picking a representative voxel for it, probing another
// surface's field at that voxel, and folding the resulting nesting parity into the signs.

// Retain mask selecting one surface's voxels out of the original grid: one block per leaf, one thread
// per voxel offset, bit ON iff the voxel carries the target surface label.
template <typename BuildT>
struct SurfaceMaskFunctor // TODO: How about changing it to RetainMaskFunctor or something like that?
{
    static constexpr int MaxThreadsPerBlock         = 512;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(const NanoGrid<BuildT>* d_grid, const uint32_t* d_surfaceLabel,
                               uint32_t target, Mask<3>* d_masks)
    {
        const int leafID = blockIdx.x, n = threadIdx.x;
        const auto& leaf = d_grid->tree().template getFirstNode<0>()[leafID];
        auto&       mask = d_masks[leafID];
        if (n < int(Mask<3>::WORD_COUNT)) mask.words()[n] = 0UL;
        __syncthreads();
        if (auto v = leaf.data()->getValue(uint32_t(n)))          // v != 0 => active voxel
            if (d_surfaceLabel[v] == target) mask.setOnAtomic(uint32_t(n));
    }
};

// Pack a voxel coordinate into one sortable key so a per-surface atomicMin picks a deterministic
// representative voxel. 21 bits per axis covers |coord| < 2^20, far beyond any rasterized grid.
__hostdev__ inline unsigned long long packCoord(const Coord& c)
{
    return ((unsigned long long)(c[0] + (1 << 20)) << 42) |
           ((unsigned long long)(c[1] + (1 << 20)) << 21) |
            (unsigned long long)(c[2] + (1 << 20));
}
// One representative voxel per surface (the packed-coordinate minimum, so it is deterministic).
template <typename BuildT>
struct SurfaceRepFunctor
{
    static constexpr int MaxThreadsPerBlock         = 512;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(const NanoGrid<BuildT>* d_grid, const uint32_t* d_surfaceLabel,
                               uint32_t surfaceCount, unsigned long long* d_repKey)
    {
        const int leafID = blockIdx.x, n = threadIdx.x;
        const auto& leaf = d_grid->tree().template getFirstNode<0>()[leafID];
        if (!leaf.isActive(uint32_t(n))) return;
        const uint32_t s = d_surfaceLabel[leaf.getValue(uint32_t(n))];
        if (s >= surfaceCount) return;
        const Coord ijk = leaf.origin() + NanoLeaf<BuildT>::OffsetToLocalCoord(uint32_t(n));
        atomicMin(&d_repKey[s], packCoord(ijk));
    }
};

// Ask one surface's sign field about every surface's representative voxel: out[j] < 0 means that
// surface's band lies inside this one.
template <typename BuildT>
struct InclusionProbeFunctor
{
    __device__ void operator()(size_t j, const NanoGrid<BuildT>* d_gridI,
                               const unsigned long long* d_repKey, const int8_t* d_signI,
                               const Mask<3>* d_leafI, const nanovdb::Mask<4>* d_lowI,
                               const nanovdb::Mask<5>* d_upI, const uint8_t* d_rootI,
                               Coord rootMin, Coord rootDims, int8_t* d_out) const
    {
        const unsigned long long k = d_repKey[j];
        const Coord ijk(int((k >> 42) & 0x1FFFFF) - (1 << 20),
                                 int((k >> 21) & 0x1FFFFF) - (1 << 20),
                                 int( k        & 0x1FFFFF) - (1 << 20));
        d_out[j] = signedSignAt<BuildT>(
            *d_gridI, ijk, d_signI, d_leafI, d_lowI, d_upI, d_rootI, rootMin, rootDims);
    }
};

// Flip each voxel's sign once per enclosing surface: sign *= (-1)^depth(surface).
/// @brief CUDA functor: turn per-surface signs into the composed sign, under either nesting rule.
///        One thread per sidecar slot via lambdaKernel.
///
///        Both rules are the same count. A voxel belongs to exactly one surface, and its sign there
///        already says whether it is inside THAT one, so the number of surfaces enclosing it is the
///        surface's own nesting depth plus one if the voxel sits inside it. EvenOdd then calls the
///        voxel interior on an odd count, Solid on any non-zero count -- which leaves only the
///        outermost boundary of each object separating inside from outside.
struct ResolveNestingFunctor
{
    __device__ void operator()(size_t v, const uint32_t* d_surfaceLabel, const uint32_t* d_depth,
                               uint32_t surfaceCount, bool evenOdd, int8_t* d_sign) const
    {
        if (v == 0) return;                                  // slot 0 is the background
        const uint32_t s = d_surfaceLabel[v];
        if (s >= surfaceCount) return;
        const uint32_t enclosing = d_depth[s] + (d_sign[v] < int8_t(0) ? 1u : 0u);
        const bool interior = evenOdd ? ((enclosing & 1u) != 0u) : (enclosing != 0u);
        d_sign[v] = interior ? int8_t(-1) : int8_t(1);
    }
};// sdf_detail::ResolveNestingFunctor

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
template <typename BufferT>
GridHandle<BufferT>
SurfaceSigner<BuildT>::computeDerivedTopology(const GridT* d_srcGrid, const float* d_udf,
                                          float voxelSize, float isoValue, const BufferT& buffer)
{
    using PruneOp = UDFBarrierPruneMaskFunctor<BuildT>; //todo: this PruneOp is too misleading let's be more specific. PruneBarrierOp or something like that.

    // Barrier threshold √3/2 voxels expressed in the sidecar's WORLD units, squared.
    const float    barrierSqWorld = 0.75f * voxelSize * voxelSize;
    const uint32_t srcLeafCount = leafCountOf(d_srcGrid);

    // Leaf-indexed retain mask: one Mask<3> (512 bits) per source leaf (device-only).
    auto  retainMask   = nanovdb::cuda::DeviceBuffer::create(
        std::size_t(srcLeafCount) * sizeof(nanovdb::Mask<3>), nullptr, false);
    auto* d_retainMask = static_cast<nanovdb::Mask<3>*>(retainMask.deviceData());
    // todo: here we may have to consider that barrier voxel should be voxel neighbor to outside voxels.
    if (mVerbose==1) mTimer.start("Prune barrier shell -> derived topology");
    util::cuda::operatorKernel<PruneOp><<<srcLeafCount, PruneOp::MaxThreadsPerBlock, 0, mStream>>>(
        d_srcGrid, d_udf, isoValue, barrierSqWorld, d_retainMask);
    cudaCheckError();

    // Topological pruning -> clean, topology-only derived index grid (UDF no longer needed).
    PruneGrid<BuildT> pruner(d_srcGrid, d_retainMask, mStream);
    pruner.setVerbose(mVerbose);
    auto handle = pruner.template getHandle<BufferT>(buffer);
    if (mVerbose==1) mTimer.stop();
    return handle;
}// SurfaceSigner<BuildT>::computeDerivedTopology

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
void SurfaceSigner<BuildT>::signNonBarrier(const GridT* d_grid, const uint32_t* d_voxelLabel)
{
    const uint32_t leafCount = leafCountOf(d_grid);
    mExteriorRep = 0;
    if (leafCount == 0) return;  // every materialized leaf has >=1 component, so leafCount==0 => K==0

    const uint64_t activeCount = activeCountOf(d_grid);

    // (1) The exterior representative = component of the grid's minimum-x active voxel.
    if (mVerbose==1) mTimer.start("Sign: find exterior component");
    auto minKeyBuf = nanovdb::cuda::DeviceBuffer::create(sizeof(unsigned long long), nullptr, false);
    auto* d_minKey = static_cast<unsigned long long*>(minKeyBuf.deviceData());
    cudaCheck(cudaMemsetAsync(d_minKey, 0xFF, sizeof(unsigned long long), mStream));   // ~0ull
    using FindOp = FindExteriorRepFunctor<BuildT>;
    util::cuda::operatorKernel<FindOp><<<leafCount, FindOp::MaxThreadsPerBlock, 0, mStream>>>(
        d_grid, d_voxelLabel, d_minKey);
    cudaCheckError();

    unsigned long long minKey = 0;   // low 32 bits of the min key = the exterior representative
    cudaCheck(cudaMemcpyAsync(&minKey, d_minKey, sizeof(minKey), cudaMemcpyDeviceToHost, mStream));
    cudaCheck(cudaStreamSynchronize(mStream));
    mExteriorRep = uint64_t(uint32_t(minKey & 0xFFFFFFFFull));
    if (mVerbose==1) mTimer.stop();

    // (2) Per-voxel sign: +1 exterior / -1 interior (slot 0 = background +1).
    mVoxelSign = nanovdb::cuda::DeviceBuffer::create((activeCount + 1) * sizeof(int8_t), nullptr, false);
    cudaCheck(cudaMemsetAsync(mVoxelSign.deviceData(), 1, (activeCount + 1) * sizeof(int8_t), mStream));// all +1
    using SignOp = SignNonBarrierFunctor<BuildT>;
    if (mVerbose==1) mTimer.start("Sign: write per-voxel signs");
    util::cuda::operatorKernel<SignOp><<<leafCount, SignOp::MaxThreadsPerBlock, 0, mStream>>>(
        d_grid, d_voxelLabel, uint32_t(mExteriorRep), deviceVoxelSign());
    cudaCheckError();
    if (mVerbose==1) mTimer.stop();
}// SurfaceSigner<BuildT>::signNonBarrier

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
void SurfaceSigner<BuildT>::injectSignsToOriginal(const GridT* d_origGrid, const GridT* d_derivedGrid)
{
    const uint64_t origActive = activeCountOf(d_origGrid);
    const uint32_t derivedLeafCount = leafCountOf(d_derivedGrid);

    // Sentinel 0 ("unsigned barrier") everywhere; slot 0 = background (+1). Non-barrier voxels are
    // overwritten by the injection below; barrier voxels (in original but not derived) keep 0.
    mOriginalVoxelSign = nanovdb::cuda::DeviceBuffer::create((origActive + 1) * sizeof(int8_t), nullptr, false);
    cudaCheck(cudaMemsetAsync(mOriginalVoxelSign.deviceData(), 0, (origActive + 1) * sizeof(int8_t), mStream));
    cudaCheck(cudaMemsetAsync(mOriginalVoxelSign.deviceData(), 1, sizeof(int8_t), mStream)); // slot 0 = +1

    if (derivedLeafCount == 0) return;  // nothing signed -> all voxels stay sentinel

    // Inject derived signs into the original sidecar at the intersection (= every non-barrier voxel,
    // since derived ⊂ original). One block per derived (source) leaf.
    using InjectOp = util::cuda::InjectGridDataFunctor<BuildT, int8_t>;
    if (mVerbose==1) mTimer.start("Inject derived signs -> original grid");
    util::cuda::operatorKernel<InjectOp><<<derivedLeafCount, InjectOp::MaxThreadsPerBlock, 0, mStream>>>(
        d_derivedGrid, d_origGrid, deviceVoxelSign(), deviceOriginalVoxelSign());
    cudaCheckError();
    if (mVerbose==1) mTimer.stop();
}// SurfaceSigner<BuildT>::injectSignsToOriginal

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
void SurfaceSigner<BuildT>::signBarrierAsInterior(const GridT* d_grid)
{
    const uint64_t activeCount = activeCountOf(d_grid);
    const uint64_t slots       = activeCount + 1;

    mSignedVoxelSign = nanovdb::cuda::DeviceBuffer::create(slots * sizeof(int8_t), nullptr, false);
    cudaCheck(cudaMemsetAsync(mSignedVoxelSign.deviceData(), 0, slots * sizeof(int8_t), mStream));
    cudaCheck(cudaMemsetAsync(mSignedVoxelSign.deviceData(), 1, sizeof(int8_t), mStream)); // slot 0 = +1
    if (leafCountOf(d_grid) == 0) return;

    if (mVerbose==1) mTimer.start("Sign: barrier voxels (all interior)");
    util::cuda::lambdaKernel<<<(unsigned int)((slots + 255) / 256), 256, 0, mStream>>>(
        slots, BarrierToInteriorFunctor{}, deviceOriginalVoxelSign(), deviceSignedVoxelSign());
    cudaCheckError();
    // The kernel also writes slot 0, which holds +1 rather than a voxel sign; restore it.
    const int8_t one = 1;
    cudaCheck(cudaMemcpyAsync(mSignedVoxelSign.deviceData(), &one, sizeof(int8_t),
                              cudaMemcpyHostToDevice, mStream));
    cudaCheck(cudaStreamSynchronize(mStream));
    if (mVerbose==1) mTimer.stop();
}// SurfaceSigner<BuildT>::signBarrierAsInterior

template <typename BuildT>
void SurfaceSigner<BuildT>::signBarrier(const GridT* d_grid, const uint32_t* d_index,
                                    const nanovdb::Vec3f* d_points, const nanovdb::Vec3i* d_triangles,
                                    const nanovdb::Map& map, float isoValue, float voxelSize)
{
    const uint64_t activeCount = activeCountOf(d_grid);
    const uint32_t leafCount = leafCountOf(d_grid);

    // Output: every active voxel ends up ±1. Start at 0, set slot 0 (background) = +1; the kernel
    // writes every active voxel (carrying non-barrier signs through, filling barriers).
    mSignedVoxelSign = nanovdb::cuda::DeviceBuffer::create((activeCount + 1) * sizeof(int8_t), nullptr, false);
    cudaCheck(cudaMemsetAsync(mSignedVoxelSign.deviceData(), 0, (activeCount + 1) * sizeof(int8_t), mStream));
    cudaCheck(cudaMemsetAsync(mSignedVoxelSign.deviceData(), 1, sizeof(int8_t), mStream)); // slot 0 = +1
    if (leafCount == 0) return;

    using Op = SignBarrierFunctor<BuildT>;
    if (mVerbose==1) mTimer.start("Sign: barrier voxels (intersecting-voxel-sign mirror)");
    util::cuda::operatorKernel<Op><<<leafCount, Op::MaxThreadsPerBlock, 0, mStream>>>(
        d_grid, deviceOriginalVoxelSign(), deviceSignedVoxelSign(),
        d_index, d_points, d_triangles, map,
        (voxelSize > 0.f) ? double(isoValue) / double(voxelSize) : 0.0);
    cudaCheckError();
    if (mVerbose==1) mTimer.stop();
}// SurfaceSigner<BuildT>::signBarrier

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

/// Ball-intersection certification of the barrier voxels. Seeds are the non-barrier signs already in
/// deviceOriginalVoxelSign(); each Jacobi round lets an undecided voxel take the sign of any certain
/// neighbour whose ball overlaps its own. Ping-pongs two label buffers so no round reads what it
/// writes, and stops when a round decides nothing.
template <typename BuildT>
void SurfaceSigner<BuildT>::signBarrierByBalls(const GridT* d_grid, const float* d_udf,
                                               float voxelSize, int maxRounds, int radius,
                                               float isoValue)
{
    const uint64_t activeCount = activeCountOf(d_grid);
    const uint32_t leafCount   = leafCountOf(d_grid);
    const std::size_t bytes    = std::size_t(activeCount + 1) * sizeof(int8_t);

    mBallVoxelSign = nanovdb::cuda::DeviceBuffer::create(bytes, nullptr, false);
    mBallUndecided = mBallContradictions = mBallRounds = 0;
    if (leafCount == 0) return;

    // Seed both buffers from the non-barrier signs; barrier voxels start at 0.
    auto scratch = nanovdb::cuda::DeviceBuffer::create(bytes, nullptr, false);
    cudaCheck(cudaMemcpyAsync(mBallVoxelSign.deviceData(), deviceOriginalVoxelSign(), bytes,
                              cudaMemcpyDeviceToDevice, mStream));
    cudaCheck(cudaMemcpyAsync(scratch.deviceData(), deviceOriginalVoxelSign(), bytes,
                              cudaMemcpyDeviceToDevice, mStream));

    auto  counters = nanovdb::cuda::DeviceBuffer::create(2 * sizeof(uint32_t), nullptr, false);
    auto* d_counters = static_cast<uint32_t*>(counters.deviceData());

    // One BIT per slot, zeroed ONCE and never per round, so a contradicted voxel is counted the
    // first time it is seen and not again: it is re-detected every round (a contradiction writes
    // back 0, which is not a change, so the voxel never settles).
    const std::size_t contraWords = std::size_t((activeCount + 1 + 31) / 32);
    auto  everContra   = nanovdb::cuda::DeviceBuffer::create(contraWords * sizeof(uint32_t), nullptr, false);
    auto* d_everContra = static_cast<uint32_t*>(everContra.deviceData());
    cudaCheck(cudaMemsetAsync(d_everContra, 0, contraWords * sizeof(uint32_t), mStream));

    cudaCheck(cudaMemsetAsync(d_counters, 0, 2 * sizeof(uint32_t), mStream));

    int8_t* labelIn  = static_cast<int8_t*>(mBallVoxelSign.deviceData());
    int8_t* labelOut = static_cast<int8_t*>(scratch.deviceData());

    using Op = BallCertifyFunctor<BuildT>;
    if (mVerbose==1) mTimer.start("Sign: barrier voxels (ball certification)");
    uint32_t host[2] = {0, 0};
    for (int r = 0; r < maxRounds; ++r) {
        // Only 'changed' resets; the contradiction tally accumulates across rounds by construction.
        cudaCheck(cudaMemsetAsync(d_counters, 0, sizeof(uint32_t), mStream));
        util::cuda::operatorKernel<Op><<<leafCount, Op::MaxThreadsPerBlock, 0, mStream>>>(
            d_grid, labelIn, labelOut, d_udf, voxelSize, d_counters, d_counters + 1, d_everContra,
            radius, isoValue);
        cudaCheckError();
        cudaCheck(cudaMemcpyAsync(host, d_counters, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost, mStream));
        cudaCheck(cudaStreamSynchronize(mStream));
        std::swap(labelIn, labelOut);
        mBallContradictions = host[1];              // running total of distinct contradicted voxels
        if (host[0] == 0) { mBallRounds = uint32_t(r); break; }
        mBallRounds = uint32_t(r + 1);
    }
    if (mVerbose==1) mTimer.stop();

    // labelIn holds the newest labels; make sure that is the buffer we hand back.
    if (labelIn != mBallVoxelSign.deviceData())
        cudaCheck(cudaMemcpyAsync(mBallVoxelSign.deviceData(), labelIn, bytes,
                                  cudaMemcpyDeviceToDevice, mStream));

    // Complete the field: unproven voxels default to interior, and are counted.
    mSignedVoxelSign = nanovdb::cuda::DeviceBuffer::create(bytes, nullptr, false);
    cudaCheck(cudaMemsetAsync(d_counters, 0, sizeof(uint32_t), mStream));
    util::cuda::lambdaKernel<<<(unsigned int)((activeCount + 256) / 256), 256, 0, mStream>>>(
        activeCount + 1, BallFinalizeFunctor{}, deviceBallVoxelSign(), deviceSignedVoxelSign(),
        d_counters);
    cudaCheckError();
    cudaCheck(cudaMemcpyAsync(&mBallUndecided, d_counters, sizeof(uint32_t),
                              cudaMemcpyDeviceToHost, mStream));
    cudaCheck(cudaStreamSynchronize(mStream));
}// SurfaceSigner<BuildT>::signBarrierByBalls

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
void SurfaceSigner<BuildT>::fillLeafInvertMask(const GridT* d_grid, const int8_t* d_sign)
{
    const int8_t* sign = d_sign ? d_sign : deviceSignedVoxelSign();  // external signs override the member
    const uint32_t leafCount = leafCountOf(d_grid);
    if (leafCount == 0) { mLeafInvertMask = nanovdb::cuda::DeviceBuffer(); return; }

    mLeafInvertMask = nanovdb::cuda::DeviceBuffer::create(
        std::size_t(leafCount) * sizeof(nanovdb::Mask<3>), nullptr, false);
    cudaCheck(cudaMemsetAsync(mLeafInvertMask.deviceData(), 0,
                              std::size_t(leafCount) * sizeof(nanovdb::Mask<3>), mStream));

    using Op = FillLeafInvertMaskFunctor<BuildT>;
    if (mVerbose==1) mTimer.start("Fill leaf invert mask (inactive-voxel interior flood)");
    util::cuda::operatorKernel<Op><<<leafCount, Op::MaxThreadsPerBlock, 0, mStream>>>(
        d_grid, sign, deviceLeafInvertMask());
    cudaCheckError();
    if (mVerbose==1) mTimer.stop();
}// SurfaceSigner<BuildT>::fillLeafInvertMask

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
void SurfaceSigner<BuildT>::fillCoarseInvertMasks(const GridT* d_grid, const int8_t* d_sign)
{
    const int8_t* sign = d_sign ? d_sign : deviceSignedVoxelSign();  // external signs override the member
    const auto     treeData   = util::cuda::DeviceGridTraits<BuildT>::getTreeData(d_grid);
    const uint32_t leafCount  = treeData.mNodeCount[0];
    const uint32_t lowerCount = treeData.mNodeCount[1];
    const uint32_t upperCount = treeData.mNodeCount[2];

    const std::size_t lowerBytes = std::size_t(lowerCount) * sizeof(nanovdb::Mask<4>);
    const std::size_t upperBytes = std::size_t(upperCount) * sizeof(nanovdb::Mask<5>);
    mLowerInvertMask = lowerCount ? nanovdb::cuda::DeviceBuffer::create(lowerBytes, nullptr, false)
                                  : nanovdb::cuda::DeviceBuffer();
    mUpperInvertMask = upperCount ? nanovdb::cuda::DeviceBuffer::create(upperBytes, nullptr, false)
                                  : nanovdb::cuda::DeviceBuffer();
    if (lowerCount) cudaCheck(cudaMemsetAsync(mLowerInvertMask.deviceData(), 0, lowerBytes, mStream));
    if (upperCount) cudaCheck(cudaMemsetAsync(mUpperInvertMask.deviceData(), 0, upperBytes, mStream));
    if (leafCount == 0 || lowerCount == 0) return;  // nothing to seed from

    // Temporary per-tile evidence accumulators (freed at scope exit).
    auto lowSawIntBuf = nanovdb::cuda::DeviceBuffer::create(lowerBytes, nullptr, false);
    auto lowSawExtBuf = nanovdb::cuda::DeviceBuffer::create(lowerBytes, nullptr, false);
    auto upSawIntBuf  = nanovdb::cuda::DeviceBuffer::create(upperBytes, nullptr, false);
    auto upSawExtBuf  = nanovdb::cuda::DeviceBuffer::create(upperBytes, nullptr, false);
    auto* d_lowSawInt = static_cast<nanovdb::Mask<4>*>(lowSawIntBuf.deviceData());
    auto* d_lowSawExt = static_cast<nanovdb::Mask<4>*>(lowSawExtBuf.deviceData());
    auto* d_upSawInt  = static_cast<nanovdb::Mask<5>*>(upSawIntBuf.deviceData());
    auto* d_upSawExt  = static_cast<nanovdb::Mask<5>*>(upSawExtBuf.deviceData());
    cudaCheck(cudaMemsetAsync(d_lowSawInt, 0, lowerBytes, mStream));
    cudaCheck(cudaMemsetAsync(d_lowSawExt, 0, lowerBytes, mStream));
    cudaCheck(cudaMemsetAsync(d_upSawInt,  0, upperBytes, mStream));
    cudaCheck(cudaMemsetAsync(d_upSawExt,  0, upperBytes, mStream));

    // (1) Leaf faces seed the childless lower/upper tiles across them.
    using LeafSeedOp = LeafFaceSeedFunctor<BuildT>;
    if (mVerbose==1) mTimer.start("Coarse invert: seed from leaf faces");
    util::cuda::operatorKernel<LeafSeedOp><<<leafCount, LeafSeedOp::MaxThreadsPerBlock, 0, mStream>>>(
        d_grid, sign, deviceLeafInvertMask(),
        d_lowSawInt, d_lowSawExt, d_upSawInt, d_upSawExt);
    cudaCheckError();
    if (mVerbose==1) mTimer.stop();

    // (2) Finalize (sawInt && !sawExt gate) + flood the lower level.
    using LowerFloodOp = CoarseInvertFloodFunctor<BuildT, 1>;
    if (mVerbose==1) mTimer.start("Coarse invert: flood lower nodes");
    util::cuda::operatorKernel<LowerFloodOp><<<lowerCount, LowerFloodOp::MaxThreadsPerBlock, 0, mStream>>>(
        d_grid, d_lowSawInt, d_lowSawExt, deviceLowerInvertMask());
    cudaCheckError();
    if (mVerbose==1) mTimer.stop();

    // (3) Lower faces seed the childless upper tiles across them.
    using LowerSeedOp = LowerFaceSeedFunctor<BuildT>;
    if (mVerbose==1) mTimer.start("Coarse invert: seed from lower faces");
    util::cuda::operatorKernel<LowerSeedOp><<<lowerCount, LowerSeedOp::MaxThreadsPerBlock, 0, mStream>>>(
        d_grid, deviceLowerInvertMask(), d_upSawInt, d_upSawExt);
    cudaCheckError();
    if (mVerbose==1) mTimer.stop();

    // (4) Finalize + flood the upper level.
    if (upperCount) {
        using UpperFloodOp = CoarseInvertFloodFunctor<BuildT, 2>;
        if (mVerbose==1) mTimer.start("Coarse invert: flood upper nodes");
        util::cuda::operatorKernel<UpperFloodOp><<<upperCount, UpperFloodOp::MaxThreadsPerBlock, 0, mStream>>>(
            d_grid, d_upSawInt, d_upSawExt, deviceUpperInvertMask());
        cudaCheckError();
        if (mVerbose==1) mTimer.stop();
    }

    // The evidence accumulators go out of scope here; sync so their frees can't outrun the kernels.
    cudaCheck(cudaStreamSynchronize(mStream));
}// SurfaceSigner<BuildT>::fillCoarseInvertMasks

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
void SurfaceSigner<BuildT>::fillRootInteriorMask(const GridT* d_grid, const int8_t* d_sign)
{
    const int8_t* sign = d_sign ? d_sign : deviceSignedVoxelSign();  // external signs override the member
    const auto     treeData   = util::cuda::DeviceGridTraits<BuildT>::getTreeData(d_grid);
    const uint32_t leafCount  = treeData.mNodeCount[0];
    const uint32_t lowerCount = treeData.mNodeCount[1];
    const uint32_t upperCount = treeData.mNodeCount[2];
    if (leafCount == 0) { mRootInterior = nanovdb::cuda::DeviceBuffer(); mRootDims = nanovdb::Coord(0); return; }

    // C1: provisional cell range = the grid bbox at root-tile (4096^3) granularity. >>12 is floor
    // division by 4096 for negative coords too (arithmetic shift).
    const auto bbox = util::cuda::DeviceGridTraits<BuildT>::getIndexBBox(d_grid, treeData);
    mRootTileMin = nanovdb::Coord(bbox.min()[0] >> 12, bbox.min()[1] >> 12, bbox.min()[2] >> 12);
    const nanovdb::Coord tileMax(bbox.max()[0] >> 12, bbox.max()[1] >> 12, bbox.max()[2] >> 12);
    mRootDims = nanovdb::Coord(tileMax[0] - mRootTileMin[0] + 1,
                               tileMax[1] - mRootTileMin[1] + 1,
                               tileMax[2] - mRootTileMin[2] + 1);
    const std::size_t total = std::size_t(mRootDims[0]) * mRootDims[1] * mRootDims[2];

    mRootInterior    = nanovdb::cuda::DeviceBuffer::create(total, nullptr, false);
    auto wallBuf     = nanovdb::cuda::DeviceBuffer::create(total, nullptr, false);
    auto sawIntBuf   = nanovdb::cuda::DeviceBuffer::create(total, nullptr, false);
    auto sawExtBuf   = nanovdb::cuda::DeviceBuffer::create(total, nullptr, false);
    auto* d_wall     = static_cast<uint8_t*>(wallBuf.deviceData());
    auto* d_sawInt   = static_cast<uint8_t*>(sawIntBuf.deviceData());
    auto* d_sawExt   = static_cast<uint8_t*>(sawExtBuf.deviceData());
    cudaCheck(cudaMemsetAsync(mRootInterior.deviceData(), 0, total, mStream));
    cudaCheck(cudaMemsetAsync(d_wall,   0, total, mStream));
    cudaCheck(cudaMemsetAsync(d_sawInt, 0, total, mStream));
    cudaCheck(cudaMemsetAsync(d_sawExt, 0, total, mStream));

    if (mVerbose==1) mTimer.start("Root interior: mark walls + seed + flood");

    // C1: walls = pre-existing root entries.
    constexpr unsigned int kWallThreads = 128;
    util::cuda::lambdaKernel<<<unsigned((total + kWallThreads - 1) / kWallThreads), kWallThreads, 0, mStream>>>(
        total, RootWallMarkFunctor<BuildT>{ d_grid, d_wall, mRootTileMin, mRootDims });
    cudaCheckError();

    // C2: interior/exterior evidence from ALL THREE levels into abutting absent root cells.
    using Seed0 = RootFaceSeedFunctor<BuildT, 0>;
    using Seed1 = RootFaceSeedFunctor<BuildT, 1>;
    using Seed2 = RootFaceSeedFunctor<BuildT, 2>;
    util::cuda::operatorKernel<Seed0><<<leafCount, Seed0::MaxThreadsPerBlock, 0, mStream>>>(
        d_grid, sign, deviceLeafInvertMask(), deviceLowerInvertMask(),
        deviceUpperInvertMask(), d_sawInt, d_sawExt, mRootTileMin, mRootDims);
    cudaCheckError();
    if (lowerCount) {
        util::cuda::operatorKernel<Seed1><<<lowerCount, Seed1::MaxThreadsPerBlock, 0, mStream>>>(
            d_grid, sign, deviceLeafInvertMask(), deviceLowerInvertMask(),
            deviceUpperInvertMask(), d_sawInt, d_sawExt, mRootTileMin, mRootDims);
        cudaCheckError();
    }
    if (upperCount) {
        util::cuda::operatorKernel<Seed2><<<upperCount, Seed2::MaxThreadsPerBlock, 0, mStream>>>(
            d_grid, sign, deviceLeafInvertMask(), deviceLowerInvertMask(),
            deviceUpperInvertMask(), d_sawInt, d_sawExt, mRootTileMin, mRootDims);
        cudaCheckError();
    }

    // C3: gate + multi-seed flood (single block; the array is a handful of cells).
    using FloodOp = RootInteriorFloodFunctor;
    util::cuda::operatorKernel<FloodOp><<<1, FloodOp::MaxThreadsPerBlock, 0, mStream>>>(
        d_wall, d_sawInt, d_sawExt, deviceRootInterior(), mRootDims);
    cudaCheckError();
    if (mVerbose==1) mTimer.stop();

    // Temporaries go out of scope here; sync so their frees can't outrun the kernels.
    cudaCheck(cudaStreamSynchronize(mStream));
}// SurfaceSigner<BuildT>::fillRootInteriorMask

} // namespace sdf_detail

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// MeshToSDF - partition the rasterized band into closed surfaces, sign each one alone, compose.
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
void MeshToSDF<BuildT>::
    build()
{
    // Time each phase into mPhaseMs (see phaseMs()). The stream sync before every mark makes the
    // marks true phase boundaries rather than kernel-launch boundaries, so the five numbers sum to
    // build()'s wall time; the syncs themselves cost five per build and are not measurable here.
    int  phase = 0;
    auto mark  = [&, prev = std::chrono::steady_clock::time_point{}]() mutable {
        cudaCheck(cudaStreamSynchronize(mStream));
        const auto now = std::chrono::steady_clock::now();
        if (phase) mPhaseMs[phase - 1] = std::chrono::duration<float, std::milli>(now - prev).count();
        prev = now;
        ++phase;
    };

    // The isovalue only ever moves the surface outward, and a negative one would ask for a level set
    // the unsigned distance does not have.
    if (mIsoValue < 0.f)
        throw std::runtime_error("MeshToSDF: setIsoValue() must be >= 0");

    mark();
    this->rasterize();          // mesh -> narrow band, with the UDF and nearest-triangle sidecars
    mark();
    this->partition();          // components of the UN-pruned band = one per closed surface
    mark();
    for (uint32_t i = 0; i < uint32_t(mSurfaces.size()); ++i)
        this->signSurface(i);   // carve surface i out, prune its barrier shell, label, and sign it alone
    mark();
    this->composeByInclusion();  // nesting parity per surface, then merge the signs onto the band
    mark();
    this->finalizeMagnitudes();  // signs are settled: report |udf - isoValue|, floor the interior
    this->fillOnOriginal();      // extend those signs off the band as invert masks
    mark();
}// MeshToSDF<BuildT>::build

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
void MeshToSDF<BuildT>::rasterize()
{
    const float voxelSize = float(mMap.getVoxelSize()[0]);

    // The signed surface stands mIsoValue out from the mesh, so the band has to reach that far again
    // to still hold mBandWidth voxels beyond it. Rasterizing 3 + isoValue/voxelSize wide and signing
    // the isosurface lands mBandWidth voxels of band outside it -- the width the caller asked for,
    // measured where they meant it.
    //TODO: in the future, we may have to prune inside voxels to keep bandwidth 3 after iso value setting.
    const float extra = (voxelSize > 0.f) ? mIsoValue / voxelSize : 0.f;

    MeshToGrid<BuildT> converter(mPoints, mPointCount, mTriangles, mTriangleCount, mMap, mStream);
    converter.setVerbose(mVerbose);
    converter.setNarrowBandWidth(mBandWidth + extra);
    std::tie(mGridHandle, mUDF, mIndex) = converter.getHandleAndUDFAndIndex();
}// MeshToSDF<BuildT>::rasterize

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

/// Connected components on the UN-PRUNED band. The barrier shell is what glues a surface's inner and
/// outer sides together, so leaving it in place makes each closed surface exactly one component — and
/// the component count the surface count.
template <typename BuildT>
void MeshToSDF<BuildT>::partition()
{
    mSurfaceCC = std::make_unique<ConnectedComponents<BuildT>>(this->deviceGrid(), mStream);
    mSurfaceCC->setVerbose(mVerbose);
    mSurfaceLabels = mSurfaceCC->getVoxelLabelsAndCount();
    cudaCheck(cudaStreamSynchronize(mStream));
    mSurfaces.resize(std::size_t(mSurfaceLabels.second));
}// MeshToSDF<BuildT>::partition

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

/// Carve one closed surface out of the rasterized band and run the whole signing sequence on it, so
/// the result is the sign field that surface would have if it were the only object in the scene.
///
/// The global exterior seed is legitimate here precisely because the grid holds one closed surface:
/// its minimum-x active voxel is necessarily outside it. A mesh with a single closed surface needs no
/// carving at all — the rasterized band already IS that surface's band — so subGrid/subUdf/subIndex
/// stay empty and the rasterized arrays are used directly. That is a memory decision: carving
/// duplicates the grid plus the UDF, index and sign sidecars, ~9 bytes per voxel.
template <typename BuildT>
void MeshToSDF<BuildT>::signSurface(uint32_t surface) //todo: change name from Surface to Component
{
    using Traits = util::cuda::DeviceGridTraits<BuildT>;

    SurfaceField&  sf         = mSurfaces[surface];
    const auto*    d_orig     = this->deviceGrid();
    const uint32_t origLeaves = Traits::getTreeData(d_orig).mNodeCount[0]; //todo: rename more clearer (origLeafCounts or something)
    const float    voxelSize  = float(mMap.getVoxelSize()[0]);

    // (a) Carve this surface out, and re-index onto the carved grid the two sidecars the stages below
    //     read. Carving renumbers the value slots, so the transfer goes through the injection functor
    //     (leaf-origin pairing + popcount rank) rather than a memcpy.
    // TODO: specify that most case: surface count is zero. single closed surface is the most common case input
    // TODO: could be a function? such as carveComponent(i)
    if (mSurfaces.size() > 1) {
        util::cuda::Timer timer(mStream);
        if (mVerbose==1) timer.start("Carve closed surface out of the band");
        auto  maskBuf    = Buffer::create(std::size_t(origLeaves) * sizeof(Mask<3>), nullptr, false);
        auto* d_partMask = static_cast<Mask<3>*>(maskBuf.deviceData());
        using MaskOp = sdf_detail::SurfaceMaskFunctor<BuildT>;
        util::cuda::operatorKernel<MaskOp><<<origLeaves, MaskOp::MaxThreadsPerBlock, 0, mStream>>>(
            d_orig, mSurfaceLabels.first, surface, d_partMask);
        cudaCheckError();

        PruneGrid<BuildT> pruner(d_orig, d_partMask, mStream);
        sf.subGrid = pruner.getHandle();

        const auto*    d_sub     = sf.subGrid.template deviceGrid<BuildT>();
        const uint64_t subActive = Traits::getActiveVoxelCount(d_sub);
        sf.subUdf   = Buffer::create((subActive + 1) * sizeof(float), nullptr, false);
        sf.subIndex = Buffer::create((subActive + 1) * sizeof(uint32_t), nullptr, false);
        cudaCheck(cudaMemsetAsync(sf.subUdf.deviceData(),   0,    (subActive + 1) * sizeof(float), mStream));
        cudaCheck(cudaMemsetAsync(sf.subIndex.deviceData(), 0xFF, (subActive + 1) * sizeof(uint32_t), mStream));

        using InjectUdf   = util::cuda::InjectGridDataFunctor<BuildT, float>;
        using InjectIndex = util::cuda::InjectGridDataFunctor<BuildT, uint32_t>;
        util::cuda::operatorKernel<InjectUdf><<<origLeaves, InjectUdf::MaxThreadsPerBlock, 0, mStream>>>(
            d_orig, d_sub, this->deviceUDF(), static_cast<float*>(sf.subUdf.deviceData()));
        cudaCheckError();
        util::cuda::operatorKernel<InjectIndex><<<origLeaves, InjectIndex::MaxThreadsPerBlock, 0, mStream>>>(
            d_orig, d_sub, this->deviceIndex(), static_cast<uint32_t*>(sf.subIndex.deviceData()));
        cudaCheckError();
        cudaCheck(cudaStreamSynchronize(mStream));
        if (mVerbose==1) timer.stop();
    }

    const auto* d_grid = this->surfaceGrid(surface);
    // TODO: Seems signer does many things. it not only signs the narrow band, it fills the inside. In that sense, we may have to reconsider the class name
    // TODO: or we can reconsider creating another class which "fills inside"
    sf.signer = std::make_unique<Signer>(mStream);
    sf.signer->setVerbose(mVerbose);

    // (b) Drop the barrier shell, then label what remains. This surface's inner and outer sides fall
    //     apart into separate components — the split the signing rule relies on.
    sf.derived = sf.signer->computeDerivedTopology(d_grid, this->surfaceUdf(surface), voxelSize, mIsoValue);
    const auto* d_derived = sf.derived.template deviceGrid<BuildT>();

    sf.cc = std::make_unique<ConnectedComponents<BuildT>>(d_derived, mStream);
    sf.cc->setVerbose(mVerbose);
    sf.ccLabels = sf.cc->getVoxelLabelsAndCount();
    cudaCheck(cudaStreamSynchronize(mStream));

    // (c) Sign the non-barrier voxels, carry the signs onto the un-pruned surface grid, then sign the
    //     barrier shell that (b) set aside.
    sf.signer->signNonBarrier(d_derived, sf.ccLabels.first);
    sf.signer->injectSignsToOriginal(d_grid, d_derived);
    switch (mBarrierSigning) {
    case BarrierSigning::Ball:
        sf.signer->signBarrierByBalls(d_grid, this->surfaceUdf(surface), voxelSize,
                                      32, mBallStencilRadius, mIsoValue);
        break;
    case BarrierSigning::Heuristic:
        sf.signer->signBarrier(d_grid, this->surfaceIndex(surface), mPoints, mTriangles, mMap,
                               mIsoValue, voxelSize);
        break;
    case BarrierSigning::Interior:
    default:
        sf.signer->signBarrierAsInterior(d_grid);
        break;
    }

    // (d) Extend the sign off the band. That is what lets the inclusion test query this field at
    //     another surface's band — and, for a single-surface mesh, it is already the finished result.
    sf.signer->fillLeafInvertMask(d_grid);
    sf.signer->fillCoarseInvertMasks(d_grid);
    sf.signer->fillRootInteriorMask(d_grid);
    cudaCheck(cudaStreamSynchronize(mStream));
}// MeshToSDF<BuildT>::signSurface

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

/// Recover the nesting depth of every closed surface from the per-surface fields, then merge those
/// fields into one sign array on the rasterized band, negating the odd-depth surfaces on the way in.
/// Requires every field to be complete through its invert-mask fill.
template <typename BuildT>
void MeshToSDF<BuildT>::composeByInclusion() //todo: change this function name since we may add another parity policty
{
    using Traits = util::cuda::DeviceGridTraits<BuildT>;

    const uint32_t numSurfaces = uint32_t(mSurfaces.size());
    if (numSurfaces == 0) return;

    // Nothing encloses a lone surface, and an uncarved one already carries its signs on the rasterized
    // band — so there is no depth to recover and nothing to gather. Aliasing here is what keeps the
    // single-surface case free of the extra full-length sign array a merge would allocate.
    if (numSurfaces == 1 && !mSurfaces[0].subGrid.bufferSize()) {
        // Depth 0 under either rule: the surface's own field is already the answer.
        mNestingDepth.assign(1, 0u);
        mSign = mSurfaces[0].signer->deviceSignedVoxelSign();
        return;
    }

    const auto*    d_orig     = this->deviceGrid();
    const uint32_t origLeaves = Traits::getTreeData(d_orig).mNodeCount[0];
    const uint64_t origActive = Traits::getActiveVoxelCount(d_orig);

    util::cuda::Timer timer(mStream);
    if (mVerbose==1) timer.start("Inclusion: nesting depth + merge onto the rasterized band");

    // (1) One representative voxel per surface. Any voxel of a surface's band serves: the band hugs
    //     its own surface, so it lies wholly inside, or wholly outside, every other surface.
    auto  repBuf   = Buffer::create(numSurfaces * sizeof(unsigned long long), nullptr, false);
    auto* d_repKey = static_cast<unsigned long long*>(repBuf.deviceData());
    cudaCheck(cudaMemsetAsync(d_repKey, 0xFF, numSurfaces * sizeof(unsigned long long), mStream));
    {
        using RepOp = sdf_detail::SurfaceRepFunctor<BuildT>;
        util::cuda::operatorKernel<RepOp><<<origLeaves, RepOp::MaxThreadsPerBlock, 0, mStream>>>(
            d_orig, mSurfaceLabels.first, numSurfaces, d_repKey);
        cudaCheckError();
    }

    // (2) Ask each surface's own field about every surface's representative. Each was completed
    //     through the invert-mask fill, so it answers off its band too — including at the other bands.
    // TODO: The variable name incBuf is a bit unclear, let's consider it from inc to inclusion for clarity.
    auto  incBuf = Buffer::create(std::size_t(numSurfaces) * numSurfaces * sizeof(int8_t), nullptr, false);
    auto* d_inc  = static_cast<int8_t*>(incBuf.deviceData());  // d_inc[i*numSurfaces+j] = field i's sign at surface j
    for (uint32_t i = 0; i < numSurfaces; ++i) {
        auto&       phi   = *mSurfaces[i].signer;
        const auto* d_sub = this->surfaceGrid(i);
        using ProbeOp = sdf_detail::InclusionProbeFunctor<BuildT>;
        util::cuda::lambdaKernel<<<1, numSurfaces, 0, mStream>>>(
            numSurfaces, ProbeOp{}, d_sub, d_repKey, phi.deviceSignedVoxelSign(),
            phi.deviceLeafInvertMask(), phi.deviceLowerInvertMask(), phi.deviceUpperInvertMask(),
            phi.deviceRootInterior(), phi.rootTileMin(), phi.rootTileDims(), d_inc + std::size_t(i) * numSurfaces);
        cudaCheckError();
    }

    // (3) Nesting depth = how many other surfaces report this one as inside them. Counting a column is
    //     enough: enclosure is transitive between non-intersecting surfaces, so a surface nested d deep
    //     is reported inside by exactly d others — the inclusion forest never has to be built.
    std::vector<int8_t> inc(std::size_t(numSurfaces) * numSurfaces);
    cudaCheck(cudaMemcpyAsync(inc.data(), d_inc, inc.size() * sizeof(int8_t), cudaMemcpyDeviceToHost, mStream));
    cudaCheck(cudaStreamSynchronize(mStream));

    mNestingDepth.assign(numSurfaces, 0u);
    for (uint32_t j = 0; j < numSurfaces; ++j)
        for (uint32_t i = 0; i < numSurfaces; ++i)
            if (i != j && inc[std::size_t(i) * numSurfaces + j] < 0) ++mNestingDepth[j];  // field i says j is inside

    // (4) Merge. Gather every surface's signs back onto the rasterized band — the surfaces partition
    //     its active voxels, so the per-surface injections write disjoint slots and together cover all —
    //     then resolve each voxel against the nesting rule in place.
    mComposedSign = Buffer::create((origActive + 1) * sizeof(int8_t), nullptr, false);
    mSign = static_cast<int8_t*>(mComposedSign.deviceData());
    cudaCheck(cudaMemsetAsync(mSign, 1, (origActive + 1) * sizeof(int8_t), mStream));  // slot 0 = background +1
    using InjectOp = util::cuda::InjectGridDataFunctor<BuildT, int8_t>;
    for (uint32_t i = 0; i < numSurfaces; ++i) {
        const auto*    d_sub     = this->surfaceGrid(i);
        const uint32_t subLeaves = Traits::getTreeData(d_sub).mNodeCount[0];
        util::cuda::operatorKernel<InjectOp><<<subLeaves, InjectOp::MaxThreadsPerBlock, 0, mStream>>>(
            d_sub, d_orig, mSurfaces[i].signer->deviceSignedVoxelSign(), mSign);
        cudaCheckError();
    }
    auto  depthBuf = Buffer::create(numSurfaces * sizeof(uint32_t), nullptr, false);
    auto* d_depth  = static_cast<uint32_t*>(depthBuf.deviceData());
    cudaCheck(cudaMemcpyAsync(d_depth, mNestingDepth.data(), numSurfaces * sizeof(uint32_t),
                              cudaMemcpyHostToDevice, mStream));
    util::cuda::lambdaKernel<<<(unsigned int)((origActive + 256) / 256), 256, 0, mStream>>>(
        origActive + 1, sdf_detail::ResolveNestingFunctor{}, mSurfaceLabels.first, d_depth,
        numSurfaces, mNestingRule == NestingRule::EvenOdd, mSign);
    cudaCheckError();
    cudaCheck(cudaStreamSynchronize(mStream));
    if (mVerbose==1) timer.stop();
}// MeshToSDF<BuildT>::composeByInclusion

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

/// @details Step 4b. Rewrites the UDF sidecar from "distance to the mesh" into the magnitude this
///          pipeline reports, now that every sign is settled: udf -> |udf - mIsoValue|, then interior
///          magnitudes floored at half a voxel diagonal. See sdf_detail::IsoMagnitudeFunctor for why
///          each step is what it is.
///
///          The ordering is the whole point. Folding about the isovalue throws away which side of the
///          MESH a voxel is on, and every stage before this one needed exactly that -- the barrier
///          test, the components, the min-vertex seed. Running here costs one pass over the sidecar
///          and leaves the signs untouched, so nothing downstream has to be redone.
///
///          Runs before fillOnOriginal() only for tidiness; the fill reads signs, not magnitudes.
template <typename BuildT>
void MeshToSDF<BuildT>::finalizeMagnitudes()
{
    if (mIsoValue == 0.f || mSurfaces.empty() || mSign == nullptr) return;

    const float voxelSize = float(mMap.getVoxelSize()[0]);

    // sqrt(3)/2 voxels: the same half-diagonal the barrier test uses, so the floor lands exactly at
    // the edge of the shell the oracle was responsible for.
    const float interiorFloor = 0.8660254f * voxelSize;

    using Op = sdf_detail::IsoMagnitudeFunctor<BuildT>;
    const uint32_t leaves = util::cuda::DeviceGridTraits<BuildT>::getTreeData(this->deviceGrid()).mNodeCount[0];
    if (leaves)
        util::cuda::operatorKernel<Op><<<leaves, Op::MaxThreadsPerBlock, 0, mStream>>>(
            this->deviceGrid(), static_cast<float*>(mUDF.deviceData()), mSign, mIsoValue, interiorFloor);
    cudaCheckError();

    // Slot 0 is the background sentinel, not a voxel. The kernel above walks active voxels and never
    // reaches it, but it is restated here so the value is the exterior background whatever the
    // rasterizer left, rather than something an earlier stage happened to write.
    const float bg = mBandWidth * voxelSize;
    cudaCheck(cudaMemcpyAsync(mUDF.deviceData(), &bg, sizeof(float), cudaMemcpyHostToDevice, mStream));
    cudaCheck(cudaStreamSynchronize(mStream));
}// MeshToSDF<BuildT>::finalizeMagnitudes

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

/// Extend the composed sign off the band, over the whole rasterized grid. When the composition aliased
/// a lone uncarved surface, the fill it already ran covered this very grid with these very signs, so
/// it is adopted as-is.
template <typename BuildT>
void MeshToSDF<BuildT>::fillOnOriginal()
{
    if (mSurfaces.empty()) return;

    if (!mComposedSign.size()) {                 // composition aliased surface 0 -> its fill is the answer
        mFinalSigner = mSurfaces[0].signer.get();
        return;
    }

    const auto* d_orig = this->deviceGrid();
    mOrigSigner = std::make_unique<Signer>(mStream);
    mOrigSigner->setVerbose(mVerbose);
    mOrigSigner->fillLeafInvertMask(d_orig, mSign);
    mOrigSigner->fillCoarseInvertMasks(d_orig, mSign);
    mOrigSigner->fillRootInteriorMask(d_orig, mSign);
    cudaCheck(cudaStreamSynchronize(mStream));
    mFinalSigner = mOrigSigner.get();
}// MeshToSDF<BuildT>::fillOnOriginal

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
const GridHandle<nanovdb::cuda::DeviceBuffer>&
MeshToSDF<BuildT>::surfaceGridHandle(uint32_t i) const
{ return mSurfaces[i].subGrid.bufferSize() ? mSurfaces[i].subGrid : mGridHandle; }

template <typename BuildT>
const NanoGrid<BuildT>* MeshToSDF<BuildT>::surfaceGrid(uint32_t i) const
{ return this->surfaceGridHandle(i).template deviceGrid<BuildT>(); }

template <typename BuildT>
const float* MeshToSDF<BuildT>::surfaceUdf(uint32_t i) const
{ return static_cast<const float*>(mSurfaces[i].subUdf.size() ? mSurfaces[i].subUdf.deviceData()
                                                              : mUDF.deviceData()); }

template <typename BuildT>
const uint32_t* MeshToSDF<BuildT>::surfaceIndex(uint32_t i) const
{ return static_cast<const uint32_t*>(mSurfaces[i].subIndex.size() ? mSurfaces[i].subIndex.deviceData()
                                                                   : mIndex.deviceData()); }

template <typename BuildT>
const Mask<3>* MeshToSDF<BuildT>::deviceLeafInvertMask() const
{ return mFinalSigner->deviceLeafInvertMask(); }

template <typename BuildT>
const Mask<4>* MeshToSDF<BuildT>::deviceLowerInvertMask() const
{ return mFinalSigner->deviceLowerInvertMask(); }

template <typename BuildT>
const Mask<5>* MeshToSDF<BuildT>::deviceUpperInvertMask() const
{ return mFinalSigner->deviceUpperInvertMask(); }

template <typename BuildT>
const uint8_t* MeshToSDF<BuildT>::deviceRootInterior() const
{ return mFinalSigner->deviceRootInterior(); }

template <typename BuildT>
Coord MeshToSDF<BuildT>::rootTileMin() const { return mFinalSigner->rootTileMin(); }

template <typename BuildT>
Coord MeshToSDF<BuildT>::rootTileDims() const { return mFinalSigner->rootTileDims(); }

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

} // namespace tools::cuda

} // namespace nanovdb

#endif // NVIDIA_TOOLS_CUDA_MESHTOSDF_CUH_HAS_BEEN_INCLUDED
