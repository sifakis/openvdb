// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file MeanCurvature.cuh

    \brief One explicit Euler step of motion by mean curvature of a narrow-band level set held as a NanoVDB index grid
           plus a value sidecar, the form tools::cuda::LevelSetTracker maintains:

               phi <- phi + dt (H - lambda) |grad phi|,     dt = dtFactor dx^2

           H is the mean curvature (k1 + k2) / 2 of the level set through the voxel, so a sphere of radius R shrinks
           at 1 / R. With lambda = 0 this is the update of openvdb::tools::LevelSetFilter::meanCurvature(), whose
           time step is dtFactor = 1 / 3. With lambda = the average of H over the surface, the enclosed volume is
           kept: the surface moves by -(H - lambda) along its outward normal, and that integrates to zero over it.
           The caller hands the result to a LevelSetTracker and calls track(), as OpenVDB's filter does.

    \note The band must be 5 voxels wide on either side of the surface. H reads 19 taps, and a tap outside the band
          only has the background; with a band of 3 voxels the voxels beyond 1.6 voxels from the surface move
          wrongly, and the renormalization of track() carries that error to the surface.

    \warning The header file contains cuda device code so be sure
             to only include it in .cu files (or other .cuh files)
*/

#ifndef NANOVDB_EXAMPLES_MEANCURVATURE_CUH_HAS_BEEN_INCLUDED
#define NANOVDB_EXAMPLES_MEANCURVATURE_CUH_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h>
#include <nanovdb/cuda/Buffer.h>
#include <nanovdb/cuda/DeviceResource.h>
#include <nanovdb/tools/cuda/LevelSetTracker.cuh>
#include <nanovdb/util/cuda/DeviceGridTraits.cuh>
#include <nanovdb/util/cuda/Util.h> // for operatorKernel and lambdaKernel

#include <cuda/std/functional> // cuda::std::plus
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>

#include <cmath>   // cos
#include <cstddef> // size_t
#include <cstdint> // uint64_t

namespace curvature {

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace detail {

/// @brief One thread per voxel of a leaf: writes motion = H |grad phi| and gradient = |grad phi| of every active
///        voxel, both of phi / dx, so that they are in voxels and the test for a vanishing gradient means the same
///        for every voxel size. MoveVoxelFunctor turns the two into the new phi.
/// @details The 19 taps and the differences are those of math::CurvatureStencil::meanCurvatureNormGrad(). A tap
///          outside the band takes the background or its negation, as told by the invert masks of PropagateSign.
template<typename BuildT, typename ValueT>
struct CurvatureMotionFunctor
{
    static constexpr int MaxThreadsPerBlock = nanovdb::NanoLeaf<BuildT>::SIZE;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(const nanovdb::NanoGrid<BuildT>* grid,
                               const ValueT* phi,
                               const nanovdb::Mask<3>* leafMasks,
                               const nanovdb::Mask<4>* lowerMasks,
                               const nanovdb::Mask<5>* upperMasks,
                               const nanovdb::Coord* rootOrigins,
                               uint64_t rootOriginCount,
                               ValueT background,
                               ValueT invDx,
                               ValueT* motion,
                               ValueT* gradient) const
    {
        using nanovdb::tools::cuda::levelset::detail::isInverted;
        const auto&    leaf = grid->tree().template getFirstNode<0>()[blockIdx.x];
        const uint32_t n    = threadIdx.x;
        if (!leaf.isActive(n)) return;
        const nanovdb::Coord ijk = leaf.offsetToGlobalCoord(n);

        // The taps in the order of math::CurvPt: the center, the 6 face neighbors, the 12 edge neighbors.
        const int offsets[19][3] = {{0, 0, 0},
                                    {-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1},
                                    {-1, -1, 0}, {1, -1, 0}, {-1, 1, 0}, {1, 1, 0},
                                    {-1, 0, -1}, {1, 0, -1}, {-1, 0, 1}, {1, 0, 1},
                                    {0, -1, -1}, {0, 1, -1}, {0, -1, 1}, {0, 1, 1}};
        const auto acc = grid->getAccessor();
        ValueT v[19];
        for (int t = 0; t < 19; ++t) {
            const nanovdb::Coord tap = ijk.offsetBy(offsets[t][0], offsets[t][1], offsets[t][2]);
            const uint64_t index = acc.getValue(tap); // 0: the tap lies outside the band
            const ValueT value = index ? phi[index]
                                       : isInverted(*grid, tap, leafMasks, lowerMasks, upperMasks, rootOrigins,
                                                    rootOriginCount) ? -background : background;
            v[t] = value * invDx;
        }

        const ValueT Dx = ValueT(0.5) * (v[2] - v[1]), Dy = ValueT(0.5) * (v[4] - v[3]), Dz = ValueT(0.5) * (v[6] - v[5]);
        const ValueT Dx2 = Dx * Dx, Dy2 = Dy * Dy, Dz2 = Dz * Dz, normGrad2 = Dx2 + Dy2 + Dz2;
        ValueT curvatureNormGrad = 0;
        if (normGrad2 > nanovdb::math::Tolerance<ValueT>::value()) {
            const ValueT Dxx = v[2] - 2 * v[0] + v[1], Dyy = v[4] - 2 * v[0] + v[3], Dzz = v[6] - 2 * v[0] + v[5];
            const ValueT Dxy = ValueT(0.25) * (v[10] - v[8] + v[7] - v[9]);
            const ValueT Dxz = ValueT(0.25) * (v[14] - v[12] + v[11] - v[13]);
            const ValueT Dyz = ValueT(0.25) * (v[18] - v[16] + v[15] - v[17]);
            const ValueT alpha = Dx2 * (Dyy + Dzz) + Dy2 * (Dxx + Dzz) + Dz2 * (Dxx + Dyy) -
                                 2 * (Dx * (Dy * Dxy + Dz * Dxz) + Dy * Dz * Dyz);
            curvatureNormGrad = alpha / (2 * normGrad2);
        }
        const uint64_t index = leaf.getValue(n);
        motion[index]   = curvatureNormGrad;
        gradient[index] = nanovdb::math::Sqrt(normGrad2);
    }
}; // CurvatureMotionFunctor

} // namespace detail

/// @brief The delta function of the surface, smeared over 2 voxels on either side; @a phiInVoxels is phi / dx.
///        Summed over the voxels along a grid line it gives exactly 1 wherever the surface crosses the line, so a
///        sum of delta |grad phi| f over the band approximates the integral of f over the surface, in voxels^2.
__hostdev__ inline double surfaceDelta(double phiInVoxels)
{
    const double halfWidth = 2.0, pi = 3.14159265358979323846;
    return phiInVoxels > -halfWidth && phiInVoxels < halfWidth ? (1.0 + cos(pi * phiInVoxels / halfWidth)) / (2.0 * halfWidth)
                                                               : 0.0;
}

namespace detail {

/// @brief Sums over the band that give the integral of H over the surface (motion) and its area (gradient).
struct SurfaceSums
{
    double motion, gradient;
    __hostdev__ SurfaceSums operator+(const SurfaceSums& other) const { return {motion + other.motion, gradient + other.gradient}; }
};

/// @brief The contribution of the voxel with sidecar index @a i to SurfaceSums.
template<typename ValueT>
struct SurfaceSumsFunctor
{
    const ValueT *phi, *motion, *gradient;
    ValueT        invDx;
    __device__ SurfaceSums operator()(uint64_t i) const
    {
        const double delta = surfaceDelta(double(phi[i]) * invDx);
        return {delta * motion[i], delta * gradient[i]};
    }
};

/// @brief result = phi + dt / dx (motion - lambda gradient) of the voxel with sidecar index @a tid + 1; slot 0 of
///        the sidecar is the background.
template<typename ValueT>
struct MoveVoxelFunctor
{
    __device__ void operator()(size_t tid, const ValueT* phi, const ValueT* motion, const ValueT* gradient,
                               ValueT dtOverDx, ValueT lambda, ValueT* result) const
    {
        const size_t i = tid + 1;
        result[i] = phi[i] + dtOverDx * (motion[i] - lambda * gradient[i]);
    }
};

} // namespace detail

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

/// @brief phi + dt (H - lambda) |grad phi| of every active voxel of @a state, with dt = dtFactor dx^2.
/// @tparam TrackerT The tools::cuda::LevelSetTracker of a ValueOnIndex grid with a float sidecar whose State
///         @a state is.
/// @param state The level set, with the invert masks of its present values (as a tracker releases it).
/// @param background Magnitude of phi outside the band, slot 0 of the sidecar.
/// @param dx Voxel size.
/// @param dtFactor Time step in units of dx^2; 1 / 3 is OpenVDB's.
/// @param preserveVolume If true, lambda is the average of H over the surface; otherwise 0.
/// @param averageCurvature If not null, receives lambda in 1 / voxel.
/// @return The new sidecar. Slot 0 is not written: the tracker's constructor sets it.
template<typename TrackerT>
typename TrackerT::template BufferT<float> meanCurvatureStep(const typename TrackerT::State& state, float background,
                                                             float dx, float dtFactor, bool preserveVolume,
                                                             cudaStream_t stream = 0, double* averageCurvature = nullptr)
{
    using BuildT    = nanovdb::ValueOnIndex;
    using ResourceT = nanovdb::cuda::DeviceResource;
    using BufferT   = typename TrackerT::template BufferT<float>;
    const nanovdb::cuda::ResourceRef<ResourceT> ref(nanovdb::cuda::default_resource<ResourceT>());

    const size_t size = state.phi.size(); // the active voxels and slot 0
    BufferT      result(stream, ref, size, nanovdb::cuda::noInit);
    if (averageCurvature) *averageCurvature = 0.0;

    const auto*    grid = state.grid.template deviceGrid<BuildT>();
    const uint32_t leafCount = nanovdb::util::cuda::DeviceGridTraits<BuildT>::getTreeData(grid, stream).mNodeCount[0];
    if (!leafCount) return result;

    BufferT motion(stream, ref, size, nanovdb::cuda::noInit), gradient(stream, ref, size, nanovdb::cuda::noInit);
    using MotionOp = detail::CurvatureMotionFunctor<BuildT, float>;
    nanovdb::util::cuda::operatorKernel<MotionOp><<<leafCount, MotionOp::MaxThreadsPerBlock, 0, stream>>>(
        grid, state.phi.data(), state.leafMasks.data(), state.lowerMasks.data(), state.upperMasks.data(),
        state.rootOrigins.data(), state.rootOrigins.size(), background, 1.f / dx, motion.data(), gradient.data());
    cudaCheckError();

    float lambda = 0.f; // in 1 / voxel
    if (preserveVolume) {
        const detail::SurfaceSums sums = thrust::transform_reduce(
            thrust::cuda::par.on(stream), thrust::counting_iterator<uint64_t>(1), thrust::counting_iterator<uint64_t>(size),
            detail::SurfaceSumsFunctor<float>{state.phi.data(), motion.data(), gradient.data(), 1.f / dx},
            detail::SurfaceSums{0.0, 0.0}, ::cuda::std::plus<detail::SurfaceSums>());
        if (sums.gradient > 0.0) lambda = float(sums.motion / sums.gradient);
        if (averageCurvature) *averageCurvature = lambda;
    }

    static constexpr unsigned int threadsPerBlock = 128;
    const size_t voxelCount = size - 1;
    nanovdb::util::cuda::lambdaKernel<<<nanovdb::util::cuda::blocksPerGrid(voxelCount, threadsPerBlock), threadsPerBlock, 0, stream>>>(
        voxelCount, detail::MoveVoxelFunctor<float>{}, state.phi.data(), motion.data(), gradient.data(), dtFactor * dx,
        lambda, result.data());
    cudaCheckError();
    return result;
}// meanCurvatureStep

} // namespace curvature

#endif // NANOVDB_EXAMPLES_MEANCURVATURE_CUH_HAS_BEEN_INCLUDED
