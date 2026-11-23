// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

// advect.cuh: semi-Lagrangian and MacCormack advection of one sidecar of a ValueOnIndex grid.
//
// Counterpart: openvdb::tools::VolumeAdvection (tools/VolumeAdvect.h) with Scheme::SEMI or Scheme::MAC and
// Scheme::CLAMP. Like it, every trace is one explicit Euler step from the sample's own position:
//   forward(x)  = in(x - d(x))                     d(x) = displacement over the step, in voxels
//   backward(x) = forward(x + d(x))
//   out(x)      = forward(x) + (in(x) - backward(x)) / 2, limited to the range of the 8 values forward(x) read
// Reused from NanoVDB: ChannelAccessor (a sidecar read at any voxel, slot 0 outside the grid), TrilinearSampler
// (math/SampleFromVoxels.h) and util::cuda::lambdaKernel.
//
// A sidecar's sample of voxel ijk sits at index-space position ijk + offset: offset 0 for a cell-centered value,
// -0.5 along axis a for component a of a staggered velocity (the convention of OpenVDB's StaggeredBoxSampler).
#pragma once
#include <nanovdb/NanoVDB.h>
#include <nanovdb/math/SampleFromVoxels.h>
#include <nanovdb/util/cuda/Util.h>
#include <nanovdb/tools/cuda/LevelSetTracker.cuh> // isInverted

namespace advect {

using BuildT = nanovdb::ValueOnIndex;
using GridT  = nanovdb::NanoGrid<BuildT>;
#ifdef ADVECT_DOUBLE
using RealT = double; // positions in double, as in OpenVDB
#else
using RealT = float;
#endif
using PosT = nanovdb::math::Vec3<RealT>;

// MacCormackRevert: where the corrected value leaves the range of the 8 values the forward step read, keep the
// forward (semi-Lagrangian) value instead of clamping (openvdb::tools::Scheme::REVERT).
enum Scheme { SemiLagrangian, MacCormack, MacCormackClamp, MacCormackRevert };

// Makes the accessor of a sidecar whose value outside the grid is its slot 0.
struct Channel
{
    using AccT = nanovdb::ChannelAccessor<float, BuildT>; // ChannelAccessor<const float> does not work with the sampler
    __hostdev__ AccT operator()(const GridT& grid, const float* values) const
    {
        return AccT(grid, const_cast<float*>(values));
    }
};

// The invert masks of PropagateSign.
struct InvertMasks
{
    const nanovdb::Mask<3>* leafMasks;
    const nanovdb::Mask<4>* lowerMasks;
    const nanovdb::Mask<5>* upperMasks;
    const nanovdb::Coord*   rootOrigins;
    uint64_t                rootOriginCount;
};

// Makes the accessor of a level set: outside the band, slot 0 (the background) or its negation, by the invert masks.
struct SignedChannel
{
    InvertMasks masks;
    struct AccT
    {
        using ValueType = float;
        using CoordType = nanovdb::Coord;
        const GridT&                         grid;
        const float*                         phi;
        InvertMasks                          masks;
        nanovdb::DefaultReadAccessor<BuildT> acc;
        __hostdev__ float getValue(const nanovdb::Coord& ijk) const
        {
            const uint64_t index = acc.getValue(ijk);
            if (index) return phi[index];
            return nanovdb::tools::cuda::levelset::detail::isInverted(grid, ijk, masks.leafMasks, masks.lowerMasks,
                       masks.upperMasks, masks.rootOrigins, masks.rootOriginCount) ? -phi[0] : phi[0];
        }
    };
    __hostdev__ AccT operator()(const GridT& grid, const float* values) const
    {
        return AccT{grid, values, masks, grid.getAccessor()};
    }
};

// Trilinear interpolation at the index-space position xyz, and the range of the 8 values it reads.
template <typename AccT>
__hostdev__ inline float interpolate(const AccT& acc, const PosT& xyz, float& lowest, float& highest)
{
    nanovdb::Coord ijk(int(floor(xyz[0])), int(floor(xyz[1])), int(floor(xyz[2])));
    const PosT     uvw(xyz[0] - RealT(ijk[0]), xyz[1] - RealT(ijk[1]), xyz[2] - RealT(ijk[2]));
    float          v[2][2][2];
    nanovdb::math::TrilinearSampler<AccT>(acc).stencil(ijk, v);
    lowest = highest = v[0][0][0];
    for (int n = 1; n < 8; ++n) {
        const float value = v[n >> 2][(n >> 1) & 1][n & 1];
        lowest = value < lowest ? value : lowest, highest = value > highest ? value : highest;
    }
    return nanovdb::math::TrilinearSampler<AccT>::sample(uvw, v);
}

// Displacement over one step, in voxels, from a staggered velocity stored as three sidecars (world units per unit
// time; slot 0 = 0): scale = time step / voxel size.
struct StaggeredVelocity
{
    const GridT* grid;
    const float* u[3];
    RealT        scale;
    __hostdev__ PosT operator()(const PosT& x) const
    {
        PosT  d;
        float lowest, highest;
        for (int a = 0; a < 3; ++a) {
            PosT p = x;
            p[a] += RealT(0.5);
            d[a] = scale * RealT(interpolate(Channel()(*grid, u[a]), p, lowest, highest));
        }
        return d;
    }
};

/// @brief out = in advected over one step.
/// @param channel Channel or SignedChannel: how @a in reads outside the grid.
/// @param offset Position of a voxel's sample relative to the voxel's coordinate.
/// @param displacement Functor PosT(const PosT& x): displacement over the step at index-space position x, in voxels.
/// @param scratch 3 x valueCount floats; unused by SemiLagrangian.
/// @param firstOrder Null, or one flag per value: a voxel whose flag is set keeps the semi-Lagrangian value.
/// @note @a out must not alias @a in. Slot 0 of @a out is set to slot 0 of @a in.
template <typename ChannelT, typename DisplacementT>
void advect(const GridT* grid, uint32_t leafCount, uint64_t valueCount, ChannelT channel, const float* in, float* out,
            PosT offset, DisplacementT displacement, Scheme scheme, float* scratch, cudaStream_t stream = 0,
            float correctionLimit = 3.0e38f, const unsigned char* firstOrder = nullptr)
{
    using nanovdb::util::cuda::blocksPerGrid;
    using nanovdb::util::cuda::lambdaKernel;
    const size_t voxels    = size_t(leafCount) * 512;
    float*       backward  = scratch;
    float*       cornerMin = scheme == SemiLagrangian ? nullptr : scratch + valueCount;
    float*       cornerMax = scheme == SemiLagrangian ? nullptr : scratch + 2 * valueCount;

    // direction +1: to[x] = from(x - d(x)); direction -1: to[x] = from(x + d(x)).
    auto trace = [=] __device__(size_t tid, const float* from, float* to, RealT direction, float* lo, float* hi) {
        const auto&    leaf = grid->tree().getFirstLeaf()[tid >> 9];
        const uint32_t n    = uint32_t(tid & 511);
        if (!leaf.isActive(n)) return;
        const nanovdb::Coord ijk = leaf.offsetToGlobalCoord(n);
        const PosT           x   = PosT(RealT(ijk[0]), RealT(ijk[1]), RealT(ijk[2])) + offset;
        const PosT           p   = x - displacement(x) * direction - offset; // in the sidecar's own index space
        const uint64_t       i   = leaf.getValue(n);
        float                lowest, highest;
        to[i] = interpolate(channel(*grid, from), p, lowest, highest);
        if (lo) lo[i] = lowest, hi[i] = highest;
    };

    cudaCheck(cudaMemcpyAsync(out, in, sizeof(float), cudaMemcpyDeviceToDevice, stream)); // slot 0, read by the trace back
    lambdaKernel<<<blocksPerGrid(voxels, 256), 256, 0, stream>>>(voxels, trace, in, out, RealT(1), cornerMin, cornerMax);
    cudaCheck(cudaGetLastError());
    if (scheme == SemiLagrangian) return;

    lambdaKernel<<<blocksPerGrid(voxels, 256), 256, 0, stream>>>(voxels, trace, out, backward, RealT(-1),
                                                                 static_cast<float*>(nullptr), static_cast<float*>(nullptr));
    cudaCheck(cudaGetLastError());
    auto correct = [=] __device__(size_t i) {
        if (i == 0 || (firstOrder && firstOrder[i])) return;
        // correctionLimit (experiment): a correction larger than this is dropped, leaving the semi-Lagrangian value.
        const float correction = 0.5f * (in[i] - backward[i]);
        float value = fabsf(correction) > correctionLimit ? out[i] : out[i] + correction;
        if (scheme == MacCormackClamp) value = fminf(fmaxf(value, cornerMin[i]), cornerMax[i]);
        if (scheme == MacCormackRevert && (value < cornerMin[i] || value > cornerMax[i])) value = out[i];
        out[i] = value;
    };
    lambdaKernel<<<blocksPerGrid(valueCount, 256), 256, 0, stream>>>(valueCount, correct);
    cudaCheck(cudaGetLastError());
}

} // namespace advect
