// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/tools/cuda/CertifyExterior.cuh

    \authors JaeHyun Lee and Efty Sifakis

    \brief Certifies the exterior of a narrow-band unsigned distance field on a NanoVDB index grid by ball overlap.

    \warning The header file contains cuda device code so be sure
             to only include it in .cu files (or other .cuh files)
*/

#ifndef NVIDIA_TOOLS_CUDA_CERTIFYEXTERIOR_CUH_HAS_BEEN_INCLUDED
#define NVIDIA_TOOLS_CUDA_CERTIFYEXTERIOR_CUH_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h>

#include <algorithm> // std::sort
#include <cmath>     // std::sqrt, std::nextafter
#include <cstdint>   // int32_t
#include <limits>    // std::numeric_limits
#include <numeric>   // std::gcd
#include <vector>    // std::vector

namespace nanovdb {

namespace tools::cuda {

/// @brief Vocabulary shared by CertifyExterior and the device code implementing it. Lengths are in voxels.
struct CertifyExteriorBase
{
    /// @brief Half-width W of the narrow band: no active voxel is farther than W from the surface.
    /// @note Fixed, so the halos below are fixed too. A wider band needs larger halos; halo 8, the largest
    ///       that stays within the 3x3x3 leaves around a leaf, has a 24^3 float tile (55 KB), more than the
    ///       48 KB of static shared memory a block can use.
    static constexpr float BandWidth = 3.f;

    /// @brief Barrier threshold D: voxels closer than D to the surface are barrier voxels, left out of the
    ///        seed. Face neighbours at least D from the surface are on the same side of it, since 2D > 1;
    ///        D = 0.55 rather than 1/2 keeps the voxels of a surface at a half-integer offset (udf = 1/2
    ///        exactly) in the barrier.
    static constexpr float BarrierThreshold = 0.55f;

    /// @brief Halos H of the default sweep, whose taps w have |w| < W + D, and of the pedantic shell pass,
    ///        whose taps have W + D <= |w| < 2W. A primitive w has a component above H only if
    ///        |w| >= sqrt((H + 1)^2 + 1), so each tap set fits in [-H, H]^3.
    static constexpr int SweepHalo = 3;
    static constexpr int ShellHalo = 5;

    /// @brief One stencil offset w: its offset in the halo tile, and its length |w|.
    struct Tap { int32_t delta; float length; };

    static_assert(BarrierThreshold > .5f && BarrierThreshold <= 1.f && BandWidth >= 3.f * BarrierThreshold,
                  "Seeding by face neighbours needs D > 1/2, and leaving out multiples of taps needs D <= 1 and W >= 3D");
    static_assert((BandWidth + BarrierThreshold) * (BandWidth + BarrierThreshold) <= (SweepHalo + 1) * (SweepHalo + 1) + 1,
                  "The default sweep taps must fit in the halo [-SweepHalo, SweepHalo]^3");
    static_assert(4.f * BandWidth * BandWidth <= (ShellHalo + 1) * (ShellHalo + 1) + 1,
                  "The shell pass taps must fit in the halo [-ShellHalo, ShellHalo]^3");
};// tools::cuda::CertifyExteriorBase

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace certification::detail {

// Setup: the tap tables.

/// @brief The halo tile of a leaf: its voxels plus Halo voxels beyond each face, in z-fastest order.
template <typename BuildT, int Halo>
struct HaloTile
{
    static constexpr int Side      = int(NanoLeaf<BuildT>::DIM) + 2 * Halo;
    static constexpr int CellCount = Side * Side * Side;
};

/// @brief The taps of the primitive offsets w in [-Halo, Halo]^3 with minLength <= |w| < maxLength, sorted
///        by length.
/// @details Multiples k*w are left out: one hop of k*w is replaced by k hops of w, through lattice points
///          that are in the band and get certified first, so the certified set at the fixpoint is the same.
///          Sorting puts the shortest taps first, so a candidate can stop scanning once |w| >= r_q + W.
template <typename BuildT, int Halo>
std::vector<CertifyExteriorBase::Tap> makeTaps(float minLength, float maxLength)
{
    using TapT = CertifyExteriorBase::Tap;
    constexpr int side = HaloTile<BuildT, Halo>::Side;

    std::vector<TapT> taps;
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
                taps.push_back({(dx * side + dy) * side + dz, length});
            }
    std::sort(taps.begin(), taps.end(), [](const TapT& a, const TapT& b) {
        return a.length < b.length || (a.length == b.length && a.delta < b.delta); });
    return taps;
}

} // namespace certification::detail

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

}// namespace tools::cuda

}// namespace nanovdb

#endif // NVIDIA_TOOLS_CUDA_CERTIFYEXTERIOR_CUH_HAS_BEEN_INCLUDED
