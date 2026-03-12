// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file Rasterization.cuh

    \author Efty Sifakis

    \brief Implements GPU kernels for rasterizing triangle mesh geometry into
           NanoVDB sparse tree topology (upper/lower internal node masks).
*/

#ifndef NANOVDB_UTIL_CUDA_RASTERIZATION_CUH_HAS_BEEN_INCLUDED
#define NANOVDB_UTIL_CUDA_RASTERIZATION_CUH_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h>

namespace nanovdb::util {

namespace rasterization {

namespace cuda {

/// @brief Scatters leaf/triangle pair origins into the upper and lower internal
///        node topology masks of a rasterized NanoVDB tree.
///
///        Launched with 1 thread per leaf/triangle pair. Each thread:
///          1. Locates its root tile via probeTile() + pointer-difference index
///          2. Computes upper and lower child offsets via CoordToOffset()
///          3. Atomically sets the corresponding bits in mUpperMasks and mLowerMasks
///
///        Both setOnAtomic() calls are required since multiple pairs can share
///        the same upper node (competing for the upper mask) or the same lower
///        node (competing for the lower mask).
template<typename BuildT, typename PairT>
struct RasterizeInternalNodesFunctor
{
    using RootT  = NanoRoot<BuildT>;
    using UpperT = NanoUpper<BuildT>;
    using LowerT = NanoLower<BuildT>;

    const PairT*      dPairs;
    const RootT*      dRoot;
    Mask<5>*          dUpperMasks;
    Mask<4>           (*dLowerMasks)[Mask<5>::SIZE];

    __device__ void operator()(size_t pairID) const
    {
        const auto& pair = dPairs[pairID];

        // Locate the root tile containing this leaf origin
        const auto* tile = dRoot->probeTile(pair.origin);
        uint64_t tileIdx = util::PtrDiff(tile, dRoot->tile(0))
                           / sizeof(typename RootT::Tile);

        // Offsets of the enclosing upper and lower nodes
        const uint32_t upperBit = UpperT::CoordToOffset(pair.origin);
        const uint32_t lowerBit = LowerT::CoordToOffset(pair.origin);

        dUpperMasks[tileIdx].setOnAtomic(upperBit);
        dLowerMasks[tileIdx][upperBit].setOnAtomic(lowerBit);
    }
};

} // namespace cuda

} // namespace rasterization

} // namespace nanovdb::util

#endif // NANOVDB_UTIL_CUDA_RASTERIZATION_CUH_HAS_BEEN_INCLUDED
