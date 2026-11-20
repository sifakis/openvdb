// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file csg_timing.cu
///
/// @brief Wall time of the CSG operations of tools/cuda/CSG.cuh on two sphere pairs and, with two band files as
///        arguments (see mesh_to_band.cu), on two mesh level sets. csgUnionPropagateSign, the baseline that
///        reruns PropagateSign on the result, is timed next to csgUnion.
///
/// usage: csg_timing [a.band b.band]

#include "CsgExampleUtil.cuh"

static void timeAll(const char* name, const StateT& a, const StateT& b, ResourceT& res)
{
    const auto count = [](const StateT& s) { return (unsigned long long)util::cuda::DeviceGridTraits<BuildT>::getTreeData(s.grid.template deviceGrid<BuildT>(), 0).mVoxelCount; };
    printf("%s: active voxels %llu + %llu\n", name, count(a), count(b));
    timeIt("csgUnionPropagateSign", [&] { auto r = tools::cuda::csgUnionPropagateSign<BuildT, float, ResourceT>(a, b, W, 0, res); });
    timeIt("csgUnion", [&] { auto r = tools::cuda::csgUnion<BuildT, float, ResourceT>(a, b, 0, res); });
    timeIt("csgIntersection", [&] { auto r = tools::cuda::csgIntersection<BuildT, float, ResourceT>(a, b, 0, res); });
    timeIt("csgDifference", [&] { auto r = tools::cuda::csgDifference<BuildT, float, ResourceT>(a, b, 0, res); });
    StateT c = tools::cuda::csgUnion<BuildT, float, ResourceT>(a, a, 0, res); // a copy of a, complemented in place an even number of times
    timeIt("csgComplement", [&] { tools::cuda::csgComplement<BuildT, float, ResourceT>(c, 0); }, 42);
}

int main(int argc, char** argv)
{
    ResourceT& res = nanovdb::cuda::default_resource<ResourceT>();
    { StateT a = makeSphere({64, 64, 64, 120}, res), b = makeSphere({190.3f, 70.1f, 64, 100}, res); timeAll("spheres r=120, 100", a, b, res); }
    { StateT a = makeSphere({10, 20, 30, 300}, res), b = makeSphere({260.5f, 100.2f, 30, 250}, res); timeAll("spheres r=300, 250", a, b, res); }
    if (argc >= 3) { StateT a = makeFromBand(loadBand(argv[1]), Coord(0), res), b = makeFromBand(loadBand(argv[2]), Coord(0, 0, 60), res); timeAll("dragon, armadillo z+60", a, b, res); }
    return 0;
}
