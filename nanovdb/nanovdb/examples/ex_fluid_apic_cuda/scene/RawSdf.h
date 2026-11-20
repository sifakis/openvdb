// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <nanovdb/NanoVDB.h>
namespace surface
{
using Vec = nanovdb::Vec3d;
using Coord = nanovdb::Coord;
using Build = nanovdb::ValueOnIndex;
using Grid = nanovdb::NanoGrid<Build>;
inline Coord floorCoord(Vec x) { return Coord(int(floor(x[0])), int(floor(x[1])), int(floor(x[2]))); }
// Exact CPU counterpart of the existing MeshToSDF signed sidecar / invert-mask decoder.
struct RawSdf
{
    const Grid *grid = nullptr;
    double h = .025, band = .075;
    double voxel(Coord q) const
    {
        const auto &tree = grid->tree();
        const auto *tile = tree.root().probeTile(q);
        if (!tile || !tile->isChild())
        {
            if (grid->blindDataCount() < 5)
                return band;
            // The committed MeshToSDF uses a dense root bitmap plus root_extent.
            // Newer local versions use a sparse list of int32 root-tile origins.
            if (grid->blindMetaData(4).mDataType == nanovdb::GridType::UInt8)
            {
                if (grid->blindDataCount() < 6)
                    return band;
                const auto *extent = grid->getBlindData<int32_t>(5);
                const int i = (q[0] >> 12) - extent[0];
                const int j = (q[1] >> 12) - extent[1];
                const int k = (q[2] >> 12) - extent[2];
                if (i < 0 || j < 0 || k < 0 || i >= extent[3] || j >= extent[4] || k >= extent[5])
                    return band;
                const uint64_t index = (uint64_t(i) * extent[4] + j) * extent[5] + k;
                return grid->getBlindData<uint8_t>(4)[index] ? -band : band;
            }
            const auto *roots = reinterpret_cast<const Coord *>(grid->getBlindData<int32_t>(4));
            Coord origin = (q >> 12) << 12;
            for (uint64_t k = 0; k < grid->blindMetaData(4).mValueCount / 3; ++k)
                if (roots[k] == origin)
                    return -band;
            return band;
        }
        const auto *upper = tree.root().getChild(tile);
        const auto *lower = upper->probeChild(q);
        if (!lower)
        {
            auto idx = upper - tree.getFirstNode<2>();
            auto masks = reinterpret_cast<const nanovdb::Mask<5> *>(grid->getBlindData<uint64_t>(3));
            return masks[idx].isOn(nanovdb::NanoUpper<Build>::CoordToOffset(q)) ? -band : band;
        }
        const auto *leaf = lower->probeChild(q);
        if (!leaf)
        {
            auto idx = lower - tree.getFirstNode<1>();
            auto masks = reinterpret_cast<const nanovdb::Mask<4> *>(grid->getBlindData<uint64_t>(2));
            return masks[idx].isOn(nanovdb::NanoLower<Build>::CoordToOffset(q)) ? -band : band;
        }
        unsigned n = nanovdb::NanoLeaf<Build>::CoordToOffset(q);
        if (leaf->isActive(n))
            return reinterpret_cast<const float *>(grid->getBlindData<float>(0))[leaf->getValue(n)];
        auto idx = leaf - tree.getFirstNode<0>();
        auto masks = reinterpret_cast<const nanovdb::Mask<3> *>(grid->getBlindData<uint64_t>(1));
        return masks[idx].isOn(n) ? -band : band;
    }
    double sdf(Vec world) const
    {
        Vec p = grid->worldToIndex(world);
        Coord q = floorCoord(p);
        Vec f = p - Vec(q[0], q[1], q[2]);
        double value = 0;
        for (int x = 0; x < 2; ++x)
            for (int y = 0; y < 2; ++y)
                for (int z = 0; z < 2; ++z)
                    value += (x ? f[0] : 1 - f[0]) * (y ? f[1] : 1 - f[1]) * (z ? f[2] : 1 - f[2]) *
                             voxel(q + Coord(x, y, z));
        return value;
    }
    Vec normal(Vec x) const
    {
        Vec n;
        for (int a = 0; a < 3; ++a)
        {
            Vec e(0);
            e[a] = h * .25;
            n[a] = sdf(x + e) - sdf(x - e);
        }
        double len = n.length();
        return len > 1e-12 ? n / len : Vec(0);
    }
};
} // namespace surface
