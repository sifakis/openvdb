// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

// SdfField.cuh: the signed distance field of a triangle mesh, and its value and gradient at any world
// position, for collisions of cloth against the mesh.
//
// The field is made by nanovdb::tools::cuda::MeshToSDF (a narrow band of signed distances on a ValueOnIndex grid)
// followed by PropagateSign (the sign of every voxel outside the band). Reading it reuses NanoVDB's
// TrilinearSampler (math/SampleFromVoxels.h): its stencil, its interpolation and the gradient of the interpolant.
// OpenVDB counterparts, used only to check this file: tools::meshToLevelSet and tools::BoxSampler.
#pragma once
#include <nanovdb/tools/cuda/MeshToSDF.cuh>
#include <nanovdb/tools/cuda/PropagateSign.cuh>
#include <nanovdb/math/SampleFromVoxels.h>

namespace cloth {

using BuildT    = nanovdb::ValueOnIndex;
using GridT     = nanovdb::NanoGrid<BuildT>;
using ResourceT = nanovdb::cuda::DeviceResource;
template <typename T>
using BufferT = nanovdb::cuda::Buffer<T, nanovdb::cuda::ResourceRef<ResourceT>>;

/// @brief What a kernel needs to read the field. World position x sits at index position x / voxelSize.
struct SdfView
{
    const GridT*            grid;
    const float*            sdf; // world units; slot 0 = the positive background
    const nanovdb::Mask<3>* leafMasks; // the invert masks of PropagateSign
    const nanovdb::Mask<4>* lowerMasks;
    const nanovdb::Mask<5>* upperMasks;
    const nanovdb::Coord*   rootOrigins;
    uint64_t                rootOriginCount;
    float                   voxelSize;

    /// @brief The value at a voxel: its own inside the band, otherwise the background, negated inside the mesh.
    struct AccT
    {
        using ValueType = float;
        using CoordType = nanovdb::Coord;
        const SdfView&                       view;
        nanovdb::DefaultReadAccessor<BuildT> acc;
        __hostdev__ float getValue(const nanovdb::Coord& ijk) const
        {
            const uint64_t index = acc.getValue(ijk);
            if (index) return view.sdf[index];
            const bool inside = nanovdb::tools::cuda::sdf::detail::isInverted(*view.grid, ijk, view.leafMasks,
                view.lowerMasks, view.upperMasks, view.rootOrigins, view.rootOriginCount);
            return inside ? -view.sdf[0] : view.sdf[0];
        }
    };

    /// @brief Trilinear interpolation of the field at world position @a x.
    /// @param gradient Derivative of the interpolant with respect to the world position (close to a unit vector
    ///        inside the band; near zero where all 8 voxels around @a x are outside the band).
    /// @return The signed distance in world units, negative inside the mesh.
    __hostdev__ float sample(const nanovdb::Vec3f& x, nanovdb::Vec3f& gradient) const
    {
        using SamplerT = nanovdb::math::TrilinearSampler<AccT>;
        const nanovdb::Vec3f p = x / voxelSize;
        nanovdb::Coord       ijk(int(floorf(p[0])), int(floorf(p[1])), int(floorf(p[2])));
        const nanovdb::Vec3f uvw(p[0] - float(ijk[0]), p[1] - float(ijk[1]), p[2] - float(ijk[2]));
        const AccT           acc{*this, grid->getAccessor()};
        float                v[2][2][2];
        SamplerT(acc).stencil(ijk, v);
        gradient = SamplerT::gradient(uvw, v) / voxelSize;
        return SamplerT::sample(uvw, v);
    }
};

/// @brief Owns the field of one mesh.
struct SdfField
{
    using HandleT = nanovdb::tools::cuda::MeshToSDF<BuildT>::HandleT;
    HandleT                   handle;
    BufferT<std::byte>        sdf;
    BufferT<nanovdb::Mask<3>> leafMasks;
    BufferT<nanovdb::Mask<4>> lowerMasks;
    BufferT<nanovdb::Mask<5>> upperMasks;
    BufferT<nanovdb::Coord>   rootOrigins;
    float                     voxelSize;
    uint64_t                  valueCount; // active voxels + 1

    /// @brief Run MeshToSDF and PropagateSign on a mesh that is on the device.
    /// @param voxelSize World size of a voxel; voxel (i, j, k) sits at world position (i, j, k) * voxelSize.
    /// @param bandWidth Half-width of the narrow band, in voxels.
    /// @param solid Whether a point enclosed by several closed surfaces of the mesh is inside (NestingRule::Solid)
    ///        or inside only for an odd number of them (NestingRule::EvenOdd).
    SdfField(const nanovdb::Vec3f* devicePoints, uint32_t pointCount, const nanovdb::Vec3i* deviceTriangles,
             uint32_t triangleCount, float voxelSize, float bandWidth, bool solid, cudaStream_t stream = 0)
        : SdfField(build(devicePoints, pointCount, deviceTriangles, triangleCount, voxelSize, bandWidth, solid, stream),
                   voxelSize, stream)
    {
    }

    SdfView view() const
    {
        return SdfView{handle.template deviceGrid<BuildT>(), reinterpret_cast<const float*>(sdf.data()), leafMasks.data(),
                       lowerMasks.data(), upperMasks.data(), rootOrigins.data(), rootOrigins.size(), voxelSize};
    }

private:
    using ConverterT = nanovdb::tools::cuda::MeshToSDF<BuildT>;
    static std::pair<HandleT, BufferT<std::byte>> build(const nanovdb::Vec3f* devicePoints, uint32_t pointCount,
        const nanovdb::Vec3i* deviceTriangles, uint32_t triangleCount, float voxelSize, float bandWidth, bool solid,
        cudaStream_t stream)
    {
        nanovdb::Map map;
        map.set(double(voxelSize), nanovdb::Vec3d(0.0), 1.0);
        ConverterT converter(devicePoints, pointCount, deviceTriangles, triangleCount, map, stream);
        converter.setNarrowBandWidth(bandWidth);
        converter.setNestingRule(solid ? ConverterT::NestingRule::Solid : ConverterT::NestingRule::EvenOdd);
        return converter.getHandleAndSDF();
    }

    SdfField(std::pair<HandleT, BufferT<std::byte>>&& result, float voxelSize, cudaStream_t stream)
        : handle(std::move(result.first))
        , sdf(std::move(result.second))
        , leafMasks(stream, resource(), 0, nanovdb::cuda::noInit)
        , lowerMasks(stream, resource(), 0, nanovdb::cuda::noInit)
        , upperMasks(stream, resource(), 0, nanovdb::cuda::noInit)
        , rootOrigins(stream, resource(), 0, nanovdb::cuda::noInit)
        , voxelSize(voxelSize)
        , valueCount(sdf.size_bytes() / sizeof(float))
    {
        const GridT* grid = handle.template deviceGrid<BuildT>();
        const auto   tree = nanovdb::util::cuda::DeviceGridTraits<BuildT>::getTreeData(grid, stream);
        leafMasks  = BufferT<nanovdb::Mask<3>>(stream, resource(), tree.mNodeCount[0], nanovdb::cuda::noInit);
        lowerMasks = BufferT<nanovdb::Mask<4>>(stream, resource(), tree.mNodeCount[1], nanovdb::cuda::noInit);
        upperMasks = BufferT<nanovdb::Mask<5>>(stream, resource(), tree.mNodeCount[2], nanovdb::cuda::noInit);
        rootOrigins = nanovdb::tools::cuda::PropagateSign<BuildT>(grid, reinterpret_cast<const float*>(sdf.data()), stream)
                          .propagate(leafMasks.data(), lowerMasks.data(), upperMasks.data()).first;
    }

    static nanovdb::cuda::ResourceRef<ResourceT> resource()
    {
        return nanovdb::cuda::ResourceRef<ResourceT>(nanovdb::cuda::default_resource<ResourceT>());
    }
};

} // namespace cloth
