// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file mesh_to_band.cu
///
/// @brief OBJ mesh -> narrow-band level set with tools::cuda::MeshToSDF, written as a band file: a uint64 voxel
///        count, then per voxel int32 x, y, z and a float value, in index space with voxel size 1. The mesh is
///        centered at the origin and scaled so that its longest extent is the given number of voxels; the band
///        half-width is 3 voxels. The other programs of this example read these files as mesh inputs.
///
/// usage: mesh_to_band in.obj longestExtentInVoxels out.band [heuristic|interior|ball]   (barrier signing; default heuristic)

#include <nanovdb/tools/cuda/MeshToSDF.cuh>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

using BuildT = nanovdb::ValueOnIndex;
using ResourceT = nanovdb::cuda::DeviceResource;
template<typename T> using BufT = nanovdb::cuda::Buffer<T, nanovdb::cuda::ResourceRef<ResourceT>>;

int main(int argc, char** argv)
{
    if (argc < 4) { printf("usage: %s in.obj longestExtentInVoxels out.band\n", argv[0]); return 1; }
    const float target = std::stof(argv[2]), halfWidth = 3.0f;
    std::vector<nanovdb::Vec3f> pts; std::vector<nanovdb::Vec3i> tris;
    std::ifstream in(argv[1]); std::string line;
    while (std::getline(in, line)) {
        std::istringstream ss(line); std::string tag; ss >> tag;
        if (tag == "v") { float x, y, z; ss >> x >> y >> z; pts.emplace_back(x, y, z); }
        else if (tag == "f") {
            std::vector<int> idx; std::string tok;
            while (ss >> tok) { int i = std::stoi(tok.substr(0, tok.find('/'))); idx.push_back(i < 0 ? int(pts.size()) + i : i - 1); }
            for (size_t k = 2; k < idx.size(); ++k) tris.emplace_back(idx[0], idx[k - 1], idx[k]);
        }
    }
    nanovdb::Vec3f lo(1e30f), hi(-1e30f);
    for (auto& p : pts) for (int c = 0; c < 3; ++c) { lo[c] = std::min(lo[c], p[c]); hi[c] = std::max(hi[c], p[c]); }
    const nanovdb::Vec3f ext = hi - lo, mid = (hi + lo) * 0.5f;
    const float s = target / std::max(ext[0], std::max(ext[1], ext[2]));
    for (auto& p : pts) p = (p - mid) * s; // centered at the origin, in voxels

    ResourceT& res = nanovdb::cuda::default_resource<ResourceT>();
    nanovdb::cuda::ResourceRef<ResourceT> ref(res);
    BufT<nanovdb::Vec3f> dPts(cudaStream_t(0), ref, pts.size(), nanovdb::cuda::noInit);
    BufT<nanovdb::Vec3i> dTris(cudaStream_t(0), ref, tris.size(), nanovdb::cuda::noInit);
    cudaCheck(cudaMemcpy(dPts.data(), pts.data(), pts.size() * sizeof(nanovdb::Vec3f), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dTris.data(), tris.data(), tris.size() * sizeof(nanovdb::Vec3i), cudaMemcpyHostToDevice));

    nanovdb::tools::cuda::MeshToSDF<BuildT, ResourceT> converter(dPts.data(), uint32_t(pts.size()), dTris.data(), uint32_t(tris.size()), nanovdb::Map(), 0, res);
    converter.setNarrowBandWidth(halfWidth);
    using SigningT = nanovdb::tools::cuda::MeshToSDF<BuildT, ResourceT>::BarrierSigning;
    const std::string signing = argc > 4 ? argv[4] : "heuristic";
    if (signing == "interior") converter.setBarrierSigning(SigningT::Interior);
    else if (signing == "ball") converter.setBarrierSigning(SigningT::Ball);
    else if (signing != "heuristic") { printf("unknown barrier signing %s\n", signing.c_str()); return 1; }
    auto [handle, sdfBytes] = converter.getHandleAndSDF();

    std::vector<char> hostGrid(handle.buffer().size());
    cudaCheck(cudaMemcpy(hostGrid.data(), handle.buffer().data(), hostGrid.size(), cudaMemcpyDeviceToHost));
    std::vector<float> sdf(sdfBytes.size() / sizeof(float));
    cudaCheck(cudaMemcpy(sdf.data(), sdfBytes.data(), sdf.size() * sizeof(float), cudaMemcpyDeviceToHost));
    const auto* grid = reinterpret_cast<const nanovdb::NanoGrid<BuildT>*>(hostGrid.data());
    const auto& tree = grid->tree();

    FILE* f = fopen(argv[3], "wb");
    uint64_t n = 0, neg = 0, dropped = 0; fwrite(&n, 8, 1, f);
    for (uint32_t l = 0; l < tree.nodeCount(0); ++l) {
        const auto& leaf = tree.template getFirstNode<0>()[l];
        for (uint32_t i = 0; i < 512; ++i) {
            if (!leaf.isActive(i)) continue;
            const float v = sdf[leaf.getValue(i)];
            if (!(std::fabs(v) < halfWidth)) { ++dropped; continue; }
            const nanovdb::Coord c = leaf.offsetToGlobalCoord(i); const int32_t xyz[3] = {c[0], c[1], c[2]};
            fwrite(xyz, 4, 3, f); fwrite(&v, 4, 1, f); ++n; if (v < 0) ++neg;
        }
    }
    fseek(f, 0, SEEK_SET); fwrite(&n, 8, 1, f); fclose(f);
    printf("%s: %zu vertices, %zu triangles, extent in voxels %.1f x %.1f x %.1f, voxel size %g, MeshToSDF active voxels %llu, background %g, band voxels written %llu (negative %llu), active voxels with |value| >= %g not written %llu\n",
           argv[1], pts.size(), tris.size(), ext[0] * s, ext[1] * s, ext[2] * s, grid->voxelSize()[0], (unsigned long long)grid->activeVoxelCount(), sdf[0],
           (unsigned long long)n, (unsigned long long)neg, halfWidth, (unsigned long long)dropped);
    return 0;
}
