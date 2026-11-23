// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

// gen_sdf.cu <mesh.obj> <res> <band in voxels> <out.sdf>: run MeshToSDF on a triangle mesh and write the result as the
// obstacle file that eulerian.cu reads (OBSTACLE=...). The voxel size is the largest extent of the mesh divided by
// res. File: uint64 grid bytes, uint64 value count, double voxel size, float background, then the grid buffer and
// the float sidecar (slot 0 = background).
#include <nanovdb/tools/cuda/MeshToSDF.cuh>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using BuildT    = nanovdb::ValueOnIndex;
using ResourceT = nanovdb::cuda::DeviceResource;
template <typename T>
using BufferT = nanovdb::cuda::Buffer<T, nanovdb::cuda::ResourceRef<ResourceT>>;
static BufferT<std::byte> emptyBuffer()
{
    return BufferT<std::byte>(cudaStream_t(0), nanovdb::cuda::default_resource<ResourceT>(), 0, nanovdb::cuda::noInit);
}

static void readObj(const char* path, std::vector<nanovdb::Vec3f>& pts, std::vector<nanovdb::Vec3i>& tris)
{
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.size() < 2) continue;
        if (line[0] == 'v' && line[1] == ' ') {
            float x, y, z;
            if (std::sscanf(line.c_str() + 2, "%f %f %f", &x, &y, &z) == 3) pts.emplace_back(x, y, z);
        } else if (line[0] == 'f' && line[1] == ' ') {
            std::istringstream iss(line.substr(2));
            std::string tok;
            std::vector<int> idx;
            while (iss >> tok) {
                const int i = std::stoi(tok.substr(0, tok.find('/')));
                idx.push_back(i > 0 ? i - 1 : int(pts.size()) + i);
            }
            for (size_t k = 2; k < idx.size(); ++k) tris.emplace_back(idx[0], idx[k - 1], idx[k]);
        }
    }
}


int main(int argc, char** argv)
{
    if (argc < 5) { std::printf("usage: gen_sdf <mesh.obj> <res> <band> <out.sdf>\n"); return 1; }
    std::vector<nanovdb::Vec3f> pts;
    std::vector<nanovdb::Vec3i> tris;
    readObj(argv[1], pts, tris);
    const int   res  = std::atoi(argv[2]);
    const float band = float(std::atof(argv[3]));
    float lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};
    for (const auto& p : pts)
        for (int a = 0; a < 3; ++a) lo[a] = std::min(lo[a], p[a]), hi[a] = std::max(hi[a], p[a]);
    const double vs = std::max({hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]}) / double(res);

    nanovdb::cuda::Buffer<nanovdb::Vec3f> dPts(cudaStream_t(0), pts.size(), nanovdb::cuda::noInit);
    nanovdb::cuda::Buffer<nanovdb::Vec3i> dTris(cudaStream_t(0), tris.size(), nanovdb::cuda::noInit);
    cudaCheck(cudaMemcpy(dPts.data(), pts.data(), dPts.size_bytes(), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dTris.data(), tris.data(), dTris.size_bytes(), cudaMemcpyHostToDevice));
    nanovdb::Map map;
    map.set(vs, nanovdb::Vec3d(0.0), 1.0);

    nanovdb::tools::cuda::MeshToSDF<BuildT> converter(dPts.data(), uint32_t(pts.size()), dTris.data(), uint32_t(tris.size()), map);
    converter.setNarrowBandWidth(band);
    auto [handle, sdf] = converter.getHandleAndSDF();
    cudaCheck(cudaDeviceSynchronize());

    std::vector<char> grid(handle.buffer().size());
    cudaCheck(cudaMemcpy(grid.data(), handle.buffer().data(), grid.size(), cudaMemcpyDeviceToHost));
    std::vector<float> phi(sdf.size_bytes() / sizeof(float));
    cudaCheck(cudaMemcpy(phi.data(), sdf.data(), sdf.size_bytes(), cudaMemcpyDeviceToHost));
    float maxAbs = 0;
    uint64_t negative = 0;
    for (size_t k = 1; k < phi.size(); ++k) maxAbs = std::max(maxAbs, std::fabs(phi[k])), negative += phi[k] < 0;
    const uint64_t gridBytes = grid.size(), valueCount = phi.size();
    const float background = phi[0];
    std::ofstream out(argv[4], std::ios::binary);
    out.write(reinterpret_cast<const char*>(&gridBytes), 8).write(reinterpret_cast<const char*>(&valueCount), 8);
    out.write(reinterpret_cast<const char*>(&vs), 8).write(reinterpret_cast<const char*>(&background), 4);
    out.write(grid.data(), grid.size()).write(reinterpret_cast<const char*>(phi.data()), phi.size() * sizeof(float));
    std::printf("%s res %d band %g: voxel size %g, %llu active voxels (%llu negative), background %g = %g voxels, "
                "max |sdf| %g voxels -> %s\n", argv[1], res, band, vs, (unsigned long long)(valueCount - 1),
                (unsigned long long)negative, background, background / vs, maxAbs / vs, argv[4]);
    return 0;
}
