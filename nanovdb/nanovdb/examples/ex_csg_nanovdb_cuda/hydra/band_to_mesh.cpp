// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

// band_to_mesh.cpp <in.band> <out.meshbin | out.obj>: the zero level set of a band file (see ../mesh_to_band.cu:
// a uint64 voxel count, then per voxel int32 x, y, z and a float value) as triangles. OpenVDB: the band's values
// go into a FloatGrid with background 3, signedFloodFill gives the inactive voxels their sign, volumeToMesh
// extracts the surface. Vertices are in voxels; triangles face from the inside (negative values) to the outside.
// Output by extension: .obj, or the binary .meshbin that render.py reads: uint64 magic "MESH0001", uint64 vertex
// count, uint64 triangle count, float32 x y z per vertex, int32 a b c per triangle.
#include <openvdb/openvdb.h>
#include <openvdb/tools/SignedFloodFill.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    if (argc < 3) { std::printf("usage: band_to_mesh <in.band> <out.meshbin | out.obj>\n"); return 1; }
    std::FILE* in = std::fopen(argv[1], "rb");
    if (!in) { std::printf("cannot open %s\n", argv[1]); return 1; }
    uint64_t count = 0;
    if (std::fread(&count, 8, 1, in) != 1) return 1;
    struct Record { int32_t xyz[3]; float value; };
    std::vector<Record> records(count);
    if (std::fread(records.data(), sizeof(Record), count, in) != count) return 1;
    std::fclose(in);

    openvdb::initialize();
    openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create(3.0f);
    {
        auto acc = grid->getAccessor();
        for (const Record& r : records) acc.setValue(openvdb::Coord(r.xyz[0], r.xyz[1], r.xyz[2]), r.value);
    }
    openvdb::tools::signedFloodFill(grid->tree());
    std::vector<openvdb::Vec3s> points;
    std::vector<openvdb::Vec3I> triangles;
    std::vector<openvdb::Vec4I> quads;
    openvdb::tools::volumeToMesh(*grid, points, triangles, quads, 0.0, 0.0);
    for (const auto& q : quads) triangles.emplace_back(q[0], q[1], q[2]), triangles.emplace_back(q[0], q[2], q[3]);

    // Orientation: the normal of a triangle must point where the level set grows.
    auto acc = grid->getConstAccessor();
    long outward = 0, inward = 0;
    for (size_t t = 0; t < triangles.size(); t += std::max<size_t>(1, triangles.size() / 2000)) {
        const openvdb::Vec3s a = points[triangles[t][0]], b = points[triangles[t][1]], c = points[triangles[t][2]];
        openvdb::Vec3s n = (b - a).cross(c - a);
        if (n.lengthSqr() < 1e-12f) continue;
        n.normalize();
        const openvdb::Vec3s m = (a + b + c) / 3.0f;
        const openvdb::Coord front = openvdb::Coord::round(m + n), back = openvdb::Coord::round(m - n);
        (acc.getValue(front) > acc.getValue(back) ? outward : inward) += 1;
    }
    if (inward > outward) for (auto& t : triangles) std::swap(t[1], t[2]);

    const std::string path = argv[2];
    std::FILE* out = std::fopen(argv[2], "wb");
    if (!out) { std::printf("cannot write %s\n", argv[2]); return 1; }
    if (path.size() > 4 && path.substr(path.size() - 4) == ".obj") {
        for (const auto& p : points) std::fprintf(out, "v %g %g %g\n", p[0], p[1], p[2]);
        for (const auto& t : triangles) std::fprintf(out, "f %u %u %u\n", t[0] + 1, t[1] + 1, t[2] + 1);
    } else {
        const uint64_t head[3] = {0x4D45534830303031ull, points.size(), triangles.size()};
        std::fwrite(head, sizeof(uint64_t), 3, out);
        std::fwrite(points.data(), sizeof(openvdb::Vec3s), points.size(), out);
        std::vector<int32_t> faces(triangles.size() * 3);
        for (size_t t = 0; t < triangles.size(); ++t) for (int a = 0; a < 3; ++a) faces[3 * t + a] = int32_t(triangles[t][a]);
        std::fwrite(faces.data(), sizeof(int32_t), faces.size(), out);
    }
    std::fclose(out);
    std::printf("%s: %llu band voxels -> %zu vertices, %zu triangles -> %s\n", argv[1], (unsigned long long)count, points.size(), triangles.size(), argv[2]);
    return 0;
}
