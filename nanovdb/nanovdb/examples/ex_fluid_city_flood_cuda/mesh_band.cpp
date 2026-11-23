// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

// mesh_band.cpp <in.band> <out.meshbin>: the water surface of one frame as triangles, from a band file written by
// eulerian.cu (BAND=...). OpenVDB: the band's values go into a FloatGrid, signedFloodFill gives the inactive
// voxels their sign, volumeToMesh extracts the zero level set. Vertices are in cell-center coordinates (voxels).
// Output (the format the APIC example's render_scene.py reads): uint64 magic "MESH0001", uint64 vertex count,
// uint64 triangle count, float32 x y z per vertex, int32 a b c per triangle. Triangles face from water to air.
#include <openvdb/openvdb.h>
#include <openvdb/tools/SignedFloodFill.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <cstdint>
#include <cstdio>
#include <vector>

int main(int argc, char** argv)
{
    if (argc < 3) { std::printf("usage: mesh_band <in.band> <out.meshbin>\n"); return 1; }
    std::FILE* in = std::fopen(argv[1], "rb");
    if (!in) { std::printf("cannot open %s\n", argv[1]); return 1; }
    int32_t header[4];
    if (std::fread(header, sizeof(int32_t), 4, in) != 4) return 1;
    std::vector<char> records(size_t(header[0]) * 10);
    if (std::fread(records.data(), 1, records.size(), in) != records.size()) return 1;
    std::fclose(in);

    openvdb::initialize();
    openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create(2.0f);
    {
        auto acc = grid->getAccessor();
        for (int32_t k = 0; k < header[0]; ++k) {
            const int16_t* xyz = reinterpret_cast<const int16_t*>(records.data() + size_t(k) * 10);
            const float    v   = *reinterpret_cast<const float*>(records.data() + size_t(k) * 10 + 6);
            acc.setValue(openvdb::Coord(xyz[0], xyz[1], xyz[2]), v);
        }
    }
    openvdb::tools::signedFloodFill(grid->tree());
    std::vector<openvdb::Vec3s> points;
    std::vector<openvdb::Vec3I> triangles;
    std::vector<openvdb::Vec4I> quads;
    openvdb::tools::volumeToMesh(*grid, points, triangles, quads, 0.0, 0.0);
    for (const auto& q : quads) triangles.emplace_back(q[0], q[1], q[2]), triangles.emplace_back(q[0], q[2], q[3]);

    // Orientation: the normal of a triangle must point where the level set grows (from water to air).
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

    std::FILE* out = std::fopen(argv[2], "wb");
    const uint64_t head[3] = {0x4D45534830303031ull, points.size(), triangles.size()};
    std::fwrite(head, sizeof(uint64_t), 3, out);
    std::fwrite(points.data(), sizeof(openvdb::Vec3s), points.size(), out);
    std::vector<int32_t> faces(triangles.size() * 3);
    for (size_t t = 0; t < triangles.size(); ++t) for (int a = 0; a < 3; ++a) faces[3 * t + a] = int32_t(triangles[t][a]);
    std::fwrite(faces.data(), sizeof(int32_t), faces.size(), out);
    std::fclose(out);
    std::printf("%s: %d band cells -> %zu vertices, %zu triangles (%s) -> %s\n", argv[1], header[0], points.size(), triangles.size(),
                inward > outward ? "flipped" : "as extracted", argv[2]);
    return 0;
}
