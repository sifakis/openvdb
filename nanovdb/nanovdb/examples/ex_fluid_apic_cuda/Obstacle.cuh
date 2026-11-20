// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifndef EX_MESH_TO_SDF_OBSTACLE_CUH
#define EX_MESH_TO_SDF_OBSTACLE_CUH
#include "Fluid.cuh"
#include <nanovdb/tools/cuda/MeshToSDF.cuh>

namespace fluid {
// These three motions retain the same eight vertices and twelve triangles.
// Breathing changes the dimensions while preserving the cuboid's volume.
struct Motion
{
    int kind = 0;
    double time = 0;
    __hostdev__ Vec origin() const { return Vec(.85 + (kind == 1 ? .12 * sin(6 * time) : 0), .38, .6); }
    __hostdev__ Vec half() const
    {
        double s = kind == 2 ? 1 + .2 * sin(6 * time) : 1;
        return Vec(.18 * s, .18 / s, .18);
    }
    __hostdev__ Vec velocity(Vec x) const
    {
        if (kind == 1)
            return Vec(.72 * cos(6 * time), 0, 0);
        if (kind == 2) {
            double rate = 1.2 * cos(6 * time) / (1 + .2 * sin(6 * time));
            Vec q = x - origin();
            return Vec(rate * q[0], -rate * q[1], 0);
        }
        return Vec(0);
    }
    __hostdev__ double exact(Vec x) const
    {
        Vec q = x - origin(), h = half(), outside;
        for (int a = 0; a < 3; ++a) {
            q[a] = fabs(q[a]) - h[a];
            outside[a] = maxValue(q[a], 0.);
        }
        return outside.length() + minValue(maxValue(q[0], maxValue(q[1], q[2])), 0.);
    }
};
struct ObstacleView
{
    const Grid *grid = nullptr;
    Motion motion;
    double h = .025, band = .075;
    __device__ double voxel(Coord q) const
    {
        const auto &tree = grid->tree();
        const auto *tile = tree.root().probeTile(q);
        if (!tile || !tile->isChild()) {
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
        if (!lower) {
            auto idx = upper - tree.getFirstNode<2>();
            auto masks = reinterpret_cast<const nanovdb::Mask<5> *>(grid->getBlindData<uint64_t>(3));
            return masks[idx].isOn(nanovdb::NanoUpper<Build>::CoordToOffset(q)) ? -band : band;
        }
        const auto *leaf = lower->probeChild(q);
        if (!leaf) {
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
    __device__ double sdf(GeometryVec world) const
    {
        GeometryVec p = grid->worldToIndex(world);
        Coord q = floorCoord(p);
        GeometryVec f = p - GeometryVec(q[0], q[1], q[2]);
        double value = 0;
        for (int x = 0; x < 2; ++x)
            for (int y = 0; y < 2; ++y)
                for (int z = 0; z < 2; ++z)
                    value += (x ? f[0] : 1 - f[0]) * (y ? f[1] : 1 - f[1]) * (z ? f[2] : 1 - f[2]) *
                             voxel(q + Coord(x, y, z));
        return value;
    }
    __device__ GeometryVec normal(GeometryVec x) const
    {
        GeometryVec n;
        for (int a = 0; a < 3; ++a) {
            GeometryVec e(0);
            e[a] = h * .25;
            n[a] = sdf(x + e) - sdf(x - e);
        }
        double len = n.length();
        return len > 1e-12 ? n / len : GeometryVec(0);
    }
};
class Obstacle
{
    using Converter = nanovdb::tools::cuda::MeshToSDF<Build>;
    Buffer<nanovdb::Vec3f> vertices = allocate<nanovdb::Vec3f>(8);
    Buffer<nanovdb::Vec3i> triangles = allocate<nanovdb::Vec3i>(12);

  public:
    Converter::HandleT handle{nanovdb::tools::cuda::sdf_detail::allocate<std::byte>(0, 0)};
    ObstacleView view;
    std::vector<nanovdb::Vec3f> hostVertices;
    explicit Obstacle(double voxelSize = .025)
    {
        view.h = voxelSize;
        view.band = 3 * voxelSize;
        nanovdb::Vec3i ids[] = {{0, 2, 3}, {0, 3, 1}, {4, 5, 7}, {4, 7, 6}, {0, 1, 5}, {0, 5, 4},
                                {2, 6, 7}, {2, 7, 3}, {0, 4, 6}, {0, 6, 2}, {1, 3, 7}, {1, 7, 5}};
        cudaCheck(cudaMemcpy(triangles.data(), ids, sizeof(ids), cudaMemcpyHostToDevice));
    }
    double build(Motion motion)
    {
        Timer timer;
        view.motion = motion;
        hostVertices.clear();
        for (int j = 0; j < 8; ++j) {
            Vec p = motion.origin(), h = motion.half();
            for (int a = 0; a < 3; ++a)
                p[a] += ((j >> a) & 1) ? h[a] : -h[a];
            hostVertices.emplace_back(p);
        }
        cudaCheck(cudaMemcpy(vertices.data(), hostVertices.data(), 8 * sizeof(nanovdb::Vec3f),
                             cudaMemcpyHostToDevice));
        nanovdb::Map map;
        map.set(view.h, nanovdb::Vec3d(0), 1.0);
        Converter converter(vertices.data(), 8, triangles.data(), 12, map);
        converter.setNarrowBandWidth(float(view.band / view.h));
        converter.setBarrierSigning(Converter::BarrierSigning::Interior);
        handle = converter.getHandle();
        view.grid = handle.deviceGrid<Build>();
        return timer.ms();
    }
    void write(const std::string &path)
    {
        std::ofstream f(path + "_obstacle.csv");
        f << "x,y,z\n";
        for (auto p : hostVertices)
            f << p[0] << ',' << p[1] << ',' << p[2] << '\n';
    }
};
__global__ void obstacleCells(Domain d, ObstacleView obstacle)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (!i || i >= d.count)
        return;
    d.cells[i].phi = obstacle.sdf(center(d.coord(i)) * d.h);
}
__global__ void obstacleFaces(Domain d, ObstacleView obstacle)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (!i || i >= d.count)
        return;
    Cell &c = d.cells[i];
    for (int a = 0; a < 3; ++a) {
        if (!d.inside(d.coord(i)) || !d.inside(d.coord(i) - axis(a)))
            continue;
        Vec x = face(d.coord(i), a) * d.h;
        c.wall[a] = obstacle.motion.velocity(x)[a];
        // 4x4 face quadrature approximates the open area. Solid-centered
        // neighbors are fully blocked, a conservative first-order cell scheme.
        int open = 0;
        for (int u = 0; u < 4; ++u)
            for (int v = 0; v < 4; ++v) {
                Vec p = x;
                p[(a + 1) % 3] += ((u + .5) / 4 - .5) * d.h;
                p[(a + 2) % 3] += ((v + .5) / 4 - .5) * d.h;
                open += obstacle.sdf(p) > 0;
            }
        c.open[a] = open / 16.;
        if (c.phi <= 0 || d.cells[d.neighbor(i, 2 * a)].phi <= 0)
            c.open[a] = 0;
    }
}
struct Contact
{
    int attempted = 0, remaining = 0, exactRemaining = 0, contacts = 0;
    double before = 0, after = 0, exactDepth = 0, normalError = 0;
};
__global__ void collide(Particle *ps, int n, ObstacleView obstacle, Contact *result, double margin)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n)
        return;
    Particle &p = ps[k];
    Contact r;
    double phi = obstacle.sdf(p.x);
    r.attempted = phi < 0;
    r.before = maxValue(-phi, 0.);
    bool contact = false;
    for (int j = 0; j < 12 && phi < margin; ++j) {
        Vec normal = obstacle.normal(p.x);
        if (normal.length() < .5)
            break;
        p.x += normal * (margin - phi);
        phi = obstacle.sdf(p.x);
        contact = true;
    }
    if (contact) {
        Vec normal = obstacle.normal(p.x), wall = obstacle.motion.velocity(p.x);
        // Unilateral free-slip contact preserves separation and tangential C.
        double relative = (p.v - wall).dot(normal);
        if (relative < 0) {
            p.v -= normal * relative;
            Vec normalRow = p.c[0]*normal[0] + p.c[1]*normal[1] + p.c[2]*normal[2];
            for (int a = 0; a < 3; ++a) p.c[a] -= normalRow * normal[a];
        }
        r.contacts = 1;
        r.normalError = maxValue(0., -(p.v - wall).dot(normal));
    }
    r.remaining = phi < -1e-7;
    r.after = maxValue(-phi, 0.);
    double exact = obstacle.motion.exact(p.x);
    r.exactRemaining = exact < -1e-7;
    r.exactDepth = maxValue(-exact, 0.);
    result[k] = r;
}
__global__ void sdfProbe(ObstacleView o, double *out)
{
    if (threadIdx.x || blockIdx.x)
        return;
    out[0] = o.sdf(o.motion.origin());
    out[1] = o.sdf(o.motion.origin() + Vec(.8, 0, 0));
    double error = 0;
    for (int a = 0; a < 3; ++a)
        for (int s = -1; s <= 1; s += 2) {
            Vec p = o.motion.origin();
            p[a] += s * (o.motion.half()[a] + .02);
            error = maxValue(error, fabs(o.sdf(p) - .02));
        }
    out[2] = error;
    out[3] = o.grid->tree().getValue(floorCoord(o.motion.origin() / o.h));
}
inline void validateSdf(Obstacle &o)
{
    auto out = allocate<double>(4);
    sdfProbe<<<1, 1>>>(o.view, out.data());
    double values[4];
    cudaCheck(cudaMemcpy(values, out.data(), sizeof(values), cudaMemcpyDeviceToHost));
    std::cout << "sdf deep_interior=" << values[0] << " exterior=" << values[1] << " face_error=" << values[2]
              << " interior_index=" << values[3] << std::endl;
    if (values[0] >= 0 || values[1] <= 0 || values[2] > o.view.h * .5 || values[3] != 0)
        throw std::runtime_error("SDF sign or world-space sampling failed");
}
inline void validateContact(Obstacle &o)
{
    Particle p;
    p.x = o.view.motion.origin() + Vec(o.view.motion.half()[0] - .005, 0, 0);
    p.v = o.view.motion.velocity(p.x) - Vec(1, 0, 0);
    auto particle = allocate<Particle>(1);
    auto result = allocate<Contact>(1);
    cudaCheck(cudaMemcpy(particle.data(), &p, sizeof(p), cudaMemcpyHostToDevice));
    collide<<<1, 1>>>(particle.data(), 1, o.view, result.data(), o.view.h * .5);
    Contact c;
    cudaCheck(cudaMemcpy(&c, result.data(), sizeof(c), cudaMemcpyDeviceToHost));
    std::cout << "manufactured contact before=" << c.before << " after=" << c.after
              << " exact=" << c.exactDepth << " normal=" << c.normalError << std::endl;
    if (c.attempted != 1 || c.remaining || c.exactRemaining || c.normalError > 1e-8)
        throw std::runtime_error("manufactured contact failed");
    const double eps = 1e-6;
    double error = 0;
    for (int j = 0; j < 8; ++j) {
        Motion m = o.view.motion, plus = m, minus = m;
        plus.time += eps;
        minus.time -= eps;
        Vec a = plus.origin(), b = minus.origin(), x = m.origin();
        for (int d = 0; d < 3; ++d) {
            double sign = ((j >> d) & 1) ? 1 : -1;
            a[d] += sign * plus.half()[d];
            b[d] += sign * minus.half()[d];
            x[d] += sign * m.half()[d];
        }
        error = std::max<double>(error, ((a - b) / (2 * eps) - m.velocity(x)).length());
    }
    if (error > 1e-8)
        throw std::runtime_error("mesh vertex velocity derivative failed");
    std::cout << "vertex velocity derivative error=" << error << std::endl;
}
inline Contact contacts(Solver &s, Obstacle &o, double margin = -1)
{
    if (margin < 0)
        margin = o.view.h * .5;
    auto result = allocate<Contact>(s.particleCount);
    collide<<<blocks(s.particleCount), 256>>>(s.particles.data(), s.particleCount, o.view, result.data(),
                                              margin);
    std::vector<Contact> host(s.particleCount);
    cudaCheck(cudaMemcpy(host.data(), result.data(), host.size() * sizeof(Contact), cudaMemcpyDeviceToHost));
    Contact r;
    for (auto c : host) {
        r.attempted += c.attempted;
        r.remaining += c.remaining;
        r.exactRemaining += c.exactRemaining;
        r.contacts += c.contacts;
        r.before = std::max(r.before, c.before);
        r.after = std::max(r.after, c.after);
        r.exactDepth = std::max(r.exactDepth, c.exactDepth);
        r.normalError = std::max(r.normalError, c.normalError);
    }
    if (r.remaining || r.exactRemaining || r.normalError > 1e-8) {
        std::cerr << "contact failure sdf=" << r.remaining << " depth=" << r.after
                  << " mesh=" << r.exactRemaining << " depth=" << r.exactDepth << " normal=" << r.normalError
                  << std::endl;
        throw std::runtime_error("obstacle contact validation failed");
    }
    return r;
}
inline Contact combine(Contact a, const Contact &b)
{
    a.attempted += b.attempted;
    a.remaining += b.remaining;
    a.exactRemaining += b.exactRemaining;
    a.contacts += b.contacts;
    a.before = std::max(a.before, b.before);
    a.after = std::max(a.after, b.after);
    a.exactDepth = std::max(a.exactDepth, b.exactDepth);
    a.normalError = std::max(a.normalError, b.normalError);
    return a;
}
struct BoundaryMetrics
{
    double error = 0, divergence = 0;
    int faces = 0, partial = 0;
};
inline BoundaryMetrics boundaryMetrics(Solver &s, Obstacle &o)
{
    BoundaryMetrics result;
    auto cells = s.downloadCells();
    int count = 0;
    auto wall = [&](const HostCell &c, int a) {
        // Independently reconstruct prescribed face velocity from the motion,
        // rather than trusting the wall sidecar used to assemble the matrix.
        const Coord q = c.ijk;
        if (q[0] < 0 || q[1] < 0 || q[2] < 0 || q[0] >= s.d.nx || q[1] >= s.d.ny || q[2] >= s.d.nz ||
            q[a] == 0)
            return 0.;
        Vec x(q[0] + .5, q[1] + .5, q[2] + .5);
        x[a] -= .5;
        return double(o.view.motion.velocity(x * s.d.h)[a]);
    };
    auto flux = [&](const HostCell &c, int a) { return c.open[a] * c.u[a] + (1 - c.open[a]) * wall(c, a); };
    for (size_t i = 1; i < cells.size(); ++i) {
        const auto &c = cells[i];
        if (c.fluid) {
            double net = 0;
            for (int a = 0; a < 3; ++a)
                net += flux(cells[c.neighbor[2 * a + 1]], a) - flux(c, a);
            result.divergence += net * net / (s.d.h * s.d.h);
            ++count;
        }
        for (int a = 0; a < 3; ++a) {
            if (c.open[a] == 1 || !(c.fluid || cells[c.neighbor[2 * a]].fluid))
                continue;
            double trace = c.open[a] == 0 ? c.u[a] : c.wall[a];
            result.error = std::max(result.error, fabs(trace - wall(c, a)));
            ++result.faces;
            result.partial += c.open[a] > 0;
        }
    }
    result.divergence = sqrt(result.divergence / std::max(count, 1));
    if (result.error > 1e-8 || result.divergence > 1e-6)
        throw std::runtime_error("independent moving boundary flux validation failed");
    return result;
}
} // namespace fluid
#endif
