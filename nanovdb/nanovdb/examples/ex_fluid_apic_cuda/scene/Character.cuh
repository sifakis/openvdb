// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifndef FLUID_SCENE_CHARACTER_CUH
#define FLUID_SCENE_CHARACTER_CUH
#include "../Obstacle.cuh"
#include <algorithm>
#include <cuda_profiler_api.h>
#include <limits>
#include <numeric>

namespace fluid
{
struct BvhNode
{
    nanovdb::Vec3f lo, hi;
    int left = -1, right = -1, start = 0, count = 0;
};
struct MeshView
{
    ObstacleView field;
    const nanovdb::Vec3f *vertices = nullptr, *velocities = nullptr;
    const nanovdb::Vec3i *triangles = nullptr;
    const BvhNode *nodes = nullptr;
    const int *order = nullptr, *components = nullptr;
    int componentCount = 0;
    __device__ double boxDistance(GeometryVec p, const BvhNode &n) const
    {
        double d = 0;
        for (int a = 0; a < 3; ++a)
        {
            double x = maxValue(maxValue(n.lo[a] - p[a], p[a] - n.hi[a]), 0.);
            d += x * x;
        }
        return d;
    }
    struct Nearest
    {
        double distance = 1e30;
        GeometryVec point{0}, normal{0}, velocity{0};
        int triangle = -1;
    };
    __device__ __noinline__ Nearest nearest(GeometryVec p) const
    {
        Nearest best;
        int stack[64], size = 1;
        stack[0] = 0;
        while (size)
        {
            int idx = stack[--size];
            const auto &node = nodes[idx];
            if (boxDistance(p, node) > (best.distance + 1e-10) * (best.distance + 1e-10))
                continue;
            if (node.count)
            {
                for (int j = 0; j < node.count; ++j)
                {
                    int t = order[node.start + j];
                    auto ids = triangles[t];
                    GeometryVec a(vertices[ids[0]]), b(vertices[ids[1]]), c(vertices[ids[2]]);
                    double u, v;
                    GeometryVec q = nanovdb::math::closestPointOnTriangleToPoint(a, b, c, p, u, v);
                    double distance = (q - p).length();
                    if (distance < best.distance - 1e-10 ||
                        (fabs(distance - best.distance) <= 1e-10 && t < best.triangle))
                    {
                        best.distance = distance;
                        best.point = q;
                        best.triangle = t;
                        GeometryVec normal = (b - a).cross(c - a);
                        double len = normal.length();
                        best.normal = len > 1e-14 ? normal / len : GeometryVec(0);
                        best.velocity = GeometryVec(velocities[ids[0]]) * (1 - u - v) + GeometryVec(velocities[ids[1]]) * u +
                                        GeometryVec(velocities[ids[2]]) * v;
                    }
                }
            }
            else
            {
                double a = boxDistance(p, nodes[node.left]), b = boxDistance(p, nodes[node.right]);
                if (a < b)
                {
                    stack[size++] = node.right;
                    stack[size++] = node.left;
                }
                else
                {
                    stack[size++] = node.left;
                    stack[size++] = node.right;
                }
            }
        }
        return best;
    }
    __device__ GeometryVec velocity(GeometryVec p) const
    {
        return nearest(p).velocity;
    }
    __device__ bool resolveInterior(GeometryVec &p, double margin, bool includeSdf = false) const
    {
        if (!inside(p) && (!includeSdf || field.sdf(p) >= 0))
            return false;
        auto q = nearest(p);
        GeometryVec direction = q.normal;
        if (direction.length() < .5)
            direction = GeometryVec(0, 1, 0);
        // Search the nearest-normal direction and independent local directions
        // at each radius. A concave component intersection can block the entire
        // normal line, even when a nearby lateral exit exists.
        const GeometryVec origin = p;
        for (int shell = 0; shell < 16; ++shell) {
            const double radius = margin * pow(1.4, shell);
            double bestDistance = 1e30;
            GeometryVec best;
            for (int k = 0; k < 28; ++k) {
                GeometryVec candidate;
                if (k < 2) {
                    candidate = q.point + direction * (k == 0 ? radius : -radius);
                } else {
                    int slot = k - 2;
                    if (slot >= 13) ++slot; // Skip the zero vector in the 3^3 cube.
                    GeometryVec d(slot / 9 - 1, (slot / 3) % 3 - 1, slot % 3 - 1);
                    candidate = origin + d * (radius / d.length());
                }
                candidate = GeometryVec(Vec(candidate));
                if (field.sdf(candidate) < margin * .25 || inside(candidate)) continue;
                const double distance = (candidate - origin).length();
                if (distance < bestDistance) {bestDistance = distance; best = candidate;}
            }
            if (bestDistance < 1e29) {p = best; return true;}
        }
        return true;
    }

    // A mesh-only push may choose the underside of a foot, below the floor.
    // Enforce the intersection of the two feasible regions. If the small gap
    // cannot hold a particle, search nearby above-floor escape directions.
    __device__ bool resolveFloor(GeometryVec &p, double margin, double floorY) const
    {
        if (p[1] >= floorY)
            return false;
        p[1] = floorY;
        if (field.sdf(p) >= margin && !inside(p))
            return true;
        const GeometryVec origin = p;
        for (int shell = 0; shell < 13; ++shell)
        {
            double r = margin * pow(1.4, shell);
            for (int ring = 0; ring < 3; ++ring)
                for (int k = 0; k < (ring == 2 ? 1 : 16); ++k)
                {
                    double angle = 2 * M_PI * k / 16;
                    double up = ring == 0 ? 0 : ring == 1 ? .7071067811865475 : 1;
                    double lateral = sqrt(1 - up * up);
                    GeometryVec candidate = origin + GeometryVec(lateral * cos(angle), up, lateral * sin(angle)) * r;
                    if (field.sdf(candidate) >= margin && !inside(candidate))
                    {
                        p = candidate;
                        return true;
                    }
                }
        }
        // Keep the unresolved point visible to the independent penetration check.
        return true;
    }

    // Independent union-of-closed-components parity; no SDF/sign sidecar is used.
    __device__ bool inside(GeometryVec p) const
    {
        const auto &root = nodes[0];
        for (int a = 0; a < 3; ++a)
            if (p[a] < root.lo[a] || p[a] > root.hi[a])
                return false;
        const GeometryVec ray(1., .3713906763541037, .1732050807568877);
        uint64_t parity[8] = {0};
        int stack[64], size = 1;
        stack[0] = 0;
        while (size)
        {
            const auto &node = nodes[stack[--size]];
            double lo = 0, hi = 1e30;
            for (int a = 0; a < 3; ++a)
            {
                lo = maxValue(lo, (node.lo[a] - p[a]) / ray[a]);
                hi = minValue(hi, (node.hi[a] - p[a]) / ray[a]);
            }
            if (hi < lo)
                continue;
            if (node.count)
            {
                for (int j = 0; j < node.count; ++j)
                {
                    int t = order[node.start + j];
                    auto ids = triangles[t];
                    GeometryVec v0(vertices[ids[0]]), e1 = GeometryVec(vertices[ids[1]]) - v0,
                                              e2 = GeometryVec(vertices[ids[2]]) - v0;
                    GeometryVec h = ray.cross(e2);
                    double det = e1.dot(h);
                    if (fabs(det) < 1e-12)
                        continue;
                    GeometryVec s = p - v0;
                    double u = s.dot(h) / det;
                    if (u < 0 || u > 1)
                        continue;
                    GeometryVec q = s.cross(e1);
                    double v = ray.dot(q) / det;
                    if (v < 0 || u + v > 1)
                        continue;
                    if (e2.dot(q) / det > 1e-9)
                    {
                        int c = components[t];
                        parity[c / 64] ^= uint64_t(1) << (c % 64);
                    }
                }
            }
            else
            {
                stack[size++] = node.left;
                stack[size++] = node.right;
            }
        }
        for (int k = 0; k < 8; ++k)
            if (parity[k])
                return true;
        return false;
    }
#ifdef FLUID_USE_FLOAT
    // Keep collision queries and animated mesh evaluation independent of the
    // fluid storage precision. Position is rounded only at the adapter boundary.
    __device__ bool resolveInterior(Vec &p,double margin,bool includeSdf=false) const {
        GeometryVec q(p);bool touched=resolveInterior(q,margin,includeSdf);p=Vec(q);return touched;
    }
    __device__ bool resolveFloor(Vec &p,double margin,double floorY) const {
        GeometryVec q(p);bool touched=resolveFloor(q,margin,floorY);p=Vec(q);return touched;
    }
#endif

};
class Character
{
    using V = nanovdb::Vec3f;
    using T = nanovdb::Vec3i;
    using Converter = nanovdb::tools::cuda::MeshToSDF<Build>;
    std::vector<V> sequence, positions, velocity;
    std::vector<T> triangles;
    std::vector<int> order, components;
    std::vector<BvhNode> nodes;
    Buffer<V> devicePositions, deviceVelocity;
    Buffer<T> deviceTriangles;
    Buffer<BvhNode> deviceNodes;
    Buffer<int> deviceOrder, deviceComponents;
    Converter::HandleT handle{nanovdb::tools::cuda::sdf_detail::allocate<std::byte>(0, 0)};
    int buildNode(int begin, int end)
    {
        int id = int(nodes.size());
        nodes.push_back({});
        nodes[id].start = begin;
        nodes[id].count = end - begin;
        refitLeaf(id);
        if (end - begin > 4)
        {
            V extent = nodes[id].hi - nodes[id].lo;
            int a = extent[1] > extent[0] ? 1 : 0;
            if (extent[2] > extent[a])
                a = 2;
            int mid = (begin + end) / 2;
            auto center = [&](int t) {
                auto q = triangles[t];
                return positions[q[0]][a] + positions[q[1]][a] + positions[q[2]][a];
            };
            std::nth_element(order.begin() + begin, order.begin() + mid, order.begin() + end,
                             [&](int x, int y) { return center(x) < center(y); });
            int l = buildNode(begin, mid), r = buildNode(mid, end);
            nodes[id].left = l;
            nodes[id].right = r;
            nodes[id].count = 0;
        }
        return id;
    }
    void refitLeaf(int id)
    {
        auto &n = nodes[id];
        n.lo = V(1e30f);
        n.hi = V(-1e30f);
        for (int j = 0; j < n.count; ++j)
        {
            auto t = triangles[order[n.start + j]];
            for (int k = 0; k < 3; ++k)
                for (int a = 0; a < 3; ++a)
                {
                    n.lo[a] = std::min(n.lo[a], positions[t[k]][a]);
                    n.hi[a] = std::max(n.hi[a], positions[t[k]][a]);
                }
        }
    }
    void refit()
    {
        for (int i = int(nodes.size()) - 1; i >= 0; --i)
        {
            auto &n = nodes[i];
            if (n.count)
                refitLeaf(i);
            else
                for (int a = 0; a < 3; ++a)
                {
                    n.lo[a] = std::min(nodes[n.left].lo[a], nodes[n.right].lo[a]);
                    n.hi[a] = std::max(nodes[n.left].hi[a], nodes[n.right].hi[a]);
                }
        }
    }
    GeometryVec at(int f, int i) const
    {
        return GeometryVec(sequence[std::clamp(f, 0, frameCount - 1) * vertexCount + i]);
    }
    GeometryVec tangent(int f, int i) const
    {
        int lo = std::max(0, f - 1), hi = std::min(frameCount - 1, f + 1);
        return (at(hi, i) - at(lo, i)) * (fps / (hi - lo));
    }

  public:
    MeshView view;
    bool profileBuild = false, traceSdfPipeline = false;
    float sdfPhaseMs[5]{};
    double prepareMs = 0, sdfPipelineMs = 0;
    int frameCount = 0, vertexCount = 0, triangleCount = 0;
    double fps = 30, maxWallSpeed = 0, timeOffset = 0;
    Character(const std::string &directory, double voxel, double offset = 0) : timeOffset(offset)
    {
        std::ifstream info(directory + "/sequence.txt");
        info >> frameCount >> vertexCount >> triangleCount >> fps;
        if (!info || frameCount < 2 || !vertexCount || !triangleCount)
            throw std::runtime_error("invalid character sequence.txt");
        sequence.resize(size_t(frameCount) * vertexCount);
        triangles.resize(triangleCount);
        std::ifstream v(directory + "/vertices.bin", std::ios::binary),
            t(directory + "/triangles.bin", std::ios::binary);
        v.read(reinterpret_cast<char *>(sequence.data()), sequence.size() * sizeof(V));
        t.read(reinterpret_cast<char *>(triangles.data()), triangles.size() * sizeof(T));
        if (!v || !t)
            throw std::runtime_error("truncated character sequence");
        positions.assign(sequence.begin(), sequence.begin() + vertexCount);
        velocity.resize(vertexCount);
        order.resize(triangleCount);
        std::iota(order.begin(), order.end(), 0);
        std::vector<int> parent(vertexCount);
        std::iota(parent.begin(), parent.end(), 0);
        auto find = [&](int i) {
            while (parent[i] != i)
            {
                parent[i] = parent[parent[i]];
                i = parent[i];
            }
            return i;
        };
        for (auto tri : triangles)
            for (int k = 1; k < 3; ++k)
                parent[find(tri[k])] = find(tri[0]);
        std::map<int, int> ids;
        components.resize(triangleCount);
        for (int i = 0; i < triangleCount; ++i)
        {
            int root = find(triangles[i][0]);
            if (!ids.count(root))
                ids[root] = int(ids.size());
            components[i] = ids[root];
        }
        if (ids.size() > 512)
            throw std::runtime_error("character exceeds 512 closed components");
        buildNode(0, triangleCount);
        devicePositions = allocate<V>(vertexCount);
        deviceVelocity = allocate<V>(vertexCount);
        deviceTriangles = allocate<T>(triangleCount);
        deviceNodes = allocate<BvhNode>(nodes.size());
        deviceOrder = allocate<int>(order.size());
        deviceComponents = allocate<int>(components.size());
        cudaCheck(cudaMemcpy(deviceTriangles.data(), triangles.data(), triangles.size() * sizeof(T),
                             cudaMemcpyHostToDevice));
        cudaCheck(
            cudaMemcpy(deviceOrder.data(), order.data(), order.size() * sizeof(int), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(deviceComponents.data(), components.data(), components.size() * sizeof(int),
                             cudaMemcpyHostToDevice));
        view.vertices = devicePositions.data();
        view.velocities = deviceVelocity.data();
        view.triangles = deviceTriangles.data();
        view.nodes = deviceNodes.data();
        view.order = deviceOrder.data();
        view.components = deviceComponents.data();
        view.componentCount = int(ids.size());
        view.field.h = voxel;
        view.field.band = 3 * voxel;
    }
    void evaluate(double time)
    {
        double u = std::clamp((time + timeOffset) * fps, 0., double(frameCount - 1));
        int frame = std::min(int(u), frameCount - 2);
        double s = u - frame;
        maxWallSpeed = 0;
        for (int i = 0; i < vertexCount; ++i)
        {
            GeometryVec a = at(frame, i), b = at(frame + 1, i), m0 = tangent(frame, i) / fps,
                m1 = tangent(frame + 1, i) / fps;
            GeometryVec p = a * (2 * s * s * s - 3 * s * s + 1) + m0 * (s * s * s - 2 * s * s + s) +
                    b * (-2 * s * s * s + 3 * s * s) + m1 * (s * s * s - s * s);
            GeometryVec vel = (a * (6 * s * s - 6 * s) + m0 * (3 * s * s - 4 * s + 1) + b * (-6 * s * s + 6 * s) +
                       m1 * (3 * s * s - 2 * s)) *
                      fps;
            positions[i] = V(p);
            velocity[i] = V(vel);
            maxWallSpeed = std::max<double>(maxWallSpeed, vel.length());
        }
    }
    double build(double time)
    {
        Timer timer;
        evaluate(time);
        refit();
        cudaCheck(cudaMemcpy(devicePositions.data(), positions.data(), positions.size() * sizeof(V),
                             cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(deviceVelocity.data(), velocity.data(), velocity.size() * sizeof(V),
                             cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(deviceNodes.data(), nodes.data(), nodes.size() * sizeof(BvhNode),
                             cudaMemcpyHostToDevice));
        nanovdb::Map map;
        map.set(view.field.h, nanovdb::Vec3d(0), 1.);
        Converter converter(devicePositions.data(), vertexCount, deviceTriangles.data(), triangleCount, map);
        converter.setNarrowBandWidth(3);
        converter.setBarrierSigning(Converter::BarrierSigning::Interior);
        converter.setNestingRule(Converter::NestingRule::Solid);
        if (profileBuild) {
            // Stop preparation before the MeshToSDF pipeline. Timers synchronize
            // so asynchronous kernels cannot spill across the measured boundary.
            prepareMs = timer.ms();
            if (traceSdfPipeline) cudaCheck(cudaProfilerStart());
            Timer pipelineTimer;
            auto result = converter.getHandle();
            sdfPipelineMs = pipelineTimer.ms();
            if (traceSdfPipeline) cudaCheck(cudaProfilerStop());
            std::copy(converter.phaseMs(), converter.phaseMs() + 5, sdfPhaseMs);
            handle = std::move(result);
        } else {
            handle = converter.getHandle();
        }
        view.field.grid = handle.deviceGrid<Build>();
        return timer.ms();
    }
    std::vector<char> downloadGrid()
    {
        nanovdb::GridData data;
        cudaCheck(cudaMemcpy(&data, view.field.grid, sizeof(data), cudaMemcpyDeviceToHost));
        std::vector<char> bytes(data.mGridSize);
        cudaCheck(cudaMemcpy(bytes.data(), view.field.grid, bytes.size(), cudaMemcpyDeviceToHost));
        return bytes;
    }
    double validateDerivative(double time)
    {
        evaluate(time);
        auto reference = velocity;
        evaluate(time + 1e-4);
        auto plus = positions;
        evaluate(time - 1e-4);
        auto minus = positions;
        double error = 0;
        for (int i = 0; i < vertexCount; ++i)
            error = std::max<double>(error, ((GeometryVec(plus[i]) - GeometryVec(minus[i])) / (2e-4) - GeometryVec(reference[i])).length());
        evaluate(time);
        return error;
    }
};
__global__ void characterFaces(Domain d, MeshView mesh)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (!i || i >= d.count)
        return;
    Cell &c = d.cells[i];
    for (int a = 0; a < 3; ++a)
    {
        if (!d.inside(d.coord(i)) || !d.inside(d.coord(i) - axis(a)))
            continue;
        auto &lo = d.cells[d.neighbor(i, 2 * a)];
        Vec x = face(d.coord(i), a) * d.h;
        // Cell-center signs do not describe a cut face. In particular, the
        // old "either center is solid" rule falsely sealed liquid under a foot.
        // Only use center distances for conservative far-from-interface exits.
        if (maxValue(c.phi, lo.phi) < -d.h)
            c.open[a] = 0;
        else if (minValue(c.phi, lo.phi) > d.h)
            c.open[a] = 1;
        else
        {
            int open = 0;
            for (int u = 0; u < 4; ++u)
                for (int v = 0; v < 4; ++v)
                {
                    Vec p = x;
                    p[(a + 1) % 3] += ((u + .5) / 4 - .5) * d.h;
                    p[(a + 2) % 3] += ((v + .5) / 4 - .5) * d.h;
                    open += mesh.field.sdf(p) > 0;
                }
            c.open[a] = open / 16.;
        }
        if (c.open[a] < 1)
            c.wall[a] = mesh.velocity(x)[a];
    }
}
// Minimum-change projection onto mesh and (when active) stationary-floor
// velocity half-spaces. Sequential clamping can undo the other contact.
__device__ Vec contactVelocity(Vec v, Vec normal, Vec wall, bool floorActive)
{
    double b = normal.dot(wall);
    Vec meshOnly = v + normal * maxValue(0., b - normal.dot(v));
    if (!floorActive || meshOnly[1] >= 0)
        return meshOnly;
    Vec floorOnly = v;
    floorOnly[1] = 0;
    if (normal.dot(floorOnly) >= b)
        return floorOnly;
    double horizontal = normal[0] * normal[0] + normal[2] * normal[2];
    if (horizontal > 1e-12)
    {
        double delta = (b - normal.dot(floorOnly)) / horizontal;
        return floorOnly + Vec(normal[0], 0, normal[2]) * delta;
    }
    return meshOnly; // Infeasible coincident planes remain a validation failure.
}
struct SceneContact
{
    int sdfRemaining = 0, meshRemaining = 0, contacts = 0, attempted = 0;
    double depth = 0, normalError = 0, impulse = 0;
};
struct ContactCombine
{
    __hostdev__ SceneContact operator()(SceneContact a,SceneContact b) const {
        a.sdfRemaining+=b.sdfRemaining;a.meshRemaining+=b.meshRemaining;a.contacts+=b.contacts;a.attempted+=b.attempted;
        a.depth=maxValue(a.depth,b.depth);a.normalError=maxValue(a.normalError,b.normalError);a.impulse=maxValue(a.impulse,b.impulse);return a;
    }
};
__global__ void characterContact(Particle *particles, int n, MeshView mesh, double margin,
                                 SceneContact *results, bool independent, double floorY = 0, bool compact = false)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    SceneContact result;
    if(k<n) {
    Particle &p = particles[k];
    double phi = mesh.field.sdf(p.x);
    result.attempted = phi < 0;
    bool touched = false;
    for (int j = 0; j < 16 && phi < margin; ++j)
    {
        Vec normal = mesh.field.normal(p.x);
        if (normal.length() < .5) break;
        const Vec origin = p.x;
        double distance = minValue(margin - phi, mesh.field.band);
        bool improved = false;
        for (int trial = 0; trial < 8; ++trial) {
            const Vec candidate = origin + normal * distance;
            const double candidatePhi = mesh.field.sdf(candidate);
            if (candidatePhi > phi + 1e-9) {
                p.x = candidate; phi = candidatePhi; improved = true; break;
            }
            distance *= .5;
        }
        if (!improved) break;
        touched = true;
    }
    // Concave/intersecting components can leave the interpolated SDF negative
    // after the bounded gradient iterations even outside the triangle union.
    // Enforce both representations, as the secondary-particle path already does.
    touched = mesh.resolveInterior(p.x, margin, true) || touched;
    const bool floorMoved = mesh.resolveFloor(p.x, margin, floorY);
    if(floorMoved && !touched && p.x[1]<=floorY+contactTolerance && p.v[1]<0) {
        p.v[1]=0;p.c[1]=Vec(0);
    }
    // A floor-only correction must NOT activate a distant character's wall
    // velocity. Float roundoff at the tank margin made this latent bug frequent.
    // Preserve the mesh-contact flag separately from a floor clamp.
    phi = mesh.field.sdf(p.x);
    if (touched)
    {
        auto q = mesh.nearest(p.x);
        Vec normal = mesh.field.normal(p.x);
        if (normal.length() < .5)
            normal = q.normal;
        double relative = (p.v - q.velocity).dot(normal);
        // Unilateral, frictionless contact: a separating particle may leave.
        result.impulse = maxValue(0., -relative);
        if (relative < 0)
        {
            p.v = contactVelocity(p.v, normal, q.velocity, p.x[1] <= floorY + 1e-10);
            // Remove only the normal output of C. Preserve tangential shear/spin.
            Vec normalRow = p.c[0] * normal[0] + p.c[1] * normal[1] + p.c[2] * normal[2];
            for (int a = 0; a < 3; ++a)
                p.c[a] -= normalRow * normal[a];
        }
        if (p.x[1] <= floorY + 1e-10)
        {
            p.v = contactVelocity(p.v, normal, q.velocity, true);
            if (p.v[1] <= 0)
                p.c[1] = Vec(0);
        }
        result.contacts = 1;
        result.normalError = maxValue(0., -(p.v - q.velocity).dot(normal));
    }
    result.sdfRemaining = phi < -contactTolerance;
    result.depth = maxValue(0., -phi);
    if (independent && mesh.inside(p.x))
    {
        auto q = mesh.nearest(p.x);
        if (q.distance > 1e-6)
        {
            result.meshRemaining = 1;
            result.depth = maxValue(result.depth, q.distance);
        }
    }
    if (independent && (result.sdfRemaining || result.meshRemaining)) {
        auto q = mesh.nearest(p.x);
        printf("CONTACT DETAIL particle=%d x=(%.17g,%.17g,%.17g) sdf=%.17g mesh=%d "
               "distance=%.17g triangle=%d normal=(%g,%g,%g) margin=%g floor=%g touched=%d floorMoved=%d\n",
               k, double(p.x[0]), double(p.x[1]), double(p.x[2]), phi, result.meshRemaining,
               q.distance, q.triangle, q.normal[0], q.normal[1], q.normal[2], margin, floorY,
               int(touched), int(floorMoved));
    }
    }
    if(compact) {
        using Reduce=cub::BlockReduce<SceneContact,256>;
        __shared__ typename Reduce::TempStorage temporary;
        auto combined=Reduce(temporary).Reduce(result,ContactCombine{});
        if(!threadIdx.x && results)results[blockIdx.x]=combined;
    } else if(k<n && results) results[k]=result;
}
} // namespace fluid
#endif
