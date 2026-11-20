// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#include <openvdb/openvdb.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/tools/LevelSetRebuild.h>
#include <openvdb/tools/LevelSetFilter.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <tbb/task_arena.h>
#include <nanovdb/HostBuffer.h>
#include "RawSdf.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <chrono>
#include <unordered_map>
#include <iomanip>
#include <thread>
using V = openvdb::Vec3d;
using M = openvdb::math::Mat3<double>;
using C = openvdb::Coord;
struct Clock
{
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    double ms()
    {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }
};
struct Hash
{
    size_t operator()(const C &c) const
    {
        return size_t(c.x()) * 73856093u ^ size_t(c.y()) * 19349663u ^ size_t(c.z()) * 83492791u;
    }
};
struct P
{
    uint64_t id;
    V x, v;
};
struct Kernel
{
    V x;
    M covariance;
};
int main(int argc, char **argv)
{
    try
    {
        if (argc < 6)
            throw std::runtime_error(
                "usage: mesh_surface CACHE OUTPUT MESH_H PARTICLE_VOLUME OBSTACLE_H "
                "[ISOVALUE=.5] [SUPPORT=2.2] [SMOOTH=.12] [FIRST=0] [LAST=100000] [CLOSED_SIDES=0] "
                "[FLOOR_BOUNDARY=0] [WAIT_FOR_FRAMES=0] [CLOSING_RADIUS_VOXELS=0] [PCA_THREADS=4]");
        std::string folder = argv[1], out = argv[2];
        double h = std::stod(argv[3]), vp = std::stod(argv[4]), obstacleH = std::stod(argv[5]),
               iso = argc > 6 ? std::stod(argv[6]) : .5, support = argc > 7 ? std::stod(argv[7]) : 2.2,
               smoothing = argc > 8 ? std::stod(argv[8]) : .12;
        int first = argc > 9 ? std::stoi(argv[9]) : 0, last = argc > 10 ? std::stoi(argv[10]) : 100000;
        bool closedSides = argc > 11 && std::stoi(argv[11]) != 0;
        bool floorBoundary = argc > 12 && std::stoi(argv[12]) != 0;
        bool waitForFrames = argc > 13 && std::stoi(argv[13]) != 0;
        const double closingRadius = argc > 14 ? std::stod(argv[14]) : 0;
        const int pcaThreads = argc > 15 ? std::stoi(argv[15]) : 4;
        if (pcaThreads < 1) throw std::runtime_error("PCA_THREADS must be positive");
        tbb::task_arena pcaArena(pcaThreads);
        if (!std::isfinite(closingRadius) || closingRadius < 0 || closingRadius > 1)
            throw std::runtime_error("closing radius must be in [0,1] meshing voxels");
        double dx = cbrt(vp), search = 3 * dx;
        if (h <= 0 || vp <= 0 || iso <= 0 || support < 1)
            throw std::runtime_error("invalid meshing parameters");
        openvdb::initialize();
        std::filesystem::create_directories(out);
        std::ofstream report(out + "/meshing.csv");
        report << "frame,particles,vertices,triangles,material_volume,kernel_integral,clipped_integral,mesh_"
                  "volume,volume_ratio,relative_volume_change,max_anisotropy,max_smoothing,isolated,read_ms,"
                  "pca_ms,grid_ms,clip_ms,contour_ms,write_ms,foam_projected,max_foam_render_shift,"
                  "foam_total,foam_buried_before,foam_buried_after,floor_halfspace_integral,"
                  "filter_ms,unfiltered_mesh_volume,filter_volume_ratio\n";
        std::unordered_map<uint64_t, Kernel> previous;
        double previousRatio = 0;
        for (int frame = first; frame <= last; ++frame)
        {
            if (waitForFrames)
            {
                Clock waiting;
                while (true)
                {
                    // Atomic checkpoint publication happens after every CSV
                    // and SDF output is closed. FLUID003/004 share this prefix.
                    std::ifstream checkpoint(folder + "/checkpoint.bin", std::ios::binary);
                    uint64_t magic = 0;
                    int32_t nextFrame = 0;
                    checkpoint.read(reinterpret_cast<char *>(&magic), sizeof(magic));
                    checkpoint.seekg(152);
                    checkpoint.read(reinterpret_cast<char *>(&nextFrame), sizeof(nextFrame));
                    if (checkpoint && (magic == 0x464c554944303033ULL || magic == 0x464c554944303034ULL) &&
                        nextFrame > frame)
                        break;
                    if (waiting.ms() > 3600000)
                        throw std::runtime_error("timed out waiting for committed simulation frame " +
                                                 std::to_string(frame));
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
            }
            char tag[32];
            std::snprintf(tag, sizeof(tag), "%06d", frame);
            std::string base = folder + "/frame_" + tag;
            std::ifstream input(base + ".csv");
            if (!input)
                break;
            Clock read;
            std::string line;
            std::getline(input, line);
            std::vector<P> p;
            while (std::getline(input, line))
            {
                std::replace(line.begin(), line.end(), ',', ' ');
                std::istringstream row(line);
                P q;
                row >> q.id >> q.x[0] >> q.x[1] >> q.x[2] >> q.v[0] >> q.v[1] >> q.v[2];
                if (!row)
                    throw std::runtime_error("invalid particle row");
                p.push_back(q);
            }
            std::ifstream raw(base + "_obstacle.grid", std::ios::binary | std::ios::ate);
            if (!raw)
                throw std::runtime_error("missing obstacle grid");
            auto bytes = nanovdb::HostBuffer::create(raw.tellg());
            raw.seekg(0);
            raw.read((char *)bytes.data(), bytes.size());
            surface::RawSdf sdf;
            sdf.grid = reinterpret_cast<const surface::Grid *>(bytes.data());
            sdf.h = obstacleH;
            sdf.band = 3 * obstacleH;
            if (!raw || sdf.grid->gridSize() != bytes.size() ||
                fabs(sdf.grid->voxelSize()[0] - obstacleH) > 1e-9 || sdf.grid->blindDataCount() < 4)
                throw std::runtime_error("invalid obstacle cache or mismatched world transform");

            double readMs = read.ms();
            Clock pca;
            std::unordered_map<C, std::vector<int>, Hash> bins;
            auto cell = [&](V x)
            { return C(int(floor(x[0] / search)), int(floor(x[1] / search)), int(floor(x[2] / search))); };
            for (size_t i = 0; i < p.size(); ++i)
                bins[cell(p[i].x)].push_back(i);
            struct Shape
            {
                V x, axis[3], r;
                double amplitude;
            };
            std::vector<Shape> shapes(p.size());
            std::unordered_map<uint64_t, Kernel> next;
            next.reserve(p.size());
            std::vector<Kernel *> destinations(p.size());
            for (size_t i = 0; i < p.size(); ++i) {
                const auto inserted = next.emplace(p[i].id, Kernel{});
                if (!inserted.second) throw std::runtime_error("duplicate particle ID in surface cache");
                destinations[i] = &inserted.first->second;
            }
            // Bucket membership, neighbor order, covariance history and scalar
            // field accumulation are unchanged. Only independent particles run
            // concurrently; each has its own destination and scratch storage.
            const auto &spatialBins = bins;
            const auto &history = previous;
            struct PcaStats {
                double maxAnisotropy = 1, maxSmooth = 0;
                int isolated = 0;
            };
            const auto stats = pcaArena.execute([&] {
                return tbb::parallel_reduce(tbb::blocked_range<size_t>(0, p.size(), 1024), PcaStats{},
                    [&](const tbb::blocked_range<size_t> &range, PcaStats stats) {
                        for (size_t i = range.begin(); i < range.end(); ++i)
                        {
                            auto q = p[i];
                            C c = cell(q.x);
                            double sum = 0;
                            V mean(0);
                            M cov = M::zero();
                            int neighbors = 0;
                            std::vector<std::pair<int, double>> near;
                            for (int x = -1; x <= 1; ++x)
                                for (int y = -1; y <= 1; ++y)
                                    for (int z = -1; z <= 1; ++z)
                                    {
                                        auto it = spatialBins.find(c + C(x, y, z));
                                        if (it == spatialBins.end())
                                            continue;
                                        for (int j : it->second)
                                        {
                                            V delta = p[j].x - q.x;
                                            double rr = delta.lengthSqr() / (search * search);
                                            if (rr >= 1)
                                                continue;
                                            double w = pow(1 - rr, 3);
                                            sum += w;
                                            mean += p[j].x * w;
                                            near.emplace_back(j, w);
                                            ++neighbors;
                                        }
                                    }
                            mean /= sum;
                            for (auto [j, w] : near)
                            {
                                V r = p[j].x - mean;
                                for (int a = 0; a < 3; ++a)
                                    for (int b = 0; b < 3; ++b)
                                        cov(a, b) += w * r[a] * r[b] / sum;
                            }
                            V rendered = q.x + (mean - q.x) * smoothing;
                            auto prior = history.find(q.id);
                            if (prior != history.end())
                                cov = cov * .7 + prior->second.covariance * .3;
                            *destinations[i] = {rendered, cov};
                            stats.maxSmooth = std::max(stats.maxSmooth, (rendered - q.x).length());
                            if (floorBoundary)
                            {
                                // At a wall, vertically collapsed particle neighborhoods
                                // must not become a detached, arbitrarily thin PCA sheet.
                                // Regularize only the wall-normal variance; keep measured
                                // tangential anisotropy and the original simulation data.
                                const double w = std::clamp(2. - q.x[1] / (support * dx), 0., 1.);
                                cov(1, 1) += w * std::max(0., std::max(cov(0, 0), cov(2, 2)) - cov(1, 1));
                            }
                            M eigenvectors;
                            V eigenvalues;
                            openvdb::math::diagonalizeSymmetricMatrix(cov, eigenvectors, eigenvalues);
                            V r(support * dx);
                            if (neighbors >= 8)
                            {
                                double maximum = std::max({eigenvalues[0], eigenvalues[1], eigenvalues[2], 1e-12});
                                for (int a = 0; a < 3; ++a)
                                    r[a] = sqrt(std::max(eigenvalues[a], maximum / 9.));
                                double scale = support * dx / cbrt(r[0] * r[1] * r[2]);
                                r *= scale;
                            }
                            double amp = vp * 315 / (64 * M_PI * r[0] * r[1] * r[2]);
                            if (neighbors <= 2)
                            {
                                ++stats.isolated;
                                r = V(1.6 * dx);
                                double sphere = cbrt(3 * vp / (4 * M_PI));
                                amp = iso / pow(1 - pow(sphere / r[0], 2), 3);
                            }
                            double minR = std::min({r[0], r[1], r[2]}), maxR = std::max({r[0], r[1], r[2]});
                            stats.maxAnisotropy = std::max(stats.maxAnisotropy, maxR / minR);
                            Shape shape;
                            shape.x = rendered;
                            shape.r = r;
                            shape.amplitude = amp;
                            for (int a = 0; a < 3; ++a)
                                shape.axis[a] = eigenvectors.col(a);
                            shapes[i] = shape;
                        }
                        return stats;
                    }, [](PcaStats a, const PcaStats &b) {
                        a.maxAnisotropy = std::max(a.maxAnisotropy, b.maxAnisotropy);
                        a.maxSmooth = std::max(a.maxSmooth, b.maxSmooth);
                        a.isolated += b.isolated;
                        return a;
                    });
            });
            const double maxAnisotropy = stats.maxAnisotropy, maxSmooth = stats.maxSmooth;
            const int isolated = stats.isolated;
            previous.swap(next);
            double pcaMs = pca.ms();
            Clock gridClock;
            auto grid = openvdb::FloatGrid::create(0.f);
            grid->setTransform(openvdb::math::Transform::createLinearTransform(h));
            grid->setName("anisotropic_particle_density");
            grid->setGridClass(openvdb::GRID_FOG_VOLUME);
            auto acc = grid->getAccessor();
            for (const auto &s : shapes)
            {
                V extent(0);
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b)
                        extent[b] += pow(s.r[a] * s.axis[a][b], 2);
                for (int b = 0; b < 3; ++b)
                    extent[b] = sqrt(extent[b]);
                C lo(int(floor((s.x[0] - extent[0]) / h)), int(floor((s.x[1] - extent[1]) / h)),
                     int(floor((s.x[2] - extent[2]) / h))),
                    hi(int(ceil((s.x[0] + extent[0]) / h)), int(ceil((s.x[1] + extent[1]) / h)),
                       int(ceil((s.x[2] + extent[2]) / h)));
                for (int x = lo.x(); x <= hi.x(); ++x)
                    for (int y = lo.y(); y <= hi.y(); ++y)
                        for (int z = lo.z(); z <= hi.z(); ++z)
                        {
                            V delta = V(x * h, y * h, z * h) - s.x;
                            double r = 0;
                            for (int a = 0; a < 3; ++a)
                                r += pow(delta.dot(s.axis[a]) / s.r[a], 2);
                            if (r < 1)
                            {
                                C c(x, y, z);
                                acc.setValue(c, acc.getValue(c) + float(s.amplitude * pow(1 - r, 3)));
                            }
                        }
            }
            double gridMs = gridClock.ms();
            Clock clip;
            double integral = 0, clipped = 0;
            std::vector<std::pair<C, float>> reflected;
            for (auto it = grid->cbeginValueOn(); it; ++it)
            {
                integral += *it * h * h * h;
                C c = it.getCoord();
                if (floorBoundary && c.y() <= 0)
                    reflected.emplace_back(C(c.x(), -c.y(), c.z()), *it);
            }
            // Even extension at the floor returns the part of each normalized
            // kernel inside the solid to the liquid half-space. The unbounded
            // kernel integral above and half-space trapezoid sum below are
            // recorded independently; the plane's quadrature weight is 1/2.
            for (const auto &q : reflected)
                acc.setValue(q.first, acc.getValue(q.first) + q.second);
            double floorIntegral = 0;
            for (auto it = grid->beginValueOn(); it; ++it)
            {
                auto c = it.getCoord();
                double v = *it;
                if (!floorBoundary || c.y() >= 0)
                    floorIntegral += v * h * h * h * (floorBoundary && c.y() == 0 ? .5 : 1.);
                double phi = sdf.sdf(surface::Vec(c.x() * h, c.y() * h, c.z() * h));
                v = std::min({v, std::max(0., iso + phi / h), std::max(0., iso + double(c.y()))});
                if (closedSides)
                {
                    double side = std::min({c.x() * h, 3.2 - c.x() * h, c.z() * h, 2.4 - c.z() * h});
                    v = std::min(v, std::max(0., iso + side / h));
                }
                it.setValue(float(v));
                clipped += v * h * h * h;
            }
            double clipMs = clip.ms();
            auto surfaceGrid = grid;
            double surfaceIso = iso, filterMs = 0, unfilteredVolume = 0;
            if (closingRadius > 0) {
                Clock filtering;
                std::vector<openvdb::Vec3s> rawPoints;
                std::vector<openvdb::Vec3I> rawTriangles;
                std::vector<openvdb::Vec4I> rawQuads;
                openvdb::tools::volumeToMesh(*grid, rawPoints, rawTriangles, rawQuads, iso, 0.0);
                auto triangleVolume = [&](int a, int b, int c) {
                    return V(rawPoints[a]).dot(V(rawPoints[b]).cross(V(rawPoints[c]))) / 6;
                };
                for (auto q : rawTriangles) unfilteredVolume += triangleVolume(q[0], q[1], q[2]);
                for (auto q : rawQuads)
                    unfilteredVolume += triangleVolume(q[0], q[1], q[2]) + triangleVolume(q[0], q[2], q[3]);
                unfilteredVolume = fabs(unfilteredVolume);
                // The APIC reconstruction is density, not a signed-distance field.
                // Rebuild its chosen isosurface before using level-set operations.
                surfaceGrid = openvdb::tools::levelSetRebuild(*grid, float(iso), 3.f);
                openvdb::tools::LevelSetFilter<openvdb::FloatGrid> filter(*surfaceGrid);
                auto strength = openvdb::FloatGrid::create(.5f);
                strength->setTransform(surfaceGrid->transform().copy());
                filter.offset(float(-closingRadius * h)); // Negative-inside: dilate.
                filter.meanCurvature(strength.get());      // One half-strength step.
                filter.offset(float(closingRadius * h));  // Erode by the same distance.
                // Restore the solid half-spaces after filtering the render field.
                for (auto it = surfaceGrid->beginValueOn(); it; ++it) {
                    const auto x = surfaceGrid->indexToWorld(it.getCoord());
                    double value = std::max({double(*it), -sdf.sdf(surface::Vec(x[0], x[1], x[2])), -x[1]});
                    if (closedSides) value = std::max({value, -x[0], x[0] - 3.2, -x[2], x[2] - 2.4});
                    it.setValue(float(value));
                }
                surfaceIso = 0;
                filterMs = filtering.ms();
            }
            Clock contour;
            std::vector<openvdb::Vec3s> points;
            std::vector<openvdb::Vec3I> triangles;
            std::vector<openvdb::Vec4I> quads;
            openvdb::tools::volumeToMesh(*surfaceGrid, points, triangles, quads, surfaceIso, 0.0);
            for (auto q : quads)
            {
                triangles.emplace_back(q[0], q[1], q[2]);
                triangles.emplace_back(q[0], q[2], q[3]);
            }
            double signedVolume = 0;
            for (auto tri : triangles)
            {
                V a(points[tri[0]]), b(points[tri[1]]), c(points[tri[2]]);
                signedVolume += a.dot(b.cross(c)) / 6;
            }
            if (signedVolume < 0)
                for (auto &tri : triangles)
                    std::swap(tri[1], tri[2]);
            double meshVolume = fabs(signedVolume), ratio = meshVolume / (p.size() * vp),
                   change = previousRatio ? fabs(ratio - previousRatio) / previousRatio : 0;
            previousRatio = ratio;
            if (closingRadius == 0) unfilteredVolume = meshVolume;
            double contourMs = contour.ms();
            Clock write;
            std::ofstream obj(out + "/water_" + tag + ".obj");
            obj << std::setprecision(8);
            for (auto x : points)
                obj << "v " << x[0] << ' ' << x[1] << ' ' << x[2] << '\n';
            for (auto q : triangles)
                obj << "f " << q[0] + 1 << ' ' << q[1] + 1 << ' ' << q[2] + 1 << '\n';
            obj.close();
            // Fast local renderer cache; the OBJ remains the audited export.
            static_assert(sizeof(openvdb::Vec3s) == 12 && sizeof(openvdb::Vec3I) == 12);
            std::ofstream binary(out + "/water_" + tag + ".meshbin", std::ios::binary);
            uint64_t header[3] = {0x4d45534830303031ULL, points.size(), triangles.size()};
            binary.write(reinterpret_cast<const char *>(header), sizeof(header));
            binary.write(reinterpret_cast<const char *>(points.data()), points.size() * 12);
            binary.write(reinterpret_cast<const char *>(triangles.data()), triangles.size() * 12);
            binary.close();
            // Match visible foam to the independently reconstructed surface. This changes
            // only the rendering cache, never simulation/secondary trajectories.
            int foamProjected = 0;
            int foamTotal = 0, foamBuriedBefore = 0, foamBuriedAfter = 0;
            double maxFoamShift = 0;
            std::ifstream ww(base + "_whitewater.csv");
            std::ofstream renderWW(out + "/whitewater_" + tag + ".csv");
            std::getline(ww, line);
            renderWW << line << '\n';
            auto scalar = [&](V x)
            {
                const double value = openvdb::tools::BoxSampler::sample(surfaceGrid->tree(), x / h);
                // Keep the density-style sign convention used by foam placement.
                return closingRadius > 0 ? iso - value / h : value;
            };
            while (std::getline(ww, line))
            {
                std::istringstream row(line);
                std::string item;
                std::vector<std::string> fields;
                while (std::getline(row, item, ','))
                    fields.push_back(item);
                if (fields.size() != 11)
                    throw std::runtime_error("invalid whitewater CSV");
                V original(std::stod(fields[1]), std::stod(fields[2]), std::stod(fields[3])), x = original;
                double radius = std::stod(fields[7]);
                if (std::stoi(fields[10]) == 0)
                {
                    ++foamTotal;
                    foamBuriedBefore += scalar(original) > iso;
                    V normal(0);
                    for (int k = 0; k < 10; ++k)
                    {
                        V grad;
                        for (int a = 0; a < 3; ++a)
                        {
                            V e(0);
                            e[a] = h * .5;
                            grad[a] = (scalar(x + e) - scalar(x - e)) / h;
                        }
                        double length = grad.length();
                        if (length < 1e-6)
                            break;
                        normal = -grad / length;
                        double delta = std::clamp((scalar(x) - iso) / length, -h * .5, h * .5);
                        x += normal * delta;
                        if (fabs(delta) < 1e-5)
                            break;
                    }
                    x += normal * (radius * .7);
                    auto allowed = [&](V v)
                    {
                        return (v - original).length() <= .06 && scalar(v) <= iso &&
                               (!closedSides || (v[0] >= radius && v[0] <= 3.2 - radius && v[2] >= radius &&
                                                 v[2] <= 2.4 - radius)) &&
                               sdf.sdf(surface::Vec(v[0], v[1], v[2])) > radius + obstacleH * .5 &&
                               v[1] >= radius;
                    };
                    if (!allowed(x))
                        x = original;
                    // Near a clipped solid surface Newton may choose the solid-side
                    // interface. Look for a nearby fluid-air exit instead, bounded
                    // to 6 cm and changing only the render copy.
                    if (scalar(x) > iso)
                    {
                        double best = .06;
                        for (int a = 0; a < 3; ++a)
                            for (int sign : {-1, 1})
                            {
                                V direction(0);
                                direction[a] = sign;
                                V lo = original;
                                for (double distance = h * .5; distance <= .06; distance += h * .5)
                                {
                                    V hi = original + direction * distance;
                                    if (scalar(hi) <= iso)
                                    {
                                        for (int j = 0; j < 10; ++j)
                                        {
                                            V mid = (lo + hi) * .5;
                                            if (scalar(mid) > iso)
                                                lo = mid;
                                            else
                                                hi = mid;
                                        }
                                        V candidate = hi + direction * (radius * .9);
                                        double shift = (candidate - original).length();
                                        if (allowed(candidate) && shift < best)
                                        {
                                            best = shift;
                                            x = candidate;
                                        }
                                        break;
                                    }
                                    lo = hi;
                                }
                            }
                    }
                    double shift = (x - original).length();
                    if (shift > 1e-8)
                    {
                        ++foamProjected;
                        maxFoamShift = std::max(maxFoamShift, shift);
                    }
                    foamBuriedAfter += scalar(x) > iso;
                }
                renderWW << fields[0] << ',' << std::setprecision(10) << x[0] << ',' << x[1] << ',' << x[2];
                for (size_t j = 4; j < fields.size(); ++j)
                    renderWW << ',' << fields[j];
                renderWW << '\n';
            }
            renderWW.close();
            double writeMs = write.ms();
            report << std::setprecision(12) << frame << ',' << p.size() << ',' << points.size() << ','
                   << triangles.size() << ',' << p.size() * vp << ',' << integral << ',' << clipped << ','
                   << meshVolume << ',' << ratio << ',' << change << ',' << maxAnisotropy << ',' << maxSmooth
                   << ',' << isolated << ',' << readMs << ',' << pcaMs << ',' << gridMs << ',' << clipMs
                   << ',' << contourMs << ',' << writeMs << ',' << foamProjected << ',' << maxFoamShift << ','
                   << foamTotal << ',' << foamBuriedBefore << ',' << foamBuriedAfter << ',' << floorIntegral
                   << ',' << filterMs << ',' << unfilteredVolume << ',' << meshVolume / unfilteredVolume
                   << '\n';
            report.flush();
            std::cout << "MESH " << frame << " triangles=" << triangles.size() << " volume ratio=" << ratio
                      << " filter volume ratio=" << meshVolume / unfilteredVolume
                      << " ms=" << readMs + pcaMs + gridMs + clipMs + filterMs + contourMs + writeMs << std::endl;
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
