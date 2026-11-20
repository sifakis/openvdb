// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#include "Whitewater.cuh"

#include "Checkpoint.h"
#include "../scalable/ParticleOps.cuh"
#include <iostream>
using namespace fluid;
__global__ void initialPoolMask(MeshView mesh, const Particle *p, int n, double margin, int *keep)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        keep[i] = mesh.field.sdf(p[i].x) >= margin && !mesh.inside(p[i].x);
}
__global__ void wallCheck(Domain d, MeshView mesh, double *error)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (!i || i >= d.count)
        return;
    double e = 0;
    for (int a = 0; a < 3; ++a)
        if (d.cells[i].open[a] < 1 && d.inside(d.coord(i)) && d.inside(d.coord(i) - axis(a)))
        {
            double wall = mesh.velocity(face(d.coord(i), a) * d.h)[a];
            e = fmax(e, fabs((1 - d.cells[i].open[a]) * (d.cells[i].wall[a] - wall)));
            if (d.cells[i].open[a] == 0)
                e = fmax(e, fabs(d.cells[i].u[a] - wall));
        }
    error[i] = e;
}
__global__ void speedGuard(Domain d, const Particle *particles, int n, double limit, double time, int phase,
                           int *errors)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n || particles[k].v.length() <= limit)
        return;
    int slot = atomicAdd(errors, 1);
    if (slot >= 4)
        return;
    auto p = particles[k];
    int i = d.index(floorCoord(p.x / d.h));
    printf("SPEED t=%.12g phase=%d particle=%d x=(%g,%g,%g) v=(%g,%g,%g) cell=%d phi=%g fluid=%d\n", time,
           phase, k, p.x[0], p.x[1], p.x[2], p.v[0], p.v[1], p.v[2], i, d.cells[i].phi, d.cells[i].fluid);
    for (int a = 0; a < 3; ++a)
    {
        int hi = d.neighbor(i, 2 * a + 1);
        printf(" axis=%d open=(%g,%g) u=(%g,%g) wall=(%g,%g)\n", a, d.cells[i].open[a], d.cells[hi].open[a],
               d.cells[i].u[a], d.cells[hi].u[a], d.cells[i].wall[a], d.cells[hi].wall[a]);
    }
}
int main(int argc, char **argv)
{
    try
    {
        std::map<std::string, std::string> args;
        for (int i = 1; i < argc; i += 2)
        {
            if (i + 1 >= argc)
                throw std::runtime_error("arguments require --key value");
            args[argv[i]] = argv[i + 1];
        }
        auto number = [&](std::string key, double fallback) {
            return args.count(key) ? std::stod(args[key]) : fallback;
        };
        const std::string character = args.count("--character") ? args["--character"] : "output/character",
                          output = args.count("--output") ? args["--output"] : "output/scene_low";
        const double h = number("--h", .08), duration = number("--duration", 1.2), fps = number("--fps", 30),
                     ppc = number("--ppc", 8), maxDt = number("--dt", .004),
                     radius = number("--emitter-radius", .26), speed = number("--emitter-speed", 1.1),
                     rate = number("--whitewater-rate", 40000), offset = number("--animation-offset", 0),
                     obstacleH = number("--obstacle-h", h * .5);
        const double emitterX = number("--emitter-x", 1.6), emitterZ = number("--emitter-z", 1.2);
        const double initialDepth = number("--initial-depth", 0),
                     foamRadiusScale = number("--foam-radius-scale", 1),
                     foamLifeScale = number("--foam-life-scale", 1);
        const double maxSpeed = number("--max-speed", 50);
        const bool writeCache=number("--write-cache",1)!=0;
        const bool profile = number("--profile", 0) != 0;
        const bool diagnoseTransfer = number("--diagnose-transfer", 0) != 0;
        const double drainSetting = number("--drain-enabled", 1);
        if (drainSetting != 0 && drainSetting != 1)
            throw std::runtime_error("--drain-enabled must be 0 or 1");
        const bool drainEnabled = drainSetting != 0;
        for (double value : {h, duration, fps, ppc, maxDt, radius, speed, rate, offset, obstacleH, emitterX,
                             emitterZ, initialDepth, foamRadiusScale, foamLifeScale, maxSpeed})
            if (!std::isfinite(value))
                throw std::runtime_error("nonfinite scene parameter");
        if (radius <= 0 || speed <= 0 || rate < 0 || obstacleH <= 0 || offset < 0 || emitterX - radius <= 0 ||
            emitterX + radius >= 3.2 || emitterZ - radius <= 0 || emitterZ + radius >= 2.4)
            throw std::runtime_error("invalid emitter, obstacle resolution or animation offset");
        const double volume = h * h * h / ppc, flow = M_PI * radius * radius * speed;
        if (h <= 0 || ppc <= 0 || maxDt <= 0 || duration <= 0 || fps <= 0)
            throw std::runtime_error("positive parameters required");
        if (initialDepth < 0 || initialDepth > .3 || foamRadiusScale <= 0 || foamLifeScale <= 0 ||
            maxSpeed <= 0)
            throw std::runtime_error("invalid pool depth or foam scale");
        std::filesystem::create_directories(output);
        std::vector<uint64_t> ids;
        uint64_t nextId = 0, emitted = 0, drained = 0;
        double fraction = 0;
        auto emit = [&](std::vector<Particle> &p, int n, double depth) {
            for (int i = 0; i < n; ++i)
            {
                uint64_t id = nextId++;
                double r = radius * sqrt(random01(id * 3 + 19)), angle = 2 * M_PI * random01(id * 3 + 20);
                Particle q;
                q.x = Vec(emitterX + r * cos(angle), 2.55 - depth * random01(id * 3 + 21),
                          emitterZ + r * sin(angle));
                q.v = Vec(0, -speed, 0);
                p.push_back(q);
                ids.push_back(id);
            }
        };
        Character mesh(character, obstacleH, offset);
        mesh.profileBuild = profile;
        if (offset + duration > (mesh.frameCount - 1) / mesh.fps + 1e-10)
            throw std::runtime_error("requested duration exceeds the baked character sequence");
        bool resume = number("--resume", 0) != 0;
        std::vector<Particle> initial;
        uint64_t initialCount = 0, initialPoolCount = 0;
        if (!resume)
        {
            emit(initial, std::max(1, int(M_PI * radius * radius * .12 / volume)), .12);
            if (initialDepth > 0)
            {
                mesh.build(0);
                const double dx = cbrt(volume);
                // Equal-volume strata. A partial top stratum is sampled in
                // proportion to its physical thickness, without changing Vp.
                std::vector<Particle> pool;
                for (int y = 0; y * dx < initialDepth - 1e-12; ++y)
                    for (int x = 0; (x + .5) * dx < 3.2; ++x)
                        for (int z = 0; (z + .5) * dx < 2.4; ++z)
                        {
                            double thickness = std::min(dx, initialDepth - y * dx);
                            uint64_t key = uint64_t(x) * 73856093ULL + uint64_t(y) * 19349663ULL +
                                           uint64_t(z) * 83492791ULL;
                            if (thickness < dx * (1 - 1e-10) && random01(key) >= thickness / dx)
                                continue;
                            Particle q;
                            q.x = Vec((x + .5) * dx, y * dx + thickness * .5, (z + .5) * dx);
                            pool.push_back(q);
                        }
                if (!pool.empty())
                {
                    auto candidates = allocate<Particle>(pool.size());
                    auto keep = allocate<int>(pool.size());
                    cudaCheck(cudaMemcpy(candidates.data(), pool.data(), pool.size() * sizeof(Particle),
                                         cudaMemcpyHostToDevice));
                    initialPoolMask<<<blocks(pool.size()), 256>>>(mesh.view, candidates.data(), pool.size(),
                                                                  .5 * obstacleH, keep.data());
                    std::vector<int> mask(pool.size());
                    cudaCheck(cudaMemcpy(mask.data(), keep.data(), mask.size() * sizeof(int),
                                         cudaMemcpyDeviceToHost));
                    for (size_t i = 0; i < pool.size(); ++i)
                        if (mask[i])
                        {
                            initial.push_back(pool[i]);
                            ids.push_back(nextId++);
                            ++initialPoolCount;
                        }
                }
            }
            initialCount = initial.size();
        }
        CheckpointHeader restart;
        std::vector<Secondary> restartWW;
        if (resume)
        {
            std::ifstream f(output + "/checkpoint.bin", std::ios::binary);
            f.read(reinterpret_cast<char *>(&restart), sizeof(restart));
            if (!f || restart.magic != 0x464c554944303034ULL || restart.particleSize != sizeof(Particle) ||
                restart.secondarySize != sizeof(Secondary) || restart.h != h || restart.ppc != ppc ||
                restart.radius != radius || restart.speed != speed || restart.emitterX != emitterX ||
                restart.emitterZ != emitterZ || restart.drainEnabled != uint64_t(drainEnabled) ||
                restart.initialDepth != initialDepth || restart.foamRadiusScale != foamRadiusScale ||
                restart.foamLifeScale != foamLifeScale || restart.whitewaterRate != rate)
                throw std::runtime_error("checkpoint configuration mismatch");
            initial.resize(restart.count);
            ids.resize(restart.count);
            restartWW.resize(restart.secondaryCount);
            f.read(reinterpret_cast<char *>(initial.data()), initial.size() * sizeof(Particle));
            f.read(reinterpret_cast<char *>(ids.data()), ids.size() * sizeof(uint64_t));
            f.read(reinterpret_cast<char *>(restartWW.data()), restartWW.size() * sizeof(Secondary));
            if (!f)
                throw std::runtime_error("truncated checkpoint");
            nextId = restart.nextId;
            emitted = restart.emitted;
            drained = restart.drained;
            fraction = restart.fraction;
            initialCount = restart.initialCount;
            initialPoolCount = restart.initialPoolCount;
            trimCsv(output + "/metrics.csv", restart.time);
            trimCsv(output + "/frames.csv", restart.time);
        }
        Solver solver(initial, true, h, Vec(3.2, 3.2, 2.4));
        solver.pressureDiagnostic = output + "/pressure_failure.csv";
        solver.includeCutCells = true;
        const std::string pressureMethod=args.count("--pressure")?args["--pressure"]:"mgpcg";
        if(pressureMethod!="mgpcg" && pressureMethod!="jacobi") throw std::runtime_error("--pressure must be mgpcg or jacobi");
        solver.jacobiPressure=pressureMethod=="jacobi";
        solver.pressureMaxIterations=int(number("--pressure-max-iterations",600));
        if(solver.pressureMaxIterations<1)throw std::runtime_error("positive pressure iteration limit required");
        ParticleManager particleManager;particleManager.upload(ids);
        std::ofstream settings(output + "/settings.json");
        settings << std::setprecision(14) << "{\"solver_revision\":\"gpu-mgpcg-11\",\"barrier_signing\":\"interior\",\"h\":" << h
                 << ",\"precision\":\"" << (singlePrecision?"float":"double") << "\",\"pressure_method\":\"" << pressureMethod
                 << "\",\"particle_bytes\":" << sizeof(Particle) << ",\"cell_bytes\":" << sizeof(Cell)
                 << ",\"pressure_check_tolerance\":" << pressureCheckTolerance << ",\"divergence_tolerance\":" << divergenceTolerance
                 << ",\"contact_tolerance\":" << contactTolerance << ",\"obstacle_h\":" << obstacleH << ",\"duration\":" << duration << ",\"fps\":" << fps
                 << ",\"ppc\":" << ppc << ",\"particle_volume\":" << volume << ",\"dt_max\":" << maxDt
                 << ",\"emitter_radius\":" << radius << ",\"emitter_speed\":" << speed
                 << ",\"flow_m3_s\":" << flow << ",\"whitewater_rate_m3_s\":" << rate
                 << ",\"animation_offset\":" << offset << ",\"domain\":[3.2,3.2,2.4],\"emitter_center\":["
                 << emitterX << ",2.55," << emitterZ << "],\"drain_width\":0.14,\"drain_height\":0.18,"
                 << "\"drain_enabled\":" << (drainEnabled ? "true" : "false")
                 << ",\"closed_sidewalls\":" << (drainEnabled ? "false" : "true")
                 << ",\"initial_depth\":" << initialDepth
                 << ",\"initial_pool_particles\":" << initialPoolCount
                 << ",\"initial_pool_volume\":" << initialPoolCount * volume
                 << ",\"max_particle_speed_m_s\":" << maxSpeed << ",\"foam_radius_scale\":" << foamRadiusScale
                 << ",\"foam_life_scale\":" << foamLifeScale << "}";
        settings.close();

        double derivativeError = mesh.validateDerivative(.413);
        if (derivativeError > .005)
            throw std::runtime_error("character velocity derivative error");
        mesh.build(resume ? restart.time : 0);
        Whitewater ww;
        ww.closedSides = !drainEnabled;
        ww.foamRadiusScale = foamRadiusScale;
        ww.foamLifeScale = foamLifeScale;
        if (resume)
        {
            ww.host = restartWW;
            ww.spawned = restart.spawned;
            ww.expired = restart.expired;
            ww.expected = restart.expected;
        }
        std::ofstream metrics(output + "/metrics.csv", resume ? std::ios::app : std::ios::out);
        if (!resume) {
            metrics << "step,time,dt,particles,emitted,drained,mass_volume,ledger_"
                       "error,occupied_volume,leaves,"
                       "missing,residual,residual_absolute,rhs_norm,div_before,flux_"
                       "div_after,wall_flux_error,"
                       "contacts,penetrations,penetration_depth,contact_normal_error,"
                       "cfl,foam,spray,ww_spawned,"
                       "ww_expected,ww_expired,ww_outside,ww_depth,ww_speed,gpu_used_"
                       "mb,topology_ms,mesh_sdf_ms,"
                       "boundary_ms,p2g_ms,pressure_ms,g2p_ms,contact_ms,whitewater_"
                       "ms,transfer_ms,io_ms,"
                       "unresolved_contact_droplets,limited_affine_rows,pressure_iterations,mg_levels,mg_setup_ms,mg_solve_ms,mg_bytes";
            if (profile)
                metrics << ",meshtosdf_ms,mesh_prepare_ms,fluid_active_voxels,obstacle_active_voxels,fluid_pressure_cells,profile_substep_ms,profile_cache_transfer_ms";
            metrics << '\n';
        }
        std::ofstream manifest(output + "/frames.csv", resume ? std::ios::app : std::ios::out);
        if (!resume)
            manifest << "frame,time,particles,foam,spray\n";
        double t = resume ? restart.time : 0;
        int frame = resume ? restart.nextFrame : 0, step = resume ? restart.step : 0;
        double lastTransfer = 0, lastIo = 0;
        auto save = [&]() {
            if(!writeCache) {++frame;lastTransfer=lastIo=0;return;}
            Timer transfer;
            auto particles = solver.download();
            ids=particleManager.download(solver.particleCount);
            ww.download();
            auto grid = mesh.downloadGrid();
            auto leafData = solver.downloadBlocks();
            lastTransfer = transfer.ms();
            Timer io;
            char tag[32];
            std::snprintf(tag, sizeof(tag), "%06d", frame);
            std::string prefix = output + "/frame_" + tag;
            std::ofstream f(prefix + ".csv");
            f << "id,x,y,z,vx,vy,vz\n" << std::setprecision(10);
            for (size_t i = 0; i < particles.size(); ++i)
            {
                auto p = particles[i];
                f << ids[i] << ',' << p.x[0] << ',' << p.x[1] << ',' << p.x[2] << ',' << p.v[0] << ','
                  << p.v[1] << ',' << p.v[2] << '\n';
            }
            std::ofstream w(prefix + "_whitewater.csv");
            w << "id,x,y,z,vx,vy,vz,radius,age,life,kind\n" << std::setprecision(9);
            for (auto p : ww.host)
                w << p.id << ',' << p.x[0] << ',' << p.x[1] << ',' << p.x[2] << ',' << p.v[0] << ',' << p.v[1]
                  << ',' << p.v[2] << ',' << p.radius << ',' << p.age << ',' << p.life << ',' << p.kind
                  << '\n';
            std::ofstream g(prefix + "_obstacle.grid", std::ios::binary);
            g.write(grid.data(), grid.size());
            std::set<std::tuple<int, int, int>> leaves, core;
            for (auto leaf:leafData) {
                auto q=leaf.origin;auto key=std::make_tuple(q[0],q[1],q[2]);
                leaves.insert(key);if(leaf.core)core.insert(key);
            }
            std::ofstream b(prefix + "_blocks.csv");
            b << "x,y,z,core\n";
            for (auto q : leaves)
                b << std::get<0>(q) * h << ',' << std::get<1>(q) * h << ',' << std::get<2>(q) * h << ','
                  << core.count(q) << '\n';
            manifest << frame << ',' << std::setprecision(12) << t << ',' << particles.size() << ','
                     << ww.foam << ',' << ww.spray << '\n';
            manifest.flush();
            f.close();
            w.close();
            g.close();
            b.close();
            CheckpointHeader header;
            header.count = particles.size();
            header.secondaryCount = ww.host.size();
            header.nextId = nextId;
            header.emitted = emitted;
            header.drained = drained;
            header.spawned = ww.spawned;
            header.expired = ww.expired;
            header.time = t;
            header.fraction = fraction;
            header.expected = ww.expected;
            header.h = h;
            header.ppc = ppc;
            header.radius = radius;
            header.speed = speed;
            header.emitterX = emitterX;
            header.emitterZ = emitterZ;
            header.nextFrame = frame + 1;
            header.step = step;
            header.drainEnabled = drainEnabled;
            header.initialDepth = initialDepth;
            header.initialCount = initialCount;
            header.initialPoolCount = initialPoolCount;
            header.foamRadiusScale = foamRadiusScale;
            header.foamLifeScale = foamLifeScale;
            header.whitewaterRate = rate;
            std::ofstream cp(output + "/checkpoint.tmp", std::ios::binary);
            cp.write(reinterpret_cast<char *>(&header), sizeof(header));
            cp.write(reinterpret_cast<char *>(particles.data()), particles.size() * sizeof(Particle));
            cp.write(reinterpret_cast<char *>(ids.data()), ids.size() * sizeof(uint64_t));
            cp.write(reinterpret_cast<char *>(ww.host.data()), ww.host.size() * sizeof(Secondary));
            cp.close();
            std::filesystem::rename(output + "/checkpoint.tmp", output + "/checkpoint.bin");
            lastIo = io.ms();
            std::cout << "FRAME " << frame << " t=" << t << " fluid=" << particles.size()
                      << " foam=" << ww.foam << " spray=" << ww.spray << std::endl;
            ++frame;
        };
        solver.reset();
        if (!resume)
            save();
        while (t < duration - 1e-10)
        {
            const auto substepStart = std::chrono::steady_clock::now();
            double target = std::min(duration, frame / fps);
            double dt = std::min({maxDt, solver.timestep(), .2 * h / (mesh.maxWallSpeed + 1), target - t});
            if (dt < 1e-10)
            {
                save();
                continue;
            }
            double meshMs = mesh.build(t + dt);
            // Count actual active voxels, not leaf capacity. This metadata read
            // is outside the MeshToSDF timer and included in substep overhead.
            const uint64_t obstacleActive = profile
                ? nanovdb::util::cuda::DeviceGridTraits<Build>::getActiveVoxelCount(mesh.view.field.grid) : 0;
            Timer transfer;
            if(drainEnabled) drained+=particleManager.drain(solver);
            fraction+=flow*dt/volume;
            int add=int(fraction);fraction-=add;
            particleManager.emit(solver,add,nextId,radius,speed,emitterX,emitterZ,speed*dt);
            nextId+=add;emitted+=add;
            double transferMs = transfer.ms();
            auto contact = allocate<SceneContact>(blocks(solver.particleCount));
            // Include the pre-contact impulse in the secondary-particle source.
            auto before = allocate<Vec>(solver.particleCount);
            copyVelocities<<<blocks(solver.particleCount),256>>>(solver.particles.data(),before.data(),solver.particleCount);
            Timer ct;
            characterContact<<<blocks(solver.particleCount), 256>>>(solver.particles.data(),
                                                                    solver.particleCount, mesh.view, h * .25,
                                                                    nullptr, false, double(Real(.1)*solver.d.h));
            fluid::sync();
            double contactMs = ct.ms();
            solver.rebuild();
            if (step % 100 == 0)
                solver.validateTopology();
            solver.reset();
            Timer boundary;
            obstacleCells<<<blocks(solver.d.count), 256>>>(solver.d, mesh.view.field);
            characterFaces<<<blocks(solver.d.count), 256>>>(solver.d, mesh.view);
            double boundaryMs = boundary.ms();
            solver.transfer(dt);
            auto diagnoseGrid = [&](const char *phase) {
                if (!diagnoseTransfer)
                    return;
                auto cc = solver.downloadCells();
                double maxU = 0, energy = 0;
                int best = 0, axisBest = 0;
                for (int j = 1; j < solver.d.count; ++j)
                    for (int a = 0; a < 3; ++a)
                    {
                        energy += cc[j].mass[a] * cc[j].u[a] * cc[j].u[a];
                        if (fabs(cc[j].u[a]) > maxU)
                        {
                            maxU = fabs(cc[j].u[a]);
                            best = j;
                            axisBest = a;
                        }
                    }
                auto c = cc[best];
                std::cout << "GRID " << phase << " t=" << std::setprecision(12) << t + dt << " max=" << maxU
                          << " E=" << energy << " coord=" << c.ijk[0] << "," << c.ijk[1] << "," << c.ijk[2]
                          << " axis=" << axisBest << " mass=" << c.mass[axisBest]
                          << " open=" << c.open[axisBest] << " wall=" << c.wall[axisBest] << std::endl;
            };
            diagnoseGrid("p2g");
            solver.pressure();
            diagnoseGrid("pressure");
            auto errors = allocate<double>(solver.d.count);
            cudaCheck(cudaMemset(errors.data(), 0, sizeof(double) * solver.d.count));
            wallCheck<<<blocks(solver.d.count), 256>>>(solver.d, mesh.view, errors.data());
            double wallError=thrust::reduce(thrust::device,errors.data(),errors.data()+solver.d.count,0.,cuda::maximum<double>());
            if (wallError > (singlePrecision?2e-5:1e-9))
            {
                std::cerr << std::setprecision(17) << "WALL FAIL t=" << t + dt << " error=" << wallError
                          << std::endl;
                throw std::runtime_error("wall flux mismatch");
            }
            solver.advect(dt);
            auto speedErrors = allocate<int>(1);
            cudaCheck(cudaMemset(speedErrors.data(), 0, sizeof(int)));
            speedGuard<<<blocks(solver.particleCount), 256>>>(solver.d, solver.particles.data(),
                                                              solver.particleCount, maxSpeed, t + dt, 0,
                                                              speedErrors.data());
            int speedError = 0;
            cudaCheck(cudaMemcpy(&speedError, speedErrors.data(), sizeof(int), cudaMemcpyDeviceToHost));
            if (speedError)
                throw std::runtime_error("excessive particle speed after grid transfer");
            Timer post;
            characterContact<<<blocks(solver.particleCount), 256>>>(solver.particles.data(),
                                                                    solver.particleCount, mesh.view, h * .25,
                                                                    contact.data(), true, double(Real(.1)*solver.d.h), true);
            auto checks=thrust::reduce(thrust::device,contact.data(),contact.data()+blocks(solver.particleCount),SceneContact{},ContactReduce{});
            contactMs+=post.ms();
            int remaining=checks.sdfRemaining+checks.meshRemaining,contacts=checks.contacts;
            double depth=checks.depth,normal=checks.normalError;
            if (remaining)
            {
                std::cerr << std::setprecision(17) << "CONTACT FAIL " << remaining << " depth " << depth
                          << " time " << t << " mesh_time " << t + dt << std::endl;
                throw std::runtime_error("character penetration");
            }
            speedGuard<<<blocks(solver.particleCount), 256>>>(solver.d, solver.particles.data(),
                                                              solver.particleCount, maxSpeed, t + dt, 1,
                                                              speedErrors.data());
            cudaCheck(cudaMemcpy(&speedError, speedErrors.data(), sizeof(int), cudaMemcpyDeviceToHost));
            if (speedError)
                throw std::runtime_error("excessive particle speed after character contact");
            double wwMs = ww.step(solver, mesh.view, before, dt, ppc, rate, step);
            double cfl = solver.particleCfl(dt);
            t += dt;
            ++step;
            size_t free, total;
            cudaCheck(cudaMemGetInfo(&free, &total));
            if (fabs(t - target) < 1e-9)
                save();
            else
                lastIo = lastTransfer = 0;
            auto s = solver.stats;
            long long ledger = static_cast<long long>(solver.particleCount) -
                               static_cast<long long>(initialCount + emitted - drained);
            metrics << std::setprecision(12) << step << ',' << t << ',' << dt << ',' << solver.particleCount
                    << ',' << emitted << ',' << drained << ',' << solver.particleCount * volume << ','
                    << ledger << ',' << s.fluidCells * h * h * h << ',' << s.leaves << ',' << s.missing << ','
                    << s.residual << ',' << s.absoluteResidual << ',' << s.rhsNorm << ',' << s.divBefore
                    << ',' << s.divAfter << ',' << wallError << ',' << contacts << ',' << remaining << ','
                    << depth << ',' << normal << ',' << cfl << ',' << ww.foam << ',' << ww.spray << ','
                    << ww.spawned << ',' << ww.expected << ',' << ww.expired << ',' << ww.outside << ','
                    << ww.maxDepth << ',' << ww.maxSpeed << ',' << (total - free) / 1048576. << ','
                    << s.topology << ',' << meshMs << ',' << boundaryMs << ',' << s.p2g << ',' << s.pressure
                    << ',' << s.g2p << ',' << contactMs << ',' << wwMs << ',' << transferMs + lastTransfer
                    << ',' << lastIo << ',' << s.unresolved << ',' << s.affineLimited << ',' << s.iterations
                    << ',' << s.mgLevels << ',' << s.mgSetup << ',' << s.mgSolve << ',' << s.mgBytes;
            if (profile) {
                fluid::sync();
                const double substepMs = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - substepStart).count();
                metrics << ',' << mesh.sdfPipelineMs << ',' << mesh.prepareMs
                        << ',' << (solver.d.count - 1) << ',' << obstacleActive
                        << ',' << s.fluidCells << ',' << substepMs << ',' << lastTransfer;
            }
            metrics << '\n';
            metrics.flush();
            if (step % 100 == 0)
                std::cout << "STEP " << step << " t=" << t << " residual=" << s.residual
                          << " div=" << s.divAfter << std::endl;
        }
        std::ofstream done(output + "/complete.json");
        done << "{\"steps\":" << step << ",\"frames\":" << frame << ",\"initial_particles\":" << initialCount
             << ",\"character_velocity_fd_error\":" << derivativeError << "}";
        std::cout << "PASS scene" << std::endl;
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "FAIL: " << e.what() << std::endl;
        return 1;
    }
}
