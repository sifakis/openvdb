// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#include <iostream>
#include "Obstacle.cuh"
#include "Validation.cuh"
#include "Run.cuh"
using namespace fluid;
void obstacleDemo(const std::string &mode, const std::string &out, RunOptions options)
{
    int kind = mode == "static" ? 0 : mode == "moving" ? 1 : 2;
    Obstacle obstacle(options.h * .5);
    obstacle.build(Motion{kind, 0});
    validateSdf(obstacle);
    validateContact(obstacle);
    auto initial = dam(options.h);
    initial.erase(std::remove_if(initial.begin(), initial.end(),
                                 [&](const Particle &p) {
                                     return obstacle.view.motion.exact(p.x) < obstacle.view.h * .5;
                                 }),
                  initial.end());
    Solver s(initial, true, options.h);
    std::filesystem::create_directories(out);
    std::ofstream log(out + "/metrics.csv");
    log << "step,time,dt,topology_ms,mesh_ms,boundary_ms,p2g_ms,pressure_ms,g2p_ms,collision_ms,cfl,residual,"
           "div_before,div_after,iterations,fluid_cells,leaves,missing,wall_error,wall_faces,partial_faces,"
           "physical_divergence,contact_normal_error,penetration_attempts,max_attempt_depth,remaining,max_"
           "remaining_depth,exact_remaining,exact_depth,contacts";
    particleColumns(log);
    log << "\n";
    s.reset();
    classify<<<blocks(s.particleCount), 256>>>(s.d, s.particles.data(), s.particleCount, s.missing.data());
    Run run(options, out, s);
    obstacle.write(run.write(s, out, true));
    while (run.active()) {
        const double dt = run.advance(s.timestep());
        const int k = run.step;
        const double time = run.time;
        // Geometry is evaluated at the end of this substep; vertex velocities
        // are the exact time derivatives at that same time (backward Euler wall).
        double meshMs = obstacle.build(Motion{kind, time});
        if (k == 1 || !run.active())
            validateSdf(obstacle);
        s.rebuild();
        s.reset();
        Timer collisionTimer;
        Contact before = contacts(s, obstacle);
        double collisionMs = collisionTimer.ms();
        Timer boundaryTimer;
        obstacleCells<<<blocks(s.d.count), 256>>>(s.d, obstacle.view);
        obstacleFaces<<<blocks(s.d.count), 256>>>(s.d, obstacle.view);
        double boundaryMs = boundaryTimer.ms();
        s.transfer(dt);
        s.pressure();
        auto boundary = boundaryMetrics(s, obstacle);
        double wall = boundary.error;
        s.advect(dt);
        Timer postCollisionTimer;
        Contact c = combine(before, contacts(s, obstacle));
        collisionMs += postCollisionTimer.ms();
        double cfl = s.particleCfl(dt);
        auto particles = particleMetrics(s, dt);
        if (k == 1 || k % 20 == 0)
            s.validateTopology();
        auto st = s.stats;
        log << std::setprecision(12) << k << ',' << time << ',' << dt << ',' << st.topology << ',' << meshMs
            << ',' << boundaryMs << ',' << st.p2g << ',' << st.pressure << ',' << st.g2p << ',' << collisionMs
            << ',' << cfl << ',' << st.residual << ',' << st.divBefore << ',' << st.divAfter << ','
            << st.iterations << ',' << st.fluidCells << ',' << st.leaves << ',' << st.missing << ',' << wall
            << ',' << boundary.faces << ',' << boundary.partial << ',' << boundary.divergence << ','
            << c.normalError << ',' << c.attempted << ',' << c.before << ',' << c.remaining << ',' << c.after
            << ',' << c.exactRemaining << ',' << c.exactDepth << ',' << c.contacts;
        particleRow(log, s, particles);
        log << '\n';
        if (wall > 1e-8)
            throw std::runtime_error("wall normal velocity failed");
        if (run.save()) {
            obstacle.write(run.write(s, out));
            log.flush();
            std::cout << mode << " step=" << k << " time=" << time << " dt=" << dt
                      << " speed=" << particles.maxSpeed << " div=" << st.divAfter << " wall=" << wall
                      << " contacts=" << c.contacts << " penetration=" << c.remaining << std::endl;
        }
    }
    std::cout << "PASS " << mode << std::endl;
}
int main(int argc, char **argv)
{
    try {
        std::string mode = argc > 1 ? argv[1] : "stage1", out = argc > 2 ? argv[2] : "fluid_output";
        RunOptions options;
        for (int i = 3; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--duration" && i + 1 < argc)
                options.duration = std::stod(argv[++i]);
            else if (arg == "--fps" && i + 1 < argc)
                options.fps = std::stod(argv[++i]);
            else if (arg == "--voxel-size" && i + 1 < argc)
                options.h = std::stod(argv[++i]);
            else if (arg == "--max-steps" && i + 1 < argc)
                options.maxSteps = std::stoi(argv[++i]);
            else if (i == 3 && arg[0] != '-')
                options.steps = std::stoi(arg);
            else
                throw std::runtime_error("unknown or incomplete argument: " + arg);
        }
        if (!std::isfinite(options.duration) || options.duration < 0 || !std::isfinite(options.fps) ||
            options.fps <= 0 || !std::isfinite(options.h) || options.h <= 0 || options.steps <= 0 ||
            options.maxSteps <= 0)
            throw std::runtime_error("invalid run duration, fps or steps");
        if (mode == "static" || mode == "moving" || mode == "deforming") {
            obstacleDemo(mode, out, options);
            return 0;
        }
        if (mode != "stage1" && mode != "stage2")
            throw std::runtime_error("mode must be stage1, stage2, static, moving, or deforming");
        transferTest(options.h);
        Solver s(dam(options.h), mode == "stage2", options.h);
        std::filesystem::create_directories(out);
        std::ofstream log(out + "/metrics.csv");
        log << "step,time,dt,topology_ms,p2g_ms,pressure_ms,g2p_ms,residual,div_before,div_after,iterations,"
               "fluid_cells,leaves,missing";
        particleColumns(log);
        log << "\n";
        s.reset();
        classify<<<blocks(s.particleCount), 256>>>(s.d, s.particles.data(), s.particleCount,
                                                   s.missing.data());
        Run run(options, out, s);
        run.write(s, out, true);
        while (run.active()) {
            double dt = run.advance(s.timestep());
            int k = run.step;
            double time = run.time;
            s.step(dt);
            s.particleCfl(dt);
            auto particles = particleMetrics(s, dt);
            if (k == 1 || k % 20 == 0)
                s.validateTopology();
            auto st = s.stats;
            log << std::setprecision(12) << k << ',' << time << ',' << dt << ',' << st.topology << ','
                << st.p2g << ',' << st.pressure << ',' << st.g2p << ',' << st.residual << ',' << st.divBefore
                << ',' << st.divAfter << ',' << st.iterations << ',' << st.fluidCells << ',' << st.leaves
                << ',' << st.missing;
            particleRow(log, s, particles);
            log << '\n';
            if (run.save()) {
                run.write(s, out);
                log.flush();
                std::cout << "step=" << k << " time=" << time << " dt=" << dt
                          << " speed=" << particles.maxSpeed << " div=" << st.divBefore << " -> "
                          << st.divAfter << " residual=" << st.residual << std::endl;
            }
        }
        std::cout << "PASS " << mode << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "FAIL: " << e.what() << std::endl;
        return 1;
    }
}
