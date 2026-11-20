// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifndef EX_FLUID_RUN_CUH
#define EX_FLUID_RUN_CUH
#include "Fluid.cuh"

namespace fluid {
struct RunOptions
{
    int steps = 60;
    double duration = 0, fps = 30, h = .05;
    int maxSteps = 50000;
};
struct Run
{
    RunOptions options;
    double time = 0;
    int step = 0, frame = 0;
    std::ofstream frames;
    explicit Run(RunOptions o, const std::string &out, const Solver &solver)
        : options(o), frames(out + "/frames.csv")
    {
        frames << "frame,step,time,particles\n" << std::setprecision(17);
        std::ofstream config(out + "/scene.json");
        config << std::setprecision(17) << "{\n  \"voxel_size\": " << solver.d.h << ",\n  \"domain_cells\": ["
               << solver.d.nx << ',' << solver.d.ny << ',' << solver.d.nz
               << "],\n  \"world_size\": [1.6,1.6,1.2],\n  \"obstacle_voxel_size\": " << solver.d.h * .5
               << ",\n  \"contact_margin\": " << solver.d.h * .25
               << ",\n  \"marker_radius\": " << solver.d.h * .14
               << ",\n  \"initial_particle_count\": " << solver.particleCount
               << ",\n  \"duration\": " << options.duration << ",\n  \"fps\": " << options.fps << "\n}\n";
    }
    bool active() const
    {
        return options.duration > 0 ? time < options.duration - 1e-12 : step < options.steps;
    }
    double advance(double proposed)
    {
        if (step >= options.maxSteps || !std::isfinite(proposed) || proposed < 1e-7)
            throw std::runtime_error("simulation made insufficient progress");
        double dt = proposed;
        if (options.duration > 0) {
            const double target = std::min(options.duration, double(frame + 1) / options.fps);
            dt = std::min(dt, target - time);
        }
        if (!(dt > 0))
            throw std::runtime_error("invalid frame scheduling timestep");
        time += dt;
        ++step;
        return dt;
    }
    bool save() const
    {
        if (options.duration > 0)
            return time >= std::min(options.duration, double(frame + 1) / options.fps) - 1e-12;
        return step % 5 == 0 || step == options.steps;
    }
    std::string write(Solver &s, const std::string &out, bool initial = false)
    {
        if (!initial)
            ++frame;
        char name[64];
        snprintf(name, sizeof(name), "/frame_%04d", options.duration > 0 ? frame : step);
        frames << frame << ',' << step << ',' << time << ',' << s.particleCount << '\n';
        frames.flush();
        writeFrame(s, out + name);
        return out + name;
    }
};
struct ParticleMetrics
{
    double maxSpeed = 0, meanSpeed = 0, kinetic = 0, potential = 0, maxAffine = 0, tankDepth = 0;
    Vec com{0};
    int tankRemaining = 0;
};
inline ParticleMetrics particleMetrics(Solver &s, double dt)
{
    ParticleMetrics r;
    for (const auto &p : s.download()) {
        double affine2 = 0;
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b) {
                if (!std::isfinite(p.c[a][b]))
                    throw std::runtime_error("nonfinite APIC affine matrix");
                affine2 += p.c[a][b] * p.c[a][b];
            }
        double speed = p.v.length();
        if (!std::isfinite(speed))
            throw std::runtime_error("nonfinite particle speed");
        r.maxSpeed = std::max(r.maxSpeed, speed);
        r.meanSpeed += speed;
        r.maxAffine = std::max(r.maxAffine, sqrt(affine2));
        r.kinetic += .5 * (p.v.lengthSqr() + .25 * s.d.h * s.d.h * affine2);
        r.potential += 9.81 * p.x[1];
        r.com += p.x;
        for (int a = 0; a < 3; ++a) {
            if (!std::isfinite(p.x[a]))
                throw std::runtime_error("nonfinite particle position");
            double depth = std::max(-p.x[a], p.x[a] - (a == 0 ? s.d.nx : a == 1 ? s.d.ny : s.d.nz) * s.d.h);
            r.tankDepth = std::max(r.tankDepth, depth);
            r.tankRemaining += depth > 1e-7;
        }
    }
    r.meanSpeed /= s.particleCount;
    r.kinetic /= s.particleCount;
    r.potential /= s.particleCount;
    r.com /= s.particleCount;
    if (r.tankRemaining || r.maxSpeed * dt / s.d.h > .5)
        throw std::runtime_error("tank/CFL stability validation failed");
    return r;
}
inline void particleColumns(std::ostream &out)
{
    out << ",residual_absolute,rhs_norm,particle_count,max_speed,mean_speed,kinetic_energy,potential_energy,total_energy,max_affine,com_"
           "x,com_y,com_z,tank_remaining,tank_depth";
}
inline void particleRow(std::ostream &out, const Solver &s, const ParticleMetrics &p)
{
    out << ',' << s.stats.absoluteResidual << ',' << s.stats.rhsNorm << ',' << s.particleCount << ',' << p.maxSpeed << ',' << p.meanSpeed << ',' << p.kinetic << ','
        << p.potential << ',' << (p.kinetic + p.potential) << ',' << p.maxAffine << ',' << p.com[0] << ','
        << p.com[1] << ',' << p.com[2] << ',' << p.tankRemaining << ',' << p.tankDepth;
}
} // namespace fluid
#endif
