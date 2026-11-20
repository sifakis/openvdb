// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#include "Whitewater.cuh"
#include <iostream>
using namespace fluid;
__global__ void verifyWall(Domain d, MeshView mesh, double *error)
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
        }
    error[i] = e;
}
struct FieldCheck
{
    int mismatch = 0, tested = 0, interior = 0, inactiveInterior = 0;
    double distance = 0;
};
__global__ void fieldCheck(MeshView mesh, FieldCheck *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    Vec p(.6 + 2 * random01(3 * i + 71), 2 * random01(3 * i + 72), .4 + 1.6 * random01(3 * i + 73));
    FieldCheck r;
    auto nearest = mesh.nearest(p);
    if (nearest.distance > 2 * mesh.field.h)
    {
        bool inside = mesh.inside(p);
        double sdf = mesh.field.sdf(p);
        r.tested = 1;
        r.interior = inside;
        r.mismatch = inside != (sdf < 0);
        r.distance = r.mismatch ? nearest.distance : 0;
        Coord c = floorCoord(mesh.field.grid->worldToIndex(p));
        r.inactiveInterior = inside && !mesh.field.grid->tree().isActive(c);
    }
    out[i] = r;
}
double sourceTest(int copies, bool rotational)
{
    std::vector<Particle> p;
    double h = .08;
    for (int x = 0; x < 12; ++x)
        for (int y = 0; y < 12; ++y)
            for (int z = 0; z < 12; ++z)
                for (int k = 0; k < copies; ++k)
                {
                    Particle q;
                    q.x = Vec(.4 + .04 * x, .4 + .04 * y, .4 + .04 * z);
                    q.v = Vec(.4, .2, -.1);
                    p.push_back(q);
                }
    Solver s(p, true, h);
    s.reset();
    s.transfer(.002);
    s.advect(.002, false);
    auto after = s.download();
    if (rotational)
    {
        for (auto &q : after)
        {
            q.c[0][1] = -20;
            q.c[1][0] = 20;
        }
        cudaCheck(cudaMemcpy(s.particles.data(), after.data(), after.size() * sizeof(Particle),
                             cudaMemcpyHostToDevice));
    }
    auto before = allocate<Vec>(p.size());
    std::vector<Vec> initialVelocity;for(auto q:p)initialVelocity.push_back(q.v);
    cudaCheck(cudaMemcpy(before.data(),initialVelocity.data(),p.size()*sizeof(Vec),cudaMemcpyHostToDevice));
    auto births = allocate<Secondary>(8192);
    auto count = allocate<int>(1);
    auto expected = allocate<double>(1);
    cudaCheck(cudaMemset(count.data(), 0, sizeof(int)));
    cudaCheck(cudaMemset(expected.data(), 0, sizeof(double)));
    spawnWhitewater<<<blocks(p.size()), 256>>>(s.d, s.particles.data(), before.data(), p.size(), .002,
                                               8 * copies, pow(h, 3) / (8 * copies), 40000, 7, births.data(),
                                               8192, count.data(), expected.data());
    double value;
    cudaCheck(cudaMemcpy(&value, expected.data(), sizeof(double), cudaMemcpyDeviceToHost));
    return value;
}
int main(int argc, char **argv)
{
    try
    {
        if (argc < 2)
            throw std::runtime_error("verify_scene CHARACTER [OBSTACLE_H]");
        Character mesh(argv[1], argc > 2 ? std::stod(argv[2]) : .016);
        if (argc > 3)
        {
            std::vector<Particle> p(1);
            p[0].x = Vec(1.6, 1, 1);
            Solver solver(p, false, .032, Vec(3.2, 3.2, 2.4));
            auto errors = allocate<double>(solver.d.count);
            for (int k = 0; k < 30; ++k)
            {
                double time = 2.265 + k * .0005;
                mesh.build(time);
                solver.reset();
                obstacleCells<<<blocks(solver.d.count), 256>>>(solver.d, mesh.view.field);
                characterFaces<<<blocks(solver.d.count), 256>>>(solver.d, mesh.view);
                cudaCheck(cudaMemset(errors.data(), 0, solver.d.count * sizeof(double)));
                verifyWall<<<blocks(solver.d.count), 256>>>(solver.d, mesh.view, errors.data());
                std::vector<double> e(solver.d.count);
                cudaCheck(
                    cudaMemcpy(e.data(), errors.data(), e.size() * sizeof(double), cudaMemcpyDeviceToHost));
                int worst = std::max_element(e.begin(), e.end()) - e.begin();
                std::cout << std::setprecision(17) << "WALL time=" << time << " error=" << e[worst]
                          << " index=" << worst << std::endl;
            }
            return 0;
        }
        int n = 100000;
        auto data = allocate<FieldCheck>(n);
        int mismatch = 0, interior = 0, inactive = 0, tested = 0;
        double maxError = 0;
        for (double t : {0., .5, 1., 2., 3., 4., 5., 6.})
        {
            mesh.build(t);
            fieldCheck<<<blocks(n), 256>>>(mesh.view, data.data(), n);
            std::vector<FieldCheck> out(n);
            cudaCheck(cudaMemcpy(out.data(), data.data(), n * sizeof(FieldCheck), cudaMemcpyDeviceToHost));
            int bad = 0;
            for (auto r : out)
            {
                bad += r.mismatch;
                tested += r.tested;
                interior += r.interior;
                inactive += r.inactiveInterior;
                maxError = std::max(maxError, r.distance);
            }
            mismatch += bad;
            std::cout << "SDF t=" << t << " mismatches=" << bad << std::endl;
        }
        double ballistic = sourceTest(1, false), one = sourceTest(1, true), two = sourceTest(2, true);
        if (ballistic > 1e-10 || fabs(one - two) > 1e-10 * std::max(one, 1.))
            throw std::runtime_error("whitewater physical source normalization failed");
        std::vector<Particle> p(1);
        p[0].x = Vec(.5, .5, .5);
        Solver solver(p, true, .08);
        solver.reset();
        Secondary q;
        q.x = Vec(3.5, 2.8, 1.2);
        auto secondary = allocate<Secondary>(1);
        auto check = allocate<WWCheck>(1);
        cudaCheck(cudaMemcpy(secondary.data(), &q, sizeof(q), cudaMemcpyHostToDevice));
        updateWhitewater<<<1, 1>>>(solver.d, mesh.view, secondary.data(), 1, .1, 8, check.data(), true);
        cudaCheck(cudaMemcpy(&q, secondary.data(), sizeof(q), cudaMemcpyDeviceToHost));
        double analyticV = -.981 * exp(-.035),
               ballisticError = fabs(q.v[1] - analyticV) + fabs(q.x[1] - (2.8 + .1 * analyticV));
        if (ballisticError > 1e-10)
            throw std::runtime_error("out-of-grid spray motion failed");
        std::cout << std::setprecision(12) << "RESULT tested=" << tested << " interior=" << interior
                  << " inactive_interior=" << inactive << " mismatch=" << mismatch
                  << " mismatch_depth=" << maxError << " freefall_births_expected=" << ballistic
                  << " source_ppc8=" << one << " source_ppc16=" << two
                  << " outside_spray_error=" << ballisticError << std::endl;
        if (mismatch)
            throw std::runtime_error("SDF sign differs from closed-component union beyond two voxels");
        std::cout << "PASS scene verification" << std::endl;
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "FAIL " << e.what() << std::endl;
        return 1;
    }
}
