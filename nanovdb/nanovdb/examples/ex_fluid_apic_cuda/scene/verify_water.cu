// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#include "Whitewater.cuh"
#include <iostream>
using namespace fluid;

// Constant tangential translation must survive transfers at a frictionless
// floor.
double floorTransfer(int repeats, double y)
{
    Particle p;
    p.x = Vec(.8, y, .6);
    p.v = Vec(1, 0, 0);
    Solver solver({p}, true, .05);
    for (int k = 0; k < repeats; ++k)
    {
        solver.reset();
        solver.transfer(0, 0);
        solver.advect(0, false);
    }
    return solver.download()[0].v[0];
}
__global__ void makeContact(MeshView mesh, Particle *p)
{
    Vec x(1.62, 1.7, 1.0);
    auto near = mesh.nearest(x);
    x = near.point + near.normal * .008;
    for (int j = 0; j < 30; ++j)
        x += mesh.field.normal(x) * (.008 - mesh.field.sdf(x));
    Vec n = mesh.field.normal(x);
    near = mesh.nearest(x);
    p[0].x = x;
    p[0].v = near.velocity + n;
    p[0].c[0] = Vec(0, 2, 0);
}
__global__ void excludePressure(Domain d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < d.count)
        d.cells[i].phi = -1;
}
__global__ void concaveContact(MeshView mesh, double *result)
{
    Vec p(1.65858954371, .776458885029, 1.08814207154);
    result[0] = mesh.field.sdf(p);
    result[1] = mesh.inside(p);
    mesh.resolveInterior(p, .00623486, true);
    result[2] = mesh.field.sdf(p);
    result[3] = mesh.inside(p);
}
__global__ void missingVerticalTransport(Domain d, bool represented)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (!i || i >= d.count)
        return;
    auto &c = d.cells[i];
    c.phi = 1;
    c.fluid = 1;
    c.open[1] = represented ? 1 : 0;
    for (int a = 0; a < 3; ++a)
    {
        c.valid[a] = a != 1;
        c.transport[a] = 4 + a;
    }
}
__global__ void coupledVelocityCheck(double *error)
{
    const Vec normal(.6, -.8, 0), wall(1, 0, 0);
    Vec v = contactVelocity(Vec(0), normal, wall, true);
    error[0] = (v - Vec(1, 0, 0)).length();
    error[1] = (contactVelocity(Vec(2, 0, 3), normal, wall, true) - Vec(2, 0, 3)).length();
    error[2] = fmax(0., -v[1]) + fmax(0., -(v - wall).dot(normal));
}
int main(int argc, char **argv)
{
    try
    {
        auto coupled = allocate<double>(3);
        coupledVelocityCheck<<<1, 1>>>(coupled.data());
        double coupledError[3];
        cudaCheck(cudaMemcpy(coupledError, coupled.data(), sizeof(coupledError), cudaMemcpyDeviceToHost));
        std::cout << "coupled_floor_mesh_velocity_error=" << coupledError[0]
                  << " separating_error=" << coupledError[1] << " constraint_error=" << coupledError[2]
                  << std::endl;
        for (double e : coupledError)
            if (e > 1e-12)
                throw std::runtime_error("coupled floor/mesh velocity regression");
        bool baseline = argc > 2 && std::string(argv[2]) == "baseline";
        double bulk = floorTransfer(1, .5), one = floorTransfer(1, .005), many = floorTransfer(100, .005);
        std::cout << std::setprecision(16) << "bulk_v=" << bulk << " floor_1_v=" << one
                  << " floor_100_v=" << many << std::endl;
        if (argc < 2)
            throw std::runtime_error("verify_water CHARACTER [baseline]");
        Character mesh(argv[1], .016);
        mesh.build(0);
        auto p = allocate<Particle>(1);
        cudaCheck(cudaMemset(p.data(), 0, sizeof(Particle)));
        makeContact<<<1, 1>>>(mesh.view, p.data());
        Particle before, after;
        cudaCheck(cudaMemcpy(&before, p.data(), sizeof(Particle), cudaMemcpyDeviceToHost));
        auto result = allocate<SceneContact>(1);
        characterContact<<<1, 1>>>(p.data(), 1, mesh.view, .009, result.data(), true);
        cudaCheck(cudaMemcpy(&after, p.data(), sizeof(Particle), cudaMemcpyDeviceToHost));
        SceneContact check;
        cudaCheck(cudaMemcpy(&check, result.data(), sizeof(check), cudaMemcpyDeviceToHost));
        double velocityChange = (after.v - before.v).length(),
               affineChange = (after.c[0] - before.c[0]).length();
        std::cout << "separating_contact_velocity_change=" << velocityChange
                  << " affine_change=" << affineChange << " contacts=" << check.contacts
                  << " penetration=" << check.meshRemaining + check.sdfRemaining << std::endl;
        if (!baseline && (fabs(bulk - 1) > 1e-12 || fabs(one - 1) > 1e-12 || fabs(many - 1) > 1e-10 ||
                          velocityChange > 1e-12 || affineChange > 1e-12 || !check.contacts ||
                          check.meshRemaining + check.sdfRemaining))
            throw std::runtime_error("free-slip/separating contact regression");
        // Regression: a stationary-floor correction far from the character
        // must not impose the nearest (but inactive) animated mesh constraint.
        Particle floorOnly;
        floorOnly.x=Vec(.1,-1e-6,.1);floorOnly.v=Vec(.4,-.1,.3);
        floorOnly.c[0]=Vec(.2,.3,.4);
        cudaCheck(cudaMemcpy(p.data(),&floorOnly,sizeof(Particle),cudaMemcpyHostToDevice));
        characterContact<<<1,1>>>(p.data(),1,mesh.view,.009,result.data(),true,.005);
        cudaCheck(cudaMemcpy(&after,p.data(),sizeof(Particle),cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(&check,result.data(),sizeof(check),cudaMemcpyDeviceToHost));
        double floorOnlyError=(after.v-Vec(.4,0,.3)).length();
        std::cout<<"floor_only_velocity_error="<<floorOnlyError<<" mesh_contacts="<<check.contacts<<std::endl;
        if(floorOnlyError>1e-12 || check.contacts || after.x[1]<.005 || check.meshRemaining || check.sdfRemaining)
            throw std::runtime_error("floor-only contact activated a distant character");
        // Floor contact must have only the configured time-based air drag.
        Particle dummy;
        dummy.x = Vec(.4, .4, .4);
        Solver empty({dummy}, true, .05);
        empty.reset();
        Secondary spray;
        spray.x = Vec(3.0, .0001, 2.2);
        spray.v = Vec(1, 0, 0);
        auto secondary = allocate<Secondary>(1);
        auto secondaryCheck = allocate<WWCheck>(1);
        cudaCheck(cudaMemcpy(secondary.data(), &spray, sizeof(spray), cudaMemcpyHostToDevice));
        updateWhitewater<<<1, 1>>>(empty.d, mesh.view, secondary.data(), 1, .002, 8, secondaryCheck.data(),
                                   true);
        cudaCheck(cudaMemcpy(&spray, secondary.data(), sizeof(spray), cudaMemcpyDeviceToHost));
        double floorDragError = fabs(spray.v[0] - exp(-.35 * .002));
        std::cout << "secondary_floor_drag_error=" << floorDragError << std::endl;
        spray = Secondary();
        spray.x = Vec(.001, 2.8, .001);
        spray.v = Vec(-1, .7, -2);
        cudaCheck(cudaMemcpy(secondary.data(), &spray, sizeof(spray), cudaMemcpyHostToDevice));
        updateWhitewater<<<1, 1>>>(empty.d, mesh.view, secondary.data(), 1, .002, 8, secondaryCheck.data(),
                                   true, true);
        cudaCheck(cudaMemcpy(&spray, secondary.data(), sizeof(spray), cudaMemcpyDeviceToHost));
        double sideError = fabs(spray.x[0] - spray.radius) + fabs(spray.x[2] - spray.radius) +
                           fabs(spray.v[0]) + fabs(spray.v[2]) +
                           fabs(spray.v[1] - (.7 - 9.81 * .002) * exp(-.35 * .002));
        std::cout << "secondary_closed_wall_error=" << sideError << std::endl;
        if (sideError > 1e-12)
            throw std::runtime_error("closed wall secondary regression");
        std::vector<Particle> cloud;
        for (int x = 0; x < 16; ++x)
            for (int y = 0; y < 16; ++y)
                for (int z = 0; z < 16; ++z)
                {
                    Particle q;
                    q.x = Vec(.2 + .025 * x, .4 + .025 * y, .2 + .025 * z);
                    cloud.push_back(q);
                }
        Solver liquid(cloud, true, .05);
        liquid.reset();
        liquid.transfer(0, 0);
        liquid.advect(0, false);
        spray = Secondary();
        spray.x = Vec(.4, .6, .4);
        spray.v = Vec(0, 1, 0);
        cudaCheck(cudaMemcpy(secondary.data(), &spray, sizeof(spray), cudaMemcpyHostToDevice));
        updateWhitewater<<<1, 1>>>(liquid.d, mesh.view, secondary.data(), 1, .002, 8, secondaryCheck.data(),
                                   true);
        cudaCheck(cudaMemcpy(&spray, secondary.data(), sizeof(spray), cudaMemcpyDeviceToHost));
        std::cout << "newborn_spray_kind=" << spray.kind << std::endl;
        if (!baseline && (floorDragError > 1e-12 || spray.kind != 1))
            throw std::runtime_error("whitewater contact/launch regression");
        // Reabsorbed spray gets a foam lifetime measured from conversion,
        // rather than disappearing at its old, short spray expiry time.
        spray = Secondary();
        spray.x = Vec(.4, .74, .4);
        spray.v = Vec(0, -1, 0);
        spray.radius = .003;
        spray.age = .2;
        spray.life = .21;
        cudaCheck(cudaMemcpy(secondary.data(), &spray, sizeof(spray), cudaMemcpyHostToDevice));
        updateWhitewater<<<1, 1>>>(liquid.d, mesh.view, secondary.data(), 1, .002, 8, secondaryCheck.data(),
                                   true, false, 1.5, 1.8);
        cudaCheck(cudaMemcpy(&spray, secondary.data(), sizeof(spray), cudaMemcpyDeviceToHost));
        std::cout << "converted_foam_kind=" << spray.kind << " radius=" << spray.radius
                  << " remaining_life=" << spray.life - spray.age << std::endl;
        if (spray.kind != 0 || fabs(spray.radius - .006) > 1e-12 || spray.life - spray.age < 1.98)
            throw std::runtime_error("spray to foam lifetime regression");
        // A particle with no represented pressure cell and no velocity extension
        // is explicitly ballistic; it must not disappear or receive a zero field.
        dummy.v = Vec(1, 0, 0);
        Solver droplet({dummy}, true, .05);
        droplet.reset();
        excludePressure<<<blocks(droplet.d.count), 256>>>(droplet.d);
        droplet.transfer(0, 0);
        droplet.pressure();
        droplet.advect(0, false);
        double dropletError = (droplet.download()[0].v - dummy.v).length();
        std::cout << "unrepresented_pressure_cell_transfer_error=" << dropletError << std::endl;
        if (dropletError > 1e-12 || droplet.stats.unresolved != 1)
            throw std::runtime_error("unresolved droplet regression");
        // A fluid cell may have both vertical MAC faces blocked by a thin slit.
        // Only that component is ballistic; represented components retain G2P.
        dummy.v = Vec(1, 2, 3);
        dummy.c[1] = Vec(7, 8, 9);
        Solver slit({dummy}, true, .05);
        slit.reset();
        missingVerticalTransport<<<blocks(slit.d.count), 256>>>(slit.d, false);
        g2p<<<1, 1>>>(slit.d, slit.particles.data(), 1, .002, slit.missing.data(), false, -9.81);
        auto slitParticle = slit.download()[0];
        int counters[2];
        cudaCheck(cudaMemcpy(counters, slit.missing.data(), sizeof(counters), cudaMemcpyDeviceToHost));
        Vec expected(4, 2 - 9.81 * .002, 6);
        double slitError = (slitParticle.v - expected).length() + slitParticle.c[1].length() +
                           (slitParticle.x - dummy.x - expected * .002).length();
        std::cout << "blocked_component_transfer_error=" << slitError << std::endl;
        if (slitError > 1e-12 || counters[0] || counters[1] != 1)
            throw std::runtime_error("blocked component transfer regression");
        // An actual hole in a represented component must remain a hard error.
        cudaCheck(cudaMemcpy(slit.particles.data(), &dummy, sizeof(dummy), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemset(slit.missing.data(), 0, sizeof(counters)));
        missingVerticalTransport<<<blocks(slit.d.count), 256>>>(slit.d, true);
        g2p<<<1, 1>>>(slit.d, slit.particles.data(), 1, 0, slit.missing.data(), false, 0);
        cudaCheck(cudaMemcpy(counters, slit.missing.data(), sizeof(counters), cudaMemcpyDeviceToHost));
        std::cout << "represented_missing_component_errors=" << counters[0] << std::endl;
        if (!counters[0] || counters[1])
            throw std::runtime_error("represented transport hole was masked");
        mesh.build(2.2354);
        auto concave = allocate<double>(4);
        concaveContact<<<1, 1>>>(mesh.view, concave.data());
        double c[4];
        cudaCheck(cudaMemcpy(c, concave.data(), sizeof(c), cudaMemcpyDeviceToHost));
        std::cout << "concave_sdf_before=" << c[0] << " mesh_before=" << c[1] << " sdf_after=" << c[2]
                  << " mesh_after=" << c[3] << std::endl;
        if (c[0] >= 0 || c[1] != 0 || c[2] < 0 || c[3] != 0)
            throw std::runtime_error("concave sampled SDF contact regression");
        std::cout << (baseline ? "BASELINE measured" : "PASS water regressions") << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
