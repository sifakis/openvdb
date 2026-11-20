// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#include "Character.cuh"
#include <iostream>
using namespace fluid;

__global__ void inspectContact(MeshView mesh, Particle *p, double *out)
{
    out[0] = mesh.field.sdf(p[0].x);
    out[1] = mesh.inside(p[0].x);
}

int main(int argc, char **argv)
{
    try {
        if (argc != 2) throw std::runtime_error("verify_contact_escape GROUNDED_CHARACTER");
        Character mesh(argv[1], 1.6 / 384);
        struct Case {double time; Vec position;};
        const Case cases[] = {
            {.72899025169262355, Vec(1.6434561014175415, .82601839303970337, 1.0975772142410278)},
            {1.7722409880587067, Vec(1.681855320930481, .73508858680725098, 1.0907140970230103)}};
        auto device = allocate<Particle>(1);
        auto values = allocate<double>(2);
        auto checks = allocate<SceneContact>(1);
        for (const auto &c : cases) {
            mesh.build(c.time);
            Particle p; p.x = c.position;
            cudaCheck(cudaMemcpy(device.data(), &p, sizeof(p), cudaMemcpyHostToDevice));
            inspectContact<<<1, 1>>>(mesh.view, device.data(), values.data());
            double before[2];
            cudaCheck(cudaMemcpy(before, values.data(), sizeof(before), cudaMemcpyDeviceToHost));
            if (before[0] >= 0 || before[1] != 0)
                throw std::runtime_error("recorded SDF/triangle discrepancy was not reproduced");
            characterContact<<<1, 1>>>(device.data(), 1, mesh.view, (3.2 / 384) * .25,
                                       checks.data(), true, double(Real(.1) * Real(3.2 / 384)));
            Particle after; SceneContact result;
            cudaCheck(cudaMemcpy(&after, device.data(), sizeof(after), cudaMemcpyDeviceToHost));
            cudaCheck(cudaMemcpy(&result, checks.data(), sizeof(result), cudaMemcpyDeviceToHost));
            const double displacement = (after.x - p.x).length();
            std::cout << std::setprecision(12) << "time=" << c.time << " sdf_before=" << before[0]
                      << " depth_after=" << result.depth << " displacement=" << displacement
                      << " inward_speed=" << result.normalError << std::endl;
            if (result.meshRemaining || result.sdfRemaining || result.normalError > contactTolerance ||
                displacement > .02 || !std::isfinite(after.v.length()) || after.v.length() > 50)
                throw std::runtime_error("contact escape regression failed");
        }
        std::cout << "PASS contact escape" << std::endl;
    } catch (const std::exception &e) {std::cerr << e.what() << std::endl; return 1;}
}
