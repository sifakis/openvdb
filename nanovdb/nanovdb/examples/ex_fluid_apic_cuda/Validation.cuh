// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifndef EX_FLUID_VALIDATION_CUH
#define EX_FLUID_VALIDATION_CUH
#include "Fluid.cuh"
namespace fluid {
void transferTest(double h = .05)
{
    for (bool affine : {false, true}) {
        std::vector<Particle> initial;
        for (int x = 5; x < 19; ++x)
            for (int y = 5; y < 19; ++y)
                for (int z = 5; z < 19; ++z)
                    for (int j = 0; j < 8; ++j) {
                        Particle p;
                        p.x = Vec(x + 0.25 + 0.5 * (j & 1), y + 0.25 + 0.5 * ((j >> 1) & 1),
                                  z + 0.25 + 0.5 * ((j >> 2) & 1)) *
                              h;
                        if (affine) {
                            p.c[0] = Vec(0.2, -0.3, 0.1);
                            p.c[1] = Vec(0.4, 0.1, -0.2);
                            p.c[2] = Vec(-0.1, 0.2, -0.3);
                        }
                        for (int a = 0; a < 3; ++a)
                            p.v[a] = Vec(0.7, -0.2, 0.3)[a] + p.c[a].dot(p.x);
                        initial.push_back(p);
                    }
        Solver s(initial, false, h);
        s.reset();
        s.transfer(0, 0);
        s.advect(0, false);
        auto actual = s.download();
        double v = 0, c = 0;
        for (size_t k = 0; k < initial.size(); ++k)
            for (int a = 0; a < 3; ++a) {
                v = std::max<double>(v, fabs(actual[k].v[a] - initial[k].v[a]));
                for (int b = 0; b < 3; ++b)
                    c = std::max<double>(c, fabs(actual[k].c[a][b] - initial[k].c[a][b]));
            }
        std::cout << "transfer affine=" << affine << " velocity_max=" << v << " affine_max=" << c
                  << std::endl;
        if (v > (singlePrecision?5e-6:1e-10) || c > (singlePrecision?5e-4:1e-9))
            throw std::runtime_error("APIC transfer failed");
    }
}
} // namespace fluid
#endif
