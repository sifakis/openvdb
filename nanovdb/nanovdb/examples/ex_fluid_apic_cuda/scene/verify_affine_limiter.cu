// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#include "Whitewater.cuh"

#include "Checkpoint.h"
#include <iostream>
using namespace fluid;
int main(int argc, char **argv)
{
    try {
        if (argc != 3) throw std::runtime_error("verify_affine_limiter CHARACTER CACHE");
        CheckpointHeader header;
        std::ifstream input(std::string(argv[2]) + "/checkpoint.bin", std::ios::binary);
        input.read(reinterpret_cast<char *>(&header), sizeof(header));
        if (!input || header.magic != 0x464c554944303034ULL || header.particleSize != sizeof(Particle))
            throw std::runtime_error("invalid checkpoint");
        std::vector<Particle> source(header.count);
        input.read(reinterpret_cast<char *>(source.data()), source.size()*sizeof(Particle));
        if (!input) throw std::runtime_error("truncated particles");
        Character mesh(argv[1], header.h*.5);
        mesh.build(header.time);
        Solver solver(source, true, header.h, Vec(3.2,3.2,2.4));
        solver.includeCutCells = true;
        solver.reset();
        obstacleCells<<<blocks(solver.d.count),256>>>(solver.d,mesh.view.field);
        characterFaces<<<blocks(solver.d.count),256>>>(solver.d,mesh.view);
        solver.transfer(0,0);
        solver.pressure();
        solver.transportScratch();
        prepareTransport<<<blocks(solver.d.count),256>>>(solver.d);
        for (int k=0;k<6;++k) {
            extrapolate<<<blocks(solver.d.count),256>>>(solver.d);
            finishExtrapolate<<<blocks(solver.d.count),256>>>(solver.d);
        }
        std::vector<Particle> reference;
        double energy[2]={0,0}, affine[2]={0,0}, meanError=0;
        for (int limited=0;limited<2;++limited) {
            cudaCheck(cudaMemcpy(solver.particles.data(),source.data(),source.size()*sizeof(Particle),cudaMemcpyHostToDevice));
            cudaCheck(cudaMemset(solver.missing.data(),0,3*sizeof(int)));
            g2p<<<blocks(source.size()),256>>>(solver.d,solver.particles.data(),source.size(),0,solver.missing.data(),false,0,limited);
            auto result=solver.download();
            if (!limited) reference=result;
            for(size_t i=0;i<result.size();++i) {
                auto p=result[i];
                meanError=std::max(meanError,(p.v-reference[i].v).length());
                double c2=0;
                for(int a=0;a<3;++a)c2+=p.c[a].lengthSqr();
                affine[limited]+=.25*header.h*header.h*c2;
                energy[limited]+=p.v.lengthSqr()+.25*header.h*header.h*c2;
            }
        }
        double fraction=(energy[0]-energy[1])/std::max(1e-30,energy[0]);
        double mass=1000*header.h*header.h*header.h/header.ppc;
        std::cout << std::setprecision(14) << "{\"time\":" << header.time
            << ",\"particles\":" << source.size() << ",\"mean_velocity_max_difference\":" << meanError
            << ",\"apic_energy_before_J\":" << .5*mass*energy[0]
            << ",\"apic_energy_after_J\":" << .5*mass*energy[1]
            << ",\"affine_energy_before_J\":" << .5*mass*affine[0]
            << ",\"affine_energy_after_J\":" << .5*mass*affine[1]
            << ",\"removed_energy_fraction\":" << fraction << "}" << std::endl;
        if (meanError>1e-12 || fraction< -1e-12 || !std::isfinite(fraction))
            throw std::runtime_error("affine-only limiter energy regression");
    } catch (const std::exception &e) {std::cerr << e.what() << std::endl; return 1;}
}
