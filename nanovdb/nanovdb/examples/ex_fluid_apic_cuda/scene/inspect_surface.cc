// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
// Geometry-only audit: the same particle cache can be compared across mesh settings.
#include "RawSdf.h"
#include <nanovdb/HostBuffer.h>
#include <algorithm>
#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <vector>
using V = surface::Vec;
using C = surface::Coord;
struct Hash {
    size_t operator()(const C& c) const {
        return size_t(c.x()) * 73856093u ^ size_t(c.y()) * 19349663u ^ size_t(c.z()) * 83492791u;
    }
};
struct Sample { V x, v; };
double quantile(std::vector<double> v, double f) {
    if (v.empty()) return 0;
    std::sort(v.begin(),v.end());
    return v[std::min(size_t(f*(v.size()-1)),v.size()-1)];
}
int main(int argc,char** argv) {
    try {
        if(argc<6 || argc>7) throw std::runtime_error("usage: inspect_surface CACHE SURFACE FRAME OBSTACLE_H PARTICLE_SPACING [BODY_BAND]");
        char tag[32];std::snprintf(tag,sizeof(tag),"%06d",std::stoi(argv[3]));
        std::string base=std::string(argv[1])+"/frame_"+tag;
        double dx=std::stod(argv[5]);
        std::ifstream raw(base+"_obstacle.grid",std::ios::binary|std::ios::ate);
        if(!raw) throw std::runtime_error("missing obstacle");
        auto buffer=nanovdb::HostBuffer::create(raw.tellg());raw.seekg(0);
        raw.read((char*)buffer.data(),buffer.size());
        surface::RawSdf sdf; sdf.grid=(const surface::Grid*)buffer.data();
        sdf.h=std::stod(argv[4]);sdf.band=3*sdf.h;
        const double bodyBand=argc>6 ? std::stod(argv[6]) : 2*sdf.h;
        if(!(bodyBand>0 && bodyBand<sdf.band && dx>0))throw std::runtime_error("invalid probe band/spacing");
        if(!raw || sdf.grid->gridSize()!=buffer.size())throw std::runtime_error("bad SDF");
        std::ifstream csv(base+".csv");std::string line;std::getline(csv,line);
        std::vector<Sample> particles;
        std::unordered_map<C,std::vector<int>,Hash> bins;
        auto cell=[&](V x){return surface::floorCoord(x/dx);};
        std::vector<double> wallSpeed,wallVertical,wallSeparation;
        while(std::getline(csv,line)) {
            std::replace(line.begin(),line.end(),',',' ');std::istringstream row(line);
            uint64_t id;Sample p;row>>id>>p.x[0]>>p.x[1]>>p.x[2]>>p.v[0]>>p.v[1]>>p.v[2];
            if(!row)throw std::runtime_error("bad particle");
            bins[cell(p.x)].push_back(particles.size());particles.push_back(p);
            double phi=sdf.sdf(p.x);
            if(p.x[1]>.3 && phi<bodyBand) {
                wallSpeed.push_back(p.v.length());wallVertical.push_back(p.v[1]);wallSeparation.push_back(phi);
            }
        }
        std::ifstream mesh(std::string(argv[2])+"/water_"+tag+".meshbin",std::ios::binary);
        uint64_t header[3];mesh.read((char*)header,sizeof(header));
        if(!mesh || header[0]!=0x4d45534830303031ULL)throw std::runtime_error("bad mesh");
        std::vector<std::array<float,3>> vertices(header[1]);mesh.read((char*)vertices.data(),vertices.size()*12);
        if(!mesh)throw std::runtime_error("truncated mesh");
        std::vector<double> distances,wallDistances;uint64_t unsupported=0;
        for(auto q:vertices) {
            V x(q[0],q[1],q[2]);if(x[1]<.3)continue;
            C c=cell(x);double best=1e6;
            for(int a=-3;a<=3;++a)for(int b=-3;b<=3;++b)for(int z=-3;z<=3;++z) {
                auto it=bins.find(c+C(a,b,z));if(it==bins.end())continue;
                for(int j:it->second) best=std::min(best,(particles[j].x-x).lengthSqr());
            }
            double distance=sqrt(best); distances.push_back(distance);
            unsupported += distance>dx;
            if(sdf.sdf(x)<bodyBand)wallDistances.push_back(distance);
        }
        std::cout<<std::setprecision(12)<<"{\n\"frame\":"<<std::stoi(argv[3])
            <<",\n\"body_band_m\":"<<bodyBand
            <<",\n\"body_particles\":"<<wallSpeed.size()
            <<",\n\"body_particle_speed_median_m_s\":"<<quantile(wallSpeed,.5)
            <<",\n\"body_particle_vertical_median_m_s\":"<<quantile(wallVertical,.5)
            <<",\n\"body_particle_clearance_p95_m\":"<<quantile(wallSeparation,.95)
            <<",\n\"airborne_mesh_vertices\":"<<distances.size()
            <<",\n\"nearest_particle_distance_p95_m\":"<<quantile(distances,.95)
            <<",\n\"body_nearest_particle_distance_p95_m\":"<<quantile(wallDistances,.95)
            <<",\n\"vertices_farther_than_particle_spacing_fraction\":"<<double(unsupported)/std::max(size_t(1),distances.size())
            <<",\n\"note\":\"Distances measure render geometry support, not fluid mass or viscosity; body region is y>0.3 and obstacle SDF<body_band_m.\"\n}\n";
    }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
