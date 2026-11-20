// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#include "Character.cuh"
#include <nanovdb/tools/cuda/PropagateSign.cuh>
#include <type_traits>
#include <filesystem>
#include <sstream>
#include <iomanip>
#include <iostream>
using namespace fluid;
// Both the committed and local PropagateSign APIs are supported. One keeps
// outputs on the operator; the other returns an owning tuple.
template<class Operator>
double timePropagation(Operator& op)
{
    Timer timer;
    if constexpr(std::is_void_v<decltype(op.propagate())>) {
        op.propagate();
        return timer.ms();
    } else {
        auto masks=op.propagate();
        return timer.ms();
    }
}
int main(int argc, char** argv)
{
    try {
        std::map<std::string,std::string> args;
        for(int i=1;i<argc;i+=2) {
            if(i+1>=argc) throw std::runtime_error("arguments require --key value");
            args[argv[i]]=argv[i+1];
        }
        const auto character=args.at("--character"), metrics=args.at("--metrics"), output=args.at("--output");
        const double h=args.count("--voxel-size")?std::stod(args["--voxel-size"]):1./240.;
        const int passes=args.count("--passes")?std::stoi(args["--passes"]):3;
        const int limit=args.count("--limit")?std::stoi(args["--limit"]):0;
        const bool trace=args.count("--trace")&&args["--trace"]=="1";
        if(passes<1||limit<0||h<=0) throw std::runtime_error("invalid profiling settings");
        std::ifstream source(metrics);std::string line;
        if(!std::getline(source,line)) throw std::runtime_error("missing metrics header");
        std::vector<double> times, dt;
        while(std::getline(source,line)) {
            if(line.empty())continue;
            std::istringstream row(line);std::string step,t,d;
            if(!std::getline(row,step,',')||!std::getline(row,t,',')||!std::getline(row,d,','))
                throw std::runtime_error("invalid metrics row");
            times.push_back(std::stod(t));dt.push_back(std::stod(d));
        }
        if(times.empty())throw std::runtime_error("empty replay");
        if(limit&&times.size()>size_t(limit)) {times.resize(limit);dt.resize(limit);}
        Character mesh(character,h);mesh.profileBuild=true;
        // Rebuild at the original restart time to warm the same pipeline.
        mesh.build(times.front()-dt.front());
        if(args.count("--diagnostic") && args["--diagnostic"]=="1") {
            mesh.build(times.front());
            nanovdb::tools::cuda::ConnectedComponents<Build> components(mesh.view.field.grid);
            auto labels=components.getVoxelLabelsAndCount();
            std::cout<<"DIAGNOSTIC voxel_surface_components="<<labels.second
                     <<" triangle_components="<<mesh.view.componentCount<<std::endl;
            nanovdb::Map map;map.set(h,nanovdb::Vec3d(0),1.);
            nanovdb::tools::cuda::MeshToGrid<Build> raster(mesh.view.vertices,mesh.vertexCount,
                mesh.view.triangles,mesh.triangleCount,map);
            raster.setNarrowBandWidth(3);raster.setVerbose(1);
            auto prototype=nanovdb::tools::cuda::sdf_detail::allocate<std::byte>(0,0);
            using Storage=decltype(prototype);
            Timer timer;
            auto result=raster.template getHandleAndUDFAndIndex<Storage,Storage>(prototype,prototype);
            std::cout<<"DIAGNOSTIC raster_api_wall_ms="<<timer.ms()<<std::endl;
            // Exact helper sequence from processSurface for one voxel component.
            // These extra synchronized diagnostic boundaries are not used in
            // the primary phaseMs replay timing above.
            if(labels.second==1) {
                namespace detail=nanovdb::tools::cuda::sdf_detail;
                const auto* grid=std::get<0>(result).template deviceGrid<Build>();
                const auto* udf=reinterpret_cast<const float*>(std::get<1>(result).data());
                Timer pruneTimer;
                auto derived=detail::pruneBarrier(grid,udf,float(h),0.f,0);
                std::cout<<"DIAGNOSTIC prune_barrier_ms="<<pruneTimer.ms()<<std::endl;
                const auto* derivedGrid=derived.template deviceGrid<Build>();
                Timer componentTimer;
                nanovdb::tools::cuda::ConnectedComponents<Build> derivedComponents(derivedGrid);
                auto componentLabels=derivedComponents.getVoxelLabelsAndCount();
                std::cout<<"DIAGNOSTIC nonbarrier_components_ms="<<componentTimer.ms()<<std::endl;
                std::cout<<"DIAGNOSTIC nonbarrier_components="<<componentLabels.second<<std::endl;
                Timer signTimer;
                auto derivedSign=detail::signNonBarrierComponents(derivedGrid,componentLabels.first.data(),0);
                std::cout<<"DIAGNOSTIC sign_nonbarrier_ms="<<signTimer.ms()<<std::endl;
                Timer injectTimer;
                auto partial=detail::injectNonBarrierSigns(grid,derivedGrid,derivedSign,0);
                std::cout<<"DIAGNOSTIC inject_sign_ms="<<injectTimer.ms()<<std::endl;
                Timer barrierTimer;
                auto sign=detail::signBarrierAsInterior(grid,partial,0);
                std::cout<<"DIAGNOSTIC interior_barrier_sign_ms="<<barrierTimer.ms()<<std::endl;
                nanovdb::tools::cuda::PropagateSign<Build> propagator(grid,sign.data());
                std::cout<<"DIAGNOSTIC propagate_masks_ms="<<timePropagation(propagator)<<std::endl;
            }
            return 0;
        }
        std::filesystem::create_directories(std::filesystem::path(output).parent_path());
        std::ofstream result(output);
        result<<"pass,sample,time,api_wall_ms,character_build_ms,prepare_ms,rasterize_ms,partition_ms,process_surfaces_ms,resolve_signs_ms,masks_finalize_ms,active_voxels,leaves,grid_bytes,barrier_signing\n";
        for(int pass=0;pass<passes;++pass) {
            for(size_t i=0;i<times.size();++i) {
                mesh.traceSdfPipeline=trace&&pass==0&&i==0;
                const double total=mesh.build(times[i]);
                const auto tree=nanovdb::util::cuda::DeviceGridTraits<Build>::getTreeData(mesh.view.field.grid);
                const auto bytes=nanovdb::util::cuda::DeviceGridTraits<Build>::getGridSize(mesh.view.field.grid);
                result<<std::setprecision(12)<<pass<<','<<i<<','<<times[i]<<','<<mesh.sdfPipelineMs<<','<<total<<','<<mesh.prepareMs;
                for(float phase:mesh.sdfPhaseMs)result<<','<<phase;
                result<<','<<tree.mVoxelCount<<','<<tree.mNodeCount[0]<<','<<bytes<<",interior\n";
            }
            result.flush();
            std::cout<<"PASS replay "<<pass<<" samples="<<times.size()<<std::endl;
        }
    } catch(const std::exception& e) {
        std::cerr<<"FAIL "<<e.what()<<std::endl;return 1;
    }
}
