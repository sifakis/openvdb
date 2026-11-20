// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <thrust/copy.h>
#include <cuda/std/tuple>
#include <thrust/iterator/zip_iterator.h>
namespace fluid {
template<class T> void reserveDevice(Buffer<T> &buffer, size_t required, size_t live=0)
{
    if(buffer.size()>=required) return;
    size_t capacity=std::max(required,buffer.size()+buffer.size()/2+256);
    auto next=allocate<T>(capacity);
    if(live) cudaCheck(cudaMemcpyAsync(next.data(),buffer.data(),live*sizeof(T),cudaMemcpyDeviceToDevice));
    buffer=std::move(next);
}
struct KeepFluid
{
    template<class Tuple> __device__ bool operator()(const Tuple &q) const
    {
        const auto &p=cuda::std::get<0>(q);
        return !(p.x[1]<.18 && (p.x[0]<.14 || p.x[0]>3.06 || p.x[2]<.14 || p.x[2]>2.26));
    }
};
__global__ void emitDevice(Particle *particles,uint64_t *ids,int start,int count,uint64_t first,
                           double radius,double speed,double x,double z,double depth)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count) return;
    uint64_t id=first+i;
    double r=radius*sqrt(random01(id*3+19)),angle=2*M_PI*random01(id*3+20);
    Particle p;
    p.x=Vec(x+r*cos(angle),2.55-depth*random01(id*3+21),z+r*sin(angle));
    p.v=Vec(0,-speed,0);particles[start+i]=p;ids[start+i]=id;
}
struct ParticleManager
{
    Buffer<uint64_t> ids, scratchIds;
    Buffer<Particle> scratch;
    void upload(const std::vector<uint64_t>& initial) {
        reserveDevice(ids,initial.size());
        cudaCheck(cudaMemcpy(ids.data(),initial.data(),initial.size()*sizeof(uint64_t),cudaMemcpyHostToDevice));
    }
    std::vector<uint64_t> download(int n) {
        std::vector<uint64_t> out(n);
        cudaCheck(cudaMemcpy(out.data(),ids.data(),n*sizeof(uint64_t),cudaMemcpyDeviceToHost));return out;
    }
    uint64_t drain(Solver &s) {
        int old=s.particleCount;
        reserveDevice(scratch,old);reserveDevice(scratchIds,old);
        auto in=thrust::make_zip_iterator(cuda::std::make_tuple(s.particles.data(),ids.data()));
        auto out=thrust::make_zip_iterator(cuda::std::make_tuple(scratch.data(),scratchIds.data()));
        auto end=thrust::copy_if(thrust::device,in,in+old,out,KeepFluid{});
        s.particleCount=int(end-out);
        std::swap(s.particles,scratch);std::swap(ids,scratchIds);
        return old-s.particleCount;
    }
    void emit(Solver &s,int n,uint64_t first,double radius,double speed,double x,double z,double depth) {
        if(!n) return;
        if(uint64_t(s.particleCount)+n>=uint64_t(INT_MAX)) throw std::runtime_error("particle index capacity exceeded");
        reserveDevice(s.particles,s.particleCount+n,s.particleCount);reserveDevice(ids,s.particleCount+n,s.particleCount);
        emitDevice<<<blocks(n),256>>>(s.particles.data(),ids.data(),s.particleCount,n,first,radius,speed,x,z,depth);
        s.particleCount+=n;
    }
};
struct ContactReduce
{
    __hostdev__ SceneContact operator()(SceneContact a,SceneContact b) const {
        a.sdfRemaining+=b.sdfRemaining;a.meshRemaining+=b.meshRemaining;a.contacts+=b.contacts;a.attempted+=b.attempted;
        a.depth=maxValue(a.depth,b.depth);a.normalError=maxValue(a.normalError,b.normalError);a.impulse=maxValue(a.impulse,b.impulse);return a;
    }
};
} // namespace fluid
