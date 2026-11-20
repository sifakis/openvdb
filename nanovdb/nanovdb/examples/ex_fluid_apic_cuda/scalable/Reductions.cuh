// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Scalar control/readback only. All O(particles/cells) work remains on device.
struct ParticleMeasure
{
    const Particle *p; Domain d; bool affine;
    __device__ double operator()(int i) const
    {
        double speed=affine?1:0;
        for(int a=0;a<3;++a) {
            if(!isfinite(p[i].x[a]) || !isfinite(p[i].v[a])) return INFINITY;
            if(!affine && (p[i].x[a]<0 || p[i].x[a]>(a==0?d.nx:a==1?d.ny:d.nz)*d.h+contactTolerance)) return INFINITY;
            double bound=fabs(double(p[i].v[a]));
            if(affine) for(int b=0;b<3;++b) {
                if(!isfinite(p[i].c[a][b])) return INFINITY;
                bound+=1.5*d.h*fabs(double(p[i].c[a][b]));
            }
            speed=maxValue(speed,bound);
        }
        return speed;
    }
};
struct FluxSummary {double square=0;int count=0;};
struct FluxMeasure
{
    Domain d;
    __device__ FluxSummary operator()(int i) const {
        if(!i || !d.cells[i].fluid) return {};
        double f=netFlux(d,i)/d.h;return {f*f,1};
    }
};
struct FluxAdd
{
    __hostdev__ FluxSummary operator()(FluxSummary a,FluxSummary b) const {return {a.square+b.square,a.count+b.count};}
};
struct TopologyMeasure
{
    Domain d;
    __device__ int operator()(int i) const {
        if(!i) return 0;
        auto c=d.coord(i);if(d.index(c)!=i) return 1;
        for(int a=0;a<6;++a) {int j=d.neighbor(i,a);if(j && (j>=d.count || d.neighbor(j,a^1)!=i)) return 1;}
        return 0;
    }
};
template<class F, class T, class Op> T reduceDevice(int n, F f, T initial, Op op)
{
    return thrust::transform_reduce(thrust::device,thrust::counting_iterator<int>(0),thrust::counting_iterator<int>(n),f,initial,op);
}
