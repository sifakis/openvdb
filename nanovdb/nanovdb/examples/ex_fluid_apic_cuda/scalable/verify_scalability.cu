// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#include "../scene/Whitewater.cuh"
#include "ParticleOps.cuh"
#include <iostream>
#include "../Validation.cuh"
using namespace fluid;
__device__ bool active(Coord c,int n,int mode)
{
    if(mode==2) return c[0]>=0&&c[1]>=0&&c[2]>=0&&c[0]<n&&c[1]<n&&c[2]<n;
    bool inside=c[0]>=1&&c[1]>=1&&c[2]>=1&&c[0]<n-1&&c[1]<n-1&&c[2]<n-1;
    if(mode==1) inside=inside && c[0]!=n/2 && !(c[1]>n/3&&c[1]<n/2&&c[2]<n/2);
    return inside;
}
__device__ Real solution(Coord c,int n,int mode)
{
    if(!active(c,n,mode)) return 0;
    return Real(sin(.071*c[0])+cos(.093*c[1])+.25*sin(.117*c[2]));
}
__global__ void manufacture(Domain d,int mode)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(!i||i>=d.count)return;
    auto q=d.coord(i);auto &c=d.cells[i];c.fluid=active(q,d.nx,mode);
    for(int a=0;a<3;++a) {
        // Fractional/open/blocked faces with nonuniform coefficients.
        c.open[a]=(q[0]+3*q[1]+5*q[2]+a)%11==0 ? Real(.125) : Real(1);
        if(!d.inside(q)||!d.inside(q-axis(a))) c.open[a]=0;
        c.u[a]=solution(q,d.nx,mode)-solution(q-axis(a),d.nx,mode);
        c.wall[a]=0;
    }
}
struct PressureError
{
    Domain d;
    __device__ double operator()(int i)const {
        if(!i)return 0;
        double e=0;for(int a=0;a<3;++a) if(d.cells[i].fluid||d.cells[d.neighbor(i,2*a)].fluid)
            e=maxValue(e,fabs(double(d.cells[i].u[a])));return e;
    }
};
void particlesTest()
{
    std::vector<Particle> p(128);std::vector<uint64_t> ids(128);
    for(int i=0;i<128;++i){p[i].x=Vec(i%2?.5:.05,.1,.5);p[i].v=Vec(1,2,3);ids[i]=i;}
    Solver s(p,true,.05,Vec(1.6,1.6,1.2));ParticleManager manager;manager.upload(ids);
    auto removed=manager.drain(s);if(removed!=64||s.particleCount!=64) throw std::runtime_error("compaction count");
    auto out=manager.download(64);for(int i=0;i<64;++i)if(out[i]!=uint64_t(2*i+1))throw std::runtime_error("stable ID compaction");
    manager.emit(s,101,128,.21,.36,1.0,.6,.004);
    auto state=s.download();auto keys=manager.download(s.particleCount);
    for(int i=64;i<s.particleCount;++i){uint64_t id=128+i-64;double r=.21*sqrt(random01(id*3+19)),angle=2*M_PI*random01(id*3+20);
        Vec x(1.+r*cos(angle),2.55-.004*random01(id*3+21),.6+r*sin(angle));
        if(keys[i]!=id||(state[i].x-x).length()>1e-6)throw std::runtime_error("emitter reference mismatch");}
    // Compare GPU affine CFL maximum to a CPU reference without changing state.
    double speed=1;for(auto q:state)for(int a=0;a<3;++a){double v=fabs(double(q.v[a]));for(int b=0;b<3;++b)v+=1.5*s.d.h*fabs(double(q.c[a][b]));speed=std::max(speed,v);}
    if(fabs(s.timestep()-std::min(.005,.25*s.d.h/(speed+1)))>1e-12)throw std::runtime_error("GPU timestep reduction");
}
int main(int argc,char **argv)
{
    try {
        int n=argc>1?std::stoi(argv[1]):32,mode=argc>2?std::stoi(argv[2]):0;
        std::string method=argc>3?argv[3]:"mgpcg";
        particlesTest();
        if(n==32 && mode==1) transferTest();
        std::vector<Particle> initial(1);initial[0].x=Vec(.5);
        Solver s(initial,false,1./n,Vec(1));s.jacobiPressure=method=="jacobi";s.pressureMaxIterations=2000;s.reset();s.validateTopology();
        manufacture<<<blocks(s.d.count),256>>>(s.d,mode);
        auto rhs=allocate<Real>(s.d.count),diagonal=allocate<Real>(s.d.count);
        assemble<<<blocks(s.d.count),256>>>(s.d,rhs.data(),diagonal.data(),s.missing.data());
        double galerkin=0;
        {Multigrid hierarchy(s.d,diagonal.data(),false);galerkin=hierarchy.verifyGalerkin();}
        if(galerkin>(singlePrecision?2e-5:1e-12)) throw std::runtime_error("Galerkin identity failed");
        double before=s.divergence();Timer timer;s.pressure();double total=timer.ms();
        double error=reduceDevice(s.d.count,PressureError{s.d},0.,cuda::maximum<double>());
        double bound=singlePrecision?2e-3:2e-6;
        if(error>bound)throw std::runtime_error("manufactured pressure gradient error="+std::to_string(error));
        auto st=s.stats;
        std::cout<<std::setprecision(12)<<"{\"pass\":true,\"precision\":\""<<(singlePrecision?"float":"double")
            <<"\",\"n\":"<<n<<",\"mode\":"<<mode<<",\"method\":\""<<method<<"\",\"cells\":"<<s.d.count
            <<",\"cell_bytes\":"<<sizeof(Cell)<<",\"particle_bytes\":"<<sizeof(Particle)
            <<",\"iterations\":"<<st.iterations<<",\"levels\":"<<st.mgLevels
            <<",\"relative_residual\":"<<st.residual<<",\"div_before\":"<<before<<",\"div_after\":"<<st.divAfter
            <<",\"galerkin_error\":"<<galerkin<<",\"gradient_error\":"<<error<<",\"setup_ms\":"<<st.mgSetup<<",\"solve_ms\":"<<st.mgSolve
            <<",\"total_ms\":"<<total<<",\"mg_bytes\":"<<st.mgBytes<<"}"<<std::endl;
    } catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<std::endl;return 1;}
}
