// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Included inside namespace fluid. Geometric aggregation P is constant over
// 2^3 blocks, R=P^T, and coarse A=P^T A P (including cut faces and air terms).
// Symmetric weighted-Jacobi pre/post smoothing makes the fixed V cycle an SPD
// preconditioner. No residual-dependent inner solve changes that operator.
struct MGLevel
{
    Domain domain{};
    const Grid *grid = nullptr;
    uint32_t *address = nullptr;
    int n = 0, *parent = nullptr;
    Real *diag = nullptr, *low = nullptr, *b = nullptr, *x = nullptr, *tmp = nullptr;
    bool fine = false;
    __device__ Coord coord(int i) const
    {
        if (fine) return domain.coord(i);
        auto a=address[i];
        return grid->tree().getFirstNode<0>()[a>>9].offsetToGlobalCoord(a&511);
    }
    __device__ int neighbor(int i, int j) const
    {
        if (fine) return domain.neighbor(i,j);
        Coord c=coord(i); c[j/2]+=(j&1)?1:-1;
        auto acc=grid->getAccessor(); return int(acc.getValue(c));
    }
    __device__ Real weight(int i, int j) const
    {
        const int k=(j&1)?neighbor(i,j):i;
        return fine ? domain.cells[k].open[j/2] : low[3*k+j/2];
    }
    __device__ Real apply(const Real *v, int i) const
    {
        if (!i || diag[i]<=0) return 0;
        Real y=diag[i]*v[i];
        auto acc=(fine?domain.grid:grid)->getAccessor();const Coord c=coord(i);
        for(int j=0;j<6;++j) {
            Coord q=c;q[j/2]+=(j&1)?1:-1;int k=int(acc.getValue(q));
            if(k && diag[k]>0) {
                int faceIndex=(j&1)?k:i;
                Real w=fine?domain.cells[faceIndex].open[j/2]:low[3*faceIndex+j/2];
                y-=w*v[k];
            }
        }
        return y;
    }
};
__global__ void mgCoordinates(MGLevel level, Coord *out)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<level.n-1) out[i]=level.coord(i+1)>>1;
}
__global__ void mgEnumerate(MGLevel level)
{
    int leaf=blockIdx.x, offset=threadIdx.x;
    const auto &node=level.grid->tree().getFirstNode<0>()[leaf];
    if(node.isActive(offset)) level.address[node.getValue(offset)]=(uint32_t(leaf)<<9)|offset;
}
__global__ void mgParents(MGLevel f, MGLevel c)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<f.n) { auto acc=c.grid->getAccessor(); f.parent[i]=i?int(acc.getValue(f.coord(i)>>1)):0; }
}
__global__ void mgGalerkin(MGLevel f, MGLevel c)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(!i || i>=f.n || f.diag[i]<=0) return;
    int p=f.parent[i];
    Real boundary=f.diag[i];
    for(int j=0;j<6;++j) {
        int k=f.neighbor(i,j);Real w=f.weight(i,j);
        bool fluid=k && f.diag[k]>0;
        if(!fluid) continue;
        boundary-=w;
        if(f.parent[k]==p) continue;
        atomicAdd(c.diag+p,w);
        if(!(j&1)) atomicAdd(c.low+3*p+j/2,w);
    }
    // Preserve accumulated Dirichlet contributions at EVERY level, rather
    // than losing them after the first coarsening of the fluid/air boundary.
    atomicAdd(c.diag+p,boundary);

}
struct CGState
{
    double norm=0, residual=0, rz=0, alpha=0, beta=0, sum=0;
    int iterations=0, breakdown=0, restarts=0;
};
namespace mg_detail {
namespace cg = cooperative_groups;
__device__ double reduce(cg::grid_group g, double value, double *partials, CGState *state)
{
    __shared__ double shared[256];
    shared[threadIdx.x]=value; __syncthreads();
    for(int d=128;d;d>>=1) {
        if(threadIdx.x<d) shared[threadIdx.x]+=shared[threadIdx.x+d];
        __syncthreads();
    }
    if(!threadIdx.x) partials[blockIdx.x]=shared[0];
    g.sync();
    if(!g.thread_rank()) { double sum=0; for(unsigned i=0;i<gridDim.x;++i) sum+=partials[i]; state->sum=sum; }
    g.sync(); return state->sum;
}
__device__ double dot(cg::grid_group g, int n, const Real *a, const Real *b, double *scratch, CGState *state)
{
    double v=0;
    for(int i=int(g.thread_rank());i<n;i+=int(g.size())) v+=double(a[i])*double(b[i]);
    return reduce(g,v,scratch,state);
}
__device__ void smooth(cg::grid_group g, MGLevel l, int iterations)
{
    for(int pass=0;pass<iterations;++pass) {
        for(int i=int(g.thread_rank());i<l.n;i+=int(g.size()))
            l.tmp[i]=l.diag[i]>0 ? l.x[i]+Real(2./3.)*(l.b[i]-l.apply(l.x,i))/l.diag[i] : 0;
        g.sync();
        for(int i=int(g.thread_rank());i<l.n;i+=int(g.size())) l.x[i]=l.tmp[i];
        g.sync();
    }
}
__device__ void cycle(cg::grid_group g, MGLevel *levels, int count, const Real *r, bool jacobi)
{
    auto f=levels[0];
    for(int i=int(g.thread_rank());i<f.n;i+=int(g.size())) {
        f.b[i]=r[i]; f.x[i]=jacobi && f.diag[i]>0 ? r[i]/f.diag[i] : 0;
    }
    g.sync(); if(jacobi) return;
    for(int k=0;k<count-1;++k) {
        auto l=levels[k], c=levels[k+1]; smooth(g,l,2);
        for(int i=int(g.thread_rank());i<c.n;i+=int(g.size())) c.x[i]=c.b[i]=0;
        g.sync();
        for(int i=int(g.thread_rank());i<l.n;i+=int(g.size()))
            if(i && l.diag[i]>0) atomicAdd(c.b+l.parent[i],l.b[i]-l.apply(l.x,i));
        g.sync();
    }
    smooth(g,levels[count-1],48);
    for(int k=count-2;k>=0;--k) {
        auto l=levels[k], c=levels[k+1];
        for(int i=int(g.thread_rank());i<l.n;i+=int(g.size()))
            if(i && l.diag[i]>0) l.x[i]+=c.x[l.parent[i]];
        g.sync(); smooth(g,l,2);
    }
}
} // namespace mg_detail
__global__ void mgpcgPersistent(MGLevel *levels, int levelCount, const Real *b,
                               Real *p, Real *r, Real *s, Real *as,
                               double *scratch, CGState *state, int maxIterations,
                               double tolerance, bool jacobi)
{
    auto g=cooperative_groups::this_grid(); auto f=levels[0]; const int n=f.n;
    for(int i=int(g.thread_rank());i<n;i+=int(g.size())) {p[i]=0;r[i]=b[i];}
    g.sync();
    double norm=sqrt(mg_detail::dot(g,n,b,b,scratch,state));
    if(!g.thread_rank()) {state->norm=norm;state->residual=norm;}
    g.sync();
    const double target=maxValue(1e-12,norm*tolerance);
    if(norm<=target) return;
    mg_detail::cycle(g,levels,levelCount,r,jacobi);
    double rz=mg_detail::dot(g,n,r,f.x,scratch,state);
    if(!g.thread_rank()) state->rz=rz;
    for(int i=int(g.thread_rank());i<n;i+=int(g.size())) s[i]=f.x[i];
    g.sync();
    for(int it=0;it<maxIterations;++it) {
        for(int i=int(g.thread_rank());i<n;i+=int(g.size())) as[i]=f.apply(s,i);
        g.sync();
        double denom=mg_detail::dot(g,n,s,as,scratch,state);
        if(!g.thread_rank()) {
            if(!(denom>0) || !isfinite(denom)) state->breakdown=1;
            else state->alpha=state->rz/denom;
        }
        g.sync(); if(state->breakdown) break;
        for(int i=int(g.thread_rank());i<n;i+=int(g.size())) {
            p[i]+=Real(state->alpha)*s[i];r[i]-=Real(state->alpha)*as[i];
        }
        g.sync();
        double rn=sqrt(mg_detail::dot(g,n,r,r,scratch,state));
        // Reliable true-residual checks prevent false convergence in float.
        bool restart=rn<=target || (singlePrecision && (it+1)%64==0);
        if(restart) {
            for(int i=int(g.thread_rank());i<n;i+=int(g.size())) r[i]=b[i]-f.apply(p,i);
            g.sync(); rn=sqrt(mg_detail::dot(g,n,r,r,scratch,state));
        }
        if(!g.thread_rank()) {state->residual=rn;state->iterations=it+1;if(restart) ++state->restarts;}
        g.sync(); if(rn<=target) break;
        mg_detail::cycle(g,levels,levelCount,r,jacobi);
        double next=mg_detail::dot(g,n,r,f.x,scratch,state);
        if(!g.thread_rank()) {state->beta=restart?0:next/state->rz;state->rz=next;}
        g.sync();
        for(int i=int(g.thread_rank());i<n;i+=int(g.size())) s[i]=f.x[i]+Real(state->beta)*s[i];
        g.sync();
    }
    for(int i=int(g.thread_rank());i<n;i+=int(g.size())) r[i]=b[i]-f.apply(p,i);
    g.sync();
    double rn=sqrt(mg_detail::dot(g,n,r,r,scratch,state));
    if(!g.thread_rank()) state->residual=rn;
}
__global__ void mgCoarseTestVector(MGLevel c)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=c.n)return;
    auto q=c.coord(i);c.x[i]=i && c.diag[i]>0 ? Real(sin(.13*q[0])+cos(.17*q[1])+.2*sin(.11*q[2])):0;
    c.b[i]=0;
}
__global__ void mgTestProlong(MGLevel f,MGLevel c)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<f.n) f.x[i]=i&&f.diag[i]>0?c.x[f.parent[i]]:0;
}
__global__ void mgTestRestrict(MGLevel f,MGLevel c)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i && i<f.n && f.diag[i]>0) atomicAdd(c.b+f.parent[i],f.apply(f.x,i));
}
struct MGInvariant
{
    MGLevel level;
    __device__ double operator()(int i)const {return fabs(double(level.apply(level.x,i)-level.b[i]))/maxValue(1.,double(level.diag[i]));}
};
class Multigrid
{
    struct Storage {
        Handle grid;
        Buffer<uint32_t> address;
        Buffer<int> parent;
        Buffer<Real> diag, low, b, x, tmp;
    };
    std::vector<std::unique_ptr<Storage>> storage;
    std::vector<MGLevel> levels;
public:
    double setupMs=0, solveMs=0;
    size_t bytes=0;
    int levelCount() const {return int(levels.size());}
    Multigrid(Domain d, Real *diagonal, bool jacobi)
    {
        Timer timer;
        MGLevel fine; fine.domain=d;fine.fine=true;fine.n=d.count;fine.diag=diagonal;
        levels.push_back(fine);
        for(int k=0;k<10;++k) {
            auto own=std::make_unique<Storage>(); auto &l=levels[k];
            own->b=allocate<Real>(l.n);own->x=allocate<Real>(l.n);own->tmp=allocate<Real>(l.n);
            l.b=own->b.data();l.x=own->x.data();l.tmp=own->tmp.data(); bytes+=size_t(l.n)*3*sizeof(Real);
            bool stop=jacobi || l.n<128 || k==9;
            if(!stop) {own->parent=allocate<int>(l.n);l.parent=own->parent.data();bytes+=size_t(l.n)*sizeof(int);}
            storage.push_back(std::move(own));
            if(stop) break;
            auto coords=allocate<Coord>(l.n-1);
            mgCoordinates<<<blocks(l.n-1),256>>>(l,coords.data());
            auto coarse=std::make_unique<Storage>();
            coarse->grid=nanovdb::tools::cuda::voxelsToGrid<Build>(coords.data(),l.n-1,1.);
            nanovdb::GridData meta;
            cudaCheck(cudaMemcpy(&meta,coarse->grid.deviceGrid<Build>(),sizeof(meta),cudaMemcpyDeviceToHost));
            auto tree=nanovdb::util::cuda::DeviceGridTraits<Build>::getTreeData(coarse->grid.deviceGrid<Build>());
            MGLevel c;c.n=int(meta.mData1);c.grid=coarse->grid.deviceGrid<Build>();
            coarse->address=allocate<uint32_t>(c.n);c.address=coarse->address.data();
            coarse->diag=allocate<Real>(c.n);coarse->low=allocate<Real>(3*size_t(c.n));
            c.diag=coarse->diag.data();c.low=coarse->low.data();
            cudaCheck(cudaMemset(c.diag,0,sizeof(Real)*c.n));
            cudaCheck(cudaMemset(c.low,0,sizeof(Real)*3*c.n));
            cudaCheck(cudaMemset(c.address,0,sizeof(uint32_t)));
            mgEnumerate<<<tree.mNodeCount[0],512>>>(c);
            mgParents<<<blocks(l.n),256>>>(l,c);
            mgGalerkin<<<blocks(l.n),256>>>(l,c);
            bytes+=meta.mGridSize+size_t(c.n)*(4*sizeof(Real)+sizeof(uint32_t));
            // Ownership of topology/coefficient arrays may live independently
            // from ownership of the next level's work vectors.
            storage.push_back(std::move(coarse));levels.push_back(c);
        }
        setupMs=timer.ms();
    }
    double verifyGalerkin()
    {
        double error=0;
        for(size_t k=1;k<levels.size();++k) {
            auto f=levels[k-1],c=levels[k];
            mgCoarseTestVector<<<blocks(c.n),256>>>(c);
            mgTestProlong<<<blocks(f.n),256>>>(f,c);
            mgTestRestrict<<<blocks(f.n),256>>>(f,c);
            error=std::max(error,reduceDevice(c.n,MGInvariant{c},0.,cuda::maximum<double>()));
        }
        return error;
    }
    CGState solve(const Real *b, Real *p, Real *r, Real *s, Real *as, bool jacobi, int maxIterations=600)
    {
        Timer timer; int device=0, cooperative=0, sms=0, active=0;
        cudaCheck(cudaGetDevice(&device));
        cudaCheck(cudaDeviceGetAttribute(&cooperative,cudaDevAttrCooperativeLaunch,device));
        if(!cooperative) throw std::runtime_error("MGPCG requires CUDA cooperative launch support");
        cudaCheck(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device));
        cudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,mgpcgPersistent,256,0));
        int count=std::min(blocks(levels[0].n),active*sms), nlevels=levelCount();
        auto dev=allocate<MGLevel>(nlevels);auto state=allocate<CGState>(1);auto scratch=allocate<double>(count);
        cudaCheck(cudaMemcpy(dev.data(),levels.data(),nlevels*sizeof(MGLevel),cudaMemcpyHostToDevice));
        cudaCheck(cudaMemset(state.data(),0,sizeof(CGState)));
        auto lv=dev.data();auto st=state.data();auto work=scratch.data();double tol=pressureTolerance;
        void *args[]={&lv,&nlevels,&b,&p,&r,&s,&as,&work,&st,&maxIterations,&tol,&jacobi};
        cudaCheck(cudaLaunchCooperativeKernel((void*)mgpcgPersistent,count,256,args));
        CGState result;
        cudaCheck(cudaMemcpy(&result,state.data(),sizeof(result),cudaMemcpyDeviceToHost));
        solveMs=timer.ms();return result;
    }
};
