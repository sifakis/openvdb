// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifndef EX_MESH_TO_SDF_FLUID_CUH
#define EX_MESH_TO_SDF_FLUID_CUH
#include <nanovdb/tools/cuda/PointsToGrid.cuh>
#include <nanovdb/tools/cuda/DilateGrid.cuh>
#include <nanovdb/cuda/Buffer.h>
#include <thrust/device_ptr.h>
#include <thrust/inner_product.h>
#include <thrust/transform_reduce.h>
#include <cooperative_groups.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <memory>
#include <climits>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <map>
#include <set>

namespace fluid {
#ifdef FLUID_USE_FLOAT
using Real = float;
#else
using Real = double;
#endif
// Mixed float/double calls must not select libstdc++ host-only constexpr overloads.
template<class A,class B> __hostdev__ auto maxValue(A a,B b) -> decltype(a+b) {return a!=a?b:b!=b?a:a>b?a:b;}
template<class A,class B> __hostdev__ auto minValue(A a,B b) -> decltype(a+b) {return a!=a?b:b!=b?a:a<b?a:b;}
using Vec = nanovdb::math::Vec3<Real>;
using GeometryVec = nanovdb::Vec3d;
constexpr bool singlePrecision = sizeof(Real) == 4;
constexpr double pressureTolerance = singlePrecision ? 2e-5 : 1e-9;
constexpr double pressureCheckTolerance = singlePrecision ? 1e-4 : 2e-8;
constexpr double divergenceTolerance = singlePrecision ? 2e-3 : 1e-6;
constexpr double contactTolerance = singlePrecision ? 2e-5 : 1e-7;
using Coord = nanovdb::Coord;
using Build = nanovdb::ValueOnIndex;
using Grid = nanovdb::NanoGrid<Build>;
using Handle = nanovdb::GridHandle<nanovdb::cuda::DualDeviceBuffer>;
template <class T> using Buffer = nanovdb::cuda::Buffer<T>;
template <class T> Buffer<T> allocate(size_t n)
{
    return Buffer<T>(0, n, nanovdb::cuda::noInit);
}
template<class T> void growBuffer(Buffer<T>& buffer,size_t required,size_t live=0)
{
    if(buffer.size()>=required) return;
    auto next=allocate<T>(std::max(required,buffer.size()+buffer.size()/2+256));
    if(live) cudaCheck(cudaMemcpyAsync(next.data(),buffer.data(),live*sizeof(T),cudaMemcpyDeviceToDevice));
    buffer=std::move(next);
}
inline void sync()
{
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
}
struct Timer
{
    std::chrono::steady_clock::time_point start;
    Timer()
    {
        sync();
        start = std::chrono::steady_clock::now();
    }
    double ms()
    {
        sync();
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }
};
struct Particle
{
    Vec x{0}, v{0}, c[3]{Vec(0), Vec(0), Vec(0)};
};
struct Cell
{
    int fluid;
    Real mass[3], u[3], transport[3];
    unsigned char valid[3];
    Real open[3], wall[3], phi;
};
// Coordinates/connectivity exist only in explicit host exports, never in the
// resident Cell. A compact inverse address is rebuilt with each topology.
struct HostCell : Cell { Coord ijk; int neighbor[6]; };
struct Domain
{
    const Grid *grid;
    Cell *cells;
    uint32_t *address = nullptr;
    Real *extrap = nullptr;
    unsigned char *extrapValid = nullptr;
    int count;
    int nx = 32, ny = 32, nz = 24;
    Real h = Real(0.05);
    __device__ bool inside(Coord c) const
    {
        return c[0] >= 0 && c[1] >= 0 && c[2] >= 0 && c[0] < nx && c[1] < ny && c[2] < nz;
    }
    __device__ int index(Coord c) const { auto acc = grid->getAccessor(); return int(acc.getValue(c)); }
    __device__ Coord coord(int i) const
    {
        uint32_t a = address[i];
        return grid->tree().getFirstNode<0>()[a >> 9].offsetToGlobalCoord(a & 511);
    }
    __device__ int neighbor(int i, int j) const
    {
        Coord c = coord(i); c[j / 2] += (j & 1) ? 1 : -1;
        return index(c);
    }
};
inline int blocks(int n)
{
    return (n + 255) / 256;
}
__device__ inline Coord axis(int a)
{
    Coord c(0);
    c[a] = 1;
    return c;
}
__device__ inline Vec center(Coord c)
{
    return Vec(c[0] + 0.5, c[1] + 0.5, c[2] + 0.5);
}
__device__ inline Vec face(Coord c, int a)
{
    Vec x = center(c);
    x[a] -= 0.5;
    return x;
}
__device__ inline Coord floorCoord(Vec x)
{
    return Coord(int(floor(x[0])), int(floor(x[1])), int(floor(x[2])));
}
__global__ void enumerate(Domain d)
{
    int leaf = blockIdx.x, offset = threadIdx.x;
    const auto &node = d.grid->tree().getFirstNode<0>()[leaf];
    if (!node.isActive(offset))
        return;
    int i = int(node.getValue(offset));
    d.address[i] = (uint32_t(leaf) << 9) | uint32_t(offset);
}
__global__ void clear(Domain d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d.count)
        return;
    Cell &c = d.cells[i];
    c.fluid = 0;
    c.phi = 1;
    for (int a = 0; a < 3; ++a) {
        c.mass[a] = c.u[a] = c.transport[a] = c.wall[a] = 0;
        c.valid[a] = 0;
        c.open[a] = i && d.inside(d.coord(i)) && d.inside(d.coord(i) - axis(a)) ? 1 : 0;
    }
}
__global__ void particleCoords(const Particle *p, int n, Coord *out, double h)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = floorCoord(p[i].x / h);
}
__device__ inline Real weight(Real x)
{
    x = fabs(x);
    return x < Real(.5) ? Real(.75) - x*x : (x < Real(1.5) ? Real(.5)*(Real(1.5)-x)*(Real(1.5)-x) : Real(0));
}
__global__ void classify(Domain d, const Particle *p, int n, int *missing, bool includeCutCells = false)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n)
        return;
    int i = d.index(floorCoord(p[k].x / d.h));
    if (!i) {
        atomicAdd(missing, 1);
        return;
    }
    // Contact-validated particles may occupy the fluid portion of a cell
    // whose center lies in the solid. Such cells still need pressure DOFs.
    if (includeCutCells || d.cells[i].phi > 0)
        atomicExch(&d.cells[i].fluid, 1);
}
__global__ void p2g(Domain d, const Particle *ps, int n, int *missing)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n)
        return;
    const Particle p = ps[k];
    auto accessor = d.grid->getAccessor();
    for (int a = 0; a < 3; ++a) {
        Vec offset(0.5);
        offset[a] = 0;
        Coord base = floorCoord(p.x / d.h - offset - Vec(0.5));
        for (int x = 0; x < 3; ++x)
            for (int y = 0; y < 3; ++y)
                for (int z = 0; z < 3; ++z) {
                    Coord q = base + Coord(x, y, z);
                    Vec delta = face(q, a) * d.h - p.x;
                    Real w = weight(delta[0] / d.h) * weight(delta[1] / d.h) * weight(delta[2] / d.h);
                    int i = int(accessor.getValue(q));
                    if (!i) {
                        if (w > 1e-14)
                            atomicAdd(missing, 1);
                        continue;
                    }
                    atomicAdd(&d.cells[i].mass[a], w);
                    atomicAdd(&d.cells[i].u[a], w * (p.v[a] + p.c[a].dot(delta)));
                }
    }
}
__global__ void normalize(Domain d, Real dt, Real gravity)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0 || i >= d.count)
        return;
    Cell &c = d.cells[i];
    for (int a = 0; a < 3; ++a) {
        c.valid[a] = c.mass[a] > 1e-14;
        c.u[a] = c.valid[a] ? c.u[a] / c.mass[a] + (a == 1 ? gravity * dt : 0) : 0;
        if (c.open[a] == 0) {
            c.u[a] = c.wall[a];
            c.valid[a] = 1;
        }
    }
}
__device__ inline Real flux(const Cell &c, int a)
{
    return c.open[a] * c.u[a] + (1 - c.open[a]) * c.wall[a];
}
__device__ inline Real netFlux(Domain d, int i)
{
    Real f = 0;
    const Cell &c = d.cells[i];
    for (int a = 0; a < 3; ++a)
        f += flux(d.cells[d.neighbor(i, 2 * a + 1)], a) - flux(c, a);
    return f;
}
__global__ void assemble(Domain d, Real *b, Real *diag, int *missing)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d.count)
        return;
    b[i] = diag[i] = 0;
    if (!d.cells[i].fluid)
        return;
    Cell &c = d.cells[i];
    b[i] = -netFlux(d, i);
    for (int a = 0; a < 3; ++a) {
        int lo = d.neighbor(i, 2 * a), hi = d.neighbor(i, 2 * a + 1);
        if (!lo || !hi)
            atomicAdd(missing, 1);
        diag[i] += c.open[a] + d.cells[hi].open[a];
    }
}
__global__ void applyMatrix(Domain d, const Real *x, Real *out, const Real *diag)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d.count)
        return;
    Real y = diag[i] * x[i];
    const Cell &c = d.cells[i];
    if (c.fluid)
        for (int a = 0; a < 3; ++a) {
            int lo = d.neighbor(i, 2 * a), hi = d.neighbor(i, 2 * a + 1);
            y -= c.open[a] * x[lo] + d.cells[hi].open[a] * x[hi];
        }
    out[i] = y;
}
__global__ void precondition(int n, const Real *r, const Real *diag, Real *z)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        z[i] = diag[i] > 0 ? r[i] / diag[i] : 0;
}
__global__ void axpy(int n, Real *x, const Real *y, Real a)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] += a * y[i];
}
__global__ void direction(int n, Real *p, const Real *z, Real beta)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        p[i] = z[i] + beta * p[i];
}
inline double dot(int n, const Real *a, const Real *b)
{
    return thrust::inner_product(thrust::device_pointer_cast(a), thrust::device_pointer_cast(a + n),
                                 thrust::device_pointer_cast(b), 0.0);
}
__global__ void project(Domain d, const Real *p)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (!i || i >= d.count)
        return;
    Cell &c = d.cells[i];
    for (int a = 0; a < 3; ++a) {
        int lo = d.neighbor(i, 2 * a);
        c.valid[a] = c.fluid || d.cells[lo].fluid || c.open[a] == 0;
        if (c.open[a] > 0 && (c.fluid || d.cells[lo].fluid)) {
            c.u[a] -= p[i] - p[lo];
            c.valid[a] = 1;
        }
        if (c.open[a] == 0)
            c.u[a] = c.wall[a];
    }
}
// Pressure velocities retain the physical wall flux. Particle interpolation uses
// a separate constant extension from open fluid faces into air AND solid ghosts.
// Treating every solid ghost component as wall velocity imposes spurious no-slip.
__global__ void prepareTransport(Domain d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (!i || i >= d.count) return;
    for (int a = 0; a < 3; ++a) {
        d.cells[i].transport[a] = d.cells[i].u[a];
        d.cells[i].valid[a] = d.cells[i].valid[a] && d.cells[i].open[a] > 0;
    }
}
__global__ void extrapolate(Domain d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (!i || i >= d.count)
        return;
    Cell &c = d.cells[i];
    for (int a = 0; a < 3; ++a) {
        d.extrap[3*i+a] = c.transport[a];
        d.extrapValid[3*i+a] = c.valid[a];
        if (c.valid[a])
            continue;
        Real sum = 0;
        int n = 0;
        for (int j = 0; j < 6; ++j) {
            int nb = d.neighbor(i, j);
            if (nb && d.cells[nb].valid[a]) {
                sum += d.cells[nb].transport[a];
                ++n;
            }
        }
        if (n) {
            d.extrap[3*i+a] = sum / n;
            d.extrapValid[3*i+a] = 1;
        }
    }
}
__global__ void finishExtrapolate(Domain d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (!i || i >= d.count)
        return;
    for (int a = 0; a < 3; ++a) {
        d.cells[i].transport[a] = d.extrap[3*i+a];
        d.cells[i].valid[a] = d.extrapValid[3*i+a];
    }
}
__global__ void g2p(Domain d, Particle *ps, int n, Real dt, int *missing, bool walls, Real gravity, bool limitAffine = true)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n)
        return;
    Particle p = ps[k];
    const Vec priorVelocity = p.v;
    int invalidTransport = 0, invalidComponents = 0;
    auto accessor = d.grid->getAccessor();
    for (int a = 0; a < 3; ++a) {
        Vec offset(0.5);
        offset[a] = 0;
        Coord base = floorCoord(p.x / d.h - offset - Vec(0.5));
        Real v = 0, minimum = 1e30, maximum = -1e30;
        bool extrapolated = false;
        Vec c(0);
        for (int x = 0; x < 3; ++x)
            for (int y = 0; y < 3; ++y)
                for (int z = 0; z < 3; ++z) {
                    Coord q = base + Coord(x, y, z);
                    Vec delta = face(q, a) * d.h - p.x;
                    Real w = weight(delta[0] / d.h) * weight(delta[1] / d.h) * weight(delta[2] / d.h);
                    int i = int(accessor.getValue(q));
                    if (!i) {
                        if (w > 1e-14)
                            atomicAdd(missing, 1);
                        continue;
                    }
                    if (w > 1e-14 && !d.cells[i].valid[a]) {
                        ++invalidTransport;
                        invalidComponents |= 1 << a;
                    }
                    if (w > 1e-14) {
                        const auto &cell = d.cells[i];
                        minimum = minValue(minimum, cell.transport[a]);
                        maximum = maxValue(maximum, cell.transport[a]);
                        extrapolated |= !(cell.open[a] > 0 &&
                            (cell.fluid || d.cells[d.neighbor(i, 2*a)].fluid));
                    }
                    v += w * d.cells[i].transport[a];
                    c += delta * (w * d.cells[i].transport[a] * 4 / (d.h * d.h));
                }
        // Extrapolated ghost velocities need not form a consistent affine
        // field. Limit only an overshooting affine reconstruction there; the
        // mean velocity and all fully projected bulk stencils are untouched.
        // A true affine field already lies in its stencil range (alpha == 1).
        if (limitAffine && extrapolated && minimum <= maximum) {
            Real alpha = 1;
            const Real tolerance = (singlePrecision ? 2e-6 : 1e-12) * maxValue(1., maxValue(fabs(minimum), fabs(maximum)));
            for (int x=0; x<3; ++x) for (int y=0; y<3; ++y) for (int z=0; z<3; ++z) {
                Vec delta = face(base + Coord(x,y,z), a) * d.h - p.x;
                Real w = weight(delta[0]/d.h)*weight(delta[1]/d.h)*weight(delta[2]/d.h);
                if (w <= 1e-14) continue;
                Real change = c.dot(delta);
                if (change > tolerance) alpha = minValue(alpha, maxValue(0., (maximum + tolerance - v)/change));
                if (change < -tolerance) alpha = minValue(alpha, maxValue(0., (minimum - tolerance - v)/change));
            }
            if (alpha < 1 - 1e-10) atomicAdd(missing + 2, 1);
            c *= alpha;
        }
        p.v[a] = v;
        p.c[a] = c;
    }
    if (invalidTransport) {
        int cell = d.index(floorCoord(p.x / d.h));
        if (cell && d.cells[cell].phi <= 0 && !d.cells[cell].fluid) {
            // Outside the actual surface, but inside a solid-classified voxel:
            // no represented pressure cell. An isolated subcell droplet moves
            // ballistically, then the same SDF/mesh collision enforces contact.
            // Do not bootstrap unprojected low-mass faces into an APIC feedback.
            p.v = priorVelocity + Vec(0, gravity * dt, 0);
            for (int a=0; a<3; ++a) p.c[a] = Vec(0);
            atomicAdd(missing + 1, 1);
        } else {
            bool resolved = cell && d.cells[cell].fluid;
            for (int a=0; a<3 && resolved; ++a) {
                if (!(invalidComponents & (1 << a))) continue;
                const int hi = d.neighbor(cell, 2*a+1);
                resolved = hi && d.cells[cell].open[a] == 0 && d.cells[hi].open[a] == 0;
            }
            if (resolved) {
                // A positive cell can still have BOTH MAC faces blocked for
                // one component (a sub-voxel slit). There is no projected DOF
                // for that component. Retain the other projected components;
                // advance only the unrepresented one ballistically and let
                // the subsequent SDF/triangle contact constrain it.
                for (int a=0; a<3; ++a) if (invalidComponents & (1 << a)) {
                    p.v[a] = priorVelocity[a] + (a == 1 ? gravity * dt : 0);
                    p.c[a] = Vec(0);
                }
                atomicAdd(missing + 1, 1);
            } else if (atomicAdd(missing, invalidTransport) == 0) {
                printf("INVALID fluid transport particle=(%g,%g,%g) cell_phi=%g axes=%d fluid=%d\n",
                       p.x[0],p.x[1],p.x[2],cell ? d.cells[cell].phi : 0., invalidComponents,
                       cell ? d.cells[cell].fluid : 0);
                if (cell) for (int a=0; a<3; ++a)
                    printf("  axis=%d open_lo=%g open_hi=%g\n", a, d.cells[cell].open[a],
                           d.cells[d.neighbor(cell, 2*a+1)].open[a]);
            }
        }
    }
    p.x += p.v * dt;
    if (walls)
        for (int a = 0; a < 3; ++a) {
            Real hi = (a == 0 ? d.nx : a == 1 ? d.ny : d.nz) * d.h - 0.1 * d.h;
            const Real lower = Real(.1) * d.h;
            if (p.x[a] < lower) {
                p.x[a] = lower;
                p.v[a] = maxValue(0., p.v[a]);
                p.c[a] = Vec(0);
            }
            if (p.x[a] > hi) {
                p.x[a] = hi;
                p.v[a] = minValue(0., p.v[a]);
                p.c[a] = Vec(0);
            }
        }
    ps[k] = p;
}
struct LeafSummary {Coord origin;int core;};
__global__ void exportLeaves(Domain d,LeafSummary *out)
{
    const auto &node=d.grid->tree().getFirstNode<0>()[blockIdx.x];
    int offset=threadIdx.x,core=node.isActive(offset)&&d.cells[node.getValue(offset)].fluid;
    __shared__ int occupied[512];occupied[offset]=core;__syncthreads();
    for(int stride=256;stride;stride>>=1){if(offset<stride)occupied[offset]|=occupied[offset+stride];__syncthreads();}
    if(!offset) out[blockIdx.x]={node.offsetToGlobalCoord(0),occupied[0]};
}
__global__ void copyVelocities(const Particle *p,Vec *v,int n)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)v[i]=p[i].v;
}
__global__ void exportCells(Domain d, HostCell *out)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=d.count) return;
    static_cast<Cell&>(out[i])=d.cells[i];
    out[i].ijk=d.coord(i);
    for(int j=0;j<6;++j) out[i].neighbor[j]=i ? d.neighbor(i,j) : 0;
}
struct Stats
{
    double topology = 0, p2g = 0, pressure = 0, g2p = 0, mesh = 0, boundary = 0;
    double residual = 0, absoluteResidual = 0, rhsNorm = 0, divBefore = 0, divAfter = 0;
    double mgSetup = 0, mgSolve = 0; size_t mgBytes = 0; int mgLevels = 0;
    int iterations = 0, missing = 0, unresolved = 0, affineLimited = 0, fluidCells = 0, leaves = 0;
};
#include "scalable/Reductions.cuh"
#include "scalable/Multigrid.cuh"
class Solver
{
  public:
    Handle topology;
    Buffer<Particle> particles;
    Buffer<Cell> cells;
    Buffer<uint32_t> address;
    Buffer<Real> extrapScratch;
    Buffer<unsigned char> validScratch;
    Buffer<int> missing;
    Domain d{};
    int particleCount;
    bool dynamic;
    Stats stats;
    double lastGravity = -9.81;
    std::string pressureDiagnostic;
    bool includeCutCells = false;
    bool jacobiPressure = false;
    int pressureMaxIterations = 600;
    Solver(const std::vector<Particle> &initial, bool sparse = false, double voxelSize = .05,
           Vec extent = Vec(1.6, 1.6, 1.2))
        : particles(allocate<Particle>(initial.size())), missing(allocate<int>(3)),
          particleCount(int(initial.size())), dynamic(sparse)
    {
        d.h = voxelSize;
        d.nx = int(std::lround(extent[0] / d.h));
        d.ny = int(std::lround(extent[1] / d.h));
        d.nz = int(std::lround(extent[2] / d.h));
        if (fabs(d.nx * d.h - extent[0]) > (singlePrecision?1e-6:1e-9) || fabs(d.ny * d.h - extent[1]) > (singlePrecision?1e-6:1e-9) ||
            fabs(d.nz * d.h - extent[2]) > (singlePrecision?1e-6:1e-9))
            throw std::runtime_error("voxel size must divide the physical tank dimensions");
        cudaCheck(cudaMemcpy(particles.data(), initial.data(), sizeof(Particle) * initial.size(),
                             cudaMemcpyHostToDevice));
        rebuild();
    }
    void rebuild()
    {
        Timer timer;
        if (dynamic) {
            auto coords = allocate<Coord>(particleCount);
            particleCoords<<<blocks(particleCount), 256>>>(particles.data(), particleCount, coords.data(),
                                                           d.h);
            topology = nanovdb::tools::cuda::voxelsToGrid<Build>(coords.data(), particleCount, d.h);
            // Six Chebyshev layers contain the quadratic MAC and pressure stencils
            // plus the local paths used by six face-neighbor extrapolation sweeps.
            for (int k = 0; k < 6; ++k) {
                nanovdb::tools::cuda::DilateGrid<Build> dilate(topology.deviceGrid<Build>());
                topology = dilate.getHandle();
            }
        } else {
            std::vector<Coord> host;
            for (int x = -8; x < d.nx + 8; ++x)
                for (int y = -8; y < d.ny + 8; ++y)
                    for (int z = -8; z < d.nz + 8; ++z)
                        host.emplace_back(x, y, z);
            auto coords = allocate<Coord>(host.size());
            cudaCheck(
                cudaMemcpy(coords.data(), host.data(), host.size() * sizeof(Coord), cudaMemcpyHostToDevice));
            topology = nanovdb::tools::cuda::voxelsToGrid<Build>(coords.data(), host.size(), d.h);
        }
        nanovdb::GridData metadata;
        cudaCheck(
            cudaMemcpy(&metadata, topology.deviceGrid<Build>(), sizeof(metadata), cudaMemcpyDeviceToHost));
        d.count = int(metadata.mData1);
        auto td = nanovdb::util::cuda::DeviceGridTraits<Build>::getTreeData(topology.deviceGrid<Build>());
        stats.leaves = td.mNodeCount[0];
        if (metadata.mData1 > uint64_t(INT_MAX) || stats.leaves >= (1 << 23))
            throw std::runtime_error("32-bit sidecar address capacity exceeded");
        cells = allocate<Cell>(d.count);
        address = allocate<uint32_t>(d.count);
        d.address = address.data();
        cudaCheck(cudaMemset(address.data(), 0, sizeof(uint32_t)));
        cudaCheck(cudaMemset(cells.data(), 0, d.count * sizeof(Cell)));
        d.grid = topology.deviceGrid<Build>();
        d.cells = cells.data();
        enumerate<<<stats.leaves, 512>>>(d);
        stats.topology = timer.ms();
    }
    void reset()
    {
        cudaCheck(cudaMemset(missing.data(), 0, 3 * sizeof(int)));
        clear<<<blocks(d.count), 256>>>(d);
    }
    std::vector<Particle> download()
    {
        std::vector<Particle> out(particleCount);
        cudaCheck(
            cudaMemcpy(out.data(), particles.data(), out.size() * sizeof(Particle), cudaMemcpyDeviceToHost));
        return out;
    }
    std::vector<HostCell> downloadCells()
    {
        auto staging = allocate<HostCell>(d.count);
        exportCells<<<blocks(d.count),256>>>(d, staging.data());
        std::vector<HostCell> out(d.count);
        cudaCheck(cudaMemcpy(out.data(), staging.data(), out.size()*sizeof(HostCell), cudaMemcpyDeviceToHost));
        return out;
    }
    std::vector<LeafSummary> downloadBlocks()
    {
        auto staging=allocate<LeafSummary>(stats.leaves);
        exportLeaves<<<stats.leaves,512>>>(d,staging.data());
        std::vector<LeafSummary> out(stats.leaves);
        cudaCheck(cudaMemcpy(out.data(),staging.data(),out.size()*sizeof(LeafSummary),cudaMemcpyDeviceToHost));
        return out;
    }
    void transportScratch()
    {
        extrapScratch = allocate<Real>(3*size_t(d.count));
        validScratch = allocate<unsigned char>(3*size_t(d.count));
        d.extrap = extrapScratch.data(); d.extrapValid = validScratch.data();
    }
    void validateTopology()
    {
        if(reduceDevice(d.count,TopologyMeasure{d},0,cuda::maximum<int>()))
            throw std::runtime_error("sidecar coordinate/neighbor correspondence failed");
    }
    double timestep()
    {
        double speed=reduceDevice(particleCount,ParticleMeasure{particles.data(),d,true},1.,cuda::maximum<double>());
        if(!std::isfinite(speed)) throw std::runtime_error("nonfinite particle state");
        return std::min(.005,.25*d.h/(speed+1));
    }
    double particleCfl(double dt)
    {
        double speed=reduceDevice(particleCount,ParticleMeasure{particles.data(),d,false},0.,cuda::maximum<double>());
        if(!std::isfinite(speed) || speed*dt/d.h>.5) throw std::runtime_error("invalid particle or CFL exceeds .5");
        return speed*dt/d.h;
    }
    double divergence()
    {
        auto value=reduceDevice(d.count,FluxMeasure{d},FluxSummary{},FluxAdd{});
        stats.fluidCells=value.count;
        return sqrt(value.square/std::max(value.count,1));
    }
    void transfer(double dt, double gravity = -9.81)
    {
        Timer t;
        lastGravity = gravity;
        classify<<<blocks(particleCount), 256>>>(d, particles.data(), particleCount, missing.data(), includeCutCells);
        p2g<<<blocks(particleCount), 256>>>(d, particles.data(), particleCount, missing.data());
        normalize<<<blocks(d.count), 256>>>(d, dt, gravity);
        stats.p2g = t.ms();
    }
    void pressure()
    {
        stats.divBefore = divergence();
        Timer t;
        const int n = d.count;
        auto b=allocate<Real>(n), diag=allocate<Real>(n), p=allocate<Real>(n),
             r=allocate<Real>(n), s=allocate<Real>(n), as=allocate<Real>(n);
        assemble<<<blocks(n),256>>>(d,b.data(),diag.data(),missing.data());
        Multigrid mg(d,diag.data(),jacobiPressure);
        auto result=mg.solve(b.data(),p.data(),r.data(),s.data(),as.data(),jacobiPressure,pressureMaxIterations);
        stats.mgSetup=mg.setupMs;stats.mgSolve=mg.solveMs;stats.mgBytes=mg.bytes;stats.mgLevels=mg.levelCount();
        stats.iterations=result.iterations;
        const double norm=result.norm, absoluteResidual=result.residual;
        stats.residual=absoluteResidual/std::max(norm,1e-12);
        stats.absoluteResidual=absoluteResidual;stats.rhsNorm=norm;
        cudaCheck(cudaMemcpy(as.data(),r.data(),n*sizeof(Real),cudaMemcpyDeviceToDevice));
        if(result.breakdown) throw std::runtime_error("GPU pressure matrix breakdown");
        project<<<blocks(n), 256>>>(d, p.data());
        stats.pressure = t.ms();
        stats.divAfter = divergence();
        if (!std::isfinite(stats.residual) || !std::isfinite(stats.divAfter) || absoluteResidual > std::max(1e-12, norm * pressureCheckTolerance) ||
            stats.divAfter > divergenceTolerance)
        {
            if (!pressureDiagnostic.empty()) {
                auto host = downloadCells();
                std::vector<Real> rhs(n), diagonal(n), pressure(n), residual(n);
                cudaCheck(cudaMemcpy(rhs.data(), b.data(), n*sizeof(Real), cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(diagonal.data(), diag.data(), n*sizeof(Real), cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(pressure.data(), p.data(), n*sizeof(Real), cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(residual.data(), as.data(), n*sizeof(Real), cudaMemcpyDeviceToHost));
                std::ofstream file(pressureDiagnostic);
                file << "id,x,y,z,rhs,diag,pressure,residual,lo0,hi0,lo1,hi1,lo2,hi2,wlo0,whi0,wlo1,whi1,wlo2,whi2\n";
                file << std::setprecision(17);
                for (int i=1; i<n; ++i) if(host[i].fluid) {
                    auto &c=host[i];
                    file << i << ',' << c.ijk[0] << ',' << c.ijk[1] << ',' << c.ijk[2]
                         << ',' << rhs[i] << ',' << diagonal[i] << ',' << pressure[i] << ',' << residual[i];
                    for(int j=0;j<6;++j) file << ',' << (host[c.neighbor[j]].fluid ? c.neighbor[j] : 0);
                    for(int a=0;a<3;++a) file << ',' << c.open[a] << ',' << host[c.neighbor[2*a+1]].open[a];
                    file << '\n';
                }
            }
            throw std::runtime_error("projection validation failed: residual=" + std::to_string(stats.residual) + " divergence=" + std::to_string(stats.divAfter) + " iterations=" + std::to_string(stats.iterations));
        }
    }
    void advect(double dt, bool walls = true)
    {
        Timer t;
        transportScratch();
        prepareTransport<<<blocks(d.count), 256>>>(d);
        for (int k = 0; k < 6; ++k) {
            extrapolate<<<blocks(d.count), 256>>>(d);
            finishExtrapolate<<<blocks(d.count), 256>>>(d);
        }
        g2p<<<blocks(particleCount), 256>>>(d, particles.data(), particleCount, dt, missing.data(), walls, lastGravity);
        stats.g2p = t.ms();
        extrapScratch = {}; validScratch = {}; d.extrap=nullptr;d.extrapValid=nullptr;
        cudaCheck(cudaMemcpy(&stats.missing, missing.data(), sizeof(int), cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(&stats.unresolved, missing.data() + 1, sizeof(int), cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(&stats.affineLimited, missing.data() + 2, sizeof(int), cudaMemcpyDeviceToHost));
        if (stats.missing)
            throw std::runtime_error("missing particle/pressure stencil");
    }
    void step(double dt)
    {
        if (dynamic)
            rebuild();
        else
            stats.topology = 0;
        reset();
        transfer(dt);
        pressure();
        advect(dt);
    }
};
inline std::vector<Particle> dam(double h = .05)
{
    // Preserve the physical dam bounds. Two samples per cell width in each
    // dimension give approximately eight particles per cell at every resolution.
    const Vec lo(.15, .05, .15), hi(.70, 1.10, 1.05);
    const Vec width = hi - lo;
    const int nx = int(std::lround(2 * width[0] / h));
    const int ny = int(std::lround(2 * width[1] / h));
    const int nz = int(std::lround(2 * width[2] / h));
    std::vector<Particle> particles;
    particles.reserve(size_t(nx) * ny * nz);
    // Match the original particle ordering at h=.05 so saved low-resolution
    // baselines remain comparable to runs using this parameterized initializer.
    for (int x = 0; x < (nx + 1) / 2; ++x)
        for (int y = 0; y < (ny + 1) / 2; ++y)
            for (int z = 0; z < (nz + 1) / 2; ++z)
                for (int j = 0; j < 8; ++j) {
                    int i[3] = {2 * x + (j & 1), 2 * y + ((j >> 1) & 1), 2 * z + ((j >> 2) & 1)};
                    if (i[0] >= nx || i[1] >= ny || i[2] >= nz)
                        continue;
                    Particle p;
                    p.x = lo + Vec((i[0] + .5) * width[0] / nx, (i[1] + .5) * width[1] / ny,
                                   (i[2] + .5) * width[2] / nz);
                    particles.push_back(p);
                }
    return particles;
}
inline void writeFrame(Solver &s, const std::string &path)
{
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::ofstream f(path + ".csv");
    f << "x,y,z,vx,vy,vz\n" << std::setprecision(10);
    for (const auto &p : s.download())
        f << p.x[0] << ',' << p.x[1] << ',' << p.x[2] << ',' << p.v[0] << ',' << p.v[1] << ',' << p.v[2]
          << '\n';
    std::ofstream b(path + "_blocks.csv");
    b << "x,y,z,fluid_cells,active_voxels\n";
    std::map<std::tuple<int, int, int>, std::pair<int, int>> leaves;
    auto hostCells = s.downloadCells();
    for (size_t i = 1; i < hostCells.size(); ++i) {
        const auto &c = hostCells[i];
        auto key = std::make_tuple((c.ijk[0] >> 3) * 8, (c.ijk[1] >> 3) * 8, (c.ijk[2] >> 3) * 8);
        auto &counts = leaves[key];
        counts.first += c.fluid;
        counts.second++;
    }
    for (const auto &[key, count] : leaves)
        b << std::get<0>(key) << ',' << std::get<1>(key) << ',' << std::get<2>(key) << ',' << count.first
          << ',' << count.second << '\n';
}
} // namespace fluid
#endif
