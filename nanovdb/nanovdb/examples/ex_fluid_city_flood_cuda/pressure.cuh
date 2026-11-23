// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

// pressure.cuh: pressure projection with the free surface placed where the level set says it is.
//
// The fluid cells are the voxels with phi < 0 inside the tank. On a face between a fluid cell and an air cell the
// pressure is zero at the surface, which cuts the segment between the two cell centers at the fraction
//   theta = phi_fluid / (phi_fluid - phi_air)                      (0 < theta <= 1, measured from the fluid cell)
// The ghost fluid method (Gibou et al. 2002) keeps the 7-point matrix and its symmetry and only divides that
// face's diagonal term, and its pressure gradient, by theta. theta = 1 is the first-order rule (zero pressure at
// the air cell's center) of the APIC example and of OpenVDB's DirichletBoundaryOp.
//
// Reused from the APIC example (ex_fluid_apic_cuda/Fluid.cuh, unmodified): Domain, Cell, enumerate, clear,
// netFlux and the matrix-free multigrid-preconditioned CG (scalable/Multigrid.cuh). Its matrix is given by the
// diagonal and the cells' open fractions, so the change above enters through the diagonal alone.
// As in that code the unknown is the scaled potential q = dt p / (rho h): A q = -(net face flux), u -= difference of q.
#pragma once
#include <nanovdb/cuda/Buffer.h>
// Fluid.cuh names cuda::DualDeviceBuffer, which NanoVDB 14 removed; it only uses it as a device buffer of bytes.
namespace nanovdb { namespace cuda { using DualDeviceBuffer = Buffer<std::byte>; } }
#include "Fluid.cuh" // ex_fluid_apic_cuda/Fluid.cuh: put that directory on the include path
#include <nanovdb/tools/cuda/LevelSetTracker.cuh> // isInverted

namespace pressure {

using fluid::Cell;
using fluid::Domain;
using fluid::Real;

// Where the solids are: psi >= 0 in free space and < 0 inside a solid (tank walls and obstacles alike), with the
// distance to the solid's surface in voxels as its magnitude: the smaller of the tank's own field, computed from
// the tank's size, and the obstacle's field. The obstacle is stored sparsely, as MeshToSDF made it: an index grid
// with values in a band of a few voxels around the mesh's surface, and PropagateSign's invert masks, which say on
// which side of the mesh a voxel outside the band is (there the field is +-band). One voxel of the grid is one cell
// of the tank. Beyond `pad` cells around the tank everything is solid (-1000).
struct Solid
{
    const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* grid; // the obstacle's grid, or null without an obstacle
    const float*            values;                       // its values, in the grid's length unit
    const nanovdb::Mask<3>* leafMasks;                    // the invert masks
    const nanovdb::Mask<4>* lowerMasks;
    const nanovdb::Mask<5>* upperMasks;
    const nanovdb::Coord*   rootOrigins;
    uint64_t                rootOriginCount;
    double                  voxel;      // the grid's voxel size: values / voxel is in voxels
    float                   band;       // the band's half width in voxels
    float                   offset;     // subtracted from the field: everything within this distance of the mesh is solid
    int                     shift[3], axes[3]; // tank cell c is the grid's voxel m with m[axes[a]] = c[a] + shift[a]
    int                     tank[3], pad;
    // The obstacle's field alone at a tank cell.
    __hostdev__ float obstacle(const nanovdb::Coord& c) const
    {
        nanovdb::Coord m;
        for (int a = 0; a < 3; ++a) m[axes[a]] = c[a] + shift[a];
        const uint64_t index = grid->tree().getValue(m);
        if (index) return float(values[index] / voxel) - offset;
        using nanovdb::tools::cuda::levelset::detail::isInverted;
        return (isInverted(*grid, m, leafMasks, lowerMasks, upperMasks, rootOrigins, rootOriginCount) ? -band : band) - offset;
    }
    __hostdev__ float at(const nanovdb::Coord& c) const
    {
        float beyond = 0.0f, inside = -1.0e30f; // the signed distance to the tank's box, as Tank::distance
        for (int a = 0; a < 3; ++a) {
            if (c[a] < -pad || c[a] >= tank[a] + pad) return -1000.0f;
            const float q = fabsf(float(c[a]) - 0.5f * float(tank[a] - 1)) - 0.5f * float(tank[a]);
            if (q > 0.0f) beyond += q * q;
            inside = q > inside ? q : inside;
        }
        const float value = -(beyond > 0.0f ? sqrtf(beyond) : inside);
        return grid ? fminf(value, this->obstacle(c)) : value;
    }
    __hostdev__ bool solid(const nanovdb::Coord& c) const { return this->at(c) < 0.0f; }
};

// Fraction of the segment from a fluid cell to an air cell that lies in the fluid, not below minimum.
// minimum >= 1 gives the first-order rule.
__hostdev__ inline Real theta(Real phiFluid, Real phiAir, Real minimum)
{
    const Real t = phiFluid / (phiFluid - phiAir);
    return t > minimum ? (t < Real(1) ? t : Real(1)) : (minimum < Real(1) ? minimum : Real(1));
}

// After fluid::clear: a face is open if the cells on both sides are free of solids, and a cell is fluid if its phi
// is negative and it is free of solids. (fluid::clear knows the tank only; this replaces what it decided.)
__global__ void setFluid(Domain d, const float* phi, Solid solid)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (!i || i >= d.count) return;
    const bool free = !solid.solid(d.coord(i));
    d.cells[i].phi   = Real(phi[i]);
    d.cells[i].fluid = phi[i] < 0.0f && free;
    for (int a = 0; a < 3; ++a) {
        const int lo = d.neighbor(i, 2 * a);
        d.cells[i].open[a] = lo && free && !solid.solid(d.coord(lo)) ? Real(1) : Real(0);
    }
}

// Is cell k of the grid inside a solid? (Cell 0 stands for "no cell".)
__device__ inline bool isWall(const Domain& d, const Solid& solid, int k) { return k && solid.solid(d.coord(k)); }

// Wall faces. A wall face is the face between a free cell and a solid cell. It is closed (no flow, the
// APIC example's open = 0) unless it is released: the wall then lets go of the water, and the face becomes a free
// surface with zero pressure at the face itself, half a cell from the fluid cell's center (theta = 0.5).
// released[i] holds, as bit a, whether the lower face of cell i along axis a is a released wall face.
constexpr Real kWallTheta = Real(0.5);

// Set open on the wall faces from the released bits. Only a face of a fluid cell can be released.
__global__ void applyRelease(Domain d, const float* released, Solid solid)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (!i || i >= d.count) return;
    const bool in = !solid.solid(d.coord(i));
    for (int a = 0; a < 3; ++a) {
        const int lo = d.neighbor(i, 2 * a);
        if (!lo) { d.cells[i].open[a] = 0; continue; }
        const bool lowerIn = !solid.solid(d.coord(lo));
        if (in == lowerIn) continue; // not a wall face: setFluid decided
        const bool fluid = d.cells[in ? i : lo].fluid;
        d.cells[i].open[a] = fluid && ((int(released[i]) >> a) & 1) ? Real(1) : Real(0);
    }
}

// Decide every wall face of a fluid cell again from the pressure just solved: the face is released exactly when,
// released, the water would move away from the wall. ustar is the velocity before the projection (the sidecars,
// plus gravity on the vertical faces). changes counts the faces whose state flips, count the released faces.
// closeOnly: a closed face stays closed; only released faces that would let water into the wall are closed.
__global__ void updateRelease(Domain d, const Real* q, const float* u0, const float* u1, const float* u2, Real gravityStep,
                              float* released, int* changes, int* count, bool closeOnly, int axes, Solid solid)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (!i || i >= d.count) return;
    const float* u[3] = {u0, u1, u2};
    const bool   in   = !solid.solid(d.coord(i));
    int          bits = int(released[i]);
    for (int a = 0; a < 3; ++a) {
        const int lo = d.neighbor(i, 2 * a);
        if (!lo) { bits &= ~(1 << a); continue; }
        const bool lowerIn = !solid.solid(d.coord(lo));
        if (in == lowerIn) { bits &= ~(1 << a); continue; }
        const int tankCell = in ? i : lo;
        if (!d.cells[tankCell].fluid) { bits &= ~(1 << a); continue; }
        const Real ustar = Real(u[a][i]) - (a == 1 ? gravityStep : Real(0));
        // The face is the lower face of i. Tank cell i: the wall is below, away from it is +a. Tank cell lo: -a.
        const Real velocity = in ? ustar - q[i] / kWallTheta : ustar + q[lo] / kWallTheta;
        const Real away     = in ? velocity : -velocity;
        // Above the noise: a ten-thousandth of what gravity adds in a step, and never below 1e-6.
        const Real noise    = Real(1e-4) * (gravityStep < 0 ? -gravityStep : gravityStep) + Real(1e-6);
        const bool now      = (bits >> a) & 1;
        // Which walls may let go. axes = 0 (the rule in use): only the faces from which gravity pulls the water away,
        // that is, faces with the wall above and the water below. axes != 0 (experiments): bit a = walls normal to
        // axis a. Measured on a dam break with a 32-voxel column: letting the side walls go as well made the speeds
        // grow fivefold and lost 14% of the volume by T* = 6.5; floor and ceiling only stayed within 0.7%.
        const bool allowed  = axes ? ((axes >> a) & 1) != 0 : (a == 1 && !in);
        const bool want     = allowed && (closeOnly ? now && away > 0 : away > noise);
        if (want != now) atomicAdd(changes, 1);
        if (want) { bits |= 1 << a; atomicAdd(count, 1); }
        else bits &= ~(1 << a);
    }
    released[i] = float(bits);
}

// Diagonal of the matrix. missing counts the open faces of fluid cells whose other cell is not in the grid.
__global__ void assemble(Domain d, Real* diag, Real minimumTheta, int* missing, Solid solid)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d.count) return;
    diag[i] = 0;
    if (!i || !d.cells[i].fluid) return;
    const Cell& c = d.cells[i];
    Real        sum = 0;
    for (int j = 0; j < 6; ++j) {
        const int  k = d.neighbor(i, j);
        const Real w = d.cells[(j & 1) ? k : i].open[j / 2];
        if (!k) { if (!(j & 1) && w > 0) atomicAdd(missing, 1); continue; }
        if (w == 0) continue;
        if (d.cells[k].fluid) sum += w;
        else sum += w / (isWall(d, solid, k) ? kWallTheta : theta(c.phi, d.cells[k].phi, minimumTheta));
    }
    diag[i] = sum;
}

// Right-hand side of the projection: minus the net flux out of each fluid cell.
__global__ void fluxRhs(Domain d, Real* b)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d.count) return;
    b[i] = i && d.cells[i].fluid ? -fluid::netFlux(d, i) : Real(0);
}

// Subtract the gradient of q from the velocity of every open face that touches a fluid cell, and mark exactly
// those faces valid: they are the seeds of the extrapolation into the air (fluid::prepareTransport).
__global__ void project(Domain d, const Real* q, Real minimumTheta, Solid solid)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (!i || i >= d.count) return;
    Cell& c = d.cells[i];
    for (int a = 0; a < 3; ++a) {
        c.valid[a] = 0;
        if (c.open[a] == 0) { c.u[a] = c.wall[a]; continue; }
        const int   lo = d.neighbor(i, 2 * a);
        const Cell& l  = d.cells[lo];
        c.valid[a] = c.fluid || l.fluid;
        if (c.fluid && l.fluid) c.u[a] -= q[i] - q[lo];
        else if (c.fluid) c.u[a] -= q[i] / (isWall(d, solid, lo) ? kWallTheta : theta(c.phi, l.phi, minimumTheta));
        else if (l.fluid) c.u[a] += q[lo] / (isWall(d, solid, i) ? kWallTheta : theta(l.phi, c.phi, minimumTheta));
    }
}

// The cells of a filled grid, with the solids given by solid.
struct Cells
{
    fluid::Buffer<Cell>     cells;
    fluid::Buffer<uint32_t> address;
    fluid::Buffer<int>      missing;
    Domain                  d{};
    Solid                   solid;

    Cells(const fluid::Grid* grid, uint32_t leafCount, uint64_t valueCount, const float* phi, const Solid& where, int nx, int ny, int nz)
        : cells(fluid::allocate<Cell>(valueCount)), address(fluid::allocate<uint32_t>(valueCount)), missing(fluid::allocate<int>(1))
        , solid(where)
    {
        d.grid = grid, d.cells = cells.data(), d.address = address.data(), d.count = int(valueCount);
        d.nx = nx, d.ny = ny, d.nz = nz, d.h = 1;
        cudaCheck(cudaMemset(address.data(), 0, sizeof(uint32_t)));
        cudaCheck(cudaMemset(missing.data(), 0, sizeof(int)));
        fluid::enumerate<<<leafCount, 512>>>(d);
        fluid::clear<<<fluid::blocks(d.count), 256>>>(d);
        setFluid<<<fluid::blocks(d.count), 256>>>(d, phi, solid);
        cudaCheck(cudaDeviceSynchronize());
    }

    struct Result { int iterations, levels, missing; double residual, norm, setupMs, solveMs; };

    // Solve A q = b. b is zero outside the fluid cells.
    Result solve(const Real* b, Real* q, Real minimumTheta, int maxIterations = 2000)
    {
        const int n = d.count;
        auto diag = fluid::allocate<Real>(n), r = fluid::allocate<Real>(n), s = fluid::allocate<Real>(n), as = fluid::allocate<Real>(n);
        cudaCheck(cudaMemset(missing.data(), 0, sizeof(int)));
        assemble<<<fluid::blocks(n), 256>>>(d, diag.data(), minimumTheta, missing.data(), solid);
        fluid::Multigrid mg(d, diag.data(), false);
        const auto state = mg.solve(b, q, r.data(), s.data(), as.data(), false, maxIterations);
        if (state.breakdown) throw std::runtime_error("pressure: matrix breakdown");
        int miss = 0;
        cudaCheck(cudaMemcpy(&miss, missing.data(), sizeof(int), cudaMemcpyDeviceToHost));
        return {state.iterations, mg.levelCount(), miss, state.residual, state.norm, mg.setupMs, mg.solveMs};
    }
};

} // namespace pressure
