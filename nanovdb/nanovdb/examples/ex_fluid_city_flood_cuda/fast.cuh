// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

// fast.cuh: gravity, pressure projection and velocity extrapolation of one step, with the same rules
// as pressure.cuh + the APIC example's Cell kernels (fluid::clear, netFlux, prepareTransport, extrapolate) and
// eulerian.cu's loadVelocity / storeVelocity, but without the Cell struct:
//   - the velocity stays in its three float sidecars and is changed in place;
//   - what else a cell needs is one byte of flags (fluid, the three open lower faces, the three valid components);
//   - a kernel visits the voxels leaf by leaf (one block of 512 threads per leaf), and finds a neighbor inside
//     the same leaf or, across a leaf face, in the neighbor leaf listed in a table of 6 leaves per leaf. No tree
//     is walked per voxel.
// The pressure system is solved by mg.cuh on the fluid cells only.
#pragma once
#include "mg.cuh"
#include <optional>
#include <thrust/reduce.h>
#include <thrust/scan.h>

namespace fast {

using fluid::Coord;
using pressure::kWallTheta;
using pressure::Solid;
using GridT = fluid::Grid;

constexpr unsigned char kFluid = 1, kOpen = 2, kValid = 16; // kOpen << a: the lower face along a is open; kValid << a: component a is valid

// The domain grid as the kernels see it.
struct Mesh
{
    const GridT*    grid;
    const uint32_t* leafNeighbor; // 6 per leaf (-x, +x, -y, +y, -z, +z): the index of the leaf across that face, or ~0u
    uint32_t        leafCount;
    int             count; // voxels + 1
    // The value index of the neighbor of voxel n of leaf l in direction j, or 0 if that voxel is not in the grid.
    __device__ int neighbor(uint32_t l, uint32_t n, int j) const
    {
        const int      shift = 6 - 3 * (j >> 1); // a leaf's voxel n is (x << 6) | (y << 3) | z
        const uint32_t c     = (n >> shift) & 7u;
        uint32_t       L     = l, m;
        if (j & 1) {
            if (c < 7u) m = n + (1u << shift);
            else L = leafNeighbor[6 * l + j], m = n - (7u << shift);
        } else {
            if (c > 0u) m = n - (1u << shift);
            else L = leafNeighbor[6 * l + j], m = n + (7u << shift);
        }
        if (L == ~0u) return 0;
        const auto& node = grid->tree().getFirstNode<0>()[L];
        return node.isActive(m) ? int(node.getValue(m)) : 0;
    }
};
// For a kernel launched with one block of 512 threads per leaf: this thread's voxel, or return.
#define FAST_VOXEL(mesh)                                                \
    const uint32_t l = blockIdx.x, n = threadIdx.x;                     \
    const auto&    node = (mesh).grid->tree().getFirstNode<0>()[l];     \
    if (!node.isActive(n)) return;                                      \
    const int   i   = int(node.getValue(n));                            \
    const Coord ijk = node.offsetToGlobalCoord(n);

__global__ void leafNeighbors(const GridT* grid, uint32_t* out, uint32_t leafCount)
{
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= 6 * leafCount) return;
    const auto* first = grid->tree().getFirstNode<0>();
    const int   j     = int(t % 6);
    Coord       c     = first[t / 6].origin();
    c[j >> 1] += (j & 1) ? 8 : -8;
    const auto* other = grid->tree().root().probeLeaf(c);
    out[t] = other ? uint32_t(other - first) : ~0u;
}

// pressure::setFluid and pressure::applyRelease: a cell is fluid if phi < 0 and it is free of solids. A lower face
// is open if the cell below is in the grid and either both cells are free, or it is a wall face (one of them
// solid) of a fluid cell that is released.
__global__ void classify(Mesh mesh, const float* phi, const float* released, Solid solid, unsigned char* flags)
{
    FAST_VOXEL(mesh)
    const bool    free = !solid.solid(ijk);
    unsigned char f    = phi[i] < 0.0f && free ? kFluid : 0;
    for (int a = 0; a < 3; ++a) {
        const int lo = mesh.neighbor(l, n, 2 * a);
        if (!lo) continue;
        Coord lower = ijk;
        lower[a] -= 1;
        const bool lowerFree = !solid.solid(lower);
        bool       open;
        if (free == lowerFree) open = free;
        else open = (free ? (f & kFluid) != 0 : phi[lo] < 0.0f) && ((int(released[i]) >> a) & 1);
        if (open) f |= kOpen << a;
    }
    flags[i] = f;
}

// Numbering of the fluid cells: after the running sum of the fluid flags, zero the others.
__global__ void markFluid(const unsigned char* flags, int* fluidOf, int count)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) fluidOf[i] = i && (flags[i] & kFluid) ? 1 : 0;
}
__global__ void numberFluid(const unsigned char* flags, int* fluidOf, int count)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count && !(i && (flags[i] & kFluid))) fluidOf[i] = 0;
}
// The neighbors of each fluid cell among the fluid cells, and its block and place for the coarser levels.
__global__ void fineTopology(Mesh mesh, const unsigned char* flags, const int* fluidOf, mg::Level fine, Coord* block, unsigned char* place)
{
    FAST_VOXEL(mesh)
    if (!(flags[i] & kFluid)) return;
    const int f = fluidOf[i];
    for (int j = 0; j < 6; ++j) fine.nbr[j][f] = fluidOf[mesh.neighbor(l, n, j)]; // fluidOf[0] = 0
    block[f - 1] = ijk >> 1;
    place[f - 1] = (unsigned char)((ijk[0] & 1) | ((ijk[1] & 1) << 1) | ((ijk[2] & 1) << 2));
}

// The velocity of the lower face along a of cell k before the projection: the sidecar plus gravity on an open
// face, zero on a closed one (eulerian.cu loadVelocity and fluid::flux with a wall at rest).
__device__ inline double faceVelocity(const unsigned char* flags, const float* const u[3], int k, int a, double gravityStep)
{
    return (flags[k] & (kOpen << a)) ? double(u[a][k]) - (a == 1 ? gravityStep : 0.0) : 0.0;
}

// For every fluid cell: b = divergence - (net flux out of it) (pressure::fluxRhs) and the diagonal (pressure::assemble).
// divergence: the net flux every fluid cell is to have after the projection (zero, or the volume control's value:
// Kim et al. 2007, Simulation of Bubbles in Foam With The Volume Control Method, eq. 4).
__global__ void rightHandSideAndDiagonal(Mesh mesh, const unsigned char* flags, const float* phi, const float* u0, const float* u1,
                                         const float* u2, double gravityStep, double minimumTheta, Solid solid, const int* fluidOf,
                                         float* b, float* diag, double divergence, const float* source)
{
    FAST_VOXEL(mesh)
    if (!(flags[i] & kFluid)) return;
    const float* const u[3] = {u0, u1, u2};
    double flux = 0, sum = 0;
    for (int j = 0; j < 6; ++j) {
        const int a = j >> 1, k = mesh.neighbor(l, n, j);
        const int faceCell = (j & 1) ? k : i; // the cell whose lower face this is
        flux += ((j & 1) ? 1.0 : -1.0) * faceVelocity(flags, u, faceCell, a, gravityStep); // flags[0] = 0: no face
        if (!k || !(flags[faceCell] & (kOpen << a))) continue;
        if (flags[k] & kFluid) { sum += 1.0; continue; }
        Coord other = ijk;
        other[a] += (j & 1) ? 1 : -1;
        sum += 1.0 / (solid.solid(other) ? double(kWallTheta) : double(pressure::theta(pressure::Real(phi[i]), pressure::Real(phi[k]), pressure::Real(minimumTheta))));
    }
    b[fluidOf[i]]    = float(divergence + (source ? double(source[i]) : 0.0) - flux);
    diag[fluidOf[i]] = float(sum);
}

// pressure::updateRelease with q numbered by fluid cell.
__global__ void updateRelease(Mesh mesh, const unsigned char* flags, const double* q, const int* fluidOf, const float* u0, const float* u1,
                              const float* u2, double gravityStep, float* released, int* changes, int* count, bool closeOnly, int axes,
                              Solid solid)
{
    FAST_VOXEL(mesh)
    const float* const u[3] = {u0, u1, u2};
    const bool   in   = !solid.solid(ijk);
    int          bits = int(released[i]);
    for (int a = 0; a < 3; ++a) {
        const int lo = mesh.neighbor(l, n, 2 * a);
        if (!lo) { bits &= ~(1 << a); continue; }
        Coord lower = ijk;
        lower[a] -= 1;
        const bool lowerIn = !solid.solid(lower);
        if (in == lowerIn) { bits &= ~(1 << a); continue; }
        const int tankCell = in ? i : lo;
        if (!(flags[tankCell] & kFluid)) { bits &= ~(1 << a); continue; }
        const double ustar    = double(u[a][i]) - (a == 1 ? gravityStep : 0.0);
        const double velocity = in ? ustar - q[fluidOf[i]] / double(kWallTheta) : ustar + q[fluidOf[lo]] / double(kWallTheta);
        const double away     = in ? velocity : -velocity;
        const double noise    = 1e-4 * (gravityStep < 0 ? -gravityStep : gravityStep) + 1e-6;
        const bool   now      = (bits >> a) & 1;
        const bool   allowed  = axes ? ((axes >> a) & 1) != 0 : (a == 1 && !in);
        const bool   want     = allowed && (closeOnly ? now && away > 0 : away > noise);
        if (want != now) atomicAdd(changes, 1);
        if (want) { bits |= 1 << a; atomicAdd(count, 1); }
        else bits &= ~(1 << a);
    }
    released[i] = float(bits);
}

// pressure::project, in place on the sidecars: gravity, minus the gradient of q on every open face that touches a
// fluid cell; those faces become valid. A closed face gets the wall's velocity, zero.
__global__ void project(Mesh mesh, unsigned char* flags, const float* phi, float* u0, float* u1, float* u2, const double* q,
                        const int* fluidOf, double gravityStep, double minimumTheta, Solid solid)
{
    FAST_VOXEL(mesh)
    float*        u[3] = {u0, u1, u2};
    unsigned char f    = flags[i] & (kFluid | (kOpen * 7));
    const bool    fluid = f & kFluid;
    for (int a = 0; a < 3; ++a) {
        if (!(f & (kOpen << a))) { u[a][i] = 0.0f; continue; }
        const int  lo = mesh.neighbor(l, n, 2 * a);
        const bool lowerFluid = flags[lo] & kFluid;
        double     value = double(u[a][i]) - (a == 1 ? gravityStep : 0.0);
        Coord      lower = ijk;
        lower[a] -= 1;
        if (fluid && lowerFluid) value -= q[fluidOf[i]] - q[fluidOf[lo]];
        else if (fluid) value -= q[fluidOf[i]] / (solid.solid(lower) ? double(kWallTheta) : double(pressure::theta(pressure::Real(phi[i]), pressure::Real(phi[lo]), pressure::Real(minimumTheta))));
        else if (lowerFluid) value += q[fluidOf[lo]] / (solid.solid(ijk) ? double(kWallTheta) : double(pressure::theta(pressure::Real(phi[lo]), pressure::Real(phi[i]), pressure::Real(minimumTheta))));
        if (fluid || lowerFluid) f |= kValid << a;
        u[a][i] = float(value);
    }
    flags[i] = f;
}

// Per leaf: the sum over its fluid cells of the squared (net flux after the projection - divergence), and their number.
__global__ void leafFlux(Mesh mesh, const unsigned char* flags, const float* u0, const float* u1, const float* u2, double* square, int* cells,
                         double divergence, const float* source)
{
    __shared__ double partial[512];
    __shared__ int    counted[512];
    const uint32_t l = blockIdx.x, n = threadIdx.x;
    const auto&    node = mesh.grid->tree().getFirstNode<0>()[l];
    double         value = 0;
    int            one = 0;
    if (node.isActive(n)) {
        const int i = int(node.getValue(n));
        if (flags[i] & kFluid) {
            const float* const u[3] = {u0, u1, u2};
            double flux = 0;
            for (int a = 0; a < 3; ++a)
                flux += faceVelocity(flags, u, mesh.neighbor(l, n, 2 * a + 1), a, 0.0) - faceVelocity(flags, u, i, a, 0.0);
            flux -= divergence + (source ? double(source[i]) : 0.0);
            value = flux * flux, one = 1;
        }
    }
    partial[n] = value, counted[n] = one;
    __syncthreads();
    for (int width = 256; width; width >>= 1) {
        if (n < width) partial[n] += partial[n + width], counted[n] += counted[n + width];
        __syncthreads();
    }
    if (!n) square[l] = partial[0], cells[l] = counted[0];
}

// Per leaf: the water volume in its free cells, each cell counting by the fraction a flat surface at distance phi
// cuts from it (the measure of the driver's reports).
__global__ void leafVolume(Mesh mesh, const float* phi, Solid solid, double* volume)
{
    __shared__ double partial[512];
    const uint32_t l = blockIdx.x, n = threadIdx.x;
    const auto&    node = mesh.grid->tree().getFirstNode<0>()[l];
    double         value = 0;
    if (node.isActive(n) && !solid.solid(node.offsetToGlobalCoord(n)))
        value = fmin(1.0, fmax(0.0, 0.5 - double(phi[node.getValue(n)])));
    partial[n] = value;
    __syncthreads();
    for (int width = 256; width; width >>= 1) {
        if (n < width) partial[n] += partial[n + width];
        __syncthreads();
    }
    if (!n) volume[l] = partial[0];
}

// Wall friction on one velocity component, before the projection. A face slides along a wall where the two cells
// next to it across the face (in a direction b other than the component's axis) are both solid; a face can have up to
// four such walls. The wall's shear stress tau = rho Cf |U| u / 2 acts on the one layer of cells at the wall, so
//   du/dt = -Cf |U| u / (2 h) per wall,   h = one voxel,   U = the velocity at the face
// (the other two components are averaged from the four faces around it). Taken implicitly in u, with lengths in
// voxels: factor = 1 / (1 + walls * Cf * |U| * dt / 2). The kernel writes the factor; scaleBy applies it, so every
// component is damped from the same, unchanged velocity. Faces of a solid cell are left alone (factor 1).
__global__ void wallFriction(Mesh mesh, const float* u0, const float* u1, const float* u2, int axis, Solid solid, float frictionStep,
                             float* factor)
{
    FAST_VOXEL(mesh)
    factor[i] = 1.0f;
    Coord lower = ijk;
    lower[axis] -= 1;
    if (solid.solid(ijk) || solid.solid(lower)) return;
    int walls = 0;
    for (int b = 0; b < 3; ++b) {
        if (b == axis) continue;
        for (int side = -1; side <= 1; side += 2) {
            Coord p = ijk, q = lower;
            p[b] += side, q[b] += side;
            walls += solid.solid(p) && solid.solid(q);
        }
    }
    if (!walls) return;
    const float* u[3] = {u0, u1, u2};
    const auto   acc  = mesh.grid->getAccessor();
    float        square = u[axis][i] * u[axis][i];
    for (int b = 0; b < 3; ++b) {
        if (b == axis) continue;
        Coord upper = ijk, lowerUpper = lower; // component b on the lower faces of ijk, lower and of their +b neighbors
        upper[b] += 1, lowerUpper[b] += 1;
        const float mean = 0.25f * (u[b][i] + u[b][acc.getValue(lower)] + u[b][acc.getValue(upper)] + u[b][acc.getValue(lowerUpper)]);
        square += mean * mean;
    }
    factor[i] = 1.0f / (1.0f + float(walls) * frictionStep * 0.5f * sqrtf(square));
}

__global__ void scaleBy(float* value, const float* factor, int count)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > 0 && i < count) value[i] *= factor[i];
}

// One sweep of fluid::extrapolate for one component: a value that is not valid becomes the mean of its valid
// neighbors, and valid, if it has any.
__global__ void extrapolateSweep(Mesh mesh, const float* in, const unsigned char* validIn, float* out, unsigned char* validOut)
{
    FAST_VOXEL(mesh)
    if (validIn[i]) { out[i] = in[i], validOut[i] = 1; return; }
    float sum   = 0;
    int   found = 0;
    for (int j = 0; j < 6; ++j) {
        const int k = mesh.neighbor(l, n, j);
        if (k && validIn[k]) sum += in[k], ++found;
    }
    out[i] = found ? sum / float(found) : in[i], validOut[i] = found > 0;
}
__global__ void getValid(const unsigned char* flags, int a, unsigned char* valid, int count)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) valid[i] = i ? (flags[i] >> (4 + a)) & 1 : 0;
}
__global__ void setValid(unsigned char* flags, int a, const unsigned char* valid, int count)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i && i < count) flags[i] = (unsigned char)((flags[i] & ~(kValid << a)) | (valid[i] ? (kValid << a) : 0));
}

// eulerian.cu storeVelocity: a component the extrapolation did not reach is zero (invalid counts those on faces
// of free cells); a closed wall face keeps a velocity only if it points away from the wall and the free cell is air.
__global__ void finish(Mesh mesh, const unsigned char* flags, float* u0, float* u1, float* u2, int* invalid, Solid solid)
{
    FAST_VOXEL(mesh)
    float*     u[3] = {u0, u1, u2};
    const bool in   = !solid.solid(ijk);
    for (int a = 0; a < 3; ++a) {
        Coord lower = ijk;
        lower[a] -= 1;
        const bool lowerIn = !solid.solid(lower);
        float      value   = u[a][i];
        if (!(flags[i] & (kValid << a))) { value = 0.0f; if (in || lowerIn) atomicAdd(invalid, 1); }
        if (in != lowerIn && !(flags[i] & (kOpen << a))) {
            const bool intoWall = in ? value < 0.0f : value > 0.0f; // the face is the lower face of i
            if (intoWall || (flags[in ? i : mesh.neighbor(l, n, 2 * a)] & kFluid)) value = 0.0f;
        }
        u[a][i] = value;
    }
}

// Steps 5 and 6 of Simulation::step on the domain grid.
class Pipeline
{
    fluid::Buffer<uint32_t>      leafNeighbor;
    fluid::Buffer<unsigned char> flags;
    fluid::Buffer<int>           fluidOf, counters;
    std::optional<mg::Multigrid> multigrid;
    Mesh                         mesh;
    Solid                        solid;
    static int blocks(int n) { return (n + 255) / 256; }

public:
    double setupMs = 0, solveMs = 0;
    int    fluidCells = 0;
    double divergence = 0; // the net flux every fluid cell is to have after the projection (volume control)
    const float* source = nullptr; // per voxel, or null: a further net flux for that cell (volume per second; spray.cuh)

    // The water volume in the grid's free cells.
    double volume(const float* phi)
    {
        auto perLeaf = fluid::allocate<double>(mesh.leafCount);
        leafVolume<<<mesh.leafCount, 512>>>(mesh, phi, solid, perLeaf.data());
        return thrust::reduce(thrust::device, perLeaf.data(), perLeaf.data() + mesh.leafCount, 0.0);
    }

    Pipeline(const GridT* grid, uint32_t leafCount, uint64_t valueCount, const Solid& where)
        : leafNeighbor(fluid::allocate<uint32_t>(6 * size_t(leafCount))), flags(fluid::allocate<unsigned char>(valueCount)),
          fluidOf(fluid::allocate<int>(1)), counters(fluid::allocate<int>(2)), solid(where)
    {
        leafNeighbors<<<blocks(6 * int(leafCount)), 256>>>(grid, leafNeighbor.data(), leafCount);
        cudaCheck(cudaMemset(flags.data(), 0, valueCount)); // slot 0 stays 0
        mesh = Mesh{grid, leafNeighbor.data(), leafCount, int(valueCount)};
    }

    void classify(const float* phi, const float* released)
    {
        fast::classify<<<mesh.leafCount, 512>>>(mesh, phi, released, solid, flags.data());
    }

    // The levels of the pressure solver, for the fluid cells classify() found. They hold for every later
    // classify() of the same phi: releasing wall faces changes the open faces, not the fluid cells.
    void buildSolver(float correctionScale, int sweeps)
    {
        fluid::Timer timer;
        fluidOf = fluid::allocate<int>(mesh.count);
        markFluid<<<blocks(mesh.count), 256>>>(flags.data(), fluidOf.data(), mesh.count);
        thrust::inclusive_scan(thrust::device, fluidOf.data(), fluidOf.data() + mesh.count, fluidOf.data());
        cudaCheck(cudaMemcpy(&fluidCells, fluidOf.data() + mesh.count - 1, sizeof(int), cudaMemcpyDeviceToHost));
        numberFluid<<<blocks(mesh.count), 256>>>(flags.data(), fluidOf.data(), mesh.count);
        multigrid.emplace(fluidCells);
        multigrid->correctionScale = correctionScale, multigrid->sweeps = sweeps;
        auto block = fluid::allocate<Coord>(std::max(fluidCells, 1));
        auto place = fluid::allocate<unsigned char>(std::max(fluidCells, 1));
        fineTopology<<<mesh.leafCount, 512>>>(mesh, flags.data(), fluidOf.data(), multigrid->fine(), block.data(), place.data());
        multigrid->buildCoarse(std::move(block), std::move(place));
        setupMs = timer.ms();
    }

    // Gravity and the pressure of the current open faces. warm: start from the previous solve's pressure.
    mg::Multigrid::Result solve(const float* phi, const float* const u[3], double gravityStep, double minimumTheta, bool warm, double tolerance)
    {
        fluid::Timer timer;
        rightHandSideAndDiagonal<<<mesh.leafCount, 512>>>(mesh, flags.data(), phi, u[0], u[1], u[2], gravityStep, minimumTheta, solid,
                                                         fluidOf.data(), multigrid->b(), multigrid->diag(), divergence, source);
        const auto result = multigrid->solve(warm, tolerance);
        solveMs += timer.ms();
        return result;
    }

    // Decide the released wall faces again from the pressure. Returns the number of faces that changed; released
    // faces are counted into releasedFaces.
    int updateRelease(const float* const u[3], double gravityStep, float* released, bool closeOnly, int axes, int& releasedFaces)
    {
        cudaCheck(cudaMemset(counters.data(), 0, 2 * sizeof(int)));
        fast::updateRelease<<<mesh.leafCount, 512>>>(mesh, flags.data(), multigrid->q(), fluidOf.data(), u[0], u[1], u[2], gravityStep,
                                                    released, counters.data(), counters.data() + 1, closeOnly, axes, solid);
        int host[2];
        cudaCheck(cudaMemcpy(host, counters.data(), 2 * sizeof(int), cudaMemcpyDeviceToHost));
        releasedFaces = host[1];
        return host[0];
    }

    // Project the sidecars in place. Returns the root mean square of the net flux of the fluid cells afterwards.
    double project(const float* phi, float* const u[3], double gravityStep, double minimumTheta)
    {
        fast::project<<<mesh.leafCount, 512>>>(mesh, flags.data(), phi, u[0], u[1], u[2], multigrid->q(), fluidOf.data(), gravityStep,
                                              minimumTheta, solid);
        auto square = fluid::allocate<double>(mesh.leafCount);
        auto cells  = fluid::allocate<int>(mesh.leafCount);
        leafFlux<<<mesh.leafCount, 512>>>(mesh, flags.data(), u[0], u[1], u[2], square.data(), cells.data(), divergence, source);
        const double sum   = thrust::reduce(thrust::device, square.data(), square.data() + mesh.leafCount, 0.0);
        const long   count = thrust::reduce(thrust::device, cells.data(), cells.data() + mesh.leafCount, 0L);
        return count ? std::sqrt(sum / double(count)) : 0.0;
    }

    // Wall friction with the coefficient Cf (wallFriction above); frictionStep = Cf * dt.
    void friction(float* const u[3], float frictionStep)
    {
        fluid::Buffer<float> factor[3] = {fluid::allocate<float>(mesh.count), fluid::allocate<float>(mesh.count), fluid::allocate<float>(mesh.count)};
        for (int a = 0; a < 3; ++a) {
            cudaCheck(cudaMemset(factor[a].data(), 0, sizeof(float)));
            wallFriction<<<mesh.leafCount, 512>>>(mesh, u[0], u[1], u[2], a, solid, frictionStep, factor[a].data());
        }
        for (int a = 0; a < 3; ++a) scaleBy<<<blocks(mesh.count), 256>>>(u[a], factor[a].data(), mesh.count);
    }

    // The solver's memory is not needed after the projection.
    void releaseSolver() { multigrid.reset(), fluidOf = fluid::allocate<int>(1); }

    // Extend the velocity from the valid faces by `sweeps` sweeps, then the rules at closed wall faces. Returns the
    // number of components on faces of free cells that no sweep reached.
    int extrapolate(float* const u[3], int sweeps)
    {
        auto other      = fluid::allocate<float>(mesh.count);
        auto valid      = fluid::allocate<unsigned char>(mesh.count);
        auto otherValid = fluid::allocate<unsigned char>(mesh.count);
        for (int a = 0; a < 3; ++a) {
            getValid<<<blocks(mesh.count), 256>>>(flags.data(), a, valid.data(), mesh.count);
            cudaCheck(cudaMemset(otherValid.data(), 0, 1));
            float*         in  = u[a];
            float*         out = other.data();
            unsigned char *validIn = valid.data(), *validOut = otherValid.data();
            for (int k = 0; k < sweeps; ++k) {
                extrapolateSweep<<<mesh.leafCount, 512>>>(mesh, in, validIn, out, validOut);
                std::swap(in, out), std::swap(validIn, validOut);
            }
            if (in != u[a]) cudaCheck(cudaMemcpy(u[a], in, size_t(mesh.count) * sizeof(float), cudaMemcpyDeviceToDevice));
            setValid<<<blocks(mesh.count), 256>>>(flags.data(), a, validIn, mesh.count);
        }
        cudaCheck(cudaMemset(counters.data(), 0, sizeof(int)));
        finish<<<mesh.leafCount, 512>>>(mesh, flags.data(), u[0], u[1], u[2], counters.data(), solid);
        int invalid = 0;
        cudaCheck(cudaMemcpy(&invalid, counters.data(), sizeof(int), cudaMemcpyDeviceToHost));
        for (int a = 0; a < 3; ++a) cudaCheck(cudaMemset(u[a], 0, sizeof(float))); // slot 0
        return invalid;
    }
};

} // namespace fast
