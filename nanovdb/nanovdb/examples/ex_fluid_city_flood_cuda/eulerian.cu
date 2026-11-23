// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

// eulerian.cu: a level-set free-surface solver on NanoVDB index grids, put together from pieces checked one by one:
//   advect.cuh (MacCormack + clamp), LevelSetTracker (dilate, normalize, prune), FillInterior.cuh, pressure.cuh
//   (ghost-fluid free surface + the APIC example's multigrid CG) and the APIC example's velocity extrapolation.
//
// Units: lengths in voxels, time in seconds. Cell ijk is centered at ijk; velocity component a of a cell sits on its
// lower face along a. The tank is the cells [0, nx) x [0, ny) x [0, nz), closed on all six sides, free slip.
//
// State: the band (index grid + phi + invert masks) and the domain grid (band + interior + a halo of air) with the
// velocity as three float sidecars. The level set is kept closed by letting the water continue `extension` voxels
// behind every wall it touches: in a cell behind a wall, phi = max(phi of the nearest tank cell, distance to the
// tank grown by extension).
//
// One step:
//   1. dt from the largest speed (cfl)
//   2. phi and the velocity are carried by the velocity
//   3. band: dilate, wall rule, normalize, prune
//   4. domain: fill the interior, add the halo, move the velocity over
//   5. gravity, pressure solve, projection
//   6. velocity extrapolated from the fluid's faces into the air and behind the walls
//
// Modes:
//   hydrostatic <N> <height> <steps>          water at rest in an N^3 tank; it must stay at rest
//   dam <nx> <ny> <nz> <width> <height> <T*>  a column of water against the wall x = 0, released; runs until the
//                                             dimensionless time T* = t sqrt(2 g / width)
#include <nanovdb/NanoVDB.h>
#include <nanovdb/HostBuffer.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/GridBuilder.h>
#include <nanovdb/tools/cuda/DilateGrid.cuh>
#include <nanovdb/tools/cuda/LevelSetTracker.cuh>
#include "FillInterior.cuh"
#include "advect.cuh"
#include "pressure.cuh"
#include "pressure2.cuh"
#include "fast.cuh"
#include "pls.cuh"
#include "spray.cuh"
#include <thrust/count.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <optional>
#include <string>
#include <thread>
#include <vector>

using BuildT    = nanovdb::ValueOnIndex;
using GridT     = nanovdb::NanoGrid<BuildT>;
using ResourceT = nanovdb::cuda::DeviceResource;
using TrackerT  = nanovdb::tools::cuda::LevelSetTracker<BuildT, float, ResourceT>;
using StateT    = TrackerT::State;
template <typename T>
using BufferT = TrackerT::BufferT<T>;
using HandleT = TrackerT::HandleT;
using pressure::Real;

static nanovdb::cuda::ResourceRef<ResourceT> ref()
{
    return nanovdb::cuda::ResourceRef<ResourceT>(nanovdb::cuda::default_resource<ResourceT>());
}

template <typename T>
static BufferT<T> makeBuffer(size_t n) { return BufferT<T>(cudaStream_t(0), ref(), n, nanovdb::cuda::noInit); }

template <typename T>
static std::vector<T> hostCopy(const BufferT<T>& b)
{
    std::vector<T> v(b.size());
    if (!v.empty()) cudaCheck(cudaMemcpy(static_cast<void*>(v.data()), b.data(), b.size_bytes(), cudaMemcpyDeviceToHost));
    return v;
}

struct Tank
{
    int n[3];
    int extension; // the water's level set ends this many voxels behind a wall face
    __hostdev__ bool inside(const nanovdb::Coord& c) const
    {
        return c[0] >= 0 && c[1] >= 0 && c[2] >= 0 && c[0] < n[0] && c[1] < n[1] && c[2] < n[2];
    }
    __hostdev__ nanovdb::Coord nearest(const nanovdb::Coord& c) const
    {
        nanovdb::Coord r;
        for (int a = 0; a < 3; ++a) r[a] = c[a] < 0 ? 0 : (c[a] >= n[a] ? n[a] - 1 : c[a]);
        return r;
    }
    // Signed distance to the tank grown by margin voxels, negative inside.
    __hostdev__ float distance(const nanovdb::Coord& c, float margin) const
    {
        float outside = 0.0f, inside = -1.0e30f;
        for (int a = 0; a < 3; ++a) {
            const float half = 0.5f * float(n[a]) + margin;
            const float q    = fabsf(float(c[a]) - 0.5f * float(n[a] - 1)) - half;
            if (q > 0.0f) outside += q * q;
            inside = q > inside ? q : inside;
        }
        return outside > 0.0f ? sqrtf(outside) : inside;
    }
    __hostdev__ float grown(const nanovdb::Coord& c) const { return this->distance(c, float(extension)); }
};

// Signed distance to the box [lo, hi], negative inside.
static double boxSdf(const double p[3], const double lo[3], const double hi[3])
{
    double outside = 0, inside = -1e30;
    for (int a = 0; a < 3; ++a) {
        const double q = std::fabs(p[a] - 0.5 * (lo[a] + hi[a])) - 0.5 * (hi[a] - lo[a]);
        if (q > 0) outside += q * q;
        inside = std::max(inside, q);
    }
    return outside > 0 ? std::sqrt(outside) : inside;
}

// The invert masks of a state, for device code.
struct InvertView
{
    const nanovdb::Mask<3>* leaf;
    const nanovdb::Mask<4>* lower;
    const nanovdb::Mask<5>* upper;
    const nanovdb::Coord*   roots;
    uint64_t                rootCount;
};

// VMAX: a velocity component held to [-limit, limit].
struct ClampSpeed
{
    float limit;
    __hostdev__ float operator()(float v) const { return v > limit ? limit : (v < -limit ? -limit : v); }
};
struct Faster
{
    float limit;
    __hostdev__ bool operator()(float v) const { return v > limit || v < -limit; }
};

struct AbsoluteValue
{
    __hostdev__ float operator()(float v) const { return v < 0 ? -v : v; }
};

// TRACE: the face with the largest |velocity|, and what the cells on its two sides look like.
struct Peak
{
    float value;
    int   cell, axis; // the face is the lower face of this cell along this axis
};
struct PeakMax
{
    __hostdev__ Peak operator()(const Peak& a, const Peak& b) const { return b.value > a.value ? b : a; }
};
struct ProjectedPeak // over the faces the projection updated
{
    const fluid::Cell* cells;
    __hostdev__ Peak operator()(int i) const
    {
        Peak p{0.0f, 0, 0};
        for (int a = 0; i && a < 3; ++a) {
            const float v = fabsf(float(cells[i].u[a]));
            if (cells[i].valid[a] && v > p.value) p = Peak{v, i, a};
        }
        return p;
    }
};
struct SidecarPeak
{
    const float* u[3];
    __hostdev__ Peak operator()(int i) const
    {
        Peak p{0.0f, 0, 0};
        for (int a = 0; i && a < 3; ++a)
            if (fabsf(u[a][i]) > p.value) p = Peak{fabsf(u[a][i]), i, a};
        return p;
    }
};
// TRACE with PIPE=2: what is around a face of the domain grid. The face is the lower face along `axis` of the voxel
// with the given index. The report is about the fluid cell F of the face's two cells (the upper one if both or
// neither are fluid): F and its six neighbors in the order -x +x -y +y -z +z ([0] = F, [1..6] = neighbors), and the
// velocities of F's six faces in the same order before the projection and now (positive along the axis).
struct VoxelReport
{
    int   coord[3], face;         // F, and which of its faces (0..5) the reported face is
    int   present[7];
    float phi[7], psi[7], source[7], released[7];
    float before[6], after[6];
};
__global__ void describeVoxel(const GridT* grid, uint64_t index, int axis, const float* phi, pressure::Solid solid, const float* source,
                              const float* released, const float* b0, const float* b1, const float* b2, const float* a0, const float* a1,
                              const float* a2, VoxelReport* out)
{
    const auto&    leaf = grid->tree().getFirstLeaf()[blockIdx.x];
    const uint32_t m    = threadIdx.x;
    if (!leaf.isActive(m) || leaf.getValue(m) != index) return;
    const float* const before[3] = {b0, b1, b2};
    const float* const after[3]  = {a0, a1, a2};
    nanovdb::Coord     c   = leaf.offsetToGlobalCoord(m);
    const auto         acc = grid->getAccessor();
    VoxelReport        r{};
    nanovdb::Coord     lower = c;
    lower[axis] -= 1;
    const uint64_t lowerIndex = acc.getValue(lower);
    const bool     upperFluid = phi[index] < 0.0f && !solid.solid(c);
    const bool     lowerFluid = lowerIndex && phi[lowerIndex] < 0.0f && !solid.solid(lower);
    r.face = 2 * axis; // the lower face of the upper cell
    if (!upperFluid && lowerFluid) c = lower, r.face = 2 * axis + 1;
    for (int a = 0; a < 3; ++a) r.coord[a] = c[a];
    for (int k = 0; k < 7; ++k) {
        nanovdb::Coord n = c;
        if (k) n[(k - 1) / 2] += (k & 1) ? -1 : 1;
        const uint64_t i = acc.getValue(n);
        r.present[k] = i != 0, r.phi[k] = i ? phi[i] : 3.0f, r.psi[k] = solid.at(n);
        r.source[k] = i && source ? source[i] : 0.0f, r.released[k] = i && released ? released[i] : 0.0f;
        if (k) { // the face between F and this neighbor: the lower face of the upper of the two
            const int      a = (k - 1) / 2;
            const uint64_t f = (k & 1) ? acc.getValue(c) : i;
            r.before[k - 1] = f ? before[a][f] : 0.0f, r.after[k - 1] = f ? after[a][f] : 0.0f;
        }
    }
    *out = r;
}

struct FaceReport
{
    int   coord[3], fluid[2], fluidNeighbors[2]; // [0] = the cell below the face, [1] = the cell above it; fluid -1 = no cell
    float phi[2], psi[2], q[2], open, theta, before, after;
    // Around the fluid cell of the face (the cell above it if both are fluid): itself, then its 6 neighbors in the order
    // -x +x -y +y -z +z; velocity = the face between them, positive out of the cell.
    int   center[3], nbFluid[7];
    float nbPhi[7], nbPsi[7], nbQ[7], nbOpen[7], nbBefore[7], nbAfter[7];
};
// before: the sidecar of this axis as it was before gravity and the projection. after: the sidecar after the
// extrapolation, or null for the velocity the projection left in the cells.
__global__ void describeFace(pressure::Domain d, int i, int a, const Real* q, const float* before, Real gravityStep,
                             Real minimumTheta, pressure::Solid solid, const float* after, FaceReport* out,
                             const float* before0, const float* before1, const float* before2)
{
    if (blockIdx.x || threadIdx.x) return;
    FaceReport r{};
    const int            cell[2] = {d.neighbor(i, 2 * a), i};
    const nanovdb::Coord ijk     = d.coord(i);
    for (int k = 0; k < 3; ++k) r.coord[k] = ijk[k];
    for (int s = 0; s < 2; ++s) {
        const int c = cell[s];
        if (!c) { r.fluid[s] = -1; continue; }
        r.fluid[s] = d.cells[c].fluid, r.phi[s] = float(d.cells[c].phi), r.psi[s] = solid.at(d.coord(c)), r.q[s] = float(q[c]);
        for (int j = 0; j < 6; ++j) {
            const int n = d.neighbor(c, j);
            r.fluidNeighbors[s] += n && d.cells[n].fluid;
        }
    }
    r.open = float(d.cells[i].open[a]);
    if (cell[0] && r.fluid[0] != r.fluid[1])
        r.theta = float(r.fluid[1] ? pressure::theta(d.cells[i].phi, d.cells[cell[0]].phi, minimumTheta)
                                   : pressure::theta(d.cells[cell[0]].phi, d.cells[i].phi, minimumTheta));
    r.before = before[i] - (a == 1 && r.open > 0 ? float(gravityStep) : 0.0f);
    r.after  = after ? after[i] : float(d.cells[i].u[a]);
    const int            f = r.fluid[1] == 1 || !cell[0] ? i : cell[0];
    const nanovdb::Coord fc = d.coord(f);
    const float*         old[3] = {before0, before1, before2};
    for (int k = 0; k < 3; ++k) r.center[k] = fc[k];
    r.nbFluid[0] = d.cells[f].fluid, r.nbPhi[0] = float(d.cells[f].phi), r.nbPsi[0] = solid.at(fc), r.nbQ[0] = float(q[f]);
    for (int j = 0; j < 6; ++j) {
        const int n = d.neighbor(f, j), owner = (j & 1) ? n : f, axis = j / 2; // the face is the lower face of owner
        const float sign = (j & 1) ? 1.0f : -1.0f;
        r.nbFluid[j + 1] = n ? d.cells[n].fluid : -1;
        if (!n) continue;
        nanovdb::Coord nc = fc;
        nc[axis] += (j & 1) ? 1 : -1;
        r.nbPhi[j + 1] = float(d.cells[n].phi), r.nbPsi[j + 1] = solid.at(nc), r.nbQ[j + 1] = float(q[n]);
        r.nbOpen[j + 1]   = float(d.cells[owner].open[axis]);
        r.nbBefore[j + 1] = sign * (old[axis][owner] - (axis == 1 && r.nbOpen[j + 1] > 0 ? float(gravityStep) : 0.0f));
        r.nbAfter[j + 1]  = sign * float(d.cells[owner].u[axis]);
    }
    *out = r;
}

// Sidecars -> cells, with gravity added to the open vertical faces.
__global__ void loadVelocity(pressure::Domain d, const float* u0, const float* u1, const float* u2, Real gravityStep)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (!i || i >= d.count) return;
    fluid::Cell& c = d.cells[i];
    c.u[0] = u0[i], c.u[1] = u1[i], c.u[2] = u2[i];
    if (c.open[1] > 0) c.u[1] -= gravityStep;
}

// Extrapolated cells -> sidecars. A closed face between a tank cell and a wall cell carries no flow. A component the
// extrapolation did not reach is set to zero; invalid counts those on faces of tank cells (the rest lie in the
// corners behind two walls, where nothing reads them).
__global__ void storeVelocity(pressure::Domain d, float* u0, float* u1, float* u2, int* invalid, pressure::Solid solid)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d.count) return;
    float* u[3] = {u0, u1, u2};
    if (!i) { u0[0] = u1[0] = u2[0] = 0.0f; return; }
    const fluid::Cell&   c   = d.cells[i];
    const nanovdb::Coord ijk = d.coord(i);
    for (int a = 0; a < 3; ++a) {
        nanovdb::Coord lower = ijk;
        lower[a] -= 1;
        float value = float(c.transport[a]);
        const bool in = !solid.solid(ijk), lowerIn = !solid.solid(lower);
        if (!c.valid[a]) { value = 0.0f; if (in || lowerIn) atomicAdd(invalid, 1); }
        // A closed wall face lets nothing into the wall. A released face keeps its flow. A face with air in front of
        // it keeps the water's velocity continued across the air only when that points away from the wall: this is
        // what moves a surface that has left the wall and sits between it and the first cell center; a velocity
        // into the wall would carry a thin layer of water through it.
        if (in != lowerIn && c.open[a] == 0) {
            const bool intoWall = in ? value < 0.0f : value > 0.0f; // the face is the lower face of i
            if (intoWall || d.cells[in ? i : d.neighbor(i, 2 * a)].fluid) value = 0.0f;
        }
        u[a][i] = value;
    }
}

// Samples the device's free memory twice per millisecond. peak() = the most memory in use since the last call of
// reset(), over what was in use when start() was called (so a context created before start() does not count).
struct MemoryWatch
{
    std::atomic<bool>   stop{false};
    std::atomic<size_t> lowestFree{~size_t(0)};
    size_t              freeAtStart = 0;
    std::thread         thread;
    static size_t freeNow()
    {
        size_t free = 0, total = 0;
        cudaCheck(cudaMemGetInfo(&free, &total));
        return free;
    }
    void start()
    {
        freeAtStart = freeNow(), lowestFree = freeAtStart;
        thread = std::thread([this] {
            while (!stop) {
                const size_t free = freeNow();
                size_t       seen = lowestFree;
                while (free < seen && !lowestFree.compare_exchange_weak(seen, free)) {}
                std::this_thread::sleep_for(std::chrono::microseconds(500));
            }
        });
    }
    void   reset() { lowestFree = freeNow(); }
    double peak() const { return (double(freeAtStart) - double(std::min(lowestFree.load(), freeNow()))) / 1048576.0; } // MiB
    ~MemoryWatch()
    {
        stop = true;
        if (thread.joinable()) thread.join();
    }
};
static MemoryWatch* gMemory = nullptr;

// What the step was doing and when, for the message of an exception (the simulation is gone by the time it is caught).
static const char* gStage = "start";
static double      gTime  = 0;

// An obstacle as pressure::Solid reads it: MeshToSDF's grid and values with the invert masks, on the device (state) and
// on the host, and how a tank cell maps to a voxel of the grid.
struct Obstacle
{
    std::optional<StateT>         state;      // the device's copy: grid, values (phi) and invert masks
    nanovdb::HostBuffer           grid;       // the host's copy
    std::vector<float>            values;
    std::vector<nanovdb::Mask<3>> leafMasks;
    std::vector<nanovdb::Mask<4>> lowerMasks;
    std::vector<nanovdb::Mask<5>> upperMasks;
    std::vector<nanovdb::Coord>   rootOrigins;
    double                        voxel = 1;
    float                         band = 0, offset = 0;
    int                           shift[3] = {0, 0, 0}, axes[3] = {0, 1, 2};
};

struct Simulation
{
    Tank   tank;
    float  background    = 3.0f;
    double gravity       = 0;    // voxels / s^2
    double cfl           = 0.5;
    Real   minimumTheta  = Real(1e-3);
    int    halo          = 2;    // layers of air added around the filled band
    int    sweeps        = 12;   // extrapolation sweeps
    bool   ruleAfterTrack = false; // experiment: apply the wall rule again after normalize and prune
    bool   separate       = true;  // walls let go of water they would have to pull on (pressure.cuh, released faces)
    int    maxReleaseIterations = 8;
    int    releaseAxes    = 0;     // 0: only walls above the water let go; otherwise bit a = walls normal to axis a (experiments)
    int    normCount      = 3;     // pseudo-time steps of the tracker's normalize per step (its default)
    int    bandGrow       = 3;     // dilations before the wall rule: the wall needs 3 layers behind any water it touches
    advect::Scheme phiScheme = advect::MacCormackClamp, velocityScheme = advect::MacCormackClamp;
    float  phiCorrectionLimit = 3.0e38f; // experiment (PHI_LIMIT): largest MacCormack correction of phi kept, in voxels
    // Volume control (Kim et al. 2007, Simulation of Bubbles in Foam With The Volume Control Method, section 4.1), with
    // the whole water as one region: each step the pressure projection leaves the divergence c in every fluid cell,
    //   x = (V - target) / target,  c dt = (-(2.3 / nP) x - (2.3 / (2 zeta nP))^2 sum of x over the steps so far) / (x + 1),
    // which is their eq. 3 with the gains of eq. 5 and 6 when dt is constant. 0: off, 1: first term only, 2: both.
    double wallFriction   = 0;   // FRICTION: friction coefficient Cf of the solids (fast.cuh wallFriction); 0 = free slip
    int    volumeControl  = 0;
    double controlSteps   = 25;  // nP: steps in which the first term alone removes 90% of an error
    double controlDamping = 2;   // zeta
    double targetVolume   = 0;   // 0: the volume measured at the first step
    double volumeErrorSum = 0;   // sum of x over the steps so far
    double volumeNow = 0, volumeError = 0, controlDivergence = 0; // of the last step
    // Particle level set (pls.cuh; PLS=<particles per cell>): marker particles move with the flow and rebuild the level
    // set where it lost or gained water against them. Not in the checkpoints: a continued run seeds them again.
    std::optional<pls::Particles> markers;
    // SPRAY=1: escaped negative markers become drops that carry the volume the level set loses (spray.cuh).
    std::optional<spray::Drops> drops;
    double waterTarget    = 0; // the volume of the level set at the first step: level set + drops are held to it
    double levelSetVolume = 0, sprayMissing = 0, sprayEmitted = 0, sprayDeposited = 0;
    size_t sprayNew       = 0; // drops made in the last step
    long   airRemoved     = 0; // escaped air markers removed so far
    int    reseedEvery    = 10;  // PLS_RESEED: steps between reseedings
    long   markerEscaped[2] = {0, 0}, markerSignChanges = 0, markerLost = 0; // sums over the run
    double gainedByMarkers[2] = {0, 0}; // DIAG: volume added by the correction after the advection and after normalize
    bool   diagnose       = false; // DIAG=1: measure the volume after each part of the level-set update
    double gained[6]      = {0, 0, 0, 0, 0, 0}; // volume added by: advection, dilate, wall rule, normalize, prune, end-of-step rule
    double implied[4]     = {0, 0, 0, 0}; // DIAG: volume change the velocity's divergence implies (divergenceBudget), by kind of cell
    double waterIn[4]     = {0, 0, 0, 0}; // DIAG: the water volume in each kind of cell, at the last step
    double time          = 0;
    float  firstOrderNearSolid = 0; // U_SL_NEAR_SOLID: velocity faces with a solid within this many voxels keep the semi-Lagrangian value
    bool   keepInterface = false; // NORM_KEEP=1: normalize leaves the voxels next to the surface as they were (restoreInterface)
    double speedLimit    = 0;    // VMAX=<m/s>: after the projection every velocity component is held to this size (0: no limit)
    int    trace         = 0;    // TRACE=N: every N steps print the largest speed after each part of the step, and where
    double traceSpike    = 1.5;  // TRACE_SPIKE: also print when the projection raises the largest speed by this factor
    int    spikeLines    = 0;
    int    pipeline       = 1;       // PIPE: 1 = the Cell kernels of the APIC example (steps 5 and 6 below), 2 = fast.cuh
    float  mgScale        = 1.8f;    // MG_SCALE: fast.cuh's multigrid, factor on every coarse correction
    int    mgSweeps       = 2;       // MG_SWEEPS: Jacobi sweeps before and after each coarse correction
    int    pressureSolver = 1;       // PSOLVER: 1 = pressure::Cells::solve (the APIC example's solver), 2 = pressure2::Solver
    double pressureTolerance = 1e-6; // PTOL: solver 2 stops when |b - A q| <= this times |b|
    bool   pressureCheck = false;    // PCHECK=1: solve every system with both solvers and print how the solutions differ

    // The solids (pressure::Solid), up to psiPad cells around the tank: the obstacle's field on the host and on the
    // device; the tank's own field is computed.
    static constexpr int psiPad = 8;
    Obstacle             obstacle;
    pressure::Solid      solid{}, hostSolid{};
    float hostPsiAt(const nanovdb::Coord& c) const { return hostSolid.at(c); }
    bool isFree(const nanovdb::Coord& c) const { return this->hostPsiAt(c) >= 0.0f; } // in the tank and outside every obstacle

    std::optional<StateT>       band;
    std::optional<HandleT>      domain;
    std::vector<BufferT<float>> u;         // 3 sidecars of the domain grid
    std::optional<BufferT<float>> domainPhi;
    std::optional<BufferT<float>> released;  // per cell of the domain grid: bit a = its lower face along a is a released wall face
    uint32_t                    domainLeaves = 0;
    uint64_t                    domainCount  = 0;

    struct Stats
    {
        double   dt = 0, maxSpeed = 0, divergence = 0, residual = 0, ms = 0;
        int      iterations = 0, missing = 0, invalid = 0, releaseIterations = 0, releasedFaces = 0;
        int      maxSolves = 0; // largest number of pressure solves in one step, over the run
        // Wall-clock time over the run, ms: advection, band update, domain rebuild, pressure (all solves), extrapolation,
        // end-of-step wall rule; and the number of steps and of pressure solves.
        double   partMs[6] = {0, 0, 0, 0, 0, 0};
        double   partPeak[6] = {0, 0, 0, 0, 0, 0}; // most device memory in use during each of those parts, MiB (BENCH)
        double   setupMs = 0, solveMs = 0;          // of the pressure time: building the multigrid levels, and iterating
        long     iterationSum = 0;                  // pressure iterations over all solves
        long     steps = 0, solves = 0;
        long     limited = 0; // velocity components VMAX cut in the last step
        long     forced = 0; // voxels the safety net of the wall rule moved to the other side of zero, over the run
        uint64_t fluidCells = 0, bandVoxels = 0, domainVoxels = 0;
    } stats;

    const GridT* bandGrid() const { return band->grid.deviceGrid<BuildT>(); }
    const GridT* domainGrid() const { return domain->deviceGrid<BuildT>(); }

    // Water = the box [lo, hi] (cell-center coordinates); a face on or beyond a wall is moved behind it.
    // given: none, or an obstacle (loadObstacle).
    Simulation(const Tank& t, const double waterLo[3], const double waterHi[3], Obstacle&& given = {}) : tank(t), obstacle(std::move(given))
    {
        for (int a = 0; a < 3; ++a) hostSolid.tank[a] = tank.n[a], hostSolid.shift[a] = obstacle.shift[a], hostSolid.axes[a] = obstacle.axes[a];
        hostSolid.pad = psiPad, hostSolid.voxel = obstacle.voxel, hostSolid.band = obstacle.band, hostSolid.offset = obstacle.offset;
        solid = hostSolid;
        if (obstacle.state) {
            hostSolid.grid = reinterpret_cast<const GridT*>(obstacle.grid.data()), hostSolid.values = obstacle.values.data();
            hostSolid.leafMasks = obstacle.leafMasks.data(), hostSolid.lowerMasks = obstacle.lowerMasks.data();
            hostSolid.upperMasks = obstacle.upperMasks.data(), hostSolid.rootOrigins = obstacle.rootOrigins.data();
            const StateT& s = *obstacle.state;
            solid.grid = s.grid.deviceGrid<BuildT>(), solid.values = s.phi.data();
            solid.leafMasks = s.leafMasks.data(), solid.lowerMasks = s.lowerMasks.data(), solid.upperMasks = s.upperMasks.data();
            solid.rootOrigins = s.rootOrigins.data();
            hostSolid.rootOriginCount = solid.rootOriginCount = obstacle.rootOrigins.size();
        }

        double lo[3], hi[3];
        for (int a = 0; a < 3; ++a) {
            lo[a] = waterLo[a] <= -0.5 ? -0.5 - tank.extension : waterLo[a];
            hi[a] = waterHi[a] >= tank.n[a] - 0.5 ? tank.n[a] - 0.5 + tank.extension : waterHi[a];
        }
        nanovdb::tools::build::FloatGrid builder(background);
        {
            auto      acc = builder.getAccessor();
            const int pad = int(background) + 2;
            for (int i = int(std::floor(lo[0])) - pad; i <= int(std::ceil(hi[0])) + pad; ++i)
                for (int j = int(std::floor(lo[1])) - pad; j <= int(std::ceil(hi[1])) + pad; ++j)
                    for (int k = int(std::floor(lo[2])) - pad; k <= int(std::ceil(hi[2])) + pad; ++k) {
                        const double p[3] = {double(i), double(j), double(k)};
                        const double d    = boxSdf(p, lo, hi);
                        if (std::fabs(d) < background) acc.setValue(nanovdb::Coord(i, j, k), float(d));
                    }
        }
        auto        handle = nanovdb::tools::createNanoGrid<nanovdb::tools::build::FloatGrid, BuildT>(builder, 0u, false, false);
        const auto* host   = handle.grid<BuildT>();
        std::vector<float> values(host->tree().activeVoxelCount() + 1, background);
        for (uint32_t l = 0; l < host->tree().nodeCount(0); ++l) {
            const auto& leaf = host->tree().getFirstLeaf()[l];
            for (uint32_t m = 0; m < 512; ++m)
                if (leaf.isActive(m)) {
                    const auto   c    = leaf.offsetToGlobalCoord(m);
                    const double p[3] = {double(c[0]), double(c[1]), double(c[2])};
                    values[leaf.getValue(m)] = float(boxSdf(p, lo, hi));
                }
        }
        BufferT<std::byte> gridBuffer = makeBuffer<std::byte>(handle.buffer().size());
        cudaCheck(cudaMemcpy(gridBuffer.data(), handle.buffer().data(), handle.buffer().size(), cudaMemcpyHostToDevice));
        BufferT<float> phi = makeBuffer<float>(values.size());
        cudaCheck(cudaMemcpy(phi.data(), values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice));
        TrackerT tracker(HandleT(std::move(gridBuffer)), std::move(phi), background);
        band.emplace(tracker.release());
        this->rebuildDomain(nullptr, 0, nullptr, nullptr);
    }

    // Domain grid = the band with its interior filled, plus the halo. phi and the velocity are moved onto it; the
    // velocity of a voxel the old domain did not have is zero until the extrapolation sets it.
    void rebuildDomain(const GridT* oldDomain, uint32_t oldLeaves, const std::vector<BufferT<float>>* oldU, const float* oldReleased)
    {
        using nanovdb::util::cuda::DeviceGridTraits;
        using nanovdb::util::cuda::lambdaKernel;
        using nanovdb::util::cuda::operatorKernel;
        using InjectOp = nanovdb::util::cuda::InjectGridDataFunctor<BuildT, float>;

        fill::FillInterior<BuildT, ResourceT> filler(this->bandGrid(), band->leafMasks.data(), band->lowerMasks.data(),
                                                     band->upperMasks.data(), band->rootOrigins.size());
        HandleT        filled    = filler.build();
        BufferT<float> filledPhi = fill::fillValues<BuildT, float, ResourceT>(this->bandGrid(), band->phi.data(),
                                                                              filled.deviceGrid<BuildT>(), background);
        const uint32_t filledLeaves = DeviceGridTraits<BuildT>::getTreeData(filled.deviceGrid<BuildT>(), cudaStream_t(0)).mNodeCount[0];

        std::optional<HandleT> grown;
        for (int layer = 0; layer < halo; ++layer) {
            nanovdb::tools::cuda::DilateGrid<BuildT, ResourceT> dilate(grown ? grown->deviceGrid<BuildT>() : filled.deviceGrid<BuildT>());
            HandleT next = dilate.build();
            grown.emplace(std::move(next));
        }
        if (!grown) throw std::runtime_error("halo must be at least 1");
        const GridT* grid = grown->deviceGrid<BuildT>();
        const auto   tree = DeviceGridTraits<BuildT>::getTreeData(grid, cudaStream_t(0));
        const uint64_t count = tree.mVoxelCount + 1;

        auto constant = [&](float value) {
            BufferT<float> b   = makeBuffer<float>(count);
            float*         out = b.data();
            lambdaKernel<<<nanovdb::util::cuda::blocksPerGrid(count, 256), 256>>>(count, [=] __device__(size_t i) { out[i] = value; });
            cudaCheckError();
            return b;
        };
        BufferT<float> phi = constant(background);
        operatorKernel<InjectOp><<<filledLeaves, InjectOp::MaxThreadsPerBlock>>>(filled.deviceGrid<BuildT>(), grid, filledPhi.data(), phi.data());
        cudaCheckError();
        std::vector<BufferT<float>> velocity;
        for (int a = 0; a < 3; ++a) {
            velocity.push_back(constant(0.0f));
            if (oldDomain && oldLeaves) {
                operatorKernel<InjectOp><<<oldLeaves, InjectOp::MaxThreadsPerBlock>>>(oldDomain, grid, (*oldU)[a].data(), velocity[a].data());
                cudaCheckError();
            }
        }
        BufferT<float> releasedNow = constant(0.0f);
        if (oldDomain && oldLeaves && oldReleased) {
            operatorKernel<InjectOp><<<oldLeaves, InjectOp::MaxThreadsPerBlock>>>(oldDomain, grid, oldReleased, releasedNow.data());
            cudaCheckError();
        }
        cudaCheck(cudaDeviceSynchronize());
        domain.emplace(std::move(*grown));
        domainPhi.emplace(std::move(phi));
        released.emplace(std::move(releasedNow));
        u = std::move(velocity);
        domainLeaves = tree.mNodeCount[0], domainCount = count;
    }

    // DIAG: the volume the water gains over a step of length dt if its surface moves with the stored velocity:
    // d(volume)/dt = integral of div(u) over the water, here the sum over the domain's free cells of
    // (water fraction of the cell, as in bandVolume) * (sum over its 6 faces of the outward face velocity).
    // out[0]: cells deeper than half a voxel (fraction 1); out[1]: cells with -0.5 < phi < 0 (pressure unknowns at
    // the surface); out[2]: cells with 0 <= phi < 0.5 (partly water, but air to the pressure solve: their
    // velocities come from the extrapolation) that have a face neighbor with phi < 0; out[3]: the same without such a
    // neighbor (water thinner than the grid resolves: no pressure unknown next to it). out[4..7]: the water volume
    // in each kind. Reads the domain's phi, which the end-of-step wall rule has not seen.
    void divergenceBudget(double dt, double out[8]) const
    {
        using nanovdb::util::cuda::lambdaKernel;
        auto sums = makeBuffer<double>(8);
        cudaCheck(cudaMemset(sums.data(), 0, 8 * sizeof(double)));
        const GridT*          grid = this->domainGrid();
        const float*          phi  = domainPhi->data();
        const float*          u0 = u[0].data(), *u1 = u[1].data(), *u2 = u[2].data();
        const pressure::Solid where = solid;
        double*               sum = sums.data();
        const size_t          voxels = size_t(domainLeaves) * 512;
        lambdaKernel<<<nanovdb::util::cuda::blocksPerGrid(voxels, 256), 256>>>(voxels, [=] __device__(size_t tid) {
            const auto&    leaf = grid->tree().getFirstLeaf()[tid >> 9];
            const uint32_t n    = uint32_t(tid & 511);
            if (!leaf.isActive(n)) return;
            const nanovdb::Coord ijk = leaf.offsetToGlobalCoord(n);
            if (where.solid(ijk)) return;
            const uint64_t i        = leaf.getValue(n);
            const float    fraction = fminf(1.0f, fmaxf(0.0f, 0.5f - phi[i]));
            if (fraction <= 0.0f) return;
            const auto   acc     = grid->getAccessor();
            const float* face[3] = {u0, u1, u2};
            float        divergence = 0.0f;
            bool         nextToFluid = false;
            for (int a = 0; a < 3; ++a) {
                nanovdb::Coord q = ijk;
                q[a] += 1;
                const uint64_t upper = acc.getValue(q);
                q[a] -= 2;
                const uint64_t lower = acc.getValue(q);
                divergence += face[a][upper] - face[a][i];
                nextToFluid = nextToFluid || (upper && phi[upper] < 0.0f) || (lower && phi[lower] < 0.0f);
            }
            const int kind = phi[i] <= -0.5f ? 0 : (phi[i] < 0.0f ? 1 : (nextToFluid ? 2 : 3));
            atomicAdd(sum + kind, double(fraction) * double(divergence));
            atomicAdd(sum + 4 + kind, double(fraction));
        });
        cudaCheckError();
        cudaCheck(cudaDeviceSynchronize());
        cudaCheck(cudaMemcpy(out, sums.data(), 8 * sizeof(double), cudaMemcpyDeviceToHost));
        for (int k = 0; k < 4; ++k) out[k] *= dt;
    }

    advect::InvertMasks masksOf(const StateT& s) const
    {
        return {s.leafMasks.data(), s.lowerMasks.data(), s.upperMasks.data(), s.rootOrigins.data(), s.rootOrigins.size()};
    }
    void reseedMarkers()
    {
        const GridT*   grid   = this->bandGrid();
        const uint32_t leaves = nanovdb::util::cuda::DeviceGridTraits<BuildT>::getTreeData(grid, cudaStream_t(0)).mNodeCount[0];
        markers->reseed(grid, leaves, band->phi.size(), band->phi.data(), this->masksOf(*band), solid, uint32_t(stats.steps) * 2654435761u + 12345u);
        markerLost += markers->lost;
    }
    void correctWithMarkers(StateT& s)
    {
        markers->correct(s.grid.deviceGrid<BuildT>(), s.phi.size(), s.phi.data(), this->masksOf(s), solid);
        markerEscaped[0] += markers->escapedPositive, markerEscaped[1] += markers->escapedNegative;
        markerSignChanges += markers->signChanges;
    }

    // NORM_KEEP: the tracker's normalize moves the surface a little every time (it has no sub-cell fix), which wears a
    // layer of one or two voxels away. This puts back what normalize changed next to the surface: every voxel that had
    // a face neighbor of the other sign before normalize gets its value from before. `before` is phi before normalize,
    // on the same grid.
    void restoreInterface(StateT& s, const BufferT<float>& before)
    {
        const GridT*   grid   = s.grid.deviceGrid<BuildT>();
        const uint32_t leaves = nanovdb::util::cuda::DeviceGridTraits<BuildT>::getTreeData(grid, cudaStream_t(0)).mNodeCount[0];
        const size_t   voxels = size_t(leaves) * 512;
        const float*   old    = before.data();
        float*         phi    = s.phi.data();
        using nanovdb::util::cuda::blocksPerGrid;
        using nanovdb::util::cuda::lambdaKernel;
        lambdaKernel<<<blocksPerGrid(voxels, 256), 256>>>(voxels, [=] __device__(size_t tid) {
            const auto&    leaf = grid->tree().getFirstLeaf()[tid >> 9];
            const uint32_t n    = uint32_t(tid & 511);
            if (!leaf.isActive(n)) return;
            const nanovdb::Coord ijk   = leaf.offsetToGlobalCoord(n);
            const uint64_t       index = leaf.getValue(n);
            const float          value = old[index];
            const auto           acc   = grid->getAccessor();
            for (int j = 0; j < 6; ++j) {
                nanovdb::Coord q = ijk;
                q[j >> 1] += (j & 1) ? 1 : -1;
                const uint64_t other = acc.getValue(q);
                if (other && (old[other] < 0.0f) != (value < 0.0f)) {
                    phi[index] = value;
                    return;
                }
            }
        });
        cudaCheckError();
    }
    BufferT<float> copyOf(const BufferT<float>& b)
    {
        BufferT<float> c = makeBuffer<float>(b.size());
        cudaCheck(cudaMemcpy(c.data(), b.data(), b.size() * sizeof(float), cudaMemcpyDeviceToDevice));
        return c;
    }

    // Step 3 of step(): the band after the advection of phi. Wall rule, dilate, wall rule, normalize, prune (track()
    // with the wall rule around the dilation). With markers: wall rule, dilate, correct, wall rule, normalize, correct,
    // radii, prune (Enright et al. 2002, end of section 2.3.4).
    // The values behind the walls are set by the rule, not by the advection: the advected ones need not close
    // the level set, and a tracker cannot be built on a level set that is open.
    void updateBand(BufferT<float>&& newPhi)
    {
        if (markers) {
            double v[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            if (diagnose) v[0] = this->bandVolume(*band, band->phi.data()), v[1] = this->bandVolume(*band, newPhi.data());
            band->phi = std::move(newPhi);
            this->applyWallRule(*band);
            gStage = "band update: tracker on the advected values";
            this->checkClosure("after advection", *band);
            TrackerT dilated(std::move(band->grid), std::move(band->phi), background);
            for (int k = 0; k < bandGrow; ++k) dilated.dilate();
            StateT s = dilated.release();
            if (diagnose) v[2] = this->bandVolume(s, s.phi.data());
            this->correctWithMarkers(s);
            if (diagnose) v[3] = this->bandVolume(s, s.phi.data());
            this->applyWallRule(s);
            if (diagnose) v[4] = this->bandVolume(s, s.phi.data());
            this->checkClosure("band update", s);
            gStage = "band update: tracker after the dilation, the markers and the wall rule";
            BufferT<float> beforeNormalize = keepInterface ? this->copyOf(s.phi) : makeBuffer<float>(0);
            TrackerT tracker(std::move(s.grid), std::move(s.phi), background);
            tracker.setNormCount(normCount);
            tracker.normalize();
            StateT n = tracker.release();
            if (keepInterface) this->restoreInterface(n, beforeNormalize);
            if (diagnose) v[5] = this->bandVolume(n, n.phi.data());
            this->correctWithMarkers(n);
            markers->adjustRadii(n.grid.deviceGrid<BuildT>(), n.phi.data(), this->masksOf(n));
            if (diagnose) v[6] = this->bandVolume(n, n.phi.data());
            this->checkClosure("after the markers", n, true);
            gStage = "band update: tracker after normalize and the markers";
            TrackerT pruner(std::move(n.grid), std::move(n.phi), background);
            pruner.prune(background);
            band.emplace(pruner.release());
            if (diagnose) {
                v[7] = this->bandVolume(*band, band->phi.data());
                gained[0] += v[1] - v[0], gained[1] += v[2] - v[1], gained[2] += v[4] - v[3], gained[3] += v[5] - v[4], gained[4] += v[7] - v[6];
                gainedByMarkers[0] += v[3] - v[2], gainedByMarkers[1] += v[6] - v[5];
            }
            return;
        }
        double v[6] = {0, 0, 0, 0, 0, 0};
        if (diagnose) v[0] = this->bandVolume(*band, band->phi.data()), v[1] = this->bandVolume(*band, newPhi.data());
        band->phi = std::move(newPhi);
        this->applyWallRule(*band);
        gStage = "band update: tracker on the advected values";
        this->checkClosure("after advection", *band);
        if (diagnose) {
            TrackerT dilated(std::move(band->grid), std::move(band->phi), background);
            for (int k = 0; k < bandGrow; ++k) dilated.dilate();
            StateT s = dilated.release();
            v[2] = this->bandVolume(s, s.phi.data());
            this->applyWallRule(s);
            v[3] = this->bandVolume(s, s.phi.data());
            BufferT<float> beforeNormalize = keepInterface ? this->copyOf(s.phi) : makeBuffer<float>(0);
            TrackerT normalizer(std::move(s.grid), std::move(s.phi), background);
            normalizer.setNormCount(normCount);
            normalizer.normalize();
            StateT n = normalizer.release();
            if (keepInterface) this->restoreInterface(n, beforeNormalize);
            v[4] = this->bandVolume(n, n.phi.data());
            TrackerT pruner(std::move(n.grid), std::move(n.phi), background);
            pruner.prune(background);
            band.emplace(pruner.release());
            v[5] = this->bandVolume(*band, band->phi.data());
            for (int k = 0; k < 5; ++k) gained[k] += v[k + 1] - v[k];
        } else {
            TrackerT dilated(std::move(band->grid), std::move(band->phi), background);
            for (int k = 0; k < bandGrow; ++k) dilated.dilate();
            StateT s = dilated.release();
            this->applyWallRule(s);
            this->checkClosure("band update", s);
            gStage = "band update: tracker after the dilation and the wall rule";
            BufferT<float> beforeNormalize = keepInterface ? this->copyOf(s.phi) : makeBuffer<float>(0);
            TrackerT tracker(std::move(s.grid), std::move(s.phi), background);
            tracker.setNormCount(normCount);
            tracker.normalize();
            if (keepInterface) {
                StateT n = tracker.release();
                this->restoreInterface(n, beforeNormalize);
                TrackerT pruner(std::move(n.grid), std::move(n.phi), background);
                pruner.prune(background);
                band.emplace(pruner.release());
            } else {
                tracker.prune(background);
                band.emplace(tracker.release());
            }
            if (ruleAfterTrack) {
                this->applyWallRule(*band);
                TrackerT again(std::move(band->grid), std::move(band->phi), background); // masks for the changed values
                band.emplace(again.release());
            }
        }
    }

    // One step of the band alone under a given motion (no pressure, no domain): the advection of phi and of the markers
    // with `displacement` (PosT(const PosT&): displacement over the step, in voxels), then updateBand().
    template <typename DisplacementT>
    void transportBand(DisplacementT displacement)
    {
        if (markers && stats.steps % reseedEvery == 0) this->reseedMarkers();
        const uint32_t bandLeaves = nanovdb::util::cuda::DeviceGridTraits<BuildT>::getTreeData(this->bandGrid(), cudaStream_t(0)).mNodeCount[0];
        const size_t   bandCount  = band->phi.size();
        BufferT<float> scratch    = makeBuffer<float>(3 * bandCount);
        BufferT<float> newPhi     = makeBuffer<float>(bandCount);
        advect::advect(this->bandGrid(), bandLeaves, bandCount, advect::SignedChannel{this->masksOf(*band)}, band->phi.data(), newPhi.data(),
                       advect::PosT(0), displacement, phiScheme, scratch.data(), cudaStream_t(0), phiCorrectionLimit);
        if (markers) markers->advect(displacement);
        cudaCheck(cudaDeviceSynchronize());
        this->updateBand(std::move(newPhi));
        stats.steps += 1;
    }

    // Sum over the cells around the box [lo, hi] of |water fraction of the band - water fraction of that box|, both
    // counted as in bandVolume: how far the band's water is from the box.
    // Check of fast::wallFriction against its formula: every x-component is set to speed, the other components to
    // zero, and one friction step is applied. A face of two free cells with w walls beside it must come out as
    // speed / (1 + w * frictionStep * speed / 2); every other face must keep its value. Returns the largest error and
    // counts the faces by their number of walls.
    double frictionCheck(float speed, float frictionStep, long walls[5])
    {
        std::vector<float> host(domainCount, speed);
        host[0] = 0;
        cudaCheck(cudaMemcpy(u[0].data(), host.data(), domainCount * sizeof(float), cudaMemcpyHostToDevice));
        for (int a = 1; a < 3; ++a) cudaCheck(cudaMemset(u[a].data(), 0, domainCount * sizeof(float)));
        fast::Pipeline pipe(this->domainGrid(), domainLeaves, domainCount, solid);
        float* const   uOut[3] = {u[0].data(), u[1].data(), u[2].data()};
        pipe.friction(uOut, frictionStep);
        const Snapshot snap = this->snapshot();
        double worst = 0;
        for (int w = 0; w < 5; ++w) walls[w] = 0;
        for (size_t k = 1; k < snap.coord.size(); ++k) {
            const nanovdb::Coord c = snap.coord[k], lower = c - nanovdb::Coord(1, 0, 0);
            int w = 0;
            if (this->isFree(c) && this->isFree(lower))
                for (int b = 1; b < 3; ++b)
                    for (int side = -1; side <= 1; side += 2) {
                        nanovdb::Coord p = c, q = lower;
                        p[b] += side, q[b] += side;
                        w += !this->isFree(p) && !this->isFree(q);
                    }
            ++walls[w];
            const double expected = double(speed) / (1.0 + w * double(frictionStep) * double(speed) * 0.5);
            worst = std::max(worst, std::fabs(double(snap.u[0][k]) - expected));
            for (int a = 1; a < 3; ++a) worst = std::max(worst, std::fabs(double(snap.u[a][k])));
        }
        return worst;
    }

    double differenceFromBox(const double lo[3], const double hi[3]) const
    {
        using nanovdb::tools::cuda::levelset::detail::isInverted;
        const StateT& s = *band;
        auto bytes = nanovdb::HostBuffer::create(s.grid.buffer().size());
        cudaCheck(cudaMemcpy(bytes.data(), s.grid.buffer().data(), bytes.size(), cudaMemcpyDeviceToHost));
        const auto* grid = reinterpret_cast<const GridT*>(bytes.data());
        const std::vector<float> phi = hostCopy(s.phi);
        const auto leaf = hostCopy(s.leafMasks);
        const auto lower = hostCopy(s.lowerMasks);
        const auto upper = hostCopy(s.upperMasks);
        const auto roots = hostCopy(s.rootOrigins);
        const auto acc = grid->getAccessor();
        double difference = 0;
        for (int i = 0; i < tank.n[0]; ++i)
            for (int j = 0; j < tank.n[1]; ++j)
                for (int k = 0; k < tank.n[2]; ++k) {
                    const nanovdb::Coord c(i, j, k);
                    const double   p[3] = {double(i), double(j), double(k)};
                    const double   exact = std::min(1.0, std::max(0.0, 0.5 - boxSdf(p, lo, hi)));
                    const uint64_t idx = acc.getValue(c);
                    const double   here = idx ? std::min(1.0, std::max(0.0, 0.5 - double(phi[idx])))
                                              : (isInverted(*grid, c, leaf.data(), lower.data(), upper.data(), roots.data(), roots.size()) ? 1.0 : 0.0);
                    difference += std::fabs(here - exact);
                }
        return difference;
    }

    double maxSpeed() const
    {
        float m = 0;
        for (int a = 0; a < 3; ++a)
            m = std::max(m, thrust::transform_reduce(thrust::device, u[a].data(), u[a].data() + domainCount, AbsoluteValue(), 0.0f,
                                                     thrust::maximum<float>()));
        return m;
    }

    void step(double dtLimit)
    {
        using nanovdb::util::cuda::DeviceGridTraits;
        const auto t0 = std::chrono::steady_clock::now();
        if (gMemory) gMemory->reset();

        // 1. Time step. sqrt(gravity) bounds the speed gravity adds over a voxel (Bridson, Fluid Simulation, ch. 3).
        stats.maxSpeed = this->maxSpeed();
        const double dt = std::min(dtLimit, cfl / (stats.maxSpeed + std::sqrt(gravity)));
        stats.dt = dt;
        if (markers && stats.steps % reseedEvery == 0) this->reseedMarkers();
        const advect::StaggeredVelocity carried{this->domainGrid(), {u[0].data(), u[1].data(), u[2].data()}, advect::RealT(dt)};
        if (diagnose) {
            double budget[8];
            this->divergenceBudget(dt, budget);
            for (int k = 0; k < 4; ++k) implied[k] += budget[k], waterIn[k] = budget[4 + k];
        }

        // 2. Advection of phi (on the band) and of the velocity (on the domain).
        const uint32_t bandLeaves = DeviceGridTraits<BuildT>::getTreeData(this->bandGrid(), cudaStream_t(0)).mNodeCount[0];
        const size_t   bandCount  = band->phi.size();
        BufferT<float> scratch    = makeBuffer<float>(3 * std::max<size_t>(bandCount, domainCount));
        BufferT<float> newPhi     = makeBuffer<float>(bandCount);
        const advect::SignedChannel signedChannel{{band->leafMasks.data(), band->lowerMasks.data(), band->upperMasks.data(),
                                                   band->rootOrigins.data(), band->rootOrigins.size()}};
        advect::advect(this->bandGrid(), bandLeaves, bandCount, signedChannel, band->phi.data(), newPhi.data(), advect::PosT(0),
                       carried, phiScheme, scratch.data(), cudaStream_t(0), phiCorrectionLimit);
        if (markers) markers->advect(carried);
        // Selle et al. 2008 (An Unconditionally Stable MacCormack Method), section 5: the error estimate is poor where a
        // characteristic can come from a solid, and they keep the semi-Lagrangian value there.
        std::vector<BufferT<unsigned char>> firstOrder;
        if (firstOrderNearSolid > 0) {
            for (int a = 0; a < 3; ++a) firstOrder.push_back(makeBuffer<unsigned char>(domainCount));
            const GridT*          grid = this->domainGrid();
            const pressure::Solid where = solid;
            const float           near  = firstOrderNearSolid;
            unsigned char *m0 = firstOrder[0].data(), *m1 = firstOrder[1].data(), *m2 = firstOrder[2].data();
            const size_t voxels = size_t(domainLeaves) * 512;
            nanovdb::util::cuda::lambdaKernel<<<nanovdb::util::cuda::blocksPerGrid(voxels, 256), 256>>>(voxels, [=] __device__(size_t tid) {
                const auto&    leaf = grid->tree().getFirstLeaf()[tid >> 9];
                const uint32_t n    = uint32_t(tid & 511);
                if (!leaf.isActive(n)) return;
                const nanovdb::Coord ijk = leaf.offsetToGlobalCoord(n);
                const uint64_t       i   = leaf.getValue(n);
                unsigned char*       m[3] = {m0, m1, m2};
                for (int a = 0; a < 3; ++a) {
                    nanovdb::Coord lower = ijk;
                    lower[a] -= 1;
                    m[a][i] = fminf(where.at(ijk), where.at(lower)) < near;
                }
            });
            cudaCheckError();
        }
        std::vector<BufferT<float>> newU;
        for (int a = 0; a < 3; ++a) {
            newU.push_back(makeBuffer<float>(domainCount));
            advect::PosT offset(0);
            offset[a] = advect::RealT(-0.5);
            advect::advect(this->domainGrid(), domainLeaves, domainCount, advect::Channel(), u[a].data(), newU[a].data(), offset,
                           carried, velocityScheme, scratch.data(), cudaStream_t(0), 3.0e38f, firstOrder.empty() ? nullptr : firstOrder[a].data());
        }
        cudaCheck(cudaDeviceSynchronize());
        if (pipeline == 2) {
            // Not read again: the advection's scratch, and the domain's old velocity and phi (rebuildDomain makes the new
            // ones from newU and the band).
            scratch = makeBuffer<float>(0);
            firstOrder.clear();
            u.clear();
            domainPhi.reset();
        }

        cudaCheck(cudaDeviceSynchronize());
        const auto t1 = std::chrono::steady_clock::now();
        if (gMemory) stats.partPeak[0] = std::max(stats.partPeak[0], gMemory->peak()), gMemory->reset();
        // 3. Band: wall rule, dilate, wall rule, normalize, prune. (track() with the wall rule around the dilation.)
        // The values behind the walls are set by the rule, not by the advection: the advected ones need not close
        // the level set, and a tracker cannot be built on a level set that is open.
        this->updateBand(std::move(newPhi));

        cudaCheck(cudaDeviceSynchronize());
        const auto t2 = std::chrono::steady_clock::now();
        if (gMemory) stats.partPeak[1] = std::max(stats.partPeak[1], gMemory->peak()), gMemory->reset();
        // 4. Domain.
        gStage = "domain rebuild and pressure";
        {
            HandleT        oldDomain   = std::move(*domain);
            BufferT<float> oldReleased = std::move(*released);
            this->rebuildDomain(oldDomain.deviceGrid<BuildT>(), domainLeaves, &newU, oldReleased.data());
        }
        // rebuildDomain read domainLeaves before overwriting it: it is the old grid's leaf count at that point.
        const double advectedMax = trace ? this->maxSpeed() : 0.0;

        cudaCheck(cudaDeviceSynchronize());
        const auto t3 = std::chrono::steady_clock::now();
        if (gMemory) stats.partPeak[2] = std::max(stats.partPeak[2], gMemory->peak()), gMemory->reset();
        // 5. Gravity, pressure, projection. With separation the set of released wall faces is part of the answer:
        // solve, decide every wall face again from the pressure, and repeat until no face changes.
        std::chrono::steady_clock::time_point t4;
        if (pipeline == 2) {
            // The advection's buffers are not needed any more.
            newU.clear(), firstOrder.clear();
            scratch = makeBuffer<float>(0);
            fast::Pipeline     pipe(this->domainGrid(), domainLeaves, domainCount, solid);
            const float* const uIn[3]  = {u[0].data(), u[1].data(), u[2].data()};
            float* const       uOut[3] = {u[0].data(), u[1].data(), u[2].data()};
            mg::Multigrid::Result solved{};
            stats.releasedFaces = 0;
            if (volumeControl) {
                volumeNow = pipe.volume(domainPhi->data());
                if (targetVolume == 0) targetVolume = volumeNow;
                volumeError = (volumeNow - targetVolume) / targetVolume;
                volumeErrorSum += volumeError;
                const double p = 2.3 / controlSteps, i = volumeControl == 2 ? p / (2 * controlDamping) : 0.0;
                controlDivergence = (-p * volumeError - i * i * volumeErrorSum) / ((volumeError + 1) * dt);
                pipe.divergence = controlDivergence;
            }
            std::vector<BufferT<float>> advectedU; // TRACE: the velocity before the projection
            for (int a = 0; trace && a < 3; ++a) {
                advectedU.push_back(makeBuffer<float>(domainCount));
                cudaCheck(cudaMemcpy(advectedU[a].data(), u[a].data(), domainCount * sizeof(float), cudaMemcpyDeviceToDevice));
            }
            // TRACE: the face with the largest |velocity| now, and what is around it.
            auto traceFace = [&](Peak& peak, VoxelReport& report, const float* source) {
                peak = thrust::transform_reduce(thrust::device, thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(int(domainCount)),
                                                SidecarPeak{{u[0].data(), u[1].data(), u[2].data()}}, Peak{0.0f, 0, 0}, PeakMax());
                auto out = fluid::allocate<VoxelReport>(1);
                describeVoxel<<<domainLeaves, 512>>>(this->domainGrid(), uint64_t(peak.cell), peak.axis, domainPhi->data(), solid, source,
                                                    released->data(), advectedU[0].data(), advectedU[1].data(), advectedU[2].data(), u[0].data(),
                                                    u[1].data(), u[2].data(), out.data());
                cudaCheck(cudaMemcpy(&report, out.data(), sizeof(VoxelReport), cudaMemcpyDeviceToHost));
            };
            if (wallFriction > 0) pipe.friction(uOut, float(wallFriction * dt));
            BufferT<float> deposit = makeBuffer<float>(0);
            if (drops) {
                levelSetVolume = pipe.volume(domainPhi->data());
                if (waterTarget == 0) waterTarget = levelSetVolume;
                deposit = makeBuffer<float>(domainCount);
                cudaCheck(cudaMemset(deposit.data(), 0, domainCount * sizeof(float)));
                const float* const velocity[3] = {u[0].data(), u[1].data(), u[2].data()};
                const double deposited = drops->step(dt, gravity, this->domainGrid(), domainPhi->data(), velocity, solid, deposit.data());
                sprayDeposited += deposited;
                // What the level set and the drops together are short of. What was deposited in this step is on its
                // way: the projection gives those cells that much divergence, and the next advection moves the surface.
                sprayMissing = waterTarget - levelSetVolume - drops->volume() - deposited;
                sprayEmitted += drops->emit(*markers, this->bandGrid(), band->phi.data(), this->masksOf(*band), this->domainGrid(), velocity,
                                            sprayMissing, sprayNew);
                airRemoved += long(markers->removeEscapedAir(this->bandGrid(), band->phi.data(), this->masksOf(*band)));
                pipe.source = deposit.data();
            }
            bool closeOnly = false; // see the two phases described below
            for (int iteration = 1;; ++iteration) {
                stats.releaseIterations = iteration;
                pipe.classify(domainPhi->data(), released->data());
                if (iteration == 1) pipe.buildSolver(mgScale, mgSweeps);
                solved = pipe.solve(domainPhi->data(), uIn, gravity * dt, double(minimumTheta), iteration > 1, pressureTolerance);
                stats.iterationSum += solved.iterations;
                if (!separate) break;
                if (iteration >= maxReleaseIterations) closeOnly = true;
                if (pipe.updateRelease(uIn, gravity * dt, released->data(), closeOnly, releaseAxes, stats.releasedFaces) == 0) break;
                if (iteration >= 4 * maxReleaseIterations) throw std::runtime_error("the released wall faces did not settle");
            }
            stats.maxSolves  = std::max(stats.maxSolves, stats.releaseIterations);
            stats.divergence = pipe.project(domainPhi->data(), uOut, gravity * dt, double(minimumTheta));
            if (speedLimit > 0) { // in voxels per second: gravity is 9.81 m/s^2 in voxels
                const float limit = float(speedLimit * gravity / 9.81);
                stats.limited = 0;
                for (int a = 0; a < 3; ++a) {
                    stats.limited += thrust::count_if(thrust::device, u[a].data(), u[a].data() + domainCount, Faster{limit});
                    thrust::transform(thrust::device, u[a].data(), u[a].data() + domainCount, u[a].data(), ClampSpeed{limit});
                }
            }
            stats.fluidCells = pipe.fluidCells;
            stats.iterations = solved.iterations, stats.missing = 0;
            stats.residual   = solved.norm > 0 ? solved.residual / solved.norm : 0.0;
            stats.setupMs += pipe.setupMs, stats.solveMs += pipe.solveMs;
            pipe.releaseSolver();
            Peak        tracePeak[2] = {{0.0f, 0, 0}, {0.0f, 0, 0}}; // after the projection, after the extrapolation
            VoxelReport traceReport[2] = {};
            if (trace) traceFace(tracePeak[0], traceReport[0], drops ? deposit.data() : nullptr);
            cudaCheck(cudaDeviceSynchronize());
            t4 = std::chrono::steady_clock::now();
            if (gMemory) stats.partPeak[3] = std::max(stats.partPeak[3], gMemory->peak()), gMemory->reset();
            stats.invalid = pipe.extrapolate(uOut, sweeps);
            cudaCheck(cudaDeviceSynchronize());
            if (trace) {
                traceFace(tracePeak[1], traceReport[1], drops ? deposit.data() : nullptr);
                if ((stats.steps + 1) % trace == 0) {
                    std::printf("trace step %ld t %.5f dt %.2e: largest speed at the start %.1f, advected %.1f, projected %.1f, stored %.1f\n",
                                stats.steps + 1, time, dt, stats.maxSpeed, advectedMax, tracePeak[0].value, tracePeak[1].value);
                    const char* names[2] = {"projected", "stored"};
                    for (int k = 0; k < 2; ++k) {
                        const VoxelReport& r = traceReport[k];
                        std::printf("      %s: face %d of cell (%d, %d, %d), gravity step %.2f; cell and neighbors -x +x -y +y -z +z: phi", names[k], r.face,
                                    r.coord[0], r.coord[1], r.coord[2], gravity * dt);
                        for (int j = 0; j < 7; ++j) std::printf(" %+.3f", r.phi[j]);
                        std::printf(", psi");
                        for (int j = 0; j < 7; ++j) std::printf(" %.2f", r.psi[j]);
                        std::printf(", source");
                        for (int j = 0; j < 7; ++j) std::printf(" %.3g", r.source[j]);
                        std::printf(", released");
                        for (int j = 0; j < 7; ++j) std::printf(" %g", r.released[j]);
                        std::printf(", faces before");
                        for (int j = 0; j < 6; ++j) std::printf(" %.1f", r.before[j]);
                        std::printf(", after");
                        for (int j = 0; j < 6; ++j) std::printf(" %.1f", r.after[j]);
                        std::printf("\n");
                    }
                }
            }
        } else {
        pressure::Cells cells(this->domainGrid(), domainLeaves, domainCount, domainPhi->data(), solid, tank.n[0], tank.n[1], tank.n[2]);
        pressure::Domain d = cells.d;
        const int blocks = fluid::blocks(d.count);
        auto b = fluid::allocate<Real>(d.count), q = fluid::allocate<Real>(d.count);
        auto counters = fluid::allocate<int>(2);
        pressure::Cells::Result solved{};
        std::optional<pressure2::Solver> fast; // built at the first solve of the step, reused by the others
        stats.releasedFaces = 0;
        // Two phases. First, every wall face is decided again after each solve, until no face changes. If that
        // has not settled after maxReleaseIterations solves, faces are only closed from then on: a released face in
        // front of which the water pushes toward the wall would be a hole in the wall. Closing only shrinks the set,
        // so this ends, with no released face that lets water into the wall.
        bool closeOnly = false;
        for (int iteration = 1;; ++iteration) {
            stats.releaseIterations = iteration;
            pressure::applyRelease<<<blocks, 256>>>(d, released->data(), solid);
            loadVelocity<<<blocks, 256>>>(d, u[0].data(), u[1].data(), u[2].data(), Real(gravity * dt));
            pressure::fluxRhs<<<blocks, 256>>>(d, b.data());
            if (pressureSolver == 2) {
                if (!fast) {
                    fast.emplace(d);
                    if (const char* v = std::getenv("MG_SCALE")) fast->correctionScale = float(std::atof(v));
                    if (const char* v = std::getenv("MG_SWEEPS")) fast->preSweeps = fast->postSweeps = std::atoi(v);
                }
                const auto f = fast->solve(b.data(), q.data(), minimumTheta, solid, cells.missing.data(), iteration > 1, pressureTolerance);
                solved = {f.iterations, f.levels, f.missing, f.residual, f.norm, iteration == 1 ? f.setupMs : 0.0, f.solveMs};
                if (pressureCheck) {
                    auto reference = fluid::allocate<Real>(d.count);
                    const auto old = cells.solve(b.data(), reference.data(), minimumTheta);
                    const double difference = fluid::reduceDevice(d.count, pressure2::SquaredDifference{q.data(), reference.data()}, 0.0, thrust::plus<double>());
                    const double size       = fluid::reduceDevice(d.count, pressure2::SquaredDifference{reference.data(), nullptr}, 0.0, thrust::plus<double>());
                    std::printf("PCHECK step %ld solve %d: %d fluid cells, %d levels | new: %d iterations (%d restarts), residual %.1e, %.1f ms (setup %.1f) | "
                                "old: %d iterations, residual %.1e, %.1f ms (setup %.1f) | solutions differ by %.2e (relative)\n", stats.steps + 1, iteration,
                                fast->fluidCells, f.levels, f.iterations, f.restarts, f.norm > 0 ? f.residual / f.norm : 0.0, f.solveMs, f.setupMs,
                                old.iterations, old.norm > 0 ? old.residual / old.norm : 0.0, old.solveMs, old.setupMs, size > 0 ? std::sqrt(difference / size) : 0.0);
                }
            } else
                solved = cells.solve(b.data(), q.data(), minimumTheta);
            stats.setupMs += solved.setupMs, stats.solveMs += solved.solveMs, stats.iterationSum += solved.iterations;
            if (!separate) break;
            if (iteration >= maxReleaseIterations) closeOnly = true;
            cudaCheck(cudaMemset(counters.data(), 0, 2 * sizeof(int)));
            pressure::updateRelease<<<blocks, 256>>>(d, q.data(), u[0].data(), u[1].data(), u[2].data(), Real(gravity * dt),
                                                    released->data(), counters.data(), counters.data() + 1, closeOnly, releaseAxes, solid);
            int host[2];
            cudaCheck(cudaMemcpy(host, counters.data(), 2 * sizeof(int), cudaMemcpyDeviceToHost));
            stats.releasedFaces = host[1];
            if (host[0] == 0) break;
            if (iteration >= 4 * maxReleaseIterations) throw std::runtime_error("the released wall faces did not settle");
        }
        stats.maxSolves = std::max(stats.maxSolves, stats.releaseIterations);
        pressure::project<<<blocks, 256>>>(d, q.data(), minimumTheta, solid);
        const auto flux = fluid::reduceDevice(d.count, fluid::FluxMeasure{d}, fluid::FluxSummary{}, fluid::FluxAdd{});
        stats.fluidCells = flux.count;
        stats.divergence = flux.count ? std::sqrt(flux.square / flux.count) : 0.0;
        stats.iterations = solved.iterations, stats.missing = solved.missing;
        stats.residual   = solved.norm > 0 ? solved.residual / solved.norm : 0.0;
        Peak       projectedPeak{0.0f, 0, 0}, storedPeak{0.0f, 0, 0};
        FaceReport projectedFace{}, storedFace{};
        auto       faceReport = fluid::allocate<FaceReport>(1);
        std::vector<BufferT<float>> beforeProjection;
        if (trace) {
            projectedPeak = thrust::transform_reduce(thrust::device, thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(d.count),
                                                     ProjectedPeak{d.cells}, Peak{0.0f, 0, 0}, PeakMax());
            describeFace<<<1, 1>>>(d, projectedPeak.cell, projectedPeak.axis, q.data(), u[projectedPeak.axis].data(), Real(gravity * dt),
                                   minimumTheta, solid, nullptr, faceReport.data(), u[0].data(), u[1].data(), u[2].data());
            cudaCheck(cudaMemcpy(&projectedFace, faceReport.data(), sizeof(FaceReport), cudaMemcpyDeviceToHost));
            for (int a = 0; a < 3; ++a) {
                beforeProjection.push_back(makeBuffer<float>(domainCount));
                cudaCheck(cudaMemcpy(beforeProjection[a].data(), u[a].data(), domainCount * sizeof(float), cudaMemcpyDeviceToDevice));
            }
        }

        cudaCheck(cudaDeviceSynchronize());
        t4 = std::chrono::steady_clock::now();
        if (gMemory) stats.partPeak[3] = std::max(stats.partPeak[3], gMemory->peak()), gMemory->reset();
        // 6. Extrapolation (the APIC example's kernels), then back to the sidecars.
        auto extrap = fluid::allocate<Real>(3 * size_t(d.count));
        auto extrapValid = fluid::allocate<unsigned char>(3 * size_t(d.count));
        auto invalid = fluid::allocate<int>(1);
        cudaCheck(cudaMemset(invalid.data(), 0, sizeof(int)));
        d.extrap = extrap.data(), d.extrapValid = extrapValid.data();
        fluid::prepareTransport<<<blocks, 256>>>(d);
        for (int k = 0; k < sweeps; ++k) {
            fluid::extrapolate<<<blocks, 256>>>(d);
            fluid::finishExtrapolate<<<blocks, 256>>>(d);
        }
        storeVelocity<<<blocks, 256>>>(d, u[0].data(), u[1].data(), u[2].data(), invalid.data(), solid);
        cudaCheck(cudaDeviceSynchronize());
        cudaCheck(cudaMemcpy(&stats.invalid, invalid.data(), sizeof(int), cudaMemcpyDeviceToHost));
        if (trace) {
            storedPeak = thrust::transform_reduce(thrust::device, thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(d.count),
                                                  SidecarPeak{{u[0].data(), u[1].data(), u[2].data()}}, Peak{0.0f, 0, 0}, PeakMax());
            describeFace<<<1, 1>>>(d, storedPeak.cell, storedPeak.axis, q.data(), beforeProjection[storedPeak.axis].data(), Real(gravity * dt),
                                   minimumTheta, solid, u[storedPeak.axis].data(), faceReport.data(), beforeProjection[0].data(),
                                   beforeProjection[1].data(), beforeProjection[2].data());
            cudaCheck(cudaMemcpy(&storedFace, faceReport.data(), sizeof(FaceReport), cudaMemcpyDeviceToHost));
            const bool spike = projectedPeak.value > traceSpike * (advectedMax + gravity * dt) && spikeLines < 400;
            if ((stats.steps + 1) % trace == 0 || spike) {
                spikeLines += spike;
                std::printf("trace step %ld t %.5f dt %.2e: largest speed at the start %.1f, advected %.1f, projected %.1f, stored %.1f%s\n",
                            stats.steps + 1, time, dt, stats.maxSpeed, advectedMax, projectedPeak.value, storedPeak.value, spike ? "  SPIKE" : "");
                const char*       names[2] = {"projected", "stored"};
                const Peak*       peaks[2] = {&projectedPeak, &storedPeak};
                const FaceReport* faces[2] = {&projectedFace, &storedFace};
                for (int k = 0; k < 2; ++k) {
                    const FaceReport& r = *faces[k];
                    std::printf("      %s: lower %c face of cell (%d, %d, %d): below/above fluid %d/%d, phi %+.3f/%+.3f, psi %.2f/%.2f, fluid neighbors %d/%d, "
                                "q %.3g/%.3g, open %.0f, theta %.4f, before %.1f, after %.1f\n", names[k], "xyz"[peaks[k]->axis], r.coord[0], r.coord[1],
                                r.coord[2], r.fluid[0], r.fluid[1], r.phi[0], r.phi[1], r.psi[0], r.psi[1], r.fluidNeighbors[0], r.fluidNeighbors[1],
                                r.q[0], r.q[1], r.open, r.theta, r.before, r.after);
                }
                if (spike) {
                    const FaceReport& r = projectedFace;
                    std::printf("      around the fluid cell (%d, %d, %d): phi %+.4f, psi %.2f, q %.4g\n", r.center[0], r.center[1], r.center[2],
                                r.nbPhi[0], r.nbPsi[0], r.nbQ[0]);
                    const char* names6[6] = {"-x", "+x", "-y", "+y", "-z", "+z"};
                    for (int j = 1; j <= 6; ++j)
                        std::printf("        %s: fluid %d, phi %+.4f, psi %.2f, q %.4g, open %.0f, outflow before %.1f, after %.1f\n", names6[j - 1],
                                    r.nbFluid[j], r.nbPhi[j], r.nbPsi[j], r.nbQ[j], r.nbOpen[j], r.nbBefore[j], r.nbAfter[j]);
                }
            }
        }

        }
        const auto t5 = std::chrono::steady_clock::now();
        if (gMemory) stats.partPeak[4] = std::max(stats.partPeak[4], gMemory->peak()), gMemory->reset();
        // 7. The band's values at the walls follow the release state just decided, before the next advection reads them.
        gStage = "end of step: tracker before the wall rule";
        const double beforeEnd = diagnose && separate ? this->bandVolume(*band, band->phi.data()) : 0.0;
        if (separate) {
            TrackerT dilated(std::move(band->grid), std::move(band->phi), background);
            for (int k = 0; k < bandGrow; ++k) dilated.dilate();
            StateT s = dilated.release();
            this->applyWallRule(s);
            this->checkClosure("end of step", s);
            this->checkClosure("end of step", s, true);
            gStage = "end of step: tracker after the wall rule";
            TrackerT again(std::move(s.grid), std::move(s.phi), background); // masks for the changed values
            again.prune(background);
            band.emplace(again.release());
            if (diagnose) gained[5] += this->bandVolume(*band, band->phi.data()) - beforeEnd;
        }

        time += dt;
        gTime = time;
        stats.bandVoxels = band->phi.size() - 1, stats.domainVoxels = domainCount - 1;
        cudaCheck(cudaDeviceSynchronize());
        const auto t6 = std::chrono::steady_clock::now();
        if (gMemory) stats.partPeak[5] = std::max(stats.partPeak[5], gMemory->peak()), gMemory->reset();
        const std::chrono::steady_clock::time_point marks[7] = {t0, t1, t2, t3, t4, t5, t6};
        for (int k = 0; k < 6; ++k) stats.partMs[k] += std::chrono::duration<double, std::milli>(marks[k + 1] - marks[k]).count();
        stats.steps += 1, stats.solves += stats.releaseIterations;
        stats.ms = std::chrono::duration<double, std::milli>(t6 - t0).count();
    }

    // The whole state in one file, to restart a run shortly before something of interest. extra: numbers of the driver.
    void save(const std::string& path, const double extra[3]) const
    {
        std::ofstream os(path + ".tmp", std::ios::binary);
        auto put = [&](const void* device, size_t bytes) {
            std::vector<char> host(bytes);
            if (bytes) cudaCheck(cudaMemcpy(host.data(), device, bytes, cudaMemcpyDeviceToHost));
            const uint64_t n = bytes;
            os.write(reinterpret_cast<const char*>(&n), sizeof(n));
            os.write(host.data(), std::streamsize(bytes));
        };
        os.write(reinterpret_cast<const char*>(&time), sizeof(time));
        os.write(reinterpret_cast<const char*>(extra), 3 * sizeof(double));
        put(band->grid.buffer().data(), band->grid.buffer().size());
        put(band->phi.data(), band->phi.size_bytes());
        put(domain->buffer().data(), domain->buffer().size());
        put(domainPhi->data(), domainPhi->size_bytes());
        put(released->data(), released->size_bytes());
        for (int a = 0; a < 3; ++a) put(u[a].data(), u[a].size_bytes());
        os.close();
        std::rename((path + ".tmp").c_str(), path.c_str());
    }
    template <typename T>
    static BufferT<T> readBuffer(std::istream& is)
    {
        uint64_t n = 0;
        is.read(reinterpret_cast<char*>(&n), sizeof(n));
        std::vector<char> host(n);
        is.read(host.data(), std::streamsize(n));
        if (!is) throw std::runtime_error("checkpoint: short file");
        BufferT<T> b = makeBuffer<T>(n / sizeof(T));
        if (n) cudaCheck(cudaMemcpy(static_cast<void*>(b.data()), host.data(), n, cudaMemcpyHostToDevice));
        return b;
    }
    void load(const std::string& path, double extra[3])
    {
        std::ifstream is(path, std::ios::binary);
        if (!is) throw std::runtime_error("checkpoint: cannot open " + path);
        is.read(reinterpret_cast<char*>(&time), sizeof(time));
        is.read(reinterpret_cast<char*>(extra), 3 * sizeof(double));
        BufferT<std::byte> bandBytes = readBuffer<std::byte>(is);
        BufferT<float>     bandPhi   = readBuffer<float>(is);
        TrackerT tracker(HandleT(std::move(bandBytes)), std::move(bandPhi), background);
        band.emplace(tracker.release());
        domain.emplace(HandleT(readBuffer<std::byte>(is)));
        domainPhi.emplace(readBuffer<float>(is));
        released.emplace(readBuffer<float>(is));
        for (int a = 0; a < 3; ++a) u[a] = readBuffer<float>(is);
        const auto tree = nanovdb::util::cuda::DeviceGridTraits<BuildT>::getTreeData(this->domainGrid(), cudaStream_t(0));
        domainLeaves = tree.mNodeCount[0], domainCount = tree.mVoxelCount + 1;
        gTime = time;
    }

    // CHECK=1: an exact test of what PropagateSign needs, before a tracker is built or prunes. On the host, flood
    // the inactive voxels from a corner of the bounding box; a negative active voxel next to a flooded voxel is where
    // the level set is open. pruned = true treats the voxels with |phi| >= background as already removed.
    void checkClosure(const char* where, const StateT& s, bool pruned = false) const
    {
        if (!std::getenv("CHECK")) return;
        auto bytes = nanovdb::HostBuffer::create(s.grid.buffer().size());
        cudaCheck(cudaMemcpy(bytes.data(), s.grid.buffer().data(), bytes.size(), cudaMemcpyDeviceToHost));
        const auto* grid = reinterpret_cast<const GridT*>(bytes.data());
        const std::vector<float> phi = hostCopy(s.phi);
        nanovdb::Coord lo(1 << 30), hi(-(1 << 30));
        for (uint32_t l = 0; l < grid->tree().nodeCount(0); ++l) {
            const nanovdb::Coord origin = grid->tree().getFirstLeaf()[l].origin();
            lo.minComponent(origin), hi.maxComponent(origin + nanovdb::Coord(7));
        }
        lo -= nanovdb::Coord(1), hi += nanovdb::Coord(1);
        const int nx = hi[0] - lo[0] + 1, ny = hi[1] - lo[1] + 1, nz = hi[2] - lo[2] + 1;
        // 0 = inactive, 1 = active and positive, 2 = active and negative, 3 = inactive and reached from outside
        std::vector<uint8_t> kind(size_t(nx) * ny * nz, 0);
        auto at = [&](int x, int y, int z) -> uint8_t& { return kind[(size_t(x - lo[0]) * ny + (y - lo[1])) * nz + (z - lo[2])]; };
        for (uint32_t l = 0; l < grid->tree().nodeCount(0); ++l) {
            const auto& leaf = grid->tree().getFirstLeaf()[l];
            for (uint32_t m = 0; m < 512; ++m) {
                if (!leaf.isActive(m)) continue;
                const float v = phi[leaf.getValue(m)];
                if (pruned && std::fabs(v) >= background) continue;
                const nanovdb::Coord c = leaf.offsetToGlobalCoord(m);
                at(c[0], c[1], c[2]) = v < 0 ? 2 : 1;
            }
        }
        std::vector<nanovdb::Coord> queue{lo};
        at(lo[0], lo[1], lo[2]) = 3;
        int found = 0;
        for (size_t head = 0; head < queue.size(); ++head) {
            const nanovdb::Coord c = queue[head];
            for (int j = 0; j < 6; ++j) {
                nanovdb::Coord q = c;
                q[j / 2] += (j & 1) ? 1 : -1;
                if (q[0] < lo[0] || q[1] < lo[1] || q[2] < lo[2] || q[0] > hi[0] || q[1] > hi[1] || q[2] > hi[2]) continue;
                uint8_t& k = at(q[0], q[1], q[2]);
                if (k == 0) { k = 3; queue.push_back(q); }
                else if (k == 2 && found++ < 8)
                    std::printf("  CHECK %s%s (after t %.5f): voxel (%d, %d, %d) phi %.3f, %s, is open toward (%d, %d, %d)\n", where,
                                pruned ? " once pruned" : "", time, q[0], q[1], q[2], phi[grid->getAccessor().getValue(q)],
                                this->isFree(q) ? "in free space" : "inside a solid", c[0], c[1], c[2]);
            }
        }
        if (found) std::printf("  CHECK %s%s: %d open faces\n", where, pruned ? " once pruned" : "", found);
    }

    // Volume of the water in the tank, in cells, from a band state: an active voxel counts by the fraction a flat
    // surface at distance phi would cut from it, an inactive one by its sign. phi overrides the state's values.
    double bandVolume(const StateT& s, const float* devicePhi) const
    {
        using nanovdb::tools::cuda::levelset::detail::isInverted;
        auto bytes = nanovdb::HostBuffer::create(s.grid.buffer().size());
        cudaCheck(cudaMemcpy(bytes.data(), s.grid.buffer().data(), bytes.size(), cudaMemcpyDeviceToHost));
        const auto* grid = reinterpret_cast<const GridT*>(bytes.data());
        std::vector<float> phi(s.phi.size());
        cudaCheck(cudaMemcpy(phi.data(), devicePhi, phi.size() * sizeof(float), cudaMemcpyDeviceToHost));
        const auto leaf  = hostCopy(s.leafMasks);
        const auto lower = hostCopy(s.lowerMasks);
        const auto upper = hostCopy(s.upperMasks);
        const auto roots = hostCopy(s.rootOrigins);
        const auto acc = grid->getAccessor();
        double volume = 0;
        for (int i = 0; i < tank.n[0]; ++i)
            for (int j = 0; j < tank.n[1]; ++j)
                for (int k = 0; k < tank.n[2]; ++k) {
                    const nanovdb::Coord c(i, j, k);
                    if (!this->isFree(c)) continue;
                    const uint64_t idx = acc.getValue(c);
                    if (idx) volume += std::min(1.0, std::max(0.0, 0.5 - double(phi[idx])));
                    else volume += isInverted(*grid, c, leaf.data(), lower.data(), upper.data(), roots.data(), roots.size()) ? 1.0 : 0.0;
                }
        return volume;
    }

    // The level set inside the solids (tank walls and obstacles), from the release state of the domain grid.
    //   A solid that holds the water (closed face): the water continues `extension` voxels into it. In a solid cell,
    //     phi = max(the free side's phi continued into the solid, depth below the solid's surface - extension).
    //   A solid that has let go (released face): the water ends at the face and the solid side is air. In a solid
    //     cell phi = max(the continued phi, depth); in the free cells below the face, phi = max(phi, -distance to
    //     the face).
    // Only faces with the solid above and the water below are ever released (pressure::updateRelease).
    void applyWallRule(StateT& s)
    {
        const GridT*   grid   = s.grid.deviceGrid<BuildT>();
        const uint32_t leaves = nanovdb::util::cuda::DeviceGridTraits<BuildT>::getTreeData(grid, cudaStream_t(0)).mNodeCount[0];
        const size_t   voxels = size_t(leaves) * 512;
        float*         phi    = s.phi.data();
        const float    bg     = background;
        const float    extension = float(tank.extension);
        const GridT*   cells  = this->domainGrid();
        const float*   flags  = released->data(); // slot 0 = 0: a cell the domain grid does not have is not released
        const pressure::Solid where = solid;
        const advect::SignedChannel channel{{s.leafMasks.data(), s.lowerMasks.data(), s.upperMasks.data(), s.rootOrigins.data(),
                                             s.rootOrigins.size()}};
        // Is the face above the free cell c a released wall face? It is the lower face of the solid cell above c.
        auto releasedAbove = [=] __device__(nanovdb::Coord c) {
            c[1] += 1;
            return where.solid(c) && ((int(flags[cells->getAccessor().getValue(c)]) >> 1) & 1) != 0;
        };
        using nanovdb::util::cuda::blocksPerGrid;
        using nanovdb::util::cuda::lambdaKernel;
        // Free cells below a released face. Writes free cells; reads only the solids and the release state.
        lambdaKernel<<<blocksPerGrid(voxels, 256), 256>>>(voxels, [=] __device__(size_t tid) {
            const auto&    leaf = grid->tree().getFirstLeaf()[tid >> 9];
            const uint32_t n    = uint32_t(tid & 511);
            if (!leaf.isActive(n)) return;
            const nanovdb::Coord ijk = leaf.offsetToGlobalCoord(n);
            if (where.solid(ijk)) return;
            for (int k = 1; float(k) - 0.5f < bg; ++k) {
                nanovdb::Coord up = ijk;
                up[1] += k;
                if (!where.solid(up)) continue;
                up[1] -= 1; // the free cell under the face
                const float toFace = float(k) - 0.5f;
                if (releasedAbove(up) && -toFace > phi[leaf.getValue(n)]) phi[leaf.getValue(n)] = -toFace;
                break;
            }
        });
        cudaCheckError();
        // Solid cells. Reads phi (free cells, and solid cells where the line into free space crosses a corner), so
        // it writes into a copy.
        BufferT<float> result = makeBuffer<float>(s.phi.size());
        cudaCheck(cudaMemcpy(result.data(), phi, s.phi.size() * sizeof(float), cudaMemcpyDeviceToDevice));
        float* out = result.data();
        lambdaKernel<<<blocksPerGrid(voxels, 256), 256>>>(voxels, [=] __device__(size_t tid) {
            const auto&    leaf = grid->tree().getFirstLeaf()[tid >> 9];
            const uint32_t n    = uint32_t(tid & 511);
            if (!leaf.isActive(n)) return;
            const nanovdb::Coord ijk = leaf.offsetToGlobalCoord(n);
            if (!where.solid(ijk)) return;
            // The way out of the solid: along the gradient of psi, by the depth.
            const float depth = -where.at(ijk);
            float       g[3], length = 0.0f;
            for (int a = 0; a < 3; ++a) {
                nanovdb::Coord p = ijk, m = ijk;
                p[a] += 1, m[a] -= 1;
                g[a] = where.at(p) - where.at(m);
                length += g[a] * g[a];
            }
            length = sqrtf(length);
            if (length < 1.0e-6f) g[0] = 0.0f, g[1] = 1.0f, g[2] = 0.0f, length = 1.0f;
            // The free side's phi continued into the solid along a straight line through two points: half a voxel
            // outside the solid's surface, and one voxel further (slope limited to 1). Constant where the water
            // surface meets the solid at a right angle, growing by one per voxel where it runs along the solid.
            const float     reach = depth + 0.5f;
            advect::PosT    p0, p1;
            for (int a = 0; a < 3; ++a) {
                p0[a] = advect::RealT(float(ijk[a]) + reach * g[a] / length);
                p1[a] = p0[a] + advect::RealT(g[a] / length);
            }
            const auto  acc = channel(*grid, phi);
            float       lowest, highest;
            const float atWall = advect::interpolate(acc, p0, lowest, highest);
            float       slope  = atWall - advect::interpolate(acc, p1, lowest, highest);
            slope = slope > 1.0f ? 1.0f : (slope < -1.0f ? -1.0f : slope);
            if (atWall >= 0.0f && slope < 0.0f) slope = 0.0f; // air in front of a solid never turns into water inside it
            const float continued = atWall + reach * slope;
            const nanovdb::Coord front(int(floorf(float(p0[0]) + 0.5f)), int(floorf(float(p0[1]) + 0.5f)), int(floorf(float(p0[2]) + 0.5f)));
            const bool  letGo = !where.solid(front) && releasedAbove(front);
            const float floor = letGo ? depth : depth - extension; // air behind a released face
            float       value = continued > floor ? continued : floor;
            value = value > bg ? bg : (value < -bg ? -bg : value);
            out[leaf.getValue(n)] = value;
        });
        cudaCheckError();
        cudaCheck(cudaMemcpy(phi, out, s.phi.size() * sizeof(float), cudaMemcpyDeviceToDevice));
        // Safety net. A tracker can only be built on a level set whose inactive regions each touch active voxels
        // of one sign (PropagateSign's preconditions 2 and 3). The state's masks give the sign of every inactive
        // voxel; an active voxel next to an inactive one takes that sign, with a value just off zero, if the rule
        // or the advection left it on the other side. It reads only which voxels are active and the masks.
        const InvertView masks{s.leafMasks.data(), s.lowerMasks.data(), s.upperMasks.data(), s.rootOrigins.data(), s.rootOrigins.size()};
        auto forced = makeBuffer<int>(1);
        cudaCheck(cudaMemset(forced.data(), 0, sizeof(int)));
        int* forcedCount = forced.data();
        lambdaKernel<<<blocksPerGrid(voxels, 256), 256>>>(voxels, [=] __device__(size_t tid) {
            const auto&    leaf = grid->tree().getFirstLeaf()[tid >> 9];
            const uint32_t n    = uint32_t(tid & 511);
            if (!leaf.isActive(n)) return;
            const nanovdb::Coord ijk = leaf.offsetToGlobalCoord(n);
            const auto           acc = grid->getAccessor();
            bool nextToInside = false, nextToOutside = false;
            for (int j = 0; j < 6; ++j) {
                nanovdb::Coord q = ijk;
                q[j / 2] += (j & 1) ? 1 : -1;
                if (acc.getValue(q)) continue;
                const bool inside = nanovdb::tools::cuda::levelset::detail::isInverted(*grid, q, masks.leaf, masks.lower, masks.upper,
                                                                                         masks.roots, masks.rootCount);
                (inside ? nextToInside : nextToOutside) = true;
            }
            float& value = phi[leaf.getValue(n)];
            if (nextToInside && value >= 0.0f) { value = -1.0e-3f; atomicAdd(forcedCount, 1); }
            else if (nextToOutside && !nextToInside && value < 0.0f) { value = 1.0e-3f; atomicAdd(forcedCount, 1); }
        });
        cudaCheckError();
        // Second part of the safety net, for the prune that follows: a voxel at the background (it would be removed)
        // that has a face neighbor of the other sign stays just inside the band, so that the neighbor does not end up
        // next to an inactive voxel of the other sign. Reads the signs of the neighbors, writes only magnitudes.
        lambdaKernel<<<blocksPerGrid(voxels, 256), 256>>>(voxels, [=] __device__(size_t tid) {
            const auto&    leaf = grid->tree().getFirstLeaf()[tid >> 9];
            const uint32_t n    = uint32_t(tid & 511);
            if (!leaf.isActive(n)) return;
            float& value = phi[leaf.getValue(n)];
            if (value < bg && value > -bg) return;
            const nanovdb::Coord ijk = leaf.offsetToGlobalCoord(n);
            const auto           acc = grid->getAccessor();
            for (int j = 0; j < 6; ++j) {
                nanovdb::Coord q = ijk;
                q[j / 2] += (j & 1) ? 1 : -1;
                const uint64_t k = acc.getValue(q);
                if (k && (phi[k] < 0.0f) != (value < 0.0f)) {
                    value = value < 0.0f ? -0.99f * bg : 0.99f * bg;
                    atomicAdd(forcedCount, 1);
                    return;
                }
            }
        });
        cudaCheckError();
        cudaCheck(cudaDeviceSynchronize());
        int host = 0;
        cudaCheck(cudaMemcpy(&host, forced.data(), sizeof(int), cudaMemcpyDeviceToHost));
        stats.forced += host;
    }

    // The domain grid, its coordinates, phi and velocity on the host.
    struct Snapshot
    {
        nanovdb::HostBuffer         bytes;
        std::vector<nanovdb::Coord> coord;
        std::vector<float>          phi, u[3];
        const GridT* grid() const { return reinterpret_cast<const GridT*>(bytes.data()); }
    };
    Snapshot snapshot() const
    {
        Snapshot s;
        s.bytes = nanovdb::HostBuffer::create(domain->buffer().size());
        cudaCheck(cudaMemcpy(s.bytes.data(), domain->buffer().data(), s.bytes.size(), cudaMemcpyDeviceToHost));
        s.coord.resize(domainCount);
        for (uint32_t l = 0; l < s.grid()->tree().nodeCount(0); ++l) {
            const auto& leaf = s.grid()->tree().getFirstLeaf()[l];
            for (uint32_t m = 0; m < 512; ++m)
                if (leaf.isActive(m)) s.coord[leaf.getValue(m)] = leaf.offsetToGlobalCoord(m);
        }
        s.phi.resize(domainCount);
        cudaCheck(cudaMemcpy(s.phi.data(), domainPhi->data(), domainCount * sizeof(float), cudaMemcpyDeviceToHost));
        for (int a = 0; a < 3; ++a) {
            s.u[a].resize(domainCount);
            cudaCheck(cudaMemcpy(s.u[a].data(), u[a].data(), domainCount * sizeof(float), cudaMemcpyDeviceToHost));
        }
        return s;
    }
};

// Volume in cells: each tank cell counts by the fraction a flat surface at distance phi would cut from it.
static double volumeOf(const Simulation& sim, const Simulation::Snapshot& snap)
{
    double volume = 0;
    for (size_t k = 1; k < snap.coord.size(); ++k)
        if (sim.isFree(snap.coord[k])) volume += std::min(1.0, std::max(0.0, 0.5 - double(snap.phi[k])));
    return volume;
}

// Kinetic and potential energy of the water, per unit density, in voxel units (voxels^5 / s^2). A cell counts by its
// water fraction as in volumeOf, a face by the mean fraction of its two cells; u[a][k] is the velocity on the lower
// face of cell k along a. Heights are measured from y = -0.5, the floor of a tank whose lowest cells are y = 0.
static void energyOf(const Simulation& sim, const Simulation::Snapshot& snap, double& kinetic, double& potential)
{
    const auto acc = snap.grid()->getAccessor();
    auto fraction = [&](const nanovdb::Coord& c, uint64_t k) {
        return k && sim.isFree(c) ? std::min(1.0, std::max(0.0, 0.5 - double(snap.phi[k]))) : 0.0;
    };
    kinetic = potential = 0;
    for (size_t k = 1; k < snap.coord.size(); ++k) {
        const nanovdb::Coord& c    = snap.coord[k];
        const double          here = fraction(c, k);
        potential += sim.gravity * (c[1] + 0.5) * here;
        for (int a = 0; a < 3; ++a) {
            nanovdb::Coord lower = c;
            lower[a] -= 1;
            const double weight = 0.5 * (here + fraction(lower, acc.getValue(lower)));
            kinetic += 0.5 * weight * double(snap.u[a][k]) * double(snap.u[a][k]);
        }
    }
}

// DUMP=<prefix>: the slice z = nz / 2 as <prefix>_<index>.pgm, one byte per cell of the tank and of 8 cells around
// it: phi mapped from [-background, background] to [0, 250], 252 = obstacle, 255 = not in the domain grid.
static void dumpSlice(const Simulation& sim, const Simulation::Snapshot& snap, int index)
{
    const char* prefix = std::getenv("DUMP");
    if (!prefix) return;
    const int  nx = sim.tank.n[0], ny = sim.tank.n[1], nz = sim.tank.n[2], pad = 8, w = nx + 2 * pad, h = ny + 2 * pad;
    const auto acc = snap.grid()->getAccessor();
    std::vector<unsigned char> image(size_t(w) * h, 255);
    for (int y = -pad; y < ny + pad; ++y)
        for (int x = -pad; x < nx + pad; ++x) {
            const nanovdb::Coord c(x, y, nz / 2);
            unsigned char& pixel = image[size_t(h - 1 - (y + pad)) * w + (x + pad)];
            if (sim.tank.inside(c) && !sim.isFree(c)) { pixel = 252; continue; } // an obstacle
            const uint64_t idx = acc.getValue(c);
            if (!idx) continue;
            const double t = std::min(1.0, std::max(-1.0, double(snap.phi[idx]) / sim.background));
            pixel = (unsigned char)std::lround((t + 1.0) * 125.0);
        }
    char name[512];
    std::snprintf(name, sizeof(name), "%s_%03d.pgm", prefix, index);
    const std::string temporary = std::string(name) + ".tmp";
    std::FILE* f = std::fopen(temporary.c_str(), "wb");
    std::fprintf(f, "P5\n%d %d\n255\n", w, h);
    std::fwrite(image.data(), 1, image.size(), f);
    std::fclose(f);
    std::rename(temporary.c_str(), name);
}

// A surface point for the web viewer: position as 3 x uint16 (cell-center coordinates + 0.5, divided by the tank's
// longest side) and the unit normal as 2 x int8 (octahedral).
static void appendPoint(std::vector<char>& points, const double p[3], const double normal[3], double longest)
{
    uint16_t q[3];
    for (int a = 0; a < 3; ++a) q[a] = uint16_t(std::lround(std::min(1.0, std::max(0.0, (p[a] + 0.5) / longest)) * 65535.0));
    const double l1 = std::fabs(normal[0]) + std::fabs(normal[1]) + std::fabs(normal[2]);
    double ox = normal[0] / l1, oy = normal[1] / l1;
    if (normal[2] < 0) {
        const double tx = (1 - std::fabs(oy)) * (ox >= 0 ? 1 : -1), ty = (1 - std::fabs(ox)) * (oy >= 0 ? 1 : -1);
        ox = tx, oy = ty;
    }
    const int8_t o[2] = {int8_t(std::lround(ox * 127)), int8_t(std::lround(oy * 127))};
    points.insert(points.end(), reinterpret_cast<const char*>(q), reinterpret_cast<const char*>(q) + 6);
    points.insert(points.end(), reinterpret_cast<const char*>(o), reinterpret_cast<const char*>(o) + 2);
}

// DUMP3D=<prefix>: the water surface as points, <prefix>_<index>.pts; with index 0 also the obstacle's surface,
// <prefix>_solid.pts. A point is a cell within 0.6 voxel of the surface, moved onto it along the gradient.
static void dumpPoints(const Simulation& sim, const Simulation::Snapshot& snap, int index)
{
    const char* prefix = std::getenv("DUMP3D");
    if (!prefix) return;
    const double longest = std::max({sim.tank.n[0], sim.tank.n[1], sim.tank.n[2]});
    auto write = [&](const std::string& name, const std::vector<char>& points) {
        std::FILE* f = std::fopen(name.c_str(), "wb");
        std::fwrite(points.data(), 1, points.size(), f);
        std::fclose(f);
    };
    char name[512];
    if (index == 0) {
        // The obstacle: cells of the tank next to a solid that is not a tank wall.
        std::vector<char> points;
        for (int i = 0; i < sim.tank.n[0]; ++i)
            for (int j = 0; j < sim.tank.n[1]; ++j)
                for (int k = 0; k < sim.tank.n[2]; ++k) {
                    const nanovdb::Coord c(i, j, k);
                    const float v = sim.hostPsiAt(c);
                    if (std::fabs(v) >= 0.6f || v == -sim.tank.distance(c, 0.0f)) continue;
                    double g[3], length = 0;
                    for (int a = 0; a < 3; ++a) {
                        nanovdb::Coord p = c, m = c;
                        p[a] += 1, m[a] -= 1;
                        g[a] = 0.5 * (sim.hostPsiAt(p) - sim.hostPsiAt(m));
                        length += g[a] * g[a];
                    }
                    length = std::sqrt(length);
                    if (length < 1e-6) continue;
                    const double pos[3] = {i - v * g[0] / length, j - v * g[1] / length, k - v * g[2] / length};
                    const double n[3] = {g[0] / length, g[1] / length, g[2] / length};
                    appendPoint(points, pos, n, longest);
                }
        write(std::string(prefix) + "_solid.pts", points);
    }
    std::vector<char> points;
    const auto acc = snap.grid()->getAccessor();
    for (size_t k = 1; k < snap.coord.size(); ++k) {
        const nanovdb::Coord c = snap.coord[k];
        if (std::fabs(snap.phi[k]) >= 0.6f || !sim.isFree(c)) continue;
        double g[3], length = 0;
        bool   complete = true;
        for (int a = 0; a < 3; ++a) {
            nanovdb::Coord p = c, m = c;
            p[a] += 1, m[a] -= 1;
            const uint64_t ip = acc.getValue(p), im = acc.getValue(m);
            if (!ip || !im) { complete = false; break; }
            g[a] = 0.5 * (double(snap.phi[ip]) - snap.phi[im]);
            length += g[a] * g[a];
        }
        if (!complete) continue;
        length = std::sqrt(length);
        if (length < 1e-6) continue;
        const double v = snap.phi[k];
        const double pos[3] = {c[0] - v * g[0] / length, c[1] - v * g[1] / length, c[2] - v * g[2] / length};
        const double n[3] = {g[0] / length, g[1] / length, g[2] / length};
        appendPoint(points, pos, n, longest);
    }
    std::snprintf(name, sizeof(name), "%s_%03d.pts", prefix, index);
    write(name, points);
}

// BAND=<prefix> with SPRAY: the drops, <prefix>_<index>.spray: int32 count, int32 every, then per drop float32 x, y, z
// (cell-center coordinates), volume (cells) and terminal speed (m/s). SPRAY_DUMP_EVERY=<n> (default 1) writes one drop
// in n (a renderer then lets each stand for n); `count` is the number written. The choice is made by a hash of the
// terminal speed, which a drop keeps for life: the same drops are written in every frame. (Without drag all drops
// have the terminal speed 0, and either all or none are written.)
static void dumpSpray(const Simulation& sim, int index)
{
    const char* prefix = std::getenv("BAND");
    if (!prefix || !sim.drops) return;
    const int                              every = std::getenv("SPRAY_DUMP_EVERY") ? std::max(1, std::atoi(std::getenv("SPRAY_DUMP_EVERY"))) : 1;
    const thrust::host_vector<spray::Drop> drops = sim.drops->drops;
    std::vector<float>                     rows;
    rows.reserve(5 * (drops.size() / every + 1));
    for (size_t k = 0; k < drops.size(); ++k) {
        const spray::Drop d = drops[k];
        if (spray::mix(spray::bitsOf(d.terminal)) % uint32_t(every) == 0) rows.insert(rows.end(), {d.x, d.y, d.z, d.volume, d.terminal});
    }
    char name[512];
    std::snprintf(name, sizeof(name), "%s_%03d.spray", prefix, index);
    std::FILE*    f = std::fopen(name, "wb");
    const int32_t header[2] = {int32_t(rows.size() / 5), int32_t(every)};
    std::fwrite(header, 4, 2, f);
    std::fwrite(rows.data(), 4, rows.size(), f);
    std::fclose(f);
}

// BAND=<prefix>: what a renderer needs to rebuild the water surface, <prefix>_<index>.band. The water is cut where
// it meets a solid: value = max(phi, -psi), so the surface closes on the tank walls and on the obstacle instead of
// continuing behind them. Records for the cells with |value| < 2: int16 x, y, z and float32 value (10 bytes), after
// a header of four int32: count, nx, ny, nz.
static void dumpBand(const Simulation& sim, const Simulation::Snapshot& snap, int index)
{
    const char* prefix = std::getenv("BAND");
    if (!prefix) return;
    std::vector<char> records;
    int32_t count = 0;
    for (size_t k = 1; k < snap.coord.size(); ++k) {
        const nanovdb::Coord c = snap.coord[k];
        const float value = std::max(snap.phi[k], -sim.hostPsiAt(c));
        if (std::fabs(value) >= 2.0f) continue;
        const int16_t xyz[3] = {int16_t(c[0]), int16_t(c[1]), int16_t(c[2])};
        records.insert(records.end(), reinterpret_cast<const char*>(xyz), reinterpret_cast<const char*>(xyz) + 6);
        records.insert(records.end(), reinterpret_cast<const char*>(&value), reinterpret_cast<const char*>(&value) + 4);
        ++count;
    }
    char name[512];
    std::snprintf(name, sizeof(name), "%s_%03d.band", prefix, index);
    const std::string temporary = std::string(name) + ".tmp"; // renamed when complete: a puller never sees half a file
    std::FILE* f = std::fopen(temporary.c_str(), "wb");
    const int32_t header[4] = {count, sim.tank.n[0], sim.tank.n[1], sim.tank.n[2]};
    std::fwrite(header, sizeof(int32_t), 4, f);
    std::fwrite(records.data(), 1, records.size(), f);
    std::fclose(f);
    std::rename(temporary.c_str(), name);
}

// Switches for experiments: EXT=<voxels behind the wall>, RULE_AFTER=1.
static void applySwitches(Simulation& sim)
{
    if (const char* v = std::getenv("RULE_AFTER")) sim.ruleAfterTrack = std::atoi(v) != 0;
    if (const char* v = std::getenv("DIAG")) sim.diagnose = std::atoi(v) != 0;
    if (const char* v = std::getenv("PLS")) { // particles per cell; 0 = off
        if (std::atoi(v) > 0) sim.markers.emplace(), sim.markers->settings.perCell = std::atoi(v);
    }
    if (sim.markers) {
        if (const char* v = std::getenv("PLS_WIDTH")) sim.markers->settings.width = float(std::atof(v));
        if (const char* v = std::getenv("PLS_REACH")) sim.markers->settings.reach = std::atoi(v);
        if (const char* v = std::getenv("PLS_RMAX")) sim.markers->settings.rMax = float(std::atof(v));
        if (const char* v = std::getenv("PLS_AIR")) sim.markers->settings.airCorrects = std::atoi(v) != 0; // 0: air particles do not correct
        if (const char* v = std::getenv("PLS_RESEED")) sim.reseedEvery = std::atoi(v);
        std::printf("markers: %d per cell within %.1f voxels of the surface, radius %.2f..%.2f, reach %d, reseeding every %d steps\n",
                    sim.markers->settings.perCell, sim.markers->settings.width, sim.markers->settings.rMin, sim.markers->settings.rMax,
                    sim.markers->settings.reach, sim.reseedEvery);
    }
    if (const char* v = std::getenv("VOLCTL")) sim.volumeControl = std::atoi(v); // 0 off, 1 proportional, 2 proportional + integral
    if (const char* v = std::getenv("FRICTION")) sim.wallFriction = std::atof(v);
    if (const char* v = std::getenv("SPRAY")) { // 1: drops carry the volume the level set loses; needs PLS and PIPE=2
        if (std::atoi(v) > 0) sim.drops.emplace();
    }
    if (sim.drops) {
        if (const char* v = std::getenv("SPRAY_LARGEST")) sim.drops->largest = float(std::atof(v));
        if (const char* v = std::getenv("SPRAY_FRICTION")) sim.drops->friction = float(std::atof(v));
        if (const char* v = std::getenv("SPRAY_MERGE")) sim.drops->mergeTime = float(std::atof(v));
        if (const char* v = std::getenv("SPRAY_TERMINAL")) { // "<low> <high>" in m/s; "0 0": no air drag
            double low = 0, high = 0;
            if (std::sscanf(v, "%lf %lf", &low, &high) != 2) throw std::runtime_error("SPRAY_TERMINAL needs two numbers");
            sim.drops->terminalLow = float(low), sim.drops->terminalHigh = float(high);
        }
        if (const char* v = std::getenv("SPRAY_BOUNCE")) sim.drops->bounce = float(std::atof(v));
        if (const char* v = std::getenv("SPRAY_JITTER")) sim.drops->jitter = float(std::atof(v));
        std::printf("drops: at most %.2f cells each, friction %.2f, merge time %.2f s, terminal speed %.1f..%.1f m/s, bounce up to %.2f, "
                    "velocity jitter %.2f\n", sim.drops->largest, sim.drops->friction, sim.drops->mergeTime, sim.drops->terminalLow,
                    sim.drops->terminalHigh, sim.drops->bounce, sim.drops->jitter);
    }
    if (const char* v = std::getenv("VOLCTL_NP")) sim.controlSteps = std::atof(v);
    if (const char* v = std::getenv("VOLCTL_ZETA")) sim.controlDamping = std::atof(v);
    if (const char* v = std::getenv("SEPARATE")) sim.separate = std::atoi(v) != 0;
    if (const char* v = std::getenv("BANDGROW")) sim.bandGrow = std::atoi(v);
    if (const char* v = std::getenv("RELEASE_AXES")) sim.releaseAxes = std::atoi(v);
    if (const char* v = std::getenv("NORM")) sim.normCount = std::atoi(v);
    if (const char* v = std::getenv("TRACE")) sim.trace = std::atoi(v);
    if (const char* v = std::getenv("VMAX")) sim.speedLimit = std::atof(v);
    if (const char* v = std::getenv("NORM_KEEP")) sim.keepInterface = std::atoi(v) != 0;
    if (const char* v = std::getenv("PIPE")) sim.pipeline = std::atoi(v);
    if (const char* v = std::getenv("MG_SCALE")) sim.mgScale = float(std::atof(v));
    if (const char* v = std::getenv("MG_SWEEPS")) sim.mgSweeps = std::atoi(v);
    if (const char* v = std::getenv("PSOLVER")) sim.pressureSolver = std::atoi(v);
    if (const char* v = std::getenv("PTOL")) sim.pressureTolerance = std::atof(v);
    if (const char* v = std::getenv("PCHECK")) sim.pressureCheck = std::atoi(v) != 0;
    if (const char* v = std::getenv("TRACE_SPIKE")) sim.traceSpike = std::atof(v);
    if (const char* v = std::getenv("THETA")) sim.minimumTheta = Real(std::atof(v)); // smallest theta of the ghost fluid boundary
    if (const char* v = std::getenv("CFL")) sim.cfl = std::atof(v);
    if (const char* v = std::getenv("PHI_LIMIT")) sim.phiCorrectionLimit = float(std::atof(v));
    if (const char* v = std::getenv("SWEEPS")) sim.sweeps = std::atoi(v);
    if (const char* v = std::getenv("HALO")) sim.halo = std::atoi(v);
    if (const char* v = std::getenv("PHI_SCHEME"))
        sim.phiScheme = std::atoi(v) == 0 ? advect::SemiLagrangian : std::atoi(v) == 2 ? advect::MacCormackRevert : advect::MacCormackClamp;
    if (const char* v = std::getenv("U_SCHEME")) // 0 semi-Lagrangian, 1 MacCormack + clamp, 2 MacCormack + revert
        sim.velocityScheme = std::atoi(v) == 0 ? advect::SemiLagrangian : (std::atoi(v) == 2 ? advect::MacCormackRevert : advect::MacCormackClamp);
    if (const char* v = std::getenv("U_SL_NEAR_SOLID")) sim.firstOrderNearSolid = float(std::atof(v));
    auto name = [](advect::Scheme scheme) {
        return scheme == advect::SemiLagrangian ? "semi-Lagrangian" : scheme == advect::MacCormackRevert ? "MacCormack+revert" : "MacCormack+clamp";
    };
    std::printf("cfl %.2f, sweeps %d, halo %d, phi %s, velocity %s\n", sim.cfl, sim.sweeps, sim.halo, name(sim.phiScheme),
                name(sim.velocityScheme));
    std::printf("switches: extension %d, rule after track %d, walls let go %d, wall friction %g\n", sim.tank.extension, int(sim.ruleAfterTrack),
                int(sim.separate), sim.wallFriction);
}
static int extensionSwitch() { return std::getenv("EXT") ? std::atoi(std::getenv("EXT")) : 2; }

static const Simulation* gSimulation = nullptr; // for the exception message

// An obstacle from a file written by gen_sdf (MeshToSDF: index grid + sidecar, with a band of a few voxels): the grid
// and its values as they are, with the invert masks the tracker's PropagateSign makes for them. One voxel of the file
// is one cell of the tank. axes[a] is the mesh axis that lies along tank axis a; the corner of the mesh's bounding box
// with the smallest coordinates goes to the tank cell `at`.
static Obstacle loadObstacle(const char* path, const Tank& tank, const int atGiven[3], const int axes[3], bool meshFrame = false)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error(std::string("cannot open ") + path);
    uint64_t gridBytes, valueCount;
    double   dx;
    float    fileBackground;
    in.read(reinterpret_cast<char*>(&gridBytes), 8).read(reinterpret_cast<char*>(&valueCount), 8);
    in.read(reinterpret_cast<char*>(&dx), 8).read(reinterpret_cast<char*>(&fileBackground), 4);
    Obstacle result;
    result.grid = nanovdb::HostBuffer::create(gridBytes);
    result.values.resize(valueCount);
    in.read(reinterpret_cast<char*>(result.grid.data()), gridBytes).read(reinterpret_cast<char*>(result.values.data()), valueCount * 4);
    const std::vector<float>& values = result.values;
    // The invert masks say on which side of the mesh a voxel outside the band is.
    BufferT<std::byte> gridBuffer = makeBuffer<std::byte>(gridBytes);
    cudaCheck(cudaMemcpy(gridBuffer.data(), result.grid.data(), gridBytes, cudaMemcpyHostToDevice));
    BufferT<float> phi = makeBuffer<float>(valueCount);
    cudaCheck(cudaMemcpy(phi.data(), values.data(), valueCount * 4, cudaMemcpyHostToDevice));
    TrackerT tracker(HandleT(std::move(gridBuffer)), std::move(phi), fileBackground);
    result.state.emplace(tracker.release());
    const StateT& state = *result.state;
    if (hostCopy(state.phi) != values) throw std::runtime_error("loadObstacle: the tracker changed the obstacle's values");
    result.leafMasks = hostCopy(state.leafMasks), result.lowerMasks = hostCopy(state.lowerMasks);
    result.upperMasks = hostCopy(state.upperMasks), result.rootOrigins = hostCopy(state.rootOrigins);
    const auto* grid = reinterpret_cast<const GridT*>(result.grid.data());
    // lo, hi: the voxels inside the mesh; activeLo, activeHi: all voxels of the band.
    nanovdb::Coord lo(1 << 30), hi(-(1 << 30)), activeLo(1 << 30), activeHi(-(1 << 30));
    uint64_t       active = 0;
    for (uint32_t l = 0; l < grid->tree().nodeCount(0); ++l) {
        const auto& node = grid->tree().getFirstLeaf()[l];
        for (uint32_t m = 0; m < 512; ++m) {
            if (!node.isActive(m)) continue;
            const nanovdb::Coord ijk = node.offsetToGlobalCoord(m);
            activeLo.minComponent(ijk), activeHi.maxComponent(ijk), ++active;
            if (values[node.getValue(m)] < 0) lo.minComponent(ijk), hi.maxComponent(ijk);
        }
    }
    int at[3]; // meshFrame: the file's voxel (i, j, k) is the tank cell (i, j, k) along the tank's axes
    for (int a = 0; a < 3; ++a) at[a] = meshFrame ? lo[axes[a]] : atGiven[a];
    result.voxel = dx, result.band = float(fileBackground / dx);
    for (int a = 0; a < 3; ++a) result.axes[a] = axes[a], result.shift[a] = lo[axes[a]] - at[a];
    const double deviceBytes = double(gridBytes) + double(valueCount) * 4 + double(state.leafMasks.size_bytes()) +
                               double(state.lowerMasks.size_bytes()) + double(state.upperMasks.size_bytes());
    std::printf("obstacle %s: %d x %d x %d voxels along the tank's axes, band %.1f voxels, %llu voxels stored, %.0f MiB on the device\n",
                path, hi[axes[0]] - lo[axes[0]] + 1, hi[axes[1]] - lo[axes[1]] + 1, hi[axes[2]] - lo[axes[2]] + 1, result.band,
                (unsigned long long)active, deviceBytes / 1048576.0);
    // OBS_OFFSET=<voxels>: the solid is everything within this distance of the mesh's solid (psi -= offset), so that a
    // sheet thinner than a voxel (a glass panel, a fence) still holds the water back.
    if (const char* v = std::getenv("OBS_OFFSET")) {
        result.offset = float(std::atof(v));
        std::printf("obstacle offset %.2f voxels\n", result.offset);
    }
    // OBS_DUMP=<file>: psi as a dense array for a look at it, over the tank cells of the file's band (cut to the tank
    // and Simulation::psiPad cells around it): int32 lo[3], size[3], then size[0] * size[1] * size[2] floats (x
    // outermost, z innermost), in voxels.
    if (const char* name = std::getenv("OBS_DUMP")) {
        pressure::Solid field{};
        field.grid = grid, field.values = values.data(), field.leafMasks = result.leafMasks.data();
        field.lowerMasks = result.lowerMasks.data(), field.upperMasks = result.upperMasks.data();
        field.rootOrigins = result.rootOrigins.data(), field.rootOriginCount = result.rootOrigins.size();
        field.voxel = result.voxel, field.band = result.band, field.offset = result.offset;
        const int pad = Simulation::psiPad;
        int       first[3], size[3];
        for (int a = 0; a < 3; ++a) {
            field.shift[a] = result.shift[a], field.axes[a] = result.axes[a];
            first[a] = std::max(-pad, activeLo[axes[a]] - lo[axes[a]] + at[a]);
            size[a]  = std::max(0, std::min(tank.n[a] + pad - 1, activeHi[axes[a]] - lo[axes[a]] + at[a]) - first[a] + 1);
        }
        std::vector<float> psi(size_t(size[0]) * size[1] * size[2]);
        for (int i = 0; i < size[0]; ++i)
            for (int j = 0; j < size[1]; ++j)
                for (int k = 0; k < size[2]; ++k)
                    psi[(size_t(i) * size[1] + j) * size[2] + k] = field.obstacle(nanovdb::Coord(i + first[0], j + first[1], k + first[2]));
        std::ofstream out(name, std::ios::binary);
        out.write(reinterpret_cast<const char*>(first), 12).write(reinterpret_cast<const char*>(size), 12);
        out.write(reinterpret_cast<const char*>(psi.data()), psi.size() * sizeof(float));
    }
    // For a renderer that places the original mesh: tank cell coordinate a = mesh coordinate axes[a] / voxel - lo[axes[a]] + at[a].
    std::printf("obstacle transform: voxel %.9g, lo %d %d %d, at %d %d %d, axes %d %d %d\n", dx, lo[0], lo[1], lo[2], at[0], at[1], at[2],
                axes[0], axes[1], axes[2]);
    return result;
}

static void printStats(int step, const Simulation& sim)
{
    const auto& s = sim.stats;
    std::printf("step %5d t %.5f dt %.2e | fluid cells %8llu, band %7llu, domain %8llu | speed %.3e | pressure %3d iterations, "
                "residual %.1e, divergence %.1e | missing %d, invalid %d | released faces %d, %d solves (max %d), forced %ld | %.0f ms | limited %ld\n", step, sim.time, s.dt,
                (unsigned long long)s.fluidCells, (unsigned long long)s.bandVoxels, (unsigned long long)s.domainVoxels, s.maxSpeed,
                s.iterations, s.residual, s.divergence, s.missing, s.invalid, s.releasedFaces, s.releaseIterations, s.maxSolves, s.forced, s.ms, s.limited);
}

static int runHydrostatic(int n, double height, int steps)
{
    const Tank   tank{{n, n, n}, 2};
    const double lo[3] = {-1, -1, -1}, hi[3] = {double(n), height, double(n)};
    Simulation   sim(tank, lo, hi);
    sim.gravity = 9.81 * n; // the tank is 1 m wide
    std::printf("hydrostatic: %d^3 tank, surface at %.2f, gravity %.1f voxels/s^2\n", n, height, sim.gravity);
    double largestSpeed = 0;
    for (int i = 1; i <= steps; ++i) {
        sim.step(1.0);
        largestSpeed = std::max(largestSpeed, sim.maxSpeed());
        if (i == 1 || i % std::max(1, steps / 8) == 0) printStats(i, sim);
    }
    // The surface: phi against y - height, away from the side walls, within 1.5 voxels of the surface.
    const auto snap = sim.snapshot();
    double worst = 0;
    for (size_t i = 1; i < snap.coord.size(); ++i) {
        const auto& c = snap.coord[i];
        if (c[0] < 4 || c[2] < 4 || c[0] >= n - 4 || c[2] >= n - 4 || std::fabs(c[1] - height) > 1.5) continue;
        worst = std::max(worst, std::fabs(double(snap.phi[i]) - (c[1] - height)));
    }
    std::printf("after %d steps (t = %.4f s): largest speed seen %.3e voxels/s (free fall over that time would reach %.1f); "
                "surface moved by at most %.3e voxel\n", steps, sim.time, largestSpeed, sim.gravity * sim.time, worst);
    return 0;
}

static int runDam(int nx, int ny, int nz, double width, double height, double endTime)
{
    const Tank   tank{{nx, ny, nz}, extensionSwitch()};
    double lo[3] = {-1, -1, -1}, hi[3] = {width - 0.5, height - 0.5, double(nz)};
    // BOX=x0,y0,z0,x1,y1,z1: the water at the start is this box (cell-center coordinates) instead of the column.
    if (const char* v = std::getenv("BOX")) std::sscanf(v, "%lf,%lf,%lf,%lf,%lf,%lf", &lo[0], &lo[1], &lo[2], &hi[0], &hi[1], &hi[2]);
    // OBSTACLE=<file.sdf> OBS_AT=x,y,z OBS_AXES=zyx: a mesh obstacle (see loadObstacle).
    Obstacle obstacle;
    if (const char* path = std::getenv("OBSTACLE")) {
        int at[3] = {nx / 2, 0, nz / 4}, axes[3] = {0, 1, 2};
        if (const char* v = std::getenv("OBS_AT")) std::sscanf(v, "%d,%d,%d", &at[0], &at[1], &at[2]);
        const bool meshFrame = std::getenv("OBS_AT") && std::string(std::getenv("OBS_AT")) == "mesh"; // tank cell = mesh / voxel
        if (const char* v = std::getenv("OBS_AXES")) for (int a = 0; a < 3; ++a) axes[a] = v[a] - 'x';
        obstacle = loadObstacle(path, tank, at, axes, meshFrame);
    }
    MemoryWatch memory;
    cudaCheck(cudaDeviceSynchronize()); // the context exists from here on
    memory.start(), gMemory = &memory;
    Simulation   sim(tank, lo, hi, std::move(obstacle));
    sim.gravity = 9.81 * nx; // the tank is 1 m long
    // VOXEL=<meters>: a cell's size, for a scene in real units. With width = 2 * gravity (in voxels), T* is seconds.
    if (const char* v = std::getenv("VOXEL")) sim.gravity = 9.81 / std::atof(v);
    applySwitches(sim);
    gSimulation = &sim;
    const double timeScale = std::sqrt(2.0 * sim.gravity / width); // T* = t * timeScale
    std::printf("dam break: tank %d x %d x %d, column %.1f x %.1f, gravity %.1f voxels/s^2\n", nx, ny, nz, width, height, sim.gravity);
    std::printf("   T*      t      front/width  height/initial  volume (cells)  max|w| / max speed\n");
    double nextReport = 0;
    double firstVolume = 0, firstEnergy = 0;
    int    reportIndex = 0;
    int    firstStep   = 1;
    // CKPT_LOAD=<file>: continue from a checkpoint. CKPT_SAVE=<prefix> CKPT_AT=<T*>,<T*>,...: write <prefix>_<T*>.ckpt
    // when the run passes each of these times.
    if (const char* path = std::getenv("CKPT_LOAD")) {
        double extra[3];
        sim.load(path, extra);
        firstVolume = extra[0], firstStep = int(extra[1]) + 1, reportIndex = int(extra[2]);
        nextReport = sim.time * timeScale;
        std::printf("continuing from %s: T* %.3f, step %d\n", path, sim.time * timeScale, firstStep - 1);
    }
    if (sim.drops && (!sim.markers || sim.pipeline != 2 || sim.volumeControl)) throw std::runtime_error("SPRAY needs PLS and PIPE=2, and no VOLCTL");
    if (sim.volumeControl) {
        if (sim.pipeline != 2) throw std::runtime_error("VOLCTL needs PIPE=2");
        if (firstVolume == 0) firstVolume = volumeOf(sim, sim.snapshot());
        sim.targetVolume = firstVolume;
        std::printf("volume control %d: target %.1f cells, nP %.0f, zeta %.1f\n", sim.volumeControl, sim.targetVolume, sim.controlSteps,
                    sim.controlDamping);
    }
    std::vector<double> saveAt;
    if (const char* v = std::getenv("CKPT_AT"))
        for (const char* c = v; *c;) {
            char* end = nullptr;
            saveAt.push_back(std::strtod(c, &end));
            c = *end ? end + 1 : end;
        }
    size_t nextSave = 0;
    // BENCH=<steps>: run that many steps and print what they cost and a summary of the state they end in.
    if (const char* v = std::getenv("BENCH")) {
        const int    steps = std::atoi(v);
        const double held  = memory.peak();
        for (int i = 0; i < steps; ++i) sim.step(1.0);
        const auto&  st = sim.stats;
        double total = 0, peak = 0;
        for (double ms : st.partMs) total += ms;
        for (double mib : st.partPeak) peak = std::max(peak, mib);
        const char* names[6] = {"advection", "band update", "domain rebuild", "pressure", "extrapolation", "end-of-step rule"};
        std::printf("BENCH %d steps: fluid cells %llu, band %llu, domain %llu voxels\n", steps, (unsigned long long)st.fluidCells,
                    (unsigned long long)st.bandVoxels, (unsigned long long)st.domainVoxels);
        std::printf("  time per step %.1f ms; %.2f pressure solves per step, %.1f iterations per solve\n", total / steps,
                    double(st.solves) / steps, double(st.iterationSum) / st.solves);
        for (int k = 0; k < 6; ++k)
            std::printf("  %-17s %8.1f ms per step (%4.1f%%), peak memory %7.0f MiB\n", names[k], st.partMs[k] / steps,
                        100.0 * st.partMs[k] / total, st.partPeak[k]);
        std::printf("  of the pressure time: multigrid setup %.1f ms, iterations %.1f ms per step\n", st.setupMs / steps, st.solveMs / steps);
        std::printf("  memory: %.0f MiB held between steps before the run, peak %.0f MiB = %.0f bytes per domain voxel\n", held, peak,
                    peak * 1048576.0 / double(st.domainVoxels));
        const auto snap = sim.snapshot();
        double volume = 0, speed = 0;
        for (size_t k = 1; k < snap.coord.size(); ++k) {
            speed = std::max({speed, double(std::fabs(snap.u[0][k])), double(std::fabs(snap.u[1][k])), double(std::fabs(snap.u[2][k]))});
            if (sim.isFree(snap.coord[k])) volume += std::min(1.0, std::max(0.0, 0.5 - double(snap.phi[k])));
        }
        double sums[4] = {0, 0, 0, 0}; // of |u|, |v|, |w| and phi over the domain's voxels
        for (size_t k = 1; k < snap.coord.size(); ++k) {
            for (int a = 0; a < 3; ++a) sums[a] += std::fabs(double(snap.u[a][k]));
            sums[3] += double(snap.phi[k]);
        }
        std::printf("  end state: t %.7f, volume %.1f cells, largest speed %.2f, last divergence %.2e, last residual %.1e\n", sim.time,
                    volume, speed, st.divergence, st.residual);
        std::printf("  end state sums: |u| %.6e, |v| %.6e, |w| %.6e, phi %.6e; invalid %d, released faces %d\n", sums[0], sums[1], sums[2],
                    sums[3], st.invalid, st.releasedFaces);
        return 0;
    }
    for (int i = firstStep; sim.time * timeScale < endTime; ++i) {
        sim.step(1.0);
        if (nextSave < saveAt.size() && sim.time * timeScale >= saveAt[nextSave] && std::getenv("CKPT_SAVE")) {
            char name[512];
            std::snprintf(name, sizeof(name), "%s_%.2f.ckpt", std::getenv("CKPT_SAVE"), saveAt[nextSave]);
            const double extra[3] = {firstVolume, double(i), double(reportIndex)};
            sim.save(name, extra);
            std::printf("checkpoint %s at T* %.3f, step %d\n", name, sim.time * timeScale, i);
            ++nextSave;
        }
        const int statsEvery = std::getenv("STATS") ? std::atoi(std::getenv("STATS")) : 0;
        if (statsEvery ? i % statsEvery == 0 : (sim.stats.missing || sim.stats.invalid)) printStats(i, sim);
        if (sim.time * timeScale < nextReport) continue;
        nextReport += std::getenv("REPORT") ? std::atof(std::getenv("REPORT")) : 0.25; // interval between reports, in T*
        // Front: where phi crosses zero along the floor row; column height: along the wall x = 0. Middle of z.
        const auto snap = sim.snapshot();
        const auto acc  = snap.grid()->getAccessor();
        auto phiAt = [&](int x, int y) {
            const uint64_t idx = acc.getValue(nanovdb::Coord(x, y, nz / 2));
            return idx ? snap.phi[idx] : sim.background;
        };
        double front = 0, top = 0;
        for (int x = 0; x + 1 < nx; ++x)
            if (phiAt(x, 0) < 0 && phiAt(x + 1, 0) >= 0) front = x + 0.5 + phiAt(x, 0) / (phiAt(x, 0) - phiAt(x + 1, 0));
        for (int y = 0; y + 1 < ny; ++y)
            if (phiAt(0, y) < 0 && phiAt(0, y + 1) >= 0) top = y + 0.5 + phiAt(0, y) / (phiAt(0, y) - phiAt(0, y + 1));
        dumpSlice(sim, snap, reportIndex);
        dumpPoints(sim, snap, reportIndex);
        dumpBand(sim, snap, reportIndex);
        dumpSpray(sim, reportIndex);
        ++reportIndex;
        // Volume: each tank cell counts by the fraction a flat surface at distance phi would cut from it.
        float  speed = 0, w = 0;
        int    wAt = -1;
        double volume = 0;
        for (size_t k = 1; k < snap.coord.size(); ++k) {
            if (std::fabs(snap.u[2][k]) > w) w = std::fabs(snap.u[2][k]), wAt = snap.coord[k][2];
            speed = std::max({speed, std::fabs(snap.u[0][k]), std::fabs(snap.u[1][k])});
            if (sim.isFree(snap.coord[k])) volume += std::min(1.0, std::max(0.0, 0.5 - double(snap.phi[k])));
        }
        if (firstVolume == 0) firstVolume = volume;
        if (std::getenv("PEAK")) { // the largest velocity component on a face that touches a fluid cell, and what is around it
            auto fluidAt = [&](const nanovdb::Coord& q) {
                const uint64_t idx = acc.getValue(q);
                return idx && snap.phi[idx] < 0 && sim.isFree(q);
            };
            float  best = -1;
            size_t at = 0;
            int    axis = 0;
            for (size_t k = 1; k < snap.coord.size(); ++k)
                for (int a = 0; a < 3; ++a) {
                    if (std::fabs(snap.u[a][k]) <= best) continue;
                    nanovdb::Coord lower = snap.coord[k];
                    lower[a] -= 1;
                    if (fluidAt(snap.coord[k]) || fluidAt(lower)) best = std::fabs(snap.u[a][k]), at = k, axis = a;
                }
            nanovdb::Coord c = snap.coord[at];
            std::printf("        PEAK: lower %c face of cell (%d, %d, %d), velocity %+.1f.", "xyz"[axis], c[0], c[1], c[2], snap.u[axis][at]);
            if (!fluidAt(c)) c[axis] -= 1; // describe the fluid cell of that face
            std::printf(" Fluid cell (%d, %d, %d): phi %+.3f; neighbors -x +x -y +y -z +z [phi, kind, velocity of the face between]:",
                        c[0], c[1], c[2], snap.phi[acc.getValue(c)]);
            for (int j = 0; j < 6; ++j) {
                nanovdb::Coord q = c;
                q[j >> 1] += (j & 1) ? 1 : -1;
                const uint64_t idx = acc.getValue(q), face = (j & 1) ? idx : acc.getValue(c);
                std::printf(" [%+.3f %s %+.1f]", idx ? snap.phi[idx] : 99.0f, !idx ? "absent" : (!sim.isFree(q) ? "solid" : (snap.phi[idx] < 0 ? "fluid" : "air")),
                            face ? snap.u[j >> 1][face] : 0.0f);
            }
            int fluidNear = 0;
            for (int x = -3; x <= 3; ++x)
                for (int y = -3; y <= 3; ++y)
                    for (int z = -3; z <= 3; ++z) fluidNear += fluidAt(c + nanovdb::Coord(x, y, z));
            std::printf("; fluid cells within 3 of it: %d of 343", fluidNear);
            // The air next to that fluid cell: the size of its connected region of air cells in the grid (6-connected), and
            // whether the region reaches the grid's edge (a cell that is not in the grid: open air).
            for (int j = 0; j < 6; ++j) {
                nanovdb::Coord start = c;
                start[j >> 1] += (j & 1) ? 1 : -1;
                const uint64_t first = acc.getValue(start);
                if (!first || !sim.isFree(start) || snap.phi[first] < 0) continue;
                std::vector<nanovdb::Coord> queue{start};
                std::vector<char>           seen(snap.coord.size(), 0);
                seen[first] = 1;
                bool   open = false;
                for (size_t head = 0; head < queue.size() && queue.size() < 2000000; ++head)
                    for (int k = 0; k < 6; ++k) {
                        nanovdb::Coord q = queue[head];
                        q[k >> 1] += (k & 1) ? 1 : -1;
                        if (!sim.isFree(q)) continue;
                        const uint64_t idx = acc.getValue(q);
                        if (!idx) { open = true; continue; }
                        if (seen[idx] || snap.phi[idx] < 0) continue;
                        seen[idx] = 1, queue.push_back(q);
                    }
                std::printf("; air region at its %s%c neighbor: %zu cells, %s", (j & 1) ? "+" : "-", "xyz"[j >> 1], queue.size(),
                            open ? "reaches open air" : "enclosed by water and walls");
            }
            std::printf("\n");
        }
        std::printf("  %5.2f  %.4f  %8.3f     %8.3f      %10.1f (%+.3f%%)  %.1e (z = %d) / %.1f   [step %d, %d iterations, %.0f ms]\n",
                    sim.time * timeScale, sim.time, front / width, top / height, volume, 100.0 * (volume - firstVolume) / firstVolume,
                    w, wAt, speed, i, sim.stats.iterations, sim.stats.ms);
        if (std::getenv("TIMES")) {
            const auto& st = sim.stats;
            double total = 0;
            for (double v : st.partMs) total += v;
            std::printf("        time per step so far (%ld steps, %.2f pressure solves per step): %.1f ms = advection %.1f + band update %.1f + "
                        "domain rebuild %.1f + pressure %.1f + extrapolation %.1f + end-of-step rule %.1f\n", st.steps,
                        double(st.solves) / st.steps, total / st.steps, st.partMs[0] / st.steps, st.partMs[1] / st.steps,
                        st.partMs[2] / st.steps, st.partMs[3] / st.steps, st.partMs[4] / st.steps, st.partMs[5] / st.steps);
            std::printf("        peak device memory so far, MiB, in the same parts: %.0f, %.0f, %.0f, %.0f, %.0f, %.0f; fluid cells %llu, band %llu, "
                        "domain %llu voxels\n", st.partPeak[0], st.partPeak[1], st.partPeak[2], st.partPeak[3], st.partPeak[4], st.partPeak[5],
                        (unsigned long long)st.fluidCells, (unsigned long long)st.bandVoxels, (unsigned long long)st.domainVoxels);
        }
        if (sim.markers)
            std::printf("        markers: %zu particles (%.0f MiB held); escaped at the last correction %ld positive, %ld negative; so far escaped "
                        "%ld / %ld, sign changes %ld, lost outside the band %ld\n", sim.markers->size(), sim.markers->bytes() / 1048576.0,
                        sim.markers->escapedPositive, sim.markers->escapedNegative, sim.markerEscaped[0], sim.markerEscaped[1],
                        sim.markerSignChanges, sim.markerLost);
        if (sim.drops) {
            const double carried = sim.drops->volume();
            std::printf("        spray: %zu drops carrying %.1f cells (%.3f%% of the water); level set %.1f + drops = %.4f%% of the start; emitted so "
                        "far %.1f, deposited so far %.1f, not assigned %+.1f; new drops in the last step %zu; air markers removed so far %ld\n", sim.drops->size(), carried,
                        100.0 * carried / sim.waterTarget, sim.levelSetVolume, 100.0 * (sim.levelSetVolume + carried) / sim.waterTarget,
                        sim.sprayEmitted, sim.sprayDeposited, sim.sprayMissing, sim.sprayNew, sim.airRemoved);
        }
        if (sim.volumeControl)
            std::printf("        volume control: volume before the pressure %.1f (%+.4f%%), divergence %+.3e per s (%+.3e per step), "
                        "sum of the errors %+.4e, divergence left after the projection %.1e\n", sim.volumeNow, 100.0 * sim.volumeError,
                        sim.controlDivergence, sim.controlDivergence * sim.stats.dt, sim.volumeErrorSum, sim.stats.divergence);
        if (std::getenv("ENERGY")) { // ENERGY=1: the water's energy at every report
            double kinetic, potential;
            energyOf(sim, snap, kinetic, potential);
            if (firstEnergy == 0) firstEnergy = kinetic + potential;
            std::printf("        energy: kinetic %.6e, potential %.6e, total %.6e (%.3f%% of the first total)\n", kinetic, potential,
                        kinetic + potential, 100.0 * (kinetic + potential) / firstEnergy);
        }
        if (sim.diagnose)
            std::printf("        gained so far: advection %+.1f, first rule and dilate %+.1f, wall rule %+.1f, normalize %+.1f, prune %+.1f, "
                        "end-of-step rule %+.1f\n", sim.gained[0], sim.gained[1], sim.gained[2], sim.gained[3], sim.gained[4], sim.gained[5]);
        if (sim.diagnose)
            std::printf("        implied by the velocity's divergence so far: cells deeper than half a voxel %+.1f, surface cells with "
                        "phi < 0 %+.1f, with phi >= 0 next to one with phi < 0 %+.1f, with phi >= 0 and no such neighbor %+.1f; "
                        "water in them now: %.0f, %.0f, %.0f, %.0f\n", sim.implied[0], sim.implied[1], sim.implied[2], sim.implied[3],
                        sim.waterIn[0], sim.waterIn[1], sim.waterIn[2], sim.waterIn[3]);
    }
    return 0;
}

// A film of water under the ceiling of a closed nx x ny x nz tank, over x < coverage * nx. With coverage 1 and
// gravity the exact answer is: nothing moves (the film cannot leave a wall that lets nothing through).
// A rigid rotation about the axis along z through `center`: the displacement over a step of `angle` radians, first
// order in the angle like the solver's own traces.
struct Rotation
{
    advect::PosT  center;
    advect::RealT angle;
    __hostdev__ advect::PosT operator()(const advect::PosT& x) const
    {
        return advect::PosT(-angle * (x[1] - center[1]), angle * (x[0] - center[0]), advect::RealT(0));
    }
};

// rotate <N> <sx> <sy> <sz> <half turns>: a box of water of that size in the middle of an N^3 tank is turned about the
// z axis through its center by the band update alone (no pressure). After every half turn the exact answer is the box
// itself. Prints the volume and the difference from the box.
static int runRotate(int n, double sx, double sy, double sz, int halfTurns)
{
    const Tank   tank{{n, n, n}, extensionSwitch()};
    const double c = 0.5 * (n - 1), size[3] = {sx, sy, sz};
    double       lo[3], hi[3];
    for (int a = 0; a < 3; ++a) lo[a] = c - 0.5 * size[a], hi[a] = c + 0.5 * size[a];
    Simulation sim(tank, lo, hi);
    applySwitches(sim);
    const double radius = 0.5 * std::sqrt(sx * sx + sy * sy);
    const int    steps  = int(std::ceil(3.14159265358979323846 / (sim.cfl / radius)));
    const double angle  = 3.14159265358979323846 / steps;
    const double first  = sim.bandVolume(*sim.band, sim.band->phi.data());
    std::printf("rotate: box %.1f x %.1f x %.1f in a %d^3 tank, %d steps per half turn (largest displacement %.3f voxels); volume %.1f cells "
                "(exact %.1f), difference from the box %.1f\n", sx, sy, sz, n, steps, angle * radius, first, sx * sy * sz,
                sim.differenceFromBox(lo, hi));
    for (int h = 1; h <= halfTurns; ++h) {
        for (int i = 0; i < steps; ++i) sim.transportBand(Rotation{advect::PosT(advect::RealT(c)), advect::RealT(angle)});
        const double v = sim.bandVolume(*sim.band, sim.band->phi.data());
        std::printf("  after %d half turns: volume %.1f (%+.2f%%), difference from the box %.1f (%.1f%% of the volume)", h, v,
                    100.0 * (v - first) / first, sim.differenceFromBox(lo, hi), 100.0 * sim.differenceFromBox(lo, hi) / first);
        if (sim.markers)
            std::printf("; markers %zu, escaped so far %ld positive / %ld negative, sign changes %ld, lost %ld", sim.markers->size(),
                        sim.markerEscaped[0], sim.markerEscaped[1], sim.markerSignChanges, sim.markerLost);
        if (sim.diagnose)
            std::printf("; gained: advection %+.1f, dilate %+.1f, normalize %+.1f, prune %+.1f, markers %+.1f and %+.1f", sim.gained[0],
                        sim.gained[1], sim.gained[3], sim.gained[4], sim.gainedByMarkers[0], sim.gainedByMarkers[1]);
        std::printf("\n");
    }
    return 0;
}

static int runFilm(int nx, int ny, int nz, double thickness, double coverage, double gravity, int steps)
{
    const Tank   tank{{nx, ny, nz}, extensionSwitch()};
    const double lo[3] = {-1, ny - 0.5 - thickness, -1}, hi[3] = {coverage >= 1 ? double(nx) : coverage * nx - 0.5, double(ny), double(nz)};
    Simulation   sim(tank, lo, hi);
    sim.gravity = gravity;
    applySwitches(sim);
    const double first = volumeOf(sim, sim.snapshot());
    double fallY = ny - 0.5 - thickness, fallV = 0; // expected lower surface and velocity of a freely falling film
    std::printf("film: tank %d x %d x %d, thickness %.2f, coverage %.2f, gravity %.1f; volume %.1f cells\n", nx, ny, nz, thickness,
                coverage, gravity, first);
    for (int i = 1; i <= steps; ++i) {
        // AIRDIV=1 (with DIAG=1): before the step, the rate at which the velocity opens up the air cells that hold
        // almost no air (0 <= phi < 0.75): the sum of their divergence. After the step, the volume the advection
        // added. If the level set cannot open those cells, the two agree.
        double thinAir = 0, otherAir = 0;
        const bool airDiv = std::getenv("AIRDIV") != nullptr;
        if (airDiv) {
            const auto snap = sim.snapshot();
            const auto acc  = snap.grid()->getAccessor();
            for (size_t k = 1; k < snap.coord.size(); ++k) {
                const nanovdb::Coord c = snap.coord[k];
                if (!sim.isFree(c) || snap.phi[k] < 0) continue;
                double div = 0;
                for (int a = 0; a < 3; ++a) {
                    nanovdb::Coord up = c;
                    up[a] += 1;
                    const uint64_t n = acc.getValue(up);
                    div += (n ? snap.u[a][n] : 0.0) - snap.u[a][k];
                }
                (snap.phi[k] < 0.75f ? thinAir : otherAir) += div;
            }
        }
        const double before = sim.gained[0];
        sim.step(gravity > 0 ? 1.0 : 0.01);
        // What a film that the ceiling lets go of must do, with the solver's own time steps: the position moves with
        // the velocity of the previous step, then the velocity takes this step's gravity.
        fallY += fallV * sim.stats.dt;
        fallV -= gravity * sim.stats.dt;
        if (airDiv && i % 2 == 0)
            std::printf("  step %4d: velocity opens thin air cells by %+8.2f cells this step (other air cells %+8.2f); advection added %+8.2f\n",
                        i, thinAir * sim.stats.dt, otherAir * sim.stats.dt, sim.gained[0] - before);
        const int every = std::getenv("EVERY") ? std::atoi(std::getenv("EVERY")) : std::max(1, steps / 8);
        if (i % every == 0 || i == 1) {
            const auto   snap = sim.snapshot();
            const double v    = volumeOf(sim, snap);
            if (std::getenv("ENERGY")) {
                double kinetic, potential;
                energyOf(sim, snap, kinetic, potential);
                std::printf("  energy: kinetic %.6e, potential %.6e, total %.6e\n", kinetic, potential, kinetic + potential);
            }
            dumpSlice(sim, snap, i);
            // Volume in each of the six layers nearest the ceiling, as a fraction of the layer.
            double layer[6] = {0, 0, 0, 0, 0, 0};
            for (size_t k = 1; k < snap.coord.size(); ++k) {
                const int below = ny - 1 - snap.coord[k][1];
                if (sim.isFree(snap.coord[k]) && below < 6) layer[below] += std::min(1.0, std::max(0.0, 0.5 - double(snap.phi[k])));
            }
            std::printf("  step %4d t %.4f: volume %9.1f (%+.2f%%), largest speed %.2e, fluid cells %llu | layers from the ceiling:", i,
                        sim.time, v, 100.0 * (v - first) / first, sim.stats.maxSpeed, (unsigned long long)sim.stats.fluidCells);
            for (double l : layer) std::printf(" %.2f", l / (nx * nz));
            if (std::getenv("FALL")) {
                // The film in the column x = nx / 2, z = nz / 2: its two surfaces and the vertical velocity in its cells.
                const auto acc = snap.grid()->getAccessor();
                auto value = [&](int y) { const uint64_t k = acc.getValue(nanovdb::Coord(nx / 2, y, nz / 2)); return k ? snap.phi[k] : sim.background; };
                double bottom = -1, top = -1, vLow = 1e30, vHigh = -1e30;
                for (int y = 0; y < ny; ++y) {
                    const float below = value(y - 1), here = value(y), above = value(y + 1); // wall cells included
                    if (below >= 0 && here < 0) bottom = y - 1 + below / (below - here);
                    if (here < 0 && above >= 0) top = y + here / (here - above);
                    if (here < 0) {
                        const uint64_t k = acc.getValue(nanovdb::Coord(nx / 2, y, nz / 2));
                        vLow = std::min(vLow, double(snap.u[1][k])), vHigh = std::max(vHigh, double(snap.u[1][k]));
                    }
                }
                std::printf("\n      lower surface %.3f (free fall %.3f), upper surface %.3f, thickness %.3f; v in the film %.3f..%.3f (free fall %.3f);"
                            " released faces %d, %d solves", bottom, fallY, top, top - bottom, vLow, vHigh, fallV,
                            sim.stats.releasedFaces, sim.stats.releaseIterations);
            }
            if (std::getenv("FLUX")) {
                // Flow through horizontal planes of faces (the lower faces of the cells of row j), over the tank.
                const auto acc = snap.grid()->getAccessor();
                std::printf("\n      flow through the plane below row j (down, up, net; cells/s):");
                for (int j = ny - 1; j >= ny - 6; --j) {
                    double down = 0, up = 0;
                    for (int x = 0; x < nx; ++x)
                        for (int z = 0; z < nz; ++z) {
                            const uint64_t idx = acc.getValue(nanovdb::Coord(x, j, z));
                            const double   v   = idx ? snap.u[1][idx] : 0.0;
                            (v < 0 ? down : up) += v;
                        }
                    std::printf("  j=%d: %.1f %+.1f = %+.1f", j, down, up, down + up);
                }
            }
            if (std::getenv("FLUX")) {
                // Divergence of the stored velocity in the fluid cells (phi < 0, in the tank), by row.
                const auto acc = snap.grid()->getAccessor();
                std::printf("\n      fluid cells per row, sum and largest |divergence| of the stored velocity:");
                for (int j = ny - 1; j >= ny - 6; --j) {
                    int    count = 0;
                    double sum = 0, largest = 0;
                    for (int x = 0; x < nx; ++x)
                        for (int z = 0; z < nz; ++z) {
                            const nanovdb::Coord c(x, j, z);
                            const uint64_t idx = acc.getValue(c);
                            if (!idx || snap.phi[idx] >= 0) continue;
                            double div = 0;
                            for (int a = 0; a < 3; ++a) {
                                nanovdb::Coord up = c;
                                up[a] += 1;
                                const uint64_t k = acc.getValue(up);
                                div += (k ? snap.u[a][k] : 0.0) - snap.u[a][idx];
                            }
                            ++count, sum += div, largest = std::max(largest, std::fabs(div));
                        }
                    std::printf("  j=%d: %d, %+.2f, %.2e", j, count, sum, largest);
                }
            }
            if (std::getenv("PRINTSTEP") && i == std::atoi(std::getenv("PRINTSTEP"))) {
                // The slice z = nz / 2, rows near the ceiling, left half of the tank: phi, then the vertical velocity
                // on the lower face of each cell, then the horizontal velocity on its left face.
                const auto acc = snap.grid()->getAccessor();
                for (int field = 0; field < 3; ++field) {
                    std::printf("\n      %s (x = 0..%d):", field == 0 ? "phi" : field == 1 ? "v on the lower face" : "u on the left face", nx / 2 - 1);
                    for (int j = ny - 1; j >= ny - 9; --j) {
                        std::printf("\n      y=%2d ", j);
                        for (int x = 0; x < nx / 2; ++x) {
                            const uint64_t idx = acc.getValue(nanovdb::Coord(x, j, nz / 2));
                            if (!idx) { std::printf("    . "); continue; }
                            std::printf(field == 0 ? "%5.1f " : "%5.0f ", field == 0 ? snap.phi[idx] : field == 1 ? snap.u[1][idx] : snap.u[0][idx]);
                        }
                    }
                }
            }
            if (sim.diagnose)
                std::printf(" | gained so far: advection %+.1f, dilate %+.1f, wall rule %+.1f, normalize %+.1f, prune %+.1f",
                            sim.gained[0], sim.gained[1], sim.gained[2], sim.gained[3], sim.gained[4]);
            std::printf("\n");
        }
    }
    return 0;
}

int main(int argc, char** argv)
{
    const std::string mode = argc > 1 ? argv[1] : "";
    try {
        if (mode == "hydrostatic" && argc >= 5) return runHydrostatic(std::atoi(argv[2]), std::atof(argv[3]), std::atoi(argv[4]));
        if (mode == "friction" && argc >= 5) { // friction <n> <speed> <Cf * dt>: fast::wallFriction against its formula
            const int    n = std::atoi(argv[2]);
            const Tank   tank{{n, n, n}, extensionSwitch()};
            const double lo[3] = {-1, -1, -1}, hi[3] = {double(n), double(n), double(n)}; // the tank full of water
            Simulation   sim(tank, lo, hi);
            long         walls[5];
            const double worst = sim.frictionCheck(float(std::atof(argv[3])), float(std::atof(argv[4])), walls);
            std::printf("friction check: faces with 0, 1, 2, 3, 4 walls: %ld %ld %ld %ld %ld; largest error %.3e (speed %s)\n", walls[0], walls[1],
                        walls[2], walls[3], walls[4], worst, argv[3]);
            return 0;
        }
        if (mode == "rotate" && argc >= 7)
            return runRotate(std::atoi(argv[2]), std::atof(argv[3]), std::atof(argv[4]), std::atof(argv[5]), std::atoi(argv[6]));
        if (mode == "film" && argc >= 9)
            return runFilm(std::atoi(argv[2]), std::atoi(argv[3]), std::atoi(argv[4]), std::atof(argv[5]), std::atof(argv[6]),
                           std::atof(argv[7]), std::atoi(argv[8]));
        if (mode == "dam" && argc >= 8)
            return runDam(std::atoi(argv[2]), std::atoi(argv[3]), std::atoi(argv[4]), std::atof(argv[5]), std::atof(argv[6]), std::atof(argv[7]));
    } catch (const std::exception& e) {
        std::printf("EXCEPTION: %s\n", e.what());
        std::printf("  during: %s, after t = %.5f\n", gStage, gTime);
        return 2;
    }
    std::printf("usage: eulerian hydrostatic <N> <height> <steps> | dam <nx> <ny> <nz> <width> <height> <T*>\n");
    return 1;
}
