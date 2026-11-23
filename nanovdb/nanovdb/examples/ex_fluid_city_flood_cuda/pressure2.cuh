// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

// pressure2.cuh: the same linear system and the same method as pressure::Cells::solve (the APIC example's
// multigrid-preconditioned CG: 2x2x2 aggregation, Galerkin coarse matrices, two weighted-Jacobi sweeps before and
// after), stored so that an iteration is cheap:
//   - the unknowns are the fluid cells only, numbered 1 .. fluid cells (0 = "no cell", whose value stays 0);
//   - the six neighbors of every cell, on every level, are looked up once per step and kept in tables, so applying
//     the matrix reads six table entries instead of walking the tree six times;
//   - the conjugate gradient runs in single precision on a correction; the solution and its residual stay in the
//     precision of Real (see Solver::solve);
//   - ordinary kernels (one per operation) instead of one kernel for the whole solve;
//   - the levels are built once per step and reused by the step's solves (the fluid cells do not change between
//     them, only the diagonal), and a solve can start from the previous solution.
// The fine matrix is A = diag - (sum over the fluid neighbors): between two fluid cells the weight is 1 (the face
// is open, pressure.cuh setFluid); air, released wall faces and missing neighbors enter through diag alone, which
// pressure::assemble computes as before.
#pragma once
#include "pressure.cuh"
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/transform_reduce.h>

namespace pressure2 {

using fluid::Coord;
using pressure::Domain;
using pressure::Real;
constexpr float kOmega = 2.0f / 3.0f; // weight of the Jacobi sweeps

// One level, as the kernels see it. Cells are 1 .. n - 1.
struct Level
{
    int    n = 0;
    int*   nbr[6]   = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr}; // -x, +x, -y, +y, -z, +z; 0 = none
    float* wgt[6]   = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr}; // off-diagonal weights; null on the finest level (all 1)
    int*   child[8] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr}; // the finer level's cells of this cell's 2x2x2 block; null on the finest level
    float *diag = nullptr, *x = nullptr, *b = nullptr, *tmp = nullptr;
    // Row i of the matrix times v. v[0] must be 0.
    __device__ float row(const float* v, int i) const
    {
        float y = diag[i] * v[i];
        if (wgt[0]) for (int j = 0; j < 6; ++j) y -= wgt[j][i] * v[nbr[j][i]];
        else for (int j = 0; j < 6; ++j) y -= v[nbr[j][i]];
        return y;
    }
};

__global__ void smoothFromZero(Level l)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < l.n) l.x[i] = i && l.diag[i] > 0 ? kOmega * l.b[i] / l.diag[i] : 0.0f;
}
__global__ void smooth(Level l) // x -> tmp
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < l.n) l.tmp[i] = i && l.diag[i] > 0 ? l.x[i] + kOmega * (l.b[i] - l.row(l.x, i)) / l.diag[i] : 0.0f;
}
__global__ void restrictResidual(Level f, Level c)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= c.n) return;
    float sum = 0;
    if (p)
        for (int s = 0; s < 8; ++s) {
            const int i = c.child[s][p];
            if (i && f.diag[i] > 0) sum += f.b[i] - f.row(f.x, i);
        }
    c.b[p] = sum;
}
// scale: the coarse correction is multiplied by it. A coarse matrix made with a prolongation that is constant over
// each block is too stiff for smooth errors (Braess 1995 scales the correction of such an aggregation method by 1.8).
__global__ void prolong(Level f, Level c, float scale)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (!p || p >= c.n) return;
    for (int s = 0; s < 8; ++s) {
        const int i = c.child[s][p];
        if (i && f.diag[i] > 0) f.x[i] += scale * c.x[p];
    }
}
// Coarse matrix = P^T A P with P constant over each 2x2x2 block: the block's diagonal is the sum of its cells'
// diagonals minus the weights of the faces inside the block; a face that leaves the block adds its weight to the
// block's face in that direction. Child s has bit a set if its coordinate along axis a is odd, so its lower
// neighbor along a is in the same block exactly when the bit is set, and its upper neighbor when it is not.
__global__ void galerkin(Level f, Level c)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= c.n) return;
    float diag = 0, w[6] = {0, 0, 0, 0, 0, 0};
    if (p)
        for (int s = 0; s < 8; ++s) {
            const int i = c.child[s][p];
            if (!i || !(f.diag[i] > 0)) continue;
            float d = f.diag[i];
            for (int j = 0; j < 6; ++j) {
                if (!f.nbr[j][i]) continue;
                const float weight = f.wgt[0] ? f.wgt[j][i] : 1.0f;
                const bool  odd    = (s >> (j / 2)) & 1;
                if ((j & 1) ? !odd : odd) d -= weight;
                else w[j] += weight;
            }
            diag += d;
        }
    c.diag[p] = diag;
    for (int j = 0; j < 6; ++j) c.wgt[j][p] = w[j];
}
// The coarsest level, in one block: Jacobi sweeps from zero.
__global__ void solveCoarsest(Level l, int sweeps)
{
    const int i = threadIdx.x;
    for (int pass = 0; pass < sweeps; ++pass) {
        float v = 0;
        if (i && i < l.n && l.diag[i] > 0) v = pass ? l.x[i] + kOmega * (l.b[i] - l.row(l.x, i)) / l.diag[i] : kOmega * l.b[i] / l.diag[i];
        __syncthreads();
        if (i < l.n) l.x[i] = v;
        __syncthreads();
    }
}

// Setup kernels.
__global__ void markFluid(Domain d, int* flag)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < d.count) flag[i] = i && d.cells[i].fluid ? 1 : 0;
}
// After the running sum of the flags: the number of a fluid cell, 0 for any other cell; and the cell of each number.
__global__ void numberFluid(Domain d, int* fluidOf, int* cellOf)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d.count) return;
    if (i && d.cells[i].fluid) cellOf[fluidOf[i]] = i;
    else fluidOf[i] = 0;
}
// Finest level: the neighbors of each fluid cell, and its block (coordinate / 2) and place in the block.
__global__ void fineTopology(Domain d, const int* fluidOf, const int* cellOf, Level l, Coord* block, unsigned char* place)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= l.n) return;
    if (!f) { for (int j = 0; j < 6; ++j) l.nbr[j][0] = 0; return; }
    const int   i   = cellOf[f];
    const Coord c   = d.coord(i);
    const auto  acc = d.grid->getAccessor();
    for (int j = 0; j < 6; ++j) {
        Coord q = c;
        q[j / 2] += (j & 1) ? 1 : -1;
        l.nbr[j][f] = fluidOf[int(acc.getValue(q))]; // fluidOf[0] = 0
    }
    block[f - 1] = c >> 1;
    place[f - 1] = (unsigned char)((c[0] & 1) | ((c[1] & 1) << 1) | ((c[2] & 1) << 2));
}
// The cells of a finer level register with their blocks, which are the voxels of grid.
__global__ void setChildren(const fluid::Grid* grid, const Coord* block, const unsigned char* place, int fineCount, Level c)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (f >= fineCount) return;
    const auto acc = grid->getAccessor();
    c.child[place[f - 1]][int(acc.getValue(block[f - 1]))] = f;
}
// A coarse level's neighbors, and its own blocks for the next level. One block of threads per leaf.
__global__ void coarseTopology(const fluid::Grid* grid, Level l, Coord* block, unsigned char* place)
{
    const auto& leaf = grid->tree().getFirstNode<0>()[blockIdx.x];
    if (!leaf.isActive(threadIdx.x)) return;
    const int   i   = int(leaf.getValue(threadIdx.x));
    const Coord c   = leaf.offsetToGlobalCoord(threadIdx.x);
    const auto  acc = grid->getAccessor();
    for (int j = 0; j < 6; ++j) {
        Coord q = c;
        q[j / 2] += (j & 1) ? 1 : -1;
        l.nbr[j][i] = int(acc.getValue(q));
    }
    block[i - 1] = c >> 1;
    place[i - 1] = (unsigned char)((c[0] & 1) | ((c[1] & 1) << 1) | ((c[2] & 1) << 2));
}

// Solve kernels on the finest level.
__global__ void gather(const Real* domainValues, const int* cellOf, float* out, int n)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f < n) out[f] = f ? float(domainValues[cellOf[f]]) : 0.0f;
}
// r = b - A q in the precision of Real, q and b over the domain's cells.
__global__ void residual(Level l, const Real* b, const Real* q, const int* cellOf, float* r)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= l.n) return;
    if (!f || !(l.diag[f] > 0)) { r[f] = 0.0f; return; }
    const int i = cellOf[f];
    Real      y = Real(l.diag[f]) * q[i];
    for (int j = 0; j < 6; ++j) {
        const int k = l.nbr[j][f];
        if (k) y -= q[cellOf[k]];
    }
    r[f] = float(b[i] - y);
}
__global__ void addCorrection(const float* e, const int* cellOf, Real* q, int n)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (f < n) q[cellOf[f]] += Real(e[f]);
}
__global__ void applyMatrix(Level l, const float* v, float* out)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f < l.n) out[f] = f && l.diag[f] > 0 ? l.row(v, f) : 0.0f;
}
__global__ void step(float alpha, const float* s, const float* as, float* q, float* r, int n)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f < n) q[f] += alpha * s[f], r[f] -= alpha * as[f];
}
__global__ void direction(float beta, const float* z, float* s, int n)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f < n) s[f] = z[f] + beta * s[f];
}
struct Product
{
    const float *a, *b;
    __device__ double operator()(int i) const { return double(a[i]) * double(b[i]); }
};
// Sum of (a - b)^2 over the domain's cells; b null = sum of a^2. For checks against the other solver.
struct SquaredDifference
{
    const Real *a, *b;
    __device__ double operator()(int i) const
    {
        const double d = double(a[i]) - (b ? double(b[i]) : 0.0);
        return d * d;
    }
};
inline double dot(const float* a, const float* b, int n)
{
    return thrust::transform_reduce(thrust::device, thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(n), Product{a, b}, 0.0,
                                    thrust::plus<double>());
}

class Solver
{
    std::vector<fluid::Buffer<int>>   ints;
    std::vector<fluid::Buffer<float>> floats;
    fluid::Buffer<int>                fluidOf, cellOf;
    fluid::Buffer<float>              e, r, s;
    std::vector<Level>                levels;
    Domain                            d;

    int* newInts(size_t n, bool zero)
    {
        ints.push_back(fluid::allocate<int>(n));
        if (zero) cudaCheck(cudaMemset(ints.back().data(), 0, n * sizeof(int)));
        bytes += n * sizeof(int);
        return ints.back().data();
    }
    float* newFloats(size_t n)
    {
        floats.push_back(fluid::allocate<float>(n));
        cudaCheck(cudaMemset(floats.back().data(), 0, sizeof(float))); // slot 0
        bytes += n * sizeof(float);
        return floats.back().data();
    }
    static int blocks(int n) { return (n + 255) / 256; }

    // One V cycle: levels[0].x = M levels[0].b.
    void cycle()
    {
        const int last = int(levels.size()) - 1;
        for (int k = 0; k < last; ++k) {
            Level& l = levels[k];
            smoothFromZero<<<blocks(l.n), 256>>>(l);
            for (int pass = 1; pass < preSweeps; ++pass) {
                smooth<<<blocks(l.n), 256>>>(l);
                std::swap(l.x, l.tmp);
            }
            restrictResidual<<<blocks(levels[k + 1].n), 256>>>(l, levels[k + 1]);
        }
        Level& coarsest = levels[last];
        if (coarsest.n <= 256) solveCoarsest<<<1, 256>>>(coarsest, 48);
        else {
            smoothFromZero<<<blocks(coarsest.n), 256>>>(coarsest);
            for (int pass = 1; pass < 48; ++pass) {
                smooth<<<blocks(coarsest.n), 256>>>(coarsest);
                std::swap(coarsest.x, coarsest.tmp);
            }
        }
        for (int k = last - 1; k >= 0; --k) {
            Level& l = levels[k];
            prolong<<<blocks(levels[k + 1].n), 256>>>(l, levels[k + 1], correctionScale);
            for (int pass = 0; pass < postSweeps; ++pass) {
                smooth<<<blocks(l.n), 256>>>(l);
                std::swap(l.x, l.tmp);
            }
        }
    }

public:
    float  correctionScale = 1.0f; // of every coarse correction (see prolong)
    int    preSweeps = 2, postSweeps = 2;
    double setupMs = 0;
    size_t bytes   = 0; // held after the constructor
    int    fluidCells = 0;
    int    levelCount() const { return int(levels.size()); }

    // The levels of the fluid cells of d (d.cells[i].fluid), which must not change while this object is used.
    explicit Solver(Domain domain) : fluidOf(fluid::allocate<int>(domain.count)), cellOf(fluid::allocate<int>(1)),
                                     e(fluid::allocate<float>(1)), r(fluid::allocate<float>(1)), s(fluid::allocate<float>(1)), d(domain)
    {
        fluid::Timer timer;
        bytes += size_t(d.count) * sizeof(int);
        markFluid<<<blocks(d.count), 256>>>(d, fluidOf.data());
        thrust::inclusive_scan(thrust::device, fluidOf.data(), fluidOf.data() + d.count, fluidOf.data());
        cudaCheck(cudaMemcpy(&fluidCells, fluidOf.data() + d.count - 1, sizeof(int), cudaMemcpyDeviceToHost));
        cellOf = fluid::allocate<int>(fluidCells + 1);
        bytes += size_t(fluidCells + 1) * sizeof(int);
        cudaCheck(cudaMemset(cellOf.data(), 0, sizeof(int)));
        numberFluid<<<blocks(d.count), 256>>>(d, fluidOf.data(), cellOf.data());

        Level fine;
        fine.n = fluidCells + 1;
        for (int j = 0; j < 6; ++j) fine.nbr[j] = newInts(fine.n, false);
        auto block = fluid::allocate<Coord>(std::max(fluidCells, 1));
        auto place = fluid::allocate<unsigned char>(std::max(fluidCells, 1));
        fineTopology<<<blocks(fine.n), 256>>>(d, fluidOf.data(), cellOf.data(), fine, block.data(), place.data());
        levels.push_back(fine);

        // Coarser levels: the blocks of a level are the voxels of a grid that is only needed to build the tables.
        for (int k = 0; k < 9 && levels[k].n >= 128; ++k) {
            const int fineCount = levels[k].n;
            auto handle = nanovdb::tools::cuda::voxelsToGrid<fluid::Build>(block.data(), fineCount - 1, 1.);
            const fluid::Grid* grid = handle.deviceGrid<fluid::Build>();
            const auto tree = nanovdb::util::cuda::DeviceGridTraits<fluid::Build>::getTreeData(grid);
            Level c;
            c.n = int(tree.mVoxelCount) + 1;
            for (int j = 0; j < 6; ++j) c.nbr[j] = newInts(c.n, true), c.wgt[j] = newFloats(c.n);
            for (int s = 0; s < 8; ++s) c.child[s] = newInts(c.n, true);
            c.diag = newFloats(c.n), c.x = newFloats(c.n), c.b = newFloats(c.n), c.tmp = newFloats(c.n);
            cudaCheck(cudaMemset(c.tmp, 0, sizeof(float)));
            setChildren<<<blocks(fineCount), 256>>>(grid, block.data(), place.data(), fineCount, c);
            auto nextBlock = fluid::allocate<Coord>(c.n - 1);
            auto nextPlace = fluid::allocate<unsigned char>(c.n - 1);
            coarseTopology<<<tree.mNodeCount[0], 512>>>(grid, c, nextBlock.data(), nextPlace.data());
            fluid::sync(); // the grid and the blocks of the finer level are released when this iteration ends
            block = std::move(nextBlock), place = std::move(nextPlace);
            levels.push_back(c);
        }
        fluid::sync();
        setupMs = timer.ms();
    }

    struct Result { int iterations, levels, missing, restarts; double residual, norm, setupMs, solveMs; };

    // Solve A q = b for the current d.cells (open faces, phi). b and q are over the domain's cells; b is zero outside
    // the fluid cells. warm: q holds a first guess (zero outside the fluid cells), otherwise it is set to zero.
    // Stops when |b - A q| <= tolerance |b|.
    //
    // Mixed precision: the solution q and the residual b - A q are computed in the precision of Real (double), and
    // the conjugate gradient works in single precision on the correction e, A e = b - A q, which is added to q
    // when it has reduced that residual by 1e-4 (or stopped improving). A single-precision solution alone cannot
    // bring the residual below about 1e-6 |b| here: rounding q itself already leaves that much.
    Result solve(const Real* b, Real* q, Real minimumTheta, const pressure::Solid& solid, int* missing, bool warm,
                 double tolerance, int maxIterations = 2000)
    {
        fluid::Timer timer;
        Level&    fine = levels[0];
        const int n    = fine.n;
        if (!fine.diag) { // first call: the vectors
            fine.diag = newFloats(n), fine.x = newFloats(n), fine.tmp = newFloats(n);
            e = fluid::allocate<float>(n), r = fluid::allocate<float>(n), s = fluid::allocate<float>(n);
            bytes += 3 * size_t(n) * sizeof(float);
            cudaCheck(cudaMemset(fine.tmp, 0, sizeof(float)));
        }
        {
            auto diag = fluid::allocate<Real>(d.count);
            cudaCheck(cudaMemset(missing, 0, sizeof(int)));
            pressure::assemble<<<blocks(d.count), 256>>>(d, diag.data(), minimumTheta, missing, solid);
            gather<<<blocks(n), 256>>>(diag.data(), cellOf.data(), fine.diag, n);
            fluid::sync();
        }
        for (size_t k = 1; k < levels.size(); ++k) galerkin<<<blocks(levels[k].n), 256>>>(levels[k - 1], levels[k]);

        Result result{0, levelCount(), 0, 0, 0.0, 0.0, setupMs, 0.0};
        if (!warm) cudaCheck(cudaMemset(q, 0, size_t(d.count) * sizeof(Real)));
        gather<<<blocks(n), 256>>>(b, cellOf.data(), r.data(), n);
        result.norm = std::sqrt(dot(r.data(), r.data(), n));
        const double target = std::max(1e-12, result.norm * tolerance);
        double rn = result.norm;
        for (int outer = 0; outer < 40 && result.iterations < maxIterations; ++outer) {
            if (warm || outer) {
                residual<<<blocks(n), 256>>>(fine, b, q, cellOf.data(), r.data());
                rn = std::sqrt(dot(r.data(), r.data(), n));
            }
            if (rn <= target) break;
            result.restarts = outer;
            // A e = r by conjugate gradients, from e = 0, until the recurrence's residual is below the larger of 1e-4 rn
            // and 0.7 target, or has not improved over 8 iterations.
            const double inner = std::max(1e-4 * rn, 0.7 * target);
            cudaCheck(cudaMemset(e.data(), 0, n * sizeof(float)));
            fine.b = r.data();
            this->cycle();
            double rz = dot(r.data(), fine.x, n), best = rn;
            int    sinceBest = 0;
            cudaCheck(cudaMemcpyAsync(s.data(), fine.x, n * sizeof(float), cudaMemcpyDeviceToDevice));
            while (result.iterations < maxIterations) {
                float* as = fine.tmp; // free between two cycles
                applyMatrix<<<blocks(n), 256>>>(fine, s.data(), as);
                const double denom = dot(s.data(), as, n);
                if (!(denom > 0) || !std::isfinite(denom)) throw std::runtime_error("pressure: matrix breakdown");
                step<<<blocks(n), 256>>>(float(rz / denom), s.data(), as, e.data(), r.data(), n);
                const double now = std::sqrt(dot(r.data(), r.data(), n));
                ++result.iterations;
                if (now < 0.9 * best) best = now, sinceBest = 0;
                else if (++sinceBest >= 8) break;
                if (now <= inner) break;
                this->cycle();
                const double next = dot(r.data(), fine.x, n);
                direction<<<blocks(n), 256>>>(float(next / rz), fine.x, s.data(), n);
                rz = next;
            }
            addCorrection<<<blocks(n), 256>>>(e.data(), cellOf.data(), q, n);
        }
        if (rn > target) { // the loop ended on its limits: report the residual of what is returned
            residual<<<blocks(n), 256>>>(fine, b, q, cellOf.data(), r.data());
            rn = std::sqrt(dot(r.data(), r.data(), n));
        }
        result.residual = rn;
        cudaCheck(cudaMemcpy(&result.missing, missing, sizeof(int), cudaMemcpyDeviceToHost));
        result.solveMs = timer.ms();
        return result;
    }
};

} // namespace pressure2
