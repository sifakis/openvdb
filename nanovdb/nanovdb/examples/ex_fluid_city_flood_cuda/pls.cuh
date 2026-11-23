// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

// pls.cuh: marker particles that correct a level set stored as a sidecar of a ValueOnIndex grid.
//
// Method: Enright, Fedkiw, Ferziger, Mitchell 2002, A Hybrid Particle Level Set Method for Improved Interface
// Capturing (J. Comput. Phys. 183), section 2.3. Particles of both signs live within `width` voxels of the surface
// and move with the flow. A particle that is on the wrong side of the surface by more than its radius has escaped:
// the level set lost what the particle still knows. Each escaped particle p carries the sphere
//   phi_p(x) = s_p (r_p - |x - x_p|)                                                      (their eq. 10)
// and the grid values around it are rebuilt as
//   phi+ = max(phi, phi_p of the escaped positive particles), phi- = min(phi, phi_p of the escaped negative ones),
//   phi  = whichever of phi+ and phi- is smaller in magnitude                             (eq. 11 - 13).
// Radii follow r_p = clamp(s_p phi(x_p), rMin, rMax) (eq. 9). Order of a step (their summary in 2.3.4): move the
// level set and the particles, correct, reinitialize, correct, set the radii.
//
// Differences from the paper:
//   - A new particle is placed at a random position of its cell and takes the sign of phi there; the paper then moves
//     it to a randomly chosen isocontour on its side ("attraction"). Here a particle that lands farther than `width`
//     from the surface is dropped instead.
//   - Particles are moved with the midpoint rule (the paper: third-order TVD Runge-Kutta).
//   - The correction writes the grid points up to `reach` beyond the 8 corners of the particle's cell (the paper: the
//     8 corners). With reach 1 every face neighbor of a corner gets a value below the band's width, so that a piece of
//     water the particles rebuild far from the rest stays surrounded by active voxels when the band is pruned.
//   - Reseeding thins a cell with too many particles at random (the paper keeps the ones closest to the surface), and
//     an escaped particle whose cell is not in the grid is removed (the paper never removes escaped particles).
//
// Positions are in index space: the value of voxel ijk sits at ijk. A particle's sign is the sign of its stored r.
#pragma once
#include "advect.cuh"
#include <nanovdb/util/cuda/DeviceGridTraits.cuh>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/iterator/counting_iterator.h>
#include <vector>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

namespace pls {

using advect::GridT;
using advect::PosT;
using advect::RealT;

struct Particle
{
    float x, y, z;
    float r; // > 0: positive particle (air side) of that radius; < 0: negative particle (water side) of radius -r; 0: none
};

struct Settings
{
    int   perCell = 16;   // particles a cell near the surface is filled up to
    float width   = 3.0f; // particles live where |phi| < width
    float rMin    = 0.1f, rMax = 0.5f;
    int   reach   = 1;    // grid points written beyond the 8 corners of an escaped particle's cell, per side
    bool  airCorrects = true; // escaped positive (air) particles rebuild air; false: only the water side is corrected
};

// The level set as the particles read it: its values in the band, the background with the invert masks' sign outside.
struct Field
{
    const GridT*        grid;
    const float*        phi;
    advect::InvertMasks masks;
    __device__ float at(const PosT& x) const
    {
        float lowest, highest;
        return advect::interpolate(advect::SignedChannel{masks}(*grid, phi), x, lowest, highest);
    }
};

__device__ inline uint32_t hash(uint32_t x)
{
    x ^= x >> 16, x *= 0x7feb352dU, x ^= x >> 15, x *= 0x846ca68bU, x ^= x >> 16;
    return x;
}
// A number in [0, 1) for the pair (a, b) of the run with this seed.
__device__ inline float uniform(uint32_t seed, uint32_t a, uint32_t b)
{
    return float(hash(seed ^ hash(a + 0x9e3779b9U * (b + 1u))) >> 8) * (1.0f / 16777216.0f);
}

__device__ inline void atomicMaxFloat(float* address, float value)
{
    int* bits = reinterpret_cast<int*>(address);
    int  old  = *bits;
    while (value > __int_as_float(old)) {
        const int assumed = old;
        old = atomicCAS(bits, assumed, __float_as_int(value));
        if (old == assumed) break;
    }
}
__device__ inline void atomicMinFloat(float* address, float value)
{
    int* bits = reinterpret_cast<int*>(address);
    int  old  = *bits;
    while (value < __int_as_float(old)) {
        const int assumed = old;
        old = atomicCAS(bits, assumed, __float_as_int(value));
        if (old == assumed) break;
    }
}

struct IsNone
{
    __device__ bool operator()(const Particle& p) const { return p.r == 0.0f; }
};

struct IsParticle
{
    __device__ bool operator()(const Particle& p) const { return p.r != 0.0f; }
};

// Storage: fixed-size chunks; a slot with r == 0 is free. Removing a particle frees its slot, a new particle takes a
// free slot, and a chunk is only ever added or dropped as a whole, so no call needs a second copy of the particles.
class Particles
{
public: // a member function that launches a device lambda cannot be private (nvcc)
    static constexpr int    kShift = 23;
    static constexpr size_t kChunk = size_t(1) << kShift; // particles per chunk (128 MiB)
    std::vector<thrust::device_vector<Particle>> chunks;
    size_t alive = 0;
    static int blocks(size_t n) { return int((n + 255) / 256); }

    size_t countChunk(size_t c) const { return size_t(thrust::count_if(thrust::device, chunks[c].begin(), chunks[c].end(), IsParticle())); }
    void   recount()
    {
        alive = 0;
        for (size_t c = 0; c < chunks.size(); ++c) alive += this->countChunk(c);
    }
    // The first `wanted` free slots of the chunks before `chunkEnd` (all of them if there are fewer), as global indices.
    thrust::device_vector<uint32_t> freeSlots(size_t wanted, size_t chunkEnd) const
    {
        thrust::device_vector<uint32_t> slots;
        size_t found = 0;
        for (size_t c = 0; c < chunkEnd && found < wanted; ++c) {
            const size_t free = kChunk - this->countChunk(c);
            if (!free) continue;
            slots.resize(found + free);
            const auto first = thrust::counting_iterator<uint32_t>(uint32_t(c * kChunk));
            thrust::copy_if(thrust::device, first, first + kChunk, chunks[c].begin(), slots.begin() + found, IsNone());
            found += free;
        }
        return slots;
    }
    thrust::device_vector<Particle*> table()
    {
        std::vector<Particle*> host;
        for (auto& c : chunks) host.push_back(c.data().get());
        return thrust::device_vector<Particle*>(host.begin(), host.end());
    }
    // Moves the particles of the last chunk into free slots of the others and drops it, while they fit.
    void defragment()
    {
        while (chunks.size() > 1) {
            const size_t last = chunks.size() - 1, moving = this->countChunk(last);
            if ((chunks.size() - 1) * kChunk - (alive - moving) < moving) break;
            if (moving) {
                thrust::device_vector<Particle> held(moving);
                thrust::copy_if(thrust::device, chunks[last].begin(), chunks[last].end(), held.begin(), IsParticle());
                const thrust::device_vector<uint32_t> slots = this->freeSlots(moving, last);
                const thrust::device_vector<Particle*> pointers = this->table();
                const Particle*  from = held.data().get();
                const uint32_t*  slot = slots.data().get();
                Particle* const* to   = pointers.data().get();
                nanovdb::util::cuda::lambdaKernel<<<blocks(moving), 256>>>(moving, [=] __device__(size_t k) {
                    to[slot[k] >> kShift][slot[k] & (kChunk - 1)] = from[k];
                });
                cudaCheckError();
                cudaCheck(cudaDeviceSynchronize());
            }
            chunks.pop_back();
        }
    }

    Settings settings;
    // Of the last call of each kind.
    long escapedPositive = 0, escapedNegative = 0; // correct(): particles that corrected the level set
    long signChanges     = 0;                      // correct(): grid values whose sign the correction changed
    long added = 0, removed = 0, lost = 0;         // reseed(): lost = escaped particles removed because their cell left the grid

    size_t size() const { return alive; }                                       // particles
    size_t bytes() const { return chunks.size() * kChunk * sizeof(Particle); }  // device memory held

    /// @brief Removes the particles that are no longer useful and fills every cell near the surface up to perCell.
    /// @param solid Functor with bool solid(Coord): cells that hold no particles.
    template <typename SolidT>
    void reseed(const GridT* grid, uint32_t leafCount, size_t valueCount, const float* phi, advect::InvertMasks masks, SolidT solid,
                uint32_t seed)
    {
        using nanovdb::util::cuda::lambdaKernel;
        const Field    field{grid, phi, masks};
        const Settings set = settings;
        const size_t   before = alive;
        thrust::device_vector<int> have(valueCount, 0), counters(1, 0);
        int* haveP = have.data().get(), *count = counters.data().get();
        // The non-escaped particles of each cell; particles that are of no use any more go.
        for (auto& chunk : chunks) {
            Particle* p = chunk.data().get();
            lambdaKernel<<<blocks(kChunk), 256>>>(kChunk, [=] __device__(size_t k) {
                Particle& q = p[k];
                if (q.r == 0.0f) return;
                const nanovdb::Coord cell(int(floorf(q.x + 0.5f)), int(floorf(q.y + 0.5f)), int(floorf(q.z + 0.5f)));
                const uint64_t       index = grid->getAccessor().getValue(cell);
                const float          s = q.r > 0 ? 1.0f : -1.0f, value = s * field.at(PosT(q.x, q.y, q.z));
                const bool           escaped = value < -fabsf(q.r);
                if (!index) { // outside the band: nothing to correct there
                    if (escaped) atomicAdd(count, 1);
                    q.r = 0.0f;
                } else if (escaped) {
                    // kept, and not counted: it marks something the level set cannot hold
                } else if (value > set.width || solid.solid(cell) || fabsf(phi[index]) >= set.width) {
                    q.r = 0.0f;
                } else {
                    atomicAdd(haveP + index, 1);
                }
            });
            cudaCheckError();
        }
        // A cell with too many: each of its non-escaped particles stays with probability perCell / count.
        for (size_t c = 0; c < chunks.size(); ++c) {
            Particle*      p    = chunks[c].data().get();
            const uint32_t base = uint32_t(c * kChunk);
            lambdaKernel<<<blocks(kChunk), 256>>>(kChunk, [=] __device__(size_t k) {
                Particle& q = p[k];
                if (q.r == 0.0f) return;
                const nanovdb::Coord cell(int(floorf(q.x + 0.5f)), int(floorf(q.y + 0.5f)), int(floorf(q.z + 0.5f)));
                const uint64_t       index = grid->getAccessor().getValue(cell);
                if (haveP[index] <= set.perCell) return;
                const float s = q.r > 0 ? 1.0f : -1.0f;
                if (s * field.at(PosT(q.x, q.y, q.z)) < -fabsf(q.r)) return; // escaped
                if (uniform(seed, base + uint32_t(k), 0x51ed27u) * float(haveP[index]) >= float(set.perCell)) q.r = 0.0f;
            });
            cudaCheckError();
        }
        cudaCheck(cudaDeviceSynchronize());
        this->recount();
        removed = long(before - alive);
        lost    = long(counters[0]);
        this->defragment();

        // How many each cell near the surface lacks, and which free slots its new particles take.
        thrust::device_vector<int> need(valueCount, 0);
        int*         needP  = need.data().get();
        const size_t voxels = size_t(leafCount) * 512;
        lambdaKernel<<<blocks(voxels), 256>>>(voxels, [=] __device__(size_t tid) {
            const auto&    leaf = grid->tree().getFirstLeaf()[tid >> 9];
            const uint32_t m    = uint32_t(tid & 511);
            if (!leaf.isActive(m)) return;
            const uint64_t i = leaf.getValue(m);
            if (fabsf(phi[i]) >= set.width || solid.solid(leaf.offsetToGlobalCoord(m))) return;
            needP[i] = set.perCell > haveP[i] ? set.perCell - haveP[i] : 0;
        });
        cudaCheckError();
        thrust::device_vector<int> first(valueCount);
        thrust::exclusive_scan(thrust::device, need.begin(), need.end(), first.begin());
        const size_t extra = size_t(int(first[valueCount - 1]) + int(need[valueCount - 1]));
        have.clear(), have.shrink_to_fit();
        if (!extra) { added = 0; return; }
        while (chunks.size() * kChunk - alive < extra) chunks.emplace_back(kChunk); // zero: every slot free
        const thrust::device_vector<uint32_t>  slots    = this->freeSlots(extra, chunks.size());
        const thrust::device_vector<Particle*> pointers = this->table();
        const int*       firstP = first.data().get();
        const uint32_t*  slot   = slots.data().get();
        Particle* const* to     = pointers.data().get();
        lambdaKernel<<<blocks(voxels), 256>>>(voxels, [=] __device__(size_t tid) {
            const auto&    leaf = grid->tree().getFirstLeaf()[tid >> 9];
            const uint32_t m    = uint32_t(tid & 511);
            if (!leaf.isActive(m)) return;
            const uint64_t i = leaf.getValue(m);
            const nanovdb::Coord ijk = leaf.offsetToGlobalCoord(m);
            for (int k = 0; k < needP[i]; ++k) {
                const uint32_t at = slot[size_t(firstP[i]) + k];
                Particle&      q  = to[at >> kShift][at & (kChunk - 1)];
                const uint32_t id = uint32_t(i) * 131u + uint32_t(k);
                q.x = float(ijk[0]) + uniform(seed, id, 1u) - 0.5f;
                q.y = float(ijk[1]) + uniform(seed, id, 2u) - 0.5f;
                q.z = float(ijk[2]) + uniform(seed, id, 3u) - 0.5f;
                const float value = field.at(PosT(q.x, q.y, q.z));
                const float size  = fminf(fmaxf(fabsf(value), set.rMin), set.rMax);
                q.r = fabsf(value) >= set.width ? 0.0f : (value > 0 ? size : -size);
            }
        });
        cudaCheckError();
        cudaCheck(cudaDeviceSynchronize());
        const size_t kept = alive;
        this->recount();
        added = long(alive - kept);
    }

    /// @brief Moves every particle over one step (midpoint rule).
    /// @param displacement Functor PosT(const PosT& x): displacement over the step at index-space position x, in voxels.
    template <typename DisplacementT>
    void advect(DisplacementT displacement)
    {
        for (auto& chunk : chunks) {
            Particle* p = chunk.data().get();
            nanovdb::util::cuda::lambdaKernel<<<blocks(kChunk), 256>>>(kChunk, [=] __device__(size_t k) {
                Particle& q = p[k];
                if (q.r == 0.0f) return;
                const PosT x(q.x, q.y, q.z);
                const PosT d = displacement(x + displacement(x) * RealT(0.5));
                q.x += float(d[0]), q.y += float(d[1]), q.z += float(d[2]);
            });
            cudaCheckError();
        }
    }

    /// @brief Rebuilds the level set around the escaped particles (eq. 11 - 13). Values of solid cells are not written.
    /// @param phi The level set's values, read through @a masks outside the band and changed in place.
    template <typename SolidT>
    void correct(const GridT* grid, size_t valueCount, float* phi, advect::InvertMasks masks, SolidT solid)
    {
        using nanovdb::util::cuda::lambdaKernel;
        const Field    field{grid, phi, masks};
        const Settings set = settings;
        escapedPositive = escapedNegative = signChanges = 0;
        if (chunks.empty()) return;
        thrust::device_vector<float> plus(valueCount), minus(valueCount);
        thrust::device_vector<int>   counters(3, 0);
        float* up = plus.data().get(), *down = minus.data().get();
        int*   count = counters.data().get();
        cudaCheck(cudaMemcpy(up, phi, valueCount * sizeof(float), cudaMemcpyDeviceToDevice));
        cudaCheck(cudaMemcpy(down, phi, valueCount * sizeof(float), cudaMemcpyDeviceToDevice));
        for (auto& chunk : chunks) {
            const Particle* p = chunk.data().get();
            lambdaKernel<<<blocks(kChunk), 256>>>(kChunk, [=] __device__(size_t k) {
                const Particle q = p[k];
                if (q.r == 0.0f) return;
                const float s = q.r > 0 ? 1.0f : -1.0f, radius = fabsf(q.r);
                if (s * field.at(PosT(q.x, q.y, q.z)) >= -radius) return; // not escaped
                atomicAdd(count + (s > 0 ? 0 : 1), 1);
                if (s > 0 && !set.airCorrects) return;
                const auto           acc = grid->getAccessor();
                const nanovdb::Coord base(int(floorf(q.x)), int(floorf(q.y)), int(floorf(q.z)));
                for (int i = -set.reach; i <= 1 + set.reach; ++i)
                    for (int j = -set.reach; j <= 1 + set.reach; ++j)
                        for (int l = -set.reach; l <= 1 + set.reach; ++l) {
                            const nanovdb::Coord c = base + nanovdb::Coord(i, j, l);
                            const uint64_t       index = acc.getValue(c);
                            if (!index || solid.solid(c)) continue;
                            const float dx = float(c[0]) - q.x, dy = float(c[1]) - q.y, dz = float(c[2]) - q.z;
                            const float value = s * (radius - sqrtf(dx * dx + dy * dy + dz * dz));
                            if (s > 0) atomicMaxFloat(up + index, value);
                            else atomicMinFloat(down + index, value);
                        }
            });
            cudaCheckError();
        }
        lambdaKernel<<<blocks(valueCount), 256>>>(valueCount, [=] __device__(size_t i) {
            if (!i) return;
            const float value = fabsf(up[i]) <= fabsf(down[i]) ? up[i] : down[i];
            if ((value < 0) != (phi[i] < 0)) atomicAdd(count + 2, 1);
            phi[i] = value;
        });
        cudaCheckError();
        escapedPositive = long(counters[0]), escapedNegative = long(counters[1]), signChanges = long(counters[2]);
    }

    /// @brief Removes the positive (air) particles that are still escaped: inside the water by more than their radius.
    /// Call it at the end of a step, after the corrections: what is left then is air the level set cannot hold. Such a
    /// particle would carve its sphere of air out of the water again in every later step (the paper keeps it: there the
    /// air is a fluid too; here it has no pressure, the water closes the hole, and the carving only removes water).
    /// @return The number of particles removed.
    size_t removeEscapedAir(const GridT* grid, const float* phi, advect::InvertMasks masks)
    {
        const Field                field{grid, phi, masks};
        thrust::device_vector<int> counter(1, 0);
        int*                       count = counter.data().get();
        for (auto& chunk : chunks) {
            Particle* p = chunk.data().get();
            nanovdb::util::cuda::lambdaKernel<<<blocks(kChunk), 256>>>(kChunk, [=] __device__(size_t k) {
                Particle& q = p[k];
                if (q.r > 0.0f && field.at(PosT(q.x, q.y, q.z)) < -q.r) q.r = 0.0f, atomicAdd(count, 1);
            });
            cudaCheckError();
        }
        cudaCheck(cudaDeviceSynchronize());
        const size_t removedNow = size_t(int(counter[0]));
        alive -= removedNow;
        return removedNow;
    }

    /// @brief Sets every particle's radius from its distance to the surface (eq. 9).
    void adjustRadii(const GridT* grid, const float* phi, advect::InvertMasks masks)
    {
        const Field    field{grid, phi, masks};
        const Settings set = settings;
        for (auto& chunk : chunks) {
            Particle* p = chunk.data().get();
            nanovdb::util::cuda::lambdaKernel<<<blocks(kChunk), 256>>>(kChunk, [=] __device__(size_t k) {
                Particle& q = p[k];
                if (q.r == 0.0f) return;
                const float s = q.r > 0 ? 1.0f : -1.0f;
                q.r = s * fminf(fmaxf(s * field.at(PosT(q.x, q.y, q.z)), set.rMin), set.rMax);
            });
            cudaCheckError();
        }
    }
};

} // namespace pls
