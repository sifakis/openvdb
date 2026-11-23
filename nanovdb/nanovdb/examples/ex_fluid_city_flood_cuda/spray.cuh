// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

// spray.cuh: the water a level set cannot hold, carried as drops.
//
// A level set loses a piece of water once it is thinner than about two voxels; the marker particles of pls.cuh that
// were inside it are then left on the air side of the surface for good ("escaped": on the wrong side by more than their
// radius, even after they corrected the level set). Here such a marker becomes a drop, and the drops carry the volume
// the level set has lost, so that the water of the level set plus the water of the drops stays the volume the run
// started with (the idea of Losasso, Talton, Kwatra, Fedkiw 2008, "Two-Way Coupled SPH and Particle Level Set Fluid
// Simulation", with ballistic drops in place of their SPH particles):
//   - emit: every escaped negative marker turns into a drop. The volume the level set is short of at that moment is
//     shared equally among the new drops, at most `largest` per drop; what is left over waits for later drops. If
//     nothing is missing, the new drops carry no water: they only show where the spray is. They move like the others,
//     but end when they come to rest on a solid.
//   - move: a drop falls through still air with quadratic drag, each drop with its own terminal speed (drawn at
//     random when it is emitted, between those of raindrops of about 1 mm and 5 mm). It meets solids one axis at a
//     time: the component into the solid is reversed and scaled by a random factor between 0 and `bounce`, and the
//     solid's Coulomb friction acts on the rest. Without the drag, the random bounce and the small random part of a
//     new drop's velocity (`jitter`), drops that left the same place move as one: all that touched a wall stay in
//     the wall's plane for good, and all reach their highest point on the same line.
//   - deposit: a drop in a fluid cell moves with the water and gives its volume, at the rate volume / mergeTime, to
//     the surface cell of the water above it as a source of the pressure equation (that cell's velocity field gets
//     that much divergence). The rate does not depend on the time step: a whole drop pushed into one cell within one
//     step would give face velocities that grow as the step shrinks. After 4 mergeTime in the water the drop ends,
//     and the 2% of its volume it still has go at once. A drop without water stays in the water as long.
//
// Positions are in index space (the value of voxel ijk sits at ijk), velocities in voxels per second, volumes in
// voxels (cells).
#pragma once
#include "pls.cuh"
#include <thrust/host_vector.h>
#include <thrust/remove.h>
#include <thrust/transform_reduce.h>

namespace spray {

using advect::GridT;
using advect::PosT;
using advect::RealT;

struct Drop
{
    float x, y, z;
    float u, v, w;
    float volume;   // 0: it carries no water; negative: to be removed
    float terminal; // its terminal speed in still air, in m/s (0: no drag)
    float wet;      // seconds it has spent in the water
};

// A pseudo-random sequence from 32 bits: an integer hash with good avalanche ("lowbias32"), and its value in [0, 1).
__hostdev__ inline uint32_t mix(uint32_t h)
{
    h ^= h >> 16, h *= 0x7feb352dU, h ^= h >> 15, h *= 0x846ca68bU, h ^= h >> 16;
    return h;
}
__hostdev__ inline float unit(uint32_t h) { return float(h >> 8) * (1.0f / 16777216.0f); }
__hostdev__ inline uint32_t bitsOf(float f)
{
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    return u;
}
__hostdev__ inline uint32_t seedOf(float x, float y, float z) { return mix(bitsOf(x) ^ mix(bitsOf(y) ^ mix(bitsOf(z)))); }

struct VolumeOf
{
    __device__ double operator()(const Drop& d) const { return double(d.volume); }
};
struct IsEmpty
{
    __device__ bool operator()(const Drop& d) const { return d.volume < 0.0f; }
};

class Drops
{
public: // a member function that launches a device lambda cannot be private (nvcc)
    thrust::device_vector<Drop> drops;
    float largest  = 0.5f; // the most volume a new drop takes
    float friction = 0.3f; // Coulomb friction of the solids
    float mergeTime = 0.1f; // a drop in the water gives up its volume at the rate volume / mergeTime, and ends after 4 mergeTime
    float terminalLow = 4.0f, terminalHigh = 9.0f; // m/s: a new drop's terminal speed is drawn between them (0 0: no drag)
    float bounce = 0.25f; // a drop leaves a solid with at most this share of the speed it arrived with
    float jitter = 0.1f;  // a new drop's velocity differs from the water's by at most this share of its speed, per axis
    static int blocks(size_t n) { return int((n + 255) / 256); }

    size_t size() const { return drops.size(); }
    double volume() const { return thrust::transform_reduce(thrust::device, drops.begin(), drops.end(), VolumeOf(), 0.0, thrust::plus<double>()); }

    /// @brief Turns the escaped negative markers into drops that share `missing` (the volume the level set is short of).
    /// @param band, bandPhi, masks The level set the markers belong to.
    /// @param domain, u The grid and the staggered velocity (voxels per second) a new drop takes its velocity from.
    /// @return The volume the new drops carry; `count` is their number.
    double emit(pls::Particles& markers, const GridT* band, const float* bandPhi, advect::InvertMasks masks, const GridT* domain,
                const float* const u[3], double missing, size_t& count)
    {
        using nanovdb::util::cuda::lambdaKernel;
        const pls::Field field{band, bandPhi, masks};
        count = 0;
        if (markers.chunks.empty()) return 0;
        // Escaped: a negative marker (r < 0) where the level set is positive by more than its radius.
        thrust::device_vector<int> counter(1, 0);
        int*                       total = counter.data().get();
        for (auto& chunk : markers.chunks) {
            const pls::Particle* p = chunk.data().get();
            lambdaKernel<<<blocks(pls::Particles::kChunk), 256>>>(pls::Particles::kChunk, [=] __device__(size_t k) {
                const pls::Particle q = p[k];
                if (q.r < 0.0f && field.at(PosT(q.x, q.y, q.z)) > -q.r) atomicAdd(total, 1);
            });
            cudaCheckError();
        }
        count = size_t(int(counter[0]));
        if (!count) return 0;
        const float                     share = float(std::max(0.0, std::min(double(largest), missing / double(count))));
        const size_t                    before = drops.size();
        drops.resize(before + count);
        Drop*                           out = drops.data().get() + before;
        const advect::StaggeredVelocity velocity{domain, {u[0], u[1], u[2]}, RealT(1)};
        const float                     low = terminalLow, high = terminalHigh, spread = jitter;
        counter[0] = 0;
        for (auto& chunk : markers.chunks) {
            pls::Particle* p = chunk.data().get();
            lambdaKernel<<<blocks(pls::Particles::kChunk), 256>>>(pls::Particles::kChunk, [=] __device__(size_t k) {
                pls::Particle& q = p[k];
                if (!(q.r < 0.0f && field.at(PosT(q.x, q.y, q.z)) > -q.r)) return;
                const PosT x(q.x, q.y, q.z);
                const PosT v = velocity(x);
                float      w[3] = {float(v[0]), float(v[1]), float(v[2])};
                const float speed = sqrtf(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
                uint32_t    h = seedOf(q.x, q.y, q.z);
                for (int a = 0; a < 3; ++a) w[a] += spread * speed * (2.0f * unit(h = mix(h + 1u)) - 1.0f);
                const float terminal = low + (high - low) * unit(h = mix(h + 1u));
                out[atomicAdd(total, 1)] = Drop{q.x, q.y, q.z, w[0], w[1], w[2], share, terminal, 0.0f};
                q.r = 0.0f; // the marker's slot is free again
            });
            cudaCheckError();
        }
        cudaCheck(cudaDeviceSynchronize());
        markers.recount();
        return double(share) * double(count);
    }

    /// @brief Moves the drops over one step and deposits the ones that are in fluid cells.
    /// @param domain, domainPhi The grid of the pressure solve and the level set on it: a cell is fluid if its value is
    ///        negative and it is free of solids.
    /// @param u The staggered velocity on @a domain (voxels per second): a drop in the water moves with it.
    /// @param solid Functor with bool solid(Coord).
    /// @param gravity In voxels per second squared; it is taken to be 9.81 m/s^2, which gives the terminal speeds in
    ///        voxels per second.
    /// @param source Per voxel of @a domain: the volume per second deposited there is added (cells per second).
    /// @return The volume deposited.
    template <typename SolidT>
    double step(double dt, double gravity, const GridT* domain, const float* domainPhi, const float* const u[3], SolidT solid, float* source)
    {
        if (drops.empty()) return 0;
        const double before = this->volume();
        Drop*        p      = drops.data().get();
        const float  step = float(dt), fall = float(gravity * dt), mu = friction, merge = mergeTime, most = bounce;
        const float  perMeter = float(gravity / 9.81); // voxels per meter
        const advect::StaggeredVelocity velocity{domain, {u[0], u[1], u[2]}, RealT(1)};
        nanovdb::util::cuda::lambdaKernel<<<blocks(drops.size()), 256>>>(drops.size(), [=] __device__(size_t k) {
            Drop& d = p[k];
            d.v -= fall;
            if (d.terminal > 0.0f) { // drag g |v| v / terminal^2 against the velocity, implicit in the speed's factor
                const float limit = d.terminal * perMeter;
                const float slow = 1.0f / (1.0f + fall * sqrtf(d.u * d.u + d.v * d.v + d.w * d.w) / (limit * limit));
                d.u *= slow, d.v *= slow, d.w *= slow;
            }
            uint32_t h = seedOf(d.x, d.y, d.z);
            float  x[3] = {d.x, d.y, d.z};
            float* v[3] = {&d.u, &d.v, &d.w};
            for (int a = 0; a < 3; ++a) { // one axis at a time: a move that would end in a solid is not made
                float moved[3] = {x[0], x[1], x[2]};
                moved[a] += *v[a] * step;
                const nanovdb::Coord cell(int(floorf(moved[0] + 0.5f)), int(floorf(moved[1] + 0.5f)), int(floorf(moved[2] + 0.5f)));
                if (!solid.solid(cell)) { x[a] = moved[a]; continue; }
                const float normal = fabsf(*v[a]);
                const float back = most * unit(h = mix(h + 1u)) * normal; // under two steps of gravity: it stays
                *v[a] = back > 2.0f * fall ? (*v[a] > 0.0f ? -back : back) : 0.0f;
                const int   b = (a + 1) % 3, c = (a + 2) % 3;
                const float tangent = sqrtf(*v[b] * *v[b] + *v[c] * *v[c]);
                const float keep = tangent > 0.0f ? fmaxf(0.0f, 1.0f - mu * (normal + fabsf(*v[a])) / tangent) : 0.0f;
                *v[b] *= keep, *v[c] *= keep;
            }
            d.x = x[0], d.y = x[1], d.z = x[2];
            // In a fluid cell: the drop's volume goes to the surface cell of the water column above it (the topmost
            // fluid cell, with air over it). That cell has a free-surface face, so its row of the pressure system is
            // never part of a block without a boundary condition, which a source would make unsolvable. A drop under
            // a column that ends at a solid, or that is taller than 32 cells, keeps its volume and moves on.
            const auto     acc = domain->getAccessor();
            nanovdb::Coord cell(int(floorf(d.x + 0.5f)), int(floorf(d.y + 0.5f)), int(floorf(d.z + 0.5f)));
            uint64_t       index = acc.getValue(cell);
            if (!index || domainPhi[index] >= 0.0f || solid.solid(cell)) {
                if (d.volume == 0.0f && d.u == 0.0f && d.v == 0.0f && d.w == 0.0f) d.volume = -1.0f; // at rest, without water
                return;
            }
            for (int k = 0; k < 32; ++k) {
                nanovdb::Coord above = cell;
                above[1] += 1;
                const uint64_t next = acc.getValue(above);
                if (!next || solid.solid(above)) return;
                if (domainPhi[next] >= 0.0f) {
                    d.wet += step;
                    const bool  last = d.wet >= 4.0f * merge;
                    const float given = last ? d.volume : d.volume * fminf(1.0f, step / merge);
                    atomicAdd(source + index, given / step);
                    d.volume = last ? -1.0f : d.volume - given;
                    const PosT w = velocity(PosT(d.x, d.y, d.z)); // carried by the water while it merges
                    d.u = float(w[0]), d.v = float(w[1]), d.w = float(w[2]);
                    return;
                }
                cell = above, index = next;
            }
        });
        cudaCheckError();
        cudaCheck(cudaDeviceSynchronize());
        drops.erase(thrust::remove_if(thrust::device, drops.begin(), drops.end(), IsEmpty()), drops.end());
        return before - this->volume();
    }
};

} // namespace spray
