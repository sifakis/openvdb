// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

// mesh_to_sdf_pile.cu <soup.bin> <voxels per unit> <out prefix> [band=3] [heuristic | interior | ball]
//                     [write band: 0 | 1] [slices]
//
// One tools::cuda::MeshToSDF conversion of a triangle soup written by make_scene.py (uint64 points, uint64 triangles,
// float32 xyz per point, int32 abc per triangle). The voxel size is 1 / <voxels per unit>; NestingRule::Solid.
// With <out prefix> = "-" nothing is written (a measurement only).
// Environment SLICES_ONLY=1: stop after the slices (no .faces file).
// Environment SHIFT_VOXELS="i j k" (integers): the soup is moved by that many voxels before the conversion, and the
// voxel coordinates written to the files are moved back, so the files are in the soup's own frame. For a soup far
// from the origin: SignCertifier (BarrierSigning::Ball) was seen to give wrong signs where a voxel's index exceeds
// 8192, where float32 world coordinates are only good to about 1/1000 voxel.
// Prints the stage timings, the active voxels, the components of the band and the high-water mark of the CUDA
// memory pool; with the files, also the index bounding box of the active voxels in the soup's own frame. Writes
//   <out prefix>.faces  the surface as the voxels see it: every interior voxel (phi < 0) that has an exterior face
//                       neighbour. char[8] "FACES001", uint64 count, double voxel size; then per voxel int16 i, j, k
//                       and a uint8 whose bit f is set when the neighbour in direction f (+x, -x, +y, -y, +z, -z)
//                       is exterior
//   <out prefix>_slice_<n>.bin  one per entry of [slices], a comma-separated list such as "z0.15,z0.05:4,y0.4": the
//                       plane's axis, its coordinate in the soup's units and, after a colon, a stride in voxels
//                       (default 1). The signed distance in voxels of every stride-th voxel of the plane: the value
//                       of an active voxel; elsewhere +band or -band by the sign tools::cuda::PropagateSign gives
//                       the inactive voxels. char[8] "SLICE002", int32 axis, index, lo0, lo1, n0, n1, stride,
//                       double voxel size, float band, int32 shift[3] (SHIFT_VOXELS: the tree's nodes are aligned to
//                       multiples of 8 and 128 of voxel coordinate + shift); then n1 rows of n0 float32, then
//                       n1 rows of n0 uint8: where the sample's sign is stored, 0 = an active voxel, 1 = a bit of
//                       a leaf (one voxel), 2 = a bit of a lower node (8^3 voxels without a leaf), 3 = a bit of an
//                       upper node (128^3 voxels without a lower node), 4 = beyond the tree (for z: rows along y, columns
//                       along x; for y: rows along z, columns along x; for x: rows along z, columns along y)
//   <out prefix>.band   only with [write band] = 1: the voxels within 2 voxels of the surface, the format mesh_band
//                       reads (int32 count, 0, 0, 0; then int16 i, j, k and float32 phi in voxels per voxel)
#include <nanovdb/tools/cuda/MeshToSDF.cuh>
#include <nanovdb/tools/cuda/ConnectedComponents.cuh>
#include <nanovdb/tools/cuda/LevelSetTracker.cuh> // levelset::detail::isInverted
#include <nanovdb/tools/cuda/PropagateSign.cuh>
#include <nanovdb/util/cuda/DeviceGridTraits.cuh>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using BuildT     = nanovdb::ValueOnIndex;
using ConverterT = nanovdb::tools::cuda::MeshToSDF<BuildT>;

// One element per sample of a slice: the signed distance in voxels of the voxel it names.
struct SliceFunctor
{
    __device__ void operator()(size_t tid, const nanovdb::NanoGrid<BuildT>* grid, const float* phi,
                               const nanovdb::Mask<3>* leafMasks, const nanovdb::Mask<4>* lowerMasks,
                               const nanovdb::Mask<5>* upperMasks, const nanovdb::Coord* rootOrigins, uint64_t rootOriginCount,
                               int axis, int index, int lo0, int lo1, int n0, int stride, float background, float invVoxel,
                               float* out, uint8_t* level) const
    {
        using nanovdb::tools::cuda::levelset::detail::isInverted;
        const int a = lo0 + stride * int(tid % uint64_t(n0)), b = lo1 + stride * int(tid / uint64_t(n0));
        const nanovdb::Coord ijk = axis == 2 ? nanovdb::Coord(a, b, index) : axis == 1 ? nanovdb::Coord(a, index, b)
                                                                                    : nanovdb::Coord(index, a, b);
        const uint64_t slot = grid->getAccessor().getValue(ijk); // 0: not an active voxel
        const float value = slot ? phi[slot]
                                 : isInverted(*grid, ijk, leafMasks, lowerMasks, upperMasks, rootOrigins, rootOriginCount)
                                       ? -background : background;
        out[tid] = value * invVoxel;
        // Where the sign comes from: the first level of the descent with no child at ijk, as in isInverted.
        const auto& tree = grid->tree();
        uint8_t where = 0;
        if (!slot) {
            const auto* tile = tree.root().probeTile(ijk);
            if (!tile) where = 4;
            else {
                const auto* upper = tree.root().getChild(tile);
                const uint32_t u = nanovdb::NanoUpper<BuildT>::CoordToOffset(ijk);
                if (!upper->isChild(u)) where = 3;
                else where = upper->getChild(u)->isChild(nanovdb::NanoLower<BuildT>::CoordToOffset(ijk)) ? 1 : 2;
            }
        }
        level[tid] = where;
    }
};

static size_t poolAttribute(cudaMemPoolAttr attribute, bool reset = false)
{
    cudaMemPool_t pool;
    cudaCheck(cudaDeviceGetDefaultMemPool(&pool, 0));
    uint64_t value = 0;
    if (reset) cudaCheck(cudaMemPoolSetAttribute(pool, attribute, &value));
    cudaCheck(cudaMemPoolGetAttribute(pool, attribute, &value));
    return size_t(value);
}

int main(int argc, char** argv)
{
    if (argc < 4) { std::printf("usage: mesh_to_sdf_pile <soup.bin> <voxels per unit> <out prefix> [band=3] [heuristic|interior|ball] [write band: 0|1] [slices]\n"); return 1; }
    const double voxelsPerUnit = std::atof(argv[2]);
    const std::string prefix   = argv[3];
    const float band           = argc > 4 ? float(std::atof(argv[4])) : 3.f;
    const std::string mode     = argc > 5 ? argv[5] : "heuristic";
    const auto signing = mode == "ball" ? ConverterT::BarrierSigning::Ball
                       : mode == "interior" ? ConverterT::BarrierSigning::Interior : ConverterT::BarrierSigning::Heuristic;

    std::FILE* in = std::fopen(argv[1], "rb");
    if (!in) { std::printf("cannot open %s\n", argv[1]); return 1; }
    uint64_t counts[2];
    if (std::fread(counts, 8, 2, in) != 2) return 1;
    std::vector<nanovdb::Vec3f> pts(counts[0]);
    std::vector<nanovdb::Vec3i> tris(counts[1]);
    if (std::fread(pts.data(), 12, pts.size(), in) != pts.size() || std::fread(tris.data(), 12, tris.size(), in) != tris.size()) return 1;
    std::fclose(in);
    const double vs = 1.0 / voxelsPerUnit;
    int shift[3] = {0, 0, 0};
    if (const char* text = std::getenv("SHIFT_VOXELS")) std::sscanf(text, "%d %d %d", &shift[0], &shift[1], &shift[2]);
    for (auto& q : pts)
        for (int a = 0; a < 3; ++a) q[a] = float(double(q[a]) + shift[a] * vs);

    nanovdb::cuda::Buffer<nanovdb::Vec3f> dPts(cudaStream_t(0), pts.size(), nanovdb::cuda::noInit);
    nanovdb::cuda::Buffer<nanovdb::Vec3i> dTris(cudaStream_t(0), tris.size(), nanovdb::cuda::noInit);
    cudaCheck(cudaMemcpy(dPts.data(), pts.data(), dPts.size_bytes(), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dTris.data(), tris.data(), dTris.size_bytes(), cudaMemcpyHostToDevice));
    nanovdb::Map map;
    map.set(vs, nanovdb::Vec3d(0.0), 1.0);
    cudaCheck(cudaDeviceSynchronize());
    poolAttribute(cudaMemPoolAttrUsedMemHigh, true);
    const auto t0 = std::chrono::steady_clock::now();
    ConverterT converter(dPts.data(), uint32_t(pts.size()), dTris.data(), uint32_t(tris.size()), map);
    converter.setNarrowBandWidth(band);
    converter.setBarrierSigning(signing);
    converter.setNestingRule(ConverterT::NestingRule::Solid);
    converter.setVerbose(1);
    auto [handle, sdf] = converter.getHandleAndSDF();
    cudaCheck(cudaDeviceSynchronize());
    const double ms   = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const size_t peak = poolAttribute(cudaMemPoolAttrUsedMemHigh), meshBytes = dPts.size_bytes() + dTris.size_bytes();

    const auto* dGrid = handle.template deviceGrid<BuildT>();
    const uint64_t voxels = nanovdb::util::cuda::DeviceGridTraits<BuildT>::getTreeData(dGrid).mVoxelCount;
    uint64_t components = 0;
    {
        nanovdb::tools::cuda::ConnectedComponents<BuildT> cc(dGrid);
        components = cc.getVoxelLabelsAndCount().second;
    }
    std::printf("\nRESULT %s at %g voxels per unit, band %g, %s: %zu triangles, %llu active voxels, %llu components, "
                "MeshToSDF %.1f ms (start-up of the device code included), pool peak %.1f MiB of which the mesh %.1f MiB\n",
                argv[1], voxelsPerUnit, band, mode.c_str(), tris.size(), (unsigned long long)voxels,
                (unsigned long long)components, ms, peak / 1048576.0, meshBytes / 1048576.0);

    if (prefix == "-") return 0;

    if (argc > 7) { // slices, on the device, before the grid is copied to the host
        const auto tree = nanovdb::util::cuda::DeviceGridTraits<BuildT>::getTreeData(dGrid);
        const float* dPhi = reinterpret_cast<const float*>(sdf.data());
        nanovdb::cuda::Buffer<nanovdb::Mask<3>> leafMasks(cudaStream_t(0), tree.mNodeCount[0], nanovdb::cuda::noInit);
        nanovdb::cuda::Buffer<nanovdb::Mask<4>> lowerMasks(cudaStream_t(0), tree.mNodeCount[1], nanovdb::cuda::noInit);
        nanovdb::cuda::Buffer<nanovdb::Mask<5>> upperMasks(cudaStream_t(0), tree.mNodeCount[2], nanovdb::cuda::noInit);
        const auto rootOrigins = nanovdb::tools::cuda::PropagateSign<BuildT>(dGrid, dPhi)
                                     .propagate(leafMasks.data(), lowerMasks.data(), upperMasks.data()).first;
        float lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};
        for (const auto& q : pts)
            for (int a = 0; a < 3; ++a) lo[a] = std::min(lo[a], q[a]), hi[a] = std::max(hi[a], q[a]);
        int ilo[3], ihi[3];
        for (int a = 0; a < 3; ++a) ilo[a] = int(std::floor(lo[a] / vs)) - int(band) - 3, ihi[a] = int(std::ceil(hi[a] / vs)) + int(band) + 3;
        std::string spec = argv[7];
        int number = 0;
        for (size_t at = 0; at < spec.size(); ++number) {
            const size_t end = std::min(spec.find(',', at), spec.size());
            const std::string item = spec.substr(at, end - at);
            at = end + 1;
            const int axis = item[0] == 'x' ? 0 : item[0] == 'y' ? 1 : 2;
            const size_t colon = item.find(':');
            const int index  = int(std::lround(std::atof(item.substr(1, colon - 1).c_str()) / vs)) + shift[axis]; // in the moved frame
            const int stride = colon == std::string::npos ? 1 : std::max(1, std::atoi(item.substr(colon + 1).c_str()));
            const int a0 = axis == 0 ? 1 : 0, a1 = axis == 2 ? 1 : 2; // the plane's two axes
            const int n0 = (ihi[a0] - ilo[a0]) / stride + 1, n1 = (ihi[a1] - ilo[a1]) / stride + 1;
            const uint64_t samples = uint64_t(n0) * n1;
            nanovdb::cuda::Buffer<float> dOut(cudaStream_t(0), samples, nanovdb::cuda::noInit);
            nanovdb::cuda::Buffer<uint8_t> dLevel(cudaStream_t(0), samples, nanovdb::cuda::noInit);
            static constexpr unsigned int threadsPerBlock = 256;
            nanovdb::util::cuda::lambdaKernel<<<nanovdb::util::cuda::blocksPerGrid(samples, threadsPerBlock), threadsPerBlock>>>(
                samples, SliceFunctor{}, dGrid, dPhi, leafMasks.data(), lowerMasks.data(), upperMasks.data(), rootOrigins.data(),
                uint64_t(rootOrigins.size()), axis, index, ilo[a0], ilo[a1], n0, stride, float(band * vs), float(1.0 / vs), dOut.data(),
                dLevel.data());
            cudaCheckError();
            std::vector<float> out(samples);
            cudaCheck(cudaMemcpy(out.data(), dOut.data(), samples * sizeof(float), cudaMemcpyDeviceToHost));
            std::vector<uint8_t> levels(samples);
            cudaCheck(cudaMemcpy(levels.data(), dLevel.data(), samples, cudaMemcpyDeviceToHost));
            const std::string path = prefix + "_slice_" + std::to_string(number) + ".bin";
            std::FILE* f = std::fopen(path.c_str(), "wb");
            const int32_t header[7] = {axis, index - shift[axis], ilo[a0] - shift[a0], ilo[a1] - shift[a1], n0, n1, stride};
            std::fwrite("SLICE002", 1, 8, f);
            std::fwrite(header, 4, 7, f);
            std::fwrite(&vs, 8, 1, f);
            std::fwrite(&band, 4, 1, f);
            std::fwrite(shift, 4, 3, f);
            std::fwrite(out.data(), 4, samples, f);
            std::fwrite(levels.data(), 1, samples, f);
            std::fclose(f);
            std::printf("slice %s: %c index %d, %d x %d samples every %d voxels -> %s\n", item.c_str(), "xyz"[axis], index, n0, n1, stride, path.c_str());
        }
        if (std::getenv("SLICES_ONLY")) return 0;
    }
    std::vector<char>  gridBytes(handle.buffer().size());
    std::vector<float> phi(sdf.size_bytes() / sizeof(float));
    cudaCheck(cudaMemcpy(gridBytes.data(), handle.buffer().data(), gridBytes.size(), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(phi.data(), sdf.data(), sdf.size_bytes(), cudaMemcpyDeviceToHost));
    const auto* grid = reinterpret_cast<const nanovdb::NanoGrid<BuildT>*>(gridBytes.data());
    const auto& tree = grid->tree();
    const auto  acc  = grid->getAccessor();

    const bool writeBand = argc > 6 && std::atoi(argv[6]) != 0;
    std::vector<char> records, faces;
    uint64_t bandCount = 0, negative = 0, surfaceVoxels = 0, faceCount = 0;
    int boxLo[3] = {INT_MAX, INT_MAX, INT_MAX}, boxHi[3] = {INT_MIN, INT_MIN, INT_MIN}; // of the active voxels
    static const int step[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
    for (uint32_t l = 0; l < tree.nodeCount(0); ++l) {
        const auto& leaf = tree.template getFirstNode<0>()[l];
        for (uint32_t n = 0; n < 512; ++n) {
            if (!leaf.isActive(n)) continue;
            const nanovdb::Coord c = leaf.offsetToGlobalCoord(n);
            const float p = float(phi[leaf.getValue(n)] / vs);
            const int16_t xyz[3] = {int16_t(c[0] - shift[0]), int16_t(c[1] - shift[1]), int16_t(c[2] - shift[2])};
            for (int a = 0; a < 3; ++a) boxLo[a] = std::min(boxLo[a], c[a] - shift[a]), boxHi[a] = std::max(boxHi[a], c[a] - shift[a]);
            if (writeBand && std::fabs(p) < 2.f) {
                records.insert(records.end(), reinterpret_cast<const char*>(xyz), reinterpret_cast<const char*>(xyz) + 6);
                records.insert(records.end(), reinterpret_cast<const char*>(&p), reinterpret_cast<const char*>(&p) + 4);
                ++bandCount;
            }
            if (p >= 0) continue;
            ++negative;
            // An exterior face neighbour of an interior voxel lies in the barrier (within 0.501 voxel of the mesh),
            // and face neighbours' distances differ by at most one voxel: deeper than 1.501 voxels there is none.
            if (p < -1.51f) continue;
            uint8_t mask = 0;
            for (int f = 0; f < 6; ++f) {
                const nanovdb::Coord q(c[0] + step[f][0], c[1] + step[f][1], c[2] + step[f][2]);
                // An interior voxel of the band has no inactive exterior neighbour: the band reaches 3 voxels outward.
                if (acc.isActive(q) && phi[acc.getValue(q)] >= 0) mask |= uint8_t(1u << f), ++faceCount;
            }
            if (!mask) continue;
            faces.insert(faces.end(), reinterpret_cast<const char*>(xyz), reinterpret_cast<const char*>(xyz) + 6);
            faces.push_back(char(mask));
            ++surfaceVoxels;
        }
    }
    {
        std::FILE* out = std::fopen((prefix + ".faces").c_str(), "wb");
        std::fwrite("FACES001", 1, 8, out);
        std::fwrite(&surfaceVoxels, 8, 1, out);
        std::fwrite(&vs, 8, 1, out);
        std::fwrite(faces.data(), 1, faces.size(), out);
        std::fclose(out);
    }
    if (writeBand) {
        const int32_t header[4] = {int32_t(bandCount), 0, 0, 0};
        std::FILE* out = std::fopen((prefix + ".band").c_str(), "wb");
        std::fwrite(header, sizeof(int32_t), 4, out);
        std::fwrite(records.data(), 1, records.size(), out);
        std::fclose(out);
    }
    std::printf("%llu interior voxels of the band; %llu of them have an exterior neighbour, with %llu faces in all -> %s.faces\n",
                (unsigned long long)negative, (unsigned long long)surfaceVoxels, (unsigned long long)faceCount, prefix.c_str());
    std::printf("active voxels: indices %d..%d, %d..%d, %d..%d; bounding box %d x %d x %d voxels\n", boxLo[0], boxHi[0], boxLo[1], boxHi[1],
                boxLo[2], boxHi[2], boxHi[0] - boxLo[0] + 1, boxHi[1] - boxLo[1] + 1, boxHi[2] - boxLo[2] + 1);
    return 0;
}
