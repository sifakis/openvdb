// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file  connected_components_cuda.cpp
///
/// @brief Host driver for the connected-components example (NanoVDB / CUDA only,
///        no OpenVDB). Reads a triangle mesh from a Wavefront .obj file, builds the
///        index<->world transform, and hands the mesh off to the CUDA side, which
///        voxelizes it into a ValueOnIndex grid + UDF sidecar. Connected-components
///        labeling on top of that grid will be added as a subsequent step.
///
///        See MeshToSDFDevelopmentPlan.md in this directory for the design notes and roadmap.

#include <nanovdb/NanoVDB.h>          // host-usable: Vec3f, Vec3i, Vec3d, Map
#include <nanovdb/GridHandle.h>       // GridHandle (header-only, host-usable)
#include <nanovdb/cuda/DeviceBuffer.h>// nanovdb::cuda::DeviceBuffer

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// Types of the rasterization result handed across the host/device seam.
using GridHandleT   = nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>;
using UDFSidecarT   = nanovdb::cuda::DeviceBuffer;
using IndexSidecarT = nanovdb::cuda::DeviceBuffer;

// Summary of computeCC's validators (definition must match connected_components_cuda_kernels.cu).
struct CCResult {
    uint64_t globalComponents            = 0;
    bool     openvdbChecked              = false;
    uint64_t confidentSignMismatches     = 0;
    uint64_t inShellTies                 = 0;
    bool     analyticChecked             = false;
    uint64_t analyticConfidentMismatches = 0;
    uint64_t analyticInShellTies         = 0;
};

/// @brief Implemented on the CUDA side (connected_components_cuda_kernels.cu):
///        uploads the mesh and voxelizes it into a ValueOnIndex grid plus a per-active-
///        voxel unsigned-distance-field sidecar.
/// @return { index-grid handle, UDF sidecar } (device buffers)
std::pair<GridHandleT, UDFSidecarT> computeUDF(
    const std::vector<nanovdb::Vec3f>& points,
    const std::vector<nanovdb::Vec3i>& triangles,
    const nanovdb::Map&                map,
    float                              bandWidth);

/// @brief Implemented on the CUDA side: like computeUDF, but also returns a per-active-voxel
///        nearest-triangle-index sidecar (uint32; 0xFFFFFFFF for background / no-hit) needed by the
///        later barrier-signing step. A built-in CPU oracle validates the index against the UDF.
/// @return { index-grid handle, UDF sidecar, nearest-triangle-index sidecar } (device buffers)
std::tuple<GridHandleT, UDFSidecarT, IndexSidecarT> computeUDFAndIndex(
    const std::vector<nanovdb::Vec3f>& points,
    const std::vector<nanovdb::Vec3i>& triangles,
    const nanovdb::Map&                map,
    float                              bandWidth);

/// @brief Implemented on the CUDA side: prints topology diagnostics for the device-
///        resident index grid (active voxels, node counts, bbox, occupancy, memory).
void printGridDiagnostics(const GridHandleT& handle, const std::string& title);

/// @brief Implemented on the CUDA side: derive the connected-components input topology by
///        pruning the surface/barrier shell (voxels within √3/2 voxels of the surface) from
///        the rasterized index grid via PruneGrid. Returns a clean, topology-only index grid.
///        voxelSize converts the √3/2-voxel barrier into the sidecar's world-space units.
GridHandleT computeDerivedTopology(const GridHandleT& srcHandle, const UDFSidecarT& udfSidecar,
                                   float voxelSize);

/// @brief Implemented on the CUDA side: run connected-components labeling + signing on the derived
///        (barrier-pruned) grid, inject the per-voxel signs back onto the original grid, then sign the
///        remaining barrier voxels (step 5) so the original grid's sign field is complete.
/// @param origHandle    the original (pre-prune) UDF index grid — injection + barrier-signing target.
/// @param derivedHandle the barrier-pruned grid CC + signing run on.
/// @param indexSidecar  the original grid's nearest-triangle-index sidecar (for barrier signing).
/// @param points,triangles,map  the source mesh + transform (re-uploaded for barrier signing).
/// @param bandWidth     narrow-band width (voxels); also the half-width for the OpenVDB sign cross-check.
/// @param analyticSpheres optional numSpheres × {Cx,Cy,Cz,R} (world) enabling the analytic union sign
///                        check (inside iff inside any ball: min_i(|p-Ci|-Ri)); else nullptr.
CCResult computeCC(const GridHandleT& origHandle, const GridHandleT& derivedHandle,
               const IndexSidecarT&               indexSidecar,
               const std::vector<nanovdb::Vec3f>& points,
               const std::vector<nanovdb::Vec3i>& triangles,
               const nanovdb::Map&                map,
               float                              bandWidth,
               const double*                      analyticSpheres = nullptr,
               int                                numAnalyticSpheres = 0);

/// @brief Minimal Wavefront .obj reader (vertices + faces) using NanoVDB types.
///
///        Polygons with more than 3 vertices are fan-triangulated. Vertex references
///        of the form `v`, `v/vt`, `v//vn`, `v/vt/vn` are accepted, as are negative
///        (relative) indices. Lines that are not `v` or `f` are ignored.
static void readOBJ(const std::string&            filename,
                    std::vector<nanovdb::Vec3f>&  points,
                    std::vector<nanovdb::Vec3i>&  triangles)
{
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Failed to open OBJ file: " + filename);

    std::string line;
    int lineNumber = 0;
    while (std::getline(file, line)) {
        ++lineNumber;
        std::istringstream iss(line);
        std::string type;
        iss >> type;

        if (type == "v") {
            float x, y, z;
            iss >> x >> y >> z;
            points.emplace_back(x, y, z);
        } else if (type == "f") {
            std::vector<int> face;
            std::string vert;
            while (iss >> vert) {
                const size_t slash = vert.find('/');
                const std::string idxStr = vert.substr(0, slash);
                if (idxStr.empty()) continue;
                int raw = std::stoi(idxStr);
                // OBJ indices are 1-based; negatives are relative to points read so far.
                int idx = (raw < 0) ? int(points.size()) + raw : raw - 1;
                if (idx < 0 || idx >= int(points.size()))
                    throw std::runtime_error("OBJ parse error on line " +
                                             std::to_string(lineNumber) +
                                             ": face index out of bounds");
                face.push_back(idx);
            }
            for (size_t i = 2; i < face.size(); ++i)
                triangles.emplace_back(face[0], face[i - 1], face[i]);
        }
    }
}

// ---------------------------------------------------------------------------------------------------
// In-code analytic mesh generators (no .obj files) for the mirror-faithfulness self-tests.
// ---------------------------------------------------------------------------------------------------

/// @brief Axis-aligned cube, side 2*halfWorld, centered at the origin then translated by
///        +0.5*voxelSize on every axis so no face lies on an integer grid plane (avoids on-plane
///        dot ties). halfWorld is snapped to an integer number of voxels so the faces land exactly
///        on half-voxel index planes (between voxel centers). 8 verts, 12 triangles, watertight.
static void makeCube(float voxelSize, float halfWorld,
                     std::vector<nanovdb::Vec3f>& P, std::vector<nanovdb::Vec3i>& T)
{
    const float h = std::round(halfWorld / voxelSize) * voxelSize;  // integer voxels
    const float s = 0.5f * voxelSize;                               // fractional-voxel shift
    const float c[8][3] = {{-h,-h,-h},{ h,-h,-h},{ h, h,-h},{-h, h,-h},
                           {-h,-h, h},{ h,-h, h},{ h, h, h},{-h, h, h}};
    for (auto& v : c) P.emplace_back(v[0] + s, v[1] + s, v[2] + s);
    const int f[12][3] = {{0,1,2},{0,2,3}, {4,6,5},{4,7,6}, {0,4,5},{0,5,1},
                          {1,5,6},{1,6,2}, {2,6,7},{2,7,3}, {3,7,4},{3,4,0}};
    for (auto& t : f) T.emplace_back(t[0], t[1], t[2]);
}

/// @brief UV sphere of radius R centered at C, tessellated nLat x nLon. With nLat large the facet
///        (chord) error R*(pi/nLat)^2/2 is << voxelSize. Watertight (two pole fans + quad rings).
static void makeUVSphere(nanovdb::Vec3f C, float R, int nLat, int nLon,
                         std::vector<nanovdb::Vec3f>& P, std::vector<nanovdb::Vec3i>& T)
{
    const float PI   = 3.14159265358979323846f;
    const int   base = int(P.size());                           // composable: append after existing verts
    const int   north = base;
    P.emplace_back(C[0], C[1], C[2] + R);                       // north pole
    for (int i = 1; i < nLat; ++i) {                            // interior rings
        const float theta = PI * float(i) / float(nLat);
        const float st = std::sin(theta), ct = std::cos(theta);
        for (int j = 0; j < nLon; ++j) {
            const float phi = 2.0f * PI * float(j) / float(nLon);
            P.emplace_back(C[0] + R * st * std::cos(phi),
                           C[1] + R * st * std::sin(phi),
                           C[2] + R * ct);
        }
    }
    const int south = int(P.size());
    P.emplace_back(C[0], C[1], C[2] - R);                       // south pole
    auto ring = [&](int r, int j) { return base + 1 + r * nLon + (j % nLon); };  // r in [0, nLat-2]
    for (int j = 0; j < nLon; ++j)                              // north cap
        T.emplace_back(north, ring(0, j), ring(0, j + 1));
    for (int r = 0; r < nLat - 2; ++r)                          // quad rings -> 2 tris each
        for (int j = 0; j < nLon; ++j) {
            T.emplace_back(ring(r, j),     ring(r + 1, j), ring(r + 1, j + 1));
            T.emplace_back(ring(r, j),     ring(r + 1, j + 1), ring(r, j + 1));
        }
    for (int j = 0; j < nLon; ++j)                              // south cap
        T.emplace_back(south, ring(nLat - 2, j + 1), ring(nLat - 2, j));
}

/// @brief Run the full mesh->SDF pipeline (rasterize -> prune -> CC + sign + cross-check) on an
///        in-memory mesh and return computeCC's validator summary.
static CCResult runPipeline(const std::string& name,
                            const std::vector<nanovdb::Vec3f>& points,
                            const std::vector<nanovdb::Vec3i>& triangles,
                            float voxelSize, float bandWidth,
                            const double* analyticSpheres, int numAnalyticSpheres)
{
    std::cout << "\n================ " << name << " : " << points.size() << " verts, "
              << triangles.size() << " tris (voxelSize=" << voxelSize
              << ", bandWidth=" << bandWidth << ") ================\n";
    nanovdb::Map map;
    map.set(double(voxelSize), nanovdb::Vec3d(0.0), 1.0);
    auto [handle, sidecar, indexSidecar] = computeUDFAndIndex(points, triangles, map, bandWidth);
    printGridDiagnostics(handle, name + " UDF grid");
    auto derivedHandle = computeDerivedTopology(handle, sidecar, voxelSize);
    return computeCC(handle, derivedHandle, indexSidecar, points, triangles, map, bandWidth,
                     analyticSpheres, numAnalyticSpheres);
}

/// @brief Run the in-code analytic self-tests (cube + sphere). Returns the number of failed checks.
static int runSelfTests(const std::string& which, float voxelSize, float bandWidth)
{
    int failures = 0;
    auto check = [&](const char* label, bool ok) {
        std::cout << "  [" << (ok ? "PASS" : "FAIL") << "] " << label << "\n";
        if (!ok) ++failures;
    };

    if (which == "--cube" || which == "--selftest") {
        std::vector<nanovdb::Vec3f> P; std::vector<nanovdb::Vec3i> T;
        makeCube(voxelSize, 15.0f * voxelSize, P, T);          // half-size ~15 voxels
        const CCResult r = runPipeline("CUBE", P, T, voxelSize, bandWidth, nullptr, 0);
        std::cout << "  cube assertions:\n";
        check("exactly 2 CC global components", r.globalComponents == 2);
        if (r.openvdbChecked) check("0 confident-region OpenVDB sign mismatches", r.confidentSignMismatches == 0);
    }

    if (which == "--sphere" || which == "--selftest") {
        std::vector<nanovdb::Vec3f> P; std::vector<nanovdb::Vec3i> T;
        const float s = 0.5f * voxelSize;
        const nanovdb::Vec3f C(s, s, s);                       // fractional-voxel center
        const float R = 20.0f * voxelSize;                     // radius ~20 voxels
        makeUVSphere(C, R, 128, 256, P, T);
        const double sphere[4] = { double(C[0]), double(C[1]), double(C[2]), double(R) };
        const CCResult r = runPipeline("SPHERE", P, T, voxelSize, bandWidth, sphere, 1);
        std::cout << "  sphere assertions:\n";
        check("exactly 2 CC global components", r.globalComponents == 2);
        if (r.openvdbChecked) check("0 confident-region OpenVDB sign mismatches", r.confidentSignMismatches == 0);
        if (r.analyticChecked) check("0 confident-region analytic sign mismatches", r.analyticConfidentMismatches == 0);
    }

    // Probe of the single-seed exterior rule (global min-x component = exterior, all others = interior):
    // two well-separated spheres. The 2nd sphere's OUTER shell is its own component, not the min-x one,
    // so the single-seed rule would mislabel it interior. This is a DIAGNOSTIC — it only reports, it
    // does NOT gate the pass/fail count.
    if (which == "--two-spheres") {
        std::vector<nanovdb::Vec3f> P; std::vector<nanovdb::Vec3i> T;
        const float  s   = 0.5f * voxelSize;                   // fractional-voxel offset
        const float  R1  = 20.0f * voxelSize, R2 = 15.0f * voxelSize;      // distinct radii
        const float  gap = 30.0f * voxelSize;                  // surface gap >> bandWidth (band ~3 vox)
        const nanovdb::Vec3f C1(s, s, s);
        const nanovdb::Vec3f C2(s + R1 + gap + R2, s, s);      // separated along +x
        makeUVSphere(C1, R1, 128, 256, P, T);
        makeUVSphere(C2, R2, 128, 256, P, T);                  // appended (composable indices)
        const double spheres[8] = { double(C1[0]), double(C1[1]), double(C1[2]), double(R1),
                                    double(C2[0]), double(C2[1]), double(C2[2]), double(R2) };
        const CCResult r = runPipeline("TWO-SPHERES", P, T, voxelSize, bandWidth, spheres, 2);

        std::cout << "  two-sphere probe (single-seed exterior rule), REPORT ONLY:\n";
        std::cout << "    CC global components            : " << r.globalComponents << " (expected 4)\n";
        if (r.openvdbChecked)
            std::cout << "    OpenVDB confident mismatches    : " << r.confidentSignMismatches << "\n";
        if (r.analyticChecked)
            std::cout << "    analytic-union confident mismatch: " << r.analyticConfidentMismatches << "\n";
        const uint64_t worst = std::max(r.confidentSignMismatches, r.analyticConfidentMismatches);
        if (worst == 0)
            std::cout << "    => single-seed rule happens to be FINE here (0 confident mismatches).\n";
        else
            std::cout << "    => LIMITATION CONFIRMED: a disconnected object's outer shell is mislabeled\n"
                      << "       interior (" << worst << " confident mismatches). The single min-x seed only\n"
                      << "       marks ONE exterior component; a multi-seed rule (every component touching the\n"
                      << "       domain boundary = exterior) would be needed.\n";
    }

    std::cout << "\nSelf-tests: " << (failures == 0 ? "ALL PASS" : "FAILED")
              << " (" << failures << " failed checks)\n";
    return failures;
}

int main(int argc, char* argv[])
{
    try {
        if (argc < 2)
            throw std::runtime_error("usage: " + std::string(argv[0]) +
                                     " <input.obj | --cube | --sphere | --selftest | --two-spheres>"
                                     " [voxelSize] [bandWidth]");

        const std::string arg1 = argv[1];

        // In-code analytic self-tests / probes (no .obj). Default voxelSize 0.02.
        if (arg1 == "--cube" || arg1 == "--sphere" || arg1 == "--selftest" || arg1 == "--two-spheres") {
            const float vs = (argc > 2) ? std::stof(argv[2]) : 0.02f;
            const float bw = (argc > 3) ? std::stof(argv[3]) : 3.0f;
            return runSelfTests(arg1, vs, bw) == 0 ? 0 : 1;
        }

        const std::string inputFile = arg1;
        const float voxelSize = (argc > 2) ? std::stof(argv[2]) : 0.01f;
        const float bandWidth = (argc > 3) ? std::stof(argv[3]) : 3.0f;

        std::vector<nanovdb::Vec3f> points;
        std::vector<nanovdb::Vec3i> triangles;
        std::cout << "Reading " << inputFile << "...\n";
        readOBJ(inputFile, points, triangles);
        std::cout << "Loaded " << points.size() << " vertices, "
                  << triangles.size() << " triangles.\n";
        if (points.empty() || triangles.empty())
            throw std::runtime_error("mesh has no triangles");

        // Index<->world transform: uniform voxel size, no translation. (nanovdb::Map)
        nanovdb::Map map;
        map.set(double(voxelSize), nanovdb::Vec3d(0.0), 1.0);

        // Step 1: rasterize the mesh into an index grid + UDF sidecar + nearest-triangle-index sidecar.
        // (The index sidecar is just carried for now; step 5 / barrier signing will consume it.)
        auto [handle, sidecar, indexSidecar] = computeUDFAndIndex(points, triangles, map, bandWidth);

        printGridDiagnostics(handle, "Rasterized UDF grid");
        std::cout << "UDF sidecar                           : "
                  << (sidecar.size() / sizeof(float)) << " floats\n";
        std::cout << "Nearest-triangle-index sidecar        : "
                  << (indexSidecar.size() / sizeof(uint32_t)) << " uint32\n";

        // Step 2: derive the CC-input topology by pruning the surface/barrier shell.
        auto derivedHandle = computeDerivedTopology(handle, sidecar, voxelSize);
        printGridDiagnostics(derivedHandle, "Derived CC-input grid");

        // Steps 3–5: connected-components labeling + signing on the derived grid, inject the signs
        // back onto the original grid, then sign the remaining barrier voxels on the original.
        computeCC(handle, derivedHandle, indexSidecar, points, triangles, map, bandWidth);

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"\n";
        return 1;
    }
}
