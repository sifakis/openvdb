// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

// the following files are from OpenVDB
#include <openvdb/tools/Morphology.h>
#include <openvdb/util/CpuTimer.h>

// the following files are from NanoVDB
#include <nanovdb/NanoVDB.h>
#include <nanovdb/cuda/DeviceBuffer.h>
#include <nanovdb/tools/CreateNanoGrid.h>

#include <random>

void mainSparseConvolutionIGEMM(
    const std::vector<nanovdb::Coord>& inputPoints,
    const std::vector<nanovdb::Coord>& outputPoints,
    uint32_t benchmark_iters);

uint32_t coordinate_bitpack(uint32_t x) {
    x &= 0x49249249; // keep only one every 3 bits
    x |= (x >> 2);
    x &= 0xc30c30c3; // Pack into pairs of bits
    x |= (x >> 4);
    x &= 0x0f00f00f; // Pack into quadruples of bits
    x |= (x >> 8);
    x &= 0xff0000ff; // Pack into quadruples of bits
    x |= (x >> 16);
    x &= 0x0000ffff; // Pack into 16-tuples (actually, 11 max) of bits
    return x;
}
/// @brief This example depends on OpenVDB, NanoVDB, and CUDA
int main(int argc, char *argv[])
{
    int benchmark_iters = 10;
    if (argc > 2) sscanf(argv[2], "%d", &benchmark_iters);

    std::random_device rd;
    // std::mt19937 generator(rd());
    std::mt19937 generator(12345);

    // static const int ambient_voxels = 16*1024;
    static const int ambient_voxels = 1024*1024*2;
    static const float input_occupancy = .75f;
    static const float output_occupancy = .75f;
    static const float overlap = .65f;
    nanovdb::Coord offset(0,0,0);

    // Mark input voxels at requested occupancy
    int target_input_voxels = (int) (input_occupancy*(float)ambient_voxels);
    std::vector<bool> voxmap(ambient_voxels);
    int active_voxels = 0;
    std::uniform_int_distribution<int> distribution(0, ambient_voxels-1);
    while (active_voxels < target_input_voxels) {
        int i = distribution(generator);
        if (!voxmap[i]) {
            voxmap[i] = true;
            active_voxels++;
        }
    }

    // Convert to coordinates
    std::vector<nanovdb::Coord> inputPoints;
    for (int i = 0; i < ambient_voxels; i++)
        if(voxmap[i]) {
        int x = coordinate_bitpack(     i & 0x49249249);
        int y = coordinate_bitpack((i>>1) & 0x49249249);
        int z = coordinate_bitpack((i>>2) & 0x49249249);
        inputPoints.emplace_back(nanovdb::Coord(x,y,z)+offset);
    }
    std::cout << inputPoints.size() << " input voxels generated" << std::endl;

    // Discard voxels until desired level of overlap
    int target_overlap_voxels = (int) (overlap*(float)ambient_voxels);
    while (active_voxels > target_overlap_voxels) {
        int i = distribution(generator);
        if (voxmap[i]) {
            voxmap[i] = false;
            active_voxels--;
        }
    }
    // Then sample more voxels until desired occupancy is met
    int target_output_voxels = (int) (output_occupancy*(float)ambient_voxels);
    while (active_voxels < target_output_voxels) {
        int i = distribution(generator);
        if (!voxmap[i]) {
            voxmap[i] = true;
            active_voxels++;
        }
    }
    // Convert to coordinates
    std::vector<nanovdb::Coord> outputPoints;
    for (int i = 0; i < ambient_voxels; i++)
        if(voxmap[i]) {
        int x = coordinate_bitpack(     i & 0x49249249);
        int y = coordinate_bitpack((i>>1) & 0x49249249);
        int z = coordinate_bitpack((i>>2) & 0x49249249);
        outputPoints.emplace_back(nanovdb::Coord(x,y,z)+offset);
    }
    std::cout << outputPoints.size() << " output voxels generated" << std::endl;

    mainSparseConvolutionIGEMM(inputPoints, outputPoints, benchmark_iters);

    return 0;
}
