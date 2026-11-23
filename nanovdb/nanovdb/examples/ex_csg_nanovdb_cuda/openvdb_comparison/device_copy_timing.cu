// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file device_copy_timing.cu
///
/// @brief Wall time of a device-to-device copy of N floats: a lower bound for a CSG step that writes the N values
///        of a result. Median of 21 copies after 300 warm-up copies.
///
/// usage: device_copy_timing N [N ...]

#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv)
{
    for (int a = 1; a < argc; ++a) {
        const size_t n = strtoull(argv[a], nullptr, 10);
        float *src, *dst;
        if (cudaMalloc(&src, n * 4) != cudaSuccess || cudaMalloc(&dst, n * 4) != cudaSuccess) { printf("allocation failed\n"); return 1; }
        cudaMemset(src, 0, n * 4);
        cudaMemset(dst, 0, n * 4);
        for (int i = 0; i < 300; ++i) cudaMemcpy(dst, src, n * 4, cudaMemcpyDeviceToDevice);
        std::vector<double> t;
        for (int i = 0; i < 21; ++i) {
            cudaDeviceSynchronize();
            const auto t0 = std::chrono::steady_clock::now();
            cudaMemcpy(dst, src, n * 4, cudaMemcpyDeviceToDevice);
            cudaDeviceSynchronize();
            t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        std::sort(t.begin(), t.end());
        printf("device copy of %zu floats: median %.3f ms\n", n, t[10]);
        cudaFree(src);
        cudaFree(dst);
    }
    return 0;
}
