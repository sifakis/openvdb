#!/bin/bash
# build.sh: builds cc_opt_bench next to this script with strict warnings.
# CCCL must point at a CCCL checkout (thrust, cub, libcudacxx); ARCH selects the code generation.
D=$(cd "$(dirname "$0")" && pwd)
NANOVDB=$(cd "$D/../.." && pwd)/..   # the directory that holds nanovdb/
CCCL=${CCCL:-/home/jaehlee/Desktop/Code/NVIDIA/learning/work/bench/partition_grid/build_pg/_deps/cccl-src}
ARCH=${ARCH:---generate-code=arch=compute_90,code=[compute_90]}
nvcc --Werror=all-warnings --expt-extended-lambda -O3 -DNDEBUG -std=c++17 $ARCH -DNANOVDB_USE_CUDA \
  -Xcompiler=-D_AMXTILEINTRIN_H_INCLUDED -Xcompiler=-D_AMXINT8INTRIN_H_INCLUDED -Xcompiler=-D_AMXBF16INTRIN_H_INCLUDED \
  -I"$NANOVDB" -I"$CCCL/thrust" -I"$CCCL/libcudacxx/include" -I"$CCCL/cub" \
  "$D/cc_opt_bench.cu" -o "$D/cc_opt_bench"
