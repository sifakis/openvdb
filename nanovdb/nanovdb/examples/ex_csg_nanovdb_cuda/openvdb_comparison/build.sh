#!/bin/bash
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# build.sh: builds the programs of the comparison with OpenVDB into bin/, straight with nvcc and g++ against the
# NanoVDB headers of this repository (they are not registered in examples/CMakeLists.txt, because two of them
# need an OpenVDB installation).
#   bin/mesh_to_band           ../mesh_to_band.cu: OBJ mesh -> band file, the input of all the others
#   bin/nanovdb_fields         the fields of the NanoVDB results, for openvdb_fields
#   bin/openvdb_fields         OpenVDB's results against them, voxel by voxel
#   bin/nanovdb_timing         wall time of the NanoVDB operations
#   bin/nanovdb_stage_timing   the same with the time of every step (stage_timers.py writes the header copy it uses)
#   bin/openvdb_timing         wall time of the OpenVDB operations
#   bin/device_copy_timing     wall time of a plain device copy
# Environment:
#   NVCC              the CUDA compiler (default: nvcc)
#   CUDA_ARCH_FLAGS   default "--generate-code=arch=compute_90,code=[compute_90]": PTX that the driver compiles for the
#                     GPU it runs on; e.g. "-arch=sm_120" for a native build
#   CCCL              a checkout of https://github.com/NVIDIA/cccl (NanoVDB's CMake build fetches one into
#                     <build>/_deps/cccl-src). Needed when the CUDA toolkit's own Thrust has no <cuda/std/...> headers.
#   OPENVDB_ROOT      an OpenVDB installation with include/ and lib/ (default: $HOME/local)
#   CXX               the host compiler (default: g++)
#   EXTRA_FLAGS       further nvcc flags
set -u
cd "$(dirname "$0")"
NVCC=${NVCC:-nvcc}
CXX=${CXX:-g++}
ARCH=${CUDA_ARCH_FLAGS:---generate-code=arch=compute_90,code=[compute_90]}
INCLUDES="-I../../../.. -I.."
if [ -n "${CCCL:-}" ]; then INCLUDES="$INCLUDES -I$CCCL/thrust -I$CCCL/libcudacxx/include -I$CCCL/cub"; fi
FLAGS="-std=c++17 --extended-lambda -O3 -DNDEBUG -DNANOVDB_USE_CUDA -DCCCL_DISABLE_CUB_UMBRELLA_HEADER_WARNING ${EXTRA_FLAGS:-}"
mkdir -p bin
STATUS=0
echo "building bin/mesh_to_band"
$NVCC $FLAGS "$ARCH" $INCLUDES ../mesh_to_band.cu -o bin/mesh_to_band || STATUS=1
for program in nanovdb_fields nanovdb_timing device_copy_timing; do
  echo "building bin/$program"
  $NVCC $FLAGS "$ARCH" $INCLUDES $program.cu -o bin/$program || STATUS=1
done
echo "building bin/nanovdb_stage_timing"
python3 stage_timers.py ../../../tools/cuda/CSG.cuh bin/stage_timers &&
  $NVCC $FLAGS "$ARCH" -DCSG_STAGE_TIMERS -Ibin/stage_timers $INCLUDES nanovdb_timing.cu -o bin/nanovdb_stage_timing || STATUS=1
OPENVDB_ROOT=${OPENVDB_ROOT:-$HOME/local}
if [ -f "$OPENVDB_ROOT/include/openvdb/openvdb.h" ]; then
  for program in openvdb_fields openvdb_timing; do
    echo "building bin/$program"
    $CXX -std=c++17 -O3 -DNDEBUG -I"$OPENVDB_ROOT/include" $program.cpp -o bin/$program \
        -L"$OPENVDB_ROOT/lib" -Wl,-rpath,"$OPENVDB_ROOT/lib" -lopenvdb -ltbb || STATUS=1
  done
else
  echo "no OpenVDB under $OPENVDB_ROOT: bin/openvdb_fields and bin/openvdb_timing not built"
  STATUS=1
fi
exit $STATUS
