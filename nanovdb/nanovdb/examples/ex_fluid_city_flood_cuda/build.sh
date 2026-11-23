#!/bin/bash
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# build.sh: builds the three programs of this example into bin/, straight with nvcc and g++ against the NanoVDB
# headers of this repository (the example is not registered in examples/CMakeLists.txt).
#   bin/eulerian   the level-set free-surface solver (eulerian.cu and the .cuh files here, plus Fluid.cuh and
#                  scalable/ of ../ex_fluid_apic_cuda, whose multigrid and cell kernels it reuses)
#   bin/gen_sdf    MeshToSDF on a triangle mesh -> the obstacle file the solver reads
#   bin/mesh_band  OpenVDB volumeToMesh on a frame of the solver -> the water surface as triangles (for rendering)
# Environment:
#   NVCC              the CUDA compiler (default: nvcc)
#   CUDA_ARCH_FLAGS   default "--generate-code=arch=compute_90,code=[compute_90]": PTX that the driver compiles for the
#                     GPU it runs on, so one build runs on any GPU from Hopper on; e.g. "-arch=sm_120" for a native build
#   CCCL              a checkout of https://github.com/NVIDIA/cccl (NanoVDB's CMake build fetches one into
#                     <build>/_deps/cccl-src). Needed when the CUDA toolkit's own Thrust has no <cuda/std/...> headers.
#   OPENVDB_ROOT      an OpenVDB installation with include/ and lib/ (default: $HOME/local), for mesh_band only;
#                     without it mesh_band is skipped
#   EXTRA_FLAGS       further nvcc flags
set -u
cd "$(dirname "$0")"
NVCC=${NVCC:-nvcc}
ARCH=${CUDA_ARCH_FLAGS:---generate-code=arch=compute_90,code=[compute_90]}
INCLUDES="-I../../.. -I../ex_fluid_apic_cuda"
if [ -n "${CCCL:-}" ]; then INCLUDES="$INCLUDES -I$CCCL/thrust -I$CCCL/libcudacxx/include -I$CCCL/cub"; fi
FLAGS="-std=c++17 --extended-lambda -O3 -DNDEBUG -DNANOVDB_USE_CUDA -DCCCL_DISABLE_CUB_UMBRELLA_HEADER_WARNING ${EXTRA_FLAGS:-}"
mkdir -p bin
STATUS=0
for program in eulerian gen_sdf; do
  echo "building bin/$program"
  $NVCC $FLAGS "$ARCH" $INCLUDES $program.cu -o bin/$program || STATUS=1
done
OPENVDB_ROOT=${OPENVDB_ROOT:-$HOME/local}
if [ -f "$OPENVDB_ROOT/include/openvdb/openvdb.h" ]; then
  echo "building bin/mesh_band"
  g++ -std=c++17 -O3 -DNDEBUG -I"$OPENVDB_ROOT/include" mesh_band.cpp -o bin/mesh_band \
      -L"$OPENVDB_ROOT/lib" -Wl,-rpath,"$OPENVDB_ROOT/lib" -lopenvdb -ltbb || STATUS=1
else
  echo "no OpenVDB under $OPENVDB_ROOT: bin/mesh_band not built (it is only needed for rendering)"
fi
exit $STATUS
