#!/bin/bash
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# build.sh: builds the programs of this example into bin/, straight with nvcc and g++ against the NanoVDB headers of
# this repository (the example is not registered in examples/CMakeLists.txt).
#   bin/mesh_to_sdf_pile   MeshToSDF on the scene's triangle soup (mesh_to_sdf_pile.cu)
#   bin/mesh_band          OpenVDB volumeToMesh on a narrow band -> triangles; make_scene.py makes the bins with it
#                          (../ex_fluid_city_flood_cuda/mesh_band.cpp, used as it is)
# Environment:
#   NVCC              the CUDA compiler (default: nvcc)
#   CUDA_ARCH_FLAGS   default "--generate-code=arch=compute_90,code=[compute_90]": PTX that the driver compiles for the
#                     GPU it runs on, so one build runs on any GPU from Hopper on; e.g. "-arch=sm_120" for a native build
#   CCCL              a checkout of https://github.com/NVIDIA/cccl (NanoVDB's CMake build fetches one into
#                     <build>/_deps/cccl-src). Needed when the CUDA toolkit's own Thrust has no <cuda/std/...> headers.
#   OPENVDB_ROOT      an OpenVDB installation with include/ and lib/ (default: $HOME/local), for mesh_band; without
#                     it mesh_band is skipped (the machine that only converts a scene made elsewhere does not need it)
#   EXTRA_FLAGS       further nvcc flags. If nvcc stops in the host compiler's AMX headers:
#                     "-Xcompiler=-D_AMXTILEINTRIN_H_INCLUDED -Xcompiler=-D_AMXINT8INTRIN_H_INCLUDED -Xcompiler=-D_AMXBF16INTRIN_H_INCLUDED"
set -u
cd "$(dirname "$0")"
NVCC=${NVCC:-nvcc}
ARCH=${CUDA_ARCH_FLAGS:---generate-code=arch=compute_90,code=[compute_90]}
INCLUDES="-I../../.."
if [ -n "${CCCL:-}" ]; then INCLUDES="$INCLUDES -I$CCCL/thrust -I$CCCL/libcudacxx/include -I$CCCL/cub"; fi
FLAGS="-std=c++17 --extended-lambda -O3 -DNDEBUG -DNANOVDB_USE_CUDA -DCCCL_DISABLE_CUB_UMBRELLA_HEADER_WARNING ${EXTRA_FLAGS:-}"
mkdir -p bin
STATUS=0
echo "building bin/mesh_to_sdf_pile"
$NVCC $FLAGS "$ARCH" $INCLUDES mesh_to_sdf_pile.cu -o bin/mesh_to_sdf_pile || STATUS=1
OPENVDB_ROOT=${OPENVDB_ROOT:-$HOME/local}
if [ -f "$OPENVDB_ROOT/include/openvdb/openvdb.h" ]; then
  echo "building bin/mesh_band"
  g++ -std=c++17 -O3 -DNDEBUG -I"$OPENVDB_ROOT/include" ../ex_fluid_city_flood_cuda/mesh_band.cpp -o bin/mesh_band \
      -L"$OPENVDB_ROOT/lib" -Wl,-rpath,"$OPENVDB_ROOT/lib" -lopenvdb -ltbb || STATUS=1
else
  echo "no OpenVDB under $OPENVDB_ROOT: bin/mesh_band not built"
fi
exit $STATUS
