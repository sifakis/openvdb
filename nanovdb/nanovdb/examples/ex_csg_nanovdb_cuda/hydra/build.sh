#!/bin/bash
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# build.sh: builds the two programs of this example into bin/, straight with nvcc and g++ against the NanoVDB
# headers of this repository (they are not registered in examples/CMakeLists.txt, because one needs an OpenVDB
# installation).
#   bin/hydra          MeshToSDF and the CSG operations: OBJ meshes -> band files
#   bin/band_to_mesh   OpenVDB volumeToMesh: band file -> the surface as triangles (for rendering)
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
echo "building bin/hydra"
$NVCC $FLAGS "$ARCH" $INCLUDES hydra.cu -o bin/hydra || STATUS=1
OPENVDB_ROOT=${OPENVDB_ROOT:-$HOME/local}
if [ -f "$OPENVDB_ROOT/include/openvdb/openvdb.h" ]; then
  echo "building bin/band_to_mesh"
  $CXX -std=c++17 -O3 -DNDEBUG -I"$OPENVDB_ROOT/include" band_to_mesh.cpp -o bin/band_to_mesh \
      -L"$OPENVDB_ROOT/lib" -Wl,-rpath,"$OPENVDB_ROOT/lib" -lopenvdb -ltbb || STATUS=1
else
  echo "no OpenVDB under $OPENVDB_ROOT: bin/band_to_mesh not built"
  STATUS=1
fi
exit $STATUS
