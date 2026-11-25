#!/bin/bash
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# run.sh <mesh directory> [voxels per unit=2048] [word=SIGGRAPH] [name=pile]: the scene and its conversion.
#   1. make_scene.py: bins shaped like the letters of <word>, filled with the meshes of <mesh directory>
#      (get_meshes.sh) -> output/<name>/scene/ (kept if it is already there)
#   2. bin/mesh_to_sdf_pile: one MeshToSDF conversion of the whole scene with BarrierSigning::Ball at
#      <voxels per unit> voxels per cap height of the letters -> output/<name>/sdf.faces (the voxels of the surface),
#      sdf_slice_0.bin .. sdf_slice_71.bin (the sign on 72 horizontal planes from height 0.38 down to the ground,
#      every 4th voxel at 2048 voxels per unit) and sdf_slice_72.bin (the plane at height 0.2462, every voxel)
# With the defaults this is the run of the paper: 104 million triangles, which need a GPU with 32 GB. A scene of one
# letter fits a 12 GB GPU: ./run.sh <mesh directory> 1024 R small
# Environment:
#   SCENE_PYTHON   python with numpy, scipy, Pillow (default: python3)
#   SHIFT_VOXELS   "i j k": the scene is moved by this many voxels for the conversion (mesh_to_sdf_pile.cu says why).
#                  Default: minus the middle of the scene along x, rounded to 0.01 unit, in voxels; "-7905 0 0" for
#                  the paper's run.
set -u
cd "$(dirname "$0")"
if [ $# -lt 1 ]; then sed -n '5,19p' "$0"; exit 1; fi
MESHES=$1
VOXELS=${2:-2048}
WORD=${3:-SIGGRAPH}
NAME=${4:-pile}
PYTHON=${SCENE_PYTHON:-python3}
OUT=output/$NAME
mkdir -p "$OUT"
if [ ! -f "$OUT/scene/soup.bin" ]; then
  "$PYTHON" make_scene.py "$OUT/scene" --mesh-dir "$MESHES" --word "$WORD" | tee "$OUT/scene.log" || exit 1
fi
SHIFT=${SHIFT_VOXELS:-$("$PYTHON" -c "
import json, sys
s = json.load(open(sys.argv[1]))
print(-int(round(round(0.5 * (s['lo'][0] + s['hi'][0]), 2) * float(sys.argv[2]))), 0, 0)" "$OUT/scene/scene.json" "$VOXELS")}
STRIDE=$(( VOXELS / 512 > 1 ? VOXELS / 512 : 1 ))
SLICES=$(awk -v s="$STRIDE" 'BEGIN { for (k = 0; k < 72; ++k) printf "%sz%.5f:%d", (k ? "," : ""), 0.38 - k * 0.38 / 71, s; printf ",z0.24620:1" }')
{
  echo "# $(date -Is): $OUT/scene/soup.bin at $VOXELS voxels per unit, ball, SHIFT_VOXELS=$SHIFT"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
} > "$OUT/run.log"
SHIFT_VOXELS="$SHIFT" bin/mesh_to_sdf_pile "$OUT/scene/soup.bin" "$VOXELS" "$OUT/sdf" 3 ball 0 "$SLICES" >> "$OUT/run.log" 2>&1
CODE=$?
grep -E "completed|RESULT|interior voxels|bounding box" "$OUT/run.log"
if [ $CODE -ne 0 ]; then tail -n 3 "$OUT/run.log"; echo "FAILED (exit code $CODE), see $OUT/run.log"; fi
exit $CODE
