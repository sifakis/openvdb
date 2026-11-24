#!/bin/bash
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# run.sh <mesh dir> [longestExtentInVoxels]: the hydra from the nine snake meshes in <mesh dir> (default: 1024 voxels
# along the body's longest extent). output/plan/ receives the plan of plan_hydra.py (made once), output/<extent>/
# the band files and the log of bin/hydra (log.txt), the surface of the result as triangles (hydra.meshbin) and, if
# RENDER_PYTHON is set, a path-traced image of it (hydra.png).
# Needs ./build.sh. PLAN_PYTHON: a Python with numpy and scipy (default: python3). RENDER_PYTHON: a Python with
# ovrtx, ovstage, pxr (usd-core), numpy and Pillow. RENDER_OPTIONS: options of render.py (default: the view from the
# front).
set -eu -o pipefail
[ $# -ge 1 ] || { echo "usage: $0 <mesh dir> [longestExtentInVoxels]"; exit 1; }
MESHES=$(realpath "$1"); EXTENT=${2:-1024}
cd "$(dirname "$0")"
D=output/$EXTENT
mkdir -p $D
[ -s output/plan/scene.txt ] || ${PLAN_PYTHON:-python3} plan_hydra.py "$MESHES" output/plan | tee output/plan_log.txt
bin/hydra output/plan/scene.txt $EXTENT $D | tee $D/log.txt
bin/band_to_mesh $D/hydra.band $D/hydra.meshbin | tee -a $D/log.txt
if [ -n "${RENDER_PYTHON:-}" ]; then
  $RENDER_PYTHON render.py $D/hydra.meshbin $D/hydra.png ${RENDER_OPTIONS:---view 0.05 0.15 1 --distance 2.6} > $D/render.log 2>&1 ||
    { echo "render failed, see $D/render.log"; exit 1; }
  echo "image: $D/hydra.png"
fi
