#!/bin/bash
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# make_movie.sh <mesh.obj> [name] [resolution] [steps] [frames] [preserve volume: 0 | 1] [schedule exponent]:
# a closed triangle mesh flows by mean curvature, and the frames become a movie. Defaults: name "flow", resolution
# 512 (voxels along the mesh's longest side), 60000 steps, 360 frames, volume preserved, schedule exponent 3 (frame k
# is taken after steps * (k / frames)^3 steps, so most frames show the start, where the small features go).
# Everything goes to output/<name>/:
#   level_set.sdf, sdf.txt   the mesh's level set with a band of 5 voxels (bin/mesh_to_sdf_file) and MeshToSDF's report
#   band/f_NNN.band, run.log the frames of the flow (bin/mean_curvature_flow) and its report, one line per frame
#   mesh/f_NNN.meshbin       the surface of every frame as triangles (bin/mesh_band, 4 at a time)
#   png/frame_NNN.png        the images (render_flow.py, one frame per process), with render_NNN.log
#   movie.mp4                24 frames per second
# A stage whose output exists is skipped, so the script can be started again after an interruption; a frame whose
# render failed is tried again, up to three passes.
#
# Needs: the programs of ./build.sh, ffmpeg, and RENDER_PYTHON: a Python with ovrtx, ovstage, pxr (usd-core), numpy
# and Pillow (default: python3).
# RENDER_OPTIONS: options of render_flow.py (default: "--up y --view 1 0.35 0.6 --samples 64 --width 1920 --height 1080").
set -u
cd "$(dirname "$0")"
[ $# -ge 1 ] || { echo "usage: make_movie.sh <mesh.obj> [name] [resolution] [steps] [frames] [preserve volume: 0 | 1] [schedule exponent]"; exit 1; }
MESH=$1; NAME=${2:-flow}; RES=${3:-512}; STEPS=${4:-60000}; FRAMES=${5:-360}; VOLUME=${6:-1}; EXPONENT=${7:-3}
PY=${RENDER_PYTHON:-python3}
OPTIONS=${RENDER_OPTIONS:---up y --view 1 0.35 0.6 --samples 64 --width 1920 --height 1080}
D=output/$NAME
for program in mesh_to_sdf_file mean_curvature_flow mesh_band; do
  [ -x bin/$program ] || { echo "bin/$program is missing: run ./build.sh first"; exit 1; }
done
mkdir -p $D/band $D/mesh $D/png
if [ ! -s $D/level_set.sdf ]; then
  bin/mesh_to_sdf_file "$MESH" $RES 5 $D/level_set.sdf > $D/sdf.txt 2>&1 || { cat $D/sdf.txt; exit 1; }
fi
if ! grep -q "^ *$FRAMES " $D/run.log 2> /dev/null; then
  bin/mean_curvature_flow $D/level_set.sdf $D/band $STEPS $FRAMES $VOLUME $EXPONENT > $D/run.log 2>&1 || { echo "the flow failed"; tail -n 3 $D/run.log; exit 1; }
fi
echo "flow done $(date +%T)"
ls $D/band/f_*.band | xargs -P 4 -I{} sh -c "b=\$(basename {} .band); [ -s $D/mesh/\$b.meshbin ] || bin/mesh_band {} $D/mesh/\$b.meshbin > /dev/null || echo mesh failed: \$b"
echo "meshes done $(date +%T): $(ls $D/mesh | wc -l)"
for pass in 1 2 3; do
  missing=0
  for ((k = 0; k <= FRAMES; ++k)); do
    name=$(printf "%03d" $k)
    [ -s $D/png/frame_$name.png ] && continue
    $PY render_flow.py $D/mesh $D/png --only $k $OPTIONS > $D/png/render_$name.log 2>&1 || { echo "render failed: frame $name (pass $pass)"; missing=1; }
  done
  [ $missing = 0 ] && break
done
echo "images done $(date +%T): $(ls $D/png/frame_*.png | wc -l)"
ffmpeg -y -v error -framerate 24 -i $D/png/frame_%03d.png -c:v libx264 -pix_fmt yuv420p -crf 17 $D/movie.mp4 && echo "$D/movie.mp4 $(date +%T)"
