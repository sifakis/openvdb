#!/bin/bash
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# render_movie.sh [name] [movie name]: meshes every frame of a run (output/<name>/out/f_NNN.band -> output/<name>/mesh/,
# bin/mesh_band) and renders output/<name>/<movie name>/movie.mp4 at 24 frames per second (the run has 8 frames per
# simulated second, so the movie plays at three times real time). Defaults: name "flood", movie name "movie", and the
# look of the paper's movie.
# A frame that has a mesh newer than its band file and an image newer than that mesh is not redone, so the script can
# be stopped and started again. A batch whose renderer crashed is tried again, up to three passes.
#
# Needs: bin/mesh_band (./build.sh with an OpenVDB installation), ffmpeg, the city pack in assets/ (./get_assets.sh)
# and RENDER_PYTHON: a Python with ovrtx, ovstage, pxr (usd-core), numpy and Pillow (default: python3).
#
# The look comes from the environment:
#   CAM     camera options of render_city.py. Default: the camera moves from a wide view from the south to a closer
#           one over the flooded park during TOTAL frames (default: the number of frames of the run).
#   SIZE    image size (default 2560 x 1440)
#   WATER   color r g b, the depth in meters over which the water takes it, and further water options
#   SPRAY   options of the drops
#   BATCH   frames per renderer process (default 2)
#   FIRST, LAST   a frame range; no movie is made then
# Water layers one cell deep are not drawn, and drops only over water three cells deep: the solver cannot carry
# thinner water (it stalls and flickers). The boxes drawn (the basin and the walls) are the scene's own: the
# RENDER_BOXES line of output/<name>/build.txt.
set -u
cd "$(dirname "$0")"
NAME=${1:-flood}; MOVIE=${2:-movie}
D=output/$NAME
PY=${RENDER_PYTHON:-python3}
H=$(cat $D/voxel.txt 2> /dev/null || echo 0.25)
M=$D/mesh
[ -x bin/mesh_band ] || { echo "bin/mesh_band is missing: ./build.sh with OPENVDB_ROOT"; exit 1; }
n=$(ls $D/out/f_*.band 2> /dev/null | wc -l)
[ $n -gt 0 ] || { echo "no frames in $D/out"; exit 1; }
first=${FIRST:-0}; last=${LAST:-$((n - 1))}
mkdir -p $M $D/$MOVIE
CAM=${CAM:---camera 775 320 1180 --target 775 0 650 --camera2 700 200 1010 --target2 770 0 600 --total ${TOTAL:-$n} --focal 35}
SIZE=${SIZE:---width 2560 --height 1440}
SPRAY=${SPRAY:---spray-above 0.6 --spray-in-air 1.0 --spray-min-depth 3 --spray-scale 2 --spray-min 0.06 --spray-max 600000 --spray-color 0.90 0.94 0.97}
WATER=${WATER:-0.40 0.62 0.60 4 --water-roughness 0.08}   # grey-green, slightly rough
BOXES=$(grep "^RENDER_BOXES=" $D/build.txt | cut -d= -f2)
LOOK="--thin --water $WATER --water-min-cells 2 --dome 1000 --sun 50 60 8 --box $BOXES --spray $D/out $SPRAY"
N=${BATCH:-2}
one() {  # render the frames $1..$2
  $PY render_city.py $D/$MOVIE --frames $M --voxel $H --origin 0 -21 0 --first $1 --count $(($2 - $1 + 1)) $CAM $SIZE --samples 48 $LOOK \
    > $D/$MOVIE/render_$1.log 2>&1 || { echo "RENDER FAILED at $1"; tail -n 3 $D/$MOVIE/render_$1.log | cut -c1-300; }
}
for pass in 1 2 3; do
  for ((i = first; i <= last; ++i)); do
    f=$(printf "f_%03d" $i)
    [ -s $M/$f.meshbin ] && [ $M/$f.meshbin -nt $D/out/$f.band ] || echo $f
  done | xargs -r -P 2 -I{} sh -c "bin/mesh_band $D/out/{}.band $M/{}.meshbin > /dev/null || echo MESH FAILED {}"
  for ((b = first; b <= last; b += N)); do
    e=$((b + N - 1 > last ? last : b + N - 1))
    todo=0
    for ((i = b; i <= e; ++i)); do [ $D/$MOVIE/$(printf "frame_%04d.png" $i) -nt $M/$(printf "f_%03d.meshbin" $i) ] || todo=1; done
    [ $todo = 1 ] && one $b $e
  done
  missing=0
  for ((i = first; i <= last; ++i)); do [ -f $D/$MOVIE/$(printf "frame_%04d.png" $i) ] || missing=1; done
  [ $missing = 0 ] && break
done
[ -n "${FIRST:-}${LAST:-}" ] && exit 0
# A frame whose image equals another frame's is rendered again on its own (the renderer can repeat a frame of a batch).
for i in $(md5sum $D/$MOVIE/frame_*.png | sort | awk '{c[$1]++; f[$1] = f[$1] " " substr($2, length($2) - 7, 4)} END {for (k in c) if (c[k] > 1) print f[k]}'); do
  i=$((10#$i)); rm -f $D/$MOVIE/$(printf "frame_%04d.png" $i); one $i $i
done
ffmpeg -y -v error -framerate 24 -i $D/$MOVIE/frame_%04d.png -c:v libx264 -pix_fmt yuv420p -crf 17 $D/$MOVIE/movie.mp4
echo "frames $(ls $D/$MOVIE/frame_*.png 2> /dev/null | wc -l), distinct $(md5sum $D/$MOVIE/frame_*.png 2> /dev/null | awk '{print $1}' | sort -u | wc -l)"
