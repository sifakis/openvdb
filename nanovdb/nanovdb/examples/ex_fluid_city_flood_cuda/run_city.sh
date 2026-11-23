#!/bin/bash
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# run_city.sh [name] [voxel in m] [seconds] [build_city.py options]: the city flood. Builds the scene's solids
# (build_city.py: the city, the ground under it, the basin in the river, the flood wall with its openings), runs
# MeshToSDF on them and then the solver; everything goes to output/<name>/. Without arguments it is the paper's run:
# name "flood", 0.25 m voxels, 80 seconds, and the scene options listed below as SCENE.
# The solids are built once per name (output/<name>/scene.sdf is reused).
#
# Needs: bin/eulerian and bin/gen_sdf (./build.sh) and output/city.obj (./get_assets.sh). RENDER_PYTHON: a Python with
# numpy (default: python3).
#
# Output in output/<name>/:
#   build.txt    the scene: triangle counts, the tank size in cells, the basin's water box, the boxes a renderer draws
#   voxel.txt    the voxel size, for render_movie.sh
#   sdf.txt      MeshToSDF's report (voxel size, active and negative voxels)
#   scene.sdf    the solids as the solver reads them
#   out/f_NNN.band, out/f_NNN.spray   the frames, FPS (default 8) per simulated second: the level set within 2 voxels
#                of the surface, and the drops (one of every SPRAY_DUMP_EVERY, default 16)
#   out.log      the solver's reports (time, volume, time per step, peak device memory); ends with EXIT=done or EXIT=fail
#   ck_<time>.ckpt   the solver's state, written every CKPT_EVERY (default 5; 0: never) simulated seconds; only the two
#                newest are kept
#
# If the solver stops with an error it is started again from the newest state file, up to MAX_ATTEMPTS (default 6)
# times and at most twice from the same file. The water and its velocity continue exactly; the drops in the air and
# the marker particles are not in the file and start anew there. Frames after that time are written again.
#
# Solver switches (environment; the defaults are the ones of the paper's run):
#   PLS=16 PLS_WIDTH=2   marker particles (particle level set): 16 per cell within 2 cells of the surface
#   SPRAY=1          water the level set cannot hold (thinner than about two cells) is carried as drops
#   VMAX=20          velocity components are held to 20 m/s
#   OFFSET=0.6       the solid is everything within this many voxels of the mesh's solid
# Fixed here: no volume control (VOLCTL=0), semi-Lagrangian advection of the level set (PHI_SCHEME=0), MacCormack
# advection of the velocity (U_SCHEME=2), the sparse pressure pipeline (PIPE=2).
set -u
cd "$(dirname "$0")"
NAME=${1:-flood}; H=${2:-0.25}; END=${3:-80}
if [ $# -gt 3 ]; then shift 3; SCENE=("$@")
else SCENE=(--basin 550 1000 100 --head 10 --gate 655 700 740 810 855 900 --parapet 520 1150 4 --clip 12 --region 150 1500 0 930); fi
PY=${RENDER_PYTHON:-python3}
D=output/$NAME
LOG=$D/out.log
EVERY=${CKPT_EVERY:-5}
[ -x bin/eulerian ] && [ -x bin/gen_sdf ] || { echo "run ./build.sh first"; exit 1; }
[ -s output/city.obj ] || { echo "run ./get_assets.sh first"; exit 1; }
mkdir -p $D/out
echo $H > $D/voxel.txt
if [ ! -s $D/scene.sdf ]; then
  $PY -I build_city.py $D/scene.obj $H "${SCENE[@]}" > $D/build.txt 2>&1 || { cat $D/build.txt; exit 1; }
  RES=$(grep -o "resolution for this voxel: [0-9.]*" $D/build.txt | grep -o "[0-9.]*$")
  bin/gen_sdf $D/scene.obj $(printf "%.0f" $RES) 6 $D/scene.sdf > $D/sdf.txt 2>&1 || { cat $D/sdf.txt; exit 1; }
  rm -f $D/scene.obj
fi
TANK=$(grep "^tank" $D/build.txt | awk '{print $2, $3, $4}')
BOX=$(grep "^BOX=" $D/build.txt)
W=$(awk "BEGIN { printf \"%.10g\", 2 * 9.81 / $H }")   # with this column width the solver's T* is seconds
REPORT=$(awk "BEGIN { printf \"%.10g\", 1 / ${FPS:-8} }")
{
  echo "# $(date -Is), scene ${SCENE[*]}, voxel $H m, $END s"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
} > $LOG 2>&1
(while :; do sleep 60; ls -t $D/ck_*.ckpt 2> /dev/null | tail -n +3 | while read -r old; do rm -f "$old"; done; done) &
KEEPER=$!
ATTEMPT=1; FROM=0; LOAD=""; SAME=0
while :; do
  # the times of the state files still to come: every EVERY seconds after FROM, before END
  AT=$(awk -v from="$FROM" -v end="$END" -v every="$EVERY" 'BEGIN { s = ""; if (every > 0) for (t = every; t < end - 0.01; t += every) if (t > from + 0.01) s = s (s == "" ? "" : ",") t; print s }')
  env PIPE=2 U_SCHEME=2 VOLCTL=0 PHI_SCHEME=0 PLS=${PLS-16} PLS_WIDTH=${PLS_WIDTH:-2} SPRAY=${SPRAY-1} VMAX=${VMAX-20} TIMES=1 STATS=200 \
      REPORT=$REPORT VOXEL=$H $BOX OBSTACLE=$D/scene.sdf OBS_AT=mesh OBS_AXES=xyz OBS_OFFSET=${OFFSET:-0.6} \
      BAND=$D/out/f SPRAY_DUMP_EVERY=${SPRAY_DUMP_EVERY:-16} ${AT:+CKPT_SAVE=$D/ck CKPT_AT=$AT} ${LOAD:+CKPT_LOAD=$LOAD} \
      stdbuf -oL bin/eulerian dam $TANK $W 1 "$END" >> $LOG 2>&1
  CODE=$?
  echo "# attempt $ATTEMPT finished $(date -Is), exit code $CODE" >> $LOG
  if [ $CODE -eq 0 ]; then echo "EXIT=done" >> $LOG; break; fi
  LAST=$(ls -t $D/ck_*.ckpt 2> /dev/null | head -1)
  [ "$LAST" = "$LOAD" ] && SAME=$((SAME + 1)) || SAME=0
  if [ -z "$LAST" ] || [ $ATTEMPT -ge ${MAX_ATTEMPTS:-6} ] || [ $SAME -ge 2 ]; then echo "EXIT=fail" >> $LOG; break; fi
  LOAD=$LAST; FROM=$(basename $LAST .ckpt | sed 's/^ck_//'); ATTEMPT=$((ATTEMPT + 1))
  echo "# attempt $ATTEMPT continues from $LAST" >> $LOG
done
kill $KEEPER
tail -n 1 $LOG
