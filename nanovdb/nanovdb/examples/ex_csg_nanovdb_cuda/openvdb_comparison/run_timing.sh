#!/bin/bash
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# run_timing.sh: times the NanoVDB CSG operations and OpenVDB's on the same mesh level sets and writes
# output/timing.txt. Rows:
#   the dragon and the armadillo, partly overlapping, at 220, 1024, 2048 and 3072 voxels along the longest extent
#   the dragon and the dragon at the same place, at 1024
# and, for the number of values of every union, the time of a plain device copy. The GPU must be idle: the script
# stops if another compute process is on it (the names in ALLOW_GPU_PROCESSES, a grep pattern, are tolerated).
#
# usage: ./run_timing.sh dragon.obj armadillo.obj     (after ./build.sh)
# Environment:
#   EXTENTS   the extents to run (default "220 1024 2048 3072", which fits a GPU with 12 GB and needs 4.5 GB of
#             disk for the band files). An extent that does not fit the GPU is reported and skipped.
set -eu -o pipefail
cd "$(dirname "$0")"
if [ $# -ne 2 ]; then echo "usage: $0 dragon.obj armadillo.obj"; exit 1; fi
EXTENTS=${EXTENTS:-220 1024 2048 3072}
ALLOW=${ALLOW_GPU_PROCESSES:-^$}
mkdir -p output
others() { nvidia-smi --query-compute-apps=name --format=csv,noheader | grep -Ev "$ALLOW" || true; }
if [ -n "$(others)" ]; then echo "other compute processes on the GPU:"; others; exit 1; fi

BAND() { echo "output/$1_$2.band"; }
for extent in $EXTENTS; do
  [ -f "$(BAND dragon $extent)" ] || bin/mesh_to_band "$1" $extent "$(BAND dragon $extent)"
  [ -f "$(BAND armadillo $extent)" ] || bin/mesh_to_band "$2" $extent "$(BAND armadillo $extent)"
done

# extent -> offset of the armadillo along z in voxels, NanoVDB runs, OpenVDB runs, PropagateSign baseline (0 or 1)
row() {
  case $1 in
    220) echo "58 31 15 1" ;; 1024) echo "265 31 15 1" ;; 2048) echo "530 21 9 1" ;; 3072) echo "795 15 7 1" ;;
    4096) echo "1060 11 5 0" ;; *) echo "$((265 * $1 / 1024)) 15 7 1" ;;
  esac
}
WARMUP=25   # seconds of unions before the NanoVDB timings
time_pair() { # label, band 1, band 2, offset, NanoVDB runs, OpenVDB runs, PropagateSign baseline (0 or 1)
  echo "=== $1 ==="
  if ! bin/nanovdb_timing "$2" "$3" 2 $4 $5 $WARMUP $7; then echo "the NanoVDB run failed: skipped"; return 0; fi
  bin/nanovdb_stage_timing "$2" "$3" 2 $4 $5 10 0 | grep -v "^nanovdb"
  if [ -n "$(others)" ]; then echo "another compute process appeared on the GPU: $(others)"; exit 1; fi
  bin/openvdb_timing "$2" "$3" 2 $4 $6
}
{
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
  for extent in $EXTENTS; do
    read offset nano ovdb baseline <<< "$(row $extent)"
    time_pair "dragon and armadillo at $extent, armadillo moved by $offset along z" \
        "$(BAND dragon $extent)" "$(BAND armadillo $extent)" $offset $nano $ovdb $baseline
  done
  if [ -f "$(BAND dragon 1024)" ]; then
    time_pair "dragon and dragon at 1024, same place" "$(BAND dragon 1024)" "$(BAND dragon 1024)" 0 31 15 1
  fi
} 2>&1 | tee output/timing.txt
# A plain device copy of as many floats as each union has active voxels.
counts=$(grep -A1 "^=== dragon and armadillo" output/timing.txt | grep "1 | 2" | sed 's/.*active //')
echo "=== device copy ===" | tee -a output/timing.txt
bin/device_copy_timing $counts | tee -a output/timing.txt
