#!/bin/bash
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# run_correctness.sh: compares the results of the NanoVDB CSG operations with OpenVDB's, voxel by voxel, on 15
# pairs of mesh level sets: disjoint, touching, overlapping and coinciding inputs, and inputs that share a part of
# their surface. Writes output/correctness.txt and prints how many of the pairs are identical.
#
# usage: ./run_correctness.sh dragon.obj armadillo.obj bunny.obj     (after ./build.sh)
set -eu
cd "$(dirname "$0")"
if [ $# -ne 3 ]; then echo "usage: $0 dragon.obj armadillo.obj bunny.obj"; exit 1; fi
mkdir -p output
EXTENT=220   # voxels along the longest extent of every mesh
BAND() { echo "output/$1_$EXTENT.band"; }
i=0
for name in dragon armadillo bunny; do
  i=$((i + 1))
  [ -f "$(BAND $name)" ] || bin/mesh_to_band "${!i}" $EXTENT "$(BAND $name)"
done
# The union of the dragon and the armadillo moved by 58 voxels along z: an input that shares the part of its
# surface outside the armadillo with the dragon.
bin/nanovdb_fields "$(BAND dragon)" "$(BAND armadillo)" 2 58 output/fields.bin "$(BAND dragon_or_armadillo)" > /dev/null

# input 1 : input 2 : axis : offset of input 2 in voxels
CASES="dragon:armadillo:2:205 dragon:armadillo:2:117 dragon:armadillo:2:58 dragon:armadillo:2:20 dragon:armadillo:2:0
       bunny:armadillo:0:60 bunny:armadillo:0:25 bunny:armadillo:0:0 armadillo:dragon:1:30 bunny:bunny:2:1
       bunny:bunny:0:0 armadillo:armadillo:2:0 dragon:dragon:2:0
       dragon:dragon_or_armadillo:2:0 dragon_or_armadillo:dragon:2:0"
: > output/correctness.txt
identical=0; total=0
for c in $CASES; do
  IFS=: read a b axis offset <<< "$c"
  total=$((total + 1))
  echo "=== $a, $b moved by $offset along axis $axis ===" >> output/correctness.txt
  bin/nanovdb_fields "$(BAND $a)" "$(BAND $b)" $axis $offset output/fields.bin >> output/correctness.txt
  if bin/openvdb_fields "$(BAND $a)" "$(BAND $b)" $axis $offset output/fields.bin >> output/correctness.txt; then
    identical=$((identical + 1))
  fi
done
rm -f output/fields.bin
echo "$identical of $total pairs identical to OpenVDB in every voxel (output/correctness.txt)"
[ $identical -eq $total ]
