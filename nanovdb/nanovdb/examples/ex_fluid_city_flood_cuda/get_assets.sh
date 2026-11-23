#!/bin/bash
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# get_assets.sh: downloads NVIDIA's "AEC City Demo Pack" (about 200 MB, one of the Omniverse downloadable asset packs,
# https://docs.omniverse.nvidia.com/usd/latest/usd_content_samples/downloadable_packs.html) into assets/ and exports
# the city (buildings, streets, the river's bed and banks) as one triangle mesh in the fluid tank's frame:
# output/city.obj (about 15 MB, 207,268 triangles).
# The pack is not part of this repository; its license file (assets/PACKAGE-LICENSES/) says it is bound by the NVIDIA
# Omniverse terms.
# RENDER_PYTHON: a Python with pxr (usd-core) and numpy (default: python3).
set -eu
cd "$(dirname "$0")"
PY=${RENDER_PYTHON:-python3}
CITY=assets/Demos/AEC/TowerDemo/CityDemopack/World_CityDemopack.usd
mkdir -p assets output
if [ ! -f $CITY ]; then
  curl -L -o assets/city_demo.zip "https://d4i3qtqj3r0z5.cloudfront.net/AECO_CityDemoPack_NVD%4010011.zip"
  unzip -q -o assets/city_demo.zip -d assets
  rm assets/city_demo.zip
fi
$PY -I tools/export_city.py $CITY output/city.obj
