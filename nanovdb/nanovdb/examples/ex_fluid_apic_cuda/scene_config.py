# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Simulation scale shared by validation and visualization; old outputs remain readable."""
import json


def load_config(folder):
    path = folder / 'scene.json'
    if path.exists():
        return json.loads(path.read_text())
    return dict(voxel_size=.05, domain_cells=[32, 32, 24],
                world_size=[1.6, 1.6, 1.2], obstacle_voxel_size=.025,
                contact_margin=.0125, marker_radius=.007)
