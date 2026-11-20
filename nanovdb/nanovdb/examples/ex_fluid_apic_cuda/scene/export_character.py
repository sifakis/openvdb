# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""blender -b --python export_character.py -- --input FILE --output DIRECTORY"""

import argparse, hashlib, json, shutil, sys, time
from pathlib import Path
import bpy
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--input", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--height", type=float, default=1.75)
p.add_argument("--center-x", type=float, default=1.6)
p.add_argument("--center-z", type=float, default=1.2)
p.add_argument("--floor-clearance", type=float, default=0.035)
p.add_argument("--inspect-only", action="store_true")
a = p.parse_args(sys.argv[sys.argv.index("--") + 1 :])
a.output.mkdir(parents=True, exist_ok=True)
source = a.input.resolve()
local = (a.output / "source.fbx").resolve()
if source != local:
    shutil.copy2(source, local)
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=str(local))
scene = bpy.context.scene
objects = sorted([o for o in scene.objects if o.type == "MESH"], key=lambda o: o.name)
assert objects, "FBX contains no mesh"
actions = [(x.name, list(x.frame_range)) for x in bpy.data.actions]
assert actions, "FBX contains no animation action"
first = int(np.floor(min(r[0] for _, r in actions)))
last = int(np.ceil(max(r[1] for _, r in actions)))
fps = scene.render.fps / scene.render.fps_base
scene.frame_start = first
scene.frame_end = last
scene.frame_set(first)
deps = bpy.context.evaluated_depsgraph_get()
raw = []
triangles = []
parts = []
polygons = []
offset = 0
materials = []
for obj in objects:
    ev = obj.evaluated_get(deps)
    m = ev.to_mesh(preserve_all_data_layers=True, depsgraph=deps)
    m.calc_loop_triangles()
    co = np.empty(len(m.vertices) * 3, dtype=np.float32)
    m.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    matrix = np.array(ev.matrix_world, dtype=np.float64)
    raw.append(co @ matrix[:3, :3].T + matrix[:3, 3])
    tris = np.array([tuple(t.vertices) for t in m.loop_triangles], dtype=np.int32)
    polys = [tuple(v.vertices) for v in m.polygons]
    polygons.append(polys)
    triangles.append(tris + offset)
    mat = []
    for slot in obj.material_slots:
        material = slot.material
        color = list(material.diffuse_color[:3]) if material else [0.6, 0.6, 0.6]
        rough = 0.4
        metal = 0.0
        if material and material.use_nodes:
            for node in material.node_tree.nodes:
                if node.type == "BSDF_PRINCIPLED":
                    color = list(node.inputs["Base Color"].default_value[:3])
                    rough = float(node.inputs["Roughness"].default_value)
                    metal = float(node.inputs["Metallic"].default_value)
        mat.append(
            dict(
                name=material.name if material else "default",
                color=color,
                roughness=rough,
                metallic=metal,
            )
        )
    parts.append(
        dict(
            name=obj.name,
            vertex_offset=offset,
            vertex_count=len(co),
            triangle_offset=sum(len(t) for t in triangles[:-1]),
            triangle_count=len(tris),
            materials=mat,
            triangle_materials=[int(t.material_index) for t in m.loop_triangles],
            modifiers=[x.type for x in obj.modifiers],
            hidden_render=bool(obj.hide_render),
        )
    )
    materials.extend(mat)
    offset += len(co)
    ev.to_mesh_clear()
triangles = np.concatenate(triangles)
raw = np.concatenate(raw)
edges = np.sort(
    np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1
)
_, counts = np.unique(edges, axis=0, return_counts=True)
metadata = dict(
    source=str(source),
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    blender_version=bpy.app.version_string,
    source_fps=fps,
    source_frame_start=first,
    source_frame_end=last,
    source_actions=actions,
    source_scene_unit_scale=scene.unit_settings.scale_length,
    source_coordinate_system="Blender world, right-handed Z-up",
    source_initial_bounds=[raw.min(axis=0).tolist(), raw.max(axis=0).tolist()],
    vertex_count=offset,
    triangle_count=len(triangles),
    objects=parts,
    boundary_edges=int(np.sum(counts == 1)),
    nonmanifold_edges=int(np.sum(counts > 2)),
)
(a.output / "inspection.json").write_text(json.dumps(metadata, indent=2))
print(
    "INSPECTION",
    json.dumps({k: v for k, v in metadata.items() if k not in ["objects", "source_actions"]}),
    flush=True,
)
if a.inspect_only:
    sys.exit(0)
started = time.perf_counter()
frames = []
for frame in range(first, last + 1):
    scene.frame_set(frame)
    deps = bpy.context.evaluated_depsgraph_get()
    values = []
    for obj, part, expected in zip(objects, parts, polygons):
        ev = obj.evaluated_get(deps)
        m = ev.to_mesh(preserve_all_data_layers=True, depsgraph=deps)
        assert len(m.vertices) == part["vertex_count"], (frame, obj.name, "vertex count changed")
        assert [tuple(v.vertices) for v in m.polygons] == expected, (
            frame,
            obj.name,
            "polygon connectivity changed",
        )
        co = np.empty(len(m.vertices) * 3, dtype=np.float32)
        m.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        matrix = np.array(ev.matrix_world, dtype=np.float64)
        values.append(co @ matrix[:3, :3].T + matrix[:3, 3])
        ev.to_mesh_clear()
    frames.append(np.concatenate(values))
    if (frame - first) % 30 == 0:
        print("EVALUATED", frame, flush=True)
frames = np.array(frames, dtype=np.float32)
lo = frames.min(axis=(0, 1))
hi = frames.max(axis=(0, 1))
scale = a.height / (hi[2] - lo[2])
# right-handed Blender XYZ -> simulation XZ(-Y), then uniform scale/translation.
translation = np.array(
    [
        a.center_x - scale * (lo[0] + hi[0]) / 2,
        a.floor_clearance - scale * lo[2],
        a.center_z + scale * (lo[1] + hi[1]) / 2,
    ]
)
world = frames[:, :, [0, 2, 1]].copy()
world[:, :, 2] *= -1
world = world * scale + translation
triangles.astype("<i4").tofile(a.output / "triangles.bin")
world.astype("<f4").tofile(a.output / "vertices.bin")
np.save(a.output / "triangles.npy", triangles)
np.save(a.output / "vertices.npy", world.astype(np.float32))
face_text = "".join("f %d %d %d\n" % tuple(t + 1) for t in triangles)
face_hash = hashlib.sha256(face_text.encode()).hexdigest()
for k, points in enumerate(world):
    dest = a.output / f"frame_{k+1:06d}.obj"
    with dest.open("w") as f:
        f.write("# evaluated world-space mesh; units=m; Y-up; fixed connectivity\n")
        np.savetxt(f, points, fmt="v %.9g %.9g %.9g")
        f.write(face_text)
    # Parse actual exported connectivity, rather than only checking in-memory indices.
    with dest.open() as f:
        encoded = "".join(line for line in f if line.startswith("f "))
    assert hashlib.sha256(encoded.encode()).hexdigest() == face_hash
metadata.update(
    fps=fps,
    frame_start=1,
    frame_end=len(world),
    time_start=0,
    time_end=(len(world) - 1) / fps,
    coordinate_system="right-handed Y-up; source (x,y,z) -> (x,z,-y)",
    units="meters",
    normalization_scale=float(scale),
    simulation_translation=translation.tolist(),
    world_bounds=[world.min(axis=(0, 1)).tolist(), world.max(axis=(0, 1)).tolist()],
    triangle_sha256=hashlib.sha256(triangles.astype("<i4").tobytes()).hexdigest(),
    obj_connectivity_sha256=face_hash,
    topology_verified_frames=len(world),
    deformation_max_from_first=[
        float(np.linalg.norm(world[k] - world[0], axis=1).max())
        for k in [0, len(world) // 2, len(world) - 1]
    ],
    export_seconds=time.perf_counter() - started,
)
assert metadata["deformation_max_from_first"][1] > 0.01, "middle frame is undeformed"
metadata["maximum_adjacent_frame_motion"] = float(
    np.linalg.norm(world[1:] - world[:-1], axis=2).max()
)
metadata["last_frame_loop_error"] = metadata["deformation_max_from_first"][2]
assert metadata["maximum_adjacent_frame_motion"] > 0.01, "animation has no evaluated deformation"
(a.output / "metadata.json").write_text(json.dumps(metadata, indent=2))
(a.output / "sequence.txt").write_text(f"{len(world)} {offset} {len(triangles)} {fps:.12g}\n")
# Preserve the original rig, material graph and actions in addition to the untouched FBX copy.
scene.frame_set(first)
bpy.ops.file.pack_all()
bpy.ops.wm.save_as_mainfile(filepath=str((a.output / "source.blend").resolve()))
print(
    "PASS export",
    len(world),
    "frames",
    offset,
    "vertices",
    len(triangles),
    "triangles",
    metadata["export_seconds"],
    "s",
    flush=True,
)
