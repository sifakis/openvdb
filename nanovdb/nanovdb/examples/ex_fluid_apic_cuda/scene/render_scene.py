# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Local USD meshes + separate whitewater, rendered with the unmodified ovrtx SDK."""

import argparse, csv, json, time, subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from pxr import Usd, UsdGeom, UsdShade, UsdLux, UsdRender, Sdf, Gf, Vt


def character_pose(sequence, time_seconds, fps):
    u = np.clip(time_seconds * fps, 0, len(sequence) - 1)
    f = min(int(u), len(sequence) - 2)
    s = u - f

    def tangent(i):
        lo = max(0, i - 1)
        hi = min(len(sequence) - 1, i + 1)
        return (sequence[hi] - sequence[lo]) / (hi - lo)

    return np.asarray(
        sequence[f] * (2 * s**3 - 3 * s * s + 1)
        + tangent(f) * (s**3 - 2 * s * s + s)
        + sequence[f + 1] * (-2 * s**3 + 3 * s * s)
        + tangent(f + 1) * (s**3 - s * s),
        dtype=np.float32,
    )


def read_obj(path, prefer_binary=False):
    binary = path.with_suffix(".meshbin")
    if prefer_binary and binary.exists():
        with binary.open("rb") as stream:
            header = np.fromfile(stream, dtype="<u8", count=3)
            assert len(header) == 3 and header[0] == 0x4D45534830303031
            nv, nf = map(int, header[1:])
            assert binary.stat().st_size == 24 + 12 * (nv + nf)
            vertices = np.fromfile(stream, dtype="<f4", count=3 * nv).reshape(-1, 3)
            faces = np.fromfile(stream, dtype="<i4", count=3 * nf).reshape(-1, 3)
        return vertices, faces
    vertices = []
    faces = []
    with path.open() as stream:
        for line in stream:
            row = line.split()
            if row and row[0] == "v":
                vertices.append([float(x) for x in row[1:4]])
            elif row and row[0] == "f":
                faces.append([int(x.split("/")[0]) - 1 for x in row[1:4]])
    return np.asarray(vertices, dtype=np.float32).reshape(-1, 3), np.asarray(
        faces, dtype=np.int32
    ).reshape(-1, 3)


def table(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def normals(vertices, faces):
    n = np.zeros_like(vertices)
    if len(faces):
        tri = vertices[faces]
        face = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        for a in range(3):
            np.add.at(n, faces[:, a], face)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20)
    return n


def material(stage, name, color, roughness=0.3, metallic=0, water=False):
    m = UsdShade.Material.Define(stage, "/World/Looks/" + name)
    s = UsdShade.Shader.Define(stage, str(m.GetPath()) + "/Shader")
    module = "OmniGlass" if water else "OmniPBR"
    s.SetSourceAsset(Sdf.AssetPath(module + ".mdl"), "mdl")
    s.SetSourceAssetSubIdentifier(module, "mdl")
    for output in ("surface", "displacement", "volume"):
        m.CreateOutput("mdl:" + output, Sdf.ValueTypeNames.Token).ConnectToSource(
            s.ConnectableAPI(), "out"
        )
    s.CreateOutput("out", Sdf.ValueTypeNames.Token)
    if water:
        s.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        s.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(1.333)
        s.CreateInput("depth", Sdf.ValueTypeNames.Float).Set(0.4)
        s.CreateInput("frosting_roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        s.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(False)
    else:
        s.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        s.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(roughness)
        s.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(metallic)
    return m


def mesh_sample(mesh, points, faces, t):
    mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points), t)
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(faces.reshape(-1)), t)
    mesh.GetFaceVertexCountsAttr().Set(
        Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32)), t
    )
    mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals(points, faces)), t)
    extent = (
        np.array([points.min(0), points.max(0)], np.float32)
        if len(points)
        else np.zeros((2, 3), np.float32)
    )
    mesh.GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(extent), t)


def mesh(stage, name, mat):
    m = UsdGeom.Mesh.Define(stage, "/World/" + name)
    m.CreateSubdivisionSchemeAttr("none")
    m.SetNormalsInterpolation("vertex")
    m.CreateNormalsAttr()
    m.CreateExtentAttr()
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
    return m


def cube(stage, name, pos, size, mat):
    c = UsdGeom.Cube.Define(stage, "/World/" + name)
    c.CreateSizeAttr(1)
    c.AddTranslateOp().Set(Gf.Vec3d(*pos))
    c.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(mat)


def aim(xform, pos, target):
    xform.AddTransformOp().Set(
        Gf.Matrix4d().SetLookAt(Gf.Vec3d(*pos), Gf.Vec3d(*target), Gf.Vec3d(0, 1, 0)).GetInverse()
    )


def spheres(rows, radius_scale=1.0):
    """Render secondary particles; scaling never changes cached collision radii."""
    if not np.isfinite(radius_scale) or radius_scale <= 0:
        raise ValueError("Secondary render radius scale must be positive and finite")
    directions = np.array(
        [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)], np.float32
    )
    tris = np.array(
        [(0, 2, 4), (4, 2, 1), (1, 2, 5), (5, 2, 0), (4, 3, 0), (1, 3, 4), (5, 3, 1), (0, 3, 5)],
        np.int32,
    )
    if not rows:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.int32)
    # One subdivision avoids the tiny projected area of the old octahedra.
    # Apply an optional appearance-only scale around each cached particle center.
    vertices, edges, refined = list(directions), {}, []

    def midpoint(a, b):
        key = tuple(sorted((a, b)))
        if key not in edges:
            v = directions[a] + directions[b]
            v /= np.linalg.norm(v)
            edges[key] = len(vertices)
            vertices.append(v)
        return edges[key]

    for a, b, c in tris:
        ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
        refined.extend(((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)))
    directions = np.array(vertices, np.float32)
    tris = np.array(refined, np.int32)
    pos = np.array([[float(r[a]) for a in ("x", "y", "z")] for r in rows], np.float32)
    radii = np.array([float(r["radius"]) for r in rows], np.float32) * radius_scale
    return (pos[:, None, :] + radii[:, None, None] * directions).reshape(-1, 3), (
        tris[None, :, :] + len(directions) * np.arange(len(rows), dtype=np.int32)[:, None, None]
    ).reshape(-1, 3)


def author(args):
    folder = args.cache
    radius_scales = [getattr(args, "foam_render_scale", 1.0),
                     getattr(args, "spray_render_scale", 1.0)]
    if any(not np.isfinite(s) or s <= 0 for s in radius_scales):
        raise ValueError("Secondary render radius scales must be positive and finite")
    if radius_scales != [1.0, 1.0] and not getattr(args, "tag", ""):
        raise ValueError("Whitewater size variants require --tag to preserve the original render")
    settings = json.loads((folder / "settings.json").read_text())
    meta = json.loads((args.character / "metadata.json").read_text())
    sequence = np.load(args.character / "vertices.npy", mmap_mode="r")
    triangles = np.load(args.character / "triangles.npy")
    records = table(folder / "frames.csv")
    if args.only_frame is not None:
        records = [r for r in records if int(r["frame"]) == args.only_frame]
    if getattr(args, "last_frame", None) is not None:
        if not args.tag:
            raise ValueError("A partial movie requires --tag to preserve the full render")
        records = [r for r in records if int(r["frame"]) <= args.last_frame]
    if not records:
        raise ValueError("No cached frames match the requested range")
    suffix = "" if args.whitewater_mode == "beauty" else "_" + args.whitewater_mode
    if getattr(args, "camera_view", "full") != "full":
        suffix += "_" + args.camera_view
    if getattr(args, "tag", ""):
        suffix += "_" + args.tag
    if getattr(args, "water_style", "water") != "water":
        suffix += "_" + args.water_style
    dest = folder / (("preview" if args.only_frame is not None else "scene") + suffix + ".usdc")
    dest.unlink(missing_ok=True)
    stage = Usd.Stage.CreateNew(str(dest.resolve()))
    stage.GetRootLayer().customLayerData = {
        "foamRenderRadiusScale": radius_scales[0],
        "sprayRenderRadiusScale": radius_scales[1],
    }
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.SetStageUpAxis(stage, "Y")
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    stage.SetTimeCodesPerSecond(settings["fps"])
    stage.SetFramesPerSecond(settings["fps"])
    stage.SetStartTimeCode(float(records[0]["time"]) * settings["fps"])
    stage.SetEndTimeCode(float(records[-1]["time"]) * settings["fps"])
    style = getattr(args, "water_style", "water")
    if style == "clay":
        water = material(stage, "Water", (0.12, 0.45, 0.65), 0.5)
    else:
        water = material(stage, "Water", (0.86, 0.97, 0.99), 0.025, water=True)
    foam = material(stage, "Foam", (0.98, 0.98, 0.98), 0.65)
    floor = material(stage, "Stage", (0.09, 0.12, 0.15), 0.32, 0.25)
    metal = material(stage, "Chrome", (0.32, 0.37, 0.42), 0.18, 0.85)
    black = material(stage, "Drain", (0.008, 0.01, 0.014), 0.5)
    warm = material(stage, "Rim", (0.13, 0.085, 0.045), 0.25, 0.6)
    liquid = mesh(stage, "Water", water)
    foamMesh = mesh(stage, "Foam", foam)
    sprayMesh = mesh(stage, "Spray", water)
    if args.whitewater_mode == "nofoam":
        UsdGeom.Imageable(foamMesh).CreateVisibilityAttr("invisible")
    if args.whitewater_mode == "hidden":
        foamMesh.CreateVisibilityAttr("invisible")
        sprayMesh.CreateVisibilityAttr("invisible")
    elif args.whitewater_mode == "diagnostic":
        liquid.CreateVisibilityAttr("invisible")
        spray_debug = material(stage, "SprayDebug", (1.0, 0.25, 0.025), 0.4)
        UsdShade.MaterialBindingAPI.Apply(sprayMesh.GetPrim()).Bind(spray_debug)
    bodies = []
    for obj in meta["objects"]:
        # FBX contains untextured, uniform materials. Preserve their base values.
        m = obj["materials"][0]
        bodies.append(
            mesh(
                stage,
                obj["name"],
                material(
                    stage, obj["name"] + "Material", m["color"], m["roughness"], m["metallic"]
                ),
            )
        )
    cube(stage, "Floor", (1.6, -0.055, 1.2), (3.2, 0.11, 2.4), floor)
    cube(
        stage,
        "Backdrop",
        (1.6, -0.15, 1.2),
        (2000, 0.1, 2000),
        material(stage, "Background", (0.065, 0.085, 0.11), 0.8),
    )
    if settings.get("drain_enabled", True):
        for a in range(2):
            cube(
                stage,
                "DrainX" + str(a),
                (0.07 if a == 0 else 3.13, 0.001, 1.2),
                (0.12, 0.002, 2.4),
                black,
            )
            cube(
                stage,
                "DrainZ" + str(a),
                (1.6, 0.001, 0.07 if a == 0 else 2.33),
                (3.2, 0.002, 0.12),
                black,
            )
    else:
        # Display the lower, wetted portion of the closed computational walls.
        # Inner glass faces coincide with x=0/3.2 and z=0/2.4, not with a sink strip.
        basin = material(stage, "BasinGlass", (0.98, 0.99, 1), 0.03, water=True)
        UsdShade.Shader.Get(stage, "/World/Looks/BasinGlass/Shader").GetInput("glass_ior").Set(1.45)
        height, thickness = 0.30, 0.018
        for a in range(2):
            cube(
                stage,
                "BasinX" + str(a),
                (-thickness / 2 if a == 0 else 3.2 + thickness / 2, height / 2, 1.2),
                (thickness, height, 2.4 + 2 * thickness),
                basin,
            )
            cube(
                stage,
                "BasinZ" + str(a),
                (1.6, height / 2, -thickness / 2 if a == 0 else 2.4 + thickness / 2),
                (3.2, height, thickness),
                basin,
            )
    # Visible nozzle; the outlet coincides with the simulation emitter.
    nozzle = UsdGeom.Cylinder.Define(stage, "/World/Nozzle")
    nozzle.CreateAxisAttr("Y")
    nozzle.CreateRadiusAttr(settings["emitter_radius"] + 0.025)
    nozzle.CreateHeightAttr(0.09)
    nozzle.AddTranslateOp().Set(
        Gf.Vec3d(settings["emitter_center"][0], 2.61, settings["emitter_center"][2])
    )
    UsdShade.MaterialBindingAPI.Apply(nozzle.GetPrim()).Bind(metal)
    cube(
        stage,
        "Pipe",
        (settings["emitter_center"][0], 2.95, settings["emitter_center"][2]),
        (0.075, 0.6, 0.075),
        metal,
    )
    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    camera.CreateFocalLengthAttr(46)
    camera.CreateHorizontalApertureAttr(36)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000))
    if getattr(args, "camera_view", "full") == "feet":
        aim(camera, (2.25, 0.60, 2.35), (1.55, 0.065, 1.15))
    elif getattr(args, "camera_view", "full") == "body":
        aim(camera, (3.25, 2.15, 3.80), (1.65, 1.05, 1.15))
    else:
        aim(camera, (4.45, 2.9, 5.55), (1.6, 1.28, 1.2))
    for name, pos, target, power, color, width, height in [
        ("Key", (-1.8, 4.5, 3.2), (1.6, 1, 1.2), 3400, (1, 0.9, 0.8), 3, 3),
        ("Rim", (2.5, 3, -1.5), (1.6, 1.3, 1.2), 2300, (0.5, 0.75, 1), 2, 3),
        ("Strip", (4.5, 2, 1), (1.6, 1, 1.2), 1300, (0.7, 0.85, 1), 0.7, 3),
    ]:
        light = UsdLux.RectLight.Define(stage, "/World/" + name)
        light.CreateIntensityAttr(power)
        light.CreateColorAttr(Gf.Vec3f(*color))
        light.CreateWidthAttr(width)
        light.CreateHeightAttr(height)
        aim(light, pos, target)
    dome = UsdLux.DomeLight.Define(stage, "/World/Ambient")
    dome.CreateIntensityAttr(300)
    dome.CreateColorAttr(Gf.Vec3f(0.48, 0.58, 0.75))
    product = UsdRender.Product.Define(stage, "/Render/Camera")
    product.CreateCameraRel().SetTargets([camera.GetPath()])
    product.CreateResolutionAttr(Gf.Vec2i(args.width, args.height))
    var = UsdRender.Var.Define(stage, "/Render/Camera/LdrColor")
    var.CreateSourceNameAttr("LdrColor")
    product.CreateOrderedVarsRel().SetTargets([var.GetPath()])
    prim = product.GetPrim()
    for name, typ, value in [
        ("omni:rtx:rendermode", Sdf.ValueTypeNames.Token, "PathTracing"),
        ("omni:rtx:pt:samplesPerPixel", Sdf.ValueTypeNames.UInt, args.samples),
        ("omni:rtx:pt:denoising:enabled", Sdf.ValueTypeNames.Bool, True),
        ("omni:rtx:pt:maxBounces", Sdf.ValueTypeNames.UInt, 12),
    ]:
        prim.CreateAttribute(name, typ).Set(value)
    surface_dir = getattr(args, "surface_dir", None) or folder / "surface"
    for row in records:
        index = int(row["frame"])
        seconds = float(row["time"])
        tc = seconds * settings["fps"]
        points = character_pose(
            sequence, seconds + settings["animation_offset"], meta["source_fps"]
        )
        for obj, body in zip(meta["objects"], bodies):
            v0 = obj["vertex_offset"]
            v1 = v0 + obj["vertex_count"]
            t0 = obj["triangle_offset"]
            t1 = t0 + obj["triangle_count"]
            mesh_sample(body, points[v0:v1], triangles[t0:t1] - v0, tc)
        v, f = read_obj(surface_dir / f"water_{index:06d}.obj", prefer_binary=True)
        mesh_sample(liquid, v, f, tc)
        secondary_path = surface_dir / f"whitewater_{index:06d}.csv"
        secondary = table(
            secondary_path
            if secondary_path.exists()
            else folder / f"frame_{index:06d}_whitewater.csv"
        )
        for kind, body in [(0, foamMesh), (1, sprayMesh)]:
            v, f = spheres([q for q in secondary if int(q["kind"]) == kind], radius_scales[kind])
            mesh_sample(body, v, f, tc)
        print("USD", index, flush=True)
    stage.GetRootLayer().Save()
    return dest, records, settings


def render(args, scene, records, settings):
    import ovrtx, ovstage

    render_start = time.perf_counter()
    renderer = ovrtx.Renderer()
    stage = ovstage.Stage("fluid.dance")
    renderer.attach_ovstage(stage)
    suffix = "" if args.whitewater_mode == "beauty" else "_" + args.whitewater_mode
    if getattr(args, "camera_view", "full") != "full":
        suffix += "_" + args.camera_view
    if getattr(args, "tag", ""):
        suffix += "_" + args.tag
    if getattr(args, "water_style", "water") != "water":
        suffix += "_" + args.water_style
    dest = args.cache / (("preview" if args.only_frame is not None else "render") + suffix)
    dest.mkdir(exist_ok=True)
    timings = []
    convergence = []
    phase_timings = []
    try:
        ovstage.population.open_usd(
            stage,
            str(scene.resolve()),
            ordinal=1,
            time_code=getattr(args, "initial_time", float(records[0]["time"])),
        )
        stage.advance_write_floor(1, ovstage.Scope.ALL).wait()
        load_seconds = time.perf_counter() - render_start
        for i, row in enumerate(records):
            start = time.perf_counter()
            ordinal = i + 2
            ovstage.population.update_from_usd_time(
                stage, ordinal=ordinal, time_code=float(row["time"])
            )
            stage.advance_write_floor(ordinal, ovstage.Scope.ALL).wait()
            update_seconds = time.perf_counter() - start
            trace_start = time.perf_counter()
            # Animated topology can return an unconverged first PT pass after
            # an update. Offline frames must each reach the authored sample
            # target; clear history and align the renderer clock to this cache.
            renderer.reset(time=float(row["time"]))
            products = renderer.step(
                render_products={"/Render/Camera"}, delta_time=1 / settings["fps"], ordinal=ordinal
            )
            frame = products["/Render/Camera"].frames[0]
            trace_seconds = time.perf_counter() - trace_start
            readback_start = time.perf_counter()
            if not frame.converged:
                raise RuntimeError(f"Unconverged render at frame {row['frame']}: progression={frame.progression}")
            convergence.append({"frame": int(row["frame"]), "progression": frame.progression,
                                "converged": frame.converged})
            var = frame.render_vars["/Render/Camera/LdrColor"].map(device=ovrtx.Device.CPU)
            view = np.from_dlpack(var)
            pixels = view.copy()
            del view
            var.unmap()
            del var
            readback_seconds = time.perf_counter() - readback_start
            elapsed = time.perf_counter() - start
            png_start = time.perf_counter()
            Image.fromarray(pixels).convert("RGB").save(dest / f"frame_{int(row['frame']):06d}.png")
            phase_timings.append({"frame": int(row["frame"]), "usd_update_seconds": update_seconds,
                                  "render_step_seconds": trace_seconds, "readback_seconds": readback_seconds,
                                  "png_seconds": time.perf_counter() - png_start})
            timings.append(elapsed)
            print("RTX", row["frame"], elapsed, flush=True)
        del frame, products
    finally:
        renderer.detach_ovstage()
        stage.destroy()
        renderer.destroy()
    encode_seconds = 0.0
    if len(records) > 1:
        encode_start = time.perf_counter()
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                str(settings["fps"]),
                "-start_number",
                str(int(records[0]["frame"])),
                "-i",
                str(dest / "frame_%06d.png"),
                "-c:v",
                "libx264",
                "-crf",
                "17",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(args.cache / ("dance_water" + suffix + ".mp4")),
            ],
            check=True,
        )

        encode_seconds = time.perf_counter() - encode_start
    (dest / "timings.json").write_text(json.dumps({
        "frame_seconds": timings,
        "frame_convergence": convergence,
        "reset_history_per_frame": True,
        "samples": args.samples,
        "resolution": [args.width, args.height],
        "foam_render_radius_scale": getattr(args, "foam_render_scale", 1.0),
        "spray_render_radius_scale": getattr(args, "spray_render_scale", 1.0),
        "author_seconds": getattr(args, "author_seconds", None),
        "renderer_load_seconds": load_seconds,
        "frame_phases": phase_timings,
        "encode_seconds": encode_seconds,
        "render_function_wall_seconds": time.perf_counter() - render_start,
    }, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cache", type=Path)
    p.add_argument("--character", type=Path, required=True)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=960)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--only-frame", type=int)
    p.add_argument("--last-frame", type=int, help="Render a committed prefix; requires --tag")
    p.add_argument("--camera-view", choices=("full", "feet", "body"), default="full")
    p.add_argument("--surface-dir", type=Path)
    p.add_argument("--tag", default="")
    p.add_argument("--water-style", choices=("water", "clay"), default="water")
    p.add_argument("--author-only", action="store_true")
    p.add_argument("--foam-render-scale", type=float, default=1.0,
                   help="Appearance-only foam radius multiplier; requires --tag if changed")
    p.add_argument("--spray-render-scale", type=float, default=1.0,
                   help="Appearance-only spray radius multiplier; requires --tag if changed")
    p.add_argument(
        "--whitewater-mode", choices=("beauty", "hidden", "diagnostic", "nofoam"), default="beauty"
    )
    args = p.parse_args()
    author_start = time.perf_counter()
    scene, records, settings = author(args)
    args.author_seconds = time.perf_counter() - author_start
    if not args.author_only:
        render(args, scene, records, settings)
