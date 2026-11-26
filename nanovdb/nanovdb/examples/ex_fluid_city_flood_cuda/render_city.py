#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

"""render_city.py <out dir> [--frames DIR --voxel V] [--camera x y z --target x y z] ...

Renders the Omniverse sample city (AEC City Demo pack, World_CityDemopack.usd, referenced as it is with its own
materials; --city or ./get_assets.sh's assets/) and, if given, the water meshes of a run (DIR/f_000.meshbin, ...
written by mesh_band) and its drops, with the ovrtx SDK: a USD stage with time-sampled meshes, OmniGlass water, a sun
and a dome, a path-traced render product; then one ovrtx.Renderer step and one PNG per frame.
Everything is in the scene's world frame, in meters: x 0..1911, y up from the street level, z 0..1764;
tools/export_city.py writes the obstacle mesh in the same frame. A water vertex in cell-center coordinates c goes to
(c + 0.5) * voxel + origin (origin 0 -21 0 for build_city.py's tank).
--box x0 y0 z0 x1 y1 z1 ...: plain concrete boxes, for the basin's walls and the flood wall.
--walls FILE: an OBJ triangle mesh drawn as the same concrete: the walls along the bank, which build_city.py writes
as <scene>.bankwalls.obj.
Needs a Python with ovrtx, ovstage, pxr (usd-core), numpy and Pillow.
"""
import argparse, time
from pathlib import Path
import numpy as np
from PIL import Image
from pxr import Usd, UsdGeom, UsdShade, UsdLux, UsdRender, Sdf, Gf, Vt

CITY = Path(__file__).resolve().parent / "assets/Demos/AEC/TowerDemo/CityDemopack/World_CityDemopack.usd"


def read_meshbin(path):
    with path.open("rb") as stream:
        header = np.fromfile(stream, dtype="<u8", count=3)
        assert header[0] == 0x4D45534830303031
        nv, nf = int(header[1]), int(header[2])
        vertices = np.fromfile(stream, dtype="<f4", count=3 * nv).reshape(-1, 3)
        faces = np.fromfile(stream, dtype="<i4", count=3 * nf).reshape(-1, 3)
    return vertices, faces


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
        m.CreateOutput("mdl:" + output, Sdf.ValueTypeNames.Token).ConnectToSource(s.ConnectableAPI(), "out")
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


def mesh(stage, name, mat):
    m = UsdGeom.Mesh.Define(stage, "/World/" + name)
    m.CreateSubdivisionSchemeAttr("none")
    m.SetNormalsInterpolation("vertex")
    m.CreateNormalsAttr()
    m.CreateExtentAttr()
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
    return m


def mesh_sample(m, points, faces, t=None):
    args = () if t is None else (t,)
    m.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points), *args)
    m.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(faces.reshape(-1)), *args)
    m.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32)), *args)
    m.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals(points, faces)), *args)
    extent = np.array([points.min(0), points.max(0)], np.float32) if len(points) else np.zeros((2, 3), np.float32)
    m.GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(extent), *args)


def cube(stage, name, pos, size, mat):
    c = UsdGeom.Cube.Define(stage, "/World/" + name)
    c.CreateSizeAttr(1)
    c.AddTranslateOp().Set(Gf.Vec3d(*pos))
    c.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(mat)


def aim(xform, pos, target):
    xform.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*pos), Gf.Vec3d(*target), Gf.Vec3d(0, 1, 0)).GetInverse())


def read_spray(path):
    """The drops of a frame (written by the solver with BAND and SPRAY): positions in cell-center coordinates, volumes in
    cells, how many drops each written one stands for, and per drop a number in [0, 1) that it keeps for life (None
    for the older files, which do not have the drops' terminal speeds to derive it from)."""
    raw = np.fromfile(path, np.int32, 2)
    size = path.stat().st_size
    if size == 8 + 20 * int(raw[0]):
        rows = np.fromfile(path, np.float32, 5 * int(raw[0]), offset=8).reshape(-1, 5)
        h = np.ascontiguousarray(rows[:, 4]).view(np.uint32) ^ np.uint32(0x9E3779B9)  # an integer hash ("lowbias32")
        h ^= h >> 16; h *= np.uint32(0x7FEB352D); h ^= h >> 15; h *= np.uint32(0x846CA68B); h ^= h >> 16
        return rows[:, :3], rows[:, 3], int(raw[1]), (h >> 8) / 16777216.0
    if size == 8 + 16 * int(raw[0]):
        count, every, offset = int(raw[0]), int(raw[1]), 8
    else:  # the first format: the count alone
        count, every, offset = int(raw[0]), 1, 4
    rows = np.fromfile(path, np.float32, 4 * count, offset=offset).reshape(-1, 4)
    return rows[:, :3], rows[:, 3], every, None


def in_water(centers, path, reach=0.0):
    """Which drops are in the water or within `reach` cells of its surface, from the frame's band file (BAND: the cells
    within 2 voxels of the surface, with the level set's value there, negative in the water). A drop's cell has the
    sign of the nearest band cell of its column (same x and z): no surface lies between the two; the distance to the
    surface is taken as the distance to that cell. A drop in a column without band cells is in the air."""
    count = int(np.fromfile(path, "<i4", 1)[0])
    band = np.fromfile(path, np.dtype([("cell", "<i2", 3), ("value", "<f4")]), count, offset=16)
    if not count:
        return np.zeros(len(centers), bool)

    def keys(cell):  # one number per cell: columns in order, and within a column from the bottom up
        c = cell.astype(np.int64) + 32768
        return c[:, 0] << 32 | c[:, 2] << 16 | c[:, 1]

    key = keys(band["cell"])
    order = np.argsort(key)
    key, value = key[order], band["value"][order]
    mine = keys(np.floor(centers + 0.5))
    above = np.minimum(np.searchsorted(key, mine), count - 1)  # the first band cell at or above the drop's cell
    below = np.maximum(above - 1, 0)
    far = 1 << 20

    def gap(k):  # cells between the drop's cell and band cell k, if they are in the same column
        return np.where(key[k] >> 16 == mine >> 16, np.abs(key[k] - mine), far)

    nearest = np.where(gap(above) <= gap(below), above, below)
    return (gap(nearest) < far) & ((value[nearest] < 0) | (gap(nearest) <= reach))


def column_water(cells, path):
    """For each cell (n x 3 integers), how many cells of its column (same x and z) the frame's band file has with
    water in them (value < 0). A layer one cell deep has 1, two cells deep 2; deep water has 4 or more (two under
    the surface and two over the ground)."""
    count = int(np.fromfile(path, "<i4", 1)[0])
    band = np.fromfile(path, np.dtype([("cell", "<i2", 3), ("value", "<f4")]), count, offset=16)
    wet = band["cell"][band["value"] < 0].astype(np.int64) + 32768
    key, number = np.unique(wet[:, 0] << 16 | wet[:, 2], return_counts=True)
    mine = (cells[:, 0].astype(np.int64) + 32768) << 16 | (cells[:, 2].astype(np.int64) + 32768)
    at = np.minimum(np.searchsorted(key, mine), max(len(key) - 1, 0))
    return np.where(key[at] == mine, number[at], 0) if len(key) else np.zeros(len(cells), np.int64)


def drop_mesh(centers, radii):
    """Every drop as an octahedron: 6 vertices and 8 triangles."""
    directions = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)], np.float32)
    tris = np.array([(0, 2, 4), (4, 2, 1), (1, 2, 5), (5, 2, 0), (4, 3, 0), (1, 3, 4), (5, 3, 1), (0, 3, 5)], np.int32)
    v = (centers[:, None, :] + radii[:, None, None] * directions[None]).reshape(-1, 3).astype(np.float32)
    f = (tris[None] + 6 * np.arange(len(centers), dtype=np.int32)[:, None, None]).reshape(-1, 3)
    return v, f


def author(args, frames, dest, first=0):
    stage = Usd.Stage.CreateNew(str(dest.resolve()))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.SetStageUpAxis(stage, "Y")
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    stage.SetTimeCodesPerSecond(args.fps)
    stage.SetFramesPerSecond(args.fps)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(len(frames) + 1)  # past the last sample: the renderer does not converge on the end time code

    city = UsdGeom.Xform.Define(stage, "/World/City")
    city.GetPrim().GetReferences().AddReference(str(args.city.resolve()))
    # The city stage is in centimeters with z up: (x, y, z) there is (0.01 x + 760, 0.01 z, 773 - 0.01 y) here.
    city.AddTransformOp().Set(Gf.Matrix4d(0.01, 0, 0, 0, 0, 0, -0.01, 0, 0, 0.01, 0, 0, 760, 0, 773, 1))
    if frames:
        water = material(stage, "Water", tuple(args.water[:3]), args.water_roughness, water=True)
        shader = UsdShade.Shader(stage.GetPrimAtPath("/World/Looks/Water/Shader"))
        shader.GetInput("depth").Set(args.water[3])  # the distance over which the water takes its color
        if args.thin:
            shader.GetInput("thin_walled").Set(True)
        liquid = mesh(stage, "Water", water)
        # Shadow rays do not refract: without this the floor under the water gets no direct light and the water is black.
        UsdGeom.PrimvarsAPI(liquid.GetPrim()).CreatePrimvar("doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
    if frames and args.spray:
        foam = material(stage, "Foam", tuple(args.spray_color), 0.6)
        spray = mesh(stage, "Spray", foam)
        UsdGeom.PrimvarsAPI(spray.GetPrim()).CreatePrimvar("doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
    if args.box:  # x0 y0 z0 x1 y1 z1: a plain box, e.g. the wall that holds the water back
        gray = material(stage, "Concrete", (0.55, 0.55, 0.53), 0.8)
        for k in range(0, len(args.box), 6):
            b = args.box[k:k + 6]
            cube(stage, f"Box{k // 6}", [(b[a] + b[a + 3]) / 2 for a in range(3)], [b[a + 3] - b[a] for a in range(3)], gray)
    if args.walls:  # a triangle mesh (OBJ: v and f lines) drawn as concrete with flat faces: the walls along the bank
        lines = [line.split() for line in open(args.walls)]
        points = np.array([[float(v) for v in l[1:4]] for l in lines if l and l[0] == "v"], np.float32)
        faces = np.array([[int(v.split("/")[0]) - 1 for v in l[1:4]] for l in lines if l and l[0] == "f"], np.int32)
        walls = UsdGeom.Mesh.Define(stage, "/World/Walls")
        walls.CreateSubdivisionSchemeAttr("none")
        walls.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
        walls.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, np.int32)))
        walls.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.reshape(-1)))
        walls.CreateDoubleSidedAttr(True)
        walls.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.array([points.min(0), points.max(0)], np.float32)))
        UsdShade.MaterialBindingAPI.Apply(walls.GetPrim()).Bind(material(stage, "WallConcrete", (0.55, 0.55, 0.53), 0.8))

    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    camera.CreateFocalLengthAttr(args.focal)
    camera.CreateHorizontalApertureAttr(36)
    camera.CreateClippingRangeAttr(Gf.Vec2f(1.0, 6000))
    if args.camera2:  # the camera moves from (camera, target) at frame 0 to (camera2, target2) at frame total - 1
        op = camera.AddTransformOp()
        for index in range(max(1, len(frames))):
            t = min(1.0, (first + index) / max(1, args.total - 1))
            t = t * t * (3 - 2 * t)  # ease in and out
            pos = [a + (b - a) * t for a, b in zip(args.camera, args.camera2)]
            tgt = [a + (b - a) * t for a, b in zip(args.target, args.target2 or args.target)]
            op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*pos), Gf.Vec3d(*tgt), Gf.Vec3d(0, 1, 0)).GetInverse(), index)
    else:
        aim(camera, args.camera, args.target)
    if args.sun:  # elevation and azimuth in degrees, intensity: a distant light
        elevation, azimuth, power = args.sun
        sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
        sun.CreateIntensityAttr(power)
        sun.CreateAngleAttr(1.0)
        sun.CreateColorAttr(Gf.Vec3f(1.0, 0.95, 0.88))
        e, q = np.radians(elevation), np.radians(azimuth)
        toward = Gf.Vec3d(float(np.cos(e) * np.cos(q)), float(np.sin(e)), float(np.cos(e) * np.sin(q)))  # where the light comes from
        sun.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(toward * 1000.0, Gf.Vec3d(0, 0, 0), Gf.Vec3d(0, 1, 0)).GetInverse())
    dome = UsdLux.DomeLight.Define(stage, "/World/Ambient")
    dome.CreateIntensityAttr(args.dome)
    dome.CreateColorAttr(Gf.Vec3f(0.75, 0.82, 0.92))

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

    origin = np.asarray(args.origin, np.float32)
    for index, path in enumerate(frames):
        v, f = read_meshbin(path)
        w = ((v + 0.5) * args.voxel + origin).astype(np.float32)
        if args.water_min_cells:  # a layer thinner than this many cells is not drawn (the solver cannot carry it: it flickers)
            deep = column_water(np.floor(v[f].mean(axis=1) + 0.5).astype(np.int64), args.spray / (path.stem + ".band")) >= args.water_min_cells
            f = np.ascontiguousarray(f[deep])
        mesh_sample(liquid, w, f, index)
        if args.spray:  # the drops as white octahedra; a written drop stands for `every` drops of its volume
            centers, volume, every, lot = read_spray(args.spray / (path.stem + ".spray"))
            keep = np.ones(len(centers), bool)
            if args.spray_min_depth:  # only drops over water at least this many cells deep
                keep &= column_water(np.floor(centers + 0.5).astype(np.int64), args.spray / (path.stem + ".band")) >= args.spray_min_depth
            if args.spray_above is not None:  # drops lying on the streets are wet spots, not spray
                keep &= (centers[:, 1] + 0.5) * args.voxel + origin[1] > args.spray_above
            if args.spray_in_air is not None:  # a drop in the water or just over it is merging into it: it is not drawn
                keep &= ~in_water(centers, args.spray / (path.stem + ".band"), args.spray_in_air / args.voxel)
            left = int(keep.sum())
            if left > args.spray_max:  # by the drop's own number if it has one: the same drops in every frame
                if lot is None:
                    keep[np.random.default_rng(index + first).choice(np.flatnonzero(keep), left - args.spray_max, replace=False)] = False
                else:
                    keep &= lot < args.spray_max / left
                every = every * left / args.spray_max
            centers, volume = centers[keep], volume[keep]
            radius = np.cbrt(0.75 / np.pi * np.maximum(volume, 1e-6) * every) * args.voxel * args.spray_scale
            dv, df = drop_mesh((centers + 0.5) * args.voxel + origin, np.maximum(radius, args.spray_min))
            mesh_sample(spray, dv, df, index)
    stage.GetRootLayer().Save()


def render(args, scene, count, dest, first):
    import ovrtx, ovstage

    renderer = ovrtx.Renderer()
    stage = ovstage.Stage("city.flood")
    renderer.attach_ovstage(stage)
    try:
        ovstage.population.open_usd(stage, str(scene.resolve()), ordinal=1, time_code=0.0)
        stage.advance_write_floor(1, ovstage.Scope.ALL).wait()
        for i in range(count):
            start = time.perf_counter()
            ordinal = i + 2
            ovstage.population.update_from_usd_time(stage, ordinal=ordinal, time_code=float(i))
            stage.advance_write_floor(ordinal, ovstage.Scope.ALL).wait()
            renderer.reset(time=i / args.fps)
            for attempt in range(6):  # after the camera moved, the first step can come back unconverged
                products = renderer.step(render_products={"/Render/Camera"}, delta_time=1 / args.fps, ordinal=ordinal)
                frame = products["/Render/Camera"].frames[0]
                if frame.converged:
                    break
            if not frame.converged:
                raise RuntimeError(f"unconverged render at frame {i}: progression={frame.progression}")
            var = frame.render_vars["/Render/Camera/LdrColor"].map(device=ovrtx.Device.CPU)
            view = np.from_dlpack(var)
            pixels = view.copy()
            del view
            var.unmap()
            del var
            Image.fromarray(pixels).convert("RGB").save(dest / f"frame_{first + i:04d}.png")
            print("RTX", first + i, round(time.perf_counter() - start, 2), flush=True)
        del frame, products
    finally:
        renderer.detach_ovstage()
        stage.destroy()
        renderer.destroy()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--frames", type=Path)
    p.add_argument("--city", type=Path, default=CITY)  # World_CityDemopack.usd of the AEC City Demo pack
    p.add_argument("--voxel", type=float, default=0.25)
    p.add_argument("--origin", type=float, nargs=3, default=[0, 0, 0])  # where tank cell-corner (0, 0, 0) is, in meters
    p.add_argument("--camera", type=float, nargs=3, default=[775, 320, 1180])
    p.add_argument("--target", type=float, nargs=3, default=[775, 0, 650])
    p.add_argument("--focal", type=float, default=35)
    p.add_argument("--camera2", type=float, nargs=3)
    p.add_argument("--target2", type=float, nargs=3)
    p.add_argument("--total", type=int, default=240)  # frames of the whole movie (for the camera move)
    p.add_argument("--box", type=float, nargs="*", default=[])
    p.add_argument("--walls", type=Path)  # an OBJ triangle mesh drawn as concrete
    p.add_argument("--dome", type=float, default=800)
    p.add_argument("--sun", type=float, nargs=3)  # elevation, azimuth (degrees), intensity
    p.add_argument("--water", type=float, nargs=4, default=[0.72, 0.90, 0.97, 4.0])  # color r g b, depth in m
    p.add_argument("--water-roughness", type=float, default=0.02)  # of the water's surface: blurs reflections and what is seen through it
    p.add_argument("--spray", type=Path)  # the folder with the run's f_NNN.spray files: the drops are drawn
    p.add_argument("--spray-color", type=float, nargs=3, default=[0.95, 0.97, 0.98])
    p.add_argument("--water-min-cells", type=int, default=0)  # water layers with fewer cells in a column are not drawn (needs --spray <folder> for the band files)
    p.add_argument("--spray-min-depth", type=int, default=0)  # drops are drawn only over water with at least this many cells in the column
    p.add_argument("--spray-above", type=float)  # only drops higher than this (world y, in meters) are drawn
    p.add_argument("--spray-in-air", type=float)  # drops in the water or less than this many meters from its surface are not drawn (needs the .band files beside the .spray files)
    p.add_argument("--spray-scale", type=float, default=1.0)   # factor on a drop's radius
    p.add_argument("--spray-min", type=float, default=0.012)   # smallest drawn radius, in meters
    p.add_argument("--spray-max", type=int, default=400000)    # at most this many drops are drawn (a random choice)
    p.add_argument("--thin", action="store_true")  # the water as a thin-walled surface: no refraction, light passes
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--only-frame", type=int)
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--count", type=int, default=0)
    args = p.parse_args()
    frames = sorted(args.frames.glob("f_*.meshbin")) if args.frames else []
    first = 0
    if args.only_frame is not None and frames:
        frames, first = [frames[args.only_frame]], args.only_frame
    elif args.count and frames:  # the frames numbered first .. first + count - 1
        first = args.first
        frames = [args.frames / f"f_{i:03d}.meshbin" for i in range(first, first + args.count)]
    args.out.mkdir(parents=True, exist_ok=True)
    scene = args.out / "scene.usdc"
    scene.unlink(missing_ok=True)
    author(args, frames, scene, first)
    render(args, scene, max(1, len(frames)), args.out, first)
