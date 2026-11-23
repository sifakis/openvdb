#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""render_flow.py <mesh dir> <out dir> [options]: renders frames of a mean curvature flow (f_NNN.meshbin written by
mesh_band, vertices in voxels) with ovrtx, path traced: one glossy object over a floor, three rect lights and a dome.
The scene is scaled so that the longest side of the run's first frame is 1 m; the floor sits under that frame. The
camera looks at the middle of that frame's bounding box from --view (a direction in the mesh's axes) at --distance
meters. --up names the mesh axis that points up (x, y or z, with an optional minus sign).
Render one frame per run (--only k), as make_movie.sh does: with several frames in one run the renderer was seen to
draw the last frame's mesh for all of them.
Needs a Python with ovrtx, ovstage, pxr (usd-core), numpy and Pillow.
"""
import argparse, time
from pathlib import Path
import numpy as np
from PIL import Image
from pxr import Usd, UsdGeom, UsdLux, UsdRender, UsdShade, Sdf, Gf, Vt


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


def material(stage, name, color, roughness, metallic):
    m = UsdShade.Material.Define(stage, "/World/Looks/" + name)
    s = UsdShade.Shader.Define(stage, str(m.GetPath()) + "/Shader")
    s.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    s.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    for output in ("surface", "displacement", "volume"):
        m.CreateOutput("mdl:" + output, Sdf.ValueTypeNames.Token).ConnectToSource(s.ConnectableAPI(), "out")
    s.CreateOutput("out", Sdf.ValueTypeNames.Token)
    s.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    s.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(roughness)
    s.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(metallic)
    return m


def to_scene(args):
    """Rotation that takes the mesh's axes to the scene's (y up)."""
    up = np.zeros(3)
    up["xyz".index(args.up[-1])] = -1.0 if args.up.startswith("-") else 1.0
    forward = np.zeros(3)
    forward[next(a for a in range(3) if up[a] == 0)] = 1.0  # any axis that is not up
    right = np.cross(up, forward)
    return np.stack([right, up, forward])  # rows: scene x, y, z in mesh coordinates


def author(args, frames, reference, dest):
    rotation = to_scene(args)
    first, _ = read_meshbin(reference)
    first = first @ rotation.T
    lo, hi = first.min(0), first.max(0)
    scale = 1.0 / float((hi - lo).max())
    middle = 0.5 * (lo + hi)

    def place(v):
        p = (v @ rotation.T - middle) * scale
        p[:, 1] += (middle[1] - lo[1]) * scale  # the first frame stands on y = 0
        return p.astype(np.float32)

    stage = Usd.Stage.CreateNew(str(dest.resolve()))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.SetStageUpAxis(stage, "Y")
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    stage.SetTimeCodesPerSecond(args.fps)
    stage.SetFramesPerSecond(args.fps)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(len(frames) - 1)

    body = mesh(stage, "Body", material(stage, "Body", args.color, args.roughness, args.metallic))
    cube(stage, "Floor", (0, -0.03, 0), (30, 0.06, 30), material(stage, "Floor", args.floor, 0.55, 0.0))

    height = (hi[1] - lo[1]) * scale
    target = np.array([0.0, 0.5 * height, 0.0]) + np.array(args.shift)
    view = np.array(args.view, float) @ rotation.T
    view /= np.linalg.norm(view)
    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    camera.CreateFocalLengthAttr(args.focal)
    camera.CreateHorizontalApertureAttr(36)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100))
    position = target + args.distance * view
    aim(camera, position, target)

    # The lights sit relative to the camera: key from its upper left, rim from behind the object, fill from its right.
    side = np.cross(view, [0, 1, 0])
    side /= np.linalg.norm(side)
    for name, pos, power, color, size in [
        ("Key", target + 1.6 * view - 1.2 * side + [0, 1.4, 0], 2600, (1, 0.94, 0.86), 1.2),
        ("Rim", target - 1.4 * view + 0.9 * side + [0, 1.1, 0], 2200, (0.62, 0.80, 1), 0.9),
        ("Fill", target + 1.3 * view + 1.5 * side + [0, 0.5, 0], 700, (0.80, 0.88, 1), 0.9),
    ]:
        light = UsdLux.RectLight.Define(stage, "/World/" + name)
        light.CreateIntensityAttr(power)
        light.CreateColorAttr(Gf.Vec3f(*color))
        light.CreateWidthAttr(size)
        light.CreateHeightAttr(size)
        aim(light, pos, target)
    dome = UsdLux.DomeLight.Define(stage, "/World/Ambient")
    dome.CreateIntensityAttr(args.dome)
    dome.CreateColorAttr(Gf.Vec3f(0.62, 0.70, 0.82))

    product = UsdRender.Product.Define(stage, "/Render/Camera")
    product.CreateCameraRel().SetTargets([camera.GetPath()])
    product.CreateResolutionAttr(Gf.Vec2i(args.width, args.height))
    var = UsdRender.Var.Define(stage, "/Render/Camera/LdrColor")
    var.CreateSourceNameAttr("LdrColor")
    product.CreateOrderedVarsRel().SetTargets([var.GetPath()])
    for name, typ, value in [
        ("omni:rtx:rendermode", Sdf.ValueTypeNames.Token, "PathTracing"),
        ("omni:rtx:pt:samplesPerPixel", Sdf.ValueTypeNames.UInt, args.samples),
        ("omni:rtx:pt:denoising:enabled", Sdf.ValueTypeNames.Bool, True),
        ("omni:rtx:pt:maxBounces", Sdf.ValueTypeNames.UInt, 8),
    ]:
        product.GetPrim().CreateAttribute(name, typ).Set(value)

    for index, path in enumerate(frames):
        v, f = read_meshbin(path)
        if args.flip:
            f = f[:, [0, 2, 1]]
        if np.linalg.det(rotation) < 0:
            f = f[:, [0, 2, 1]]  # the axis change mirrors the mesh
        mesh_sample(body, place(v), f, index)
    stage.GetRootLayer().Save()


def render(args, scene, names, dest):
    import ovrtx, ovstage

    renderer = ovrtx.Renderer()
    stage = ovstage.Stage("flow")
    renderer.attach_ovstage(stage)
    try:
        ovstage.population.open_usd(stage, str(scene.resolve()), ordinal=1, time_code=0.0)
        stage.advance_write_floor(1, ovstage.Scope.ALL).wait()
        for i, name in enumerate(names):
            start = time.perf_counter()
            ordinal = i + 2
            ovstage.population.update_from_usd_time(stage, ordinal=ordinal, time_code=float(i))
            stage.advance_write_floor(ordinal, ovstage.Scope.ALL).wait()
            renderer.reset(time=i / args.fps)  # animated topology: clear the path tracer's history
            products = renderer.step(render_products={"/Render/Camera"}, delta_time=1 / args.fps, ordinal=ordinal)
            frame = products["/Render/Camera"].frames[0]
            if not frame.converged:
                raise RuntimeError(f"unconverged render at frame {i}: progression={frame.progression}")
            var = frame.render_vars["/Render/Camera/LdrColor"].map(device=ovrtx.Device.CPU)
            view = np.from_dlpack(var)
            pixels = view.copy()
            del view
            var.unmap()
            del var
            Image.fromarray(pixels).convert("RGB").save(dest / f"frame_{name}.png")
            print("RTX", name, round(time.perf_counter() - start, 2), flush=True)
        del frame, products
    finally:
        renderer.detach_ovstage()
        stage.destroy()
        renderer.destroy()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("frames", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--only", type=int, nargs="*", help="frame numbers to render (default: all)")
    p.add_argument("--up", default="y")
    p.add_argument("--view", type=float, nargs=3, default=[1, 0.35, 0.6], help="direction from the object to the camera, mesh axes")
    p.add_argument("--distance", type=float, default=2.2)
    p.add_argument("--shift", type=float, nargs=3, default=[0, 0, 0], help="moves the point the camera looks at, in m")
    p.add_argument("--focal", type=float, default=50)
    p.add_argument("--color", type=float, nargs=3, default=[0.08, 0.42, 0.38])
    p.add_argument("--roughness", type=float, default=0.22)
    p.add_argument("--metallic", type=float, default=0.0)
    p.add_argument("--floor", type=float, nargs=3, default=[0.78, 0.76, 0.72])
    p.add_argument("--dome", type=float, default=450)
    p.add_argument("--flip", action="store_true", help="reverse the triangles")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--fps", type=int, default=24)
    args = p.parse_args()
    frames = sorted(args.frames.glob("f_*.meshbin"))
    reference = frames[0]  # the scale and the floor come from the first frame of the run, whichever frames are rendered
    if args.only:
        frames = [f for f in frames if int(f.stem[2:]) in args.only]
    args.out.mkdir(parents=True, exist_ok=True)
    scene = args.out / "scene.usdc"
    scene.unlink(missing_ok=True)
    author(args, frames, reference, scene)
    render(args, scene, [f.stem[2:] for f in frames], args.out)
