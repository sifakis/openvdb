#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

"""render.py <in.meshbin> <out.png> [options]

Renders a triangle mesh written by band_to_mesh (vertices in voxels, the mesh's y axis up) with the ovrtx SDK, path
traced: one glossy object standing on a floor, three rect lights and a dome. The mesh is scaled so that the longest
side of its bounding box is 1 m. The camera looks at the middle of the bounding box from the direction --view (in
the mesh's axes), from --distance m away.
Needs a Python with ovrtx, ovstage, pxr (usd-core), numpy and Pillow.
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from pxr import Usd, UsdGeom, UsdShade, UsdLux, UsdRender, Sdf, Gf, Vt


def read_meshbin(path):
    with path.open("rb") as stream:
        header = np.fromfile(stream, dtype="<u8", count=3)
        assert header[0] == 0x4D45534830303031
        nv, nf = int(header[1]), int(header[2])
        vertices = np.fromfile(stream, dtype="<f4", count=3 * nv).reshape(-1, 3)
        faces = np.fromfile(stream, dtype="<i4", count=3 * nf).reshape(-1, 3)
    return vertices, faces


def normals(vertices, faces):
    """Vertex normals: the area-weighted sum of the normals of the triangles around each vertex."""
    n = np.zeros_like(vertices)
    tri = vertices[faces]
    face = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for a in range(3):
        np.add.at(n, faces[:, a], face)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20)
    return n


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


def aim(xform, pos, target):
    xform.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*pos), Gf.Vec3d(*target), Gf.Vec3d(0, 1, 0)).GetInverse())


def author(args, dest):
    vertices, faces = read_meshbin(args.mesh)
    lo, hi = vertices.min(0), vertices.max(0)
    scale = 1.0 / float((hi - lo).max())
    points = (vertices - 0.5 * (lo + hi)) * scale
    points[:, 1] += 0.5 * (hi[1] - lo[1]) * scale  # the mesh stands on y = 0
    points = points.astype(np.float32)

    stage = Usd.Stage.CreateNew(str(dest.resolve()))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.SetStageUpAxis(stage, "Y")
    UsdGeom.SetStageMetersPerUnit(stage, 1)

    body = UsdGeom.Mesh.Define(stage, "/World/Body")
    body.CreateSubdivisionSchemeAttr("none")
    body.SetNormalsInterpolation("vertex")
    body.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    body.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.reshape(-1)))
    body.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32)))
    body.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals(points, faces)))
    body.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.array([points.min(0), points.max(0)], np.float32)))
    UsdShade.MaterialBindingAPI.Apply(body.GetPrim()).Bind(material(stage, "Body", args.color, args.roughness, args.metallic))

    floor = UsdGeom.Cube.Define(stage, "/World/Floor")
    floor.CreateSizeAttr(1)
    floor.AddTranslateOp().Set(Gf.Vec3d(0, -0.03, 0))
    floor.AddScaleOp().Set(Gf.Vec3f(30, 0.06, 30))
    UsdShade.MaterialBindingAPI.Apply(floor.GetPrim()).Bind(material(stage, "Floor", args.floor, 0.55, 0.0))

    target = np.array([0.0, 0.5 * (hi[1] - lo[1]) * scale, 0.0]) + np.array(args.shift)
    view = np.array(args.view, float)
    view /= np.linalg.norm(view)
    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    camera.CreateFocalLengthAttr(args.focal)
    camera.CreateHorizontalApertureAttr(36)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100))
    aim(camera, target + args.distance * view, target)

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
    stage.GetRootLayer().Save()


def render(scene, dest):
    import ovrtx, ovstage

    renderer = ovrtx.Renderer()
    stage = ovstage.Stage("hydra")
    renderer.attach_ovstage(stage)
    try:
        ovstage.population.open_usd(stage, str(scene.resolve()), ordinal=1, time_code=0.0)
        stage.advance_write_floor(1, ovstage.Scope.ALL).wait()
        products = renderer.step(render_products={"/Render/Camera"}, delta_time=1 / 24, ordinal=1)
        frame = products["/Render/Camera"].frames[0]
        if not frame.converged:
            raise RuntimeError(f"unconverged render: progression={frame.progression}")
        var = frame.render_vars["/Render/Camera/LdrColor"].map(device=ovrtx.Device.CPU)
        view = np.from_dlpack(var)
        pixels = view.copy()
        del view
        var.unmap()
        del var
        Image.fromarray(pixels).convert("RGB").save(dest)
        del frame, products
    finally:
        renderer.detach_ovstage()
        stage.destroy()
        renderer.destroy()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mesh", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--view", type=float, nargs=3, default=[0.1, 0.2, -1], help="direction from the object to the camera, in the mesh's axes")
    p.add_argument("--distance", type=float, default=2.6, help="in m; the object is 1 m long")
    p.add_argument("--shift", type=float, nargs=3, default=[0, 0, 0], help="moves the point the camera looks at, in m")
    p.add_argument("--focal", type=float, default=50)
    p.add_argument("--color", type=float, nargs=3, default=[0.08, 0.42, 0.38])
    p.add_argument("--roughness", type=float, default=0.22)
    p.add_argument("--metallic", type=float, default=0.0)
    p.add_argument("--floor", type=float, nargs=3, default=[0.78, 0.76, 0.72])
    p.add_argument("--dome", type=float, default=450)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--samples", type=int, default=128)
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    scene = args.out.with_suffix(".usdc")
    scene.unlink(missing_ok=True)
    author(args, scene)
    render(scene, args.out)
