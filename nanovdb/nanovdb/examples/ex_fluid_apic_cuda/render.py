# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""CSV -> local USD/ovrtx, plus orthographic particle/block diagnostics."""
import argparse
import csv
import math
from pathlib import Path
from PIL import Image, ImageDraw
from scene_config import load_config


def read(path):
    with open(path) as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def debug(path):
    config = load_config(path.parent)
    h = config['voxel_size']
    slice_index = .6 / h
    particles = read(path)
    image = Image.new('RGB', (960, 720), '#141d29')
    draw = ImageDraw.Draw(image)
    project = lambda p: (80 + p[0] * 380, 670 - p[1] * 380)
    blocks_path = path.with_name(path.stem + '_blocks.csv')
    if blocks_path.exists():
        for b in read(blocks_path):
            if b['z'] <= slice_index < b['z'] + 8:
                x, y = b['x'] * h, b['y'] * h
                draw.rectangle([project((x, y+8*h)), project((x+8*h, y))], outline='#cf923d' if b['fluid_cells'] else '#3c626b', width=2)
    draw.rectangle([project((0, 1.6)), project((1.6, 0))], outline='white', width=2)
    for p in particles:
        if .5 < p['z'] < .7:
            x, y = project((p['x'], p['y']))
            speed = math.sqrt(sum(p[k]**2 for k in ('vx', 'vy', 'vz')))
            color = (min(255, int(45+speed*55)), 170, 245)
            draw.ellipse((x-1.4,y-1.4,x+1.4,y+1.4), fill=color)
    obstacle = path.with_name(path.stem+'_obstacle.csv')
    if obstacle.exists():
        v = read(obstacle)
        xmin, xmax = min(p['x'] for p in v), max(p['x'] for p in v)
        ymin, ymax = min(p['y'] for p in v), max(p['y'] for p in v)
        draw.rectangle([project((xmin,ymax)),project((xmax,ymin))],outline='#ff765a',width=3)
    draw.text((50, 12), path.parent.name+' / '+path.stem+f' | h = {h:.3f} m | X-Y slice 0.5 < z < 0.7', fill='white')
    draw.text((50, 32), 'orange: blocks containing fluid | cyan: halo blocks | white: container | red: obstacle', fill='white')
    image.save(path.with_suffix('.png'))


def usd(path):
    config = load_config(path.parent)
    radius = config['marker_radius']
    vertices, indices = [], []
    offsets = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    triangles = [(0,2,4),(4,2,1),(1,2,5),(5,2,0),(4,3,0),(1,3,4),(5,3,1),(0,3,5)]
    for p in read(path):
        n = len(vertices)
        vertices.extend(tuple(p[k]+radius*d for k,d in zip(('x','y','z'),off)) for off in offsets)
        indices.extend(n+i for tri in triangles for i in tri)
    def mesh(name, verts, ids, color):
        return f'''def Mesh "{name}" {{
point3f[] points = {verts}
int[] faceVertexCounts = {[3]*(len(ids)//3)}
int[] faceVertexIndices = {ids}
uniform token subdivisionScheme = "none"
color3f[] primvars:displayColor = [{color}]
}}\n'''
    text = '#usda 1.0\n(defaultPrim = "World"\n metersPerUnit = 1\n upAxis = "Y")\ndef Xform "World" {\n'
    text += mesh('Particles', vertices, indices, (.08,.4,.8))
    obstacle = path.with_name(path.stem+'_obstacle.csv')
    if obstacle.exists():
        verts = [tuple(p[k] for k in ('x','y','z')) for p in read(obstacle)]
        ids = [0,2,3,0,3,1,4,5,7,4,7,6,0,1,5,0,5,4,2,6,7,2,7,3,0,4,6,0,6,2,1,3,7,1,7,5]
        text += mesh('Obstacle', verts, ids, (.85,.15,.05))
    for a in range(3):
        others = [b for b in range(3) if b != a]
        for j in range(4):
            position = [0., 0., 0.]
            scale = [.003, .003, .003]
            dims = config['world_size']
            position[a] = dims[a]/2
            scale[a] = dims[a]
            for n, b in enumerate(others):
                position[b] = dims[b] if ((j >> n) & 1) else 0.
            text += f'def Cube "Edge{a}{j}" {{\n double size = 1\n double3 xformOp:translate = {tuple(position)}\n double3 xformOp:scale = {tuple(scale)}\n uniform token[] xformOpOrder = ["xformOp:translate","xformOp:scale"]\n color3f[] primvars:displayColor = [(0.15,0.17,0.2)]\n}}\n'
    text += '''def Cube "Floor" {
 double size = 1
 double3 xformOp:translate = (0.8,-0.04,0.6)
 double3 xformOp:scale = (1.65,0.08,1.25)
 uniform token[] xformOpOrder = ["xformOp:translate","xformOp:scale"]
 color3f[] primvars:displayColor = [(0.3,0.32,0.35)]
}
def Camera "Camera" {
 float2 clippingRange = (0.01,100)
 float focalLength = 40
 float horizontalAperture = 36
 double3 xformOp:translate = (2.6,2.0,3.5)
 float3 xformOp:rotateXYZ = (-20,30,0)
 uniform token[] xformOpOrder = ["xformOp:translate","xformOp:rotateXYZ"]
}
def DistantLight "Light" {
 float inputs:intensity = 3000
 float3 xformOp:rotateXYZ = (-35,-25,0)
 uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
}
def DomeLight "Fill" {
 float inputs:intensity = 500
}
}
def Scope "Render" {
 def RenderProduct "Camera" {
 rel camera = </World/Camera>
 int2 resolution = (960,720)
 rel orderedVars = <LdrColor>
 def RenderVar "LdrColor" {
 string sourceName = "LdrColor"
 }
 }
}
'''
    dest = path.with_suffix('.usda')
    dest.write_text(text)
    return dest


def render(path):
    import ovrtx
    import ovstage
    import numpy as np
    scene = usd(path)
    print("Creating renderer", flush=True)
    renderer = ovrtx.Renderer()
    stage = ovstage.Stage('nanovdb.fluid')
    renderer.attach_ovstage(stage)
    print("Opening local USD", flush=True)
    ovstage.population.open_usd(stage, str(scene.resolve()), ordinal=1)
    stage.advance_write_floor(1, ovstage.Scope.ALL).wait()
    for k in range(1, 9):
        if k > 1:
            stage.advance_write_floor(k, ovstage.Scope.ALL).wait()
        print("Rendering", k, flush=True)
        products = renderer.step(render_products={'/Render/Camera'}, delta_time=1/60, ordinal=k)
        for product in products.values():
            for frame in product.frames:
                var = frame.render_vars['/Render/Camera/LdrColor'].map(device=ovrtx.Device.CPU)
                view = np.from_dlpack(var)
                pixels = view.copy()
                del view
                var.unmap()
                del var
    Image.fromarray(pixels).save(path.with_name(path.stem+'_rtx.png'))
    del frame, product, products
    renderer.detach_ovstage()
    stage.destroy()
    renderer.destroy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('frames', nargs='+', type=Path)
    parser.add_argument('--rtx', action='store_true')
    parser.add_argument('--gif', type=Path)
    args = parser.parse_args()
    images = []
    for path in args.frames:
        if '_blocks' in path.stem or '_obstacle' in path.stem:
            continue
        debug(path)
        if args.rtx:
            render(path)
        if args.gif:
            images.append(Image.open(path.with_suffix('.png')))
    if args.gif:
        images[0].save(args.gif, save_all=True, append_images=images[1:], duration=100, loop=0)
