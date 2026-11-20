# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""CPU Blender diagnostic of cached geometry, independent of the ovrtx material.
blender -b --python SCRIPT -- --cache CACHE --character SEQUENCE --surface SURFACE --frame 90 --output image.png
"""
import argparse
import sys
from pathlib import Path
import bpy
import numpy as np
from mathutils import Vector


def coordinates(p):
    return np.stack((p[:, 0], -p[:, 2], p[:, 1]), axis=1)


def material(name, color, water=False):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    node = m.node_tree.nodes.get('Principled BSDF')
    node.inputs['Base Color'].default_value = (*color, 1)
    node.inputs['Roughness'].default_value = .015 if water else .5
    node.inputs['IOR'].default_value = 1.333
    node.inputs['Transmission Weight'].default_value = 1 if water else 0
    return m


def mesh(name, v, f, mat):
    data = bpy.data.meshes.new(name)
    data.from_pydata(coordinates(v), [], f)
    data.update()
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mat)
    for face in data.polygons:
        face.use_smooth = True
    return obj


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--surface', type=Path, required=True)
    p.add_argument('--character', type=Path, required=True)
    p.add_argument('--frame', type=int, default=90)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--style', choices=('clay', 'water', 'particles'), default='clay')
    p.add_argument('--samples', type=int, default=48)
    a = p.parse_args(sys.argv[sys.argv.index('--')+1:])
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = a.samples
    scene.cycles.use_denoising = True
    scene.render.threads_mode = 'FIXED'
    scene.render.threads = 8
    scene.render.resolution_x = 1200
    scene.render.resolution_y = 1200
    scene.render.resolution_percentage = 100
    scene.world.use_nodes = True
    scene.world.node_tree.nodes['Background'].inputs[0].default_value = (.22, .25, .3, 1)
    scene.world.node_tree.nodes['Background'].inputs[1].default_value = .6
    character = np.load(a.character/'vertices.npy', mmap_mode='r')
    faces = np.load(a.character/'triangles.npy')
    mesh('Character', character[a.frame], faces, material('Character', (.55, .17, .12)))
    fluid_material = material('Fluid', (.06, .32, .48) if a.style!='water' else (.98, .995, 1), a.style=='water')
    if a.style == 'particles':
        raw = np.loadtxt(a.cache/f'frame_{a.frame:06d}.csv', delimiter=',', skiprows=1, usecols=(1,2,3))
        raw = raw[(raw[:,1]>.25)&(raw[:,1]<2.1)]
        direction=np.array([(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)])*.003
        tris=np.array([(0,2,4),(4,2,1),(1,2,5),(5,2,0),(4,3,0),(1,3,4),(5,3,1),(0,3,5)])
        mesh('APIC centers (3mm glyphs)',(raw[:,None,:]+direction).reshape(-1,3),
             (tris[None,:,:]+6*np.arange(len(raw))[:,None,None]).reshape(-1,3),fluid_material)
    else:
        with (a.surface/f'water_{a.frame:06d}.meshbin').open('rb') as f:
            header=np.fromfile(f,dtype='<u8',count=3)
            assert header[0]==0x4d45534830303031
            v=np.fromfile(f,dtype='<f4',count=int(header[1])*3).reshape(-1,3)
            faces=np.fromfile(f,dtype='<i4',count=int(header[2])*3).reshape(-1,3)
        mesh('Identical cached surface',v,faces,fluid_material)
    bpy.ops.mesh.primitive_plane_add(size=200)
    bpy.context.object.data.materials.append(material('Floor',(.08,.095,.12)))
    camera=bpy.data.cameras.new('Camera');obj=bpy.data.objects.new('Camera',camera)
    bpy.context.collection.objects.link(obj)
    # World-space Y-up inputs converted to Blender Z-up.
    obj.location=(3.05,-3.5,2.05);target=Vector((1.65,-1.15,1.10))
    obj.rotation_euler=(target-obj.location).to_track_quat('-Z','Y').to_euler()
    camera.lens=55;scene.camera=obj
    for name,position,power,size in [('Key',(-1,-2.5,4),1200,3),('Rim',(3,1,3),900,2)]:
        light=bpy.data.lights.new(name,'AREA');light.energy=power;light.shape='DISK';light.size=size
        obj=bpy.data.objects.new(name,light);bpy.context.collection.objects.link(obj);obj.location=position
        obj.rotation_euler=(Vector((1.6,-1.15,1))-obj.location).to_track_quat('-Z','Y').to_euler()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    scene.render.filepath=str(a.output.resolve())
    bpy.ops.render.render(write_still=True)


if __name__=='__main__':main()
