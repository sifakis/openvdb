# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Author a time-sampled 3D USD scene, render with ovrtx, encode real-time MP4."""
import argparse
import csv
import json
import subprocess
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from pxr import Usd, UsdGeom, Vt
from render import usd, debug
from scene_config import load_config

OFFSETS=np.array([(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)],dtype=np.float32)


def manifest(folder):
    with (folder/'frames.csv').open() as stream:
        return list(csv.DictReader(stream))


def author(folder, fps):
    records=manifest(folder)
    offsets = OFFSETS * load_config(folder)['marker_radius']
    base=usd(folder/'frame_0000.csv')
    stage=Usd.Stage.Open(str(base.resolve()))
    stage.SetTimeCodesPerSecond(fps)
    stage.SetFramesPerSecond(fps)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(float(records[-1]['time'])*fps)
    attr=UsdGeom.Mesh(stage.GetPrimAtPath('/World/Particles')).GetPointsAttr()
    extent=UsdGeom.Mesh(stage.GetPrimAtPath('/World/Particles')).CreateExtentAttr()
    obstacle=stage.GetPrimAtPath('/World/Obstacle')
    obstacle_attr=UsdGeom.Mesh(obstacle).GetPointsAttr() if obstacle else None
    for record in records:
        index=int(record['frame']);t=float(record['time'])*fps
        path=folder/f'frame_{index:04d}.csv'
        values=np.loadtxt(path,delimiter=',',skiprows=1,dtype=np.float32)
        vertices=(values[:,:3,None].transpose(0,2,1)+offsets[None,:,:]).reshape(-1,3)
        attr.Set(Vt.Vec3fArray.FromNumpy(vertices),t)
        extent.Set(Vt.Vec3fArray.FromNumpy(np.array([vertices.min(axis=0),vertices.max(axis=0)],dtype=np.float32)),t)
        if obstacle_attr:
            points=np.loadtxt(path.with_name(path.stem+'_obstacle.csv'),delimiter=',',skiprows=1,dtype=np.float32)
            obstacle_attr.Set(Vt.Vec3fArray.FromNumpy(points),t)
        if index%30==0:print('USD',folder.name,index,flush=True)
    dest=folder/'scene.usdc'
    stage.GetRootLayer().Export(str(dest.resolve()))
    # Reopen exported data and verify that animation samples were actually persisted.
    exported=Usd.Stage.Open(str(dest.resolve()))
    samples=UsdGeom.Mesh(exported.GetPrimAtPath('/World/Particles')).GetPointsAttr().GetTimeSamples()
    assert len(samples)==len(records)
    return dest


def encode(folder,fps,name='animation'):
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-framerate',str(fps),'-i',str(folder/'rtx'/'frame_%04d.png'),'-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(folder/(name+'.mp4'))],check=True)


def render_animation(folder,scene,fps,warmup,limit=None):
    import ovrtx
    import ovstage
    records=manifest(folder)
    h = load_config(folder)['voxel_size']
    if limit:records=records[:limit]
    images=folder/'rtx';images.mkdir(exist_ok=True)
    renderer=ovrtx.Renderer()
    stage=ovstage.Stage('nanovdb.fluid.animation')
    renderer.attach_ovstage(stage)
    last_pixels=None; differences=[]
    try:
        ovstage.population.open_usd(stage,str(scene.resolve()),ordinal=1,time_code=0)
        stage.advance_write_floor(1,ovstage.Scope.ALL).wait()
        for _ in range(warmup):
            renderer.step(render_products={'/Render/Camera'},delta_time=1/fps,ordinal=1)
        for i,record in enumerate(records):
            t=float(record['time']);ordinal=i+2
            # ovstage accepts seconds; USD time samples use t * timeCodesPerSecond.
            ovstage.population.update_from_usd_time(stage,ordinal=ordinal,time_code=t)
            stage.advance_write_floor(ordinal,ovstage.Scope.ALL).wait()
            # A few accumulation steps at this fixed pose are renderer warmup,
            # not additional simulation time or duplicated animation samples.
            for _ in range(4):
                products=renderer.step(render_products={'/Render/Camera'},delta_time=1/fps,ordinal=ordinal)
            product=products['/Render/Camera']
            frame=product.frames[0]
            var=frame.render_vars['/Render/Camera/LdrColor'].map(device=ovrtx.Device.CPU)
            view=np.from_dlpack(var);pixels=view.copy();del view;var.unmap();del var
            if last_pixels is not None:differences.append(float(np.mean(np.abs(pixels[:,:,:3].astype(np.float32)-last_pixels[:,:,:3]))))
            last_pixels=pixels
            image=Image.fromarray(pixels).convert('RGB')
            draw=ImageDraw.Draw(image);draw.rectangle((0,0,960,25),fill='#141d29')
            draw.text((12,7),f'{folder.name} | h = {h:.3f} m | simulation t = {t:.3f} s | particle and obstacle view',fill='white')
            image.save(images/f"frame_{int(record['frame']):04d}.png")
            print('RTX',folder.name,record['frame'],'time',t,flush=True)
        del frame,product,products
    finally:
        renderer.detach_ovstage();stage.destroy();renderer.destroy()
    assert differences and max(differences)>.1,'animation produced static images'
    return dict(frames=len(records),max_frame_pixel_change=max(differences),mean_frame_pixel_change=float(np.mean(differences)))


def debug_video(folder,fps):
    records=manifest(folder)
    h = load_config(folder)['voxel_size']
    images=folder/'debug';images.mkdir(exist_ok=True)
    for record in records:
        index=int(record['frame']);path=folder/f'frame_{index:04d}.csv';debug(path)
        image=Image.open(path.with_suffix('.png')).convert('RGB')
        draw=ImageDraw.Draw(image);draw.rectangle((0,0,960,25),fill='#141d29')
        draw.text((12,7),f"{folder.name} | h = {h:.3f} m | simulation t = {float(record['time']):.3f} s | particle slab / block halo",fill='white')
        image.save(images/f'frame_{index:04d}.png')
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-framerate',str(fps),'-i',str(images/'frame_%04d.png'),'-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(folder/'debug.mp4')],check=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('folder',type=Path)
    parser.add_argument('--fps',type=float,default=30)
    parser.add_argument('--warmup',type=int,default=40)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--author-only',action='store_true')
    parser.add_argument('--reuse-scene',action='store_true')
    args=parser.parse_args()
    times={};start=time.perf_counter()
    scene=args.folder/'scene.usdc' if args.reuse_scene else author(args.folder,args.fps)
    times['usd_seconds']=time.perf_counter()-start
    if not args.author_only:
        start=time.perf_counter();times['render_check']=render_animation(args.folder,scene,args.fps,args.warmup,args.limit);times['rtx_seconds']=time.perf_counter()-start
        encode(args.folder,args.fps)
        start=time.perf_counter();debug_video(args.folder,args.fps);times['debug_seconds']=time.perf_counter()-start
    (args.folder/'animation_times.json').write_text(json.dumps(times,indent=2)+'\n')
