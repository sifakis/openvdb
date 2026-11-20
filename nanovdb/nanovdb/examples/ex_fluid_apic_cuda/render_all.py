# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Render initial/middle/final views and all particle/block debug sequences."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from PIL import Image, ImageDraw

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, default=Path(__file__).parent/'output')
parser.add_argument('--rtx', action='store_true', help='Use a Python environment with ovrtx/ovstage installed')
parser.add_argument('--reuse-rtx', action='store_true', help='Build galleries from already rendered images')
args = parser.parse_args()
script = Path(__file__).with_name('render.py')
stages = [('stage1',80),('stage2',80),('static',160),('moving',200),('deforming',200)]
timings = {}
for stage, end in stages:
    folder = args.output/stage
    frames = sorted(p for p in folder.glob('frame_*.csv') if '_blocks' not in p.stem and '_obstacle' not in p.stem)
    start = time.perf_counter()
    subprocess.run([sys.executable,str(script),'--gif',str(folder/'sequence.gif'),*map(str,frames)],check=True)
    timings[stage+'_debug_seconds'] = time.perf_counter()-start
    representative = [folder/f'frame_{n:04d}.csv' for n in (0,end//2,end)]
    if args.rtx:
        start = time.perf_counter()
        subprocess.run([sys.executable,str(script),'--rtx',*map(str,representative)],check=True)
        timings[stage+'_rtx_seconds'] = time.perf_counter()-start
    for suffix in ('', '_rtx') if args.rtx or args.reuse_rtx else ('',):
        gallery = Image.new('RGB',(1440,390),'#141d29')
        draw = ImageDraw.Draw(gallery)
        for i, frame in enumerate(representative):
            image = Image.open(frame.with_name(frame.stem+suffix+'.png')).convert('RGB')
            image.thumbnail((480,360))
            gallery.paste(image,(480*i,30))
            draw.text((480*i+12,8),stage+' / '+frame.stem,fill='white')
        gallery.save(folder/('overview'+suffix+'.png'))
(args.output/'render_times.json').write_text(json.dumps(timings,indent=2)+'\n')
