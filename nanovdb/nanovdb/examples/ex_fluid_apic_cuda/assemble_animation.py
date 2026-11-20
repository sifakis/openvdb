# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Concatenate 3D movies, verify decoded videos, and make small previews."""
import argparse
import json
import subprocess
from pathlib import Path
from PIL import Image
from scene_config import load_config

parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,default=Path(__file__).parent/'output'/'long')
args=parser.parse_args()
root=args.output
names=['static','moving','deforming']
listing=root/'videos.txt'
listing.write_text('\n'.join("file '"+str((root/m/'animation.mp4').resolve())+"'" for m in names)+'\n')
subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','concat','-safe','0','-i',str(listing),'-c','copy','-movflags','+faststart',str(root/'demo.mp4')],check=True)
checks={}
expected_frames={}
for mode in names:
    with (root/mode/'animation_times.json').open() as stream:
        expected_frames[mode]=json.load(stream)['render_check']['frames']
expected_total=sum(expected_frames.values())
all_names=names+[m for m in ['stage1','stage2'] if (root/m/'animation_times.json').exists()]
for mode in all_names:
    with (root/mode/'animation_times.json').open() as stream:
        expected_frames[mode]=json.load(stream)['render_check']['frames']
for path in [root/m/k for m in all_names for k in ['animation.mp4','debug.mp4']]+[root/'demo.mp4']:
    result=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,width,height,r_frame_rate,nb_frames,duration','-of','json',str(path)],capture_output=True,text=True,check=True)
    data=json.loads(result.stdout)['streams'][0]
    expected=expected_total if path.name=='demo.mp4' else expected_frames[path.parent.name]
    assert int(data['nb_frames'])==expected,(path,data)
    fps=30 if path.name=='demo.mp4' else load_config(path.parent).get('fps',30)
    assert data['r_frame_rate']==f'{int(fps)}/1'
    assert data['width']==960 and data['height']==720
    subprocess.run(['ffmpeg','-v','error','-i',str(path),'-f','null','-'],check=True)
    checks[str(path.relative_to(root))]=data
(root/'video_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
for mode in all_names:
    folder=root/mode
    count=len(list((folder/'rtx').glob('frame_*.png')))
    preview=[]
    for n in range(0,count,2):
        image=Image.open(folder/'rtx'/f'frame_{n:04d}.png').convert('RGB')
        image.thumbnail((480,360));preview.append(image)
    fps=load_config(folder).get('fps',30)
    preview[0].save(folder/'preview.gif',save_all=True,append_images=preview[1:],duration=round(2000/fps),loop=0)
    sheet=Image.new('RGB',(1440,740),'#141d29')
    for i,n in enumerate([0,(count-1)//10,(count-1)//5,(count-1)//2,3*(count-1)//4,count-1]):
        image=Image.open(folder/'rtx'/f'frame_{n:04d}.png').convert('RGB')
        image.thumbnail((480,360));sheet.paste(image,((i%3)*480,(i//3)*370))
    sheet.save(folder/'overview.png')
print('PASS 3D/debug movie decode and frame counts')
