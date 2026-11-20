# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Compare the same 42 cm emitter at two solver resolutions after frame 90 commits."""
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from PIL import Image, ImageDraw

EX = Path(__file__).resolve().parent.parent
REPO = EX.parents[3]
BINS = REPO/'build/release/nanovdb/nanovdb/examples'
DIAG = EX/'output/stream_diagnosis'
CHARACTER = EX/'output/character_grounded'


def run(name, command):
    with (DIAG/(name+'.log')).open('w') as log:
        subprocess.run(list(map(str, command)), stdout=log, stderr=subprocess.STDOUT, check=True)


def wait_frame(folder, frame):
    start = time.monotonic()
    while True:
        path=folder/'surface/meshing.csv'
        if path.exists():
            rows=list(csv.DictReader(path.open()))
            if rows and int(rows[-1]['frame'])>=frame:
                return
        if time.monotonic()-start>7200:
            raise RuntimeError('Timed out waiting for preview geometry: '+str(folder))
        log=DIAG/'fine_simulation.log'
        if log.exists() and 'FAIL:' in log.read_text():
            raise RuntimeError('Simulation validation failed; see '+str(log))
        time.sleep(5)


def main():
    records={}
    for name, h in [('coarse',.032),('fine',.02)]:
        folder=EX/'output'/('scene_clear_shower_'+name)
        wait_frame(folder,90)
        result=json.loads(subprocess.check_output([str(BINS/'ex_fluid_apic_surface_inspect'),str(folder),str(folder/'surface'),'90',str(h*.5),str(h*.5),'.02'],text=True))
        records[name]=result
        (DIAG/(name+'_cut_geometry.json')).write_text(json.dumps(result,indent=2)+'\n')
        run(name+'_cut_clay',[sys.executable,EX/'scene/render_scene.py',folder,'--character',CHARACTER,'--only-frame',90,'--camera-view','body','--water-style','clay','--whitewater-mode','hidden','--width',1280,'--height',960,'--samples',64])
        run(name+'_cut_preview',[sys.executable,EX/'scene/render_scene.py',folder,'--character',CHARACTER,'--only-frame',90,'--width',1600,'--height',1200,'--samples',96])
    canvas=Image.new('RGB',(1920,1500),(18,25,32));draw=ImageDraw.Draw(canvas)
    for i,name in enumerate(['coarse','fine']):
        folder=EX/'output'/('scene_clear_shower_'+name)
        for j,sub in enumerate(['preview','preview_hidden_body_clay']):
            im=Image.open(folder/sub/'frame_000090.png').convert('RGB').resize((960,720))
            canvas.paste(im,(i*960,j*750+30))
            draw.text((i*960+15,j*750+8),f'{name}: h={32 if i==0 else 20} mm | same 42 cm emitter, cut-cell boundary | t=3s',fill='white')
    canvas.save(DIAG/'resolution_comparison.jpg',quality=95)
    records['body_support_distance_ratio']=records['fine']['body_nearest_particle_distance_p95_m']/records['coarse']['body_nearest_particle_distance_p95_m']
    records['note']='Same world-space body band (20 mm); geometric support distance is not a viscosity measure. Coupled simulation/obstacle/meshing refinement, identical dimensionless kernel and material settings.'
    (DIAG/'resolution_comparison.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
