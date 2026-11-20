#!/usr/bin/env python3
"""make_viewer.py <prefix> <mesh name> <res> <wide band> <gpu> <commit> <out.html>

Embed <prefix>.json and <prefix>.bin (written by offset_demo) into viewer_template.html."""
import base64
import json
import os
import sys

prefix, mesh, res, wide_band, gpu, commit, out = sys.argv[1:8]
meta = json.load(open(prefix + ".json"))
meta.update(mesh=mesh, res=int(res), wideBand=int(wide_band), gpu=gpu, commit=commit)
data = base64.b64encode(open(prefix + ".bin", "rb").read()).decode()
template = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "viewer_template.html"), encoding="utf-8").read()
html = template.replace("/*META*/", json.dumps(meta)).replace("/*B64*/", data)
open(out, "w", encoding="utf-8").write(html)
print(f"{out}: {len(html) / 1e6:.1f} MB")
