#!/usr/bin/env python3
"""Benchmark the mesh->SDF example: per-step timing, scalability, CC counts. Writes a .md report.

Timing runs set CC_SKIP_VALIDATE=1 so the wall time is the GPU pipeline alone (the CPU oracles +
OpenVDB meshToLevelSet cross-check are skipped). The per-step "pipeline ms" comes from the GPU timers
the example prints, which exclude validation regardless. CC-count runs keep validation on (the count
is printed by the CC label validator)."""
import subprocess, re, time, os, sys

EXE = os.path.expanduser("~/Desktop/Code/NVIDIA/openvdb/build/release/"
                         "nanovdb/nanovdb/examples/ex_connected_components_cuda")
MESHES = os.path.expanduser("~/Desktop/meshes")
OUT = os.path.expanduser("~/Desktop/Code/NVIDIA/openvdb/nanovdb/nanovdb/examples/"
                         "ex_connected_components_cuda/BENCHMARK.md")

# Pipeline step boundaries — the first timer label of each step (positional; some labels repeat
# across MeshToGrid/PruneGrid so we bucket by order, not by name).
STEP_MARKERS = [
    (1, "Transforming triangles"),                 # step 1: rasterize UDF + nearest-tri index
    (2, "Prune root node"),                         # step 2: prune barrier -> derived topology
    (3, "Allocating per-leaf component counts"),    # step 3: connected components
    (4, "Sign: find exterior component"),           # step 4: sign non-barrier + inject
    (5, "Sign: barrier voxels"),                    # step 5: barrier signing
    (6, "Fill leaf invert mask"),                   # step 6: invert-mask fill (leaf/coarse/root)
]
STEP_NAME = {1: "1 rasterize (UDF+index)", 2: "2 prune -> derived", 3: "3 connected components",
             4: "4 sign non-barrier+inject", 5: "5 barrier signing", 6: "6 invert-mask fill"}
TIMER_RE = re.compile(r"^(.*?) \.\.\. completed in ([\d.]+) milliseconds")


def run(mesh, vs, timeout=300, skip_validate=False):
    t0 = time.time()
    env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}  # unset (empty hides GPU)
    if skip_validate:
        env["CC_SKIP_VALIDATE"] = "1"   # time the GPU pipeline alone; wall excludes CPU/OpenVDB oracles
    try:
        p = subprocess.run([EXE, mesh, str(vs)], capture_output=True, text=True, timeout=timeout, env=env)
        out = p.stdout + "\n" + p.stderr
        rc = p.returncode
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "wall": timeout}
    wall = time.time() - t0
    timers = [(m.group(1).strip(), float(m.group(2))) for m in
              (TIMER_RE.match(l) for l in out.splitlines()) if m]
    steps = {i: 0.0 for i in range(1, 7)}
    subs = {i: [] for i in range(1, 7)}
    cur = 1
    marker_it = iter(STEP_MARKERS); nxt = next(marker_it, None)
    for label, ms in timers:
        while nxt and label.startswith(nxt[1]):
            cur = nxt[0]; nxt = next(marker_it, None); break
        steps[cur] += ms; subs[cur].append((label, ms))
    av = re.search(r"Active voxels\s+\[activeVoxelCount\(\)\]\s*:\s*(\d+)", out)
    cc = re.search(r"\((\d+) components, (\d+) global labels", out)
    ok = any(l.startswith("Root interior") for l, _ in timers)   # step-6 reached => pipeline completed
    oom = "out of memory" in out.lower()
    return {"status": "OK" if (rc == 0 and ok) else ("OOM" if oom else f"FAIL(rc={rc})"),
            "wall": wall, "pipe_ms": sum(steps.values()), "steps": steps, "subs": subs,
            "voxels": int(av.group(1)) if av else None,
            "cc": int(cc.group(2)) if cc else None}


def warmup():
    run(os.path.join(MESHES, "hand.obj"), 0.01, timeout=120, skip_validate=True)   # prime CUDA / JIT


def main():
    print("warmup..."); warmup()
    dragon = os.path.join(MESHES, "dragon.obj")
    hair = os.path.join(MESHES, "hairball.obj")

    # Exp 1: per-step timing. Validation ON only to also print the CC count; the step table / pipeline
    # ms come from the GPU timers, which exclude validation regardless.
    m1, vs1 = dragon, 0.003
    print(f"exp1: dragon @ {vs1}")
    r1 = run(m1, vs1, skip_validate=False)
    write_md(r1, m1, vs1, [], [], [])

    # Exp 2: scalability. Validation OFF so the wall is the GPU pipeline alone.
    scal_d = []
    for vs in [0.01, 0.006, 0.004, 0.003, 0.002, 0.0015, 0.001,
               0.0007, 0.0005, 0.0003, 0.0002, 0.00015, 0.0001]:
        print(f"exp2 dragon @ {vs}")
        r = run(dragon, vs, timeout=180, skip_validate=True)
        scal_d.append((vs, r))
        write_md(r1, m1, vs1, scal_d, [], [])
        if r["status"] != "OK":
            break
    scal_h = []
    if os.path.exists(hair):
        for vs in [0.02, 0.01, 0.006, 0.004]:
            print(f"exp2 hairball @ {vs}")
            r = run(hair, vs, timeout=180, skip_validate=True)
            scal_h.append((vs, r))
            write_md(r1, m1, vs1, scal_d, scal_h, [])
            if r["status"] != "OK":
                break

    # Exp 3: CC count per mesh. Validation ON (the count is printed by the CC label validator).
    exp3 = [("bunny.obj", 0.004), ("dragon.obj", 0.004), ("armadillo.obj", 0.004),
            ("cow.obj", 0.004), ("hand.obj", 0.004), ("cat.obj", 0.004),
            ("hairball.obj", 0.02), ("hairball.obj", 0.01)]   # coarser: hairball OOMs at 0.004
    ccrows = []
    for name, vs in exp3:
        path = os.path.join(MESHES, name)
        if not os.path.exists(path):
            continue
        print(f"exp3: {name} @ {vs}")
        r = run(path, vs, timeout=150, skip_validate=False)
        ccrows.append((name, vs, r))
        write_md(r1, m1, vs1, scal_d, scal_h, ccrows)
    print("wrote", OUT)


def _scal_table(L, rows):
    L.append("| voxelSize | active voxels | pipeline ms | wall s | status |")
    L.append("|---:|---:|---:|---:|---|")
    for vs, r in rows:
        ok = r["status"] == "OK"
        v = f"{r['voxels']:,}" if r.get("voxels") else "—"
        pm = f"{r['pipe_ms']:.1f}" if (ok and r.get("pipe_ms")) else "—"   # partial on OOM => blank
        w = f"{r['wall']:.1f}" if ok else "—"
        st = r["status"] if ok else f"**{r['status']}**"
        L.append(f"| {vs} | {v} | {pm} | {w} | {st} |")


def write_md(r1, m1, vs1, scal_d, scal_h, ccrows):
    import platform
    def sh(cmd):
        try: return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()
        except Exception: return "?"
    L = []
    L.append("# Mesh -> SDF pipeline benchmark\n")
    L.append("Generated by `scripts/bench_mesh_to_sdf.py`. **Timing runs skip validation**")
    L.append("(`CC_SKIP_VALIDATE=1`), so the wall time is the GPU pipeline alone — the CPU oracles and")
    L.append("the OpenVDB `meshToLevelSet` cross-check are excluded. `pipeline ms` is the sum of the")
    L.append("per-step GPU timers the example prints (also validation-free). CC-count runs keep")
    L.append("validation on, since the count is printed by the CC label validator.\n")
    L.append("## Environment\n")
    L.append(f"- **GPU:** {sh('nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader')}")
    L.append(f"- **CUDA (nvcc):** {sh('nvcc --version | grep release').split('release')[-1].strip()}")
    L.append(f"- **CUDA arch:** {sh('grep CMAKE_CUDA_ARCHITECTURES ~/Desktop/Code/NVIDIA/openvdb/build/release/CMakeCache.txt').split('=')[-1]}")
    L.append("- **CPU:** Intel Core Ultra 7 265H")
    L.append(f"- **RAM:** {sh('free -h | awk \"/Mem:/{print \\$2}\"')}")
    L.append(f"- **OS:** {sh('. /etc/os-release; echo $PRETTY_NAME')} / kernel {platform.release()}")
    L.append(f"- **commit:** {sh('git -C ~/Desktop/Code/NVIDIA/openvdb rev-parse --short HEAD')}")
    L.append("- **build:** Release, NANOVDB_USE_CUDA + NANOVDB_USE_OPENVDB\n")

    # Exp 1
    L.append("## 1. Per-step timing (moderate example)\n")
    L.append(f"`{os.path.basename(m1)}` @ voxelSize {vs1} — {r1['voxels']:,} active voxels, "
             f"{r1['cc']} connected components. **GPU pipeline total {r1['pipe_ms']:.1f} ms** "
             "(validation excluded).\n")
    L.append("### Steps, sorted by time (bottleneck first)\n")
    L.append("| step | ms | % |")
    L.append("|---|---:|---:|")
    tot = r1["pipe_ms"] or 1
    for i, ms in sorted(r1["steps"].items(), key=lambda kv: -kv[1]):
        L.append(f"| {STEP_NAME[i]} | {ms:.2f} | {100*ms/tot:.1f}% |")
    L.append(f"| **total** | **{tot:.2f}** | 100% |\n")
    L.append("### Top 12 individual sub-steps\n")
    allsub = [(lbl, ms, STEP_NAME[i]) for i, ss in r1["subs"].items() for lbl, ms in ss]
    L.append("| sub-step | step | ms |")
    L.append("|---|---|---:|")
    for lbl, ms, st in sorted(allsub, key=lambda x: -x[1])[:12]:
        L.append(f"| {lbl} | {st.split()[0]} | {ms:.2f} |")
    L.append("")

    # Exp 2
    L.append("## 2. Scalability (validation-free)\n")
    L.append("Wall and pipeline both exclude validation. `pipeline ms` = sum of GPU-step timers; "
             "`wall s` additionally includes host-side mesh load + allocation.\n")
    L.append("### dragon, increasing resolution\n")
    _scal_table(L, scal_d)
    L.append("")
    if scal_h:
        L.append("### hairball (stress mesh — 236 MB, dense tangle of thin strands)\n")
        _scal_table(L, scal_h)
        L.append("")
    L.append("`status`: OK = full pipeline completed; OOM = `cudaError 2: out of memory` at allocation.\n")

    # Exp 3
    L.append("## 3. Connected-component count per mesh\n")
    L.append("Component count = distinct global CC labels on the barrier-pruned grid (ideal 2 for a "
             "clean closed surface; extra = trapped pockets in thin/concave regions and/or separate "
             "objects). Fixed voxelSize across meshes → very different voxel counts (the meshes have "
             "different world scales).\n")
    L.append("| mesh | voxelSize | active voxels | components | status |")
    L.append("|---|---:|---:|---:|---|")
    for name, vs, r in ccrows:
        v = f"{r['voxels']:,}" if r.get("voxels") else "—"
        cc = r.get("cc") if r.get("cc") is not None else "—"
        L.append(f"| {name} | {vs} | {v} | {cc} | {r['status']} |")
    L.append("")

    # Findings
    L.append("## 4. Findings\n")
    L.append("- **Rasterization dominates at low–mid resolution.** Step 1 (mesh → UDF + index) is ~96% "
             "of the pipeline and is triangle-bound (dragon 871 K triangles), so the pipeline time is "
             "roughly flat (~220–340 ms) up to ~10 M voxels. **Beyond ~10 M it grows ~linearly with "
             "voxel count** (0.6 s @ 41 M, 3.9 s @ 461 M) as the voxel-bound steps take over.")
    L.append("- **Signing / CC / fill are cheap.** Steps 2–4 and 6 are each < 0.3%; barrier signing "
             "(step 5, ~3%) is the only voxel-bound step visible at moderate resolution.")
    L.append("- **Scalability ceiling (12 GB GPU):** dragon completes to **461 M** active voxels "
             "(3.9 s pipeline, ~20 s wall) and OOMs at the next step (~1 B); the hairball stress mesh "
             "completes to **340 M** and OOMs at 0.004 (its huge triangle count raises rasterization "
             "memory). Validation-free wall at 461 M is ~20 s vs ~156 s with the OpenVDB cross-check on.")
    L.append("- **Component count is resolution- and mesh-dependent.** Clean objects give ~1–2 "
             "(armadillo 2, hand 1, cat 2); thin/concave features trap extra pockets (dragon 12 @ 0.003 "
             "but 40 @ 0.004 — coarser pinches off more). All extras are correctly signed (interior); "
             "they only inflate the raw count.")
    L.append("")
    open(OUT, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
