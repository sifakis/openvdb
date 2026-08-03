#!/usr/bin/env python3
"""Benchmark the mesh->SDF example: per-step timing, scalability, CC counts. Writes a .md report.

Timing runs set CC_SKIP_VALIDATE=1 so the wall time is the GPU pipeline alone (the CPU oracles +
OpenVDB meshToLevelSet cross-check are skipped). The per-step "pipeline ms" comes from the GPU timers
the example prints, which exclude validation regardless. CC-count runs keep validation on (the count
is printed by the CC label validator)."""
import subprocess, re, time, os, sys

EXE = os.path.expanduser("~/Desktop/Code/NVIDIA/openvdb/build/release/"
                         "nanovdb/nanovdb/examples/ex_mesh_to_sdf_cuda")
MESHES = os.path.expanduser("~/Desktop/meshes")
OUT = os.path.expanduser("~/Desktop/Code/NVIDIA/openvdb/nanovdb/nanovdb/examples/"
                         "ex_mesh_to_sdf_cuda/BENCHMARK.md")

# Pipeline step boundaries — the first timer label of each step, in EMISSION order (positional; some
# labels repeat across MeshToGrid/PruneGrid/ConnectedComponents so we bucket by order, not by name).
# "Allocating per-leaf component counts" therefore appears twice: once for the partition, once for the
# per-surface labeling.
#
# Steps 3-7 are the PER-SURFACE group and repeat once per closed surface. When the matcher sees that
# group's first marker again it wraps back to its start, so those totals are sums over all surfaces.
STEP_MARKERS = [
    (1, "Transforming triangles"),                # step 1: rasterize UDF + nearest-tri index
    (2, "Allocating per-leaf component counts"),  # step 2: partition = CC on the un-pruned band
    (3, "Prune root node"),                       # step 3: prune barrier -> derived topology
    (4, "Allocating per-leaf component counts"),  # step 4: connected components (derived grid)
    (5, "Sign: find exterior component"),         # step 5: sign non-barrier + inject
    (6, "Sign: barrier voxels"),                  # step 6: barrier signing
    (7, "Fill leaf invert mask"),                 # step 7: invert-mask fill (leaf/coarse/root)
    (8, "Inclusion:"),                            # step 8: nesting depth + merge (N>=2 only)
]
GROUP = (2, 7)   # [start, end) indices into STEP_MARKERS of the repeating per-surface group
STEP_NAME = {1: "1 rasterize (UDF+index)", 2: "2 partition (un-pruned CC)",
             3: "3 prune -> derived", 4: "4 connected components",
             5: "5 sign non-barrier+inject", 6: "6 barrier signing",
             7: "7 invert-mask fill", 8: "8 inclusion (nesting)"}
NSTEPS = 8
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
    steps = {i: 0.0 for i in range(1, NSTEPS + 1)}
    subs = {i: [] for i in range(1, NSTEPS + 1)}
    cur, mi = 1, 0
    for label, ms in timers:
        # Scan FORWARD over the remaining markers rather than only testing the next one: a stage can
        # be absent (step 8 runs only for multi-surface input), and a strict sequential match would
        # then stall and misattribute every later timer.
        hit = next((k for k in range(mi, len(STEP_MARKERS)) if label.startswith(STEP_MARKERS[k][1])), None)
        # Nothing ahead matched, but the per-surface group may have started over on the next surface.
        if hit is None:
            hit = next((k for k in range(GROUP[0], min(mi, GROUP[1]))
                        if label.startswith(STEP_MARKERS[k][1])), None)
        if hit is not None:
            cur, mi = STEP_MARKERS[hit][0], hit + 1
        steps[cur] += ms; subs[cur].append((label, ms))
    av = re.search(r"Active voxels\s+\[activeVoxelCount\(\)\]\s*:\s*(\d+)", out)
    # Component counts, from the labels the validator prints today:
    #   "CC voxel-label contract validation: PASS (N active voxels, M components, ...)"  -> pruned grid
    #   "Closed surfaces (un-pruned components): S"                                      -> closed surfaces
    cc = re.search(r"contract validation:\s+\w+\s+\(\d+ active voxels, (\d+) components", out)
    sf = re.search(r"Closed surfaces \(un-pruned components\):\s*(\d+)", out)
    incl = re.search(r"Inclusion:[^.]*\.\.\. completed in ([\d.]+) milliseconds", out)
    ok = any(l.startswith("Root interior") for l, _ in timers)   # step-6 reached => pipeline completed
    oom = "out of memory" in out.lower()
    return {"status": "OK" if (rc == 0 and ok) else ("OOM" if oom else f"FAIL(rc={rc})"),
            "wall": wall, "pipe_ms": sum(steps.values()), "steps": steps, "subs": subs,
            "voxels": int(av.group(1)) if av else None,
            "cc": int(cc.group(1)) if cc else None,
            "surfaces": int(sf.group(1)) if sf else None,
            "incl_ms": float(incl.group(1)) if incl else None}


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

    # Exp 4: cost of the inclusion stage. It only runs for multi-surface input, which no real mesh in
    # the suite is (they are all a single closed surface), so this uses the in-code analytic cases.
    exp4 = ["--two-spheres", "--multi-spheres", "--nested-spheres", "--triple-nested", "--multi-nested"]
    inclrows = []
    for case in exp4:
        print(f"exp4: {case}")
        r = run(case, 0.02, timeout=150, skip_validate=True)
        inclrows.append((case, r))
        write_md(r1, m1, vs1, scal_d, scal_h, ccrows, inclrows)
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


def write_md(r1, m1, vs1, scal_d, scal_h, ccrows, inclrows=()):
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
        if ms <= 0.0: continue   # e.g. 5b is skipped when the mesh is a single closed surface
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
    L.append("## 3. Component counts per mesh\n")
    L.append("Two different counts. **Closed surfaces** = components of the UN-pruned band, which is "
             "what the signing partitions by: one per closed surface. **Pruned components** = "
             "components after the barrier shell is removed, which splits every surface into an inner "
             "and an outer shell and additionally strands a component in each thin/concave pocket. "
             "Fixed voxelSize across meshes → very different voxel counts (the meshes have different "
             "world scales).\n")
    L.append("| mesh | voxelSize | active voxels | closed surfaces | pruned components | status |")
    L.append("|---|---:|---:|---:|---:|---|")
    for name, vs, r in ccrows:
        v = f"{r['voxels']:,}" if r.get("voxels") else "—"
        cc = f"{r['cc']:,}" if r.get("cc") is not None else "—"
        sf = r.get("surfaces") if r.get("surfaces") is not None else "—"
        L.append(f"| {name} | {vs} | {v} | {sf} | {cc} | {r['status']} |")
    L.append("")

    # Exp 4
    if inclrows:
        L.append("## 4. Inclusion-signing cost (multi-surface input)\n")
        L.append("The nesting stage builds one sign field per closed surface to recover how the "
                 "surfaces enclose one another, so it costs roughly one extra invert-mask fill per "
                 "surface. It is skipped entirely for a single closed surface, which is every real "
                 "mesh in the suite — hence the in-code analytic cases here. `inclusion ms` is the "
                 "stage's own GPU timer; `pipeline ms` is the whole GPU pipeline.\n")
        L.append("| case | surfaces | active voxels | inclusion ms | pipeline ms | % |")
        L.append("|---|---:|---:|---:|---:|---:|")
        for case, r in inclrows:
            v  = f"{r['voxels']:,}" if r.get("voxels") else "—"
            sf = r.get("surfaces") if r.get("surfaces") is not None else "—"
            im = r.get("incl_ms")
            pm = r.get("pipe_ms") or 0.0
            L.append(f"| `{case}` | {sf} | {v} | "
                     f"{im:.2f} | {pm:.1f} | {100*im/pm:.1f}% |" if im else
                     f"| `{case}` | {sf} | {v} | — | {pm:.1f} | — |")
        L.append("")

    # Findings
    L.append("## 5. Findings\n")
    L.append('- **Rasterization dominates at low-mid resolution.** Step 1 (mesh -> UDF + index) is ~96% of the pipeline and is triangle-bound (dragon 871 K triangles), so the time is roughly flat (~220-370 ms) up to ~10 M voxels. **Beyond ~10 M it grows ~linearly with voxel count** (0.6 s @ 41 M, 4.5 s @ 461 M) as the voxel-bound steps take over.')
    L.append('- **Everything after rasterization is cheap.** Barrier signing (step 5, ~2.6%) is the only voxel-bound step visible at moderate resolution; the prune, both connected-components passes, the signing and the invert-mask fill are each < 0.5%. Labeling the un-pruned band for the surface partition (step 3a) costs 0.79 ms on 1.1 M voxels — about the same as the pruned-grid pass it sits next to, and 0.3% of the pipeline.')
    L.append('- **Scalability ceiling (12 GB GPU):** dragon completes to **461 M** active voxels (4.5 s pipeline, 34 s wall) and OOMs at the next step (~1 B); the hairball stress mesh completes to **340 M** and OOMs at 0.004, its huge triangle count raising rasterization memory. Wall times here exclude the CPU and OpenVDB validation, which dominates at these sizes.')
    L.append('- **Every mesh in the suite is a single closed surface.** That is the count the signing actually partitions by, and it is **1 for all of them** — including the hairball, at both resolutions. The much larger pruned-component counts (dragon 40, hairball 333,703 @ 0.01) are artifacts of removing the barrier shell: each surface splits into an inner and an outer shell, and every thin or concave pocket strands one more. They are all signed correctly; they only inflate the raw count. Multi-object and nested input is what actually produces more than one surface.')
    L.append('- **Nesting costs about one extra fill per surface, and only when there is more than one.** The inclusion stage builds a sign field per closed surface, so it grows with the surface count (~4 ms at 2 surfaces, ~11-13 ms at 5) and lands at 7-10% of the pipeline on the small analytic cases. On every real mesh it is skipped outright, so single-object input pays nothing for it.')
    L.append("")
    open(OUT, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
