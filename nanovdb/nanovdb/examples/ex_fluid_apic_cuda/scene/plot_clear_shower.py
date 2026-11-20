# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Plot recorded numerical checks; never infer validation from the images."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    cache = args.cache
    if not args.allow_partial:
        assert json.loads((cache / "validation.json").read_text())["pass"]
        assert json.loads((cache / "pool_validation.json").read_text())["pass"]
    settings = json.loads((cache / "settings.json").read_text())
    def load(name):
        with (cache / name).open() as source:
            return list(csv.DictReader(source))
    data, mesh, frames = load("metrics.csv"), load("surface/meshing.csv"), load("frames.csv")
    last = float(mesh[-1]["frame"]) / settings["fps"]
    data = [row for row in data if float(row["time"]) <= last + 1e-10]
    col = lambda key: np.array([float(row[key]) for row in data])
    mcol = lambda key: np.array([float(row[key]) for row in mesh])
    t, mt = col("time"), mcol("frame") / settings["fps"]
    initial = int(frames[0]["particles"]) * settings["particle_volume"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), constrained_layout=True)
    ax = axes[0, 0]
    ax.semilogy(t, np.maximum(col("div_before"), 1e-16), label="before projection", alpha=.6)
    ax.semilogy(t, np.maximum(col("flux_div_after"), 1e-16), label="after, moving-wall flux")
    ax.axhline(1e-6, color="red", ls="--", label="acceptance 1e-6")
    ax.set_title("Full-cell flux divergence, RMS [1/s]")
    ax = axes[0, 1]
    ax.semilogy(t, np.maximum(col("residual_absolute"), 1e-16), label="true residual norm")
    ax.semilogy(t, np.maximum(1e-12, 2e-8 * col("rhs_norm")), label="acceptance bound")
    ax.set_title("Pressure system residual")
    ax = axes[0, 2]
    ax.plot(t, col("mass_volume"), label="particle material volume")
    ax.plot(mt, mcol("mesh_volume"), label="reconstructed mesh volume")
    ax.plot(t, initial + settings["flow_m3_s"] * t, ls="--", label="initial + integrated inflow")
    ax.set_title("Material and reconstructed volumes [m3]")
    ax = axes[1, 0]
    ax.plot(mt, mcol("volume_ratio"), label="mesh / material")
    ax.plot(mt, mcol("kernel_integral") / mcol("material_volume"), label="kernel integral / material")
    ax.axhspan(.8, 1.2, color="green", alpha=.08, label="mesh acceptance range")
    ax.set_ylim(.75, 1.25)
    ax.set_title("Reconstruction bias (not a mass ledger)")
    ax = axes[1, 1]
    ax.plot(t, col("foam"), label="foam")
    ax.plot(t, col("spray"), label="spray")
    ax.set_title("Secondary particles, excluded from pressure")
    ax = axes[1, 2]
    error = col("mass_volume") - initial - settings["flow_m3_s"] * t
    ax.plot(t, error * 1e6, label="material minus continuous inflow [mL]")
    ax.plot(t, col("ledger_error") * settings["particle_volume"] * 1e6, label="integer particle ledger [mL]")
    quantum = settings["particle_volume"] * 1e6
    ax.axhline(-quantum, color="red", ls="--", label="one-particle emission quantum")
    ax.set_title("Closed-basin mass accounting")
    for ax in axes.flat:
        ax.set_xlabel("simulation seconds")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    title = "PARTIAL CACHE" if args.allow_partial else "Validated 12 second refinement run"
    fig.suptitle(f"{title}: recorded through {last:g} s", fontsize=15)
    name = "validation_overview_partial.png" if args.allow_partial else "validation_overview.png"
    fig.savefig(cache / name, dpi=150)
    plt.close(fig)
    print(cache / name)


if __name__ == "__main__":
    main()
