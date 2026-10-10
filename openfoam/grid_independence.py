# python3 grid_independence.py --levels 3   run one background refinement level (one SLURM array task, see grid_independence.sh)
# Only the blockMesh background resolution changes between levels. snappy surface/box levels and the
# wall layers come from case_template_supersonic unchanged, so every level refines the whole mesh together.

import argparse
import csv
import os
import re
import subprocess
import time
import uuid

import metrics
from supersonic_run import (
    isa_atmosphere, prepare, solve, cleanup,
    ALTITUDE_M, MACH, NP, SYMMETRY, UAV_PARAMS
)

RESULTS_CSV = "grid_independence.csv"

REFINEMENT_LEVELS = [
    (1, (21,  9, 18)),
    (2, (28, 12, 24)),
    (3, (35, 15, 30)),   # template
    (4, (42, 18, 36)),
    (5, (49, 21, 42)),
]
DOMAIN_Y = 10.0          # blockMeshDict y extent, background cells are cubic

FIELDNAMES = [
    "level", "nx", "ny", "nz", "dx", "total_cells",
    "layers_avg", "layer_thickness_pct",
    "drag_N", "lift_N", "Cd", "Cl",
    "converged", "drag_drift_pct", "drag_std_N",
    "yplus_avg", "yplus_max", "max_non_ortho", "max_skewness",
    "cap_hit", "status", "mesh_time_s", "solve_time_s", "runtime_s", "job",
]

# ---------------------------------------------------------------------------
def patch_blockmesh(job_directory, nx, ny, nz):
    path = os.path.join(job_directory, "system", "blockMeshDict")
    with open(path) as f:
        content = f.read()
    patched, n = re.subn(
        r"hex\s*\([^)]+\)\s*\(\s*\d+\s+\d+\s+\d+\s*\)",
        f"hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz})",
        content,
    )
    if n != 1:
        raise RuntimeError(f"blockMeshDict hex substitution matched {n} times, expected 1")
    with open(path, "w") as f:
        f.write(patched)

def mesh_with_log(job_directory):
    log = f"{job_directory}/log.snappyHexMesh"
    commands = [
        f"surfaceFeatureExtract -case {job_directory}",
        f"blockMesh -case {job_directory}",
        f"decomposePar -case {job_directory}",
        f"set -o pipefail; mpirun -np {NP} snappyHexMesh -parallel -overwrite "
        f"-case {job_directory} 2>&1 | tee {log}",
        f"reconstructParMesh -constant -case {job_directory}",
        f"rm -rf {job_directory}/processor*",
    ]
    for cmd in commands:
        print(f"  -> {cmd}")
        if subprocess.run(cmd, shell=True, executable="/bin/bash").returncode != 0:
            print(f"  Meshing failed: {cmd}")
            return False
    return os.path.isdir(f"{job_directory}/constant/polyMesh")

def read_snappy_log(job_directory):
    out = {"cap_hit": None, "layers_avg": None, "layer_thickness_pct": None}
    log = f"{job_directory}/log.snappyHexMesh"
    if not os.path.isfile(log):
        return out
    with open(log, errors="replace") as f:
        text = f.read()

    patterns = [
        r"Stopped refin\w+",
        r"Reached (?:global )?max(?:imum)? (?:number of )?cells",
        r"exceeded max(?:imum)? number of cells",
    ]
    out["cap_hit"] = any(re.search(p, text, re.I) for p in patterns)

    # patch  faces  layers  overall-thickness[m]  [%]
    rows = re.findall(r"^\s*uav\w*\s+\d+\s+([\d.]+)\s+[\d.eE+-]+\s+([\d.]+)\s*$", text, re.M)
    if rows:
        out["layers_avg"] = float(rows[-1][0])
        out["layer_thickness_pct"] = float(rows[-1][1])
    return out

def mesh_quality(job_directory):
    out = {"total_cells": None, "max_non_ortho": None, "max_skewness": None}
    r = subprocess.run(f"checkMesh -case {job_directory}",
                       shell=True, capture_output=True, text=True,
                       executable="/bin/bash")
    text = r.stdout

    m = re.search(r"^\s*cells:\s+(\d+)", text, re.M)
    if m:
        out["total_cells"] = int(m.group(1))
    m = re.search(r"Max (?:non-orthogonality|nonOrthogonality)[^0-9-]*([\d.]+)", text)
    if m:
        out["max_non_ortho"] = float(m.group(1))
    m = re.search(r"Max skewness\s*=\s*([\d.]+)", text)
    if m:
        out["max_skewness"] = float(m.group(1))
    return out

# ---------------------------------------------------------------------------
def run_level(level, nx, ny, nz, atm, u_inf, keep=False):
    job_id = f"grid_L{level}_{uuid.uuid4().hex[:6]}"
    job_dir = f"./{job_id}"
    dx = DOMAIN_Y / ny

    print(f"\n{'=' * 60}")
    print(f"Level {level}   background ({nx},{ny},{nz})   dx = {dx:.4f} m")
    print(f"job {job_id}")
    print(f"{'=' * 60}")

    row = {k: "" for k in FIELDNAMES}
    row.update(level=level, nx=nx, ny=ny, nz=nz, dx=round(dx, 5), job=job_id,
               status="started")

    t0 = time.time()
    refs = prepare(job_dir, atm, u_inf, UAV_PARAMS)
    if refs is None:
        row["status"] = "prepare_failed"
        return row

    patch_blockmesh(job_dir, nx, ny, nz)

    t_mesh = time.time()
    if not mesh_with_log(job_dir):
        row["status"] = "mesh_failed"
        row["runtime_s"] = round(time.time() - t0, 1)
        return row
    row["mesh_time_s"] = round(time.time() - t_mesh, 1)

    row.update({k: v for k, v in mesh_quality(job_dir).items() if v is not None})
    row.update({k: v for k, v in read_snappy_log(job_dir).items() if v is not None})
    if row["cap_hit"]:
        print("  *** maxGlobalCells LIMIT REACHED - refinement was truncated. "
              "This level is NOT a valid point on the convergence curve. ***")
    print(f"  layers: avg {row['layers_avg']}, "
          f"{row['layer_thickness_pct']} % of target thickness")

    t_solve = time.time()
    try:
        solved = solve(job_dir)
    except Exception as e:
        print(f"  Exception during solving: {e}")
        solved = False
    if not solved:
        row["status"] = "solve_failed"
        row["runtime_s"] = round(time.time() - t0, 1)
        return row
    row["solve_time_s"] = round(time.time() - t_solve, 1)

    m = metrics.write_metrics(job_dir, symmetry_factor=SYMMETRY, extra=refs)
    print(metrics.format_summary(m))

    for key in ("drag_N", "lift_N", "Cd", "Cl", "converged",
                "drag_drift_pct", "drag_std_N", "yplus_avg", "yplus_max",
                "status"):
        if m.get(key) is not None:
            row[key] = m[key]

    row["runtime_s"] = round(time.time() - t0, 1)

    if not keep:
        cleanup(job_dir)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, nargs="+", default=None,
                    help="levels to run (default: all). One per SLURM array task.")
    ap.add_argument("--csv", default=RESULTS_CSV)
    ap.add_argument("--keep", action="store_true",
                    help="keep case directories instead of cleaning up")
    args = ap.parse_args()

    atm = isa_atmosphere(ALTITUDE_M)
    u_inf = MACH * atm["a"]
    print(f"ISA {ALTITUDE_M / 1000:.0f} km | Mach {MACH} | U = {u_inf:.1f} m/s | "
          f"symmetry factor x{SYMMETRY:g}")

    levels = [(lv, n) for lv, n in REFINEMENT_LEVELS
              if args.levels is None or lv in args.levels]
    if not levels:
        print("No matching levels.")
        return 1

    # Array tasks share one file, the header is written only by whichever creates it
    try:
        with open(args.csv, "x", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()
    except FileExistsError:
        pass
    print(f"[*] Results -> {args.csv}")

    bad = 0
    for level, (nx, ny, nz) in levels:
        row = run_level(level, nx, ny, nz, atm, u_inf, keep=args.keep)
        with open(args.csv, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)
        print(f"  -> L{level}: drag={row['drag_N']} N  Cd={row['Cd']}  "
              f"cells={row['total_cells']}  y+avg={row['yplus_avg']}  "
              f"layers={row['layers_avg']} ({row['layer_thickness_pct']} %)  "
              f"converged={row['converged']}  cap_hit={row['cap_hit']}  "
              f"t={row['runtime_s']}s")
        if row["status"] != "ok" or row["cap_hit"] or row["converged"] is not True:
            bad += 1

    print(f"\nDone. {len(levels) - bad}/{len(levels)} levels usable. "
          f"Results in {args.csv}")
    return 0 if bad == 0 else 2

if __name__ == "__main__":
    raise SystemExit(main())
