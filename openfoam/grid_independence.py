import argparse
import csv
import os
import re
import shutil
import subprocess
import time
import uuid

import metrics
from supersonic_run import (
    isa_atmosphere, prepare, solve, cleanup,
    ALTITUDE_M, MACH, NP, CASE_TEMPLATE, GEOMETRY_STL, SYMMETRY,
)

RESULTS_CSV = "grid_independence.csv"

# Half domain: x 30 m, y 10 m, z 20 m.  Cubic cells require
#   ny = nx * (10/30) = nx/3        nz = nx * (20/30) = 2*nx/3
# Level 4 reproduces the committed template exactly: (60, 20, 40).
REFINEMENT_LEVELS = [
    (1, (25,  8,  17)),
    (2, (35, 12,  23)),
    (3, (45, 15,  30)),
    (4, (60, 20,  40)),   # baseline - matches case_template_supersonic
    (5, (75, 25,  50)),
    (6, (90, 30,  60)),
    (7, (110, 37, 73)),
]

FIELDNAMES = [
    "level", "nx", "ny", "nz", "dx", "total_cells",
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
    """
    Same pipeline as supersonic_run.mesh(), but the snappyHexMesh output is
    teed to a log so the maxGlobalCells cap can be detected afterwards.
    """
    tri = f"{job_directory}/constant/triSurface"
    os.makedirs(tri, exist_ok=True)
    shutil.copy(GEOMETRY_STL, f"{tri}/uav.stl")

    log = f"{job_directory}/log.snappyHexMesh"
    commands = [
        f"surfaceFeatureExtract -case {job_directory}",
        f"blockMesh -case {job_directory}",
        f"decomposePar -case {job_directory}",
        f"mpirun -np {NP} snappyHexMesh -parallel -overwrite "
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


def cap_was_hit(job_directory):
    """
    Detect silent refinement truncation. snappyHexMesh reports reaching the
    global cell limit in its log and then carries on to produce a valid but
    under-refined mesh - there is no non-zero exit code to catch.
    """
    log = f"{job_directory}/log.snappyHexMesh"
    if not os.path.isfile(log):
        return None
    with open(log, errors="replace") as f:
        text = f.read()
    patterns = [
        r"Stopped refin\w+",
        r"maxGlobalCells",
        r"Reached (?:global )?max(?:imum)? (?:number of )?cells",
        r"exceeded max(?:imum)? number of cells",
    ]
    hits = [p for p in patterns if re.search(p, text, re.I)]
    # "maxGlobalCells" alone appears in the echoed dictionary, so require a
    # phrasing that indicates the limit actually bound.
    return bool([h for h in hits if h != "maxGlobalCells"])


def mesh_quality(job_directory):
    """cells / max non-orthogonality / max skewness from checkMesh."""
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
    dx = 30.0 / nx

    print(f"\n{'=' * 60}")
    print(f"Level {level}   background ({nx},{ny},{nz})   dx = {dx:.4f} m")
    print(f"job {job_id}")
    print(f"{'=' * 60}")

    row = {k: "" for k in FIELDNAMES}
    row.update(level=level, nx=nx, ny=ny, nz=nz,
               dx=round(dx, 5), job=job_id, status="started")

    t0 = time.time()
    if not prepare(job_dir, atm, u_inf):
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
    hit = cap_was_hit(job_dir)
    row["cap_hit"] = hit
    if hit:
        print("  *** maxGlobalCells LIMIT REACHED - refinement was truncated. "
              "This level is NOT a valid point on the convergence curve. ***")

    t_solve = time.time()
    if not solve(job_dir):
        row["status"] = "solve_failed"
        row["runtime_s"] = round(time.time() - t0, 1)
        return row
    row["solve_time_s"] = round(time.time() - t_solve, 1)

    m = metrics.write_metrics(job_dir, symmetry_factor=SYMMETRY)
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
                    help="levels to run (default: all). One per SLURM task.")
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

    # Append mode with a header only when new, so array tasks can share a file.
    exists = os.path.isfile(args.csv)
    with open(args.csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not exists:
            w.writeheader()
    print(f"[*] Results -> {args.csv}")

    bad = 0
    for level, (nx, ny, nz) in levels:
        row = run_level(level, nx, ny, nz, atm, u_inf, keep=args.keep)
        with open(args.csv, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)
        print(f"  -> L{level}: drag={row['drag_N']} N  Cd={row['Cd']}  "
              f"cells={row['total_cells']}  converged={row['converged']}  "
              f"cap_hit={row['cap_hit']}  t={row['runtime_s']}s")
        if row["status"] != "ok" or row["cap_hit"] or row["converged"] is not True:
            bad += 1

    print(f"\nDone. {len(levels) - bad}/{len(levels)} levels usable. "
          f"Results in {args.csv}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())