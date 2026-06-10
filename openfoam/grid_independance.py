import os
import subprocess
import shutil
import csv
import time
import re
import uuid

from supersonic_run import (
    isa_atmosphere, prepare, solve, cleanup,
    ALTITUDE_M, MACH, NP, END_TIME, WRITE_INTERVAL,
    CASE_TEMPLATE, GEOMETRY_STL
)

RESULTS_CSV = "grid_independence.csv"

REFINEMENT_LEVELS = [
    (1, (25,  17,  25)),
    (2, (35,  23,  35)),
    (3, (45,  30,  45)),
    (4, (60,  40,  60)),   # baseline
    (5, (75,  50,  75)),
    (6, (90,  60,  90)),
    (7, (110, 73, 110)),
]

def patch_blockmesh(job_directory, nx, ny, nz):
    bmd_path = os.path.join(job_directory, "system", "blockMeshDict")
    with open(bmd_path, "r") as f:
        content = f.read()
    patched = re.sub(
        r'hex\s*\([^)]+\)\s*\(\d+\s+\d+\s+\d+\)',
        f'hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz})',
        content
    )
    with open(bmd_path, "w") as f:
        f.write(patched)

def extract_cd(job_directory):
    # Look in postProcessing/forceCoeffs
    fc_base = os.path.join(job_directory, "postProcessing", "forceCoeffs")
    if not os.path.isdir(fc_base):
        return None
    # Find latest time subdir
    time_dirs = sorted(
        [d for d in os.listdir(fc_base) if _is_float(d)],
        key=float
    )
    if not time_dirs:
        return None
    coeff_file = os.path.join(fc_base, time_dirs[-1], "coefficient.dat")
    if not os.path.isfile(coeff_file):
        # try alternate name
        coeff_file = os.path.join(fc_base, time_dirs[-1], "forceCoeffs.dat")
    if not os.path.isfile(coeff_file):
        return None
    cd_vals = []
    with open(coeff_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split()
            if len(cols) >= 3:
                try:
                    cd_vals.append(float(cols[2]))  # Cd is column index 2
                except ValueError:
                    pass
    return cd_vals[-1] if cd_vals else None

def mesh(job_directory):
    os.makedirs(f"{job_directory}/constant/triSurface", exist_ok=True)
    shutil.copy(GEOMETRY_STL, f"{job_directory}/constant/triSurface/uav.stl")

    COMMANDS = [
        f"surfaceFeatureExtract -case {job_directory}",
        f"blockMesh -case {job_directory}",
        f"decomposePar -case {job_directory}",
        f"mpirun -np {NP} snappyHexMesh -parallel -overwrite -case {job_directory}",
        f"reconstructParMesh -constant -case {job_directory}",
        f"rm -rf {job_directory}/processor*",
    ]
    for cmd in COMMANDS:
        result = subprocess.run(cmd, shell=True, executable="/bin/bash")
        if result.returncode != 0:
            print(f"  Meshing failed: {cmd}")
            return False
    return os.path.isdir(f"{job_directory}/constant/polyMesh")

def _is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def count_cells(job_directory):
    mesh_info = os.path.join(job_directory, "constant", "polyMesh", "owner")
    if not os.path.isfile(mesh_info):
        return None
    result = subprocess.run(
        f"checkMesh -case {job_directory} | grep 'cells:' | tail -1",
        shell=True, capture_output=True, text=True
    )
    m = re.search(r"cells:\s+(\d+)", result.stdout)
    return int(m.group(1)) if m else None

def run_level(level, nx, ny, nz, atm, u_inf):
    job_id = f"grid_L{level}_{uuid.uuid4().hex[:6]}"
    job_dir = f"./{job_id}"
    print(f"\n{'='*50}")
    print(f"Level {level}  mesh=({nx},{ny},{nz})  job={job_id}")
    print(f"{'='*50}")

    t_start = time.time()

    if not prepare(job_dir, atm, u_inf):
        print("  prepare() failed")
        return None

    patch_blockmesh(job_dir, nx, ny, nz)

    if not mesh(job_dir):
        print("  mesh() failed")
        # cleanup(job_dir)
        return None

    cells = count_cells(job_dir)

    if not solve(job_dir):
        print("  solve() failed")
        # cleanup(job_dir)
        return None

    cd = extract_cd(job_dir)
    elapsed = time.time() - t_start

    # cleanup(job_dir)

    return {
        "level":      level,
        "nx":         nx,
        "ny":         ny,
        "nz":         nz,
        "total_cells": cells if cells else "",
        "Cd":         cd if cd is not None else "",
        "runtime_s":  round(elapsed, 1),
    }

def main():
    atm   = isa_atmosphere(ALTITUDE_M)
    u_inf = MACH * atm["a"]
    print(f"ISA {ALTITUDE_M/1000:.0f} km | Mach {MACH} | U={u_inf:.1f} m/s")

    fieldnames = ["level", "nx", "ny", "nz", "total_cells", "Cd", "runtime_s"]
    with open(RESULTS_CSV, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()
    print(f"[*] Results -> {RESULTS_CSV}")

    for level, (nx, ny, nz) in REFINEMENT_LEVELS:
        row = run_level(level, nx, ny, nz, atm, u_inf)
        if row is None:
            row = {"level": level, "nx": nx, "ny": ny, "nz": nz,
                   "total_cells": "", "Cd": "FAILED", "runtime_s": ""}
        with open(RESULTS_CSV, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)
        print(f"  -> L{level}: Cd={row['Cd']}  t={row['runtime_s']}s  cells={row['total_cells']}")

    print(f"\nDone. Results in {RESULTS_CSV}")

if __name__ == "__main__":
    main()