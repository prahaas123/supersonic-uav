import os
import subprocess
import shutil
import csv
import time
import uuid

from supersonic_run import (
    isa_atmosphere, prepare, cleanup,
    ALTITUDE_M, MACH, GEOMETRY_STL
)

RESULTS_CSV = "core_scaling.csv"
CORE_COUNTS = [10, 20, 30, 40, 50, 60, 70]

def mesh(job_directory, np_):
    os.makedirs(f"{job_directory}/constant/triSurface", exist_ok=True)
    shutil.copy(GEOMETRY_STL, f"{job_directory}/constant/triSurface/uav.stl")

    COMMANDS = [
        f"surfaceFeatureExtract -case {job_directory}",
        f"blockMesh -case {job_directory}",
        f"decomposePar -case {job_directory}",
        f"mpirun -np {np_} snappyHexMesh -parallel -overwrite -case {job_directory}",
        f"reconstructParMesh -constant -case {job_directory}",
        f"rm -rf {job_directory}/processor*",
    ]
    for cmd in COMMANDS:
        result = subprocess.run(cmd, shell=True, executable="/bin/bash")
        if result.returncode != 0:
            print(f"  Meshing failed: {cmd}")
            return False
    return os.path.isdir(f"{job_directory}/constant/polyMesh")

def set_decompose_par(job_directory, np_):
    path = os.path.join(job_directory, "system", "decomposeParDict")
    from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
    dpd = ParsedParameterFile(path)
    dpd["numberOfSubdomains"] = np_
    dpd.writeFile()

def solve_with_cores(job_directory, np_):
    from PyFoam.Execution.BasicRunner import BasicRunner

    COMMANDS = [
        f"decomposePar -case {job_directory}",
        f"mpirun -np {np_} rhoCentralFoam -parallel -case {job_directory}",
        f"reconstructPar -latestTime -case {job_directory}",
    ]
    for command in COMMANDS:
        runner = BasicRunner(argv=command.split())
        runner.start()
        if not runner.runOK():
            raise Exception(f"{command} failed")

    subprocess.run(f"rm -rf {job_directory}/processor*", shell=True)

    time_dirs = [d for d in os.listdir(job_directory)
                 if os.path.isdir(f"{job_directory}/{d}") and _is_float(d) and float(d) > 0]
    return len(time_dirs) > 0

def _is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def run_job(np_, atm, u_inf):
    job_id = f"cores_{np_:03d}_{uuid.uuid4().hex[:6]}"
    job_dir = f"./{job_id}"
    print(f"\n{'='*50}")
    print(f"NP = {np_}  job={job_id}")
    print(f"{'='*50}")

    if not prepare(job_dir, atm, u_inf):
        print("  prepare() failed")
        return None

    set_decompose_par(job_dir, np_)

    print("  Meshing...")
    t_mesh_start = time.time()
    if not mesh(job_dir, np_):
        print("  mesh() failed")
        cleanup(job_dir)
        return None
    mesh_time = time.time() - t_mesh_start

    print("  Solving...")
    t_solve_start = time.time()
    if not solve_with_cores(job_dir, np_):
        print("  solve failed")
        cleanup(job_dir)
        return None
    solve_time = time.time() - t_solve_start

    total_time = mesh_time + solve_time

    cleanup(job_dir)

    return {
        "num_cores":  np_,
        "mesh_time_s":  round(mesh_time, 2),
        "solve_time_s": round(solve_time, 2),
        "total_time_s": round(total_time, 2),
    }

def main():
    atm   = isa_atmosphere(ALTITUDE_M)
    u_inf = MACH * atm["a"]
    print(f"ISA {ALTITUDE_M/1000:.0f} km | Mach {MACH} | U={u_inf:.1f} m/s")

    fieldnames = ["num_cores", "mesh_time_s", "solve_time_s", "total_time_s"]
    with open(RESULTS_CSV, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()
    print(f"[*] Results -> {RESULTS_CSV}")

    for np_ in CORE_COUNTS:
        row = run_job(np_, atm, u_inf)
        if row is None:
            row = {"num_cores": np_, "mesh_time_s": "", "solve_time_s": "", "total_time_s": "FAILED"}
        with open(RESULTS_CSV, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)
        print(f"  -> NP={np_}: total={row['total_time_s']}s "
              f"(mesh={row['mesh_time_s']}s, solve={row['solve_time_s']}s)")

    print(f"\nDone. Results in {RESULTS_CSV}")

if __name__ == "__main__":
    main()