import os
import subprocess
import shutil
import re
import csv
import uuid
import math
from PyFoam.RunDictionary.SolutionDirectory import SolutionDirectory
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
from PyFoam.Execution.BasicRunner import BasicRunner

GEOMETRY_STL  = "uav.stl"
CASE_TEMPLATE = "case_template_supersonic"
ALTITUDE_M    = 15000.0     # [m]
MACH          = 1.5
CG_X          = 0.5         # [m] moment reference point
REF_CHORD     = 3.0         # [m]
REF_AREA      = 0.60        # [m2]
NP            = 40           # MPI processes
END_TIME      = 0.01         # [s]
WRITE_INTERVAL = 0.001       # [s]
RESULTS_CSV   = "results_supersonic.csv"


def main():
    atm   = isa_atmosphere(ALTITUDE_M)
    u_inf = MACH * atm["a"]

    print(f"Mach {MACH} | {ALTITUDE_M/1000:.0f} km ISA | U={u_inf:.1f} m/s | "
          f"p={atm['p']} Pa | T={atm['T']} K | rho={atm['rho']} kg/m3")

    initialize_results_csv()
    design_point(atm, u_inf)

def design_point(atm, u_inf):
    job_id = f"run_{uuid.uuid4().hex[:8]}"
    job_directory = f"./{job_id}"

    print(f"\n{'='*40}")
    print(f"Starting Simulation: {job_id}")
    print(f"{'='*40}")

    # 1. Prepare Case Directory
    print("[1/5] Preparing case from template...")
    if not prepare(job_directory, atm, u_inf):
        print(f"Error: Failed to prepare case for {job_id}. Skipping...")
        return None

    # 2. Mesh Generation
    print("[2/5] Generating mesh (snappyHexMesh)...")
    try:
        if not mesh(job_directory):
            print(f"Error: Meshing failed to produce polyMesh for {job_id}. Skipping...")
            return None
    except Exception as e:
        print(f"Exception during meshing: {e}")

    # 3. Solve
    print("[3/5] Solving (rhoCentralFoam)...")
    try:
        if not solve(job_directory):
            print(f"Error: Solver failed to complete for {job_id}. Skipping...")
            return None
    except Exception as e:
        print(f"Exception during solving: {e}")

    # 4. Post Processing
    print("[4/5] Post processing in Paraview...")
    results = post_process(job_id)

    # 5. Clean Up
    print("[5/5] Cleaning up heavy mesh/processor files...")
    cleanup(job_directory)

    print(f"Successfully completed {job_id}!")

def isa_atmosphere(altitude_m):
    g0 = 9.80665;  R = 287.058;  gamma = 1.4
    T0 = 288.15;   p0 = 101325.0
    L  = -0.0065;  h_tp = 11000.0

    if altitude_m <= h_tp:
        T = T0 + L * altitude_m
        p = p0 * (T / T0) ** (-g0 / (L * R))
    else:
        T_tp = T0 + L * h_tp
        p_tp = p0 * (T_tp / T0) ** (-g0 / (L * R))
        T    = T_tp
        p    = p_tp * math.exp(-g0 * (altitude_m - h_tp) / (R * T_tp))

    rho = p / (R * T)
    a   = math.sqrt(gamma * R * T)
    mu  = 1.458e-6 * T**1.5 / (T + 110.4)
    nu  = mu / rho
    return dict(T=round(T,4), p=round(p,2), rho=round(rho,6), a=round(a,4), mu=mu, nu=nu)

def initialize_results_csv():
    if os.path.exists(RESULTS_CSV):
        os.remove(RESULTS_CSV)
        print(f"[*] Deleted existing {RESULTS_CSV}")
    with open(RESULTS_CSV, mode='w', newline='') as f:
        csv.writer(f).writerow(["Alpha", "CL", "CD", "LD_ratio", "CM", "Avg_yPlus", "Max_yPlus"])
    print(f"[*] Initialized fresh {RESULTS_CSV}")

def append_result(row):
    with open(RESULTS_CSV, mode='a', newline='') as f:
        csv.DictWriter(f, fieldnames=["Alpha", "CL", "CD", "LD_ratio", "CM",
                                      "Avg_yPlus", "Max_yPlus"]).writerow(row)

def prepare(job_directory, atm, u_inf):
    try:
        # Clone template
        if os.path.exists(job_directory):
            shutil.rmtree(job_directory)
        SolutionDirectory(CASE_TEMPLATE).cloneCase(job_directory)

        # decomposeParDict
        dpd = ParsedParameterFile(f"{job_directory}/system/decomposeParDict")
        dpd["numberOfSubdomains"] = NP
        dpd.writeFile()

        # controlDict
        cd = ParsedParameterFile(f"{job_directory}/system/controlDict")
        cd["endTime"]       = END_TIME
        cd["writeInterval"] = WRITE_INTERVAL
        fc = cd["functions"]["forceCoeffs"]
        fc["magUInf"]   = u_inf
        fc["lRef"]      = REF_CHORD
        fc["Aref"]      = REF_AREA
        fc["rhoInf"]    = atm["rho"]
        fc["pRef"]      = atm["p"]
        fc["CofR"]      = f"({CG_X} 0 0)"
        cd["functions"]["forces"]["CofR"]   = f"({CG_X} 0 0)"
        cd["functions"]["forces"]["rhoInf"] = atm["rho"]
        cd["functions"]["forces"]["pRef"]   = atm["p"]
        cd.writeFile()
        
        # Freestream variables
        TI, L_mix, Cmu = 0.001, 0.02, 0.09
        k_inf     = 1.5 * (TI * u_inf) ** 2
        omega_inf = k_inf**0.5 / (Cmu**0.25 * L_mix)
        vars_path = os.path.join(job_directory, "0", "include", "freeStreamVars")
        with open(vars_path, "w") as f:
            f.write(f"Uinf  ({u_inf:.6g} 0.0 0.0);\n")
            f.write(f"pInf  {atm['p']:.6g};\n")
            f.write(f"Tinf  {atm['T']:.6g};\n")
            f.write(f"rhoInf {atm['rho']:.6g};\n")
            f.write(f"kinf  {k_inf:.6g};\n")
            f.write(f"omegaInf {omega_inf:.6g};\n")

        return True
    except Exception as e:
        print(f"Exception during preparation: {e}")
        return False

def mesh(job_directory):
    os.makedirs(f"{job_directory}/constant/triSurface", exist_ok=True)

    COMMANDS = [
        f"surfaceFeatureExtract -case {job_directory}",
        f"blockMesh -case {job_directory}",
        f"decomposePar -case {job_directory}",
        f"mpirun -np {NP} snappyHexMesh -parallel -overwrite -case {job_directory}",
        f"reconstructParMesh -constant -case {job_directory}",
        f"rm -rf {job_directory}/processor*",
    ]
    
    triSurface_dir = f"{job_directory}/constant/triSurface"
    os.makedirs(triSurface_dir, exist_ok=True)
    shutil.copy(GEOMETRY_STL, f"{job_directory}/constant/triSurface/uav.stl")
    for command in COMMANDS:
        print(f"  -> Executing: {command}")
        result = subprocess.run(command, shell=True, executable='/bin/bash')
        if result.returncode != 0:
            print(f"Meshing step failed on command: {command}")
            return False

    return os.path.isdir(f"{job_directory}/constant/polyMesh")

def solve(job_directory):
    COMMANDS = [
        f"decomposePar -case {job_directory}",
        f"mpirun -np {NP} rhoCentralFoam -parallel -case {job_directory}",
        f"reconstructPar -latestTime -case {job_directory}",
    ]

    for command in COMMANDS:
        runner = BasicRunner(argv=command.split())
        runner.start()
        if not runner.runOK():
            raise Exception(f"{command} failed")

    subprocess.run(f"rm -rf {job_directory}/processor*", shell=True)

    # Check at least one time dir beyond 0 was written
    time_dirs = [d for d in os.listdir(job_directory)
                 if os.path.isdir(f"{job_directory}/{d}") and _is_float(d) and float(d) > 0]
    return len(time_dirs) > 0

def _is_float(s):
    try: float(s); return True
    except ValueError: return False

def post_process(job_id):
    command = f"LIBGL_ALWAYS_SOFTWARE=1 pvbatch --force-offscreen-rendering post_process.py {job_id}/{job_id}.foam {job_id}/images"
    try:        
        result = subprocess.run(command, shell=True, capture_output=True, text=True, executable='/bin/bash')
        if result.returncode != 0:
            print(f"Post-processing failed: {result.stderr}")
            return None
    except Exception as e:
        print(f"Error occurred while running post-processing: {e}")
        return None

    return result

def cleanup(job_directory):
    COMMANDS = [
        f"rm -rf {job_directory}/processor*",
        f"rm -rf {job_directory}/PyFoam*",
        "rm -rf PyFoam*",
    ]
    for command in COMMANDS:
        subprocess.run(command, shell=True, capture_output=True, text=True)

if __name__ == "__main__":
    main()