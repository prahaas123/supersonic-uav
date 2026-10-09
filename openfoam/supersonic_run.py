import os
import sys
import subprocess
import shutil
import re
import json
import math
import time
from PyFoam.RunDictionary.SolutionDirectory import SolutionDirectory
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
from PyFoam.Execution.BasicRunner import BasicRunner

import metrics
from create_uav import create_uav_model, reference_values

CASE_TEMPLATE = "case_template_supersonic"
ALTITUDE_M    = 15000.0     # [m]
MACH          = 1.5
NP            = 50           # MPI processes
END_TIME      = 0.03         # [s]
WRITE_INTERVAL = END_TIME    # [s] fields written only at the final time

HALF_MODEL    = True
SYMMETRY      = 2.0 if HALF_MODEL else 1.0

# Aircraft geometry
UAV_PARAMS = dict(
    # Wing placement
    x_location=0.0,
    z_location=0.08,
    y_rotation=1.5,
    # Segment 1 (root to break)
    seg1root_chord=4.0,
    seg2root_chord=1.6,  # break chord
    seg1_span=0.75,
    seg1_sweep=73.0,
    seg1_twist=0.0,
    seg1_dihedral=0.0,
    # Segment 2 (break to tip)
    seg2tip_chord=0.8,
    seg2_span=1.0,
    seg2_sweep=48.0,
    seg2_twist=-2.5,
    seg2_dihedral=0.0,
    # Airfoils (double wedge)
    tc_root=0.025,
    tc_break=0.035,
    tc_tip=0.050,
    thick_loc=0.5,
    # Wing tessellation
    wing_tess_w=101,
    seg1_tess_u=35,
    seg2_tess_u=35,
)

def main():
    atm   = isa_atmosphere(ALTITUDE_M)
    u_inf = MACH * atm["a"]

    print(f"Mach {MACH} | {ALTITUDE_M/1000:.0f} km ISA | U={u_inf:.1f} m/s | "
          f"p={atm['p']} Pa | T={atm['T']} K | rho={atm['rho']} kg/m3")

    results = design_point(atm, u_inf)
    if results is None:
        print("Run did not produce metrics.")
        return 1
    return 0 if results.get("converged") else 2

def design_point(atm, u_inf, uav_params=UAV_PARAMS, job_directory="./design_baseline"):
    job_id = os.path.basename(os.path.normpath(job_directory))

    print(f"\n{'='*40}")
    print(f"Starting Simulation: {job_id}")
    print(f"{'='*40}")

    # 1. Prepare Case Directory
    print("[1/6] Preparing case from template and generating UAV STL (OpenVSP)...")
    refs = prepare(job_directory, atm, u_inf, uav_params)
    if refs is None:
        print(f"Error: Failed to prepare case for {job_id}. Skipping...")
        return write_failure(job_directory, "prepare_failed")

    # 2. Mesh Generation
    print("[2/6] Generating mesh (snappyHexMesh)...")
    t_start = time.perf_counter()
    try:
        if not mesh(job_directory):
            print(f"Error: Meshing failed to produce polyMesh for {job_id}. Skipping...")
            return write_failure(job_directory, "mesh_failed", refs)
    except Exception as e:
        print(f"Exception during meshing: {e}")
        return write_failure(job_directory, "mesh_failed", refs)
    t_meshed = time.perf_counter()

    # 3. Solve
    print("[3/6] Solving (rhoCentralFoam)...")
    try:
        if not solve(job_directory):
            print(f"Error: Solver failed to complete for {job_id}. Skipping...")
            return write_failure(job_directory, "solve_failed", refs)
    except Exception as e:
        print(f"Exception during solving: {e}")
        return write_failure(job_directory, "solve_failed", refs)
    t_solved = time.perf_counter()

    timings = {
        "mesh_time_s":  round(t_meshed - t_start, 1),
        "solve_time_s": round(t_solved - t_meshed, 1),
        "total_time_s": round(t_solved - t_start, 1),
    }

    # 4. Metrics
    print("[4/6] Extracting converged metrics...")
    results = metrics.write_metrics(job_directory, symmetry_factor=SYMMETRY,
                                    extra={**refs, **timings})
    print(metrics.format_summary(results))

    if not results.get("converged"):
        print(f"WARNING: {job_id} did not meet the convergence tolerance "
              f"(drag drift {results.get('drag_drift_pct')} %). "
              f"Treat this result as unusable.")

    # 5. Post Processing
    print("[5/6] Post processing in Paraview...")
    post_process(job_directory)

    # 6. Clean Up
    print("[6/6] Cleaning up mesh/processor files...")
    cleanup(job_directory)

    print(f"Successfully completed {job_id}!")
    return results

def write_failure(job_directory, status, refs=None):
    os.makedirs(job_directory, exist_ok=True)
    with open(os.path.join(job_directory, "metrics.json"), "w") as fh:
        json.dump({"case": os.path.abspath(job_directory), "status": status,
                   "converged": False, **(refs or {})}, fh, indent=2)
    cleanup(job_directory)
    return None

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

def prepare(job_directory, atm, u_inf, uav_params):
    try:
        refs = reference_values(uav_params)

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
        cofr = f"({refs['cofr_x']:.6g} 0 0)"
        fc["magUInf"]   = u_inf
        fc["lRef"]      = refs["ref_chord"]
        fc["Aref"]      = refs["ref_area"]
        fc["rhoInf"]    = atm["rho"]
        fc["pRef"]      = atm["p"]
        fc["CofR"]      = cofr
        cd["functions"]["forces"]["CofR"]   = cofr
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

        # Geometry: generate the UAV STL into the case
        stl_path = os.path.join(job_directory, "constant", "triSurface", "uav.stl")
        if not create_uav_model(stl_path=stl_path, **uav_params):
            return None

        return refs
    except Exception as e:
        print(f"Exception during preparation: {e}")
        return None

def mesh(job_directory):
    COMMANDS = [
        f"surfaceFeatureExtract -case {job_directory}",
        f"blockMesh -case {job_directory}",
        f"decomposePar -case {job_directory}",
        f"mpirun -np {NP} snappyHexMesh -parallel -overwrite -case {job_directory}",
        f"reconstructParMesh -constant -case {job_directory}",
        f"rm -rf {job_directory}/processor*",
    ]
    
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

def post_process(job_directory):
    job_id = os.path.basename(os.path.normpath(job_directory))
    command = f"LIBGL_ALWAYS_SOFTWARE=1 pvbatch --force-offscreen-rendering post_process.py {job_directory}/{job_id}.foam {job_directory}/images"
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
    ]
    for command in COMMANDS:
        subprocess.run(command, shell=True, capture_output=True, text=True)

if __name__ == "__main__":
    sys.exit(main())