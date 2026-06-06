import os
import subprocess
import shutil
import re
import csv
import glob
import math
import numpy as np
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
ALPHAS        = [0.0]       # [deg]
NP            = 50           # MPI processes
END_TIME      = 0.5         # [s]
WRITE_INTERVAL = 0.01       # [s]
RESULTS_CSV   = "results_supersonic.csv"


def main():
    atm   = isa_atmosphere(ALTITUDE_M)
    u_inf = MACH * atm["a"]

    print(f"Mach {MACH} | {ALTITUDE_M/1000:.0f} km ISA | U={u_inf:.1f} m/s | "
          f"p={atm['p']} Pa | T={atm['T']} K | rho={atm['rho']} kg/m3")

    initialize_results_csv()

    for alpha in ALPHAS:
        job_id        = f"run_alpha_{int(alpha)}"
        job_directory = f"./{job_id}"

        print(f"\n{'='*40}")
        print(f"Starting Simulation: {job_id} | Alpha: {alpha} deg")
        print(f"{'='*40}")

        # 1. Prepare Case Directory
        print("[1/5] Preparing case from template...")
        if not prepare(job_directory, alpha, atm, u_inf):
            print(f"Error: Failed to prepare case for {job_id}. Skipping...")
            continue

        # 2. Mesh Generation
        print("[2/5] Generating mesh (snappyHexMesh)...")
        try:
            if not mesh(job_directory, alpha):
                print(f"Error: Meshing failed to produce polyMesh for {job_id}. Skipping...")
                continue
        except Exception as e:
            print(f"Exception during meshing: {e}")
            continue

        # 3. Solve
        print("[3/5] Solving (rhoCentralFoam)...")
        try:
            if not solve(job_directory):
                print(f"Error: Solver failed to complete for {job_id}. Skipping...")
                continue
        except Exception as e:
            print(f"Exception during solving: {e}")
            continue

        # 4. Extract Results
        print("[4/5] Extracting results...")
        results = extract_results(job_directory, alpha, atm, u_inf)
        append_result(results)
        print(f"  CL={results['CL']}  CD={results['CD']}  CM={results['CM']}  L/D={results['LD_ratio']}")

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

def prepare(job_directory, alpha_deg, atm, u_inf):
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
        vars_path = os.path.join(job_directory, "0", "include", "freestreamVars")
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

def mesh(job_directory, alpha):
    os.makedirs(f"{job_directory}/constant/triSurface", exist_ok=True)

    COMMANDS = [
        f"{GEOMETRY_STL} {job_directory}/constant/triSurface/uav.stl",
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

def extract_results(job_directory, alpha, atm, u_inf):
    result = {"Alpha": alpha, "CL": None, "CD": None, "LD_ratio": None,
              "CM": None, "Avg_yPlus": None, "Max_yPlus": None}

    # forceCoeffs — columns: Time  Cm  Cd  Cl
    fc_files = sorted(glob.glob(f"{job_directory}/postProcessing/forceCoeffs/*/forceCoeffs.dat"))
    if fc_files:
        times, CLs, CDs, CMs = [], [], [], []
        with open(fc_files[-1]) as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"): continue
                parts = s.split()
                if len(parts) < 4: continue
                try:
                    times.append(float(parts[0])); CMs.append(float(parts[1]))
                    CDs.append(float(parts[2]));   CLs.append(float(parts[3]))
                except ValueError: continue
        if times:
            times = np.array(times); CLs = np.array(CLs); CDs = np.array(CDs); CMs = np.array(CMs)
            mask  = times >= times[-1] - 0.25 * (times[-1] - times[0])
            CL_m  = float(np.mean(CLs[mask])); CD_m = float(np.mean(CDs[mask]))
            result.update({
                "CL": round(CL_m, 6), "CD": round(CD_m, 6), "CM": round(float(np.mean(CMs[mask])), 6),
                "LD_ratio": round(CL_m / CD_m, 4) if abs(CD_m) > 1e-10 else None,
            })

    # y+
    yp_files = sorted(glob.glob(f"{job_directory}/postProcessing/yPlus/*/yPlus.dat"))
    if yp_files:
        avgs, maxs = [], []
        with open(yp_files[-1]) as f:
            for line in f:
                if line.strip().startswith("#") or not line.strip(): continue
                parts = line.split()
                if len(parts) >= 3:
                    try: avgs.append(float(parts[1])); maxs.append(float(parts[2]))
                    except ValueError: pass
        if avgs:
            result["Avg_yPlus"] = round(float(np.mean(avgs[-5:])), 2)
            result["Max_yPlus"] = round(float(np.max(maxs[-5:])),  2)

    return result

def cleanup(job_directory):
    COMMANDS = [
        f"rm -rf {job_directory}/constant/polyMesh",
        f"rm -rf {job_directory}/constant/extendedFeatureEdgeMesh",
        f"rm -rf {job_directory}/processor*",
        f"rm -rf {job_directory}/PyFoam*",
        "rm -rf PyFoam*",
    ]
    for command in COMMANDS:
        subprocess.run(command, shell=True, capture_output=True, text=True)

if __name__ == "__main__":
    main()