import os
import sys
import glob
import subprocess
import shutil
import re
import json
import math
import time
from PyFoam.RunDictionary.SolutionDirectory import SolutionDirectory
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
from PyFoam.Execution.BasicRunner import BasicRunner

TOOLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "0-tools")
sys.path.insert(0, TOOLS_DIR)

import metrics
from create_uav import create_uav_model, reference_values

CASE_TEMPLATE = "detail_template_supersonic"
DESIGN_ID     = 236
ALTITUDE_M    = 15000.0     # [m]
MACH          = 1.5
NP            = 50           # MPI processes
END_TIME      = 0.02        # [s]
WRITE_INTERVAL = END_TIME / 10   # [s] periodic writes so a timed-out job can be resumed

# Full model, no symmetry plane
SYMMETRY      = 1.0

# Boundary layer mesh: first cell height from a flat-plate estimate at the reference chord
TARGET_YPLUS  = 1.0

# Wall thermal condition: None = adiabatic (wallHeatFlux ~ 0),
# a value [K] = isothermal wall (wallHeatFlux gives the heat load at that wall temperature)
WALL_TEMPERATURE = None

# Freestream angles (alpha, beta) [deg], each case reuses the same mesh.
# Positive alpha: flow from below (+z component). Positive beta: wind from the
# right wing (+y), i.e. a -y velocity component. Keep both small (cruise).
FLOW_ANGLES = [
    (0.0, 0.0),
]

# Continue cases whose processor* directories survived a timed-out job
RESUME        = False

# Design 236 (0-results/dse_results.csv)
UAV_PARAMS = dict(
    # Wing placement
    x_location=0.0,
    y_rotation=2.181336,
    # Segment 1 (root to break)
    seg1root_chord=1.365551,
    seg1_taper=0.385108,
    seg1_span=0.355705,
    seg1_sweep=66.842209,
    seg1_twist=-0.939043,
    seg1_dihedral=0.0,
    # Segment 2 (break to tip)
    seg2_taper=0.314328,
    seg2_span=0.443973,
    seg2_sweep=50.29117,
    seg2_twist=-3.770442,
    seg2_dihedral=0.0,
    # Airfoils (double wedge)
    tc_root=0.033856,
    tc_break=0.036873,
    tc_tip=0.034586,
    thick_loc=0.332753,
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

    # 1. Mesh once, every flow angle reuses it
    mesh_directory = f"./design_{DESIGN_ID}_mesh"
    refs = reference_values(UAV_PARAMS)
    if RESUME and os.path.isdir(f"{mesh_directory}/constant/polyMesh"):
        print(f"Reusing existing mesh in {mesh_directory}")
    else:
        print("[mesh] Preparing case from template and generating UAV STL (OpenVSP)...")
        refs = prepare_mesh(mesh_directory, atm, u_inf, UAV_PARAMS)
        if refs is None:
            print("Error: Failed to prepare the mesh case.")
            return 1

        print("[mesh] Generating mesh (snappyHexMesh)...")
        t_start = time.perf_counter()
        try:
            if not mesh(mesh_directory):
                print("Error: Meshing failed to produce polyMesh.")
                return 1
        except Exception as e:
            print(f"Exception during meshing: {e}")
            return 1
        print(f"[mesh] done in {time.perf_counter() - t_start:.1f} s")

    # 2. Flow angle cases
    all_converged = True
    for alpha, beta in FLOW_ANGLES:
        results = design_point(atm, u_inf, refs, mesh_directory, alpha, beta)
        all_converged = all_converged and bool(results and results.get("converged"))

    return 0 if all_converged else 2

def design_point(atm, u_inf, refs, mesh_directory, alpha, beta):
    job_directory = f"./design_{DESIGN_ID}_a{alpha:g}_b{beta:g}"
    job_id = os.path.basename(os.path.normpath(job_directory))
    resume = RESUME and os.path.isdir(f"{job_directory}/processor0")

    print(f"\n{'='*40}")
    print(f"Starting Simulation: {job_id} | alpha {alpha} deg | beta {beta} deg"
          f"{' (resumed)' if resume else ''}")
    print(f"{'='*40}")

    # 1. Prepare Case Directory
    if not resume:
        print("[1/5] Copying mesh case and setting freestream...")
        if not prepare_case(job_directory, mesh_directory, atm, u_inf, refs, alpha, beta):
            print(f"Error: Failed to prepare case for {job_id}. Skipping...")
            return write_failure(job_directory, "prepare_failed", refs)

    # 2. Solve
    print("[2/5] Solving (rhoCentralFoam)...")
    t_start = time.perf_counter()
    try:
        if not solve(job_directory, resume):
            print(f"Error: Solver failed to complete for {job_id}. Skipping...")
            return write_failure(job_directory, "solve_failed", refs)
    except Exception as e:
        print(f"Exception during solving: {e}")
        return write_failure(job_directory, "solve_failed", refs)

    extra = {**refs, "alpha_deg": alpha, "beta_deg": beta,
             "wall_temperature_K": WALL_TEMPERATURE,
             "solve_time_s": round(time.perf_counter() - t_start, 1)}

    # 3. Metrics
    print("[3/5] Extracting converged metrics...")
    results = detail_metrics(job_directory, atm, u_inf, refs, extra)
    print(metrics.format_summary(results))
    print(format_detail_summary(results))

    if not results.get("converged"):
        print(f"WARNING: {job_id} did not meet the convergence tolerance "
              f"(Cd drift {results.get('Cd_drift_pct')} %). "
              f"Treat this result as unusable.")

    # 4. Post Processing
    # post_process.py rewrites metrics.json and the convergence plots with the
    # half-model factor, so both are written again afterwards
    print("[4/5] Post processing in Paraview...")
    post_process(job_directory)
    write_json(job_directory, results)
    plot_convergence(job_directory, results)

    # 5. Clean Up
    print("[5/5] Cleaning up processor files...")
    cleanup(job_directory)

    print(f"Successfully completed {job_id}!")
    return results

def write_failure(job_directory, status, refs=None):
    os.makedirs(job_directory, exist_ok=True)
    write_json(job_directory, {"case": os.path.abspath(job_directory), "status": status,
                               "converged": False, **(refs or {})})
    cleanup(job_directory)
    return None

def write_json(job_directory, data):
    with open(os.path.join(job_directory, "metrics.json"), "w") as fh:
        json.dump(data, fh, indent=2)

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

def first_layer_thickness(atm, u_inf, length, y_plus):
    re_l  = u_inf * length / atm["nu"]
    cf    = 0.026 / re_l ** (1.0 / 7.0)      # turbulent flat plate skin friction
    u_tau = u_inf * math.sqrt(0.5 * cf)
    return y_plus * atm["nu"] / u_tau

def flow_axes(alpha_deg, beta_deg):
    # Wind axes: drag along the freestream, lift normal to it in the x-z plane
    a, b = math.radians(alpha_deg), math.radians(beta_deg)
    drag  = (math.cos(a) * math.cos(b), -math.sin(b), math.sin(a) * math.cos(b))
    lift  = (-math.sin(a), 0.0, math.cos(a))
    pitch = (math.cos(a) * math.sin(b), math.cos(b), math.sin(a) * math.sin(b))  # lift x drag
    return drag, lift, pitch

def vec(v):
    return "(" + " ".join(f"{c:.8g}" for c in v) + ")"

def prepare_mesh(job_directory, atm, u_inf, uav_params):
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

        # Boundary layer: first cell height for TARGET_YPLUS
        y1 = first_layer_thickness(atm, u_inf, refs["ref_chord"], TARGET_YPLUS)
        shm_path = f"{job_directory}/system/snappyHexMeshDict"
        with open(shm_path, "r") as fh:
            shm = fh.read()
        shm = re.sub(r"firstLayerThickness\s+[^;]+;", f"firstLayerThickness {y1:.4g};", shm)
        shm = re.sub(r"minThickness\s+[^;]+;", f"minThickness {y1:.4g};", shm)
        with open(shm_path, "w") as fh:
            fh.write(shm)
        print(f"  first layer thickness {y1*1e6:.2f} um for y+ = {TARGET_YPLUS:g}")

        # Geometry: generate the UAV STL into the case
        stl_path = os.path.join(job_directory, "constant", "triSurface", "uav.stl")
        if not create_uav_model(stl_path=stl_path, **uav_params):
            return None

        return {**refs, "first_layer_m": y1, "target_yplus": TARGET_YPLUS}
    except Exception as e:
        print(f"Exception during preparation: {e}")
        return None

def prepare_case(job_directory, mesh_directory, atm, u_inf, refs, alpha, beta):
    try:
        if os.path.exists(job_directory):
            shutil.rmtree(job_directory)
        shutil.copytree(mesh_directory, job_directory,
                        ignore=shutil.ignore_patterns("processor*", "PyFoam*", "*.foam", "log.*"))
        job_id = os.path.basename(os.path.normpath(job_directory))
        open(os.path.join(job_directory, f"{job_id}.foam"), "w").close()

        drag, lift, pitch = flow_axes(alpha, beta)
        u_vec = tuple(u_inf * c for c in drag)

        # controlDict
        cd = ParsedParameterFile(f"{job_directory}/system/controlDict")
        cd["endTime"]       = END_TIME
        cd["writeInterval"] = WRITE_INTERVAL
        fc = cd["functions"]["forceCoeffs"]
        cofr = f"({refs['cg_x']:.6g} 0 0)"
        fc["magUInf"]   = u_inf
        fc["lRef"]      = refs["ref_chord"]
        fc["Aref"]      = refs["ref_area"]
        fc["rhoInf"]    = atm["rho"]
        fc["pRef"]      = atm["p"]
        fc["CofR"]      = cofr
        fc["dragDir"]   = vec(drag)
        fc["liftDir"]   = vec(lift)
        fc["pitchAxis"] = vec(pitch)
        cd["functions"]["forces"]["CofR"]   = cofr
        cd["functions"]["forces"]["rhoInf"] = atm["rho"]
        cd["functions"]["forces"]["pRef"]   = atm["p"]
        cd["functions"]["Cp"]["pInf"]   = atm["p"]
        cd["functions"]["Cp"]["UInf"]   = vec(u_vec)
        cd["functions"]["Cp"]["rhoInf"] = atm["rho"]
        cd.writeFile()

        # Freestream variables
        TI, L_mix, Cmu = 0.001, 0.02, 0.09
        k_inf     = 1.5 * (TI * u_inf) ** 2
        omega_inf = k_inf**0.5 / (Cmu**0.25 * L_mix)
        include_dir = os.path.join(job_directory, "0", "include")
        os.makedirs(include_dir, exist_ok=True)
        with open(os.path.join(include_dir, "freeStreamVars"), "w") as f:
            f.write(f"Uinf  {vec(u_vec)};\n")
            f.write(f"pInf  {atm['p']:.6g};\n")
            f.write(f"Tinf  {atm['T']:.6g};\n")
            f.write(f"rhoInf {atm['rho']:.6g};\n")
            f.write(f"kinf  {k_inf:.6g};\n")
            f.write(f"omegaInf {omega_inf:.6g};\n")

        # Wall thermal condition
        with open(os.path.join(include_dir, "wallThermal"), "w") as f:
            if WALL_TEMPERATURE is None:
                f.write("type            zeroGradient;\n")
            else:
                f.write("type            fixedValue;\n")
                f.write(f"value           uniform {WALL_TEMPERATURE:.6g};\n")

        return True
    except Exception as e:
        print(f"Exception during preparation: {e}")
        return False

def mesh(job_directory):
    COMMANDS = [
        f"surfaceFeatureExtract -case {job_directory}",
        f"blockMesh -case {job_directory}",
        f"decomposePar -case {job_directory}",
        f"mpirun -np {NP} snappyHexMesh -parallel -overwrite -case {job_directory}",
        f"reconstructParMesh -constant -case {job_directory}",
        f"rm -rf {job_directory}/processor*",
        # Mesh quality and layer report, kept for reference only
        f"checkMesh -constant -case {job_directory} > {job_directory}/log.checkMesh 2>&1 || true",
    ]

    for command in COMMANDS:
        print(f"  -> Executing: {command}")
        result = subprocess.run(command, shell=True, executable='/bin/bash')
        if result.returncode != 0:
            print(f"Meshing step failed on command: {command}")
            return False

    return os.path.isdir(f"{job_directory}/constant/polyMesh")

def solve(job_directory, resume=False):
    COMMANDS = [
        f"mpirun -np {NP} rhoCentralFoam -parallel -case {job_directory}",
        f"reconstructPar -latestTime -case {job_directory}",
    ]
    if not resume:
        COMMANDS.insert(0, f"decomposePar -force -case {job_directory}")

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

def read_coefficients(job_directory):
    # Coefficient names from the file header, the column order changes between OpenFOAM versions
    t, C = metrics._read_dat(job_directory, "forceCoeffs", "coefficient.dat", min_cols=4)
    if t is None:
        return None, {}
    path = sorted(glob.glob(os.path.join(
        job_directory, "postProcessing", "forceCoeffs", "*", "coefficient.dat")))[0]
    names = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("# Time"):
                names = line.lstrip("#").split()[1:]
                break
    return t, {name: C[:, i] for i, name in enumerate(names) if i < C.shape[1]}

def read_wall_heat_flux(job_directory, patch="uav"):
    # Last write time row: min, max [W/m2] and integrated heat load [W]
    files = sorted(glob.glob(os.path.join(
        job_directory, "postProcessing", "wallHeatFlux", "*", "wallHeatFlux.dat")),
        key=metrics._start_time)
    row = None
    for path in files:
        with open(path) as fh:
            for line in fh:
                parts = line.split()
                if not line.startswith("#") and len(parts) >= 5 and parts[1] == patch:
                    row = parts
    if row is None:
        return {}
    return {"heat_flux_min_W_m2": float(row[2]), "heat_flux_max_W_m2": float(row[3]),
            "heat_load_W": float(row[4])}

def detail_metrics(job_directory, atm, u_inf, refs, extra):
    m = metrics.extract_metrics(job_directory, symmetry_factor=SYMMETRY)
    m.update(extra)

    # forces/force.dat is in body (x, y, z) axes, these are the wind-axis coefficients
    tc, coeffs = read_coefficients(job_directory)
    q_s = 0.5 * atm["rho"] * u_inf**2 * refs["ref_area"]
    span = 2.0 * refs["semi_span"]
    for name in ("Cd", "Cs", "Cl", "CmRoll", "CmPitch", "CmYaw"):
        if name not in coeffs:
            continue
        st = metrics.window_stats(tc, coeffs[name], metrics.TAIL_FRACTION, SYMMETRY)
        if st:
            m[name] = st["value"]
            m[f"{name}_std"] = st["std"]
            m[f"{name}_drift_pct"] = st["drift_pct"]

    if "Cd" in m and "Cl" in m:
        m["drag_wind_N"] = m["Cd"] * q_s
        m["lift_wind_N"] = m["Cl"] * q_s
        m["LD_ratio"] = m["Cl"] / m["Cd"] if m["Cd"] else None
    if "Cs" in m:
        m["side_wind_N"] = m["Cs"] * q_s
    # forceCoeffs uses lRef (MAC) for every moment, roll and yaw are usually per span
    for name, key in (("CmRoll", "Cl_roll_span"), ("CmYaw", "Cn_yaw_span")):
        if name in m:
            m[key] = m[name] * refs["ref_chord"] / span

    m.update(read_wall_heat_flux(job_directory))

    # Converged on the wind-axis drag coefficient
    drift = m.get("Cd_drift_pct")
    m["converged"] = drift is not None and abs(drift) < metrics.DRIFT_TOLERANCE_PCT
    if m["status"] in ("ok", "not_converged"):
        m["status"] = "ok" if m["converged"] else "not_converged"

    write_json(job_directory, m)
    return m

def format_detail_summary(m):
    def fmt(key, spec=".5f", default="n/a"):
        v = m.get(key)
        return format(v, spec) if isinstance(v, (int, float)) else default

    bar = "=" * 52
    return "\n".join([
        f" WIND AXES   alpha {fmt('alpha_deg', 'g')} deg | beta {fmt('beta_deg', 'g')} deg",
        bar,
        f"  Drag  (wind)      :  {fmt('drag_wind_N', '10.3f')} N",
        f"  Lift  (wind)      :  {fmt('lift_wind_N', '10.3f')} N",
        f"  Side  (wind)      :  {fmt('side_wind_N', '10.3f')} N",
        f"  Cd / Cl / Cs      :  {fmt('Cd')} / {fmt('Cl')} / {fmt('Cs')}",
        f"  Cm pitch (MAC)    :  {fmt('CmPitch')}",
        f"  Cl roll / Cn yaw  :  {fmt('Cl_roll_span')} / {fmt('Cn_yaw_span')}  (per span)",
        f"  L/D               :  {fmt('LD_ratio', '.4f')}",
        f"  Cd drift          :  {fmt('Cd_drift_pct', '+.4f')} %",
        f"  wall heat flux    :  {fmt('heat_flux_min_W_m2', '.4g')} to "
        f"{fmt('heat_flux_max_W_m2', '.4g')} W/m2",
        f"  heat load         :  {fmt('heat_load_W', '.4g')} W  "
        f"(wall T: {fmt('wall_temperature_K', 'g', 'adiabatic')})",
        bar, "",
    ])

def plot_convergence(job_directory, m):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        tc, coeffs = read_coefficients(job_directory)
        if tc is None:
            return
        images = os.path.join(job_directory, "images")
        os.makedirs(images, exist_ok=True)
        # Body-axis force plots from post_process.py are doubled for a full model
        for stale in ("convergence_lift.png", "convergence_drag.png"):
            if os.path.exists(os.path.join(images, stale)):
                os.remove(os.path.join(images, stale))

        tail = metrics.TAIL_FRACTION
        for name in ("Cd", "Cl", "Cs", "CmPitch", "CmRoll", "CmYaw"):
            if name not in coeffs:
                continue
            t, y = tc, coeffs[name] * SYMMETRY
            t_cut = t[0] + (1.0 - tail) * (t[-1] - t[0])
            plt.figure(figsize=(8, 5))
            plt.plot(t, y, linewidth=1.2)
            plt.axvspan(t_cut, t[-1], color="0.6", alpha=0.25,
                        label=f"averaging window (last {100*tail:.0f}%)")
            if m.get(name) is not None:
                plt.axhline(m[name], color="k", linestyle="--", linewidth=1.2,
                            label=f"mean = {m[name]:.5g}")
            plt.xlabel("Time (s)")
            plt.ylabel(name)
            plt.title(f"{name} Convergence (wind axes)")
            plt.grid(True, linestyle="--", alpha=0.7)
            # Clip the y-range to the settled part, the startup spike flattens everything else
            y_tail = y[t >= t_cut]
            if len(y_tail) > 1:
                pad = max(6.0 * (y_tail.max() - y_tail.min()), 0.02 * abs(y_tail.mean()), 1e-6)
                plt.ylim(y_tail.min() - pad, y_tail.max() + pad)
            plt.legend(fontsize=8)
            plt.savefig(os.path.join(images, f"convergence_{name}.png"), dpi=200, bbox_inches="tight")
            plt.close()
    except Exception as e:
        print(f"Warning: Could not plot convergence histories. Error: {e}")

def post_process(job_directory):
    job_id = os.path.basename(os.path.normpath(job_directory))
    command = f"LIBGL_ALWAYS_SOFTWARE=1 pvbatch --force-offscreen-rendering {os.path.join(TOOLS_DIR, 'post_process.py')} {job_directory}/{job_id}.foam {job_directory}/images"
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
