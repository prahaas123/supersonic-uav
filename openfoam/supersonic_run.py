import os
import re
import csv
import glob
import math
import shutil
import argparse
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime
from PyFoam.RunDictionary.SolutionDirectory import SolutionDirectory
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
from PyFoam.Execution.BasicRunner import BasicRunner

GEOMETRY_STL      = r"case_template_supersonic/constant/geometry/uav.stl"                   # Source STL in working directory
CASE_TEMPLATE     = "case_template_supersonic"  # Template folder name
ALTITUDE_M        = 15000.0                     # [m] ISA altitude
MACH              = 1.5                         # Freestream Mach number

# UAV reference geometry — ADJUST TO YOUR UAV
CG_X              = 0.5             # Centre of gravity / moment ref, x [m]
REF_CHORD         = 3.0             # Reference chord length lRef [m]
REF_AREA          = 0.60            # Reference planform area Aref 
ALPHAS            = [0.0]           # Angles of attack [deg]
NP                = 4               # MPI processes (match decomposeParDict)

# Transient solver control
END_TIME          = 0.5             # [s] Physical end time — shocks stabilise ~0.2–0.4 s
WRITE_INTERVAL    = 0.01            # [s] Write interval for time directories
AVERAGING_FRAC    = 0.25            # Average last 25% of time history for coefficients

# Convergence tolerance for CL, CD (relative change between last two windows)
CONVERGENCE_TOL   = 0.005           # 0.5% — set None to disable check

# Results
RESULTS_CSV       = "results_supersonic.csv"

# ══════════════════════════════════════════════════════════════════════════════
#  ISA ATMOSPHERE MODEL
# ══════════════════════════════════════════════════════════════════════════════

def isa_atmosphere(altitude_m: float) -> dict:
    """
    Compute ISA standard atmosphere at given altitude.
    Valid for troposphere (0–11 km) and lower stratosphere (11–20 km).
    Returns dict with T [K], p [Pa], rho [kg/m³], a [m/s], mu [Pa·s], nu [m²/s].
    """
    g0    = 9.80665       # [m/s²]
    R     = 287.058       # [J/(kg·K)]
    gamma = 1.4
    T0    = 288.15        # [K]  sea-level temperature
    p0    = 101325.0      # [Pa] sea-level pressure
    L     = -0.0065       # [K/m] lapse rate (troposphere)
    h_tp  = 11000.0       # [m]  tropopause altitude

    if altitude_m <= h_tp:
        T   = T0 + L * altitude_m
        p   = p0 * (T / T0) ** (-g0 / (L * R))
    else:
        # Isothermal stratosphere
        T_tp = T0 + L * h_tp
        p_tp = p0 * (T_tp / T0) ** (-g0 / (L * R))
        T    = T_tp
        p    = p_tp * math.exp(-g0 * (altitude_m - h_tp) / (R * T_tp))

    rho = p / (R * T)
    a   = math.sqrt(gamma * R * T)

    # Sutherland viscosity
    As, Ts = 1.458e-6, 110.4
    mu  = As * T**1.5 / (T + Ts)
    nu  = mu / rho

    return {
        "T":   round(T,   4),
        "p":   round(p,   2),
        "rho": round(rho, 6),
        "a":   round(a,   4),
        "mu":  mu,
        "nu":  nu,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def log(msg: str, level: str = "INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{level:5s}] {msg}", flush=True)


def run_cmd(cmd: str, cwd: str = None, fatal: bool = True) -> bool:
    """Run a shell command, stream output, return success flag."""
    log(f"  → {cmd}")
    result = subprocess.run(
        cmd, shell=True, executable="/bin/bash",
        cwd=cwd, text=True,
    )
    if result.returncode != 0:
        log(f"Command FAILED (rc={result.returncode}): {cmd}", "ERROR")
        if fatal:
            raise RuntimeError(f"Command failed: {cmd}")
        return False
    return True


def run_pyfoam(cmd: str, fatal: bool = True) -> bool:
    """Run a command via PyFoam BasicRunner (captures residuals etc.)."""
    log(f"  → {cmd}")
    runner = BasicRunner(argv=cmd.split())
    runner.start()
    if not runner.runOK():
        log(f"PyFoam runner FAILED: {cmd}", "ERROR")
        if fatal:
            raise RuntimeError(f"PyFoam runner failed: {cmd}")
        return False
    return True


def foam_vec(x: float, y: float, z: float) -> str:
    """Format a 3-component OpenFOAM vector string."""
    return f"({x:.6g} {y:.6g} {z:.6g})"


def initialize_results_csv():
    """Initialise a fresh results CSV with headers."""
    if os.path.exists(RESULTS_CSV):
        os.remove(RESULTS_CSV)
        log(f"Deleted existing {RESULTS_CSV}")
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Alpha_deg", "Mach", "Altitude_m",
            "CL", "CD", "CM", "LD_ratio",
            "CL_std", "CD_std",                  # stability indicators
            "p_inf_Pa", "T_inf_K", "rho_inf",
            "U_inf_ms", "Re_chord",
            "Avg_yPlus", "Max_yPlus",
            "Converged", "Job_dir",
        ])
    log(f"Initialised {RESULTS_CSV}")


def append_result(row: dict):
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Alpha_deg", "Mach", "Altitude_m",
            "CL", "CD", "CM", "LD_ratio",
            "CL_std", "CD_std",
            "p_inf_Pa", "T_inf_K", "rho_inf",
            "U_inf_ms", "Re_chord",
            "Avg_yPlus", "Max_yPlus",
            "Converged", "Job_dir",
        ])
        writer.writerow(row)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — PREPARE CASE DIRECTORY
# ══════════════════════════════════════════════════════════════════════════════

def prepare(job_dir: str, alpha_deg: float, atm: dict, u_inf: float,
            np_cores: int, end_time: float) -> bool:
    """
    Clone template, patch all dictionaries for this AoA and flight condition.
    """
    try:
        alpha_rad = math.radians(alpha_deg)
        cos_a = math.cos(alpha_rad)
        sin_a = math.sin(alpha_rad)

        # Velocity components in body frame (flow rotated by AoA in XY plane)
        Ux = u_inf * cos_a
        Uy = u_inf * sin_a
        Uz = 0.0

        # Aerodynamic axis unit vectors (world frame)
        drag_dir  = foam_vec( cos_a,  sin_a, 0)   # stream direction
        lift_dir  = foam_vec(-sin_a,  cos_a, 0)   # perpendicular (up)
        pitch_ax  = foam_vec(0, 0, 1)              # +Z

        # ── Clone template ────────────────────────────────────────────────────
        log("Cloning case template...")
        if os.path.exists(job_dir):
            shutil.rmtree(job_dir)
        template = SolutionDirectory(CASE_TEMPLATE)
        template.cloneCase(job_dir)

        # ── decomposeParDict ──────────────────────────────────────────────────
        dpd_path = f"{job_dir}/system/decomposeParDict"
        dpd = ParsedParameterFile(dpd_path)
        dpd["numberOfSubdomains"] = np_cores
        dpd.writeFile()

        # ── controlDict ───────────────────────────────────────────────────────
        cd_path = f"{job_dir}/system/controlDict"
        cd = ParsedParameterFile(cd_path)
        cd["endTime"]      = end_time
        cd["writeInterval"] = WRITE_INTERVAL

        # forceCoeffs
        fc = cd["functions"]["forceCoeffs"]
        fc["magUInf"]   = u_inf
        fc["lRef"]      = REF_CHORD
        fc["Aref"]      = REF_AREA
        fc["rhoInf"]    = atm["rho"]
        fc["pRef"]      = atm["p"]
        fc["CofR"]      = foam_vec(CG_X, 0, 0)
        fc["liftDir"]   = lift_dir
        fc["dragDir"]   = drag_dir
        fc["pitchAxis"] = pitch_ax

        # forces
        fo = cd["functions"]["forces"]
        fo["rhoInf"] = atm["rho"]
        fo["pRef"]   = atm["p"]
        fo["CofR"]   = foam_vec(CG_X, 0, 0)

        # Cp functionObject
        cp_fo = cd["functions"]["Cp"]
        cp_fo["pInf"]   = atm["p"]
        cp_fo["UInf"]   = foam_vec(Ux, Uy, Uz)
        cp_fo["rhoInf"] = atm["rho"]

        cd.writeFile()

        # ── 0/U ──────────────────────────────────────────────────────────────
        _patch_U(job_dir, Ux, Uy, Uz)

        # ── 0/p ──────────────────────────────────────────────────────────────
        _patch_scalar(job_dir, "p",   atm["p"],
                      fixed_patches=["inlet"],
                      freestream_patches=["farfield_top","farfield_bottom","farfield_sides"])

        # ── 0/T ──────────────────────────────────────────────────────────────
        _patch_scalar(job_dir, "T",   atm["T"],
                      fixed_patches=["inlet"],
                      freestream_patches=["farfield_top","farfield_bottom","farfield_sides"])

        # ── 0/rho ─────────────────────────────────────────────────────────────
        _patch_scalar(job_dir, "rho", atm["rho"],
                      fixed_patches=["inlet"],
                      freestream_patches=["farfield_top","farfield_bottom","farfield_sides"])

        # ── Turbulence: k, omega ──────────────────────────────────────────────
        #  TI = 0.1% (high-altitude freestream)
        TI      = 0.001
        L_mix   = 0.02      # [m] 1% of wingspan
        Cmu     = 0.09
        k_inf   = 1.5 * (TI * u_inf) ** 2
        omega_inf = k_inf**0.5 / (Cmu**0.25 * L_mix)

        _patch_scalar(job_dir, "k",     k_inf,
                      fixed_patches=["inlet"],
                      freestream_patches=["farfield_top","farfield_bottom","farfield_sides"])
        _patch_scalar(job_dir, "omega", omega_inf,
                      fixed_patches=["inlet"],
                      freestream_patches=["farfield_top","farfield_bottom","farfield_sides"])

        log(f"Freestream: p={atm['p']:.1f} Pa  T={atm['T']:.2f} K  "
            f"rho={atm['rho']:.4f} kg/m³  U={u_inf:.2f} m/s")
        log(f"Turbulence: k={k_inf:.4f} m²/s²  omega={omega_inf:.1f} s⁻¹")
        log(f"AoA={alpha_deg}°  Ux={Ux:.3f} Uy={Uy:.3f} m/s")
        return True

    except Exception as e:
        log(f"Preparation failed: {e}", "ERROR")
        return False


def _patch_U(job_dir: str, Ux: float, Uy: float, Uz: float):
    """Rewrite velocity values in 0/U using regex (robust to OpenFOAM formatting)."""
    u_path = f"{job_dir}/0/U"
    vec    = f"({Ux:.6g} {Uy:.6g} {Uz:.6g})"
    with open(u_path) as f:
        content = f.read()
    # Replace any uniform (...) tuple in internalField and fixedValue patches
    content = re.sub(
        r'(internalField\s+uniform\s*)\([^)]+\)',
        rf'\g<1>{vec}', content
    )
    content = re.sub(
        r'(type\s+fixedValue;\s*value\s+uniform\s*)\([^)]+\)',
        rf'\g<1>{vec}', content
    )
    content = re.sub(
        r'(freestreamValue\s+uniform\s*)\([^)]+\)',
        rf'\g<1>{vec}', content
    )
    with open(u_path, "w") as f:
        f.write(content)


def _patch_scalar(job_dir: str, field: str, value: float,
                  fixed_patches: list, freestream_patches: list):
    """Patch a scalar field file: internalField + inlet fixedValue + freestream."""
    fpath = f"{job_dir}/0/{field}"
    with open(fpath) as f:
        content = f.read()

    # internalField
    content = re.sub(
        r'(internalField\s+uniform\s+)[0-9eE+\-\.]+',
        rf'\g<1>{value:.6g}', content
    )
    # fixedValue on fixed patches (just replace numeric value after "uniform")
    content = re.sub(
        r'(type\s+fixedValue;\s*value\s+uniform\s+)[0-9eE+\-\.]+',
        rf'\g<1>{value:.6g}', content
    )
    # freestreamValue
    content = re.sub(
        r'(freestreamValue\s+uniform\s+)[0-9eE+\-\.]+',
        rf'\g<1>{value:.6g}', content
    )

    with open(fpath, "w") as f:
        f.write(content)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — MESH
# ══════════════════════════════════════════════════════════════════════════════

def mesh(job_dir: str, alpha_deg: float, np_cores: int) -> bool:
    geom_dir = f"{job_dir}/constant/geometry"
    os.makedirs(geom_dir, exist_ok=True)

    stl_dest = f"{geom_dir}/uav.stl"

    commands = [
        # Rotate STL by -alpha around Y axis so freestream stays along +X
        (f"surfaceTransformPoints "
         f"-rotate-angle '((0 1 0) {-alpha_deg})' "
         f"{GEOMETRY_STL} {stl_dest}"),

        # Extract sharp features for snapping
        f"surfaceFeatureExtract -case {job_dir}",

        # Background hex mesh
        f"blockMesh -case {job_dir}",

        # Decompose for parallel snappy
        f"decomposePar -copyZero -case {job_dir}",

        # snappyHexMesh in parallel
        f"mpirun -np {np_cores} snappyHexMesh -parallel -overwrite -case {job_dir}",

        # Reconstruct mesh (fields stay decomposed for solver)
        f"reconstructParMesh -constant -case {job_dir}",

        # Remove processor dirs (will re-decompose for solver)
        f"rm -rf {job_dir}/processor*",
    ]

    for cmd in commands:
        if not run_cmd(cmd, fatal=False):
            return False

    # Verify polyMesh was created
    poly_mesh = f"{job_dir}/constant/polyMesh"
    if not os.path.isdir(poly_mesh):
        log("polyMesh directory missing after snappyHexMesh", "ERROR")
        return False

    # Quick mesh quality check — parse non-orthogonality
    log("Running checkMesh...")
    result = subprocess.run(
        f"checkMesh -case {job_dir}",
        shell=True, text=True, capture_output=True
    )
    _summarise_checkmesh(result.stdout + result.stderr)

    return True


def _summarise_checkmesh(output: str):
    """Extract and print key mesh quality metrics from checkMesh output."""
    metrics = {
        "Max non-orthogonality":  r"Max non-orthogonality\s*=\s*([\d.]+)",
        "Max skewness":           r"Max skewness\s*=\s*([\d.]+)",
        "Max aspect ratio":       r"Max aspect ratio\s*=\s*([\d.]+)",
        "Total cells":            r"cells:\s*(\d+)",
    }
    log("── checkMesh summary ─────────────────────────────────────")
    for label, pattern in metrics.items():
        m = re.search(pattern, output)
        log(f"  {label:<30} {m.group(1) if m else 'not found'}")
    log("──────────────────────────────────────────────────────────")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — SOLVE (rhoCentralFoam, transient)
# ══════════════════════════════════════════════════════════════════════════════

def solve(job_dir: str, np_cores: int, end_time: float) -> bool:
    """
    Decompose, run rhoCentralFoam in parallel, reconstruct the last time step.
    rhoCentralFoam is a transient explicit solver — no redistributePar needed.
    """
    commands = [
        # Decompose fields for parallel run
        f"decomposePar -case {job_dir}",

        # Run solver
        f"mpirun -np {np_cores} rhoCentralFoam -parallel -case {job_dir}",

        # Reconstruct only the final time (saves time and disk)
        f"reconstructPar -latestTime -case {job_dir}",

        # Remove processor dirs to save disk
        f"rm -rf {job_dir}/processor*",
    ]

    for cmd in commands:
        if not run_cmd(cmd, fatal=False):
            return False

    # Check that at least one time directory beyond 0 was written
    time_dirs = _find_time_dirs(job_dir)
    if not time_dirs:
        log("No time directories found after solve — solver may have crashed", "ERROR")
        return False

    latest = max(time_dirs)
    log(f"Latest time directory: {latest:.4g} s (end_time={end_time} s)")
    return True


def _find_time_dirs(job_dir: str) -> list:
    """Return list of numeric time directory values (excluding 0)."""
    times = []
    for d in Path(job_dir).iterdir():
        if d.is_dir():
            try:
                t = float(d.name)
                if t > 0:
                    times.append(t)
            except ValueError:
                pass
    return sorted(times)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — EXTRACT RESULTS (force coefficients + y+)
# ══════════════════════════════════════════════════════════════════════════════

def extract_results(job_dir: str, alpha_deg: float, atm: dict,
                    u_inf: float) -> dict:
    """
    Parse forceCoeffs.dat and yPlus data from postProcessing directory.
    Averages the last AVERAGING_FRAC of the time history.
    """
    results = {
        "Alpha_deg": alpha_deg,
        "Mach":      MACH,
        "Altitude_m": ALTITUDE_M,
        "p_inf_Pa":  atm["p"],
        "T_inf_K":   atm["T"],
        "rho_inf":   atm["rho"],
        "U_inf_ms":  round(u_inf, 3),
        "Re_chord":  round(u_inf * REF_CHORD / atm["nu"], 0),
        "CL": None, "CD": None, "CM": None, "LD_ratio": None,
        "CL_std": None, "CD_std": None,
        "Avg_yPlus": None, "Max_yPlus": None,
        "Converged": False,
        "Job_dir": job_dir,
    }

    # ── forceCoeffs ──────────────────────────────────────────────────────────
    fc_pattern = f"{job_dir}/postProcessing/forceCoeffs/*/forceCoeffs.dat"
    fc_files   = sorted(glob.glob(fc_pattern))

    if not fc_files:
        log(f"forceCoeffs.dat not found at {fc_pattern}", "WARN")
        return results

    fc_file = fc_files[-1]  # Use the latest startTime subdirectory
    log(f"Parsing force coefficients: {fc_file}")

    times, CL_vals, CD_vals, CM_vals = [], [], [], []
    with open(fc_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                # OpenFOAM forceCoeffs columns: Time  Cm  Cd  Cl  [Cl(f)  Cl(r)]
                t  = float(parts[0])
                Cm = float(parts[1])
                Cd = float(parts[2])
                Cl = float(parts[3])
                times.append(t)
                CM_vals.append(Cm)
                CD_vals.append(Cd)
                CL_vals.append(Cl)
            except ValueError:
                continue

    if not times:
        log("Could not parse any force coefficient data", "WARN")
        return results

    times   = np.array(times)
    CL_vals = np.array(CL_vals)
    CD_vals = np.array(CD_vals)
    CM_vals = np.array(CM_vals)

    # Average over the last AVERAGING_FRAC of the simulation
    t_start = times[-1] - AVERAGING_FRAC * (times[-1] - times[0])
    mask    = times >= t_start

    CL_mean = float(np.mean(CL_vals[mask]))
    CD_mean = float(np.mean(CD_vals[mask]))
    CM_mean = float(np.mean(CM_vals[mask]))
    CL_std  = float(np.std(CL_vals[mask]))
    CD_std  = float(np.std(CD_vals[mask]))

    results.update({
        "CL":     round(CL_mean, 6),
        "CD":     round(CD_mean, 6),
        "CM":     round(CM_mean, 6),
        "LD_ratio": round(CL_mean / CD_mean, 4) if abs(CD_mean) > 1e-10 else None,
        "CL_std": round(CL_std,  6),
        "CD_std": round(CD_std,  6),
    })

    # Convergence check: std / |mean| < tolerance
    if CONVERGENCE_TOL is not None:
        cl_ok = (abs(CL_mean) < 1e-6) or (CL_std / abs(CL_mean) < CONVERGENCE_TOL)
        cd_ok = (abs(CD_mean) < 1e-6) or (CD_std / abs(CD_mean) < CONVERGENCE_TOL)
        results["Converged"] = bool(cl_ok and cd_ok)
        if not results["Converged"]:
            log(f"Convergence NOT met: CL_std/CL={CL_std/max(abs(CL_mean),1e-9):.3%}  "
                f"CD_std/CD={CD_std/max(abs(CD_mean),1e-9):.3%}", "WARN")
    else:
        results["Converged"] = True  # Not checked

    # ── y+ ────────────────────────────────────────────────────────────────────
    yplus_pattern = f"{job_dir}/postProcessing/yPlus/*/yPlus.dat"
    yplus_files   = sorted(glob.glob(yplus_pattern))

    if yplus_files:
        yplus_file = yplus_files[-1]
        log(f"Parsing y+: {yplus_file}")
        yp_vals_avg, yp_vals_max = [], []
        with open(yplus_file) as f:
            for line in f:
                if line.strip().startswith("#") or not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        yp_vals_avg.append(float(parts[1]))   # mean y+
                        yp_vals_max.append(float(parts[2]))   # max y+
                    except ValueError:
                        pass
        if yp_vals_avg:
            results["Avg_yPlus"] = round(float(np.mean(yp_vals_avg[-5:])), 2)
            results["Max_yPlus"] = round(float(np.max(yp_vals_max[-5:])),  2)
    else:
        log("yPlus.dat not found — skipping y+ extraction", "WARN")

    return results


def print_result_table(results: dict):
    """Pretty-print the result for a single alpha."""
    conv = "✓" if results.get("Converged") else "✗"
    ld   = results.get("LD_ratio")
    print()
    print(f"  ┌── Results: α={results['Alpha_deg']}° ──────────────────────────────┐")
    print(f"  │  CL      = {results['CL']:>12.6f}  ± {results['CL_std']:.2e}      │")
    print(f"  │  CD      = {results['CD']:>12.6f}  ± {results['CD_std']:.2e}      │")
    print(f"  │  CM      = {results['CM']:>12.6f}                          │")
    print(f"  │  L/D     = {ld if ld is not None else 'N/A':>12}                          │")
    print(f"  │  y+_avg  = {results['Avg_yPlus']:>12}                          │")
    print(f"  │  y+_max  = {results['Max_yPlus']:>12}                          │")
    print(f"  │  Converged: {conv}                                        │")
    print(f"  └──────────────────────────────────────────────────────────┘")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — POST-PROCESSING (ParaView)
# ══════════════════════════════════════════════════════════════════════════════

def postprocess(job_dir: str, job_id: str) -> bool:
    """
    Run optional ParaView batch script (post_process.py must exist alongside).
    Generates: Mach contours, pressure/Cp surface, shock cone isosurface.
    """
    pv_script = "post_process.py"
    if not os.path.exists(pv_script):
        log("post_process.py not found — skipping ParaView rendering", "WARN")
        return True

    images_dir = f"{job_dir}/images"
    os.makedirs(images_dir, exist_ok=True)

    # Create a .foam file so paraFoam / pvbatch can open the case
    foam_file = f"{job_dir}/{job_id}.foam"
    Path(foam_file).touch()

    cmd = (
        f"LIBGL_ALWAYS_SOFTWARE=1 pvbatch --force-offscreen-rendering "
        f"{pv_script} {foam_file} {images_dir}"
    )
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        log(f"ParaView failed:\n{result.stderr}", "WARN")
        return False

    images = glob.glob(f"{images_dir}/*.png")
    log(f"Generated {len(images)} image(s) in {images_dir}")
    return len(images) > 0


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6 — CLEANUP
# ══════════════════════════════════════════════════════════════════════════════

def cleanup(job_dir: str):
    """Remove heavy mesh and processor files to save disk space."""
    targets = [
        f"{job_dir}/constant/polyMesh",
        f"{job_dir}/constant/extendedFeatureEdgeMesh",
        f"{job_dir}/processor*",
        f"{job_dir}/PyFoam*",
        "PyFoam*",
    ]
    for pattern in targets:
        for path in glob.glob(pattern):
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            elif os.path.isfile(path):
                os.remove(path)
    log("Cleanup complete")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Supersonic UAV rhoCentralFoam sweep driver"
    )
    parser.add_argument("--alpha", nargs="+", type=float,
                        help="Override AoA sweep angles in degrees")
    parser.add_argument("--mesh-only",  action="store_true",
                        help="Run mesh generation only, skip solver")
    parser.add_argument("--solve-only", action="store_true",
                        help="Skip meshing (mesh must already exist)")
    parser.add_argument("--np", type=int, default=NP,
                        help=f"MPI process count (default: {NP})")
    parser.add_argument("--end-time", type=float, default=END_TIME,
                        help=f"Simulation end time in seconds (default: {END_TIME})")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Skip post-run cleanup of heavy files")
    parser.add_argument("--no-paraview", action="store_true",
                        help="Skip ParaView rendering step")
    args = parser.parse_args()

    alphas    = args.alpha if args.alpha else ALPHAS
    np_cores  = args.np
    end_time  = args.end_time

    # ── Pre-flight checks ─────────────────────────────────────────────────────
    if not os.path.exists(GEOMETRY_STL):
        log(f"Geometry file '{GEOMETRY_STL}' not found in working directory.", "ERROR")
        log("Place your watertight UAV STL here before running.", "ERROR")
        raise SystemExit(1)

    if not os.path.isdir(CASE_TEMPLATE):
        log(f"Case template directory '{CASE_TEMPLATE}' not found.", "ERROR")
        raise SystemExit(1)

    # ── Compute ISA atmosphere ────────────────────────────────────────────────
    atm   = isa_atmosphere(ALTITUDE_M)
    u_inf = MACH * atm["a"]

    print()
    print("═" * 60)
    print(f"  Supersonic UAV Sweep Driver")
    print(f"  Mach {MACH}  |  {ALTITUDE_M/1000:.0f} km ISA  |  {u_inf:.1f} m/s")
    print(f"  T∞={atm['T']} K  p∞={atm['p']} Pa  ρ∞={atm['rho']:.4f} kg/m³")
    print(f"  Re_chord ≈ {u_inf * REF_CHORD / atm['nu']:.2e}")
    print(f"  AoA sweep: {alphas}°")
    print(f"  Cores: {np_cores}  |  End time: {end_time} s")
    print("═" * 60)
    print()

    initialize_results_csv()

    # ── Main sweep loop ───────────────────────────────────────────────────────
    for alpha in alphas:
        job_id  = f"run_alpha_{alpha:+.1f}".replace("+", "p").replace("-", "m").replace(".", "d")
        job_dir = f"./{job_id}"

        print()
        print("━" * 60)
        print(f"  Job: {job_id}  |  α = {alpha}°")
        print("━" * 60)

        # ── 1. Prepare ────────────────────────────────────────────────────────
        log("[1/5] Preparing case directory...")
        if not prepare(job_dir, alpha, atm, u_inf, np_cores, end_time):
            log(f"Preparation failed — skipping {job_id}", "ERROR")
            continue

        # ── 2. Mesh ───────────────────────────────────────────────────────────
        if not args.solve_only:
            log("[2/5] Generating mesh (snappyHexMesh)...")
            try:
                if not mesh(job_dir, alpha, np_cores):
                    log(f"Meshing failed — skipping {job_id}", "ERROR")
                    continue
            except Exception as e:
                log(f"Mesh exception: {e}", "ERROR")
                continue
        else:
            log("[2/5] Skipping mesh (--solve-only)")

        if args.mesh_only:
            log("Mesh-only mode — stopping after mesh.")
            continue

        # ── 3. Solve ──────────────────────────────────────────────────────────
        log("[3/5] Running rhoCentralFoam...")
        try:
            if not solve(job_dir, np_cores, end_time):
                log(f"Solver failed — skipping {job_id}", "ERROR")
                continue
        except Exception as e:
            log(f"Solver exception: {e}", "ERROR")
            continue

        # ── 4. Extract results ────────────────────────────────────────────────
        log("[4/5] Extracting force coefficients...")
        results = extract_results(job_dir, alpha, atm, u_inf)
        print_result_table(results)
        append_result(results)

        # ── 5. Post-process (ParaView) ────────────────────────────────────────
        if not args.no_paraview:
            log("[5a/5] Running ParaView rendering...")
            try:
                postprocess(job_dir, job_id)
            except Exception as e:
                log(f"ParaView exception: {e}", "WARN")

        # ── 6. Cleanup ────────────────────────────────────────────────────────
        if not args.no_cleanup:
            log("[5b/5] Cleaning up heavy files...")
            cleanup(job_dir)

        log(f"Completed {job_id} ✓")

    # ── Final summary ─────────────────────────────────────────────────────────
    print()
    print("═" * 60)
    print(f"  Sweep complete. Results: {RESULTS_CSV}")
    print("═" * 60)
    _print_summary_table()
    print()


def _print_summary_table():
    """Print a compact summary of all results from the CSV."""
    if not os.path.exists(RESULTS_CSV):
        return
    with open(RESULTS_CSV) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return

    print(f"\n  {'α':>6}  {'CL':>10}  {'CD':>10}  {'L/D':>8}  {'CM':>10}  {'y+_avg':>8}  {'Conv':>5}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*10}  {'─'*8}  {'─'*5}")
    for r in rows:
        conv = "✓" if r.get("Converged", "").lower() == "true" else "✗"
        print(f"  {float(r['Alpha_deg']):>5.1f}°"
              f"  {r['CL']:>10}"
              f"  {r['CD']:>10}"
              f"  {r['LD_ratio']:>8}"
              f"  {r['CM']:>10}"
              f"  {r['Avg_yPlus']:>8}"
              f"  {conv:>5}")
    print()

if __name__ == "__main__":
    main()