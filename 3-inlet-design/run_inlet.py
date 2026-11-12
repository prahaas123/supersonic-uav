import os
import re
import sys
import json
import math
import time
import shutil
import subprocess

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR     = os.path.join(SCRIPT_DIR, "..", "0-tools")
CASE_TEMPLATE = os.path.join(SCRIPT_DIR, "case_template_inlet")

# Flight condition
ALTITUDE_M     = 15000.0     # [m]
MACH           = 1.5
PT_RECOVERY_2D = 0.94        # total pressure recovery of the 2D external compression (2d-inlet-designs.csv)
MDOT_TARGET    = 0.2635      # RDE air mass flow [kg/s] (README, section 1)
ETA_DUCT_EST   = 0.96        # assumed S-duct p0 recovery, only sets the outlet p (first baseline: 0.959)

# Run settings
NP             = 50          # MPI processes
N_ITER         = 8000        # rhoSimpleFoam iterations (residualControl may stop earlier)
WRITE_INTERVAL = 500         # [iterations] periodic writes
BASE_CELL      = 0.006       # background cell [m], refined to level 2 (1.5 mm) inside the duct
TARGET_YPLUS   = 40.0        # cell-centre y+ from the 1D estimate; the baseline run puts the wall at ~30-100
YPLUS_RANGE    = (30.0, 100.0)  # wall-function range, average y+ outside it is flagged
EXT_LENGTH_D   = 1.5         # straight extension after the engine face [engine-face diameters]
N_EXT          = 13          # axial stations on the extension
TURB_INTENSITY = 0.05        # at the capture plane, behind the shock system
RAMP_ITERS     = 1500        # inlet mass flow ramps from 20 % to MDOT_TARGET over these iterations (fixed schedule)

# Convergence (iterative) checks, evaluated over the averaging window
MASS_IMBALANCE_TOL_PCT = 0.5 # |inlet - outlet| mass flow
MDOT_OSC_TOL_PCT       = 0.2 # outlet mass flow half peak-to-peak / mean (the inlet one is imposed)
P_CAP_OSC_TOL_PCT      = 0.2 # capture-plane static pressure half peak-to-peak / mean
MDOT_TARGET_TOL_PCT    = 0.5 # |mean outlet mass flow / MDOT_TARGET - 1|
CONT_ERR_TOL           = 1e-3  # mean "sum local" continuity error over the window
# Physical checks, evaluated on the averaged fields
T0_MARGIN_K            = 1.0 # duct T max may not exceed T0 by more than this (adiabatic duct)
P0_CAP_DEV_WARN_PCT    = 5.0 # warn when the capture p0 is this far from p0_capture (assumed ETA_DUCT_EST)

GAMMA = 1.4
R_GAS = 287.058

# S-duct design
INLET_PARAMS = dict(
    LD=5.5,       # duct length / engine-face diameter
    AR=1.5,       # engine-face area / capture area
    WH=2.5,       # capture width / height
    c_cl=0.5,     # centreline S-curve inflection
    c_area=0.5,   # area-distribution inflection
)

# Post processing
VIEW_SIZE  = [3840, 1440]
Y_SLICE    = 1e-4            # side slice just off the y = 0 symmetry plane (avoids cell faces)
X_OFFSET   = 1e-4            # measurement planes just downstream of the capture plane / engine face
SCALAR_BAR = {"position": [0.90, 0.05], "length": 0.90, "thickness": 48,
              "title_size": 48, "label_size": 40, "n_labels": 6}

def main():
    # usage: python3 run_inlet.py [case_dir]
    job_directory = sys.argv[1] if len(sys.argv) > 1 else "./inlet_baseline"
    results = design_point(INLET_PARAMS, job_directory)
    if results is None:
        print("Run did not produce metrics.")
        return 1
    return 0 if results.get("converged") else 2

def design_point(params, job_directory):
    job_id = os.path.basename(os.path.normpath(job_directory))
    print(f"\n{'='*40}")
    print(f"Starting Simulation: {job_id}")
    print(f"  {params}")
    print(f"{'='*40}")

    # 1. Prepare
    print("[1/6] Preparing case, S-duct STL and boundary conditions...")
    refs = prepare(job_directory, params)
    if refs is None or "status" in refs:
        status = refs["status"] if refs else "prepare_failed"
        print(f"Error: {status} for {job_id}.")
        return write_failure(job_directory, status, refs)

    # 2. Mesh
    print("[2/6] Generating mesh (snappyHexMesh)...")
    t_start = time.perf_counter()
    if not mesh(job_directory):
        return write_failure(job_directory, "mesh_failed", refs)
    t_meshed = time.perf_counter()

    # 3. Solve
    print("[3/6] Solving (rhoSimpleFoam)...")
    if not solve(job_directory):
        return write_failure(job_directory, "solve_failed", refs)
    t_solved = time.perf_counter()

    timings = {
        "mesh_time_s":  round(t_meshed - t_start, 1),
        "solve_time_s": round(t_solved - t_meshed, 1),
        "total_time_s": round(t_solved - t_start, 1),
    }

    # 4. Mass flow and convergence
    print("[4/6] Extracting mass flow and convergence...")
    results = convergence_metrics(job_directory, {**refs, **timings})

    # 5. ParaView: flow metrics merged into metrics.json, side slices
    print("[5/6] Post processing in ParaView...")
    post_process(job_directory)
    plot_convergence(job_directory)
    results = physical_checks(job_directory, read_json(job_directory) or results)
    print(format_summary(results))

    if not results.get("converged"):
        print(f"WARNING: {job_id} failed {results.get('failed_checks')}. "
              f"Treat this result as unusable.")

    # 6. Clean up
    print("[6/6] Cleaning up processor files...")
    cleanup(job_directory)
    print(f"Successfully completed {job_id}!")
    return results

# ----------------------------------------------------------------------------
# Flow conditions
# ----------------------------------------------------------------------------

def isa_atmosphere(altitude_m):
    g0 = 9.80665;  R = R_GAS
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
    a   = math.sqrt(GAMMA * R * T)
    return dict(T=round(T, 4), p=round(p, 2), rho=round(rho, 6), a=round(a, 4))

def sutherland(T):
    return 1.458e-6 * T**1.5 / (T + 110.4)

def isentropic_ratio(M):
    # T0/T, p0/p = ratio**(gamma/(gamma-1))
    return 1.0 + 0.5 * (GAMMA - 1.0) * M * M

def mass_flow(M, area, p0, T0):
    expo = -(GAMMA + 1.0) / (2.0 * (GAMMA - 1.0))
    return area * p0 * math.sqrt(GAMMA / (R_GAS * T0)) * M * isentropic_ratio(M) ** expo

def subsonic_mach(mdot, area, p0, T0):
    # None when mdot is above the choked mass flow of the section
    if mdot >= mass_flow(1.0, area, p0, T0):
        return None
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if mass_flow(mid, area, p0, T0) < mdot else (lo, mid)
    return 0.5 * (lo + hi)

def flow_conditions(params):
    import build_inlet as bi

    atm   = isa_atmosphere(ALTITUDE_M)
    pexp  = GAMMA / (GAMMA - 1.0)
    p0inf = atm["p"] * isentropic_ratio(MACH) ** pexp
    T0    = atm["T"] * isentropic_ratio(MACH)
    p0c   = PT_RECOVERY_2D * p0inf

    a_ef  = math.pi / 4.0 * bi.D_EF ** 2
    a_cap = a_ef / params["AR"]          # ellipse, W0 * H0 = D_EF^2 / AR

    # Engine face: target mass flow at the capture p0 less an assumed duct loss. The outlet p is
    # fixed to this static pressure; the capture p0 then comes out of the solution near p0c.
    p0ef = ETA_DUCT_EST * p0c
    m_ef = subsonic_mach(MDOT_TARGET, a_ef, p0ef, T0)
    if m_ef is None:
        raise ValueError("MDOT_TARGET is above the choked mass flow of the engine face")
    m_cap = subsonic_mach(MDOT_TARGET, a_cap, p0c, T0)

    p_back = p0ef / isentropic_ratio(m_ef) ** pexp
    T_ef   = T0 / isentropic_ratio(m_ef)
    U_ef   = m_ef * math.sqrt(GAMMA * R_GAS * T_ef)
    rho_ef = p_back / (R_GAS * T_ef)

    # Inlet turbulence: intensity and mixing length from the capture equivalent diameter
    U_cap = U_ef if m_cap is None else \
        m_cap * math.sqrt(GAMMA * R_GAS * T0 / isentropic_ratio(m_cap))
    l_mix = 0.07 * bi.D_EF / math.sqrt(params["AR"])
    k     = 1.5 * (TURB_INTENSITY * U_cap) ** 2
    omega = k ** 0.5 / (0.09 ** 0.25 * l_mix)

    return dict(
        p_inf_Pa=atm["p"], T_inf_K=atm["T"],
        p0_inf_Pa=p0inf, T0_K=T0, p0_capture_Pa=p0c, p_back_Pa=p_back,
        mdot_target_kg_s=MDOT_TARGET, pt_recovery_2d=PT_RECOVERY_2D, eta_duct_est=ETA_DUCT_EST,
        capture_area_m2=a_cap, engine_face_area_m2=a_ef,
        # isentropic 1D estimates at the target mass flow (capture None = choked)
        mach_capture_1d=m_cap, mach_engine_face_1d=m_ef,
        T_ef_1d_K=T_ef, U_ef_1d=U_ef, rho_ef_1d=rho_ef,
        k_init=k, omega_init=omega, mixing_length_m=l_mix,
    )

def first_layer_thickness(rho, U, T, length, y_plus):
    nu    = sutherland(T) / rho
    re_l  = U * length / nu
    cf    = 0.026 / re_l ** (1.0 / 7.0)      # turbulent flat plate skin friction
    u_tau = U * math.sqrt(0.5 * cf)
    # OpenFOAM evaluates y+ at the cell centre, half the first-layer height
    return 2.0 * y_plus * nu / u_tau

# ----------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------

def _write_solid(fh, name, tris):
    import numpy as np
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    fh.write(f"solid {name}\n")
    for nrm, tri in zip(n, tris):
        fh.write(f" facet normal {nrm[0]:.6e} {nrm[1]:.6e} {nrm[2]:.6e}\n  outer loop\n")
        for v in tri:
            fh.write(f"   vertex {v[0]:.9e} {v[1]:.9e} {v[2]:.9e}\n")
        fh.write("  endloop\n endfacet\n")
    fh.write(f"endsolid {name}\n")

def add_extension(duct_stl, out_stl, length):
    # Replace the engine-face cap with a straight duct of the engine-face section and a new outlet cap
    import numpy as np

    with open(duct_stl) as fh:
        text = fh.read()
    solids = dict(re.findall(r"solid (\w+)\n(.*?)endsolid \1\n", text, flags=re.S))

    v = np.array(re.findall(r"vertex (\S+) (\S+) (\S+)", solids["outlet"]), dtype=float)
    v = np.unique(v, axis=0)
    x_ef = float(v[0, 0])
    i_c = np.argmin([np.sum((p - v.mean(axis=0)) ** 2) for p in v])
    centre, ring = v[i_c], np.delete(v, i_c, axis=0)
    # Counter-clockwise about +x, same order as build_inlet
    ang = np.mod(np.arctan2(ring[:, 2] - centre[2], ring[:, 1] - centre[1]), 2 * np.pi)
    ring = ring[np.argsort(ang)]

    xs = x_ef + length * np.linspace(0.0, 1.0, N_EXT)
    P = np.repeat(ring[None, :, :], N_EXT, axis=0)
    P[..., 0] = xs[:, None]

    nc = len(ring)
    i = np.arange(N_EXT - 1)[:, None]
    j = np.arange(nc)[None, :]
    jn = (j + 1) % nc
    a, b, c, d = P[i, j], P[i + 1, j], P[i + 1, jn], P[i, jn]
    wall = np.concatenate([np.stack([a, d, c], -2).reshape(-1, 3, 3),
                           np.stack([a, c, b], -2).reshape(-1, 3, 3)])

    end = P[-1]
    C = np.broadcast_to(np.array([xs[-1], centre[1], centre[2]]), end.shape)
    cap = np.stack([C, end, np.roll(end, -1, axis=0)], 1)      # normal +x, outward

    with open(out_stl, "w") as fh:
        for name in ("inlet", "wall"):
            fh.write(f"solid {name}\n{solids[name]}endsolid {name}\n")
        _write_solid(fh, "extension", wall)
        _write_solid(fh, "outlet", cap)

    pts = np.array(re.findall(r"vertex (\S+) (\S+) (\S+)", open(out_stl).read()), dtype=float)
    return x_ef, float(xs[-1]), pts.min(axis=0), pts.max(axis=0)

# ----------------------------------------------------------------------------
# Case setup
# ----------------------------------------------------------------------------

def _sub(path, pattern, repl, flags=re.M):
    with open(path) as fh:
        text = fh.read()
    text, n = re.subn(pattern, repl, text, flags=flags)
    if n == 0:
        raise ValueError(f"pattern {pattern!r} not found in {path}")
    with open(path, "w") as fh:
        fh.write(text)

def prepare(job_directory, params):
    sys.path.insert(0, SCRIPT_DIR)
    import numpy as np
    import build_inlet as bi

    try:
        if os.path.exists(job_directory):
            shutil.rmtree(job_directory)
        shutil.copytree(CASE_TEMPLATE, job_directory, ignore=shutil.ignore_patterns("*.foam"))
        job_id = os.path.basename(os.path.normpath(job_directory))
        open(os.path.join(job_directory, f"{job_id}.foam"), "w").close()

        # Geometry: S-duct from build_inlet, then the straight extension
        tri_dir = os.path.join(job_directory, "constant", "triSurface")
        os.makedirs(tri_dir, exist_ok=True)
        raw = os.path.join(tri_dir, "sduct_raw.stl")
        try:
            bi.build_inlet(out_path=raw, **params)
        except (bi.GeometryError, ValueError) as e:
            print(f"  geometry rejected: {e}")
            return {"status": "geometry_failed", **params}
        L = params["LD"] * bi.D_EF
        x_ef, x_out, lo, hi = add_extension(raw, os.path.join(tri_dir, "duct.stl"),
                                            EXT_LENGTH_D * bi.D_EF)
        os.remove(raw)

        flow = flow_conditions(params)
        if flow["mach_capture_1d"] is None:
            print("  WARNING: target mass flow is above the choked flow of the capture area "
                  "(isentropic), the duct cannot pass MDOT_TARGET at this recovery.")

        # Background block: duct bounding box padded by 1.5 cells
        pad = 1.5 * BASE_CELL
        lo, hi = lo - pad, hi + pad
        n = np.maximum(np.ceil((hi - lo) / BASE_CELL).astype(int), 1)
        hi = lo + n * BASE_CELL
        verts = [(lo[0], lo[1], lo[2]), (hi[0], lo[1], lo[2]), (hi[0], hi[1], lo[2]), (lo[0], hi[1], lo[2]),
                 (lo[0], lo[1], hi[2]), (hi[0], lo[1], hi[2]), (hi[0], hi[1], hi[2]), (lo[0], hi[1], hi[2])]
        vtext = "vertices\n(\n" + "".join(
            f"    ({x:.6f} {y:.6f} {z:.6f})   // {k}\n" for k, (x, y, z) in enumerate(verts)) + ");"
        bmd = os.path.join(job_directory, "system", "blockMeshDict")
        _sub(bmd, r"^vertices\s*\(.*?^\);", vtext.replace("\\", "\\\\"), flags=re.M | re.S)
        _sub(bmd, r"hex \(0 1 2 3 4 5 6 7\) \(\d+ \d+ \d+\)",
             f"hex (0 1 2 3 4 5 6 7) ({n[0]} {n[1]} {n[2]})")

        # snappyHexMesh: point inside the duct (mid-length on the centreline), first layer
        zc, _, _ = bi._section(bi.X_LIP + 0.5 * L, L, 0, 0, params["c_cl"], params["c_area"])
        loc = (bi.X_LIP + 0.5 * L + 1.234e-4, 1.234e-4, float(zc) + 1.234e-4)
        y1 = first_layer_thickness(flow["rho_ef_1d"], flow["U_ef_1d"], flow["T_ef_1d_K"], L, TARGET_YPLUS)
        shm = os.path.join(job_directory, "system", "snappyHexMeshDict")
        _sub(shm, r"^(\s*)locationInMesh\s+[^;\n]+;",
             rf"\g<1>locationInMesh ({loc[0]:.6f} {loc[1]:.6f} {loc[2]:.6f});")
        _sub(shm, r"^(\s*)firstLayerThickness\s+[^;\n]+;", rf"\g<1>firstLayerThickness {y1:.4g};")
        _sub(shm, r"^(\s*)minThickness\s+[^;\n]+;", rf"\g<1>minThickness {0.5 * y1:.4g};")
        print(f"  {n[0]}x{n[1]}x{n[2]} background cells, first layer {y1*1e6:.1f} um "
              f"for y+ = {TARGET_YPLUS:g}")

        # decomposeParDict, controlDict
        _sub(os.path.join(job_directory, "system", "decomposeParDict"),
             r"^numberOfSubdomains\s+\d+;", f"numberOfSubdomains  {NP};")
        cd = os.path.join(job_directory, "system", "controlDict")
        _sub(cd, r"^endTime\s+[^;\n]+;", f"endTime         {N_ITER};")
        _sub(cd, r"^writeInterval\s+[^;\n]+;", f"writeInterval   {WRITE_INTERVAL};")

        # fieldAverage over the same window the convergence check uses
        sys.path.insert(0, TOOLS_DIR)
        import metrics
        avg_start = int(round(1 + (1.0 - metrics.TAIL_FRACTION) * (N_ITER - 1)))
        if RAMP_ITERS >= avg_start:
            raise ValueError("RAMP_ITERS must end before the averaging window")
        _sub(cd, r"^(\s*)timeStart\s+[^;\n]+;", rf"\g<1>timeStart       {avg_start};")

        # Inlet mass flow: fixed schedule, 20 % of the target ramped to the target, then held
        _sub(os.path.join(job_directory, "0", "U"), r"^(\s*)massFlowRate\s+[^;\n]+;",
             rf"\g<1>massFlowRate    table ((0 {0.2 * MDOT_TARGET:.6g}) ({RAMP_ITERS} {MDOT_TARGET:.6g}));")

        # Boundary / initial values
        with open(os.path.join(job_directory, "0", "include", "initialConditions"), "w") as f:
            f.write(f"p0In        {flow['p0_capture_Pa']:.6g};\n")
            f.write(f"T0In        {flow['T0_K']:.6g};\n")
            f.write(f"pOut        {flow['p_back_Pa']:.6g};\n")
            f.write(f"TInit       {flow['T_ef_1d_K']:.6g};\n")
            f.write(f"UInit       ({flow['U_ef_1d']:.6g} 0 0);\n")
            f.write(f"kInit       {flow['k_init']:.6g};\n")
            f.write(f"omegaInit   {flow['omega_init']:.6g};\n")
            f.write(f"turbIntensity {TURB_INTENSITY:.6g};\n")
            f.write(f"mixingLength  {flow['mixing_length_m']:.6g};\n")

        print(f"  inlet mass flow {MDOT_TARGET:g} kg/s (ramped over {RAMP_ITERS} it) | "
              f"fields averaged from iteration {avg_start}")
        print(f"  p0 capture (expected) {flow['p0_capture_Pa']:.0f} Pa | T0 {flow['T0_K']:.2f} K | "
              f"p back (fixed, eta_duct {ETA_DUCT_EST:g}) {flow['p_back_Pa']:.0f} Pa | 1D Mach capture "
              f"{flow['mach_capture_1d'] if flow['mach_capture_1d'] is None else round(flow['mach_capture_1d'], 3)}"
              f" / engine face {flow['mach_engine_face_1d']:.3f}")

        return {**params, **flow,
                "duct_length_m": L, "x_lip_m": bi.X_LIP, "x_engine_face_m": x_ef,
                "x_outlet_m": x_out, "extension_length_m": x_out - x_ef,
                "first_layer_m": y1, "target_yplus": TARGET_YPLUS, "base_cell_m": BASE_CELL,
                "ramp_iters": RAMP_ITERS, "avg_start_iter": avg_start}
    except Exception as e:
        print(f"Exception during preparation: {e}")
        return None

# ----------------------------------------------------------------------------
# OpenFOAM
# ----------------------------------------------------------------------------

def run_steps(job_directory, steps):
    for app, cmd in steps:
        log = os.path.join(job_directory, f"log.{app}")
        print(f"  -> {cmd}   (log.{app})")
        rc = subprocess.run(f"{cmd} > {log} 2>&1", shell=True, executable="/bin/bash").returncode
        if rc != 0:
            print(f"  {app} failed (exit {rc}), last lines of log.{app}:")
            subprocess.run(f"tail -n 30 {log}", shell=True, executable="/bin/bash")
            return False
    return True

def mesh(job_directory):
    c = job_directory
    ok = run_steps(c, [
        ("surfaceFeatureExtract", f"surfaceFeatureExtract -case {c}"),
        ("blockMesh",             f"blockMesh -case {c}"),
        ("decomposePar.mesh",     f"decomposePar -force -case {c}"),
        ("snappyHexMesh",         f"mpirun -np {NP} snappyHexMesh -parallel -overwrite -case {c}"),
        ("reconstructParMesh",    f"reconstructParMesh -constant -case {c}"),
    ])
    subprocess.run(f"rm -rf {c}/processor*", shell=True, executable="/bin/bash")
    if ok:
        # Mesh quality and layer report, kept for reference only
        subprocess.run(f"checkMesh -constant -case {c} > {c}/log.checkMesh 2>&1",
                       shell=True, executable="/bin/bash")
    return ok and os.path.isdir(f"{c}/constant/polyMesh")

def solve(job_directory):
    c = job_directory
    ok = run_steps(c, [
        ("decomposePar",   f"decomposePar -force -case {c}"),
        ("rhoSimpleFoam",  f"mpirun -np {NP} rhoSimpleFoam -parallel -case {c}"),
        ("reconstructPar", f"reconstructPar -latestTime -case {c}"),
    ])
    subprocess.run(f"rm -rf {c}/processor*", shell=True, executable="/bin/bash")
    time_dirs = [d for d in os.listdir(c) if os.path.isdir(os.path.join(c, d))
                 and _is_float(d) and float(d) > 0]
    return ok and len(time_dirs) > 0

def _is_float(s):
    try:
        float(s); return True
    except ValueError:
        return False

def cleanup(job_directory):
    subprocess.run(f"rm -rf {job_directory}/processor*", shell=True, executable="/bin/bash")

# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

def read_json(job_directory):
    try:
        with open(os.path.join(job_directory, "metrics.json")) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None

def write_json(job_directory, data):
    os.makedirs(job_directory, exist_ok=True)
    with open(os.path.join(job_directory, "metrics.json"), "w") as fh:
        json.dump(data, fh, indent=2)

def write_failure(job_directory, status, refs=None):
    refs = {k: v for k, v in (refs or {}).items() if k != "status"}
    write_json(job_directory, {"case": os.path.abspath(job_directory), "status": status,
                               "converged": False, **refs})
    cleanup(job_directory)
    return None

def convergence_metrics(job_directory, extra):
    sys.path.insert(0, TOOLS_DIR)
    import metrics

    m = {"case": os.path.abspath(job_directory), "status": "ok", "converged": False,
         "tail_fraction": metrics.TAIL_FRACTION,
         "drift_tolerance_pct": metrics.DRIFT_TOLERANCE_PCT,
         "mass_imbalance_tol_pct": MASS_IMBALANCE_TOL_PCT,
         "mdot_osc_tol_pct": MDOT_OSC_TOL_PCT,
         "p_cap_osc_tol_pct": P_CAP_OSC_TOL_PCT,
         "mdot_target_tol_pct": MDOT_TARGET_TOL_PCT,
         "cont_err_tol": CONT_ERR_TOL, **extra}

    to, out = metrics._read_dat(job_directory, "massFlowOutlet", "surfaceFieldValue.dat", 2)
    ti, inn = metrics._read_dat(job_directory, "massFlowInlet", "surfaceFieldValue.dat", 2)
    if to is None or ti is None:
        m["status"] = "no_data"
        write_json(job_directory, m)
        return m

    so = metrics.window_stats(to, out[:, 0])
    si = metrics.window_stats(ti, -inn[:, 0])          # inflow phi is negative
    m["iterations"] = int(to[-1])
    if so is None or si is None or so["n_samples"] < metrics.MIN_TAIL_SAMPLES:
        m["status"] = "too_few_samples"
        write_json(job_directory, m)
        return m

    # The inlet mass flow is imposed, so the outlet mass flow and the capture pressure are the
    # free (convergence) signals
    m["mdot_outlet_kg_s"]   = so["value"]
    m["mdot_inlet_kg_s"]    = si["value"]
    m["mdot_drift_pct"]     = so["drift_pct"]
    m["mdot_osc_pct"]       = 50.0 * (so["max"] - so["min"]) / so["value"]    # half peak-to-peak
    m["mdot_std_pct"]       = 100.0 * so["std"] / so["value"]
    m["mass_imbalance_pct"] = 100.0 * (si["value"] - so["value"]) / so["value"]
    m["mdot_ratio"]         = so["value"] / MDOT_TARGET
    m["avg_window_start"]   = so["window_start"]
    m["avg_window_end"]     = so["window_end"]

    tp, pc = metrics._read_dat(job_directory, "inletPressure", "surfaceFieldValue.dat", 2)
    sp = metrics.window_stats(tp, pc[:, 0]) if tp is not None else None
    if sp:
        m["p_cap_static_avg_Pa"] = sp["value"]
        m["p_cap_osc_pct"]       = 50.0 * (sp["max"] - sp["min"]) / sp["value"]
        m["p_cap_drift_pct"]     = sp["drift_pct"]

    m["cont_err_avg"] = continuity_error(job_directory, so["window_start"])

    m.update(metrics.read_yplus(job_directory, patch="wall"))
    ya = m.get("yplus_avg")
    m["yplus_in_range"] = ya is not None and YPLUS_RANGE[0] <= ya <= YPLUS_RANGE[1]

    drift, imb, osc = m["mdot_drift_pct"], m["mass_imbalance_pct"], m["mdot_osc_pct"]
    pdrift, posc, cerr = m.get("p_cap_drift_pct"), m.get("p_cap_osc_pct"), m["cont_err_avg"]
    checks = {
        "mdot_drift":       drift is not None and abs(drift) < metrics.DRIFT_TOLERANCE_PCT,
        "mdot_oscillation": osc < MDOT_OSC_TOL_PCT,
        "p_cap_drift":      pdrift is not None and abs(pdrift) < metrics.DRIFT_TOLERANCE_PCT,
        "p_cap_oscillation": posc is not None and posc < P_CAP_OSC_TOL_PCT,
        "imbalance":        abs(imb) < MASS_IMBALANCE_TOL_PCT,
        "mdot_target":      abs(m["mdot_ratio"] - 1.0) * 100.0 < MDOT_TARGET_TOL_PCT,
        "continuity":       cerr is not None and cerr < CONT_ERR_TOL,
    }
    m["failed_checks"] = [k for k, ok in checks.items() if not ok]
    m["converged"] = not m["failed_checks"]
    m["status"] = "ok" if m["converged"] else "not_converged"
    if not m["yplus_in_range"]:
        print(f"  WARNING: average y+ {ya} is outside {YPLUS_RANGE}, adjust TARGET_YPLUS")
    write_json(job_directory, m)
    return m

def continuity_error(job_directory, t_start):
    # Mean "sum local" continuity error from log.rhoSimpleFoam over iterations >= t_start
    path = os.path.join(job_directory, "log.rhoSimpleFoam")
    if not os.path.exists(path):
        return None
    vals, t = [], None
    with open(path) as fh:
        for line in fh:
            if line.startswith("Time = "):
                try:
                    t = float(line.split()[2])
                except (IndexError, ValueError):
                    t = None
            elif line.startswith("time step continuity errors") and t is not None and t >= t_start:
                try:
                    vals.append(float(line.split("sum local =")[1].split(",")[0]))
                except (IndexError, ValueError):
                    pass
    return sum(vals) / len(vals) if vals else None

def physical_checks(job_directory, m):
    # Run after ParaView: reject solutions that break the energy equation or create total pressure
    checks = {}
    if m.get("pt_recovery_duct") is not None:
        checks["p0_gain"] = m["pt_recovery_duct"] < 1.0
    if m.get("duct_T_max") is not None and m.get("T0_K") is not None:
        checks["T_above_T0"] = m["duct_T_max"] <= m["T0_K"] + T0_MARGIN_K
    if not checks:
        checks["post_missing"] = False
    m["failed_physical_checks"] = [k for k, ok in checks.items() if not ok]
    m["failed_checks"] = list(m.get("failed_checks") or []) + m["failed_physical_checks"]
    m["converged"] = not m["failed_checks"]
    m["status"] = "ok" if m["converged"] else "not_converged"

    if m.get("cap_p0_avg") and m.get("p0_capture_Pa"):
        m["p0_cap_dev_pct"] = 100.0 * (m["cap_p0_avg"] / m["p0_capture_Pa"] - 1.0)
        if abs(m["p0_cap_dev_pct"]) > P0_CAP_DEV_WARN_PCT:
            print(f"  WARNING: capture p0 is {m['p0_cap_dev_pct']:+.1f} % from the intended "
                  f"{m['p0_capture_Pa']:.0f} Pa, ETA_DUCT_EST ({ETA_DUCT_EST}) is off for this design")
    write_json(job_directory, m)
    return m

def format_summary(m):
    def f(key, spec=".4g", default="n/a"):
        v = m.get(key)
        return format(v, spec) if isinstance(v, (int, float)) and not isinstance(v, bool) else default

    bar = "=" * 60
    return "\n".join([
        "", bar, f" INLET METRICS   [{m.get('status', '?')}]", bar,
        f"  LD {f('LD')} | AR {f('AR')} | WH {f('WH')} | c_cl {f('c_cl')} | c_area {f('c_area')}",
        f"  iterations        :  {f('iterations', 'd')}",
        "-" * 60,
        f"  mass flow target  :  {f('mdot_target_kg_s', '.4f')} kg/s",
        f"  mass flow out/in  :  {f('mdot_outlet_kg_s', '.4f')} / {f('mdot_inlet_kg_s', '.4f')} kg/s"
        f"   (ratio {f('mdot_ratio', '.4f')})",
        f"  back pressure     :  {f('p_back_Pa', '.1f')} Pa   (fixed, eta_duct est {f('eta_duct_est', '.3f')})",
        f"  capture p0        :  {f('cap_p0_avg', '.1f')} Pa   (intended {f('p0_capture_Pa', '.1f')}, "
        f"{f('p0_cap_dev_pct', '+.2f')} %)",
        "-" * 60,
        f"  flow metrics from :  {'mean fields' if m.get('fields_averaged') else 'LAST ITERATION (no mean fields)'}"
        f", iterations {f('avg_window_start', '.0f')} -> {f('avg_window_end', '.0f')}",
        f"  capture   Mach avg          :  {f('cap_mach_avg', '.4f')}",
        f"  engine    Mach avg / max    :  {f('ef_mach_avg', '.4f')} / {f('ef_mach_max', '.4f')}",
        f"  engine    T avg             :  {f('ef_T_avg', '.2f')} K",
        f"  engine    p avg / max       :  {f('ef_p_avg', '.1f')} / {f('ef_p_max', '.1f')} Pa",
        f"  engine    p0 (mass avg)     :  {f('ef_p0_avg', '.1f')} Pa",
        f"  duct      Mach avg / max    :  {f('duct_mach_avg', '.4f')} / {f('duct_mach_max', '.4f')}",
        f"  duct      T avg / max       :  {f('duct_T_avg', '.2f')} / {f('duct_T_max', '.2f')} K",
        f"  duct      p avg / max       :  {f('duct_p_avg', '.1f')} / {f('duct_p_max', '.1f')} Pa",
        f"  p0 recovery duct / total    :  {f('pt_recovery_duct', '.4f')} / {f('pt_recovery_total', '.4f')}",
        "-" * 60,
        f"  mass flow drift   :  {f('mdot_drift_pct', '+.4f')} %   (tol {f('drift_tolerance_pct', '.2f')})",
        f"  mass flow osc     :  {f('mdot_osc_pct', '.4f')} %   (tol {f('mdot_osc_tol_pct', '.2f')}, "
        f"std {f('mdot_std_pct', '.4f')} %)",
        f"  capture p drift   :  {f('p_cap_drift_pct', '+.4f')} %   (tol {f('drift_tolerance_pct', '.2f')})",
        f"  capture p osc     :  {f('p_cap_osc_pct', '.4f')} %   (tol {f('p_cap_osc_tol_pct', '.2f')})",
        f"  mass imbalance    :  {f('mass_imbalance_pct', '+.4f')} %   (tol {f('mass_imbalance_tol_pct', '.2f')})",
        f"  continuity error  :  {f('cont_err_avg', '.3g')}   (tol {f('cont_err_tol', '.0e')})",
        f"  y+  min/avg/max   :  {f('yplus_min', '.1f')} / {f('yplus_avg', '.1f')} / {f('yplus_max', '.1f')}"
        f"   (target avg {YPLUS_RANGE[0]:g}-{YPLUS_RANGE[1]:g}"
        f"{'' if m.get('yplus_in_range') else ', OUT OF RANGE'})",
        f"  mesh / solve time :  {f('mesh_time_s', '.1f')} / {f('solve_time_s', '.1f')} s",
        bar, f"  CONVERGED: {m.get('converged')}"
        f"{'' if m.get('converged') else '   failed: ' + ', '.join(m.get('failed_checks') or [])}", bar, "",
    ])

def plot_convergence(job_directory):
    try:
        sys.path.insert(0, TOOLS_DIR)
        import glob
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import metrics

        images = os.path.join(job_directory, "images")
        os.makedirs(images, exist_ok=True)

        # Mass flow
        to, out = metrics._read_dat(job_directory, "massFlowOutlet", "surfaceFieldValue.dat", 2)
        ti, inn = metrics._read_dat(job_directory, "massFlowInlet", "surfaceFieldValue.dat", 2)
        if to is not None and ti is not None:
            t_cut = to[0] + (1.0 - metrics.TAIL_FRACTION) * (to[-1] - to[0])
            plt.figure(figsize=(8, 5))
            plt.plot(ti, -inn[:, 0], linewidth=1.2, label="capture plane")
            plt.plot(to, out[:, 0], linewidth=1.2, label="outlet")
            plt.axhline(MDOT_TARGET, color="k", linestyle="--", linewidth=1.0,
                        label=f"target {MDOT_TARGET:g} kg/s")
            plt.axvspan(t_cut, to[-1], color="0.6", alpha=0.25,
                        label=f"averaging window (last {100*metrics.TAIL_FRACTION:.0f}%)")
            tail = out[to >= t_cut, 0]        # outlet is the free signal, the inlet is imposed
            if len(tail) > 1:
                lo = min(tail.min(), MDOT_TARGET); hi = max(tail.max(), MDOT_TARGET)
                pad = max(0.5 * (hi - lo), 0.02 * hi)
                plt.ylim(lo - pad, hi + pad)
            plt.xlabel("Iteration")
            plt.ylabel("Mass flow (kg/s)")
            plt.title("Mass Flow Convergence")
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend(fontsize=8)
            plt.savefig(os.path.join(images, "convergence_mass_flow.png"), dpi=200, bbox_inches="tight")
            plt.close()

        # Residuals
        files = sorted(glob.glob(os.path.join(job_directory, "postProcessing", "*", "*", "solverInfo.dat")),
                       key=metrics._start_time)
        residuals = {}
        for path in files:
            with open(path) as fh:
                lines = fh.readlines()
            header = next((l for l in lines if l.startswith("# Time")), None)
            if header is None:
                continue
            cols = header.lstrip("#").split()
            fields = {c[:-len("_initial")]: i for i, c in enumerate(cols) if c.endswith("_initial")}
            for line in lines:
                parts = line.split()
                if line.startswith("#") or len(parts) != len(cols):
                    continue
                for name, idx in fields.items():
                    t_list, v_list = residuals.setdefault(name, ([], []))
                    if t_list and float(parts[0]) <= t_list[-1]:
                        continue
                    try:
                        v_list.append(float(parts[idx])); t_list.append(float(parts[0]))
                    except ValueError:
                        pass
        if residuals:
            plt.figure(figsize=(9, 6))
            for name, (t, v) in sorted(residuals.items()):
                plt.semilogy(t, v, label=name, linewidth=1.0)
            plt.xlabel("Iteration")
            plt.ylabel("Initial Residual")
            plt.title("Residual Convergence")
            plt.grid(True, which="both", linestyle="--", alpha=0.5)
            plt.legend()
            plt.savefig(os.path.join(images, "residuals.png"), dpi=200, bbox_inches="tight")
            plt.close()
    except Exception as e:
        print(f"Warning: Could not plot convergence histories. Error: {e}")

def post_process(job_directory):
    job_id = os.path.basename(os.path.normpath(job_directory))
    foam = os.path.join(job_directory, f"{job_id}.foam")
    images = os.path.join(job_directory, "images")
    log = os.path.join(job_directory, "log.pvbatch")
    command = (f"LIBGL_ALWAYS_SOFTWARE=1 pvbatch --force-offscreen-rendering "
               f"{os.path.abspath(__file__)} post {foam} {images} > {log} 2>&1")
    result = subprocess.run(command, shell=True, executable="/bin/bash")
    if result.returncode != 0:
        print(f"Post-processing failed, see {log}")
        subprocess.run(f"tail -n 30 {log}", shell=True, executable="/bin/bash")
        return False
    return True

# ----------------------------------------------------------------------------
# ParaView post processing (runs under pvbatch)
# ----------------------------------------------------------------------------

def paraview_post(foam, images):
    from paraview import servermanager
    from paraview.simple import (OpenDataFile, Calculator, Slice, Clip, IntegrateVariables,
                                 CreateView, Show, ColorBy, GetColorTransferFunction,
                                 GetScalarBar, Render, SaveScreenshot, Delete)

    os.makedirs(images, exist_ok=True)
    case = os.path.dirname(os.path.abspath(foam))
    m = read_json(case) or {}

    # Window-averaged fields from fieldAverage when the latest time has them, else the last iteration
    time_dirs = [d for d in os.listdir(case) if os.path.isdir(os.path.join(case, d))
                 and _is_float(d) and float(d) > 0]
    latest = max(time_dirs, key=float) if time_dirs else None
    averaged = latest is not None and all(
        os.path.exists(os.path.join(case, latest, f)) for f in ("UMean", "pMean", "TMean"))
    sfx = "Mean" if averaged else ""
    U_, p_, T_ = f"U{sfx}", f"p{sfx}", f"T{sfx}"
    if not averaged:
        print("WARNING: no UMean/pMean/TMean in the latest time, using the last iteration")

    reader = OpenDataFile(foam)
    reader.MeshRegions = ["internalMesh"]
    reader.CellArrays = [p_, T_, U_]
    reader.UpdatePipelineInformation()
    times = reader.TimestepValues
    times = list(times) if hasattr(times, "__len__") else [times]
    t = times[-1] if times else 0.0

    # Copy the chosen fields to plain names so everything below is the same either way
    src = reader
    if averaged:
        for name, fn in [("p", p_), ("T", T_), ("U", U_)]:
            src = Calculator(Input=src)
            src.AttributeType = "Cell Data"
            src.ResultArrayName = name
            src.Function = fn

    # Derived cell fields (perfect gas, same gamma / R as the 1D estimates)
    for name, fn in [
        ("Mach",    f"mag(U)/sqrt({GAMMA}*{R_GAS}*T)"),
        ("rho",     f"p/({R_GAS}*T)"),
        ("massFlux", "rho*U_X"),
        ("p0",      f"p*(1+{0.5*(GAMMA-1):g}*Mach*Mach)^{GAMMA/(GAMMA-1):g}"),
        ("p0Flux",  "p0*massFlux"),
    ]:
        src = Calculator(Input=src)
        src.AttributeType = "Cell Data"
        src.ResultArrayName = name
        src.Function = fn
    src.UpdatePipeline(t)

    def integrate(source):
        integ = IntegrateVariables(Input=source)
        integ.DivideCellDataByVolume = 0
        integ.UpdatePipeline(t)
        cd = servermanager.Fetch(integ).GetCellData()
        vals = {cd.GetArrayName(i): cd.GetArray(i).GetValue(0) for i in range(cd.GetNumberOfArrays())}
        Delete(integ)
        return vals

    def plane(prefix, x):
        s = Slice(Input=src)
        s.SliceType = "Plane"
        s.SliceType.Origin = [x, 0.0, 0.0]
        s.SliceType.Normal = [1.0, 0.0, 0.0]
        s.UpdatePipeline(t)
        I = integrate(s)
        A, mdot = I["Area"], I["massFlux"]
        out = {
            f"{prefix}_area_m2": A,
            f"{prefix}_mdot_kg_s": mdot,
            f"{prefix}_mach_avg": I["Mach"] / A,
            f"{prefix}_mach_max": s.CellData["Mach"].GetRange()[1],
            f"{prefix}_T_avg": I["T"] / A,
            f"{prefix}_p_avg": I["p"] / A,
            f"{prefix}_p_max": s.CellData["p"].GetRange()[1],
            f"{prefix}_p0_avg": I["p0Flux"] / mdot,       # mass-weighted
        }
        Delete(s)
        return out

    x_lip = m.get("x_lip_m"); x_ef = m.get("x_engine_face_m")
    if x_lip is None or x_ef is None:
        b = src.GetDataInformation().GetBounds()
        x_lip, x_ef = b[0], b[1]

    stats = {}
    stats.update(plane("cap", x_lip + X_OFFSET))
    stats.update(plane("ef", x_ef + X_OFFSET))

    # Volume statistics over the S-duct only (capture plane to engine face, no extension)
    clip = Clip(Input=src)
    clip.ClipType = "Plane"
    clip.ClipType.Origin = [x_ef, 0.0, 0.0]
    clip.ClipType.Normal = [1.0, 0.0, 0.0]
    clip.Invert = 1
    clip.UpdatePipeline(t)
    if clip.GetDataInformation().GetBounds()[1] > x_ef + 1e-3:
        clip.Invert = 0
        clip.UpdatePipeline(t)
    I = integrate(clip)
    V = I["Volume"]
    stats.update({
        "duct_volume_m3": V,
        "duct_mach_avg": I["Mach"] / V,
        "duct_mach_max": clip.CellData["Mach"].GetRange()[1],
        "duct_T_avg": I["T"] / V,
        "duct_T_max": clip.CellData["T"].GetRange()[1],
        "duct_p_avg": I["p"] / V,
        "duct_p_max": clip.CellData["p"].GetRange()[1],
    })
    Delete(clip)

    if stats.get("cap_p0_avg"):
        stats["pt_recovery_duct"] = stats["ef_p0_avg"] / stats["cap_p0_avg"]
    if stats.get("pt_recovery_duct") is not None:
        # The capture p0 is a result now, so chain the assumed 2D recovery with the duct ratio
        stats["pt_recovery_total"] = m.get("pt_recovery_2d", PT_RECOVERY_2D) * stats["pt_recovery_duct"]
    stats["post_time"] = t
    stats["fields_averaged"] = averaged

    m.update(stats)
    write_json(case, m)
    print(json.dumps(stats, indent=2))

    # Side slices (y = 0 plane, viewed from -y)
    side = Slice(Input=src)
    side.SliceType = "Plane"
    side.SliceType.Origin = [0.0, Y_SLICE, 0.0]
    side.SliceType.Normal = [0.0, 1.0, 0.0]
    side.UpdatePipeline(t)
    b = side.GetDataInformation().GetBounds()

    # Parallel camera, the slice fills the left ~85 % and the colour bar sits on the right
    aspect = VIEW_SIZE[0] / VIEW_SIZE[1]
    xspan, zspan = b[1] - b[0], b[5] - b[4]
    scale = max(0.5 * zspan / 0.85, xspan / (0.84 * 2.0 * aspect))
    width = 2.0 * scale * aspect
    focal = [b[0] - 0.03 * width + 0.5 * width, 0.0, 0.5 * (b[4] + b[5])]

    def view():
        rv = CreateView("RenderView")
        rv.ViewSize = VIEW_SIZE
        rv.ViewTime = t
        rv.OrientationAxesVisibility = 0
        rv.CameraPosition = [focal[0], -1.0, focal[2]]
        rv.CameraFocalPoint = focal
        rv.CameraViewUp = [0, 0, 1]
        rv.CameraParallelProjection = 1
        rv.CameraParallelScale = scale
        return rv

    def scalar_bar(display, rv, array, title):
        display.SetScalarBarVisibility(rv, True)
        lut = GetColorTransferFunction(array)
        bar = GetScalarBar(lut, rv)
        bar.WindowLocation = "Any Location"
        bar.Orientation = "Vertical"
        bar.Position = SCALAR_BAR["position"]
        bar.ScalarBarLength = SCALAR_BAR["length"]
        bar.ScalarBarThickness = SCALAR_BAR["thickness"]
        bar.TitleFontSize = SCALAR_BAR["title_size"]
        bar.LabelFontSize = SCALAR_BAR["label_size"]
        bar.TitleBold = 1
        lo, hi = lut.RGBPoints[0], lut.RGBPoints[-4]
        n = SCALAR_BAR["n_labels"]
        bar.UseCustomLabels = 1
        bar.CustomLabels = [lo + (hi - lo) * i / (n - 1) for i in range(1, n - 1)]
        bar.AutomaticLabelFormat = 1
        bar.RangeLabelFormat = "%-#.4g"
        bar.Title = title
        bar.ComponentTitle = ""

    for array, title, fname in [("Mach", "Mach", "slice-mach-side.png"),
                                ("p", "p [Pa]", "slice-pressure-side.png")]:
        rv = view()
        d = Show(side, rv)
        ColorBy(d, ("CELLS", array))
        lo, hi = side.CellData[array].GetRange()
        GetColorTransferFunction(array).RescaleTransferFunction(lo, hi)
        scalar_bar(d, rv, array, title)
        Render(rv)
        SaveScreenshot(os.path.join(images, fname), rv, ImageResolution=VIEW_SIZE)
        Delete(rv)

    # Mesh on the side slice
    rv = view()
    d = Show(side, rv)
    d.Representation = "Surface With Edges"
    ColorBy(d, None)
    Render(rv)
    SaveScreenshot(os.path.join(images, "mesh-side.png"), rv, ImageResolution=VIEW_SIZE)
    Delete(rv)

if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "post":
        if len(sys.argv) < 4:
            print("usage: pvbatch run_inlet.py post <case.foam> <images_dir>")
            sys.exit(1)
        paraview_post(sys.argv[2], sys.argv[3])
        sys.exit(0)
    sys.exit(main())