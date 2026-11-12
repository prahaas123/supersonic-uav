"""One S-duct inlet CFD run: geometry, mesh, rhoSimpleFoam, metrics and images.

    python3 run_inlet.py [case_dir]                        full run (see run_inlet.sh)
    pvbatch run_inlet.py post <case.foam> <images_dir>     ParaView part, called by the full run

Boundary conditions (case_template_inlet): total pressure / temperature at the capture plane and a
fixed static back pressure at the outlet, so the mass flow is a result. It matches MDOT_TARGET when
ETA_DUCT_EST equals the duct recovery of the run; the summary suggests the value to use next.
"""
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR     = os.path.join(SCRIPT_DIR, "..", "0-tools")
CASE_TEMPLATE = os.path.join(SCRIPT_DIR, "case_template_inlet")
# build_inlet and metrics need numpy, so they are imported inside the functions that use them:
# pvbatch loads this file too and its Python is not guaranteed to have numpy.
sys.path[:0] = [SCRIPT_DIR, TOOLS_DIR]

# Flight condition
ALTITUDE_M     = 15000.0     # [m]
MACH           = 1.5
PT_RECOVERY_2D = 0.94        # total pressure recovery of the 2D external compression (2d-inlet-designs.csv)
MDOT_TARGET    = 0.2635      # RDE air mass flow [kg/s] (README, section 1)
ETA_DUCT_EST   = 0.9664      # assumed S-duct p0 recovery, sets the back pressure and so the mass flow
                             # (0.9664: converged run at ETA_DUCT_EST 0.92, mass flow +17.5 %)

# S-duct design
INLET_PARAMS = dict(
    LD=5.5,       # duct length / engine-face diameter
    AR=1.5,       # engine-face area / capture area
    WH=2.5,       # capture width / height
    c_cl=0.5,     # centreline S-curve inflection
    c_area=0.5,   # area-distribution inflection
)

# Run settings
NP             = 50          # MPI processes
N_ITER         = 8000        # rhoSimpleFoam iterations (residualControl usually stops it earlier)
WRITE_INTERVAL = 500         # [iterations]
BASE_CELL      = 0.006       # background cell [m], refined to level 2 (1.5 mm) inside the duct
TARGET_YPLUS   = 40.0        # cell-centre y+ used to size the first layer
YPLUS_RANGE    = (30.0, 100.0)  # wall-function range for the average y+
EXT_LENGTH_D   = 1.5         # straight extension after the engine face [engine-face diameters]
N_EXT          = 13          # axial stations on the extension
TURB_INTENSITY = 0.05        # at the capture plane, behind the shock system
RAMP_ITERS     = 1500        # back pressure ramps down to p_back over these iterations ...
P_START_FRAC   = 0.2         # ... starting at p0_capture - P_START_FRAC * (p0_capture - p_back)

# Convergence checks over the last metrics.TAIL_FRACTION of the run
MASS_IMBALANCE_TOL_PCT = 0.5   # |inlet - outlet| mass flow
OSC_TOL_PCT            = 0.2   # half peak-to-peak / mean, outlet mass flow and capture static p
CONT_ERR_TOL           = 1e-3  # mean "sum local" continuity error
MDOT_TARGET_TOL_PCT    = 0.5   # mass flow vs MDOT_TARGET, reported only (it is a result of ETA_DUCT_EST)
T0_MARGIN_K            = 1.0   # adiabatic duct: T may not exceed T0 by more than this

GAMMA = 1.4
R_GAS = 287.058

# Post processing
SLICE_STATIONS = (0.0, 0.25, 0.5, 0.75, 1.0)   # x/L of the front-view slices, capture plane -> engine face
X_OFFSET   = 1e-4            # planes kept just inside the capture plane / downstream of the engine face
Y_SLICE    = 1e-4            # side slice just off the y = 0 symmetry plane (avoids cell faces)
VIEW_SIZE  = [3840, 1440]
SCALAR_BAR = {"position": [0.90, 0.05], "length": 0.90, "thickness": 48,
              "title_size": 48, "label_size": 40, "n_labels": 6}

# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def main():
    case = sys.argv[1] if len(sys.argv) > 1 else "./inlet_baseline"
    name = os.path.basename(os.path.normpath(case))
    print(f"\n{'=' * 40}\nStarting Simulation: {name}\n  {INLET_PARAMS}\n{'=' * 40}")

    print("[1/5] Preparing case, S-duct STL and boundary conditions...")
    try:
        refs = prepare(case, INLET_PARAMS)
    except Exception as e:
        print(f"  preparation failed: {e}")
        return fail(case, "prepare_failed", {**INLET_PARAMS, "error": str(e)})

    print("[2/5] Generating mesh (snappyHexMesh)...")
    t0 = time.perf_counter()
    if not mesh(case):
        return fail(case, "mesh_failed", refs)
    t1 = time.perf_counter()

    print("[3/5] Solving (rhoSimpleFoam)...")
    if not solve(case):
        return fail(case, "solve_failed", refs)
    t2 = time.perf_counter()
    refs.update(mesh_time_s=round(t1 - t0, 1), solve_time_s=round(t2 - t1, 1))

    print("[4/5] Convergence and ParaView post processing...")
    write_json(case, convergence_metrics(case, refs))
    post_process(case)                      # adds the flow metrics to metrics.json
    plot_convergence(case)
    m = read_json(case)
    m["failed_checks"] += physical_checks(m)
    m["converged"] = not m["failed_checks"]
    m["status"] = "ok" if m["converged"] else "not_converged"
    write_json(case, m)
    print(format_summary(m))
    print_warnings(m)

    print("[5/5] Cleaning up processor files...")
    sh(f"rm -rf {case}/processor*")
    print(f"Finished {name}")
    return 0 if m["converged"] else 2

def fail(case, status, refs):
    print(f"Error: {status}")
    write_json(case, {"case": os.path.abspath(case), "status": status, "converged": False, **refs})
    sh(f"rm -rf {case}/processor*")
    return 1

# ----------------------------------------------------------------------------
# Flow conditions
# ----------------------------------------------------------------------------

def isa_atmosphere(h):
    # Troposphere + lower stratosphere, returns (T, p)
    g0, T_sl, p_sl, lapse, h_tp = 9.80665, 288.15, 101325.0, -0.0065, 11000.0
    T = T_sl + lapse * min(h, h_tp)
    p = p_sl * (T / T_sl) ** (-g0 / (lapse * R_GAS))
    if h > h_tp:
        p *= math.exp(-g0 * (h - h_tp) / (R_GAS * T))
    return T, p

def isentropic_ratio(M):
    # T0/T; p0/p = ratio ** (gamma / (gamma - 1))
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

    T_inf, p_inf = isa_atmosphere(ALTITUDE_M)
    pexp  = GAMMA / (GAMMA - 1.0)
    T0    = T_inf * isentropic_ratio(MACH)
    p0inf = p_inf * isentropic_ratio(MACH) ** pexp
    p0c   = PT_RECOVERY_2D * p0inf
    a_ef  = math.pi / 4.0 * bi.D_EF ** 2
    a_cap = a_ef / params["AR"]

    # 1D engine face at the target mass flow with the assumed duct recovery -> back pressure
    m_ef = subsonic_mach(MDOT_TARGET, a_ef, ETA_DUCT_EST * p0c, T0)
    if m_ef is None:
        raise ValueError("MDOT_TARGET is above the choked mass flow of the engine face")
    m_cap = subsonic_mach(MDOT_TARGET, a_cap, p0c, T0)
    if m_cap is None:
        raise ValueError("MDOT_TARGET is above the choked mass flow of the capture area")

    p_back = ETA_DUCT_EST * p0c / isentropic_ratio(m_ef) ** pexp
    T_ef   = T0 / isentropic_ratio(m_ef)
    U_ef   = m_ef * math.sqrt(GAMMA * R_GAS * T_ef)

    # Capture-plane turbulence: intensity, mixing length from the capture equivalent diameter
    U_cap = m_cap * math.sqrt(GAMMA * R_GAS * T0 / isentropic_ratio(m_cap))
    l_mix = 0.07 * bi.D_EF / math.sqrt(params["AR"])
    k     = 1.5 * (TURB_INTENSITY * U_cap) ** 2
    omega = k ** 0.5 / (0.09 ** 0.25 * l_mix)

    return dict(
        p_inf_Pa=p_inf, T_inf_K=T_inf, p0_inf_Pa=p0inf, T0_K=T0,
        p0_capture_Pa=p0c, p_back_Pa=p_back, mdot_target_kg_s=MDOT_TARGET,
        pt_recovery_2d=PT_RECOVERY_2D, eta_duct_est=ETA_DUCT_EST,
        capture_area_m2=a_cap, engine_face_area_m2=a_ef,
        mach_capture_1d=m_cap, mach_engine_face_1d=m_ef,
        T_ef_1d_K=T_ef, U_ef_1d=U_ef, rho_ef_1d=p_back / (R_GAS * T_ef),
        k_init=k, omega_init=omega, mixing_length_m=l_mix,
    )

def first_layer_thickness(rho, U, T, length, y_plus):
    nu    = 1.458e-6 * T ** 1.5 / (T + 110.4) / rho      # Sutherland
    cf    = 0.026 / (U * length / nu) ** (1.0 / 7.0)     # turbulent flat plate
    u_tau = U * math.sqrt(0.5 * cf)
    return 2.0 * y_plus * nu / u_tau                     # y+ is evaluated at the cell centre

# ----------------------------------------------------------------------------
# Geometry and case setup
# ----------------------------------------------------------------------------

def write_solid(fh, name, tris):
    import build_inlet as bi
    fh.write(f"solid {name}\n")
    for n, tri in zip(bi._normals(tris), tris):
        fh.write(f" facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n  outer loop\n")
        for v in tri:
            fh.write(f"   vertex {v[0]:.9e} {v[1]:.9e} {v[2]:.9e}\n")
        fh.write("  endloop\n endfacet\n")
    fh.write(f"endsolid {name}\n")

def add_extension(duct_stl, out_stl, length):
    # Replace the engine-face cap with a straight duct of the engine-face section and a new outlet cap
    import numpy as np

    with open(duct_stl) as fh:
        solids = dict(re.findall(r"solid (\w+)\n(.*?)endsolid \1\n", fh.read(), flags=re.S))

    v = np.unique(np.array(re.findall(r"vertex (\S+) (\S+) (\S+)", solids["outlet"]), dtype=float), axis=0)
    x_ef = float(v[0, 0])
    i_c = np.argmin(np.sum((v - v.mean(axis=0)) ** 2, axis=1))
    centre, ring = v[i_c], np.delete(v, i_c, axis=0)
    ang = np.mod(np.arctan2(ring[:, 2] - centre[2], ring[:, 1] - centre[1]), 2 * np.pi)
    ring = ring[np.argsort(ang)]            # counter-clockwise about +x, same order as build_inlet

    xs = x_ef + length * np.linspace(0.0, 1.0, N_EXT)
    P = np.repeat(ring[None, :, :], N_EXT, axis=0)
    P[..., 0] = xs[:, None]
    i = np.arange(N_EXT - 1)[:, None]
    j = np.arange(len(ring))[None, :]
    jn = (j + 1) % len(ring)
    a, b, c, d = P[i, j], P[i + 1, j], P[i + 1, jn], P[i, jn]
    wall = np.concatenate([np.stack([a, d, c], -2).reshape(-1, 3, 3),
                           np.stack([a, c, b], -2).reshape(-1, 3, 3)])
    end = P[-1]
    C = np.broadcast_to(np.array([xs[-1], centre[1], centre[2]]), end.shape)
    cap = np.stack([C, end, np.roll(end, -1, axis=0)], 1)      # normal +x, outward

    with open(out_stl, "w") as fh:
        for name in ("inlet", "wall"):
            fh.write(f"solid {name}\n{solids[name]}endsolid {name}\n")
        write_solid(fh, "extension", wall)
        write_solid(fh, "outlet", cap)

    pts = np.concatenate([P.reshape(-1, 3),
                          np.array(re.findall(r"vertex (\S+) (\S+) (\S+)", solids["wall"]), dtype=float)])
    return x_ef, float(xs[-1]), pts.min(axis=0), pts.max(axis=0)

def sub(path, pattern, repl, flags=re.M):
    # Regex replace in a case file, the pattern must match
    with open(path) as fh:
        text, n = re.subn(pattern, repl, fh.read(), flags=flags)
    if n == 0:
        raise ValueError(f"pattern {pattern!r} not found in {path}")
    with open(path, "w") as fh:
        fh.write(text)

def prepare(case, params):
    import numpy as np
    import build_inlet as bi

    if os.path.exists(case):
        shutil.rmtree(case)
    shutil.copytree(CASE_TEMPLATE, case, ignore=shutil.ignore_patterns("*.foam"))
    open(os.path.join(case, os.path.basename(os.path.normpath(case)) + ".foam"), "w").close()
    path = lambda *p: os.path.join(case, *p)

    # Geometry: S-duct from build_inlet plus the straight extension
    raw = path("constant", "triSurface", "sduct_raw.stl")
    os.makedirs(os.path.dirname(raw), exist_ok=True)
    bi.build_inlet(out_path=raw, **params)
    L = params["LD"] * bi.D_EF
    x_ef, x_out, lo, hi = add_extension(raw, path("constant", "triSurface", "duct.stl"),
                                        EXT_LENGTH_D * bi.D_EF)
    os.remove(raw)
    flow = flow_conditions(params)

    # Background block: duct bounding box padded by 1.5 cells
    lo, hi = lo - 1.5 * BASE_CELL, hi + 1.5 * BASE_CELL
    n = np.maximum(np.ceil((hi - lo) / BASE_CELL).astype(int), 1)
    hi = lo + n * BASE_CELL
    corners = [(lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1])]
    verts = [(x, y, z) for z in (lo[2], hi[2]) for x, y in corners]
    vtext = "vertices\n(\n" + "".join(f"    ({x:.6f} {y:.6f} {z:.6f})   // {k}\n"
                                      for k, (x, y, z) in enumerate(verts)) + ");"
    sub(path("system", "blockMeshDict"), r"^vertices\s*\(.*?^\);", vtext, flags=re.M | re.S)
    sub(path("system", "blockMeshDict"), r"hex \(0 1 2 3 4 5 6 7\) \(\d+ \d+ \d+\)",
        f"hex (0 1 2 3 4 5 6 7) ({n[0]} {n[1]} {n[2]})")

    # snappyHexMesh: a point inside the duct (mid-length on the centreline) and the first layer
    zc = float(bi._section(bi.X_LIP + 0.5 * L, L, 0, 0, params["c_cl"], params["c_area"])[0])
    loc = (bi.X_LIP + 0.5 * L + 1.234e-4, 1.234e-4, zc + 1.234e-4)
    y1 = first_layer_thickness(flow["rho_ef_1d"], flow["U_ef_1d"], flow["T_ef_1d_K"], L, TARGET_YPLUS)
    shm = path("system", "snappyHexMeshDict")
    sub(shm, r"^(\s*)locationInMesh\s+[^;\n]+;", rf"\g<1>locationInMesh ({loc[0]:.6f} {loc[1]:.6f} {loc[2]:.6f});")
    sub(shm, r"^(\s*)firstLayerThickness\s+[^;\n]+;", rf"\g<1>firstLayerThickness {y1:.4g};")
    sub(shm, r"^(\s*)minThickness\s+[^;\n]+;", rf"\g<1>minThickness {0.5 * y1:.4g};")

    # Run control
    sub(path("system", "decomposeParDict"), r"^numberOfSubdomains\s+\d+;", f"numberOfSubdomains  {NP};")
    sub(path("system", "controlDict"), r"^endTime\s+[^;\n]+;", f"endTime         {N_ITER};")
    sub(path("system", "controlDict"), r"^writeInterval\s+[^;\n]+;", f"writeInterval   {WRITE_INTERVAL};")

    # Back pressure ramp, then boundary / initial values
    p0c, p_back = flow["p0_capture_Pa"], flow["p_back_Pa"]
    p_start = p0c - P_START_FRAC * (p0c - p_back)
    sub(path("0", "p"), r"^(\s*)uniformValue\s+[^;\n]+;",
        rf"\g<1>uniformValue    table ((0 {p_start:.6g}) ({RAMP_ITERS} {p_back:.6g}));")
    values = {"p0In": p0c, "T0In": flow["T0_K"], "pOut": p_back, "pInit": p_start,
              "TInit": flow["T_ef_1d_K"], "UInit": f"({flow['U_ef_1d']:.6g} 0 0)",
              "kInit": flow["k_init"], "omegaInit": flow["omega_init"],
              "turbIntensity": TURB_INTENSITY, "mixingLength": flow["mixing_length_m"]}
    with open(path("0", "include", "initialConditions"), "w") as fh:
        fh.writelines(f"{k:<13} {v if isinstance(v, str) else format(v, '.6g')};\n" for k, v in values.items())

    print(f"  {n[0]}x{n[1]}x{n[2]} background cells, first layer {y1 * 1e6:.1f} um for y+ = {TARGET_YPLUS:g}")
    print(f"  p0 capture {p0c:.0f} Pa, T0 {flow['T0_K']:.2f} K | back pressure {p_start:.0f} -> {p_back:.0f} Pa "
          f"over {RAMP_ITERS} it (eta_duct {ETA_DUCT_EST:g}) | 1D Mach capture {flow['mach_capture_1d']:.3f}, "
          f"engine face {flow['mach_engine_face_1d']:.3f}")

    return {**params, **flow, "duct_length_m": L, "x_lip_m": bi.X_LIP, "x_engine_face_m": x_ef,
            "x_outlet_m": x_out, "first_layer_m": y1, "target_yplus": TARGET_YPLUS,
            "base_cell_m": BASE_CELL, "ramp_iters": RAMP_ITERS, "p_back_start_Pa": p_start}

# ----------------------------------------------------------------------------
# OpenFOAM
# ----------------------------------------------------------------------------

def sh(cmd):
    return subprocess.run(cmd, shell=True, executable="/bin/bash").returncode

def run_steps(case, steps):
    for app, cmd in steps:
        log = os.path.join(case, f"log.{app}")
        print(f"  -> {cmd}   (log.{app})")
        rc = sh(f"{cmd} > {log} 2>&1")
        if rc != 0:
            print(f"  {app} failed (exit {rc}), last lines of log.{app}:")
            sh(f"tail -n 30 {log}")
            return False
    return True

def mesh(c):
    ok = run_steps(c, [
        ("surfaceFeatureExtract", f"surfaceFeatureExtract -case {c}"),
        ("blockMesh",             f"blockMesh -case {c}"),
        ("decomposePar.mesh",     f"decomposePar -force -case {c}"),
        ("snappyHexMesh",         f"mpirun -np {NP} snappyHexMesh -parallel -overwrite -case {c}"),
        ("reconstructParMesh",    f"reconstructParMesh -constant -case {c}"),
    ])
    sh(f"rm -rf {c}/processor*")
    if ok:
        sh(f"checkMesh -constant -case {c} > {c}/log.checkMesh 2>&1")     # quality report, reference only
    return ok and os.path.isdir(os.path.join(c, "constant", "polyMesh"))

def solve(c):
    ok = run_steps(c, [
        ("decomposePar",   f"decomposePar -force -case {c}"),
        ("rhoSimpleFoam",  f"mpirun -np {NP} rhoSimpleFoam -parallel -case {c}"),
        ("reconstructPar", f"reconstructPar -latestTime -case {c}"),
    ])
    sh(f"rm -rf {c}/processor*")
    return ok and latest_time(c) is not None

def latest_time(case):
    # Latest time directory after 0, None if there is none
    times = []
    for d in os.listdir(case):
        try:
            if float(d) > 0 and os.path.isdir(os.path.join(case, d)):
                times.append(d)
        except ValueError:
            pass
    return max(times, key=float) if times else None

# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

def read_json(case):
    try:
        with open(os.path.join(case, "metrics.json")) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}

def write_json(case, data):
    os.makedirs(case, exist_ok=True)
    with open(os.path.join(case, "metrics.json"), "w") as fh:
        json.dump(data, fh, indent=2)

def convergence_metrics(case, refs):
    # Iterative convergence over the last metrics.TAIL_FRACTION of the run. The capture p0 and the
    # back pressure are imposed, so the mass flow and the capture static pressure are the free signals.
    import metrics

    m = {"case": os.path.abspath(case), **refs, "tail_fraction": metrics.TAIL_FRACTION,
         "drift_tol_pct": metrics.DRIFT_TOLERANCE_PCT}
    to, out = metrics._read_dat(case, "massFlowOutlet", "surfaceFieldValue.dat", 2)
    ti, inn = metrics._read_dat(case, "massFlowInlet", "surfaceFieldValue.dat", 2)
    tp, pc  = metrics._read_dat(case, "inletPressure", "surfaceFieldValue.dat", 2)
    so = metrics.window_stats(to, out[:, 0]) if to is not None else None
    si = metrics.window_stats(ti, -inn[:, 0]) if ti is not None else None    # inflow phi is negative
    sp = metrics.window_stats(tp, pc[:, 0]) if tp is not None else None
    if not (so and si and sp) or so["n_samples"] < metrics.MIN_TAIL_SAMPLES:
        m["failed_checks"] = ["no_data"]
        return m

    osc = lambda s: 50.0 * (s["max"] - s["min"]) / s["value"]          # half peak-to-peak [%]
    m.update(
        iterations=int(to[-1]), window_start=so["window_start"], window_end=so["window_end"],
        mdot_outlet_kg_s=so["value"], mdot_inlet_kg_s=si["value"], mdot_ratio=so["value"] / MDOT_TARGET,
        mdot_drift_pct=so["drift_pct"], mdot_osc_pct=osc(so),
        mass_imbalance_pct=100.0 * (si["value"] - so["value"]) / so["value"],
        p_cap_static_Pa=sp["value"], p_cap_drift_pct=sp["drift_pct"], p_cap_osc_pct=osc(sp),
        cont_err_avg=continuity_error(case, so["window_start"]),
        **metrics.read_yplus(case, patch="wall"),
    )
    drift_ok = lambda d: d is not None and abs(d) < metrics.DRIFT_TOLERANCE_PCT
    checks = {
        "mdot_drift":        drift_ok(m["mdot_drift_pct"]),
        "mdot_oscillation":  m["mdot_osc_pct"] < OSC_TOL_PCT,
        "p_cap_drift":       drift_ok(m["p_cap_drift_pct"]),
        "p_cap_oscillation": m["p_cap_osc_pct"] < OSC_TOL_PCT,
        "imbalance":         abs(m["mass_imbalance_pct"]) < MASS_IMBALANCE_TOL_PCT,
        "continuity":        m["cont_err_avg"] is not None and m["cont_err_avg"] < CONT_ERR_TOL,
    }
    m["failed_checks"] = [k for k, ok in checks.items() if not ok]
    m["mdot_on_target"] = abs(m["mdot_ratio"] - 1.0) * 100.0 < MDOT_TARGET_TOL_PCT
    m["yplus_in_range"] = m["yplus_avg"] is not None and YPLUS_RANGE[0] <= m["yplus_avg"] <= YPLUS_RANGE[1]
    return m

def continuity_error(case, t_start):
    # Mean "sum local" continuity error in log.rhoSimpleFoam over iterations >= t_start
    vals, t = [], None
    try:
        with open(os.path.join(case, "log.rhoSimpleFoam")) as fh:
            for line in fh:
                if line.startswith("Time = "):
                    t = float(line.split()[2])
                elif line.startswith("time step continuity errors") and t is not None and t >= t_start:
                    vals.append(float(line.split("sum local =")[1].split(",")[0]))
    except (OSError, ValueError, IndexError):
        pass
    return sum(vals) / len(vals) if vals else None

def physical_checks(m):
    # Reject solutions that create total pressure or heat an adiabatic duct
    if m.get("pt_recovery_duct") is None or m.get("duct_T_max") is None:
        return ["post_missing"]
    failed = []
    if m["pt_recovery_duct"] >= 1.0:
        failed.append("p0_gain")
    if m["duct_T_max"] > m["T0_K"] + T0_MARGIN_K:
        failed.append("T_above_T0")
    return failed

def format_summary(m):
    def f(key, spec=".4g"):
        v = m.get(key)
        return format(v, spec) if isinstance(v, (int, float)) and not isinstance(v, bool) else "n/a"

    bar = "=" * 64
    lines = [
        "", bar, f" INLET METRICS   [{m.get('status', '?')}]", bar,
        f"  LD {f('LD')} | AR {f('AR')} | WH {f('WH')} | c_cl {f('c_cl')} | c_area {f('c_area')}",
        f"  iterations          {f('iterations', 'd')}   (window {f('window_start', '.0f')} -> {f('window_end', '.0f')})",
        f"  mass flow out / in  {f('mdot_outlet_kg_s', '.4f')} / {f('mdot_inlet_kg_s', '.4f')} kg/s   "
        f"target {f('mdot_target_kg_s', '.4f')}, ratio {f('mdot_ratio', '.4f')}"
        f"{'' if m.get('mdot_on_target') else '  OFF TARGET'}",
        f"  p0 capture / back p {f('p0_capture_Pa', '.0f')} / {f('p_back_Pa', '.0f')} Pa   "
        f"(eta_duct est {f('eta_duct_est', '.4f')})",
        "-" * 64,
        f"  Mach  capture avg {f('cap_mach_avg', '.3f')} | engine face avg / max {f('ef_mach_avg', '.3f')} / "
        f"{f('ef_mach_max', '.3f')} | duct max {f('duct_mach_max', '.3f')}",
        f"  T     engine face avg {f('ef_T_avg', '.1f')} K | duct max {f('duct_T_max', '.1f')} K "
        f"(T0 {f('T0_K', '.1f')})",
        f"  p0 recovery duct / total   {f('pt_recovery_duct', '.4f')} / {f('pt_recovery_total', '.4f')}",
        "  stations (x/L)      p avg [Pa]   p0 mass avg [Pa]   Mach avg",
    ]
    for s in m.get("stations", []):
        lines.append(f"    {s['x_over_L']:6.2f}          {s['p_avg']:9.0f}   {s['p0_avg']:16.0f}   {s['mach_avg']:8.3f}")
    lines += [
        "-" * 64,
        f"  mass flow drift / osc   {f('mdot_drift_pct', '+.4f')} / {f('mdot_osc_pct', '.4f')} %   "
        f"(tol {f('drift_tol_pct', 'g')} / {OSC_TOL_PCT})",
        f"  capture p drift / osc   {f('p_cap_drift_pct', '+.4f')} / {f('p_cap_osc_pct', '.4f')} %",
        f"  mass imbalance          {f('mass_imbalance_pct', '+.4f')} %   (tol {MASS_IMBALANCE_TOL_PCT})",
        f"  continuity error        {f('cont_err_avg', '.3g')}   (tol {CONT_ERR_TOL:g})",
        f"  y+ min / avg / max      {f('yplus_min', '.1f')} / {f('yplus_avg', '.1f')} / {f('yplus_max', '.1f')}"
        f"{'' if m.get('yplus_in_range') else '   OUT OF RANGE'}",
        f"  mesh / solve time       {f('mesh_time_s', '.0f')} / {f('solve_time_s', '.0f')} s",
        bar, f"  CONVERGED: {m.get('converged')}"
        f"{'' if m.get('converged') else '   failed: ' + ', '.join(m.get('failed_checks', []))}", bar, "",
    ]
    return "\n".join(lines)

def print_warnings(m):
    if m.get("mdot_ratio") is not None and not m.get("mdot_on_target"):
        hint = (f", try ETA_DUCT_EST = {m['pt_recovery_duct']:.4f} (the duct recovery of this run)"
                if m.get("pt_recovery_duct") is not None else "")
        print(f"WARNING: mass flow is {100.0 * (m['mdot_ratio'] - 1.0):+.2f} % from the target "
              f"{MDOT_TARGET:g} kg/s with ETA_DUCT_EST = {ETA_DUCT_EST}{hint}")
    if m.get("yplus_avg") is not None and not m.get("yplus_in_range"):
        print(f"WARNING: average y+ {m['yplus_avg']:.1f} is outside {YPLUS_RANGE}, adjust TARGET_YPLUS")
    if not m.get("converged"):
        print(f"WARNING: failed {m.get('failed_checks')}, treat this result as unusable")

def plot_convergence(case):
    try:
        import glob
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import metrics

        images = os.path.join(case, "images")
        os.makedirs(images, exist_ok=True)

        to, out = metrics._read_dat(case, "massFlowOutlet", "surfaceFieldValue.dat", 2)
        ti, inn = metrics._read_dat(case, "massFlowInlet", "surfaceFieldValue.dat", 2)
        if to is not None and ti is not None:
            t_cut = to[0] + (1.0 - metrics.TAIL_FRACTION) * (to[-1] - to[0])
            plt.figure(figsize=(8, 5))
            plt.plot(ti, -inn[:, 0], linewidth=1.2, label="capture plane")
            plt.plot(to, out[:, 0], linewidth=1.2, label="outlet")
            plt.axhline(MDOT_TARGET, color="k", linestyle="--", linewidth=1.0, label=f"target {MDOT_TARGET:g} kg/s")
            plt.axvspan(t_cut, to[-1], color="0.6", alpha=0.25,
                        label=f"convergence window (last {100 * metrics.TAIL_FRACTION:.0f}%)")
            tail = out[to >= t_cut, 0]
            lo, hi = min(tail.min(), MDOT_TARGET), max(tail.max(), MDOT_TARGET)
            pad = max(0.5 * (hi - lo), 0.02 * hi)
            plt.ylim(lo - pad, hi + pad)
            plt.xlabel("Iteration"); plt.ylabel("Mass flow (kg/s)"); plt.title("Mass Flow Convergence")
            plt.grid(True, linestyle="--", alpha=0.7); plt.legend(fontsize=8)
            plt.savefig(os.path.join(images, "convergence_mass_flow.png"), dpi=200, bbox_inches="tight")
            plt.close()

        # Initial residuals from solverInfo (columns "<field>_initial"), restarts keep the first value
        residuals = {}
        for path in sorted(glob.glob(os.path.join(case, "postProcessing", "residuals", "*", "solverInfo.dat")),
                           key=metrics._start_time):
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
                    t, v = residuals.setdefault(name, ([], []))
                    try:
                        if not t or float(parts[0]) > t[-1]:
                            v.append(float(parts[idx])); t.append(float(parts[0]))
                    except ValueError:
                        pass
        if residuals:
            plt.figure(figsize=(9, 6))
            for name, (t, v) in sorted(residuals.items()):
                plt.semilogy(t, v, label=name, linewidth=1.0)
            plt.xlabel("Iteration"); plt.ylabel("Initial Residual"); plt.title("Residual Convergence")
            plt.grid(True, which="both", linestyle="--", alpha=0.5); plt.legend()
            plt.savefig(os.path.join(images, "residuals.png"), dpi=200, bbox_inches="tight")
            plt.close()
    except Exception as e:
        print(f"Warning: could not plot convergence histories: {e}")

def post_process(case):
    name = os.path.basename(os.path.normpath(case))
    log = os.path.join(case, "log.pvbatch")
    rc = sh(f"LIBGL_ALWAYS_SOFTWARE=1 pvbatch --force-offscreen-rendering {os.path.abspath(__file__)} "
            f"post {os.path.join(case, name + '.foam')} {os.path.join(case, 'images')} > {log} 2>&1")
    if rc != 0:
        print(f"  post processing failed, last lines of {log}:")
        sh(f"tail -n 30 {log}")
    return rc == 0

# ----------------------------------------------------------------------------
# ParaView post processing (runs under pvbatch)
# ----------------------------------------------------------------------------

def paraview_post(foam, images):
    from paraview import servermanager
    from paraview.simple import (OpenDataFile, Calculator, Slice, Clip, Transform, Text,
                                 IntegrateVariables, CreateView, Show, ColorBy,
                                 GetColorTransferFunction, GetScalarBar, Render, SaveScreenshot, Delete)

    os.makedirs(images, exist_ok=True)
    case = os.path.dirname(os.path.abspath(foam))
    m = read_json(case)

    # Converged steady solution: the last written iteration
    reader = OpenDataFile(foam)
    reader.MeshRegions = ["internalMesh"]
    reader.CellArrays = ["p", "T", "U"]
    reader.UpdatePipelineInformation()
    t = float(latest_time(case))

    # Derived cell fields (perfect gas, same gamma / R as the 1D estimates)
    src = reader
    for name, fn in [
        ("Mach",     f"mag(U)/sqrt({GAMMA}*{R_GAS}*T)"),
        ("rho",      f"p/({R_GAS}*T)"),
        ("massFlux", "rho*U_X"),
        ("p0",       f"p*(1+{0.5 * (GAMMA - 1):g}*Mach*Mach)^{GAMMA / (GAMMA - 1):g}"),
        ("p0Flux",   "p0*massFlux"),
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

    def cut(origin, normal):
        s = Slice(Input=src)
        s.SliceType = "Plane"
        s.SliceType.Origin = origin
        s.SliceType.Normal = normal
        s.UpdatePipeline(t)
        return s

    def plane_stats(s):
        I = integrate(s)
        A, mdot = I["Area"], I["massFlux"]
        return {"area_m2": A, "mdot_kg_s": mdot, "mach_avg": I["Mach"] / A,
                "mach_max": s.CellData["Mach"].GetRange()[1], "T_avg": I["T"] / A,
                "p_avg": I["p"] / A, "p_max": s.CellData["p"].GetRange()[1],
                "p0_avg": I["p0Flux"] / mdot}                  # p0 mass-weighted

    x_lip, x_ef, L = m["x_lip_m"], m["x_engine_face_m"], m["duct_length_m"]
    stats = {}

    # Capture plane and engine face, and the front-view stations in between
    for prefix, x in (("cap", x_lip + X_OFFSET), ("ef", x_ef + X_OFFSET)):
        s = cut([x, 0, 0], [1, 0, 0])
        stats.update({f"{prefix}_{k}": v for k, v in plane_stats(s).items()})
        Delete(s)
    stations = []
    for f in SLICE_STATIONS:
        x = min(max(x_lip + f * L, x_lip + X_OFFSET), x_ef - X_OFFSET)
        s = cut([x, 0, 0], [1, 0, 0])
        stations.append((f, x, s))
    stats["stations"] = [{"x_over_L": f, "x_m": x, **plane_stats(s)} for f, x, s in stations]

    # Volume statistics over the S-duct only (capture plane to engine face, no extension)
    clip = Clip(Input=src)
    clip.ClipType = "Plane"
    clip.ClipType.Origin = [x_ef, 0.0, 0.0]
    clip.ClipType.Normal = [1.0, 0.0, 0.0]
    clip.Invert = 1                          # keep x < x_ef
    clip.UpdatePipeline(t)
    I = integrate(clip)
    V = I["Volume"]
    stats.update(duct_volume_m3=V, duct_mach_avg=I["Mach"] / V,
                 duct_mach_max=clip.CellData["Mach"].GetRange()[1],
                 duct_T_avg=I["T"] / V, duct_T_max=clip.CellData["T"].GetRange()[1],
                 duct_p_avg=I["p"] / V, duct_p_max=clip.CellData["p"].GetRange()[1])
    Delete(clip)

    stats["pt_recovery_duct"] = stats["ef_p0_avg"] / stats["cap_p0_avg"]
    stats["pt_recovery_total"] = m["pt_recovery_2d"] * stats["pt_recovery_duct"]
    stats["post_time"] = t
    m.update(stats)
    write_json(case, m)
    print(json.dumps(stats, indent=2))

    # ---- images ----
    aspect = VIEW_SIZE[0] / VIEW_SIZE[1]

    def view(position, focal, scale):
        rv = CreateView("RenderView")
        rv.ViewSize = VIEW_SIZE
        rv.ViewTime = t
        rv.OrientationAxesVisibility = 0
        rv.CameraPosition = position
        rv.CameraFocalPoint = focal
        rv.CameraViewUp = [0, 0, 1]
        rv.CameraParallelProjection = 1
        rv.CameraParallelScale = scale
        return rv

    def color(display, rv, array, title, lo, hi):
        ColorBy(display, ("CELLS", array))
        lut = GetColorTransferFunction(array)
        lut.RescaleTransferFunction(lo, hi)
        display.SetScalarBarVisibility(rv, True)
        bar = GetScalarBar(lut, rv)
        bar.WindowLocation = "Any Location"
        bar.Orientation = "Vertical"
        bar.Position = SCALAR_BAR["position"]
        bar.ScalarBarLength = SCALAR_BAR["length"]
        bar.ScalarBarThickness = SCALAR_BAR["thickness"]
        bar.TitleFontSize = SCALAR_BAR["title_size"]
        bar.LabelFontSize = SCALAR_BAR["label_size"]
        bar.TitleBold = 1
        n = SCALAR_BAR["n_labels"]
        bar.UseCustomLabels = 1
        bar.CustomLabels = [lo + (hi - lo) * i / (n - 1) for i in range(1, n - 1)]
        bar.AutomaticLabelFormat = 1
        bar.RangeLabelFormat = "%-#.4g"
        bar.Title = title
        bar.ComponentTitle = ""

    def save(rv, fname):
        Render(rv)
        SaveScreenshot(os.path.join(images, fname), rv, ImageResolution=VIEW_SIZE)
        Delete(rv)

    # Side view: y = 0 slice seen from -y, the slice fills the left ~85 %, colour bar on the right
    side = cut([0.0, Y_SLICE, 0.0], [0, 1, 0])
    b = side.GetDataInformation().GetBounds()
    scale = max(0.5 * (b[5] - b[4]) / 0.85, (b[1] - b[0]) / (0.84 * 2.0 * aspect))
    width = 2.0 * scale * aspect
    focal = [b[0] - 0.03 * width + 0.5 * width, 0.0, 0.5 * (b[4] + b[5])]
    side_view = lambda: view([focal[0], -1.0, focal[2]], focal, scale)
    for array, title, fname in [("Mach", "Mach", "slice-mach-side.png"),
                                ("p", "p [Pa]", "slice-pressure-side.png")]:
        rv = side_view()
        color(Show(side, rv), rv, array, title, *side.CellData[array].GetRange())
        save(rv, fname)
    rv = side_view()
    d = Show(side, rv)
    d.Representation = "Surface With Edges"
    ColorBy(d, None)
    save(rv, "mesh-side.png")

    # Front view: the station slices side by side, looking downstream (from upstream of the capture
    # plane), capture plane on the left, one colour range per field so the stations compare directly.
    # Looking along +x with z up, screen-right is -y, so slice i is moved to y = -i * pitch.
    bounds = [s.GetDataInformation().GetBounds() for _, _, s in stations]
    w_max = max(bb[3] - bb[2] for bb in bounds)
    h_max = max(bb[5] - bb[4] for bb in bounds)
    pitch = 1.25 * w_max
    moved = []
    for i, ((f, x, s), bb) in enumerate(zip(stations, bounds)):
        tr = Transform(Input=s)
        tr.Transform = "Transform"
        tr.Transform.Translate = [x_lip - x, -i * pitch - 0.5 * (bb[2] + bb[3]), -0.5 * (bb[4] + bb[5])]
        tr.UpdatePipeline(t)
        moved.append(tr)
    n = len(moved)
    row_w = (n - 1) * pitch + w_max
    scale = max(0.5 * h_max / 0.6, row_w / (0.82 * 2.0 * aspect))
    width = 2.0 * scale * aspect
    y_focal = -0.5 * (n - 1) * pitch - 0.075 * width          # row centred in the left ~85 %

    for array, title, fname in [("p", "p [Pa]", "slices-pressure-front.png"),
                                ("p0", "p0 [Pa]", "slices-total-pressure-front.png")]:
        lo = min(s.CellData[array].GetRange()[0] for _, _, s in stations)
        hi = max(s.CellData[array].GetRange()[1] for _, _, s in stations)
        rv = view([x_lip - 1.0, y_focal, 0.0], [x_lip, y_focal, 0.0], scale)
        for tr in moved:
            color(Show(tr, rv), rv, array, title, lo, hi)
        for i, f in enumerate(SLICE_STATIONS):
            label = Text(Text=f"x/L = {f:g}")
            ld = Show(label, rv)
            ld.WindowLocation = "Any Location"
            ld.FontSize = 40
            # screen position of panel i: offset (y_focal - y_i) to the right of the centre
            ld.Position = [0.5 + (y_focal + i * pitch) / width - 0.04, 0.5 - 0.5 * h_max / (2 * scale) - 0.12]
        title_text = Text(Text="Looking downstream: capture plane (left) to engine face (right)")
        td = Show(title_text, rv)
        td.WindowLocation = "Upper Center"
        td.FontSize = 40
        save(rv, fname)

if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "post":
        if len(sys.argv) < 4:
            print("usage: pvbatch run_inlet.py post <case.foam> <images_dir>")
            sys.exit(1)
        paraview_post(sys.argv[2], sys.argv[3])
        sys.exit(0)
    sys.exit(main())