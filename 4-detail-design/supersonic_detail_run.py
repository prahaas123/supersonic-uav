import csv
import glob
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.join(HERE, "..", "0-tools")
sys.path.insert(0, TOOLS_DIR)
import metrics     # noqa: E402  (_read_dat, window_stats, _start_time, TAIL_FRACTION)

# =====================================================================================================
# INPUTS
# =====================================================================================================
MACH      = 1.5
ALPHA_DEG = 0.0                  # angle of attack [deg]
BETA_DEG  = 0.0                  # sideslip [deg], > 0 = wind from starboard
CG        = (0.826, 0.0, 0.0)    # moment reference point / CG (x, y, z) [m], body axes

# =====================================================================================================
# ASSUMED VALUES
# =====================================================================================================
# --- atmosphere and gas
ALTITUDE_M    = 15000.0          # ISA: p_inf 12044.6 Pa, T_inf 216.65 K
GAMMA         = 1.4              # constant cp air, also used for the exhaust
R_GAS         = 287.05           # [J/kg/K]
SUTHERLAND_AS = 1.458e-6         # mu = As T^1.5 / (T + Ts)
SUTHERLAND_TS = 110.4            # [K]
PRANDTL_TURB  = 0.9              # turbulent Prandtl number (STAR-CCM+ default)
FREESTREAM_TURB_INTENSITY = 0.01     # STAR-CCM+ default
FREESTREAM_VISC_RATIO     = 10.0     # mu_t / mu, STAR-CCM+ default

# --- reference quantities (estimates from the CAD planform)
S_REF = 0.494                    # exposed wing planform area [m2]
C_REF = 0.503                    # exposed-wing mean chord [m]
B_REF = 1.107                    # span [m]

# --- engine (README section 1)
MDOT_AIR   = 0.2635              # engine-face air mass flow [kg/s]
MDOT_FUEL  = 0.0216              # [kg/s], exhaust mass flow = MDOT_AIR + MDOT_FUEL
T0_EXHAUST = 2250.0              # exhaust total temperature [K] (assumed)
P_EXHAUST  = 15065.0             # exhaust static pressure [Pa] (from the 343.75 N thrust estimate)
EXHAUST_TURB_INTENSITY = 0.01
EXHAUST_VISC_RATIO     = 10.0

# --- walls
WALL_TEMPERATURE = None          # None = adiabatic, a value [K] = isothermal walls

# =====================================================================================================
# GEOMETRY AND MESH (the mesh does not depend on the flow condition and is reused between runs)
# =====================================================================================================
GEOMETRY   = os.path.join(HERE, "uav.stl")     # ASCII, one solid per named face, metres
HALF_MODEL = None                # None: half model (symmetry plane y = 0) when BETA_DEG == 0, else full
DOMAIN     = {"x": (-4.75, 7.5), "y": (-4.5, 4.5), "z": (-4.5, 4.5)}   # extended to whole BASE_SIZE cells
BASE_SIZE  = 0.4                 # background (largest) cell [m]; all sizes below are BASE_SIZE / 2^n

# surface (target, minimum) cell size [m]
SURFACE_SIZES = {
    "airframe":     (0.003125,  0.0015625),   # STAR-CCM+ 6.25 / 1.5625 mm
    "inlet_lip":    (0.000390625, 0.000390625),  # STAR-CCM+ 0.5 / 0.25 mm, nearest BASE_SIZE / 2^n
    "inlet_duct":   (0.0015625, 0.00078125),
    "engine_ports": (0.0015625, 0.00078125),  # engine face + exhaust
}
FEATURE_EDGE_SIZE = 0.0015625    # cell size along sharp CAD edges [m]
N_CELLS_BETWEEN_LEVELS = 3
RESOLVE_FEATURE_ANGLE  = 30.0    # [deg] surfaces curving more than this get the minimum size

# volume refinement boxes, body axes: name -> (min corner, max corner, cell size [m])
# Extended upward and rearward of the STAR-CCM+ boxes to follow the shocks (Mach angle 41.8 deg at
# Mach 1.5) and the wake / plume further, plus a 6.25 mm box around the vehicle. Vehicle extent:
# x -0.246 .. 1.425, |y| <= 0.554, z -0.07 .. 0.30. Estimated ~3.5 M box cells on the half model
# (~5 M with surface refinement and prism layers), against ~0.75 M box cells with the STAR-CCM+ boxes.
REFINEMENT_BOXES = {
    "refineFar":   ((-0.80, -1.40,  -1.10),  (6.00, 1.40,  2.40), 0.05),      # STAR: x 4.00, z 1.40
    "refineMid":   ((-0.50, -0.85,  -0.40),  (3.60, 0.85,  1.30), 0.025),     # STAR: x 2.40, z 0.65
    "refineNear":  ((-0.32, -0.65,  -0.12),  (2.40, 0.65,  0.65), 0.0125),    # STAR: x 1.60, z 0.36
    "refineBody":  ((-0.30, -0.60,  -0.08),  (1.75, 0.60,  0.34), 0.00625),   # new
    "refineInlet": (( 0.62, -0.07,  -0.045), (1.06, 0.07,  0.20), 0.003125),  # STAR: z 0.14 (lip shock / spillage)
    "refinePlume": (( 1.38, -0.15,  -0.15),  (2.60, 0.15,  0.15), 0.00625),   # STAR: x 1.80, +/-0.12
}

# prism layers: airframe, then inlet lip + duct walls (engine face and exhaust get none)
PRISM_LAYERS         = 8
PRISM_FIRST_CELL     = 6.0e-4    # [m]
PRISM_TOTAL          = 0.010     # [m]
PRISM_LAYERS_DUCT    = 6
PRISM_TOTAL_DUCT     = 0.005     # [m]
PRISM_MIN_THICKNESS  = 1.0e-4    # snappyHexMesh removes layers thinner than this [m]

LOCATION_IN_MESH = (-4.0123, 2.0123, 2.0123)   # any point in the fluid with y > 0

# =====================================================================================================
# RUN CONTROL
# =====================================================================================================
CASE_TEMPLATE    = os.path.join(HERE, "detail_template_supersonic")
NP               = 50            # MPI processes
END_TIME         = 0.02          # [s] cap, ~12 flow-throughs of the 1.67 m body
WRITE_INTERVAL   = END_TIME / 10 # [s] periodic writes so a timed-out job can be resumed
MAX_COURANT      = 0.5
MONITOR_INTERVAL = 10            # time steps between monitor samples
RESUME           = False         # continue a case whose processor* directories survived a timed-out job
POLL_S           = 120           # [s] between stopping-criterion checks while the solver runs

# Stopping: all of these must stay within the given band (max - min) over the final window
ASYMPTOTE = {
    "CD":            1.0e-4,     # 1 drag count
    "CL":            5.0e-4,
    "Cm_pitch":      5.0e-4,
    "p_engine_face": 50.0,       # [Pa], the free variable at the engine face (mass flow is imposed)
}
ASYMPTOTE_WINDOW_S   = 2.0e-3    # [s] ~0.5 body flow-throughs
ASYMPTOTE_MIN_TIME_S = 6.0e-3    # [s] no check before this time

WALL_PATCHES = ("airframe", "inlet_lip", "inlet_duct")

# =====================================================================================================
# Driver
# =====================================================================================================

def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "report":
        case = sys.argv[2]
        flow = read_json(case).get("flow") or flow_conditions(geometry_info(GEOMETRY), half_model())
        return finish(case, flow)

    half = half_model()
    geo = geometry_info(GEOMETRY)
    flow = flow_conditions(geo, half)
    print_conditions(flow)

    # 1. Mesh, reused while the mesh settings and the geometry are unchanged
    mesh_dir = os.path.join(".", f"detail_mesh_{'half' if half else 'full'}")
    settings = mesh_settings(half)
    if read_json(mesh_dir).get("mesh_settings") == settings and \
            os.path.isdir(os.path.join(mesh_dir, "constant", "polyMesh")):
        print(f"[mesh] reusing {mesh_dir}")
    else:
        print(f"[mesh] building {mesh_dir} ...")
        t0 = time.perf_counter()
        prepare_mesh(mesh_dir, settings, half)
        if not mesh(mesh_dir):
            print("Error: meshing failed")
            return 1
        write_json(mesh_dir, {"mesh_settings": settings, "mesh_time_s": round(time.perf_counter() - t0, 1),
                              **mesh_stats(mesh_dir)})
        print(f"[mesh] done in {time.perf_counter() - t0:.0f} s")

    # 2. Case
    case = os.path.join(".", f"detail_M{MACH:g}_a{ALPHA_DEG:g}_b{BETA_DEG:g}")
    resume = RESUME and os.path.isdir(os.path.join(case, "processor0"))
    print(f"\n{'=' * 60}\n {os.path.basename(case)}{' (resumed)' if resume else ''}\n{'=' * 60}")
    if not resume:
        print("[1/4] Copying the mesh and writing the flow condition ...")
        prepare_case(case, mesh_dir)
    write_flow_settings(case, flow)
    write_json(case, {**read_json(case), "case": os.path.abspath(case), "flow": flow,
                      "mesh": read_json(mesh_dir)})

    # 3. Solve
    print("[2/4] Solving (rhoCentralFoam) ...")
    t0 = time.perf_counter()
    ok, stop_reason = solve(case, flow, resume)
    m = read_json(case)
    m.update(solve_time_s=round(time.perf_counter() - t0, 1), stop_reason=stop_reason)
    write_json(case, m)
    if not ok:
        print("Error: solver failed, see log.rhoCentralFoam")
        m["status"] = "solve_failed"
        write_json(case, m)
        return 1

    # 4. Reports, plots, images
    return finish(case, flow)


def finish(case, flow):
    print("[3/4] Reports and monitor plots ...")
    hist = histories(case, flow)
    if not hist:
        print("Error: no monitor data in postProcessing")
        return 1
    m = read_json(case)
    m.update(reports(case, flow, hist))
    write_json(case, m)
    write_history_csv(case, hist)
    plot_monitors(case, flow, hist, m)
    plot_residuals(case)

    print("[4/4] ParaView scenes ...")
    post_process(case)
    m = read_json(case)
    text = format_summary(m)
    print(text)
    with open(os.path.join(case, "summary.txt"), "w") as fh:
        fh.write(text)
    sh(f"rm -rf {case}/processor*")
    return 0 if m.get("converged") else 2

# =====================================================================================================
# Flow condition
# =====================================================================================================

def half_model():
    return (BETA_DEG == 0.0) if HALF_MODEL is None else bool(HALF_MODEL)


def isa_atmosphere(h):
    g0, T_sl, p_sl, lapse, h_tp, R = 9.80665, 288.15, 101325.0, -0.0065, 11000.0, 287.05287
    T = T_sl + lapse * min(h, h_tp)
    p = p_sl * (T / T_sl) ** (-g0 / (lapse * R))
    if h > h_tp:
        p *= math.exp(-g0 * (h - h_tp) / (R * T))
    return T, p


def sutherland(T):
    return SUTHERLAND_AS * T ** 1.5 / (T + SUTHERLAND_TS)


def wind_axes(alpha_deg, beta_deg):
    a, b = math.radians(alpha_deg), math.radians(beta_deg)
    drag = (math.cos(a) * math.cos(b), -math.sin(b), math.sin(a) * math.cos(b))   # along the freestream
    lift = (-math.sin(a), 0.0, math.cos(a))
    side = (math.cos(a) * math.sin(b), math.cos(b), math.sin(a) * math.sin(b))
    return drag, lift, side


def exhaust_mach(mdot, area, p, T0):
    # Exit Mach number for a mass flux at static pressure p and total temperature T0
    flux = mdot / area
    f = lambda M: p * M * math.sqrt(GAMMA * (1.0 + 0.5 * (GAMMA - 1.0) * M * M) / (R_GAS * T0))
    lo, hi = 0.0, 20.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) < flux else (lo, mid)
    return 0.5 * (lo + hi)


def turbulence(U, rho, T, intensity, visc_ratio):
    k = 1.5 * (intensity * U) ** 2
    nut = visc_ratio * sutherland(T) / rho
    return k, k / nut, nut


def flow_conditions(geo, half):
    T, p = isa_atmosphere(ALTITUDE_M)
    a = math.sqrt(GAMMA * R_GAS * T)
    U = MACH * a
    rho = p / (R_GAS * T)
    drag, lift, side = wind_axes(ALPHA_DEG, BETA_DEG)
    k, omega, nut = turbulence(U, rho, T, FREESTREAM_TURB_INTENSITY, FREESTREAM_VISC_RATIO)
    iso = 1.0 + 0.5 * (GAMMA - 1.0) * MACH ** 2
    sym = 2.0 if half else 1.0

    # 1D exhaust state (full exhaust area and mass flow)
    mdot_ex = MDOT_AIR + MDOT_FUEL
    M_ex = exhaust_mach(mdot_ex, geo["area"]["exhaust"], P_EXHAUST, T0_EXHAUST)
    T_ex = T0_EXHAUST / (1.0 + 0.5 * (GAMMA - 1.0) * M_ex ** 2)
    U_ex = M_ex * math.sqrt(GAMMA * R_GAS * T_ex)
    rho_ex = P_EXHAUST / (R_GAS * T_ex)
    k_ex, omega_ex, _ = turbulence(U_ex, rho_ex, T_ex, EXHAUST_TURB_INTENSITY, EXHAUST_VISC_RATIO)
    n_ex = geo["normal"]["exhaust"]               # outward from the body = into the fluid

    return dict(
        mach=MACH, alpha_deg=ALPHA_DEG, beta_deg=BETA_DEG, cg=list(CG), altitude_m=ALTITUDE_M,
        half_model=half, symmetry_factor=sym, gamma=GAMMA, R=R_GAS,
        sutherland_As=SUTHERLAND_AS, sutherland_Ts=SUTHERLAND_TS,
        p_inf=p, T_inf=T, rho_inf=rho, a_inf=a, V_inf=U, mu_inf=sutherland(T),
        U_vec=[U * c for c in drag], drag_dir=list(drag), lift_dir=list(lift), side_dir=list(side),
        q_inf=0.5 * GAMMA * p * MACH ** 2, p0_inf=p * iso ** (GAMMA / (GAMMA - 1.0)), T0_inf=T * iso,
        k_inf=k, omega_inf=omega, nut_inf=nut,
        S_ref=S_REF, c_ref=C_REF, b_ref=B_REF,
        mdot_air=MDOT_AIR, mdot_fuel=MDOT_FUEL, mdot_exhaust=mdot_ex,
        T0_exhaust=T0_EXHAUST, p_exhaust=P_EXHAUST,
        engine_face_area=geo["area"]["engine_face"], exhaust_area=geo["area"]["exhaust"],
        exhaust_mach_1d=M_ex, exhaust_T_1d=T_ex, exhaust_U_1d=U_ex, exhaust_rho_1d=rho_ex,
        exhaust_supersonic=M_ex >= 1.0, exhaust_U_vec=[U_ex * c for c in n_ex],
        exhaust_thrust_1d=mdot_ex * U_ex + (P_EXHAUST - p) * geo["area"]["exhaust"],
        k_exhaust=k_ex, omega_exhaust=omega_ex,
        wall_temperature=WALL_TEMPERATURE,
    )


def print_conditions(f):
    print(f"Mach {f['mach']:g} | alpha {f['alpha_deg']:g} deg | beta {f['beta_deg']:g} deg | "
          f"CG ({', '.join(f'{c:g}' for c in f['cg'])}) m | {'half' if f['half_model'] else 'full'} model")
    print(f"  {f['altitude_m'] / 1000:g} km ISA: p {f['p_inf']:.1f} Pa, T {f['T_inf']:.2f} K, "
          f"rho {f['rho_inf']:.5f} kg/m3, V {f['V_inf']:.1f} m/s, q {f['q_inf']:.0f} Pa, p0 {f['p0_inf']:.0f} Pa")
    print(f"  engine face {f['mdot_air']:g} kg/s over {f['engine_face_area'] * 1e4:.2f} cm2 | exhaust "
          f"{f['mdot_exhaust']:g} kg/s, T0 {f['T0_exhaust']:g} K, p {f['p_exhaust']:g} Pa -> 1D Mach "
          f"{f['exhaust_mach_1d']:.3f}, U {f['exhaust_U_1d']:.0f} m/s, gross thrust {f['exhaust_thrust_1d']:.1f} N")
    if not f["exhaust_supersonic"]:
        print("  WARNING: the 1D exhaust is subsonic, its static pressure is left free (zeroGradient)")

# =====================================================================================================
# Geometry
# =====================================================================================================

def read_stl(path):
    # Triangles per solid of an ASCII STL: {name: (n, 3, 3) array}
    with open(path, "rb") as fh:
        head = fh.read(512)
    if not head.lstrip().lower().startswith(b"solid") or b"facet" not in head:
        raise SystemExit(f"{path} is not an ASCII STL; export it as ASCII with one solid per named face")
    solids, name, verts = {}, None, []
    with open(path) as fh:
        for line in fh:
            w = line.split()
            if not w:
                continue
            if w[0] == "solid":
                name, verts = (w[1] if len(w) > 1 else "solid"), []
            elif w[0] == "vertex":
                verts.append(w[1:4])
            elif w[0] == "endsolid":
                tri = np.asarray(verts, dtype=float).reshape(-1, 3, 3)
                solids[name] = np.concatenate([solids[name], tri]) if name in solids else tri
    return solids


def geometry_info(path):
    solids = read_stl(path)
    area, normal, lo, hi = {}, {}, {}, {}
    for name, p in solids.items():
        n = 0.5 * np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        area[name] = float(np.linalg.norm(n, axis=1).sum())
        s = n.sum(axis=0)
        normal[name] = list(s / np.linalg.norm(s)) if np.linalg.norm(s) > 0 else [0.0, 0.0, 0.0]
        lo[name], hi[name] = list(p.reshape(-1, 3).min(axis=0)), list(p.reshape(-1, 3).max(axis=0))
    for need in ("airframe", "inlet_lip", "inlet_duct", "engine_face", "exhaust"):
        if need not in solids:
            raise SystemExit(f"{path} has no 'solid {need}' (found: {', '.join(solids)})")
    return {"area": area, "normal": normal, "min": lo, "max": hi}

# =====================================================================================================
# Mesh
# =====================================================================================================

def level(size):
    n = math.log2(BASE_SIZE / size)
    if abs(n - round(n)) > 1e-6:
        print(f"  note: {size * 1e3:g} mm is not BASE_SIZE / 2^n, using {BASE_SIZE / 2 ** round(n) * 1e3:g} mm")
    return int(round(n))


def mesh_settings(half):
    with open(GEOMETRY, "rb") as fh:
        geo_hash = hashlib.md5(fh.read()).hexdigest()
    s = dict(
        geometry=os.path.basename(GEOMETRY), geometry_md5=geo_hash, half_model=half, domain=DOMAIN,
        base_size=BASE_SIZE, surface_sizes=SURFACE_SIZES, feature_edge_size=FEATURE_EDGE_SIZE,
        n_cells_between_levels=N_CELLS_BETWEEN_LEVELS, resolve_feature_angle=RESOLVE_FEATURE_ANGLE,
        refinement_boxes=REFINEMENT_BOXES,
        prism=[PRISM_LAYERS, PRISM_FIRST_CELL, PRISM_TOTAL, PRISM_LAYERS_DUCT, PRISM_TOTAL_DUCT,
               PRISM_MIN_THICKNESS],
        location_in_mesh=LOCATION_IN_MESH,
    )
    return json.loads(json.dumps(s))     # tuples -> lists, comparable with the stored copy


def fmt_vec(v):
    return "(" + " ".join(f"{c:.8g}" for c in v) + ")"


def write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)


def sub(path, pattern, repl):
    with open(path) as fh:
        text, n = re.subn(pattern, repl, fh.read(), flags=re.M)
    if n == 0:
        raise ValueError(f"pattern {pattern!r} not found in {path}")
    with open(path, "w") as fh:
        fh.write(text)


def prepare_mesh(mesh_dir, s, half):
    if os.path.exists(mesh_dir):
        shutil.rmtree(mesh_dir)
    shutil.copytree(CASE_TEMPLATE, mesh_dir, ignore=shutil.ignore_patterns("*.foam"))
    path = lambda *p: os.path.join(mesh_dir, *p)
    os.makedirs(path("constant", "triSurface"), exist_ok=True)
    shutil.copy(GEOMETRY, path("constant", "triSurface", "uav.stl"))

    # Background box, extended outward to whole BASE_SIZE cells; y = 0 is a grid plane
    (x0, x1), (y0, y1), (z0, z1) = DOMAIN["x"], DOMAIN["y"], DOMAIN["z"]
    snap = lambda v, up: (math.ceil if up else math.floor)(round(v / BASE_SIZE, 9)) * BASE_SIZE
    x0, x1, z0, z1 = snap(x0, False), snap(x1, True), snap(z0, False), snap(z1, True)
    y0, y1 = (0.0 if half else snap(y0, False)), snap(y1, True)
    n = [round((x1 - x0) / BASE_SIZE), round((y1 - y0) / BASE_SIZE), round((z1 - z0) / BASE_SIZE)]
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    verts = "".join(f"    ({x:.6g} {y:.6g} {z:.6g})\n" for z in (z0, z1) for x, y in corners)
    outer = "(0 4 7 3) (1 2 6 5) (3 7 6 2) (0 3 2 1) (4 5 6 7)"
    boundary = (f"    farfield {{ type patch; faces ({outer}); }}\n"
                "    symmetry { type symmetry; faces ((0 1 5 4)); }\n" if half else
                f"    farfield {{ type patch; faces ({outer} (0 1 5 4)); }}\n")
    write_text(path("system", "include", "domain"),
               "// Written by supersonic_detail_run.py\n"
               f"vertices\n(\n{verts});\n\n"
               f"blocks\n(\n    hex (0 1 2 3 4 5 6 7) ({n[0]} {n[1]} {n[2]}) simpleGrading (1 1 1)\n);\n\n"
               f"boundary\n(\n{boundary});\n")

    lv = lambda key: f"({level(SURFACE_SIZES[key][0])} {level(SURFACE_SIZES[key][1])})"
    write_text(path("system", "include", "meshVars"),
               "// Written by supersonic_detail_run.py\n"
               f"nCellsBetweenLevels {N_CELLS_BETWEEN_LEVELS};\n"
               f"featureLevel        {level(FEATURE_EDGE_SIZE)};\n"
               f"resolveFeatureAngle {RESOLVE_FEATURE_ANGLE:g};\n"
               f"airframeLevel       {lv('airframe')};\n"
               f"inletLipLevel       {lv('inlet_lip')};\n"
               f"inletDuctLevel      {lv('inlet_duct')};\n"
               f"enginePortsLevel    {lv('engine_ports')};\n"
               f"nLayers             {PRISM_LAYERS};\n"
               f"nLayersDuct         {PRISM_LAYERS_DUCT};\n"
               f"firstLayerThickness {PRISM_FIRST_CELL:.6g};\n"
               f"layerThickness      {PRISM_TOTAL:.6g};\n"
               f"layerThicknessDuct  {PRISM_TOTAL_DUCT:.6g};\n"
               f"minLayerThickness   {PRISM_MIN_THICKNESS:.6g};\n"
               f"locationInMesh      {fmt_vec(LOCATION_IN_MESH)};\n")
    write_text(path("system", "include", "refinementBoxes"),
               "// Written by supersonic_detail_run.py\n" + "".join(
                   f"{k} {{ type searchableBox; min {fmt_vec(a)}; max {fmt_vec(b)}; }}\n"
                   for k, (a, b, _) in REFINEMENT_BOXES.items()))
    write_text(path("system", "include", "refinementRegions"),
               "// Written by supersonic_detail_run.py\n" + "".join(
                   f"{k} {{ mode inside; levels ((1e15 {level(sz)})); }}\n"
                   for k, (_, _, sz) in REFINEMENT_BOXES.items()))
    sub(path("system", "decomposeParDict"), r"^numberOfSubdomains\s+\d+;", f"numberOfSubdomains  {NP};")
    if half and LOCATION_IN_MESH[1] <= 0:
        raise SystemExit("LOCATION_IN_MESH must have y > 0 on a half model")


def sh(cmd):
    return subprocess.run(cmd, shell=True, executable="/bin/bash").returncode


def run_steps(case, steps):
    for app, cmd in steps:
        print(f"  -> {cmd}   (log.{app})")
        rc = sh(f"{cmd} > {case}/log.{app} 2>&1")
        if rc != 0:
            print(f"  {app} failed (exit {rc}), last lines of log.{app}:")
            sh(f"tail -n 30 {case}/log.{app}")
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
        sh(f"checkMesh -constant -case {c} > {c}/log.checkMesh 2>&1")   # quality report, reference only
        print_layer_table(c)
    return ok and os.path.isdir(os.path.join(c, "constant", "polyMesh"))


def print_layer_table(c):
    # Prism-layer coverage per patch from the end of the snappyHexMesh log
    try:
        lines = open(os.path.join(c, "log.snappyHexMesh")).read().splitlines()
    except OSError:
        return
    start = max((i for i, l in enumerate(lines) if l.strip().startswith("patch") and "layers" in l), default=None)
    if start is not None:
        print("  prism layers:")
        for l in lines[start:start + 12]:
            if not l.strip():
                break
            print("   " + l)


def mesh_stats(c):
    out = {}
    try:
        text = open(os.path.join(c, "log.checkMesh")).read()
        for key in ("cells", "faces", "points"):
            m = re.search(rf"^\s*{key}:\s+(\d+)", text, flags=re.M)
            if m:
                out[f"mesh_{key}"] = int(m.group(1))
        m = re.search(r"Mesh non-orthogonality Max:\s+(\S+)", text)
        out["max_non_orthogonality"] = float(m.group(1)) if m else None
        m = re.search(r"Max skewness = (\S+)", text)
        out["max_skewness"] = float(m.group(1)) if m else None
        out["checkMesh_ok"] = "Mesh OK." in text
    except OSError:
        pass
    return out

# =====================================================================================================
# Case
# =====================================================================================================

def prepare_case(case, mesh_dir):
    if os.path.exists(case):
        shutil.rmtree(case)
    shutil.copytree(mesh_dir, case, ignore=shutil.ignore_patterns(
        "processor*", "log.*", "*.foam", "metrics.json", "postProcessing"))
    open(os.path.join(case, os.path.basename(os.path.normpath(case)) + ".foam"), "w").close()


def write_flow_settings(case, f):
    path = lambda *p: os.path.join(case, *p)
    sym = f["symmetry_factor"]
    v = lambda x: f"{x:.8g}"
    write_text(path("0", "include", "flowVars"), "\n".join([
        "// Written by supersonic_detail_run.py",
        f"Uinf            {fmt_vec(f['U_vec'])};",
        f"pInf            {v(f['p_inf'])};",
        f"Tinf            {v(f['T_inf'])};",
        f"rhoInf          {v(f['rho_inf'])};",
        f"kinf            {v(f['k_inf'])};",
        f"omegaInf        {v(f['omega_inf'])};",
        f"nutInf          {v(f['nut_inf'])};",
        f"Prt             {v(PRANDTL_TURB)};",
        f"mdotEngineFace  {v(f['mdot_air'] / sym)};      // per simulated half" if sym > 1 else
        f"mdotEngineFace  {v(f['mdot_air'])};",
        f"T0inf           {v(f['T0_inf'])};",
        f"mdotExhaust     {v(f['mdot_exhaust'] / sym)};",
        f"T0Exhaust       {v(f['T0_exhaust'])};",
        f"gammaExhaust    {v(GAMMA)};",
        f"UExhaust        {fmt_vec(f['exhaust_U_vec'])};",
        f"TExhaust        {v(f['exhaust_T_1d'])};",
        f"rhoExhaust      {v(f['exhaust_rho_1d'])};",
        f"kExhaust        {v(f['k_exhaust'])};",
        f"omegaExhaust    {v(f['omega_exhaust'])};", ""]))
    write_text(path("0", "include", "wallThermal"),
               "type            zeroGradient;\n" if WALL_TEMPERATURE is None else
               f"type            fixedValue;\nvalue           uniform {WALL_TEMPERATURE:.6g};\n")
    write_text(path("0", "include", "exhaustPressure"),
               f"type            fixedValue;\nvalue           uniform {P_EXHAUST:.8g};\n"
               if f["exhaust_supersonic"] else "type            zeroGradient;\n")
    write_text(path("constant", "include", "gasVars"), "\n".join([
        "// Written by supersonic_detail_run.py",
        f"gasMolWeight    {v(8314.46261815324 / R_GAS)};",
        f"gasCp           {v(GAMMA * R_GAS / (GAMMA - 1.0))};",
        f"sutherlandAs    {v(SUTHERLAND_AS)};",
        f"sutherlandTs    {v(SUTHERLAND_TS)};", ""]))
    write_text(path("system", "include", "caseVars"), "\n".join([
        "// Written by supersonic_detail_run.py",
        f"tEnd            {v(END_TIME)};",
        f"tWrite          {v(WRITE_INTERVAL)};",
        f"maxCourant      {v(MAX_COURANT)};",
        f"monitorInterval {MONITOR_INTERVAL};",
        f"cg              {fmt_vec(CG)};",
        f"rhoInfRef       {v(f['rho_inf'])};", ""]))
    sub(path("system", "controlDict"), r"^stopAt\s+\w+;", "stopAt          endTime;")
    sub(path("system", "decomposeParDict"), r"^numberOfSubdomains\s+\d+;", f"numberOfSubdomains  {NP};")

# =====================================================================================================
# Solve with the asymptotic stopping criteria
# =====================================================================================================

def solve(case, flow, resume):
    if not resume and not run_steps(case, [("decomposePar", f"decomposePar -force -case {case}")]):
        return False, "decompose_failed"

    print(f"  -> mpirun -np {NP} rhoCentralFoam -parallel   (log.rhoCentralFoam)")
    log = open(os.path.join(case, "log.rhoCentralFoam"), "a")
    proc = subprocess.Popen(f"mpirun -np {NP} rhoCentralFoam -parallel -case {case}", shell=True,
                            executable="/bin/bash", stdout=log, stderr=subprocess.STDOUT)
    reason = "end_time"
    while True:
        try:
            proc.wait(timeout=POLL_S)
            break
        except subprocess.TimeoutExpired:
            pass
        if reason != "end_time":
            continue
        try:
            status = asymptote_status(histories(case, flow))
        except Exception as e:          # files mid-write, retry at the next poll
            print(f"  (monitor read failed: {e})")
            continue
        if not status:
            continue
        t = max(s["t"] for s in status.values())
        print(f"  t = {t:.5g} s | " + " | ".join(
            f"{k} band {s['width']:.3g}{' ok' if s['ok'] else ''}" for k, s in status.items()), flush=True)
        if all(s["ok"] for s in status.values()):
            print("  asymptotic stopping criteria met, writing and stopping")
            sub(os.path.join(case, "system", "controlDict"), r"^stopAt\s+\w+;", "stopAt          writeNow;")
            reason = "asymptotic_criteria"
    log.close()
    if proc.returncode != 0:
        sh(f"tail -n 30 {case}/log.rhoCentralFoam")
        return False, "solver_error"

    ok = run_steps(case, [("reconstructPar", f"reconstructPar -latestTime -case {case}")])
    sub(os.path.join(case, "system", "controlDict"), r"^stopAt\s+\w+;", "stopAt          endTime;")
    sh(f"rm -rf {case}/processor*")
    return ok and latest_time(case) is not None, reason


def latest_time(case):
    times = []
    for d in os.listdir(case):
        try:
            if float(d) > 0 and os.path.isdir(os.path.join(case, d)):
                times.append(d)
        except ValueError:
            pass
    return max(times, key=float) if times else None


def asymptote_status(hist):
    out = {}
    for name, tol in ASYMPTOTE.items():
        if name not in hist:
            continue
        t, y = hist[name]
        if len(t) < 2:
            continue
        mask = t >= t[-1] - ASYMPTOTE_WINDOW_S
        width = float(np.ptp(y[mask]))
        ok = t[-1] >= ASYMPTOTE_MIN_TIME_S and t[-1] - t[0] >= ASYMPTOTE_WINDOW_S and width <= tol
        out[name] = {"t": float(t[-1]), "width": width, "tol": tol, "ok": bool(ok)}
    return out

# =====================================================================================================
# Monitors (time histories, complete aircraft)
# =====================================================================================================

def read_columns(case, fo, fname, min_cols=2):
    # (time, columns) from postProcessing/<fo>/*/<fname>, column names from the "# Time" header
    t, data = metrics._read_dat(case, fo, fname, min_cols)
    if t is None:
        return None, None, []
    names = []
    for path in sorted(glob.glob(os.path.join(case, "postProcessing", fo, "*", fname))):
        with open(path) as fh:
            for line in fh:
                if line.startswith("# Time"):
                    names = line.lstrip("#").split()[1:]
                    break
    return t, data, names


def force_columns(case, fo, fname):
    # total, pressure, viscous vectors from a forces function object file
    t, d, names = read_columns(case, fo, fname, 4)
    if t is None:
        return None
    idx = lambda key, j: names.index(key) if key in names else j
    out = {"t": t}
    for k, (part, j) in enumerate((("total", 0), ("pressure", 3), ("viscous", 6))):
        cols = [idx(f"{part}_{a}", j + i) for i, a in enumerate("xyz")]
        out[part] = d[:, cols] if d.shape[1] > max(cols) else None
    return out


def histories(case, f):
    S = f["symmetry_factor"]
    half = f["half_model"]
    qS = f["q_inf"] * S_REF
    a, b = math.radians(ALPHA_DEG), math.radians(BETA_DEG)
    ca, sa, cb, sb = math.cos(a), math.sin(a), math.cos(b), math.sin(b)
    h = {}

    def full(vec):
        # complete aircraft: double x and z, drop y (a half model has no net side force / roll / yaw)
        v = vec * S
        if half:
            v[:, 1] = 0.0
        return v

    def full_moment(vec):
        v = vec * S
        if half:
            v[:, 0] = 0.0
            v[:, 2] = 0.0
        return v

    F = force_columns(case, "forces", "force.dat")
    M = force_columns(case, "forces", "moment.dat")
    if F is None:
        return {}
    t = F["t"]
    Fb = full(F["total"])
    Fx, Fy, Fz = Fb[:, 0], Fb[:, 1], Fb[:, 2]
    for i, ax in enumerate("xyz"):
        h[f"F{ax}_body"] = (t, Fb[:, i])
    drag = Fx * ca * cb - Fy * sb + Fz * sa * cb
    lift = -Fx * sa + Fz * ca
    side = Fx * ca * sb + Fy * cb + Fz * sa * sb
    h.update(Drag=(t, drag), Lift=(t, lift), SideForce=(t, side),
             CD=(t, drag / qS), CL=(t, lift / qS), CY=(t, side / qS),
             L_over_D=(t, np.divide(lift, drag, out=np.zeros_like(lift), where=np.abs(drag) > 1e-12)))
    if M is not None:
        Mc = full_moment(M["total"])
        n = min(len(M["t"]), len(Mc))
        tm = M["t"][:n]
        for i, ax in enumerate("xyz"):
            h[f"M{ax}_cg"] = (tm, Mc[:n, i])
        # aircraft sign convention: Cm > 0 nose up, Cl > 0 starboard wing down, Cn > 0 nose to starboard
        h["Cm_pitch"] = (tm, Mc[:n, 1] / (qS * C_REF))
        h["Cl_roll"] = (tm, -Mc[:n, 0] / (qS * B_REF))
        h["Cn_yaw"] = (tm, -Mc[:n, 2] / (qS * B_REF))

    # Engine face and exhaust
    def sfv(fo, col=0, scale=1.0):
        tt, d, _ = read_columns(case, fo, "surfaceFieldValue.dat")
        return None if tt is None or d.shape[1] <= col else (tt, d[:, col] * scale)

    for key, fo, col, scale in [
            ("mdot_engine_face", "mdot_engine_face", 0, S),
            ("mdot_exhaust", "mdot_exhaust", 0, -S),          # positive = into the domain
            ("p_engine_face", "engine_face_average", 0, 1.0),
            ("Mach_engine_face", "engine_face_average", 1, 1.0),
            ("T_engine_face", "engine_face_average", 2, 1.0),
            ("p_exhaust", "exhaust_average", 0, 1.0),
            ("Mach_exhaust", "exhaust_average", 1, 1.0),
            ("T_exhaust", "exhaust_average", 2, 1.0)]:
        r = sfv(fo, col, scale)
        if r:
            h[key] = r
    r = sfv("engine_face_p0")
    if r:
        h["PressureRecovery_engine_face"] = (r[0], r[1] / f["p0_inf"])

    # Stream thrust: integral of p n + rho u (u.n) over the engine face / exhaust, n out of the fluid.
    # Fx_net = walls + engine face + exhaust = net axial force on the vehicle with its engine stream tube
    # (> 0 net drag, < 0 net thrust); same as STAR's Fx_body + StreamThrust_face - StreamThrust_exhaust.
    for port in ("engine_face", "exhaust"):
        P = force_columns(case, f"forces_{port}", "force.dat")
        tt, mom, _ = read_columns(case, f"{port}_momentum", "surfaceFieldValue.dat", 4)
        if P is None or tt is None or P["pressure"] is None:
            continue
        pf = full(P["pressure"])
        mf = full(mom[:, :3].copy())
        I = np.column_stack([np.interp(t, P["t"], pf[:, i]) + np.interp(t, tt, mf[:, i]) for i in range(3)])
        h[f"I_{port}_x"] = (t, I[:, 0])
        h[f"I_{port}_z"] = (t, I[:, 2])
    if "I_engine_face_x" in h and "I_exhaust_x" in h:
        h["StreamThrust_engine_face"] = (t, h["I_engine_face_x"][1])
        h["StreamThrust_exhaust"] = (t, -h["I_exhaust_x"][1])
        fx_net = Fx + h["I_engine_face_x"][1] + h["I_exhaust_x"][1]
        fz_net = Fz + h["I_engine_face_z"][1] + h["I_exhaust_z"][1]
        h["Fx_net_with_engine"] = (t, fx_net)
        h["Fz_net_with_engine"] = (t, fz_net)
        h["CX_net_with_engine"] = (t, fx_net / qS)
    return h


def write_history_csv(case, hist):
    names = sorted(hist)
    t = hist["CD"][0]
    with open(os.path.join(case, "history.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time_s"] + names)
        cols = [np.interp(t, *hist[n]) for n in names]
        for i in range(len(t)):
            w.writerow([f"{t[i]:.8g}"] + [f"{c[i]:.8g}" for c in cols])

# =====================================================================================================
# Reports
# =====================================================================================================

REPORT_ORDER = [
    "Fx_body", "Fy_body", "Fz_body", "Mx_cg", "My_cg", "Mz_cg",
    "Drag", "Lift", "SideForce", "CD", "CL", "CY", "L_over_D", "Cm_pitch", "Cl_roll", "Cn_yaw",
    "mdot_engine_face", "mdot_exhaust", "PressureRecovery_engine_face", "Mach_engine_face",
    "p_engine_face", "T_engine_face", "p_exhaust", "Mach_exhaust", "T_exhaust",
    "StreamThrust_engine_face", "StreamThrust_exhaust", "Fx_net_with_engine", "Fz_net_with_engine",
    "CX_net_with_engine",
]


def reports(case, flow, hist):
    t_end = float(hist["CD"][0][-1])
    out = {"t_end_s": t_end, "window_s": ASYMPTOTE_WINDOW_S, "reports": {}, "report_std": {},
           "report_last": {}}
    for name in REPORT_ORDER:
        if name not in hist:
            continue
        t, y = hist[name]
        mask = t >= t[-1] - ASYMPTOTE_WINDOW_S
        out["reports"][name] = float(np.mean(y[mask]))
        out["report_std"][name] = float(np.std(y[mask]))
        out["report_last"][name] = float(y[-1])

    # Drift over the last metrics.TAIL_FRACTION of the run (same check as the DSE cases)
    st = metrics.window_stats(*hist["CD"], metrics.TAIL_FRACTION)
    out["CD_drift_pct"] = st["drift_pct"] if st else None

    status = asymptote_status(hist)
    out["stopping_criteria"] = status
    out["converged"] = bool(status) and all(s["ok"] for s in status.values())

    # Per-wall breakdown (complete aircraft), as uav_breakdown.java
    rows = []
    S = flow["symmetry_factor"]
    for patch in WALL_PATCHES:
        P = force_columns(case, f"forces_{patch}", "force.dat")
        if P is None:
            continue
        mask = P["t"] >= P["t"][-1] - ASYMPTOTE_WINDOW_S
        mean = lambda arr, i: float(np.mean(arr[mask, i])) * S if arr is not None else float("nan")
        rows.append({"boundary": patch, "Fx_total": mean(P["total"], 0), "Fx_pressure": mean(P["pressure"], 0),
                     "Fx_shear": mean(P["viscous"], 0), "Fz_total": mean(P["total"], 2)})
    out["breakdown"] = rows
    with open(os.path.join(case, "breakdown.txt"), "w") as fh:
        fh.write("boundary, Fx_total, Fx_pressure, Fx_shear, Fz_total [N]\n")
        for r in rows:
            fh.write(f"{r['boundary']:<22s} {r['Fx_total']:10.3f} {r['Fx_pressure']:10.3f} "
                     f"{r['Fx_shear']:10.3f} {r['Fz_total']:10.3f}\n")

    # Wall y+ at the write times (yPlus function object, per patch)
    yp = {}
    for path in sorted(glob.glob(os.path.join(case, "postProcessing", "yPlus", "*", "yPlus.dat")),
                       key=metrics._start_time):
        for line in open(path):
            p = line.split()
            if line.startswith("#") or len(p) < 5:
                continue
            yp.setdefault(float(p[0]), {})[p[1]] = (float(p[2]), float(p[3]), float(p[4]))
    out["yplus_history"] = [{"t": tt, **{k: list(v) for k, v in d.items()}} for tt, d in sorted(yp.items())]
    return out

# =====================================================================================================
# Plots (the STAR-CCM+ monitor plots)
# =====================================================================================================

def plot_monitors(case, flow, hist, m):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, no plots")
        return
    out = os.path.join(case, "plots")
    os.makedirs(out, exist_ok=True)
    half = flow["half_model"]

    plots = [
        ("CD", ["CD"], "CD"), ("CL", ["CL"], "CL"), ("CY", ["CY"], "CY"),
        ("Cm (pitch)", ["Cm_pitch"], "Cm about CG"), ("Cl (roll)", ["Cl_roll"], "Cl"),
        ("Cn (yaw)", ["Cn_yaw"], "Cn"), ("L over D", ["L_over_D"], "L/D"),
        ("CX net with engine", ["CX_net_with_engine"], "CX net"),
        ("Forces - lift drag side [N]", ["Lift", "Drag", "SideForce"], "Force [N]"),
        ("Force - net axial with engine [N]", ["Fx_body", "Fx_net_with_engine"], "Force [N]"),
        ("Engine mass flows [kg per s]", ["mdot_engine_face", "mdot_exhaust"], "Mass flow [kg/s]"),
        ("Inlet pressure recovery", ["PressureRecovery_engine_face"], "p0 / p0_inf"),
        ("Engine face Mach", ["Mach_engine_face"], "Mach"),
        ("Engine face static pressure [Pa]", ["p_engine_face"], "p [Pa]"),
    ]
    if half:   # zero by symmetry
        plots = [p for p in plots if p[1][0] not in ("CY", "Cl_roll", "Cn_yaw")]

    for title, names, ylabel in plots:
        names = [n for n in names if n in hist]
        if not names:
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        tails = []
        for n in names:
            t, y = hist[n]
            ax.plot(t, y, linewidth=1.0, label=f"{n} = {m['reports'].get(n, float('nan')):.5g}")
            tails.append(y[t >= t[-1] - ASYMPTOTE_WINDOW_S])
        t_end = max(hist[n][0][-1] for n in names)
        ax.axvspan(t_end - ASYMPTOTE_WINDOW_S, t_end, color="0.6", alpha=0.25,
                   label=f"final window ({ASYMPTOTE_WINDOW_S * 1e3:g} ms)")
        for n in names:
            if n in ASYMPTOTE:
                s = m["stopping_criteria"].get(n, {})
                ax.set_title(f"{title}   [band {s.get('width', float('nan')):.3g} / tol {ASYMPTOTE[n]:g}]")
                break
        else:
            ax.set_title(title)
        # y-range from the settled part, the start-up transient flattens everything else
        lo = min(float(np.min(x)) for x in tails if len(x))
        hi = max(float(np.max(x)) for x in tails if len(x))
        pad = max(8.0 * (hi - lo), 0.05 * max(abs(lo), abs(hi)), 1e-9)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=8)
        fig.savefig(os.path.join(out, "plot_" + re.sub(r"[^A-Za-z0-9_+-]", "_", title) + ".png"),
                    dpi=160, bbox_inches="tight")
        plt.close(fig)

    # Wall y+ (write times only)
    yh = m.get("yplus_history", [])
    if yh:
        fig, ax = plt.subplots(figsize=(9, 5))
        t = [r["t"] for r in yh]
        for patch in WALL_PATCHES:
            if all(patch in r for r in yh):
                ax.plot(t, [r[patch][1] for r in yh], "o-", label=f"{patch} max")
                ax.plot(t, [r[patch][2] for r in yh], "s--", label=f"{patch} avg")
        ax.set_xlabel("Time [s]"); ax.set_ylabel("y+"); ax.set_title("Wall y+")
        ax.grid(True, linestyle="--", alpha=0.6); ax.legend(fontsize=8)
        fig.savefig(os.path.join(out, "plot_Wall_y_.png"), dpi=160, bbox_inches="tight")
        plt.close(fig)


def plot_residuals(case):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    res = {}
    for path in sorted(glob.glob(os.path.join(case, "postProcessing", "residuals", "*", "solverInfo.dat")),
                       key=metrics._start_time):
        lines = open(path).readlines()
        header = next((l for l in lines if l.startswith("# Time")), None)
        if header is None:
            continue
        cols = header.lstrip("#").split()
        fields = {c[:-len("_initial")]: i for i, c in enumerate(cols) if c.endswith("_initial")}
        for line in lines:
            p = line.split()
            if line.startswith("#") or len(p) != len(cols):
                continue
            for name, i in fields.items():
                tt, vv = res.setdefault(name, ([], []))
                try:
                    if not tt or float(p[0]) > tt[-1]:
                        tt.append(float(p[0])); vv.append(float(p[i]))
                except ValueError:
                    pass
    if not res:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, (tt, vv) in sorted(res.items()):
        ax.semilogy(tt, vv, linewidth=0.8, label=name)
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Initial residual"); ax.set_title("Residuals")
    ax.grid(True, which="both", linestyle="--", alpha=0.5); ax.legend()
    os.makedirs(os.path.join(case, "plots"), exist_ok=True)
    fig.savefig(os.path.join(case, "plots", "residuals.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

# =====================================================================================================
# ParaView scenes (detail_post.py) and summary
# =====================================================================================================

def post_process(case):
    name = os.path.basename(os.path.normpath(case))
    log = os.path.join(case, "log.pvbatch")
    rc = sh(f"LIBGL_ALWAYS_SOFTWARE=1 pvbatch --force-offscreen-rendering {os.path.join(HERE, 'detail_post.py')} "
            f"{os.path.join(case, name + '.foam')} {os.path.join(case, 'images')} > {log} 2>&1")
    if rc != 0:
        print(f"  post processing failed, last lines of {log}:")
        sh(f"tail -n 30 {log}")
    return rc == 0


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


def format_summary(m):
    f = m.get("flow", {})
    rep, std = m.get("reports", {}), m.get("report_std", {})
    bar = "=" * 72
    lines = [
        "", bar,
        f" {os.path.basename(m.get('case', ''))}   Mach {f.get('mach')} | alpha {f.get('alpha_deg')} | "
        f"beta {f.get('beta_deg')} | CG {f.get('cg')} | {'half' if f.get('half_model') else 'full'} model",
        bar,
        f"  stopped by {m.get('stop_reason', '?')} at t = {m.get('t_end_s', float('nan')):.5g} s, "
        f"solve time {m.get('solve_time_s', float('nan')):.0f} s, "
        f"{m.get('mesh', {}).get('mesh_cells', '?')} cells",
        "", "== Stopping criteria (band over the final window) ==",
    ]
    for k, s in m.get("stopping_criteria", {}).items():
        lines.append(f"  {k + ' asymptote':<34s} band {s['width']:12.4g}  tol {s['tol']:10.4g}  "
                     f"satisfied={s['ok']}")
    lines.append(f"  CD drift over the last {100 * metrics.TAIL_FRACTION:.0f}%: "
                 f"{m.get('CD_drift_pct') if m.get('CD_drift_pct') is None else round(m['CD_drift_pct'], 4)} %")
    lines += ["", f"== Reports (mean over the final {1e3 * m.get('window_s', 0):g} ms, +/- std) =="]
    for k in REPORT_ORDER:
        if k in rep:
            lines.append(f"  {k:<32s} {rep[k]:14.6g}   +/- {std.get(k, 0):.3g}")
    lines += ["", "== Axial force breakdown (complete aircraft) [N] ==",
              f"  {'boundary':<14s} {'Fx_total':>10s} {'Fx_press':>10s} {'Fx_shear':>10s} {'Fz_total':>10s}"]
    for r in m.get("breakdown", []):
        lines.append(f"  {r['boundary']:<14s} {r['Fx_total']:10.3f} {r['Fx_pressure']:10.3f} "
                     f"{r['Fx_shear']:10.3f} {r['Fz_total']:10.3f}")
    post = m.get("post", {})
    if post:
        lines += ["", "== Wall y+ (latest time) =="]
        for k, v in post.get("yplus", {}).items():
            lines.append(f"  {k:<14s} area avg {v['avg']:8.2f}   max {v['max']:9.2f}")
        lines += ["", "== Field extremes (fluid volume) =="]
        for k, (lo, hi) in post.get("field_range", {}).items():
            lines.append(f"  {k:<26s} min {lo:14.6g}   max {hi:14.6g}")
        lp = post.get("low_pressure_cells")
        if lp:
            lines += ["", f"== Cells with p < {lp['limit']:g} Pa: {lp['count']} ==",
                      f"  bounds x {lp['bounds'][0]:.4f} .. {lp['bounds'][1]:.4f}, y {lp['bounds'][2]:.4f} .. "
                      f"{lp['bounds'][3]:.4f}, z {lp['bounds'][4]:.4f} .. {lp['bounds'][5]:.4f} m"
                      if lp["count"] else "  none"]
    lines += [bar, f"  CONVERGED (asymptotic criteria): {m.get('converged')}", bar, ""]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
