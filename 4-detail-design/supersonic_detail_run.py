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
MDOT_AIR   = 0.2635              # engine-face air mass flow target [kg/s] (a result, set by the back pressure)
MDOT_FUEL  = 0.0216              # [kg/s], exhaust mass flow = MDOT_AIR + MDOT_FUEL
T0_EXHAUST = 2250.0              # exhaust total temperature [K] (assumed)
P_EXHAUST  = 15065.0             # exhaust static pressure [Pa] (from the 343.75 N thrust estimate)
EXHAUST_TURB_INTENSITY = 0.01
EXHAUST_VISC_RATIO     = 10.0

# --- engine-face back pressure. A straight extension of the engine-face section (1.5 D, in the CAD) ends in
#     the outlet patch, which holds a fixed static pressure; the engine-face quantities, mass flow included,
#     are results sampled on the engine-face plane (as 3-inlet-design/run_inlet.py)
PT_RECOVERY_EST = 0.94 * 0.979   # engine-face p0 / p0_inf assumed for the 1D back pressure: 2D external
                                 # compression x S-duct recovery of design 39 (0-results/inlet_dse_results.csv)
P_BACK          = 35500.0        # [Pa] outlet static pressure; None = 1D value at MDOT_AIR with PT_RECOVERY_EST
RAMP_ITERS      = 1500           # back pressure ramps down to P_BACK over these iterations (freestream start) ...
P_START_FRAC    = 0.2            # ... from p0_face - P_START_FRAC * (p0_face - P_BACK)
MDOT_TARGET_TOL_PCT = 0.5        # engine-face mass flow vs MDOT_AIR, reported only (it is a result)

# --- duct start. On a freestream start the inlet duct and the extension are filled with gas at rest at the
#     starting back pressure (p_back_start, T0_inf), as in 3-inlet-design. A duct started at the supersonic
#     freestream stayed supersonic against the back pressure (detail_M1.5_a0_b0, Nov 2026: engine-face
#     Mach 1.18, p0 recovery 0.60, then a floating point exception at iteration 2519)
DUCT_INIT         = True
DUCT_INIT_X_START = None         # [m] upstream end of the filled region; None = 2 mm aft of the last point
                                 # where the inlet duct meets the inlet lip or the airframe

# --- walls
WALL_TEMPERATURE = None          # None = adiabatic, a value [K] = isothermal walls

# =====================================================================================================
# GEOMETRY AND MESH (the mesh does not depend on the flow condition and is reused between runs)
# =====================================================================================================
GEOMETRY   = os.path.join(HERE, "uav.stl")     # ASCII, one solid per named face (GEOMETRY_SOLIDS), metres
GEOMETRY_SOLIDS = ("airframe", "inlet_lip", "inlet_duct", "extension", "outlet", "exhaust")
HALF_MODEL = None                # None: half model (symmetry plane y = 0) when BETA_DEG == 0, else full
DOMAIN     = {"x": (-2.4, 10.0), "y": (-4.5, 4.5), "z": (-3.6, 4.8)}   # extended to whole BASE_SIZE cells
BASE_SIZE  = 0.4                 # background (largest) cell [m]; all sizes below are BASE_SIZE / 2^n

# surface (target, minimum) cell size [m]
SURFACE_SIZES = {
    "airframe":     (0.003125,  0.0015625),   # STAR-CCM+ 6.25 / 1.5625 mm
    "inlet_lip":    (0.000390625, 0.000390625),  # STAR-CCM+ 0.5 / 0.25 mm, nearest BASE_SIZE / 2^n
    "inlet_duct":   (0.0015625, 0.00078125),  # also the extension
    "engine_ports": (0.0015625, 0.00078125),  # outlet + exhaust
}
FEATURE_EDGE_SIZE = 0.0015625    # cell size along sharp CAD edges [m]
N_CELLS_BETWEEN_LEVELS = 3
RESOLVE_FEATURE_ANGLE  = 30.0    # [deg] surfaces curving more than this get the minimum size

# volume refinement boxes, body axes: name -> (min corner, max corner, cell size [m])
# Extended upward, downward and rearward of the STAR-CCM+ boxes to follow the shocks (Mach angle
# 41.8 deg at Mach 1.5) and the wake / plume further, plus a 6.25 mm box around the vehicle that also
# covers the belly shocks. Vehicle extent: x -0.246 .. 1.425, |y| <= 0.554, z -0.07 .. 0.30. The 4.5 M
# cell half mesh (2026-11-14) grows by ~0.6 M from the lower box bottoms.
REFINEMENT_BOXES = {
    "refineFar":   ((-0.80, -1.40,  -1.10),  (6.00, 1.40,  2.40), 0.05),      # STAR: x 4.00, z 1.40
    "refineMid":   ((-0.50, -0.85,  -0.60),  (3.60, 0.85,  1.30), 0.025),     # STAR: x 2.40, z 0.65
    "refineNear":  ((-0.32, -0.65,  -0.30),  (2.40, 0.65,  0.65), 0.0125),    # STAR: x 1.60, z 0.36
    "refineBody":  ((-0.30, -0.60,  -0.16),  (1.75, 0.60,  0.34), 0.00625),   # new
    "refineInlet": (( 0.62, -0.07,  -0.045), (1.06, 0.07,  0.20), 0.003125),  # STAR: z 0.14 (lip shock / spillage)
    "refinePlume": (( 1.38, -0.15,  -0.15),  (2.60, 0.15,  0.15), 0.00625),   # STAR: x 1.80, +/-0.12
}

PRISM_LAYERS         = 5
PRISM_LAYERS_DUCT    = 0
PRISM_LAYERS_LIP     = 0
PRISM_EXPANSION      = 1.2
PRISM_FINAL_REL      = 0.3       # outermost layer / local surface cell size
PRISM_MIN_REL        = 0.5       # snappyHexMesh removes stacks thinner than this x local cell size

LOCATION_IN_MESH = (-2.0123, 2.0123, 2.0123)   # any point in the fluid with y > 0

# =====================================================================================================
# RUN CONTROL
# =====================================================================================================
CASE_TEMPLATE    = os.path.join(HERE, "detail_template_supersonic")
NP               = 100           # MPI processes
# Local time stepping (ddtScheme localEuler): a pseudo-time march to the steady state where every cell
# takes its own largest stable step, so the ~0.1 mm wall and lip cells do not hold back the 0.4 m
# far-field cells. OpenFOAM time is the iteration count (deltaT 1).
MAX_ITERATIONS   = 30000         # cap, ~1.1 s each on 50 cores
WRITE_INTERVAL   = 1000          # [iterations] field writes (the last two are kept)
MAX_COURANT      = 0.4           # local Courant number of every cell
MONITOR_INTERVAL = 10            # iterations between monitor samples
POLL_S           = 120           # [s] between stopping-criterion checks while the solver runs
# Initial field: "auto" = the finished case on the same mesh closest in (Mach, alpha, beta), converged
# cases first; a case path = that case; None = uniform freestream. The internal fields of its latest
# time are copied into 0/ (same mesh, so cell for cell); boundary conditions stay those of this case.
INIT_FROM        = "auto"

# Stopping: all of these must stay within the given band (max - min) over the final window
ASYMPTOTE = {
    "CD":            1.0e-4,     # 1 drag count
    "CL":            5.0e-4,
    "Cm_pitch":      5.0e-4,
    "mdot_engine_face": 1.0e-3,  # [kg/s] complete aircraft, the free variable (back pressure is imposed)
}
ASYMPTOTE_WINDOW        = 2000  # [iterations]
ASYMPTOTE_MIN_ITER      = 6000  # no check before this iteration
ASYMPTOTE_MIN_ITER_INIT = 3000  # the same, for a case started from another case (INIT_FROM)

WALL_PATCHES = ("airframe", "inlet_lip", "inlet_duct")   # vehicle walls; the extension wall is the engine's

# =====================================================================================================
# Driver
# =====================================================================================================

def main():
    half = BETA_DEG == 0.0 if HALF_MODEL is None else bool(HALF_MODEL)
    geo = geometry_info(GEOMETRY)
    flow = flow_conditions(geo, half)
    print_conditions(flow)

    # 1. Mesh, reused while the mesh settings and the geometry are unchanged
    mesh_dir = f"./detail_mesh_{'half' if half else 'full'}"
    settings = mesh_settings(half)
    if read_json(mesh_dir).get("mesh_settings") == settings and os.path.isdir(f"{mesh_dir}/constant/polyMesh"):
        print(f"[mesh] reusing {mesh_dir}")
    else:
        print(f"[mesh] building {mesh_dir} ...")
        t0 = time.perf_counter()
        prepare_mesh(mesh_dir, half)
        if not mesh(mesh_dir):
            print("Error: meshing failed")
            return 1
        write_json(mesh_dir, {"mesh_settings": settings, "mesh_time_s": round(time.perf_counter() - t0, 1),
                              **mesh_stats(mesh_dir)})
        print(f"[mesh] done in {time.perf_counter() - t0:.0f} s")

    # 2. Case
    case = f"./detail_M{MACH:g}_a{ALPHA_DEG:g}_b{BETA_DEG:g}"
    print(f"\n{'=' * 60}\n {os.path.basename(case)}\n{'=' * 60}")
    print("[1/4] Copying the mesh and writing the flow condition ...")
    prepare_case(case, mesh_dir, flow)
    src = find_init_source(case, mesh_dir)
    init_from = initialise_from(case, *src) if src else None
    write_outlet_pressure(case, flow, ramp=not init_from)
    duct_init = initialise_duct(case, flow, geo["duct"]) if DUCT_INIT and not init_from else None
    write_json(case, {"case": os.path.abspath(case), "flow": flow, "mesh": read_json(mesh_dir),
                      "init_from": init_from, "duct_init": duct_init})

    # 3. Solve
    print("[2/4] Solving (rhoCentralFoam) ...")
    t0 = time.perf_counter()
    stop_reason = solve(case, flow)
    m = {**read_json(case), "solve_time_s": round(time.perf_counter() - t0, 1), "stop_reason": stop_reason}
    write_json(case, m)
    if stop_reason not in ("stopping_criteria", "max_iterations"):
        print(f"Error: {stop_reason}")
        return 1

    # 4. Reports, plots, images
    print("[3/4] Reports and monitor plots ...")
    hist = histories(case, flow)
    m.update(reports(case, flow, hist))
    write_json(case, m)
    write_history_csv(case, hist)
    plot_monitors(case, hist, m)
    plot_residuals(case)

    print("[4/4] ParaView scenes ...")
    post_process(case)
    text = format_summary(read_json(case))      # detail_post.py adds its checks to metrics.json
    print(text)
    with open(os.path.join(case, "summary.txt"), "w") as fh:
        fh.write(text)
    return 0 if m["converged"] else 2

# =====================================================================================================
# Flow condition
# =====================================================================================================

def isa_atmosphere(h):
    g0, T_sl, p_sl, lapse, h_tp, R = 9.80665, 288.15, 101325.0, -0.0065, 11000.0, 287.05287
    T = T_sl + lapse * min(h, h_tp)
    p = p_sl * (T / T_sl) ** (-g0 / (lapse * R))
    if h > h_tp:
        p *= math.exp(-g0 * (h - h_tp) / (R * T))
    return T, p

def sutherland(T):
    return SUTHERLAND_AS * T ** 1.5 / (T + SUTHERLAND_TS)

def exhaust_mach(mdot, area, p, T0):
    # Exit Mach number for a mass flux at static pressure p and total temperature T0 (bisection)
    f = lambda M: p * M * math.sqrt(GAMMA * (1.0 + 0.5 * (GAMMA - 1.0) * M * M) / (R_GAS * T0))
    lo, hi = 0.0, 20.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) < mdot / area else (lo, mid)
    return 0.5 * (lo + hi)

def subsonic_mach(mdot, area, p0, T0):
    # Subsonic Mach number of a mass flow through a section (bisection); None above the choked mass flow
    flux = lambda M: (p0 * math.sqrt(GAMMA / (R_GAS * T0)) * M
                      * (1.0 + 0.5 * (GAMMA - 1.0) * M * M) ** (-0.5 * (GAMMA + 1.0) / (GAMMA - 1.0)))
    if mdot / area >= flux(1.0):
        return None
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if flux(mid) < mdot / area else (lo, mid)
    return 0.5 * (lo + hi)

def turbulence(U, rho, T, intensity, visc_ratio):
    # (k, omega, nut) from a turbulence intensity and a viscosity ratio
    k = 1.5 * (intensity * U) ** 2
    nut = visc_ratio * sutherland(T) / rho
    return k, k / nut, nut

def flow_conditions(geo, half):
    T, p = isa_atmosphere(ALTITUDE_M)
    U = MACH * math.sqrt(GAMMA * R_GAS * T)
    rho = p / (R_GAS * T)
    a, b = math.radians(ALPHA_DEG), math.radians(BETA_DEG)
    k, omega, nut = turbulence(U, rho, T, FREESTREAM_TURB_INTENSITY, FREESTREAM_VISC_RATIO)
    iso = 1.0 + 0.5 * (GAMMA - 1.0) * MACH ** 2

    # 1D exhaust state (full exhaust area and mass flow)
    mdot_ex = MDOT_AIR + MDOT_FUEL
    M_ex = exhaust_mach(mdot_ex, geo["area"]["exhaust"], P_EXHAUST, T0_EXHAUST)
    T_ex = T0_EXHAUST / (1.0 + 0.5 * (GAMMA - 1.0) * M_ex ** 2)
    U_ex = M_ex * math.sqrt(GAMMA * R_GAS * T_ex)
    rho_ex = P_EXHAUST / (R_GAS * T_ex)
    k_ex, omega_ex, _ = turbulence(U_ex, rho_ex, T_ex, EXHAUST_TURB_INTENSITY, EXHAUST_VISC_RATIO)

    # 1D engine face at the target mass flow with the assumed recovery -> outlet back pressure
    p0_inf, T0_inf = p * iso ** (GAMMA / (GAMMA - 1.0)), T * iso
    ef = geo["engine_face"]
    p0_ef = PT_RECOVERY_EST * p0_inf
    M_ef = subsonic_mach(MDOT_AIR, ef["area"], p0_ef, T0_inf)
    if M_ef is None:
        raise SystemExit("MDOT_AIR is above the choked mass flow of the engine face with PT_RECOVERY_EST")
    p_back = P_BACK or p0_ef / (1.0 + 0.5 * (GAMMA - 1.0) * M_ef ** 2) ** (GAMMA / (GAMMA - 1.0))

    return dict(
        mach=MACH, alpha_deg=ALPHA_DEG, beta_deg=BETA_DEG, cg=list(CG), altitude_m=ALTITUDE_M,
        half_model=half, symmetry_factor=2.0 if half else 1.0, gamma=GAMMA, R=R_GAS,
        sutherland_As=SUTHERLAND_AS, sutherland_Ts=SUTHERLAND_TS,
        p_inf=p, T_inf=T, rho_inf=rho, V_inf=U,
        U_vec=[U * math.cos(a) * math.cos(b), -U * math.sin(b), U * math.sin(a) * math.cos(b)],
        q_inf=0.5 * GAMMA * p * MACH ** 2, p0_inf=p0_inf, T0_inf=T0_inf,
        k_inf=k, omega_inf=omega, nut_inf=nut,
        mdot_air=MDOT_AIR, mdot_exhaust=mdot_ex, T0_exhaust=T0_EXHAUST, p_exhaust=P_EXHAUST,
        engine_face_area=ef["area"], engine_face_x=ef["x"], engine_face_centre=ef["centre"],
        engine_face_radius=ef["radius"], outlet_x=ef["x_outlet"], pt_recovery_est=PT_RECOVERY_EST,
        mach_engine_face_1d=M_ef,
        p_back=p_back, p_back_set=P_BACK is not None, p_back_start=p0_ef - P_START_FRAC * (p0_ef - p_back), ramp_iters=RAMP_ITERS,
        exhaust_mach_1d=M_ex, exhaust_T_1d=T_ex, exhaust_U_1d=U_ex, exhaust_rho_1d=rho_ex,
        exhaust_supersonic=M_ex >= 1.0, exhaust_U_vec=[U_ex * c for c in geo["normal"]["exhaust"]],
        exhaust_thrust_1d=mdot_ex * U_ex + (P_EXHAUST - p) * geo["area"]["exhaust"],
        k_exhaust=k_ex, omega_exhaust=omega_ex,
    )

def print_conditions(f):
    print(f"Mach {f['mach']:g} | alpha {f['alpha_deg']:g} deg | beta {f['beta_deg']:g} deg | "
          f"CG ({', '.join(f'{c:g}' for c in f['cg'])}) m | {'half' if f['half_model'] else 'full'} model")
    print(f"  {f['altitude_m'] / 1000:g} km ISA: p {f['p_inf']:.1f} Pa, T {f['T_inf']:.2f} K, "
          f"rho {f['rho_inf']:.5f} kg/m3, V {f['V_inf']:.1f} m/s, q {f['q_inf']:.0f} Pa, p0 {f['p0_inf']:.0f} Pa")
    print(f"  engine face x {f['engine_face_x']:.4f} m, {f['engine_face_area'] * 1e4:.2f} cm2 | outlet back pressure "
          f"{f['p_back']:.0f} Pa " + ("(P_BACK) | 1D design point" if f["p_back_set"] else "(1D") +
          f": {f['mdot_air']:g} kg/s, Mach {f['mach_engine_face_1d']:.3f} with recovery "
          f"{f['pt_recovery_est']:.4f}" + ("" if f["p_back_set"] else ")"))
    print(f"  exhaust {f['mdot_exhaust']:g} kg/s, T0 {f['T0_exhaust']:g} K, p {f['p_exhaust']:g} Pa -> 1D Mach "
          f"{f['exhaust_mach_1d']:.3f}, U {f['exhaust_U_1d']:.0f} m/s, gross thrust {f['exhaust_thrust_1d']:.1f} N")
    if not f["exhaust_supersonic"]:
        print("  WARNING: the 1D exhaust is subsonic, its static pressure is left free (zeroGradient)")

# =====================================================================================================
# Geometry
# =====================================================================================================

def geometry_info(path):
    # Area and mean outward normal of each solid of an ASCII STL (one solid per named face)
    with open(path, "rb") as fh:
        head = fh.read(512)
    if not head.lstrip().lower().startswith(b"solid") or b"facet" not in head:
        raise SystemExit(f"{path} is not an ASCII STL; export it as ASCII with one solid per named face")
    verts = {}
    with open(path) as fh:
        for line in fh:
            w = line.split()
            if w and w[0] == "solid":
                name = w[1] if len(w) > 1 else "solid"
            elif w and w[0] == "vertex":
                verts.setdefault(name, []).append(w[1:4])
    for need in GEOMETRY_SOLIDS:
        if need not in verts:
            raise SystemExit(f"{path} has no 'solid {need}' (found: {', '.join(verts)})")
    area, normal, centre = {}, {}, {}
    for name, v in verts.items():
        p = np.asarray(v, dtype=float).reshape(-1, 3, 3)
        n = 0.5 * np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        s = n.sum(axis=0)
        a = np.linalg.norm(n, axis=1)
        area[name] = float(a.sum())
        normal[name] = list(s / np.linalg.norm(s)) if np.linalg.norm(s) > 0 else [0.0, 0.0, 0.0]
        centre[name] = [float(c) for c in (a[:, None] * p.mean(axis=1)).sum(axis=0) / a.sum()]
    # Engine face: the upstream end of the straight extension (along x), the same section as the outlet
    if abs(normal["outlet"][0]) < 0.999:
        raise SystemExit(f"{path}: the outlet must be a plane normal to x (normal {normal['outlet']})")
    x_ef = float(np.asarray(verts["extension"], dtype=float)[:, 0].min())
    engine_face = {"x": x_ef, "centre": [x_ef, centre["outlet"][1], centre["outlet"][2]], "area": area["outlet"],
                   "radius": math.sqrt(area["outlet"] / math.pi), "x_outlet": centre["outlet"][0]}
    # Duct start region: the inlet duct downstream of its last vertex shared with the inlet lip or the
    # airframe (and of the whole lip), so the capture opening lies upstream of it and the duct walls,
    # extension and outlet close it all round
    pts = lambda name: np.asarray(verts[name], dtype=float)
    keys = lambda a: {tuple(r) for r in np.round(a, 7)}
    outer = keys(pts("inlet_lip")) | keys(pts("airframe"))
    shared = [v for v in keys(pts("inlet_duct")) if v in outer]
    walls = np.vstack([pts("inlet_duct"), pts("extension")])
    duct = {"x_start": float(max([pts("inlet_lip")[:, 0].max()] + [v[0] for v in shared])),
            "n_shared_vertices": len(shared), "lo": walls.min(axis=0).tolist(), "hi": walls.max(axis=0).tolist()}
    return {"area": area, "normal": normal, "engine_face": engine_face, "duct": duct}

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
        prism=[PRISM_LAYERS, PRISM_LAYERS_DUCT, PRISM_LAYERS_LIP, PRISM_EXPANSION, PRISM_FINAL_REL,
               PRISM_MIN_REL],
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

def prepare_mesh(mesh_dir, half):
    if half and LOCATION_IN_MESH[1] <= 0:
        raise SystemExit("LOCATION_IN_MESH must have y > 0 on a half model")
    shutil.rmtree(mesh_dir, ignore_errors=True)
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
               f"nLayersLip          {PRISM_LAYERS_LIP};\n"
               f"expansionRatio      {PRISM_EXPANSION:.6g};\n"
               f"finalLayerThickness {PRISM_FINAL_REL:.6g};\n"
               f"minLayerThickness   {PRISM_MIN_REL:.6g};\n"
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
    if not ok:
        return False
    sh(f"checkMesh -constant -case {c} > {c}/log.checkMesh 2>&1")   # quality report, reference only
    # Prism-layer coverage per patch, the table at the end of the snappyHexMesh log
    lines = open(os.path.join(c, "log.snappyHexMesh")).read().splitlines()
    start = max((i for i, l in enumerate(lines) if l.strip().startswith("patch") and "layers" in l), default=None)
    if start is not None:
        print("  prism layers:")
        for l in lines[start:start + 12]:
            if not l.strip():
                break
            print("   " + l)
    return True

def mesh_stats(c):
    text = open(os.path.join(c, "log.checkMesh")).read()

    def find(pattern):
        m = re.search(pattern, text, flags=re.M)
        return float(m.group(1)) if m else None

    return {"mesh_cells": int(find(r"^\s*cells:\s+(\d+)") or 0),
            "max_non_orthogonality": find(r"Mesh non-orthogonality Max:\s+(\S+)"),
            "max_skewness": find(r"Max skewness = ([-+0-9.eE]+)"),
            "checkMesh_ok": "Mesh OK." in text}

# =====================================================================================================
# Case
# =====================================================================================================

def prepare_case(case, mesh_dir, f):
    shutil.rmtree(case, ignore_errors=True)
    shutil.copytree(mesh_dir, case, ignore=shutil.ignore_patterns(
        "processor*", "log.*", "*.foam", "metrics.json", "postProcessing"))
    # solver settings from the template, so a reused mesh does not carry older ones
    for name in ("controlDict", "fvSchemes", "fvSolution", "fvOptions"):
        shutil.copy(os.path.join(CASE_TEMPLATE, "system", name), os.path.join(case, "system", name))
    open(os.path.join(case, os.path.basename(case) + ".foam"), "w").close()

    path = lambda *p: os.path.join(case, *p)
    v = lambda x: f"{x:.8g}"
    sym = f["symmetry_factor"]
    r = f["engine_face_radius"]
    # half sizes of the box that bounds the engine-face plane. The bounds keep the cut cells whose centres lie inside, so the x half size must exceed half the largest duct cell: at 1 mm only a near-wall ring of the finer cells was sampled (25 % of the face area, detail_M1.5_a0_b0, Nov 2026)
    ef_box = (0.5 * r, 1.1 * r, 1.1 * r)
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
        f"tEnd            {MAX_ITERATIONS};",
        f"tWrite          {WRITE_INTERVAL};",
        f"maxCourant      {v(MAX_COURANT)};",
        f"monitorInterval {MONITOR_INTERVAL};",
        f"cg              {fmt_vec(CG)};",
        f"rhoInfRef       {v(f['rho_inf'])};",
        # engine-face sampling plane, bounded to the duct section (the plane also cuts the outer flow)
        f"efPoint         {fmt_vec(f['engine_face_centre'])};",
        f"efBounds        {fmt_vec([c - d for c, d in zip(f['engine_face_centre'], ef_box)])} "
        f"{fmt_vec([c + d for c, d in zip(f['engine_face_centre'], ef_box)])};", ""]))
    sub(path("system", "decomposeParDict"), r"^numberOfSubdomains\s+\d+;", f"numberOfSubdomains  {NP};")

def write_outlet_pressure(case, f, ramp):
    """Outlet back pressure: ramped down from p_back_start on a freestream start, constant otherwise."""
    value = (f"uniformValue    table ((0 {f['p_back_start']:.8g}) ({f['ramp_iters']} {f['p_back']:.8g}));\n"
             if ramp else f"uniformValue    constant {f['p_back']:.8g};\n")
    write_text(os.path.join(case, "0", "include", "outletPressure"),
               "// Written by supersonic_detail_run.py: engine back pressure at the extension outlet\n"
               "type            uniformFixedValue;\n" + value + f"value           uniform {f['p_back']:.8g};\n")

def initialise_duct(case, f, duct):
    """Fill the inlet duct and the extension (freestream start) with gas at rest at the starting back pressure.
    The cells are those of a box over the duct (from DUCT_INIT_X_START) connected to a point in the extension,
    so the outer flow inside the box is left alone. Returns a description for metrics.json."""
    x0 = DUCT_INIT_X_START if DUCT_INIT_X_START is not None else duct["x_start"] + 2e-3
    x_ef, x_out, r, c = f["engine_face_x"], f["outlet_x"], f["engine_face_radius"], f["engine_face_centre"]
    if not x0 < x_ef:
        raise SystemExit(f"duct start region begins at x {x0:.4f} m, not upstream of the engine face "
                         f"(x {x_ef:.4f} m); set DUCT_INIT_X_START")
    m = 2e-3
    lo = [x0, duct["lo"][1] - m, duct["lo"][2] - m]
    hi = [x_out + m, duct["hi"][1] + m, duct["hi"][2] + m]
    inside = [0.5 * (x_ef + x_out), c[1] + 0.3 * r, c[2] + 0.1 * r]   # in the extension, off the grid planes
    p, T = f["p_back_start"], f["T0_inf"]
    head = lambda obj: ("FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       dictionary;\n"
                        f"    object      {obj};\n}}\n// Written by supersonic_detail_run.py: duct start region\n\n")
    write_text(os.path.join(case, "system", "topoSetDict"), head("topoSetDict") +
               "actions\n(\n"
               f"    {{ name ductBox;  type cellSet; action new; source boxToCell; box {fmt_vec(lo)} {fmt_vec(hi)}; }}\n"
               "    { name ductInit; type cellSet; action new; source regionToCell; set ductBox;\n"
               f"      insidePoints ({fmt_vec(inside)}); nErode 0; }}\n);\n")
    write_text(os.path.join(case, "system", "setFieldsDict"), head("setFieldsDict") +
               "defaultFieldValues ( );\n\nregions\n(\n    cellToCell\n    {\n        set ductInit;\n"
               "        fieldValues\n        (\n"
               "            volVectorFieldValue U (0 0 0)\n"
               f"            volScalarFieldValue p {p:.8g}\n"
               f"            volScalarFieldValue T {T:.8g}\n"
               "        );\n    }\n);\n")
    print(f"  duct start: x {x0:.4f} .. {x_out:.4f} m at rest, p {p:.0f} Pa, T {T:.1f} K "
          f"({duct['n_shared_vertices']} lip / airframe vertices on the duct)")
    if not run_steps(case, [("topoSet", f"topoSet -case {case}")]):
        raise SystemExit("topoSet failed for the duct start region (DUCT_INIT = False starts from the freestream)")
    sizes = dict(re.findall(r"cellSet (\w+) now size (\d+)", open(os.path.join(case, "log.topoSet")).read()))
    n_box, n_duct = int(sizes.get("ductBox", 0)), int(sizes.get("ductInit", 0))
    if n_duct == 0:
        raise SystemExit(f"no duct cells found around {fmt_vec(inside)} (log.topoSet)")
    if not run_steps(case, [("setFields", f"setFields -case {case}")]):
        raise SystemExit("setFields failed for the duct start region")
    print(f"  duct start: {n_duct} cells set ({n_box} in the box, the rest is outer flow)")
    return {"x_start": x0, "box": [lo, hi], "inside_point": inside, "cells": n_duct, "box_cells": n_box,
            "p": p, "T": T}

def latest_time(case):
    times = [d for d in os.listdir(case) if re.fullmatch(r"[0-9.eE+-]+", d) and float(d) > 0]
    return max(times, key=float) if times else None

def mesh_signature(c):
    # "nPoints:.. nCells:.. nFaces:.. nInternalFaces:.." from the owner file header (ASCII in binary files too)
    try:
        with open(os.path.join(c, "constant", "polyMesh", "owner"), "rb") as fh:
            m = re.search(r'note\s+"([^"]*)"', fh.read(4096).decode("ascii", "replace"))
    except OSError:
        return None
    return " ".join(m.group(1).split()) if m else None

def find_init_source(case, mesh_dir):
    """(case, latest time) of the solution to start from (see INIT_FROM), or None."""
    if not INIT_FROM:
        return None
    sig = mesh_signature(mesh_dir)
    found = []
    for src in glob.glob("./detail_M*_a*_b*") if INIT_FROM == "auto" else [INIT_FROM]:
        if os.path.abspath(src) == os.path.abspath(case) or mesh_signature(src) != sig or not latest_time(src):
            continue
        m = read_json(src)
        f = m.get("flow", {})
        d = (10.0 * abs(f.get("mach", math.inf) - MACH) + abs(f.get("alpha_deg", math.inf) - ALPHA_DEG)
             + abs(f.get("beta_deg", math.inf) - BETA_DEG))
        found.append((not m.get("converged"), d, src))
    if not found:
        print("  no solution on the same mesh to start from, starting from the freestream")
        return None
    src = min(found)[2]
    return src, latest_time(src)

def initialise_from(case, src, t):
    """Copy the internal fields of src/t into case/0, keeping case/0's boundary conditions.
    Returns a description for metrics.json, or None (fields left at the freestream) on failure."""
    print(f"  initial field from {src} (iteration {t})")
    fields = [f for f in sorted(os.listdir(os.path.join(case, "0"))) if os.path.isfile(os.path.join(src, t, f))]
    tmp = os.path.join(case, t)
    os.makedirs(tmp, exist_ok=True)
    for f in fields:
        shutil.copy(os.path.join(src, t, f), tmp)
    # foamFormatConvert rewrites the copied fields in ASCII so their internalField can be spliced as text
    cdict = os.path.join(case, "system", "controlDict")
    sub(cdict, r"^writeFormat\s+binary;", "writeFormat     ascii;")
    new = {}
    try:
        if run_steps(case, [("foamFormatConvert", f"foamFormatConvert -noConstant -time {t} -case {case}")]):
            for f in fields:          # all fields spliced in memory first, so a failure leaves 0/ untouched
                new[f] = splice_internal_field(os.path.join(case, "0", f), os.path.join(tmp, f))
    except (OSError, ValueError) as e:
        print(f"  could not copy the fields ({e})")
        new = {}
    finally:
        sub(cdict, r"^writeFormat\s+ascii;", "writeFormat     binary;")
        shutil.rmtree(tmp, ignore_errors=True)
    if not new:
        print("  starting from the freestream instead")
        return None
    for f, text in new.items():
        write_text(os.path.join(case, "0", f), text)
    print(f"  copied {', '.join(new)}")
    return f"{os.path.abspath(src)} (iteration {t})"

def splice_internal_field(target, source):
    """target's text with its internalField entry replaced by source's (ASCII)."""
    with open(source) as fh:
        src = fh.read()
    i = src.find("\ninternalField")
    j = src.find("\nboundaryField", i)
    if not re.search(r"^\s*format\s+ascii;", src[:2000], flags=re.M) or i < 0 or j < 0:
        raise ValueError(f"no ASCII internalField in {source}")
    with open(target) as fh:
        text, n = re.subn(r"^internalField\b[^;]*;", lambda _: src[i + 1:j].rstrip(), fh.read(), count=1,
                          flags=re.M)
    if n != 1:
        raise ValueError(f"no internalField in {target}")
    return text

# =====================================================================================================
# Solve with the asymptotic stopping criteria
# =====================================================================================================

def solve(case, flow):
    """Run the solver, stop it once the stopping criteria hold. Returns the stop reason."""
    if not run_steps(case, [("decomposePar", f"decomposePar -force -case {case}")]):
        return "decomposePar failed"
    print(f"  -> mpirun -np {NP} rhoCentralFoam -parallel   (log.rhoCentralFoam)")
    with open(os.path.join(case, "log.rhoCentralFoam"), "w") as log:
        proc = subprocess.Popen(f"mpirun -np {NP} rhoCentralFoam -parallel -case {case}", shell=True,
                                executable="/bin/bash", stdout=log, stderr=subprocess.STDOUT)
        reason = "max_iterations"
        it_min = ASYMPTOTE_MIN_ITER_INIT if read_json(case).get("init_from") else ASYMPTOTE_MIN_ITER
        while proc.poll() is None:
            time.sleep(POLL_S)
            if reason != "max_iterations":
                continue
            try:
                hist = histories(case, flow)
            except (OSError, ValueError, IndexError, TypeError) as e:   # files mid-write, retry next poll
                print(f"  (monitor read failed: {e})")
                continue
            status = asymptote_status(hist, it_min)
            print(f"  iteration {hist['t'][-1]:.0f} | " + " | ".join(
                f"{k} band {s['width']:.3g}{' ok' if s['ok'] else ''}" for k, s in status.items()), flush=True)
            if all(s["ok"] for s in status.values()):
                print("  stopping criteria met, writing and stopping")
                sub(os.path.join(case, "system", "controlDict"), r"^stopAt\s+\w+;", "stopAt          writeNow;")
                reason = "stopping_criteria"
    if proc.returncode != 0:
        sh(f"tail -n 30 {case}/log.rhoCentralFoam")
        return "rhoCentralFoam failed"
    ok = run_steps(case, [("reconstructPar", f"reconstructPar -latestTime -case {case}")])
    sh(f"rm -rf {case}/processor*")
    return reason if ok else "reconstructPar failed"

def asymptote_status(hist, it_min):
    t = hist["t"]
    window = t >= t[-1] - ASYMPTOTE_WINDOW
    out = {}
    for name, tol in ASYMPTOTE.items():
        width = float(np.ptp(hist[name][window]))
        out[name] = {"width": width, "tol": tol, "ok": bool(t[-1] >= it_min and width <= tol)}
    return out

# =====================================================================================================
# Monitors (iteration histories, complete aircraft)
# =====================================================================================================

def read_dat(case, fo, fname):
    """(iterations, value columns) of postProcessing/<fo>/0/<fname>; vectors "(x y z)" give 3 columns."""
    rows = []
    with open(os.path.join(case, "postProcessing", fo, "0", fname)) as fh:
        for line in fh:
            if not line.startswith("#"):
                try:
                    rows.append([float(v) for v in line.replace("(", " ").replace(")", " ").split()])
                except ValueError:
                    pass
    rows = [r for r in rows if len(r) == len(rows[0])]     # drops a line still being written
    d = np.asarray(rows)
    return d[:, 0], d[:, 1:]

def histories(case, f):
    """{"t": iterations, name: values} for the complete aircraft, all on the iterations of the forces."""
    S, qS = f["symmetry_factor"], f["q_inf"] * S_REF
    a, b = math.radians(ALPHA_DEG), math.radians(BETA_DEG)
    ca, sa, cb, sb = math.cos(a), math.sin(a), math.cos(b), math.sin(b)
    t, force = read_dat(case, "forces", "force.dat")

    def on_t(fo, fname, col):      # one column of another monitor file, on the force iterations
        tt, d = read_dat(case, fo, fname)
        return np.interp(t, tt, d[:, col])

    Fx, Fy, Fz = (S * force[:, i] for i in range(3))
    Mx, My, Mz = (S * on_t("forces", "moment.dat", i) for i in range(3))
    drag = Fx * ca * cb - Fy * sb + Fz * sa * cb
    lift = -Fx * sa + Fz * ca
    # aircraft sign convention: Cm > 0 nose up, Cl > 0 starboard wing down, Cn > 0 nose to starboard
    h = {"t": t, "Fx_body": Fx, "Fz_body": Fz, "My_cg": My, "Drag": drag, "Lift": lift,
         "CD": drag / qS, "CL": lift / qS, "L_over_D": lift / drag, "Cm_pitch": My / (qS * C_REF)}
    if not f["half_model"]:     # zero by symmetry on a half model
        side = Fx * ca * sb + Fy * cb + Fz * sa * sb
        h.update(Fy_body=Fy, Mx_cg=Mx, Mz_cg=Mz, SideForce=side, CY=side / qS,
                 Cl_roll=-Mx / (qS * B_REF), Cn_yaw=-Mz / (qS * B_REF))

    # engine face: the sampling plane at the upstream end of the extension (normal +x)
    sfv = "surfaceFieldValue.dat"
    h["mdot_engine_face"] = S * on_t("engine_face_mdot", sfv, 0)
    h["p_engine_face"], h["Mach_engine_face"], h["T_engine_face"] = (
        on_t("engine_face_average", sfv, i) for i in range(3))
    h["PressureRecovery_engine_face"] = on_t("engine_face_p0", sfv, 0) / f["p0_inf"]

    # Stream thrust: x of the integral of p n + rho u (u.n) over a port, n out of the fluid. The net axial
    # force on the vehicle with its engine stream tube is walls + engine face + exhaust (> 0 net drag,
    # < 0 net thrust), as STAR's Fx_body + StreamThrust_face - StreamThrust_exhaust.
    I = {"engine_face": S * (on_t("engine_face_pressure", sfv, 0) + on_t("engine_face_momentum", sfv, 0)),
         "exhaust": S * (on_t("forces_exhaust", "force.dat", 3) + on_t("exhaust_momentum", sfv, 0))}
    h["StreamThrust_engine_face"], h["StreamThrust_exhaust"] = I["engine_face"], -I["exhaust"]
    h["Fx_net_with_engine"] = Fx + I["engine_face"] + I["exhaust"]
    h["CX_net_with_engine"] = h["Fx_net_with_engine"] / qS
    return h

def write_history_csv(case, hist):
    np.savetxt(os.path.join(case, "history.csv"), np.column_stack(list(hist.values())), fmt="%.8g",
               delimiter=",", header=",".join(["iteration"] + list(hist)[1:]), comments="")

# =====================================================================================================
# Reports
# =====================================================================================================

def reports(case, flow, hist):
    window = hist["t"] >= hist["t"][-1] - ASYMPTOTE_WINDOW
    it_min = ASYMPTOTE_MIN_ITER_INIT if read_json(case).get("init_from") else ASYMPTOTE_MIN_ITER
    status = asymptote_status(hist, it_min)
    out = {"iterations": int(hist["t"][-1]),
           "reports": {k: float(np.mean(y[window])) for k, y in hist.items() if k != "t"},
           "report_std": {k: float(np.std(y[window])) for k, y in hist.items() if k != "t"},
           "stopping_criteria": status, "converged": all(s["ok"] for s in status.values()),
           "breakdown": []}
    # Per-wall axial force breakdown (complete aircraft), as uav_breakdown.java
    for patch in WALL_PATCHES:
        t, d = read_dat(case, f"forces_{patch}", "force.dat")
        mean = flow["symmetry_factor"] * d[t >= t[-1] - ASYMPTOTE_WINDOW].mean(axis=0)
        out["breakdown"].append({"boundary": patch, "Fx_total": mean[0], "Fx_pressure": mean[3],
                                 "Fx_shear": mean[6], "Fz_total": mean[2]})
    return out

# =====================================================================================================
# Plots (the STAR-CCM+ monitor plots)
# =====================================================================================================

PLOTS = [   # (title, series, y label); series missing from the histories (half model) are skipped
    ("CD", ["CD"], "CD"), ("CL", ["CL"], "CL"), ("CY", ["CY"], "CY"),
    ("Cm (pitch)", ["Cm_pitch"], "Cm about CG"), ("Cl (roll)", ["Cl_roll"], "Cl"),
    ("Cn (yaw)", ["Cn_yaw"], "Cn"), ("L over D", ["L_over_D"], "L/D"),
    ("CX net with engine", ["CX_net_with_engine"], "CX net"),
    ("Forces - lift drag side [N]", ["Lift", "Drag", "SideForce"], "Force [N]"),
    ("Force - net axial with engine [N]", ["Fx_body", "Fx_net_with_engine"], "Force [N]"),
    ("Engine face mass flow [kg/s]", ["mdot_engine_face"], "mdot [kg/s]"),
    ("Inlet pressure recovery", ["PressureRecovery_engine_face"], "p0 / p0_inf"),
    ("Engine face Mach", ["Mach_engine_face"], "Mach"),
    ("Engine face static pressure [Pa]", ["p_engine_face"], "p [Pa]"),
]

def plot_monitors(case, hist, m):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.join(case, "plots"), exist_ok=True)
    t = hist["t"]
    window = t >= t[-1] - ASYMPTOTE_WINDOW
    for title, names, ylabel in PLOTS:
        names = [n for n in names if n in hist]
        if not names:
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        for n in names:
            ax.plot(t, hist[n], linewidth=1.0, label=f"{n} = {m['reports'][n]:.5g}")
        ax.axvspan(t[-1] - ASYMPTOTE_WINDOW, t[-1], color="0.6", alpha=0.25,
                   label=f"final window ({ASYMPTOTE_WINDOW} iterations)")
        s = m["stopping_criteria"].get(names[0])
        ax.set_title(f"{title}   [band {s['width']:.3g} / tol {s['tol']:g}]" if s else title)
        # y-range from the settled part, the start-up transient flattens everything else
        lo = min(hist[n][window].min() for n in names)
        hi = max(hist[n][window].max() for n in names)
        pad = max(8.0 * (hi - lo), 0.05 * max(abs(lo), abs(hi)), 1e-9)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=8)
        fig.savefig(os.path.join(case, "plots", "plot_" + re.sub(r"[^A-Za-z0-9_+-]", "_", title) + ".png"),
                    dpi=160, bbox_inches="tight")
        plt.close(fig)

def plot_residuals(case):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with open(os.path.join(case, "postProcessing", "residuals", "0", "solverInfo.dat")) as fh:
        lines = fh.read().splitlines()
    cols = next(l for l in lines if l.startswith("# Time"))[1:].split()
    rows = [l.split() for l in lines if not l.startswith("#") and len(l.split()) == len(cols)]
    t = [float(r[0]) for r in rows]
    num = lambda s: float(s) if re.fullmatch(r"[-+0-9.eE]+", s) else float("nan")
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, c in enumerate(cols):
        if c.endswith("_initial"):
            ax.semilogy(t, [num(r[i]) for r in rows], linewidth=0.8, label=c[:-len("_initial")])
    ax.set_xlabel("Iteration"); ax.set_ylabel("Initial residual"); ax.set_title("Residuals")
    ax.grid(True, which="both", linestyle="--", alpha=0.5); ax.legend()
    fig.savefig(os.path.join(case, "plots", "residuals.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

# =====================================================================================================
# ParaView scenes (detail_post.py) and summary
# =====================================================================================================

def post_process(case):
    log = os.path.join(case, "log.pvbatch")
    foam = os.path.join(case, os.path.basename(case) + ".foam")
    if sh(f"LIBGL_ALWAYS_SOFTWARE=1 pvbatch --force-offscreen-rendering {os.path.join(HERE, 'detail_post.py')} "
          f"{foam} {os.path.join(case, 'images')} > {log} 2>&1") != 0:
        print(f"  post processing failed, last lines of {log}:")
        sh(f"tail -n 30 {log}")

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
    f, rep, std = m["flow"], m["reports"], m["report_std"]
    bar = "=" * 72
    lines = [
        "", bar,
        f" {os.path.basename(m['case'])}   Mach {f['mach']} | alpha {f['alpha_deg']} | beta {f['beta_deg']} | "
        f"CG {f['cg']} | {'half' if f['half_model'] else 'full'} model",
        bar,
        f"  stopped by {m['stop_reason']} at iteration {m['iterations']}, solve time {m['solve_time_s']:.0f} s, "
        f"{m['mesh'].get('mesh_cells', '?')} cells",
        f"  initial field: {m['init_from'] or 'uniform freestream'}"
        + (f", duct from x {m['duct_init']['x_start']:.4f} m at rest ({m['duct_init']['cells']} cells)"
           if m.get("duct_init") else ""),
        "", "== Stopping criteria (band over the final window) ==",
    ]
    for k, s in m["stopping_criteria"].items():
        lines.append(f"  {k + ' asymptote':<34s} band {s['width']:12.4g}  tol {s['tol']:10.4g}  "
                     f"satisfied={s['ok']}")
    lines += ["", f"== Reports (mean over the final {ASYMPTOTE_WINDOW} iterations, +/- std) =="]
    lines += [f"  {k:<32s} {rep[k]:14.6g}   +/- {std[k]:.3g}" for k in rep]
    lines += ["", "== Axial force breakdown (complete aircraft) [N] ==",
              f"  {'boundary':<14s} {'Fx_total':>10s} {'Fx_press':>10s} {'Fx_shear':>10s} {'Fz_total':>10s}"]
    lines += [f"  {r['boundary']:<14s} {r['Fx_total']:10.3f} {r['Fx_pressure']:10.3f} "
              f"{r['Fx_shear']:10.3f} {r['Fz_total']:10.3f}" for r in m["breakdown"]]
    if "p_back" in f and "mdot_engine_face" in rep:
        ratio = rep["mdot_engine_face"] / f["mdot_air"]
        fixed = f.get("p_back_set", False)
        lines += ["", "== Engine face (back pressure fixed at the extension outlet, mass flow a result) ==",
                  f"  outlet back pressure {f['p_back']:.0f} Pa " + ("(set by P_BACK)" if fixed else
                                                                      f"(1D with PT_RECOVERY_EST {f['pt_recovery_est']:.4f})"),
                  f"  mass flow {rep['mdot_engine_face']:.4f} kg/s, target {f['mdot_air']:.4f} ({100 * (ratio - 1):+.2f} %)"]
        if abs(ratio - 1.0) * 100.0 > MDOT_TARGET_TOL_PCT and not fixed:
            lines.append(f"  OFF TARGET: try PT_RECOVERY_EST = {rep['PressureRecovery_engine_face']:.4f} "
                         "(the engine-face recovery of this run)")
    post = m.get("post")
    if post:
        lines += ["", "== Wall y+ (latest time) =="]
        lines += [f"  {k:<14s} area avg {v['avg']:8.2f}   max {v['max']:9.2f}" for k, v in post["yplus"].items()]
        lines += ["", "== Field extremes (fluid volume) =="]
        lines += [f"  {k:<26s} min {lo:14.6g}   max {hi:14.6g}" for k, (lo, hi) in post["field_range"].items()]
        lp = post.get("low_pressure_cells")
        if lp:
            b = lp["bounds"]
            lines += ["", f"== Cells with p < {lp['limit']:g} Pa: {lp['count']} ==",
                      f"  bounds x {b[0]:.4f} .. {b[1]:.4f}, y {b[2]:.4f} .. {b[3]:.4f}, z {b[4]:.4f} .. {b[5]:.4f} m"
                      if lp["count"] else "  none"]
    lines += [bar, f"  CONVERGED (stopping criteria): {m['converged']}", bar, ""]
    return "\n".join(lines)

if __name__ == "__main__":
    sys.exit(main())