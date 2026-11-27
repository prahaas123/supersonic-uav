"""Detail subsonic run (rhoSimpleFoam, Mach 0.1 - 0.9) of uav.stl with the engine flow, the subsonic counterpart
of supersonic_detail_run.py (same geometry, monitors, reports and metrics.json layout; detail_post.py works on
both). Template: case_template_subsonic. Run on Nibi with subsonic_detail_run.sh."""
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
MACH       = 0.5                 # freestream Mach number, 0 < MACH < 1
ALPHA_DEG  = 0.0                 # angle of attack [deg]
BETA_DEG   = 0.0                 # sideslip [deg], > 0 = wind from starboard
ALTITUDE_M = 15000.0             # ISA altitude [m]; with the corrected-flow engine model only the Reynolds
                                 # number changes with altitude at a fixed Mach number
CG         = (0.78, 0.0, 0.0)    # moment reference point / CG (x, y, z) [m], body axes
# Mach, alpha, beta, altitude, the back pressure (P_BACK) and the initial field (INIT_FROM) can also be given
# on the command line, e.g.  python3 subsonic_detail_run.py --mach 0.8 --alpha 4 --altitude 5000

# =====================================================================================================
# ASSUMED VALUES
# =====================================================================================================
# --- gas
GAMMA         = 1.4              # constant cp air, also used for the exhaust
R_GAS         = 287.05           # [J/kg/K]
SUTHERLAND_AS = 1.458e-6         # mu = As T^1.5 / (T + Ts)
SUTHERLAND_TS = 110.4            # [K]
PRANDTL_TURB  = 0.9              # turbulent Prandtl number (STAR-CCM+ default)
FREESTREAM_TURB_INTENSITY = 0.01     # STAR-CCM+ default
FREESTREAM_VISC_RATIO     = 10.0     # mu_t / mu, STAR-CCM+ default

# --- reference quantities (estimates from the CAD planform), as supersonic_detail_run.py
S_REF = 0.494                    # exposed wing planform area [m2]
C_REF = 0.503                    # exposed-wing mean chord [m]
B_REF = 1.107                    # span [m]

# --- engine (README section 1)
MDOT_AIR   = 0.2635              # design engine air mass flow [kg/s]
MDOT_FUEL  = 0.0216              # [kg/s] at MDOT_AIR; exhaust = air + fuel at this fuel-air ratio
T0_EXHAUST = 2250.0              # exhaust total temperature [K] (assumed)
EXHAUST_TURB_INTENSITY = 0.01
EXHAUST_VISC_RATIO     = 10.0

# --- engine-face back pressure. The engine runs at the corrected mass flow mdot sqrt(T0) / p0 of the Mach 1.5
#     design point (a fixed-geometry engine on its operating line), so the engine-face Mach number is the
#     design one. The outlet of the straight extension holds the 1D static pressure at that Mach number and
#     the assumed recovery below; the engine-face mass flow and recovery are results.
DESIGN_MACH        = 1.5         # design point: supersonic_detail_run.py calibration (runs/bp30000,
DESIGN_ALTITUDE_M  = 15000.0     # Mach 1.5, alpha 0: 0.2672 kg/s at engine-face recovery 0.7747)
DESIGN_MDOT_FACE   = 0.2672      # [kg/s]
DESIGN_RECOVERY    = 0.7747      # engine-face p0 / p0_inf
PT_RECOVERY_EST    = 0.97        # subsonic engine-face p0 / p0_inf assumed for the back pressure (S-duct
                                 # recovery of design 39 is 0.979; lip losses at low Mach lower it)
CAPTURE_AREA       = 0.0031225   # [m2] inlet capture area (0-results/inlet_design_39/metrics.json),
                                 # for the reported mass flow ratio only
P_BACK             = None        # outlet static pressure: a value [Pa], or None = 1D value above
RAMP_ITERS         = 500         # the back pressure ramps down to P_BACK over these iterations (freestream
P_START_FRAC       = 0.2         # start only) from p0_face - P_START_FRAC * (p0_face - P_BACK), as
                                 # supersonic_detail_run.py and 3-inlet-design. Starting at p_inf instead
                                 # choked the duct (supersonic from the lip to the outlet, Mach 1.46) and
                                 # diverged at Mach 0.9 (local test, Nov 2026)
RECOVERY_TOL       = 0.01        # summary flags |recovery - PT_RECOVERY_EST| above this

# --- exhaust: convergent nozzle of the CAD exhaust area. The 1D exit Mach number at the ambient pressure
#     decides the state: subsonic -> exit pressure left free (zeroGradient, it follows the ambient),
#     supersonic -> the nozzle chokes, sonic exit at the pressure that passes the mass flow (fixed)

# --- walls
WALL_TEMPERATURE = None          # None = adiabatic, a value [K] = isothermal walls

# =====================================================================================================
# GEOMETRY AND MESH (the mesh does not depend on the flow condition and is reused between runs)
# =====================================================================================================
GEOMETRY   = os.path.join(HERE, "uav.stl")     # ASCII, one solid per named face (GEOMETRY_SOLIDS), metres
GEOMETRY_SOLIDS = ("airframe", "inlet_lip", "inlet_duct", "extension", "outlet", "exhaust")
HALF_MODEL = None                # None: half model (symmetry plane y = 0) when BETA_DEG == 0, else full
# Subsonic disturbances decay slowly, so the outer boundary is ~7 vehicle lengths upstream, ~12 downstream
# and ~11 spans to the side, above and below (supersonic_detail_run.py: 1.4 lengths upstream). Vehicle extent:
# x -0.246 .. 1.425, |y| <= 0.554, z -0.07 .. 0.30. Extended to whole BASE_SIZE cells.
DOMAIN     = {"x": (-12.0, 20.0), "y": (-12.0, 12.0), "z": (-12.0, 12.0)}
BASE_SIZE  = 0.4                 # background (largest) cell [m]; all sizes below are BASE_SIZE / 2^n

# surface (target, minimum) cell size [m], as supersonic_detail_run.py (same near-wall resolution over the
# whole flight envelope)
SURFACE_SIZES = {
    "airframe":     (0.003125,  0.0015625),
    "inlet_lip":    (0.000390625, 0.000390625),
    "inlet_duct":   (0.0015625, 0.00078125),  # also the extension
    "engine_ports": (0.0015625, 0.00078125),  # outlet + exhaust
}
FEATURE_EDGE_SIZE = 0.0015625    # cell size along sharp CAD edges [m]
N_CELLS_BETWEEN_LEVELS = 3
RESOLVE_FEATURE_ANGLE  = 30.0    # [deg] surfaces curving more than this get the minimum size

# volume refinement boxes, body axes: name -> (min corner, max corner, cell size [m]). The boxes of
# supersonic_detail_run.py (they hold the vehicle, its near field and the wake / plume), plus refineOuter
# so the cells grow to the large far field over a wider region ahead of and around the vehicle.
REFINEMENT_BOXES = {
    "refineOuter": ((-2.50, -2.50,  -2.50),  (9.00, 2.50,  2.50), 0.1),
    "refineFar":   ((-0.80, -1.40,  -1.10),  (6.00, 1.40,  2.40), 0.05),
    "refineMid":   ((-0.50, -0.85,  -0.60),  (3.60, 0.85,  1.30), 0.025),
    "refineNear":  ((-0.32, -0.65,  -0.30),  (2.40, 0.65,  0.65), 0.0125),
    "refineBody":  ((-0.30, -0.60,  -0.16),  (1.75, 0.60,  0.34), 0.00625),
    "refineInlet": (( 0.62, -0.07,  -0.045), (1.06, 0.07,  0.20), 0.003125),  # lip flow at high suction
    "refinePlume": (( 1.38, -0.15,  -0.15),  (2.60, 0.15,  0.15), 0.00625),
}

PRISM_LAYERS         = 5
PRISM_LAYERS_DUCT    = 0
PRISM_LAYERS_LIP     = 0
PRISM_EXPANSION      = 1.2
PRISM_FINAL_REL      = 0.3       # outermost layer / local surface cell size
PRISM_MIN_REL        = 0.5       # snappyHexMesh removes stacks thinner than this x local cell size

LOCATION_IN_MESH = (-6.0123, 6.0123, 6.0123)   # any point in the fluid with y > 0

# =====================================================================================================
# RUN CONTROL
# =====================================================================================================
CASE_TEMPLATE    = os.path.join(HERE, "case_template_subsonic")
NP               = 100           # MPI processes
MAX_ITERATIONS   = 20000         # SIMPLE iteration cap
WRITE_INTERVAL   = 1000          # [iterations] field writes (the last two are kept)
MONITOR_INTERVAL = 10            # iterations between monitor samples
POLL_S           = 120           # [s] between stopping-criterion checks while the solver runs
# A freestream start runs on first-order upwind convection (system/fvSchemes.firstOrder) for this many
# iterations, then on the second-order system/fvSchemes (swapped at the first poll past it). A case started
# from another solution (INIT_FROM) starts on second order.
FIRST_ORDER_ITERS = 500
# Initial field: "auto" = the finished case on the same mesh closest in (Mach, alpha, beta), converged
# cases first; a case path = that case; None = uniform freestream. The internal fields of its latest
# time are copied into 0/ (same mesh, so cell for cell); boundary conditions stay those of this case.
INIT_FROM        = "auto"

# Stopping: all of these must stay within the given band (max - min) over the final window
ASYMPTOTE = {
    "CD":            1.0e-4,     # 1 drag count
    "CL":            5.0e-4,
    "Cm_pitch":      5.0e-4,
    "mdot_engine_face": 4.0e-3,  # RELATIVE to the 1D engine-face mass flow (that scales with the
                                 # freestream pressure); the free variable, the back pressure is imposed
}
ASYMPTOTE_RELATIVE      = ("mdot_engine_face",)
# Low-q floor: a coefficient band is never asked to be tighter than these forces (complete aircraft) over
# q_inf S_REF (the moment over q_inf S_REF C_REF). At Mach 0.1 one drag count is 0.035 N, below the ~200-iteration
# wiggle of a settled solution: Nov 2026 runs, Mach 0.1 bands 1.4 N drag, 6.0 N lift, 2.0 N m with the 1000-iteration
# means steady to 0.1 N, 0.7 N, 0.2 N m. The floors govern up to Mach 0.9 (qS 6300 N, 1 drag count 0.63 N).
ASYMPTOTE_FORCE_FLOOR   = {"CD": 2.0, "CL": 8.0, "Cm_pitch": 2.5}   # [N, N, N m]
ASYMPTOTE_WINDOW        = 1500   # [iterations]
ASYMPTOTE_MIN_ITER      = 4000   # no check before this iteration
ASYMPTOTE_MIN_ITER_INIT = 2000   # the same, for a case started from another case (INIT_FROM)

WALL_PATCHES = ("airframe", "inlet_lip", "inlet_duct")   # vehicle walls; the extension wall is the engine's

# =====================================================================================================
# Driver
# =====================================================================================================

def main():
    half = BETA_DEG == 0.0 if HALF_MODEL is None else bool(HALF_MODEL)
    geo = geometry_info(GEOMETRY)
    flow = flow_conditions(geo, half)
    print_conditions(flow)

    # 1. Mesh, reused while the mesh settings and the geometry are unchanged. Parallel runs (job array) take
    #    a lock, so only the first builds a missing mesh and the others wait for it and reuse it.
    mesh_dir = f"./detail_mesh_subsonic_{'half' if half else 'full'}"     # the supersonic mesh is detail_mesh_*
    settings = mesh_settings(half)
    import fcntl
    with open(f"{mesh_dir}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
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
    case = f"./detail_M{MACH:g}_a{ALPHA_DEG:g}_b{BETA_DEG:g}_h{ALTITUDE_M / 1000:g}"
    print(f"\n{'=' * 60}\n {os.path.basename(case)}\n{'=' * 60}")
    print("[1/4] Copying the mesh and writing the flow condition ...")
    prepare_case(case, mesh_dir, flow)
    src = find_init_source(case, mesh_dir)
    init_from = initialise_from(case, *src) if src else None
    write_outlet_pressure(case, flow, ramp=not init_from)
    first_order = not init_from and FIRST_ORDER_ITERS > 0
    if first_order:
        shutil.copy(os.path.join(CASE_TEMPLATE, "system", "fvSchemes.firstOrder"),
                    os.path.join(case, "system", "fvSchemes"))
    write_json(case, {"case": os.path.abspath(case), "flow": flow, "mesh": read_json(mesh_dir),
                      "init_from": init_from, "first_order_iters": FIRST_ORDER_ITERS if first_order else 0})

    # 3. Solve
    print("[2/4] Solving (rhoSimpleFoam) ...")
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

def mass_flux(M, p0, T0):
    # Isentropic mass flux [kg/s/m2] at Mach M from the total pressure and temperature
    return (p0 * math.sqrt(GAMMA / (R_GAS * T0)) * M
            * (1.0 + 0.5 * (GAMMA - 1.0) * M * M) ** (-0.5 * (GAMMA + 1.0) / (GAMMA - 1.0)))

def subsonic_mach(mdot, area, p0, T0):
    # Subsonic Mach number of a mass flow through a section (bisection); None above the choked mass flow
    if mdot / area >= mass_flux(1.0, p0, T0):
        return None
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if mass_flux(mid, p0, T0) < mdot / area else (lo, mid)
    return 0.5 * (lo + hi)

def turbulence(U, rho, T, intensity, visc_ratio):
    # (k, omega, nut) from a turbulence intensity and a viscosity ratio
    k = 1.5 * (intensity * U) ** 2
    nut = visc_ratio * sutherland(T) / rho
    return k, k / nut, nut

def total_conditions(M, h):
    """(p, T, p0, T0) of the ISA atmosphere at altitude h and Mach M."""
    T, p = isa_atmosphere(h)
    iso = 1.0 + 0.5 * (GAMMA - 1.0) * M * M
    return p, T, p * iso ** (GAMMA / (GAMMA - 1.0)), T * iso

def flow_conditions(geo, half):
    if not 0.0 < MACH < 1.0:
        raise SystemExit(f"Mach {MACH:g}: this template is for subsonic flight (0 < Mach < 1); "
                         "supersonic_detail_run.py covers Mach > 1")
    p, T, p0_inf, T0_inf = total_conditions(MACH, ALTITUDE_M)
    U = MACH * math.sqrt(GAMMA * R_GAS * T)
    rho = p / (R_GAS * T)
    a, b = math.radians(ALPHA_DEG), math.radians(BETA_DEG)
    k, omega, nut = turbulence(U, rho, T, FREESTREAM_TURB_INTENSITY, FREESTREAM_VISC_RATIO)

    # Engine face: Mach number of the design corrected mass flow (constant on the operating line), the 1D
    # static pressure there at the assumed recovery is the outlet back pressure
    ef = geo["engine_face"]
    _, _, p0_design, T0_design = total_conditions(DESIGN_MACH, DESIGN_ALTITUDE_M)
    M_ef = subsonic_mach(DESIGN_MDOT_FACE, ef["area"], DESIGN_RECOVERY * p0_design, T0_design)
    if M_ef is None:
        raise SystemExit(f"the design mass flow {DESIGN_MDOT_FACE:g} kg/s chokes the {ef['area'] * 1e4:.2f} cm2 "
                         "engine face")
    p0_ef = PT_RECOVERY_EST * p0_inf
    p_ef = p0_ef / (1.0 + 0.5 * (GAMMA - 1.0) * M_ef ** 2) ** (GAMMA / (GAMMA - 1.0))
    mdot_air = mass_flux(M_ef, p0_ef, T0_inf) * ef["area"]
    p_back = p_ef if P_BACK is None else float(P_BACK)

    # Exhaust, fuel at the design fuel-air ratio: convergent nozzle of the exhaust area. Subsonic at the
    # ambient pressure -> exit at p_inf; otherwise choked, sonic exit at the pressure that passes the flow
    mdot_fuel = MDOT_FUEL * mdot_air / MDOT_AIR
    mdot_ex = mdot_air + mdot_fuel
    A_ex = geo["area"]["exhaust"]
    M_amb = exhaust_mach(mdot_ex, A_ex, p, T0_EXHAUST)
    choked = M_amb >= 1.0
    M_ex = 1.0 if choked else M_amb
    p_ex = (mdot_ex / A_ex / math.sqrt(GAMMA * (1.0 + 0.5 * (GAMMA - 1.0)) / (R_GAS * T0_EXHAUST))
            if choked else p)
    T_ex = T0_EXHAUST / (1.0 + 0.5 * (GAMMA - 1.0) * M_ex ** 2)
    U_ex = M_ex * math.sqrt(GAMMA * R_GAS * T_ex)
    rho_ex = p_ex / (R_GAS * T_ex)
    k_ex, omega_ex, _ = turbulence(U_ex, rho_ex, T_ex, EXHAUST_TURB_INTENSITY, EXHAUST_VISC_RATIO)

    return dict(
        mach=MACH, alpha_deg=ALPHA_DEG, beta_deg=BETA_DEG, cg=list(CG), altitude_m=ALTITUDE_M,
        half_model=half, symmetry_factor=2.0 if half else 1.0, gamma=GAMMA, R=R_GAS,
        sutherland_As=SUTHERLAND_AS, sutherland_Ts=SUTHERLAND_TS,
        p_inf=p, T_inf=T, rho_inf=rho, V_inf=U, mu_inf=sutherland(T),
        reynolds_per_m=rho * U / sutherland(T), reynolds_cref=rho * U * C_REF / sutherland(T),
        U_vec=[U * math.cos(a) * math.cos(b), -U * math.sin(b), U * math.sin(a) * math.cos(b)],
        q_inf=0.5 * GAMMA * p * MACH ** 2, p0_inf=p0_inf, T0_inf=T0_inf,
        k_inf=k, omega_inf=omega, nut_inf=nut,
        mdot_air=mdot_air, mdot_fuel=mdot_fuel, mdot_exhaust=mdot_ex, T0_exhaust=T0_EXHAUST, p_exhaust=p_ex,
        engine_face_area=ef["area"], engine_face_x=ef["x"], engine_face_centre=ef["centre"],
        engine_face_radius=ef["radius"], outlet_x=ef["x_outlet"], pt_recovery_est=PT_RECOVERY_EST,
        design_mach=DESIGN_MACH, design_altitude_m=DESIGN_ALTITUDE_M, design_mdot_face=DESIGN_MDOT_FACE,
        design_recovery=DESIGN_RECOVERY,
        mach_engine_face_1d=M_ef, p_engine_face_1d=p_ef, p0_engine_face_1d=p0_ef,
        capture_area=CAPTURE_AREA, mass_flow_ratio_1d=mdot_air / (rho * U * CAPTURE_AREA),
        p_back=p_back, p_back_mode="1d" if P_BACK is None else "set", p_back_set=P_BACK is not None,
        p_back_start=p0_ef - P_START_FRAC * (p0_ef - p_back), ramp_iters=RAMP_ITERS,
        exhaust_mach_ambient_1d=M_amb, exhaust_choked=choked,
        exhaust_mach_1d=M_ex, exhaust_T_1d=T_ex, exhaust_U_1d=U_ex, exhaust_rho_1d=rho_ex,
        exhaust_U_vec=[U_ex * c for c in geo["normal"]["exhaust"]],
        exhaust_thrust_1d=mdot_ex * U_ex + (p_ex - p) * A_ex,
        k_exhaust=k_ex, omega_exhaust=omega_ex,
        **wall_pressure_terms(geo, CG),
    )

def print_conditions(f):
    print(f"Mach {f['mach']:g} | alpha {f['alpha_deg']:g} deg | beta {f['beta_deg']:g} deg | "
          f"CG ({', '.join(f'{c:g}' for c in f['cg'])}) m | {'half' if f['half_model'] else 'full'} model")
    print(f"  {f['altitude_m'] / 1000:g} km ISA: p {f['p_inf']:.1f} Pa, T {f['T_inf']:.2f} K, "
          f"rho {f['rho_inf']:.5f} kg/m3, V {f['V_inf']:.1f} m/s, q {f['q_inf']:.0f} Pa, p0 {f['p0_inf']:.0f} Pa, "
          f"Re/m {f['reynolds_per_m']:.3g}")
    print(f"  engine face x {f['engine_face_x']:.4f} m, {f['engine_face_area'] * 1e4:.2f} cm2 | outlet back pressure "
          f"{f['p_back']:.0f} Pa ({back_pressure_source(f)})")
    print(f"  1D engine face: Mach {f['mach_engine_face_1d']:.3f} (design corrected flow), recovery "
          f"{f['pt_recovery_est']:.4f} -> {f['mdot_air']:.4f} kg/s, p {f['p_engine_face_1d']:.0f} Pa, "
          f"mass flow ratio {f['mass_flow_ratio_1d']:.2f} of the capture stream tube")
    print(f"  exhaust {f['mdot_exhaust']:.4f} kg/s, T0 {f['T0_exhaust']:g} K -> "
          + (f"choked: sonic exit at p {f['p_exhaust']:.0f} Pa (fixed)" if f["exhaust_choked"] else
             f"subsonic exit, 1D Mach {f['exhaust_mach_1d']:.3f} at p_inf (pressure left free)")
          + f", U {f['exhaust_U_1d']:.0f} m/s, gross thrust {f['exhaust_thrust_1d']:.1f} N")

def back_pressure_source(f):
    if f.get("p_back_set"):
        return "set by P_BACK"
    return (f"1D at the design engine-face Mach {f['mach_engine_face_1d']:.3f} with PT_RECOVERY_EST "
            f"{f['pt_recovery_est']:.4f}")

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
    area, normal, centre, vector_area, area_moment = {}, {}, {}, {}, {}
    for name, v in verts.items():
        p = np.asarray(v, dtype=float).reshape(-1, 3, 3)
        n = 0.5 * np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        s = n.sum(axis=0)
        a = np.linalg.norm(n, axis=1)
        area[name] = float(a.sum())
        vector_area[name] = s.tolist()                                    # sum of n dA, n out of the vehicle
        area_moment[name] = np.cross(p.mean(axis=1), n).sum(axis=0).tolist()   # sum of r x n dA about 0
        normal[name] = list(s / np.linalg.norm(s)) if np.linalg.norm(s) > 0 else [0.0, 0.0, 0.0]
        centre[name] = [float(c) for c in (a[:, None] * p.mean(axis=1)).sum(axis=0) / a.sum()]
    # Engine face: the upstream end of the straight extension (along x), the same section as the outlet
    if abs(normal["outlet"][0]) < 0.999:
        raise SystemExit(f"{path}: the outlet must be a plane normal to x (normal {normal['outlet']})")
    x_ef = float(np.asarray(verts["extension"], dtype=float)[:, 0].min())
    engine_face = {"x": x_ef, "centre": [x_ef, centre["outlet"][1], centre["outlet"][2]], "area": area["outlet"],
                   "radius": math.sqrt(area["outlet"] / math.pi), "x_outlet": centre["outlet"][0]}
    return {"area": area, "normal": normal, "engine_face": engine_face, "vector_area": vector_area,
            "area_moment": area_moment}

def wall_pressure_terms(geo, cg):
    """Gauge-pressure terms for the wall forces (complete aircraft, from the STL): per wall patch its vector area
    Sf (out of the fluid, into the vehicle, as OpenFOAM's patch Sf = -the STL normal) and the moment of Sf about
    cg. A uniform p_inf puts p_inf x Sf on a wall; histories() takes that off the absolute-pressure monitors."""
    va, am = geo["vector_area"], geo["area_moment"]
    if va["exhaust"][0] <= 0.0:
        raise SystemExit("uav.stl: the exhaust normal points forward; the STL normals must point out of the vehicle")
    c = np.asarray(cg, dtype=float)
    return {"wall_Sf": {k: [-x for x in va[k]] for k in WALL_PATCHES},
            "wall_Sf_moment": {k: (np.cross(c, va[k]) - np.asarray(am[k])).tolist() for k in WALL_PATCHES},
            "exhaust_Sf": [-x for x in va["exhaust"]],
            # a closed STL sums to zero; what is left bounds the error of the p_inf term
            "stl_closure_area": float(np.linalg.norm(np.sum(list(va.values()), axis=0)))}

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
               "// Written by subsonic_detail_run.py\n"
               f"vertices\n(\n{verts});\n\n"
               f"blocks\n(\n    hex (0 1 2 3 4 5 6 7) ({n[0]} {n[1]} {n[2]}) simpleGrading (1 1 1)\n);\n\n"
               f"boundary\n(\n{boundary});\n")

    lv = lambda key: f"({level(SURFACE_SIZES[key][0])} {level(SURFACE_SIZES[key][1])})"
    write_text(path("system", "include", "meshVars"),
               "// Written by subsonic_detail_run.py\n"
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
               "// Written by subsonic_detail_run.py\n" + "".join(
                   f"{k} {{ type searchableBox; min {fmt_vec(a)}; max {fmt_vec(b)}; }}\n"
                   for k, (a, b, _) in REFINEMENT_BOXES.items()))
    write_text(path("system", "include", "refinementRegions"),
               "// Written by subsonic_detail_run.py\n" + "".join(
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
    # boundary conditions, gas / turbulence models and solver settings from the template, so a reused mesh
    # does not carry older ones (the mesh directory keeps the template of the day it was built)
    shutil.rmtree(os.path.join(case, "0"), ignore_errors=True)
    shutil.copytree(os.path.join(CASE_TEMPLATE, "0"), os.path.join(case, "0"))
    for name in ("thermophysicalProperties", "turbulenceProperties"):
        shutil.copy(os.path.join(CASE_TEMPLATE, "constant", name), os.path.join(case, "constant", name))
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
        "// Written by subsonic_detail_run.py",
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
               f"type            fixedValue;\nvalue           uniform {f['p_exhaust']:.8g};\n"
               if f["exhaust_choked"] else "type            zeroGradient;\n")
    write_text(path("constant", "include", "gasVars"), "\n".join([
        "// Written by subsonic_detail_run.py",
        f"gasMolWeight    {v(8314.46261815324 / R_GAS)};",
        f"gasCp           {v(GAMMA * R_GAS / (GAMMA - 1.0))};",
        f"sutherlandAs    {v(SUTHERLAND_AS)};",
        f"sutherlandTs    {v(SUTHERLAND_TS)};", ""]))
    write_text(path("system", "include", "caseVars"), "\n".join([
        "// Written by subsonic_detail_run.py",
        f"tEnd            {MAX_ITERATIONS};",
        f"tWrite          {WRITE_INTERVAL};",
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
               "// Written by subsonic_detail_run.py: engine back pressure at the extension outlet\n"
               "type            uniformFixedValue;\n" + value + f"value           uniform {f['p_back']:.8g};\n")

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
        if INIT_FROM == "auto" and m.get("stop_reason") not in ("stopping_criteria", "max_iterations", "time_limit"):
            continue        # still running (a parallel job) or failed
        f = m.get("flow", {})
        if abs(f.get("altitude_m", math.inf) - ALTITUDE_M) > 1.0:
            continue        # another altitude: pressure and density levels too far off to start from
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
    print(f"  -> mpirun -np {NP} rhoSimpleFoam -parallel   (log.rhoSimpleFoam)")
    with open(os.path.join(case, "log.rhoSimpleFoam"), "w") as log:
        proc = subprocess.Popen(f"mpirun -np {NP} rhoSimpleFoam -parallel -case {case}", shell=True,
                                executable="/bin/bash", stdout=log, stderr=subprocess.STDOUT)
        reason = "max_iterations"
        m = read_json(case)
        it_min = ASYMPTOTE_MIN_ITER_INIT if m.get("init_from") else ASYMPTOTE_MIN_ITER
        switch_at = m.get("first_order_iters") or None
        while proc.poll() is None:
            time.sleep(POLL_S)
            if reason != "max_iterations":
                continue
            try:
                hist = histories(case, flow)
            except (OSError, ValueError, IndexError, TypeError) as e:   # files mid-write, retry next poll
                print(f"  (monitor read failed: {e})")
                continue
            if switch_at and hist["t"][-1] >= switch_at:
                shutil.copy(os.path.join(CASE_TEMPLATE, "system", "fvSchemes"), os.path.join(case, "system", "fvSchemes"))
                print(f"  iteration {hist['t'][-1]:.0f}: first-order start done, second-order schemes from here", flush=True)
                write_json(case, {**read_json(case), "second_order_from": float(hist["t"][-1])})
                switch_at = None
            status = asymptote_status(hist, it_min, flow)
            print(f"  iteration {hist['t'][-1]:.0f} | " + " | ".join(
                f"{k} band {s['width']:.3g}{' ok' if s['ok'] else ''}" for k, s in status.items()), flush=True)
            if all(s["ok"] for s in status.values()):
                print("  stopping criteria met, writing and stopping")
                sub(os.path.join(case, "system", "controlDict"), r"^stopAt\s+\w+;", "stopAt          writeNow;")
                reason = "stopping_criteria"
    if proc.returncode != 0:
        sh(f"tail -n 30 {case}/log.rhoSimpleFoam")
        return "rhoSimpleFoam failed"
    ok = run_steps(case, [("reconstructPar", f"reconstructPar -latestTime -case {case}")])
    sh(f"rm -rf {case}/processor*")
    return reason if ok else "reconstructPar failed"

def asymptote_status(hist, it_min, flow):
    t = hist["t"]
    window = t >= t[-1] - ASYMPTOTE_WINDOW
    out = {}
    for name, tol in ASYMPTOTE.items():
        if name in ASYMPTOTE_RELATIVE:
            tol *= abs(flow["mdot_air"])
        elif name in ASYMPTOTE_FORCE_FLOOR:
            ref = flow["q_inf"] * S_REF * (C_REF if name == "Cm_pitch" else 1.0)
            tol = max(tol, ASYMPTOTE_FORCE_FLOOR[name] / ref)
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
    # The monitors integrate the absolute pressure (pRef 0). The walls alone are open (at the engine face and the
    # exhaust), so that carries p_inf x their vector area, no aerodynamic force: ~ p_inf x (A_exhaust - A_face)
    # axially, 290 N at sea level. Wall forces and moments from here on are with the gauge pressure p - p_inf.
    Fx_abs, pinf = Fx, f["p_inf"]
    Sf = np.sum([f["wall_Sf"][k] for k in WALL_PATCHES], axis=0)
    SfM = np.sum([f["wall_Sf_moment"][k] for k in WALL_PATCHES], axis=0)
    Fx, Fy, Fz = Fx - pinf * Sf[0], Fy - pinf * Sf[1], Fz - pinf * Sf[2]
    Mx, My, Mz = Mx - pinf * SfM[0], My - pinf * SfM[1], Mz - pinf * SfM[2]
    drag = Fx * ca * cb - Fy * sb + Fz * sa * cb
    lift = -Fx * sa + Fz * ca
    # aircraft sign convention: Cm > 0 nose up, Cl > 0 starboard wing down, Cn > 0 nose to starboard
    h = {"t": t, "Fx_body": Fx, "Fz_body": Fz, "My_cg": My, "Drag": drag, "Lift": lift,
         "CD": drag / qS, "CL": lift / qS, "L_over_D": lift / drag, "Cm_pitch": My / (qS * C_REF)}
    # airframe alone, the external drag (the duct walls carry the internal suction of the engine flow)
    Fa = [S * on_t("forces_airframe", "force.dat", i) - pinf * f["wall_Sf"]["airframe"][i] for i in range(3)]
    h["Drag_airframe"] = Fa[0] * ca * cb - Fa[1] * sb + Fa[2] * sa * cb
    h["CD_airframe"] = h["Drag_airframe"] / qS
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
    # Stream thrusts reported with p - p_inf (the exhaust one is the gross thrust). The net force stays the closed
    # balance in absolute pressure, = Fx_body + StreamThrust_engine_face - StreamThrust_exhaust up to p_inf x the
    # extension wall's axial area (zero for the straight extension) and the STL closure (summary).
    h["StreamThrust_engine_face"] = I["engine_face"] - pinf * f["engine_face_area"]
    h["StreamThrust_exhaust"] = -(I["exhaust"] - pinf * f["exhaust_Sf"][0])
    h["Fx_net_with_engine"] = Fx_abs + I["engine_face"] + I["exhaust"]
    h["Fx_body_abs"] = Fx_abs          # absolute-pressure wall force, as the monitors (for reference)
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
    status = asymptote_status(hist, it_min, flow)
    out = {"iterations": int(hist["t"][-1]),
           "reports": {k: float(np.mean(y[window])) for k, y in hist.items() if k != "t"},
           "report_std": {k: float(np.std(y[window])) for k, y in hist.items() if k != "t"},
           "stopping_criteria": status, "converged": all(s["ok"] for s in status.values()),
           "breakdown": []}
    # Per-wall axial force breakdown (complete aircraft), as uav_breakdown.java
    for patch in WALL_PATCHES:
        t, d = read_dat(case, f"forces_{patch}", "force.dat")
        mean = flow["symmetry_factor"] * d[t >= t[-1] - ASYMPTOTE_WINDOW].mean(axis=0)
        dp = [flow["p_inf"] * s for s in flow["wall_Sf"][patch]]      # p_inf part, taken off as in histories()
        out["breakdown"].append({"boundary": patch, "Fx_total": mean[0] - dp[0], "Fx_pressure": mean[3] - dp[0],
                                 "Fx_shear": mean[6], "Fz_total": mean[2] - dp[2]})
    return out

# =====================================================================================================
# Plots (the STAR-CCM+ monitor plots)
# =====================================================================================================

PLOTS = [   # (title, series, y label); series missing from the histories (half model) are skipped
    ("CD", ["CD"], "CD"), ("CD airframe", ["CD_airframe"], "CD airframe"), ("CL", ["CL"], "CL"), ("CY", ["CY"], "CY"),
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
        f"{f['altitude_m'] / 1000:g} km | CG {f['cg']} | {'half' if f['half_model'] else 'full'} model",
        bar,
        f"  stopped by {m['stop_reason']} at iteration {m['iterations']}, solve time {m['solve_time_s']:.0f} s, "
        f"{m['mesh'].get('mesh_cells', '?')} cells, Re (C_REF) {f['reynolds_cref']:.3g}",
        f"  initial field: {m['init_from'] or 'uniform freestream'}"
        + (f", first order to iteration {m['second_order_from']:.0f}" if m.get("second_order_from") else ""),
        "", "== Stopping criteria (band over the final window) ==",
    ]
    for k, s in m["stopping_criteria"].items():
        lines.append(f"  {k + ' asymptote':<34s} band {s['width']:12.4g}  tol {s['tol']:10.4g}  "
                     f"satisfied={s['ok']}")
    lines += ["", f"== Reports (mean over the final {ASYMPTOTE_WINDOW} iterations, +/- std) =="]
    if "wall_Sf" in f:
        off = f["p_inf"] * sum(f["wall_Sf"][k][0] for k in WALL_PATCHES)
        lines.append(f"  wall forces with p - p_inf: {off:+.1f} N of p_inf on the open walls taken off Fx_body "
                     f"(Fx_body_abs); STL closure {f['stl_closure_area'] * 1e4:.3g} cm2 = "
                     f"+/- {f['p_inf'] * f['stl_closure_area']:.2g} N")
    lines += [f"  {k:<32s} {rep[k]:14.6g}   +/- {std[k]:.3g}" for k in rep]
    lines += ["", "== Axial force breakdown (complete aircraft, p - p_inf) [N] ==",
              f"  {'boundary':<14s} {'Fx_total':>10s} {'Fx_press':>10s} {'Fx_shear':>10s} {'Fz_total':>10s}"]
    lines += [f"  {r['boundary']:<14s} {r['Fx_total']:10.3f} {r['Fx_pressure']:10.3f} "
              f"{r['Fx_shear']:10.3f} {r['Fz_total']:10.3f}" for r in m["breakdown"]]
    if "mdot_engine_face" in rep:
        ratio = rep["mdot_engine_face"] / f["mdot_air"]
        rec = rep["PressureRecovery_engine_face"]
        lines += ["", "== Engine face (back pressure fixed at the extension outlet, mass flow a result) ==",
                  f"  outlet back pressure {f['p_back']:.0f} Pa ({back_pressure_source(f)})",
                  f"  mass flow {rep['mdot_engine_face']:.4f} kg/s, 1D estimate {f['mdot_air']:.4f} "
                  f"({100 * (ratio - 1):+.2f} %), {rep['mdot_engine_face'] / (f['rho_inf'] * f['V_inf'] * f['capture_area']):.2f}"
                  " x the capture stream tube",
                  f"  recovery {rec:.4f} (assumed {f['pt_recovery_est']:.4f}), engine-face Mach "
                  f"{rep['Mach_engine_face']:.3f} (design {f['mach_engine_face_1d']:.3f})"]
        if abs(rec - f["pt_recovery_est"]) > RECOVERY_TOL and not f.get("p_back_set", False):
            lines.append(f"  OFF DESIGN CORRECTED FLOW: rerun with PT_RECOVERY_EST = {rec:.4f} (the recovery of this "
                         "run) to put the engine face back on the design Mach number")
        lines.append(f"  exhaust: {f['mdot_exhaust']:.4f} kg/s, "
                     + (f"choked, sonic exit at {f['p_exhaust']:.0f} Pa" if f["exhaust_choked"] else
                        f"subsonic exit (1D Mach {f['exhaust_mach_1d']:.3f})"))
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

def post_only(case):
    """The reports, monitor plots and summary of a finished case again, from its monitor files (no mesh, solve or
    ParaView; the y+ and field checks of detail_post.py stay those of the run)."""
    m = read_json(case)
    if "flow" not in m or "stop_reason" not in m:
        raise SystemExit(f"{case}: no metrics.json of a finished run")
    with open(GEOMETRY, "rb") as fh:
        md5 = hashlib.md5(fh.read()).hexdigest()
    if m.get("mesh", {}).get("mesh_settings", {}).get("geometry_md5") not in (None, md5):
        print(f"  warning: {os.path.basename(GEOMETRY)} has changed since the mesh of {case} was built")
    m["flow"] = flow = {**m["flow"], **wall_pressure_terms(geometry_info(GEOMETRY), m["flow"]["cg"])}
    hist = histories(case, flow)
    m.update(reports(case, flow, hist))
    write_json(case, m)
    write_history_csv(case, hist)
    plot_monitors(case, hist, m)
    plot_residuals(case)
    text = format_summary(m)
    print(text)
    with open(os.path.join(case, "summary.txt"), "w") as fh:
        fh.write(text)
    return 0 if m["converged"] else 2

def parse_args(argv):
    """Command-line overrides of the INPUTS (anything not given keeps its value above)."""
    import argparse
    ap = argparse.ArgumentParser(description="Detail subsonic run (rhoSimpleFoam) of uav.stl")
    ap.add_argument("--mach", type=float, help=f"freestream Mach number, 0 < Mach < 1 (default {MACH:g})")
    ap.add_argument("--alpha", type=float, help=f"angle of attack [deg] (default {ALPHA_DEG:g})")
    ap.add_argument("--beta", type=float, help=f"sideslip [deg] (default {BETA_DEG:g})")
    ap.add_argument("--altitude", type=float, help=f"ISA altitude [m] (default {ALTITUDE_M:g})")
    ap.add_argument("--recovery", type=float, help="assumed engine-face recovery for the back pressure "
                                                   f"(default {PT_RECOVERY_EST:g})")
    ap.add_argument("--p-back", help=f"outlet back pressure: a value [Pa] or '1d' (default {P_BACK or '1d'})")
    ap.add_argument("--init-from", help="initial field: 'auto', 'none' or a finished case directory "
                                        f"(default {INIT_FROM})")
    ap.add_argument("--post-only", metavar="CASE", help="redo the reports, monitor plots and summary of a finished "
                                                       "case from its monitor files (no mesh, solve or ParaView)")
    a = ap.parse_args(argv)
    g = globals()
    for opt, name in (("mach", "MACH"), ("alpha", "ALPHA_DEG"), ("beta", "BETA_DEG"), ("altitude", "ALTITUDE_M"),
                      ("recovery", "PT_RECOVERY_EST")):
        if getattr(a, opt) is not None:
            g[name] = getattr(a, opt)
    if a.p_back is not None:
        g["P_BACK"] = None if a.p_back.lower() == "1d" else float(a.p_back)
    if a.init_from is not None:
        g["INIT_FROM"] = None if a.init_from.lower() == "none" else a.init_from
    return a.post_only

if __name__ == "__main__":
    post = parse_args(sys.argv[1:])
    sys.exit(post_only(post) if post else main())
