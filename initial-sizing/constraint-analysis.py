import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider, Button
from matplotlib.lines import Line2D

G = 9.80665
S_REF = 0.60        # m^2, reference area the CFD Cd is normalised on
K_SUB = 0.15        # subsonic induced-drag factor, 1/(pi*AR*e)
BETA_FLOOR = 0.4583 # beta at M = 1.10; linear theory is invalid closer to M = 1
M_DD, M_PEAK = 0.95, 1.10

# Atmosphere model
def isa(h):
    T0, L, R, g, rho0 = 288.15, 0.0065, 287.05, 9.80665, 1.225
    if h <= 11000:
        T   = T0 - L * h
        rho = rho0 * (T / T0) ** (g / (L * R))
    else:
        T   = 216.65
        rho = 0.36392 * np.exp(-g * (h - 11000) / (R * T))
    return rho, np.sqrt(1.4 * R * T)

# Drag model
def k_induced(M, suction):
    if M <= 1.0:
        return K_SUB
    beta = max(np.sqrt(M**2 - 1.0), BETA_FLOOR)
    return (beta / 4.0) * (1.0 - suction)

def cd_zero_lift(M, cd0_sub, cd_ref, M_ref, n):
    dcdw_ref = max(cd_ref - cd0_sub, 0.0)
    beta_ref = max(np.sqrt(max(M_ref**2 - 1.0, 0.0)), BETA_FLOOR)
    if M <= M_DD:
        return cd0_sub
    if M < M_PEAK:
        t = (M - M_DD) / (M_PEAK - M_DD)
        return cd0_sub + dcdw_ref * (beta_ref / BETA_FLOOR) ** n * (t * t * (3 - 2 * t))
    beta = max(np.sqrt(M**2 - 1.0), BETA_FLOOR)
    return cd0_sub + dcdw_ref * (beta_ref / beta) ** n

def thrust_avail(h, M, T_ref, h_ref, M_ref):
    rho, _ = isa(h)
    rho_r, _ = isa(h_ref)
    return T_ref * (rho / rho_r) * (M / M_ref)

# Constraint analysis
def compute_constraints(WS, p):
    M_cr, h_cr = p["mach"], p["alt_cr"]
    rho_cr, a_cr = isa(h_cr)
    V_cr = M_cr * a_cr
    q_cr = 0.5 * rho_cr * V_cr**2

    # Zero-lift drag AREA (m^2), split fixed vs wing-proportional. Holding CD0
    # constant as S varies silently assumes all drag scales with wing area; for a
    # fuselage-dominated vehicle it does not.
    f0 = p["cd_ref"] * S_REF
    f0_wing = p["wing_share"] * f0
    f0_fix = f0 - f0_wing

    out = {}

    def add(label, M, h, extra_TW):
        rho, a = isa(h)
        V = M * a
        q = 0.5 * rho * V**2
        cd0 = cd_zero_lift(M, p["cd0_sub"], p["cd_ref"], M_cr, p["wave_n"])
        K = k_induced(M, p["suction"])
        # split cd0 into fixed/wing drag area in the same proportion
        scale = cd0 / p["cd_ref"] if p["cd_ref"] > 0 else 0.0
        A = q * f0_fix * scale
        B = q * f0_wing * scale / (WS * S_REF) + K * WS / q + extra_TW
        W_max = np.where(B > 0, (thrust_avail(h, M, p["T_ref"], h_cr, M_cr) - A) / B, np.nan)
        out[label] = W_max / G

    add(f"Cruise (M={M_cr:.2f})", M_cr, h_cr, 0.0)
    add(f"Transonic accel (M={p['M_acc']:.2f})", p["M_acc"], p["alt_acc"], p["dVdt"] / G)
    add("Supersonic climb", M_cr, h_cr, p["RC"] / V_cr)
    add("Service ceiling", M_cr, p["alt_ceil"], 0.508 / (M_cr * isa(p["alt_ceil"])[1]))

    # Rail launch removes the stall constraint. W/S is bounded instead by the
    # maximum usable CL of a thin supersonic wing at sensible alpha.
    out["WS_max"] = q_cr * p["CL_use"]
    out["_ctx"] = (V_cr, q_cr, rho_cr, f0_fix, f0_wing)
    return out

def best_design(p):
    res = compute_constraints(WS, p)
    WS_max = res.pop("WS_max")
    ctx = res.pop("_ctx")
    env = np.nanmin(np.vstack([res[l] for l in res]), axis=0)
    env = np.where((WS <= WS_max) & (env > 0), env, np.nan)
    idx = None if np.all(np.isnan(env)) else int(np.nanargmax(env))
    return res, WS_max, ctx, env, idx

def cd_to_close(p, lo=0.0005, hi=0.30, iters=40):
    q = dict(p)
    q["cd_ref"] = lo
    _, _, _, _, i = best_design(q)
    if i is None:
        return None
    lo_ok = best_design(q)[3][i] >= p["MTOW"]
    if not lo_ok:
        return None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        q["cd_ref"] = mid
        _, _, _, env, i = best_design(q)
        if i is not None and env[i] >= p["MTOW"]:
            lo = mid
        else:
            hi = mid
    return lo

STYLE = {
    "Cruise":           {"color": "#1a5fa8", "ls": "-",  "lw": 2.2},
    "Transonic accel":  {"color": "#d14e8a", "ls": "-.", "lw": 1.8},
    "Supersonic climb": {"color": "#1a8c5b", "ls": "--", "lw": 1.8},
    "Service ceiling":  {"color": "#7f5cc9", "ls": ":",  "lw": 1.8},
}

def get_style(label):
    for key, sty in STYLE.items():
        if label.startswith(key):
            return sty
    return {"color": "#666666", "ls": "-", "lw": 1.5}

DEFAULTS = dict(
    MTOW      = 18.7,     # kg   -- target, drawn as a line only
    T_ref     = 344.0,    # N    -- thrust at the cruise condition
    alt_cr    = 15000,    # m    -- matches the CFD design point
    mach      = 1.50,
    cd_ref    = 0.11628,  # measured: 1331.15 N / (19080 Pa * 0.60 m^2)
    cd0_sub   = 0.012,
    wing_share= 0.40,     # fraction of zero-lift drag that scales with wing area
    wave_n    = 1.00,     # transonic pinch severity; n=1 gives a 2.4x peak
    suction   = 0.00,     # 0 = supersonic LE (K = beta/4), 0.6 = subsonic LE
    CL_use    = 0.20,     # max usable cruise CL -> right-hand W/S bound
    M_acc     = 1.10,
    alt_acc   = 12000,    # m
    dVdt      = 2.0,      # m/s^2
    RC        = 30,       # m/s
    alt_ceil  = 18000,    # m
)

WS = np.linspace(100, 5000, 800)   # N/m²

fig = plt.figure(figsize=(15, 8.5), facecolor="#f7f7f5")
fig.canvas.manager.set_window_title("Supersonic UAV — Constraint Analysis (fixed thrust)")

ax = fig.add_axes([0.06, 0.09, 0.52, 0.84])
ax.set_facecolor("#fafaf8")
ax.set_xlim(WS[0], WS[-1])
ax.set_xlabel("Wing Loading  W/S  [N/m²]", fontsize=12)
ax.set_ylabel("Maximum sustainable mass  [kg]", fontsize=12)
ax.set_title("Fixed-thrust constraint diagram — rail launched", fontsize=13, fontweight="bold", pad=10)
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5, color="#cccccc")
ax.tick_params(labelsize=10)

panel_bg = fig.add_axes([0.61, 0.01, 0.38, 0.98])
panel_bg.set_facecolor("#eeede8")
panel_bg.set_xticks([]); panel_bg.set_yticks([])
for sp in panel_bg.spines.values():
    sp.set_edgecolor("#cccccc")

SLIDERS_DEF = [
    ("Target MTOW [kg]",    "MTOW",      2,     60,     0.1,   ),
    ("Thrust @ cruise [N]", "T_ref",     50,    4000,   10,    ),
    ("Cruise Alt [m]",      "alt_cr",    5000,  25000,  500,   ),
    ("Cruise Mach",         "mach",      1.05,  2.50,   0.05,  ),
    ("CD @ cruise (CFD)",   "cd_ref",    0.005, 0.150,  0.001, ),
    ("CD0 subsonic",        "cd0_sub",   0.005, 0.040,  0.001, ),
    ("Wing share of CD0",   "wing_share",0.0,   1.0,    0.05,  ),
    ("Wave decay exp n",    "wave_n",    0.0,   2.0,    0.1,   ),
    ("LE suction factor",   "suction",   0.0,   0.60,   0.05,  ),
    ("Usable cruise CL",    "CL_use",    0.05,  0.60,   0.01,  ),
    ("Accel Mach",          "M_acc",     1.05,  1.40,   0.05,  ),
    ("Accel Alt [m]",       "alt_acc",   5000,  20000,  500,   ),
    ("Accel dV/dt [m/s²]",  "dVdt",      0.5,   10.0,   0.5,   ),
    ("Climb rate [m/s]",    "RC",        5,     80,     1,     ),
    ("Service Ceiling [m]", "alt_ceil",  5000,  25000,  500,   ),
]

slider_objs = {}
s_h, s_gap, s_x, s_w, top = 0.040, 0.0085, 0.755, 0.185, 0.945

fig.text(0.815, 0.978, "Design Requirements", ha="center", va="top",
         fontsize=11, fontweight="bold", color="#333333")

for i, (lbl, key, vmin, vmax, vstep) in enumerate(SLIDERS_DEF):
    y_pos = top - i * (s_h + s_gap)
    sax = fig.add_axes([s_x, y_pos - s_h, s_w, s_h - 0.004])
    sl = Slider(sax, lbl, vmin, vmax, valinit=DEFAULTS[key], valstep=vstep, color="#4a86c8", track_color="#d0d0cc")
    sl.label.set_fontsize(8.5); sl.label.set_color("#444444")
    sl.valtext.set_fontsize(8.5); sl.valtext.set_color("#222222")
    slider_objs[key] = sl

btn_ax = fig.add_axes([0.755, 0.012, 0.110, 0.030])
btn_reset = Button(btn_ax, "Reset", color="#deded8", hovercolor="#c8c8c0")
btn_reset.label.set_fontsize(9)

line_objects  = {}
cl_line       = ax.axvline(x=1000, color="#e24b4a", lw=2.2, ls="-", zorder=5)
cl_fill       = [None]
feasible_fill = [None]
mtow_line     = ax.axhline(y=DEFAULTS["MTOW"], color="#444444", lw=1.6, ls="--", zorder=5)
envelope_line = ax.plot([], [], color="#1db954", lw=2.5, zorder=6)[0]
design_pt     = ax.plot([], [], "k*", ms=14, zorder=10)[0]
dp_text       = ax.text(0, 0, "", fontsize=8.5, color="#111111", va="top", ha="right", zorder=11, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#aaaaaa", alpha=0.92))
info_text     = ax.text(0.01, 0.01, "", transform=ax.transAxes, fontsize=8, color="#555555", va="bottom", fontfamily="monospace")

def get_params():
    return {key: sl.val for key, sl in slider_objs.items()}

def update(_=None):
    p = get_params()
    res, WS_max, ctx, env, idx = best_design(p)
    V_cr, q_cr, rho_cr, f0_fix, f0_wing = ctx

    for label, m_arr in res.items():
        sty = get_style(label)
        m_clip = np.where(m_arr < 0, np.nan, m_arr)
        if label in line_objects:
            line_objects[label].set_data(WS, m_clip)
        else:
            line_objects[label] = ax.plot(WS, m_clip, color=sty["color"], ls=sty["ls"], lw=sty["lw"], zorder=4)[0]

    active = set(res.keys())
    for lbl, ln in line_objects.items():
        ln.set_visible(lbl in active)

    cl_line.set_xdata([WS_max, WS_max])
    if cl_fill[0] is not None:
        cl_fill[0].remove()
    cl_fill[0] = ax.axvspan(WS_max, WS[-1], alpha=0.08, color="#e24b4a", zorder=1)

    envelope_line.set_data(WS, env)
    mtow_line.set_ydata([p["MTOW"], p["MTOW"]])

    if feasible_fill[0] is not None:
        feasible_fill[0].remove(); feasible_fill[0] = None
    valid = ~np.isnan(env)
    if idx is not None:
        feasible_fill[0] = ax.fill_between(WS, 0, env, where=valid,
                                           color="#1db954", alpha=0.18, zorder=2)
        ax.set_ylim(0, max(1.35 * np.nanmax(env), 1.25 * p["MTOW"]))
        WS_dp, m_dp = WS[idx], env[idx]
        design_pt.set_data([WS_dp], [m_dp])
        S_dp = m_dp * G / WS_dp
        binding = min(active, key=lambda l: res[l][idx])
        dp_text.set_position((WS[-1] * 0.985, ax.get_ylim()[1] * 0.97))
        dp_text.set_text(f"W/S = {WS_dp:.0f} N/m²   m = {m_dp:.1f} kg\n"
                         f"S   = {S_dp:.3f} m²\nbinding: {binding}")
        dp_text.set_visible(True)
        headroom = f"max mass {m_dp:.1f} kg vs target {p['MTOW']:.1f} kg"
    else:
        design_pt.set_data([], []); dp_text.set_visible(False)
        ax.set_ylim(0, max(1.25 * p["MTOW"], 5))
        headroom = "NO FEASIBLE DESIGN — fixed drag alone exceeds available thrust"

    # What cruise CD would close the target MTOW, against whichever constraint binds?
    cd_needed = cd_to_close(p)
    cd_str = (f"{cd_needed:.4f}  ({cd_needed / p['cd_ref']:.2f}x current {p['cd_ref']:.4f})"
              if cd_needed else "unreachable at any CD — thrust or MTOW must change")

    T_cr = thrust_avail(p["alt_cr"], p["mach"], p["T_ref"], p["alt_cr"], p["mach"])
    info_text.set_text(
        f"Cruise:  V = {V_cr:.0f} m/s   q = {q_cr:.0f} Pa   rho = {rho_cr:.4f} kg/m³\n"
        f"Drag area: fixed {f0_fix:.4f} m² -> {q_cr*f0_fix:6.0f} N   "
        f"wing {f0_wing:.4f} m²   T_avail = {T_cr:.0f} N\n"
        f"{headroom}\n"
        f"CD @ cruise needed to close at target MTOW: {cd_str}"
    )

    handles = [Line2D([0], [0], color=get_style(l)["color"], ls=get_style(l)["ls"],
                      lw=get_style(l)["lw"], label=l) for l in line_objects if line_objects[l].get_visible()]
    handles += [
        Line2D([0], [0], color="#e24b4a", lw=2.2, label=f"Usable-CL limit (W/S = {WS_max:.0f})"),
        Line2D([0], [0], color="#1db954", lw=2.5, label="Binding envelope"),
        mpatches.Patch(facecolor="#1db954", alpha=0.35, label="Feasible design space"),
        Line2D([0], [0], color="#444444", lw=1.6, ls="--", label="Target MTOW"),
        Line2D([0], [0], marker="*", color="#111111", ms=10, ls="none", label="Max-W/S design point"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8.5, framealpha=0.92,
              edgecolor="#cccccc", handlelength=2.2)
    fig.canvas.draw_idle()

def reset(_):
    for key, sl in slider_objs.items():
        sl.set_val(DEFAULTS[key])


for sl in slider_objs.values():
    sl.on_changed(update)
btn_reset.on_clicked(reset)

update()
plt.show()