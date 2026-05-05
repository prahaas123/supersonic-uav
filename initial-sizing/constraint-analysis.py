import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider, Button
from matplotlib.lines import Line2D

# Atmosphere model
def isa(h):
    T0, L, R, g, rho0 = 288.15, 0.0065, 287.05, 9.80665, 1.225
    if h <= 11000:
        T   = T0 - L * h
        rho = rho0 * (T / T0) ** (g / (L * R))
        a   = np.sqrt(1.4 * R * T)
    else:
        T   = 216.65
        rho = 0.36392 * np.exp(-g * (h - 11000) / (R * T))
        a   = np.sqrt(1.4 * R * T)
    return rho, a


# Constraint analysis
def compute_constraints(WS, p):
    g    = 9.81
    mu   = 0.04
    rho_sl, _     = isa(0)
    rho_cr, a_cr  = isa(p["alt_cr"])
    rho_ce, a_ce  = isa(p["alt_ceil"])
    V_cr   = p["mach"] * a_cr
    q_cr   = 0.5 * rho_cr * V_cr**2

    # Supersonic aero
    CD0_sup = p["CD0"] + p["dCD_w"]
    K_sup   = max(p["K"] * 0.65, 0.06) # linear theory approximation for M>1

    # Stall speed constraint
    WS_stall = 0.5 * rho_sl * p["V_stall"]**2 * p["CL_max"]
    results = {"WS_stall": WS_stall}

    # Cruise speed constraint
    if p["mach"] > 1.0:
        TW_cruise = (q_cr * CD0_sup) / WS + (K_sup * WS) / q_cr
    else:
        TW_cruise = (q_cr * p["CD0"]) / WS + (p["K"] * WS) / q_cr
    results["Cruise (M={:.2f})".format(p["mach"])] = TW_cruise

    # Rate of climb constraint
    V_rc  = np.sqrt((2 / rho_cr) * np.sqrt(p["K"] / p["CD0"]) * WS)
    q_rc  = 0.5 * rho_cr * V_rc**2
    TW_rc = p["RC"] / V_rc + (q_rc * p["CD0"]) / WS + (p["K"] * WS) / q_rc
    results["Rate of climb"] = TW_rc

    # Altitude ceiling constraint
    RC_ceil = 0.508 # 100 ft/min residual climb
    V_opt_c = np.sqrt((2 / rho_ce) * np.sqrt(p["K"] / p["CD0"]) * WS)
    q_ceil  = 0.5 * rho_ce * V_opt_c**2
    TW_ceil = RC_ceil / V_opt_c + (q_ceil * p["CD0"]) / WS + (p["K"] * WS) / q_ceil
    results["Service ceiling"] = TW_ceil

    # Max speed constraint
    TW_dash = (q_cr * CD0_sup) / WS + (K_sup * WS) / q_cr
    results["Supersonic dash"] = TW_dash

    return results

# Constraint diagram style and formatting
STYLE = {
    "Rate of climb":     {"color": "#1a8c5b", "ls": "--", "lw": 1.8},
    "Service ceiling":   {"color": "#7f5cc9", "ls": ":",  "lw": 1.8},
    "Supersonic dash":   {"color": "#d14e8a", "ls": "-.", "lw": 1.8},
    "Cruise":            {"color": "#1a5fa8", "ls": "-",  "lw": 2.2},
}

def get_style(label):
    for key, sty in STYLE.items():
        if label.startswith(key):
            return sty
    return {"color": "#666666", "ls": "-", "lw": 1.5}

DEFAULTS = dict(
    alt_cr   = 12000,   # m
    mach     = 1.50,
    V_stall  = 40,      # m/s
    CD0      = 0.012,
    dCD_w    = 0.040,
    K        = 0.15,
    CL_max   = 1.00,
    RC       = 30,      # m/s
    alt_ceil = 20000,   # m
)

WS = np.linspace(20, 1200, 600)   # N/m²
TW_MAX = 2.0                      # y-axis ceiling for shading

fig = plt.figure(figsize=(15, 8), facecolor="#f7f7f5")
fig.canvas.manager.set_window_title("Supersonic UAV — Constraint Analysis")

# Main graph
ax = fig.add_axes([0.05, 0.08, 0.53, 0.86])
ax.set_facecolor("#fafaf8")
ax.set_xlim(20, 1200)
ax.set_ylim(0, TW_MAX)
ax.set_xlabel("Wing Loading  W/S  [N/m²]", fontsize=12)
ax.set_ylabel("Thrust-to-Weight Ratio  T/W", fontsize=12)
ax.set_title("Supersonic UAV — Constraint Analysis Diagram", fontsize=13, fontweight="bold", pad=10)
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5, color="#cccccc")
ax.tick_params(labelsize=10)

# Slider panel
fig.add_axes([0.62, 0.0, 0.38, 1.0]).set_visible(False)
panel_bg = plt.axes([0.61, 0.01, 0.38, 0.98])
panel_bg.set_facecolor("#eeede8")
panel_bg.set_xticks([]); panel_bg.set_yticks([])
for sp in panel_bg.spines.values():
    sp.set_edgecolor("#cccccc")

SLIDERS_DEF = [
    # (label,              key,       min,  max,    step, fmt)
    ("Cruise Alt [m]",     "alt_cr",  5000, 25000,  500,  "{:.0f}"),
    ("Cruise Mach",        "mach",    0.3,  2.5,    0.05, "{:.2f}"),
    ("Stall Speed [m/s]",  "V_stall", 15,   80,     1,    "{:.0f}"),
    ("CD0 (subsonic)",     "CD0",     0.005,0.040,  0.001,"{:.3f}"),
    ("ΔCD wave drag",      "dCD_w",   0.01, 0.12,   0.005,"{:.3f}"),
    ("K (induced drag)",   "K",       0.05, 0.40,   0.01, "{:.2f}"),
    ("CL_max (stall)",     "CL_max",  0.6,  1.8,    0.05, "{:.2f}"),
    ("Rate of Climb [m/s]","RC",      5,    80,     1,    "{:.0f}"),
    ("Service Ceiling [m]","alt_ceil",5000, 25000,  500,  "{:.0f}"),
]

slider_objs = {}
sax_list    = []
n_sl   = len(SLIDERS_DEF)
s_h    = 0.048
s_gap  = 0.010
s_x    = 0.740 
s_w    = 0.200 
top    = 0.94

fig.text(0.81, 0.975, "Design Requirements", ha="center", va="top",
         fontsize=11, fontweight="bold", color="#333333")

for i, (lbl, key, vmin, vmax, vstep, fmt) in enumerate(SLIDERS_DEF):
    y_pos = top - i * (s_h + s_gap)
    sax   = fig.add_axes([s_x, y_pos - s_h, s_w, s_h - 0.004])
    sl    = Slider(sax, lbl, vmin, vmax,
                   valinit=DEFAULTS[key], valstep=vstep,
                   color="#4a86c8", track_color="#d0d0cc")
    sl.label.set_fontsize(9)
    sl.label.set_color("#444444")
    sl.valtext.set_fontsize(9)
    sl.valtext.set_color("#222222")
    slider_objs[key] = sl
    sax_list.append(sax)

# Reset button
btn_ax = fig.add_axes([0.750, 0.008, 0.110, 0.034])
btn_reset = Button(btn_ax, "Reset", color="#deded8", hovercolor="#c8c8c0")
btn_reset.label.set_fontsize(9)

line_objects   = {}
stall_line     = ax.axvline(x=500, color="#e24b4a", lw=2.2, ls="-",
                            label="Stall speed limit", zorder=5)
stall_fill    = [None]
feasible_fill = [None]
envelope_line  = ax.plot([], [], color="#1db954", lw=2.5, ls="-",
                         label="Constraint envelope", zorder=6)[0]
design_pt      = ax.plot([], [], "k*", ms=14, zorder=10,
                         label="Min-T/W design point")[0]
dp_text        = ax.text(0, 0, "", fontsize=8.5, color="#111111",
                         va="bottom", ha="left", zorder=11,
                         bbox=dict(boxstyle="round,pad=0.3",
                                   facecolor="white", edgecolor="#aaaaaa",
                                   alpha=0.9))

info_text = ax.text(0.01, 0.01, "", transform=ax.transAxes,
                    fontsize=8, color="#555555", va="bottom",
                    fontfamily="monospace")

def get_params():
    return {key: sl.val for key, sl in slider_objs.items()}

def update(_=None):
    global feasible_fill
    p   = get_params()
    res = compute_constraints(WS, p)
    WS_stall = res.pop("WS_stall")

    for label, TW_arr in res.items():
        sty = get_style(label)
        TW_clipped = np.where(TW_arr < 0, np.nan, TW_arr)
        if label in line_objects:
            line_objects[label].set_ydata(TW_clipped)
            line_objects[label].set_label(label)
        else:
            ln, = ax.plot(WS, TW_clipped,
                          color=sty["color"], ls=sty["ls"], lw=sty["lw"],
                          label=label, zorder=4)
            line_objects[label] = ln

    active = set(res.keys())
    for lbl, ln in line_objects.items():
        ln.set_visible(lbl in active)

    stall_line.set_xdata([WS_stall, WS_stall])
    if stall_fill[0] is not None:
        stall_fill[0].remove()
    stall_fill[0] = ax.axvspan(20, WS_stall, alpha=0.08, color="#e24b4a", zorder=1)
    TW_stack = np.full((len(res), len(WS)), np.nan)
    for i, TW_arr in enumerate(res.values()):
        TW_stack[i] = TW_arr

    TW_env = np.nanmax(TW_stack, axis=0)
    TW_env = np.where(WS <= WS_stall, TW_env, np.nan)

    envelope_line.set_xdata(WS)
    envelope_line.set_ydata(TW_env)

    if feasible_fill[0] is not None:
        feasible_fill[0].remove()
        feasible_fill[0] = None

    valid = (~np.isnan(TW_env)) & (WS <= WS_stall) & (TW_env < TW_MAX)
    if np.any(valid):
        feasible_fill[0] = ax.fill_between(
            WS, TW_env, TW_MAX,
            where=valid,
            color="#1db954", alpha=0.18, zorder=2,
            label="_nolegend_"
        )

    valid_idx = np.where(valid)[0]
    if len(valid_idx) > 0:
        idx_dp  = valid_idx[np.nanargmin(TW_env[valid_idx])]
        WS_dp   = WS[idx_dp]
        TW_dp   = TW_env[idx_dp]
        design_pt.set_data([WS_dp], [TW_dp])
        dp_x = min(WS_dp + 25, 1100)
        dp_y = min(TW_dp + 0.02, TW_MAX - 0.08)
        dp_text.set_position((dp_x, dp_y))
        dp_text.set_text(f"W/S = {WS_dp:.0f} N/m²\nT/W = {TW_dp:.3f}")
        dp_text.set_visible(True)
    else:
        design_pt.set_data([], [])
        dp_text.set_visible(False)

    rho_cr, a_cr = isa(p["alt_cr"])
    V_cr  = p["mach"] * a_cr
    q_cr  = 0.5 * rho_cr * V_cr**2
    info_text.set_text(
        f"Cruise:  V = {V_cr:.0f} m/s  |  q = {q_cr:.0f} Pa  |  ρ = {rho_cr:.4f} kg/m³"
    )

    legend_handles = []
    for label, ln in line_objects.items():
        if ln.get_visible():
            legend_handles.append(
                Line2D([0],[0], color=ln.get_color(), ls=ln.get_linestyle(),
                       lw=ln.get_linewidth(), label=label)
            )
    legend_handles += [
        Line2D([0],[0], color="#e24b4a", lw=2.2, label="Stall speed limit"),
        Line2D([0],[0], color="#1db954", lw=2.5, label="Constraint envelope"),
        mpatches.Patch(facecolor="#1db954", alpha=0.35, label="Feasible design space"),
        Line2D([0],[0], marker="*", color="#111111", ms=10,
               ls="none", label="Min-T/W design point"),
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              fontsize=8.5, framealpha=0.92, edgecolor="#cccccc",
              ncol=1, handlelength=2.2)

    fig.canvas.draw_idle()

def reset(_):
    for key, sl in slider_objs.items():
        sl.set_val(DEFAULTS[key])

for sl in slider_objs.values():
    sl.on_changed(update)
btn_reset.on_clicked(reset)

update()
plt.show()