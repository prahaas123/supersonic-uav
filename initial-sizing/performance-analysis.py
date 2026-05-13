import numpy as np
import ussa1976
import matplotlib.pyplot as plt

# Aircraft parameters
auw_weight = 20  # N
fuel_weight = 20 # N
weight_minus_fuel = auw_weight - fuel_weight
SREF = 3  # m^2
BREF = 1  # m
AR = 8
SWEEP = 30 # degrees
stall_speed = 30 # m/s

# Aerodynamic estimations
cl_0 = 0.2
cl_alpha = 5 # per rad
cl_dmin = 0.15
cd_min = 0.032
e = (4.61 * (1 - 0.045 * AR ** 0.68) * (np.cos(SWEEP) ** 0.15)) - 3.1
k = 1 / (np.pi * AR * e)

# Atmosphere
altitude       = 2000 # m
atm            = ussa1976.compute(np.array([altitude]))
atm_sl         = ussa1976.compute(np.array([0.0]))        # sea-level reference
density        = float(atm["rho"].values[0])              # kg/m^3
density_sl     = float(atm_sl["rho"].values[0])           # kg/m^3
density_ratio  = density / density_sl
ambient_temp   = float(atm["t"].values[0])                # K
sound_speed    = np.sqrt(1.4 * 287.05 * ambient_temp)     # m/s
pressure       = float(atm["p"].values[0])                # Pa
pressure_sl    = float(atm_sl["p"].values[0])             # Pa
pressure_ratio = pressure / pressure_sl

# Propulsion
specific_fuel_flow = 0.5 # kg/N
static_thrust = 100 # N
thrust = static_thrust

# Airspeed conversions
v_cas = np.linspace(20, 900, 100)  # Calibrated airspeed, m/s
v_eas = v_cas  # Equivalent airspeed, m/s
v_tas = v_eas / np.sqrt(density_ratio)  # True airspeed, m/s
mach = v_tas / sound_speed

# Aerodynamics
q = 0.5 * density * v_tas**2 # Pa
cl = auw_weight / (q * SREF)
alpha_trim = (cl - cl_0) / cl_alpha * 180 / np.pi # degrees
cd_i = k * (cl - cl_dmin) ** 2
cd = cd_min + cd_i
ld = cl/cd
drag = cd * q * SREF
max_cl = 2 * auw_weight / (SREF * density * stall_speed ** 2)

# Propulsion
req_power = drag * v_tas # W
avail_power = thrust * v_tas # W
excess_power = avail_power - req_power
max_speed = v_tas[excess_power >= 0][-1]

# PLOTS
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle(f"Aerodynamic Performance  —  Altitude: {altitude} m", fontsize=14)

# 1. Drag polar: CL vs CAS
axes[0, 0].plot(v_cas, cl, color="steelblue")
axes[0, 0].set_xlabel("CAS (m/s)")
axes[0, 0].set_ylabel("CL")
axes[0, 0].set_title("Required CL vs CAS")
axes[0, 0].grid(True)

# 2. L/D vs CAS
axes[0, 1].plot(v_cas, ld, color="darkorange")
axes[0, 1].set_xlabel("CAS (m/s)")
axes[0, 1].set_ylabel("L/D")
axes[0, 1].set_title("Lift-to-Drag Ratio vs CAS")
axes[0, 1].grid(True)

# 4. Alpha trim vs CAS
axes[1, 0].plot(v_cas, alpha_trim, color="crimson")
axes[1, 0].set_xlabel("CAS (m/s)")
axes[1, 0].set_ylabel("α trim (deg)")
axes[1, 0].set_title("Trim Angle of Attack vs CAS")
axes[1, 0].grid(True)

# 5. Dynamic pressure vs CAD
axes[1, 1].plot(v_cas, q, color="mediumpurple")
axes[1, 1].set_xlabel("CAS (m/s)")
axes[1, 1].set_ylabel("q (Pa)")
axes[1, 1].set_title("Dynamic Pressure vs CAS")
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()