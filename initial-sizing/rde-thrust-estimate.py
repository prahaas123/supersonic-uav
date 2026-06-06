import warnings
warnings.filterwarnings('ignore')

import numpy as np
import cantera as ct
from scipy.optimize import brentq

MECH  = 'gri30.yaml'
ETA   = 0.85
G0    = 9.80665

# fixed inputs
FUEL      = 'C2H4'
OXIDIZER  = 'air'        # 'air' or 'O2'
PHI       = 1.0
P1        = 1.5 * 101325  # Pa  (reactant pressure upstream of detonation)
T1        = 255.0          # K
PA        = 101325.0       # Pa  (ambient)
UC        = 300.0          # m/s (injection velocity, held fixed)

# inputs
CASES = [
    {'label': 'reference', 'R_i': 0.02450, 'R_o': 0.03375, 'L': 0.10, 'mdot': 1.12},
    {'label': 'wider_ann', 'R_i': 0.02000, 'R_o': 0.04000, 'L': 0.10, 'mdot': 1.50},
    {'label': 'larger_R',  'R_i': 0.04000, 'R_o': 0.05500, 'L': 0.15, 'mdot': 2.00},
]

def _ox_str(oxidizer):
    return 'O2:1, N2:3.76' if oxidizer == 'air' else 'O2:1'

def get_cj_state(fuel, oxidizer, phi, P1, T1):
    ox = _ox_str(oxidizer)

    gas1 = ct.Solution(MECH)
    gas1.set_equivalence_ratio(phi, fuel, ox)
    gas1.TP = T1, P1
    rho1 = gas1.density
    h1   = gas1.enthalpy_mass

    def post_det(U):
        gas = ct.Solution(MECH)
        gas.set_equivalence_ratio(phi, fuel, ox)
        T2, P2 = 2500.0, 20 * P1
        for _ in range(500):
            gas.TP = T2, P2
            gas.equilibrate('TP')
            rho2 = gas.density
            u2   = rho1 * U / rho2
            P2n  = P1 + rho1 * U**2 - rho2 * u2**2
            h2n  = h1 + 0.5 * (U**2 - u2**2)
            gas.HP = h2n, P2n
            gas.equilibrate('HP')
            err = abs(gas.T - T2) / T2 + abs(gas.P - P2) / P2
            T2, P2 = gas.T, gas.P
            if err < 1e-7:
                break
        return gas, rho1 * U / gas.density, gas.sound_speed

    def cj_res(U):
        try:
            g, u2, a2 = post_det(U)
            return u2 - a2
        except Exception:
            return 1e9

    U_v = np.linspace(1400, 4000, 60)
    r_v = np.array([cj_res(U) for U in U_v])
    idx = np.where(np.diff(np.sign(r_v)) < 0)[0]
    if len(idx) == 0:
        raise RuntimeError("CJ bracket not found - check fuel/phi/P1/T1")
    U_CJ = brentq(cj_res, U_v[idx[0]], U_v[idx[0] + 1], xtol=0.5)
    gas_cj, _, _ = post_det(U_CJ)
    return U_CJ, gas_cj, rho1, h1

def _isentrope_at_P(gas_cj, P_target):
    s2 = gas_cj.entropy_mass
    def s_res(T):
        g = ct.Solution(MECH)
        g.TPX = T, P_target, gas_cj.X
        g.equilibrate('TP')
        return g.entropy_mass - s2
    try:
        T_s = brentq(s_res, 300.0, gas_cj.T, xtol=0.5)
    except Exception:
        return None
    g = ct.Solution(MECH)
    g.TPX = T_s, P_target, gas_cj.X
    g.equilibrate('TP')
    return g

def axial_sonic_sp_thrust(gas_cj, h1, Pa):
    P2 = gas_cj.P

    def sonic_res(P):
        g = _isentrope_at_P(gas_cj, P)
        if g is None:
            return -1e6
        dh = h1 - g.enthalpy_mass
        w  = np.sqrt(2 * dh) if dh > 0 else 0.0
        return w - g.sound_speed

    P_sonic = brentq(sonic_res, P2 * 0.05, P2 * 0.95, xtol=100)
    g_s     = _isentrope_at_P(gas_cj, P_sonic)
    w_star  = np.sqrt(2 * (h1 - g_s.enthalpy_mass))
    sp      = w_star + (P_sonic - Pa) / (g_s.density * w_star)
    return sp, w_star, P_sonic

def pressure_history_sp_thrust(U_CJ, gas_cj, rho_c, P1, Pa, uc, oxidizer):
    K     = 1.02 if oxidizer == 'air' else 1.54
    dPCJ  = gas_cj.P - P1
    FI_sp = (dPCJ / (rho_c * U_CJ)) * K
    FII_sp = (P1 - Pa) / (rho_c * uc) + uc
    return FI_sp + FII_sp

def fuel_mass_fraction(fuel, oxidizer, phi, P1, T1):
    ox  = _ox_str(oxidizer)
    gas = ct.Solution(MECH)
    gas.set_equivalence_ratio(phi, fuel, ox)
    gas.TP = T1, P1
    return gas[fuel].Y[0]

def run_case(case, U_CJ, gas_cj, rho_c, h1, Yf):
    R_i  = case['R_i']
    R_o  = case['R_o']
    mdot = case['mdot']
    W    = R_o - R_i
    Ac   = np.pi * (R_o**2 - R_i**2)
    H    = mdot / (rho_c * W * U_CJ)
    
    R_bar = 0.5 * (R_i + R_o)
    if 2 * np.pi * R_bar / H < 8:
        print(f"  [WARN] {case['label']}: 2πR̄/H = {2*np.pi*R_bar/H:.1f} < 8, K=1/α approximation unreliable")
    if case['L'] / H < 8:
        print(f"  [WARN] {case['label']}: L/H = {case['L']/H:.1f} < 8, axial flow assumption may not hold")

    sp_ph  = pressure_history_sp_thrust(U_CJ, gas_cj, rho_c, P1, PA, UC, OXIDIZER)
    sp_ax, w_star, P_star = axial_sonic_sp_thrust(gas_cj, h1, PA)

    sp_avg  = 0.5 * (sp_ph + sp_ax)
    sp_corr = ETA * sp_avg

    T_total = sp_corr * mdot
    Isp_f   = sp_corr / (Yf * G0)

    return {
        'label':   case['label'],
        'W_m':     W,
        'Ac_m2':   Ac,
        'H_mm':    H * 1e3,
        'sp_PH':   sp_ph,
        'sp_ax':   sp_ax,
        'sp_avg':  sp_avg,
        'sp_corr': sp_corr,
        'T_N':     T_total,
        'Isp_f_s': Isp_f,
        'w_star':  w_star,
        'P_star':  P_star / 1e5,
    }

def main():
    print(f"\nFuel={FUEL}  Oxidizer={OXIDIZER}  phi={PHI}  P1={P1/1e5:.2f} atm  T1={T1} K\n")
    print("Computing CJ state ... ", end='', flush=True)
    U_CJ, gas_cj, rho_c, h1 = get_cj_state(FUEL, OXIDIZER, PHI, P1, T1)
    Yf = fuel_mass_fraction(FUEL, OXIDIZER, PHI, P1, T1)
    print(f"done.\n  U_CJ={U_CJ:.1f} m/s  P_CJ={gas_cj.P/1e6:.3f} MPa  "
          f"T_CJ={gas_cj.T:.0f} K  rho_c={rho_c:.4f} kg/m3  Yf={Yf:.5f}\n")

    hdr = (f"{'Case':<14} {'W(mm)':>7} {'Ac(cm²)':>9} {'H(mm)':>7} "
           f"{'sp_PH':>8} {'sp_ax':>8} {'sp_avg':>8} {'sp_corr':>9} "
           f"{'T(N)':>9} {'Isp_f(s)':>10}")
    print(hdr)
    print('-' * len(hdr))

    for case in CASES:
        r = run_case(case, U_CJ, gas_cj, rho_c, h1, Yf)
        print(f"{r['label']:<14} "
              f"{r['W_m']*1e3:>7.1f} "
              f"{r['Ac_m2']*1e4:>9.2f} "
              f"{r['H_mm']:>7.2f} "
              f"{r['sp_PH']:>8.1f} "
              f"{r['sp_ax']:>8.1f} "
              f"{r['sp_avg']:>8.1f} "
              f"{r['sp_corr']:>9.1f} "
              f"{r['T_N']:>9.1f} "
              f"{r['Isp_f_s']:>10.0f}")

    print(f"\nAll specific thrusts in m/s.  ETA={ETA}  correction applied to average of PH+axial models.")

if __name__ == '__main__':
    main()