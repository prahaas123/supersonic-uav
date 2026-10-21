#   local:  python surrogate.py, python optimize.py, push 2-optimization/candidates.csv
#   nibi:   python3 dse.py add candidates.csv
#           sbatch --array=$(python3 dse.py pending)%10 dse_run.sh
#           python3 dse.py collect, copy dse_results/results.csv to 0-results/dse_results.csv

import csv
import os

import numpy as np
from scipy.optimize import differential_evolution

import dse
from surrogate import HERE, LOWER, NAMES, RESULTS_CSV, TARGETS, UPPER, load_models

# Design targets (estimates, update after the sizing redo)
MASS_KG       = 20.0
LIFT_MIN_N    = MASS_KG * 9.81         # lift >= weight
CG_MIN_MAC    = 0.25                   # trim CG must lie in this band (fraction of MAC from its LE),
CG_MAX_MAC    = 0.45                   # i.e. where the CG can be placed, update after component placement
THRUST_N      = 343.75
THRUST_MARGIN = 0.8
DRAG_MAX_N    = THRUST_MARGIN * THRUST_N   # drag <= available thrust with margin

N_SIGMA   = 2.0      # constraints must hold at mean -/+ N_SIGMA std, guards against surrogate error
N_DESIGNS = 5        # designs proposed per round
MIN_DIST  = 0.15     # minimum spacing between proposed designs, in variables scaled to 0-1
PENALTY   = 1e4      # [N] added drag per unit of (normalised) constraint violation

MODELS_PATH    = os.path.join(HERE, "..", "0-results", "surrogates.joblib")
CANDIDATES_CSV = os.path.join(HERE, "candidates.csv")

def read_csv(path):
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))

def trim_cg(lift, moment, mac):
    return dse.CG_MAC_FRACTION - moment / (np.maximum(lift, 1.0) * mac)

def is_feasible(lift, drag, moment, mac):
    return (lift >= LIFT_MIN_N and CG_MIN_MAC <= trim_cg(lift, moment, mac) <= CG_MAX_MAC
            and drag <= DRAG_MAX_N)

def geometry(u):
    p = dict(dse.FIXED, **dict(zip(NAMES, LOWER + u * (UPPER - LOWER))))
    ref = dse.reference_values(p)
    box = max(ref["tip_te_x"] - dse.BOX_X_MAX, 0.0) + max(ref["semi_span"] - dse.BOX_Y_MAX, 0.0)
    return ref["ref_chord"], box

def predict(models, U):
    return {t: models[t].predict(U, return_std=True) for t in TARGETS}

def trim_cg_std(pred, mac):
    (ml, sl), (mm, sm) = pred["lift_N"], pred["cm_pitch_Nm"]
    ml = np.maximum(ml, 1.0)
    return np.sqrt((sm / (ml * mac)) ** 2 + (mm * sl / (ml ** 2 * mac)) ** 2)

def violations(pred, mac):
    (ml, sl), (md, sd), (mm, _) = pred["lift_N"], pred["drag_N"], pred["cm_pitch_Nm"]
    x, sx = trim_cg(ml, mm, mac), N_SIGMA * trim_cg_std(pred, mac)
    lift = np.maximum(LIFT_MIN_N - (ml - N_SIGMA * sl), 0.0) / LIFT_MIN_N
    drag = np.maximum(md + N_SIGMA * sd - DRAG_MAX_N, 0.0) / DRAG_MAX_N
    band = CG_MAX_MAC - CG_MIN_MAC
    cg = (np.maximum(CG_MIN_MAC - (x - sx), 0.0) + np.maximum(x + sx - CG_MAX_MAC, 0.0)) / band
    return lift + drag + cg

def objective(models, picks):
    def f(x):
        U = np.atleast_2d(x.T)
        pred = predict(models, U)
        mac, box = np.array([geometry(u) for u in U]).T
        v = violations(pred, mac) + box
        for p in picks:
            v += np.maximum(MIN_DIST - np.linalg.norm(U - p, axis=1), 0.0) / MIN_DIST
        return pred["drag_N"][0] + PENALTY * v
    return f

def propose(models):
    picks = []
    for k in range(N_DESIGNS):
        res = differential_evolution(objective(models, picks), [(0.0, 1.0)] * len(NAMES),
                                     seed=dse.SEED + k, maxiter=300, popsize=15, tol=1e-8,
                                     init="latinhypercube", polish=False,
                                     vectorized=True, updating="deferred")
        picks.append(res.x)
    return np.array(picks)

def check_previous(results):
    previous = read_csv(CANDIDATES_CSV)
    if not previous:
        return
    by_id = {int(r["design_id"]): r for r in results}
    print("Previous candidates, CFD vs predicted (z = (CFD - predicted) / std):")
    for c in previous:
        r = by_id.get(int(c["design_id"]), {})
        if r.get("status") != "ok":
            print(f"  design_{int(c['design_id']):03d}  {r.get('status', 'not in results yet')}")
            continue
        parts = [f"{t} {float(r[t]):7.1f} (pred {float(c[f'pred_{t}']):7.1f}, "
                 f"z {(float(r[t]) - float(c[f'pred_{t}'])) / float(c[f'std_{t}']):+5.2f})" for t in TARGETS]
        cg = trim_cg(float(r["lift_N"]), float(r["cm_pitch_Nm"]), float(r["ref_chord"]))
        print(f"  design_{int(c['design_id']):03d}  " + "  ".join(parts)
              + f"  trim CG {cg:.3f} MAC (pred {float(c['trim_cg_mac']):.3f})"
              + ("  FEASIBLE" if row_feasible(r) else ""))

def row_feasible(r):
    return is_feasible(float(r["lift_N"]), float(r["drag_N"]), float(r["cm_pitch_Nm"]), float(r["ref_chord"]))

def best_feasible(results):
    feas = [r for r in results if r.get("status") == "ok" and row_feasible(r)]
    return min(feas, key=lambda r: float(r["drag_N"])) if feas else None, len(feas)

def main():
    results = read_csv(RESULTS_CSV)
    check_previous(results)

    best, n_feas = best_feasible(results)
    print(f"\n{n_feas} feasible designs in CFD" + (
        f", best design_{int(best['design_id']):03d}: drag {float(best['drag_N']):.2f} N, "
        f"lift {float(best['lift_N']):.1f} N, trim CG "
        f"{trim_cg(float(best['lift_N']), float(best['cm_pitch_Nm']), float(best['ref_chord'])):.3f} MAC"
        if best else ""))

    print(f"\nOptimising on the surrogates ({N_DESIGNS} designs, constraints at {N_SIGMA:g} sigma)...")
    models = load_models(MODELS_PATH)
    picks = propose(models)
    pred = predict(models, picks)
    mac, box = np.array([geometry(u) for u in picks]).T
    cg, cg_std = trim_cg(pred["lift_N"][0], pred["cm_pitch_Nm"][0], mac), trim_cg_std(pred, mac)
    margins_ok = (violations(pred, mac) == 0.0) & (box == 0.0)

    start = max(int(r["design_id"]) for r in results) + 1
    rows = []
    for k, u in enumerate(picks):
        row = {"design_id": start + k}
        row.update({n: round(float(v), 6) for n, v in zip(NAMES, LOWER + u * (UPPER - LOWER))})
        for t in TARGETS:
            row[f"pred_{t}"], row[f"std_{t}"] = round(float(pred[t][0][k]), 4), round(float(pred[t][1][k]), 4)
        row["trim_cg_mac"], row["trim_cg_std"] = round(float(cg[k]), 4), round(float(cg_std[k]), 4)
        row["margins_ok"] = bool(margins_ok[k])
        rows.append(row)
        print(f"  design_{row['design_id']:03d}  " + "  ".join(
            f"{t} {row[f'pred_{t}']:7.1f} +/- {row[f'std_{t}']:5.1f}" for t in TARGETS)
            + f"  trim CG {row['trim_cg_mac']:.3f} +/- {row['trim_cg_std']:.3f} MAC"
            + ("" if row["margins_ok"] else "  CONSTRAINTS NOT MET"))

    with open(CANDIDATES_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {CANDIDATES_CSV}")

if __name__ == "__main__":
    main()