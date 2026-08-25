import glob
import json
import os
import sys

import numpy as np

SYMMETRY_FACTOR = 2.0     # half model
TAIL_FRACTION = 0.10      # average over the final 10% of the time record
DRIFT_TOLERANCE_PCT = 0.20
MIN_TAIL_SAMPLES = 20
COEFF_COLS = {"Cd": 1, "Cl": 4, "CmPitch": 7}

def _start_time(path):
    try:
        return float(os.path.basename(os.path.dirname(path)))
    except ValueError:
        return 0.0

def _read_dat(case_dir, function_object, filename, min_cols):
    pattern = os.path.join(case_dir, "postProcessing", function_object,
                           "*", filename)
    files = sorted(glob.glob(pattern), key=_start_time)
    if not files:
        return None, None

    times, rows = [], []
    for path in files:
        with open(path, "r") as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                parts = line.replace("(", " ").replace(")", " ").split()
                if len(parts) < min_cols:
                    continue
                try:
                    t = float(parts[0])
                    vals = [float(v) for v in parts[1:]]
                except ValueError:
                    continue
                # Restarts re-emit the overlap, keep the first occurrence
                if times and t <= times[-1]:
                    continue
                times.append(t)
                rows.append(vals)

    if not times:
        return None, None

    width = min(len(r) for r in rows)
    return np.asarray(times), np.asarray([r[:width] for r in rows])

def read_yplus(case_dir, patch="uav"):
    pattern = os.path.join(case_dir, "postProcessing", "yPlus", "*", "yPlus.dat")
    files = sorted(glob.glob(pattern), key=_start_time)
    avg_yp = max_yp = min_yp = None
    for path in files:
        with open(path, "r") as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 5 and parts[1] == patch:
                    min_yp, max_yp, avg_yp = (float(parts[2]), float(parts[3]),
                                              float(parts[4]))
    return {"yplus_min": min_yp, "yplus_max": max_yp, "yplus_avg": avg_yp}

def window_stats(t, y, tail_fraction=TAIL_FRACTION, scale=1.0):
    if t is None or len(t) < 2:
        return None

    t_cut = t[0] + (1.0 - tail_fraction) * (t[-1] - t[0])
    mask = t >= t_cut
    tw, yw = t[mask], y[mask]

    if len(tw) < 2:
        return None

    mean_raw = float(np.mean(yw))
    slope = float(np.polyfit(tw, yw, 1)[0]) if len(tw) > 2 else 0.0
    span = float(tw[-1] - tw[0])
    drift_pct = (100.0 * slope * span / mean_raw) if mean_raw else None

    return {
        "value": mean_raw * scale,
        "std": float(np.std(yw, ddof=1)) * abs(scale) if len(yw) > 1 else 0.0,
        "min": float(np.min(yw)) * scale,
        "max": float(np.max(yw)) * scale,
        "drift_pct": drift_pct,
        "n_samples": int(len(tw)),
        "window_start": float(tw[0]),
        "window_end": float(tw[-1]),
    }

def extract_metrics(case_dir,
                    symmetry_factor=SYMMETRY_FACTOR,
                    tail_fraction=TAIL_FRACTION,
                    drift_tolerance_pct=DRIFT_TOLERANCE_PCT,
                    patch="uav"):
    S = float(symmetry_factor)
    out = {
        "case": os.path.abspath(case_dir),
        "symmetry_factor": S,
        "tail_fraction": tail_fraction,
        "drift_tolerance_pct": drift_tolerance_pct,
        "status": "ok",
        "converged": False,
    }
    
    t, F = _read_dat(case_dir, "forces", "force.dat", min_cols=4)
    if t is None:
        out["status"] = "no_data"
        return out

    out["t_end"] = float(t[-1])
    out["n_timesteps"] = int(len(t))

    drag = window_stats(t, F[:, 0], tail_fraction, S)
    side = window_stats(t, F[:, 1], tail_fraction, 1.0)
    lift = window_stats(t, F[:, 2], tail_fraction, S)

    if drag is None or drag["n_samples"] < MIN_TAIL_SAMPLES:
        out["status"] = "too_few_samples"
        return out

    out["drag_N"] = drag["value"]
    out["drag_std_N"] = drag["std"]
    out["drag_drift_pct"] = drag["drift_pct"]
    out["lift_N"] = lift["value"]
    out["lift_std_N"] = lift["std"]
    out["lift_drift_pct"] = lift["drift_pct"]
    out["window_start_s"] = drag["window_start"]
    out["n_window_samples"] = drag["n_samples"]
    out["side_force_raw_N"] = side["value"] if side else None

    if F.shape[1] >= 9:
        pres = window_stats(t, F[:, 3], tail_fraction, S)
        visc = window_stats(t, F[:, 6], tail_fraction, S)
        out["drag_pressure_N"] = pres["value"] if pres else None
        out["drag_viscous_N"] = visc["value"] if visc else None

    # moments
    tm, M = _read_dat(case_dir, "forces", "moment.dat", min_cols=4)
    if tm is not None:
        pitch = window_stats(tm, M[:, 1], tail_fraction, S)
        out["cm_pitch_Nm"] = pitch["value"] if pitch else None

    # coefficients
    tc, C = _read_dat(case_dir, "forceCoeffs", "coefficient.dat", min_cols=8)
    if tc is not None:
        for name, col in COEFF_COLS.items():
            if C.shape[1] > col - 1:
                st = window_stats(tc, C[:, col - 1], tail_fraction, S)
                if st:
                    out[name] = st["value"]
        if out.get("Cd"):
            out["LD_ratio"] = out.get("Cl", 0.0) / out["Cd"]

    # mesh quality
    out.update(read_yplus(case_dir, patch))

    # convergence
    drift = out.get("drag_drift_pct")
    out["converged"] = drift is not None and abs(drift) < drift_tolerance_pct
    if not out["converged"]:
        out["status"] = "not_converged"

    return out


def write_metrics(case_dir, path=None, **kwargs):
    m = extract_metrics(case_dir, **kwargs)
    path = path or os.path.join(case_dir, "metrics.json")
    with open(path, "w") as fh:
        json.dump(m, fh, indent=2)
    return m

CSV_FIELDS = [
    "case", "status", "converged",
    "drag_N", "lift_N", "drag_pressure_N", "drag_viscous_N", "cm_pitch_Nm",
    "Cd", "Cl", "CmPitch", "LD_ratio",
    "drag_drift_pct", "lift_drift_pct", "drag_std_N", "lift_std_N",
    "yplus_avg", "yplus_max", "side_force_raw_N",
    "t_end", "n_window_samples",
]


def append_results_csv(m, csv_path, case_name=None):
    import csv

    row = {k: m.get(k) for k in CSV_FIELDS}
    row["case"] = case_name or os.path.basename(os.path.normpath(m.get("case", "")))

    exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    return csv_path


def format_summary(m):
    def fmt(key, spec=".4f", default="n/a"):
        v = m.get(key)
        return format(v, spec) if isinstance(v, (int, float)) else default

    bar = "=" * 52
    lines = [
        "", bar,
        f" AERODYNAMIC METRICS   [{m.get('status', '?')}]",
        bar,
        f"  averaged over final {100 * m.get('tail_fraction', 0):.0f}% "
        f"({m.get('n_window_samples', 0)} samples, "
        f"t = {fmt('window_start_s', '.5g')} -> {fmt('t_end', '.5g')} s)",
        f"  symmetry factor        x{m.get('symmetry_factor', 1):.0f}",
        "-" * 52,
        f"  Drag force  (X)   :  {fmt('drag_N', '10.3f')} N   "
        f"+/- {fmt('drag_std_N', '.3f')}",
        f"  Lift force  (Z)   :  {fmt('lift_N', '10.3f')} N   "
        f"+/- {fmt('lift_std_N', '.3f')}",
        f"    pressure        :  {fmt('drag_pressure_N', '10.3f')} N",
        f"    viscous         :  {fmt('drag_viscous_N', '10.3f')} N",
        f"  Pitch moment      :  {fmt('cm_pitch_Nm', '10.3f')} N.m",
        "-" * 52,
        f"  Cd                :  {fmt('Cd', '10.5f')}",
        f"  Cl                :  {fmt('Cl', '10.5f')}",
        f"  CmPitch           :  {fmt('CmPitch', '10.5f')}",
        f"  L/D               :  {fmt('LD_ratio', '10.4f')}",
        "-" * 52,
        f"  drag drift        :  {fmt('drag_drift_pct', '+10.4f')} %  "
        f"(tolerance {m.get('drift_tolerance_pct', 0):.2f} %)",
        f"  y+  avg / max     :  {fmt('yplus_avg', '.2f')} / {fmt('yplus_max', '.2f')}",
        f"  side force (raw)  :  {fmt('side_force_raw_N', '10.3f')} N  "
        f"[half-model diagnostic, not an aircraft load]",
        bar,
        f"  CONVERGED: {m.get('converged')}",
        bar, "",
    ]
    return "\n".join(lines)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("usage: python3 metrics.py <case_dir> [--full-model]")
        sys.exit(1)

    case = sys.argv[1]
    sf = 1.0 if "--full-model" in sys.argv else SYMMETRY_FACTOR
    result = write_metrics(case, symmetry_factor=sf)
    print(format_summary(result))
    print(f"wrote {os.path.join(case, 'metrics.json')}")
    sys.exit(0 if result.get("converged") else 2)