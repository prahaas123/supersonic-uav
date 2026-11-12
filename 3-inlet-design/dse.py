# S-duct inlet design space exploration, run from 3-inlet-design/
#
# python3 dse.py generate          write dse/designs.csv: Latin hypercube over build_inlet.BOUNDS, infeasible
#                                  designs (duct fold-over, choked capture area) replaced by feasible ones
# python3 dse.py run <design_id>   mesh, solve and post process one design (one SLURM array task, dse_run.sh)
# python3 dse.py pending           SLURM array spec of unfinished designs: sbatch --array=$(python3 dse.py pending)%10 dse_run.sh
# python3 dse.py collect           write dse_results/results.csv and copy the images of every design
#
# generate needs scipy (login node: module load python/3.11.5 scipy-stack), then
#   mkdir -p dse/logs && sbatch dse_run.sh

import csv
import json
import os
import shutil
import sys
import tempfile

import build_inlet as bi
import run_inlet as ri

N_SAMPLES   = 60
SEED        = 42
MAX_BATCHES = 50             # replacement Latin hypercube batches before generate gives up
DSE_DIR     = "dse"
DESIGNS_CSV = os.path.join(DSE_DIR, "designs.csv")
RESULTS_DIR = "dse_results"
RESULTS_CSV = os.path.join(RESULTS_DIR, "results.csv")

VARIABLES = list(bi.BOUNDS)  # LD, AR, WH, c_cl, c_area
KEEP = ("images", "metrics.json")
DONE_STATUSES = ("ok", "not_converged")

# Columns of results.csv after design_id and the variables (keys of metrics.json)
OUTPUTS = [
    # performance
    "pt_recovery_duct",      # mass-averaged p0 engine face / capture plane
    "reversed_flow_frac",    # duct volume fraction with U_x < 0 (separation)
    "duct_mach_max", "cap_mach_avg", "ef_mach_avg", "ef_p_avg",
    "mdot_outlet_kg_s", "mdot_ratio",
    # geometry
    "duct_length_m", "wetted_area_m2", "capture_area_m2",
    # run quality
    "status", "converged", "failed_checks", "iterations", "mass_imbalance_pct", "cont_err_avg",
    "yplus_min", "yplus_avg", "yplus_max",
    "duct_T_max", "duct_T_pctl", "duct_T_hot_frac",   # T_above_T0 check (duct_T_pctl at run_inlet.T_PERCENTILE)
    "solve_time_s",
]

def case_dir(design_id):
    return os.path.join(DSE_DIR, f"design_{int(design_id):03d}")

def read_designs():
    with open(DESIGNS_CSV, newline="") as fh:
        return list(csv.DictReader(fh))

def params_of(row):
    return {n: float(row[n]) for n in VARIABLES}

def read_metrics(design_id):
    try:
        with open(os.path.join(case_dir(design_id), "metrics.json")) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None

def is_done(design_id):
    m = read_metrics(design_id)
    return m is not None and m.get("status") in DONE_STATUSES

def infeasible(params, tmp):
    # Reason the design cannot be run, None if it can: the same checks prepare() makes before meshing
    try:
        ri.flow_conditions(params)
        bi.build_inlet(out_path=os.path.join(tmp, "check.stl"), **params)
    except ValueError as e:              # GeometryError is a ValueError
        return str(e)
    return None

def generate():
    from scipy.stats import qmc

    if os.path.exists(DESIGNS_CSV):
        print(f"{DESIGNS_CSV} already exists, delete it first to generate new designs")
        return 1

    lower = [bi.BOUNDS[n][0] for n in VARIABLES]
    upper = [bi.BOUNDS[n][1] for n in VARIABLES]
    sampler = qmc.LatinHypercube(d=len(VARIABLES), optimization="random-cd", seed=SEED)

    # The first batch is the design of experiments; infeasible points are replaced by feasible points of
    # further Latin hypercube batches, in order, until there are N_SAMPLES designs.
    designs, rejected = [], {}
    with tempfile.TemporaryDirectory() as tmp:
        for _ in range(MAX_BATCHES):
            for s in qmc.scale(sampler.random(N_SAMPLES), lower, upper):
                p = {n: round(float(v), 6) for n, v in zip(VARIABLES, s)}
                reason = infeasible(p, tmp)
                if reason:
                    rejected[reason] = rejected.get(reason, 0) + 1
                else:
                    designs.append(p)
                if len(designs) == N_SAMPLES:
                    break
            if len(designs) == N_SAMPLES:
                break
    if len(designs) < N_SAMPLES:
        print(f"Only {len(designs)} feasible designs after {MAX_BATCHES} batches, nothing written")
        return 1

    os.makedirs(DSE_DIR, exist_ok=True)
    with open(DESIGNS_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["design_id"] + VARIABLES)
        writer.writeheader()
        writer.writerows({"design_id": i, **p} for i, p in enumerate(designs))

    print(f"Wrote {N_SAMPLES} feasible designs ({len(VARIABLES)} variables) to {DESIGNS_CSV}")
    for n in VARIABLES:
        vals = [p[n] for p in designs]
        print(f"  {n:7s} {min(vals):.4f} - {max(vals):.4f}   (bounds {bi.BOUNDS[n][0]} - {bi.BOUNDS[n][1]})")
    for reason, count in rejected.items():
        print(f"  replaced {count} infeasible: {reason}")
    return 0

def strip_case(design_id):
    case = case_dir(design_id)
    if not os.path.isdir(case):
        return
    for name in os.listdir(case):
        if name in KEEP:
            continue
        path = os.path.join(case, name)
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path, ignore_errors=True)
        else:
            os.remove(path)

def run(design_id):
    if is_done(design_id):
        print(f"design_{design_id:03d} already completed, skipping")
        return 0
    row = next(r for r in read_designs() if int(r["design_id"]) == design_id)
    try:
        rc = ri.run_case(case_dir(design_id), params_of(row))
    finally:
        strip_case(design_id)
    if rc == 0 and os.environ.get("SLURM_ARRAY_TASK_ID") == str(design_id):
        try:                                    # keep the SLURM log only for designs that need a look
            os.remove(os.path.join(DSE_DIR, "logs", f"design_{design_id}.log"))
        except OSError:
            pass
    return rc

def array_spec(ids):
    parts, start = [], None
    for i, n in enumerate(ids):
        if start is None:
            start = n
        if i + 1 == len(ids) or ids[i + 1] != n + 1:
            parts.append(str(start) if start == n else f"{start}-{n}")
            start = None
    return ",".join(parts)

def pending():
    ids = [int(d["design_id"]) for d in read_designs() if not is_done(d["design_id"])]
    if not ids:
        print("All designs completed", file=sys.stderr)
        return
    print(f"{len(ids)} designs pending", file=sys.stderr)
    print(array_spec(ids))

def collect():
    designs = read_designs()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows, n_images = [], 0
    for d in designs:
        m = read_metrics(d["design_id"]) or {"status": "not_run", "converged": False}
        out = {k: m.get(k) for k in OUTPUTS}
        out["failed_checks"] = ";".join(m.get("failed_checks", []))
        rows.append({**d, **out})

        images = os.path.join(case_dir(d["design_id"]), "images")
        if os.path.isdir(images):
            dest = os.path.join(RESULTS_DIR, "images", os.path.basename(case_dir(d["design_id"])))
            shutil.rmtree(dest, ignore_errors=True)
            shutil.copytree(images, dest)
            n_images += 1

    with open(RESULTS_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["design_id"] + VARIABLES + OUTPUTS)
        writer.writeheader()
        writer.writerows(rows)

    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"Wrote {len(rows)} rows to {RESULTS_CSV}  " +
          "  ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    print(f"Copied images for {n_images} designs to {os.path.join(RESULTS_DIR, 'images')}")

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else None
    if cmd == "generate":
        sys.exit(generate())
    elif cmd == "run" and len(sys.argv) == 3:
        sys.exit(run(int(sys.argv[2])))
    elif cmd == "pending":
        pending()
    elif cmd == "collect":
        collect()
    else:
        print("usage: python3 dse.py generate | run <design_id> | pending | collect")
        sys.exit(1)
