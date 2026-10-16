# python3 dse.py generate          write dse/designs.csv (Latin hypercube samples)
# python3 dse.py run <design_id>   mesh, solve and extract metrics for one design (one SLURM array task, see dse_run.sh)
# python3 dse.py collect           collects all images and writes one results csv file
# python3 dse.py pending           print the SLURM array spec of pending designs: sbatch --array=$(python3 dse.py pending)%10 dse_run.sh

import csv
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "0-tools"))
from create_uav import reference_values

N_SAMPLES   = 512
SEED        = 42
DSE_DIR     = "dse"
DESIGNS_CSV = os.path.join(DSE_DIR, "designs.csv")
RESULTS_DIR = "dse_results"
RESULTS_CSV = os.path.join(RESULTS_DIR, "results.csv")

# shockAndWakeBox in snappyHexMeshDict, designs reaching outside it are flagged at generation
BOX_X_MAX = 4.2
BOX_Y_MAX = 1.6

KEEP = ("images", "metrics.json")
DONE_STATUSES = ("ok", "not_converged")

# Sampled variables: name -> (lower, upper)
VARIABLES = {
    "y_rotation":     (0.0, 3.0),
    "seg1root_chord": (1.2, 2.0),
    "seg1_taper":     (0.30, 0.70),   # break chord / root chord
    "seg2_taper":     (0.15, 0.70),   # tip chord / break chord
    "seg1_span":      (0.20, 0.40),
    "seg1_sweep":     (50.0, 75.0),
    "seg1_twist":     (-2.0, 0.0),
    "seg2_span":      (0.35, 0.65),
    "seg2_sweep":     (50.0, 70.0),
    "seg2_twist":     (-4.0, 0.0),
    "tc_root":        (0.020, 0.040),
    "tc_break":       (0.025, 0.045),
    "tc_tip":         (0.030, 0.060),
    "thick_loc":      (0.30, 0.60),
}

# constant UAV parameters
FIXED = dict(
    x_location=0.0,
    seg1_dihedral=0.0,
    seg2_dihedral=0.0,
    wing_tess_w=101,
    seg1_tess_u=35,
    seg2_tess_u=35,
)

def resolve(row):
    p = dict(FIXED)
    for name in VARIABLES:
        p[name] = float(row[name])
    return p

def case_dir(design_id):
    return os.path.join(DSE_DIR, f"design_{int(design_id):03d}")

def read_designs():
    with open(DESIGNS_CSV, newline="") as fh:
        return list(csv.DictReader(fh))

def is_done(design_id):
    try:
        with open(os.path.join(case_dir(design_id), "metrics.json")) as fh:
            return json.load(fh).get("status") in DONE_STATUSES
    except (OSError, ValueError):
        return False

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

def array_spec(ids):
    parts, start = [], None
    for i, n in enumerate(ids):
        if start is None:
            start = n
        if i + 1 == len(ids) or ids[i + 1] != n + 1:
            parts.append(str(start) if start == n else f"{start}-{n}")
            start = None
    return ",".join(parts)

def generate():
    from scipy.stats import qmc

    names = list(VARIABLES)
    lower = [VARIABLES[n][0] for n in names]
    upper = [VARIABLES[n][1] for n in names]
    unit = qmc.LatinHypercube(d=len(names), optimization="random-cd", seed=SEED).random(N_SAMPLES)
    samples = qmc.scale(unit, lower, upper)

    os.makedirs(DSE_DIR, exist_ok=True)
    rows = [{"design_id": i, **{n: round(float(v), 6) for n, v in zip(names, s)}}
            for i, s in enumerate(samples)]
    with open(DESIGNS_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["design_id"] + names)
        writer.writeheader()
        writer.writerows(rows)

    # Mesh envelope check
    refs = [reference_values(resolve(r)) for r in rows]
    outside = [r["design_id"] for r, ref in zip(rows, refs)
               if ref["tip_te_x"] > BOX_X_MAX or ref["semi_span"] > BOX_Y_MAX]
    print(f"Wrote {len(rows)} designs ({len(names)} variables) to {DESIGNS_CSV}")
    print(f"  max tip TE x   : {max(r['tip_te_x'] for r in refs):.3f} m (box {BOX_X_MAX})")
    print(f"  max semi-span  : {max(r['semi_span'] for r in refs):.3f} m (box {BOX_Y_MAX})")
    print(f"  ref area range : {min(r['ref_area'] for r in refs):.3f} - "
          f"{max(r['ref_area'] for r in refs):.3f} m2")
    if outside:
        print(f"  WARNING: {len(outside)} designs extend past shockAndWakeBox: {outside}")

def run(design_id):
    if is_done(design_id):
        print(f"design_{design_id:03d} already completed, skipping")
        return 0

    import supersonic_run as sr

    row = next(r for r in read_designs() if int(r["design_id"]) == design_id)
    atm = sr.isa_atmosphere(sr.ALTITUDE_M)
    u_inf = sr.MACH * atm["a"]
    try:
        results = sr.design_point(atm, u_inf, resolve(row), case_dir(design_id))
    finally:
        strip_case(design_id)
    if results is None:
        return 1
    if not results.get("converged"):
        return 2
    remove_log(design_id)
    return 0

def remove_log(design_id):
    if os.environ.get("SLURM_ARRAY_TASK_ID") != str(design_id):
        return
    try:
        os.remove(os.path.join(DSE_DIR, "logs", f"design_{design_id}.log"))
    except OSError:
        pass

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
    rows, metric_keys = [], []
    n_images = 0
    for d in designs:
        case = case_dir(d["design_id"])
        path = os.path.join(case, "metrics.json")
        if os.path.isfile(path):
            with open(path) as fh:
                m = json.load(fh)
        else:
            m = {"status": "not_run", "converged": False}
        m.pop("case", None)
        for k in m:
            if k not in metric_keys:
                metric_keys.append(k)
        rows.append({**d, **m})

        # Copy images to results/images/design_###
        images = os.path.join(case, "images")
        if os.path.isdir(images):
            dest = os.path.join(RESULTS_DIR, "images", os.path.basename(case))
            shutil.rmtree(dest, ignore_errors=True)
            shutil.copytree(images, dest)
            n_images += 1

    with open(RESULTS_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(designs[0]) + metric_keys)
        writer.writeheader()
        writer.writerows(rows)

    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"Wrote {len(rows)} rows to {RESULTS_CSV}  " +
          "  ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    print(f"Copied images for {n_images} designs to {os.path.join(RESULTS_DIR, 'images')}")

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("generate", "run", "collect", "pending"):
        print("usage: python3 dse.py generate | run <design_id> | collect | pending")
        sys.exit(1)
    if sys.argv[1] == "generate":
        generate()
    elif sys.argv[1] == "run":
        sys.exit(run(int(sys.argv[2])))
    elif sys.argv[1] == "pending":
        pending()
    else:
        collect()
