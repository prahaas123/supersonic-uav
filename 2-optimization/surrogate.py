import csv
import os
import sys
import warnings

import numpy as np
import joblib
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from dse import SEED, VARIABLES

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(HERE, "..", "0-results", "dse_results.csv")

TARGETS = {
    "lift_N":      "Lift [N]",
    "drag_N":      "Drag [N]",
    "cm_pitch_Nm": "Pitching moment about CG [N.m]",
}
N_FOLDS     = 5
N_RESTARTS  = 10
LENGTH_MAX  = 1e2    # length scales at this bound mean the variable has no effect

NAMES = list(VARIABLES)
LOWER = np.array([VARIABLES[n][0] for n in NAMES])
UPPER = np.array([VARIABLES[n][1] for n in NAMES])

def to_unit(x):
    return (np.asarray(x, dtype=float) - LOWER) / (UPPER - LOWER)

def make_gp():
    kernel = (ConstantKernel(1.0, (1e-3, 1e3))
              * Matern(length_scale=np.ones(len(NAMES)), length_scale_bounds=(1e-2, LENGTH_MAX), nu=2.5)
              + WhiteKernel(1e-3, (1e-8, 1e0)))
    return GaussianProcessRegressor(kernel, normalize_y=True,
                                    n_restarts_optimizer=N_RESTARTS, random_state=SEED)

def make_quadratic():
    return make_pipeline(PolynomialFeatures(2), LinearRegression())

def load_data(path, target):
    X, y = [], []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("status") != "ok" or row.get(target) in (None, ""):
                continue
            X.append([float(row[n]) for n in NAMES])
            y.append(float(row[target]))
    return to_unit(X), np.array(y)

def scores(y, mean, std=None):
    rmse = float(np.sqrt(np.mean((y - mean) ** 2)))
    r2 = 1.0 - np.sum((y - mean) ** 2) / np.sum((y - y.mean()) ** 2)
    cover = None if std is None else float(np.mean(np.abs(y - mean) <= 2.0 * std))
    return r2, rmse, cover

def cross_validate(model, X, y, with_std):
    mean, std = np.zeros_like(y), np.zeros_like(y)
    for train, test in KFold(N_FOLDS, shuffle=True, random_state=SEED).split(X):
        m = clone(model).fit(X[train], y[train])
        if with_std:
            mean[test], std[test] = m.predict(X[test], return_std=True)
        else:
            mean[test] = m.predict(X[test])
    return mean, (std if with_std else None)

def length_scales(gp):
    return gp.kernel_.k1.k2.length_scale

def fit(path):
    out_dir = os.path.dirname(os.path.abspath(path))
    models, cv = {}, {}

    for target, label in TARGETS.items():
        X, y = load_data(path, target)
        print(f"\n=== {label}   ({len(y)} ok designs)")

        gp_mean, gp_std = cross_validate(make_gp(), X, y, with_std=True)
        quad_mean, _ = cross_validate(make_quadratic(), X, y, with_std=False)
        r2, rmse, cover = scores(y, gp_mean, gp_std)
        q_r2, q_rmse, _ = scores(y, quad_mean)
        print(f"  {N_FOLDS}-fold CV   GP        R2 {r2:.4f}   RMSE {rmse:.4g}   within 2 sigma {100 * cover:.1f}%")
        print(f"               quadratic R2 {q_r2:.4f}   RMSE {q_rmse:.4g}")
        cv[target] = (y, gp_mean, gp_std)

        gp = make_gp().fit(X, y)
        models[target] = gp
        print(f"  final kernel: {gp.kernel_}")
        print("  length scales (shorter = more influential):")
        for i in np.argsort(length_scales(gp)):
            ls = length_scales(gp)[i]
            flag = "   <- no effect" if ls >= 0.99 * LENGTH_MAX else ""
            print(f"    {NAMES[i]:<15} {ls:8.3f}{flag}")

    model_path = os.path.join(out_dir, "surrogates.joblib")
    joblib.dump({"variables": VARIABLES, "models": models}, model_path)
    print(f"\nSaved models to {model_path}")
    plot_cv(cv, os.path.join(out_dir, "surrogate_cv.png"))

def plot_cv(cv, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(cv), figsize=(5 * len(cv), 4.5))
    for ax, (target, (y, mean, std)) in zip(axes, cv.items()):
        ax.errorbar(y, mean, yerr=2.0 * std, fmt="o", ms=3, alpha=0.5, elinewidth=0.5)
        lo, hi = min(y.min(), mean.min()), max(y.max(), mean.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel(f"CFD {TARGETS[target]}")
        ax.set_ylabel(f"GP {TARGETS[target]}")
        ax.set_title(f"R2 = {scores(y, mean)[0]:.4f}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved CV plot to {path}")

def load_models(path):
    return joblib.load(path)["models"]

def predict(models, x):
    # x: (n, 14) array in physical units, columns in VARIABLES order
    X = to_unit(np.atleast_2d(x))
    return {t: m.predict(X, return_std=True) for t, m in models.items()}

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    fit(sys.argv[1] if len(sys.argv) > 1 else RESULTS_CSV)
