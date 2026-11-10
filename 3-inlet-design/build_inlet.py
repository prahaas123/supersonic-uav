import math
from pathlib import Path
import numpy as np

BOUNDS = {
    "LD": (4.0, 7.0),
    "AR": (1.3, 2.0),
    "WH": (1.5, 4.0),
    "c_cl": (0.35, 0.65),
    "c_area": (0.30, 0.70),
}

D_EF = 0.080      # engine-face diameter (RDE)
OFFSET = 0.1034   # capture centre height above engine axis
X_LIP = 0.620     # capture-plane station
N_AXIAL = 121     # stations along the duct
N_CIRC = 96       # points around each section


class GeometryError(ValueError):
    """Duct wall folds over (bend radius < local half-height)."""

def _s_curve(u, c):
    # 0->1, zero slope at both ends, inflection at u=c. C2, finite curvature.
    u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
    left = c * (1 - np.cos(np.pi * u / (2 * c)))
    right = c + (1 - c) * np.sin(np.pi * (u - c) / (2 * (1 - c)))
    return np.where(u <= c, left, right)

def _section(x, L, H0, W0, c_cl, c_area):
    # Centreline z, width, height at station(s) x.
    u = (np.asarray(x, dtype=float) - X_LIP) / L
    zc = OFFSET * (1 - _s_curve(u, c_cl))
    s = _s_curve(u, c_area)
    return zc, W0 + (D_EF - W0) * s, H0 + (D_EF - H0) * s

def _normals(t):
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    return n / np.linalg.norm(n, axis=1, keepdims=True)

def build_inlet(LD, AR, WH, c_cl, c_area, out_path):
    params = dict(LD=LD, AR=AR, WH=WH, c_cl=c_cl, c_area=c_area)
    for k, v in params.items():
        lo, hi = BOUNDS[k]
        if not lo <= v <= hi:
            raise ValueError(f"{k}={v} outside bounds [{lo}, {hi}]")

    L = LD * D_EF
    H0 = D_EF / math.sqrt(AR * WH)
    W0 = WH * H0

    # fold-over check: centreline radius of curvature vs local half-height
    x = X_LIP + L * np.linspace(0, 1, 4001)
    z, _, H = _section(x, L, H0, W0, c_cl, c_area)
    zp = np.gradient(z, x)
    kappa = np.abs(np.gradient(zp, x)) / (1 + zp ** 2) ** 1.5
    if np.min(1 / np.maximum(kappa, 1e-12) / (H / 2)) <= 1.0:
        raise GeometryError("duct wall folds over")

    # surface points: cosine clustering toward both ends
    xs = X_LIP + L * 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, N_AXIAL)))
    zc, w, h = _section(xs, L, H0, W0, c_cl, c_area)
    th = np.linspace(0, 2 * np.pi, N_CIRC, endpoint=False)
    P = np.empty((N_AXIAL, N_CIRC, 3))
    P[..., 0] = xs[:, None]
    P[..., 1] = (w / 2)[:, None] * np.cos(th)[None, :]
    P[..., 2] = zc[:, None] + (h / 2)[:, None] * np.sin(th)[None, :]

    # wall triangles, outward normals
    i = np.arange(N_AXIAL - 1)[:, None]
    j = np.arange(N_CIRC)[None, :]
    jn = (j + 1) % N_CIRC
    a, b, c, d = P[i, j], P[i + 1, j], P[i + 1, jn], P[i, jn]
    wall = np.concatenate([np.stack([a, d, c], -2).reshape(-1, 3, 3),
                           np.stack([a, c, b], -2).reshape(-1, 3, 3)])

    # end caps (fan from section centre)
    def cap(ring, centre, outward_x):
        C = np.broadcast_to(centre, ring.shape)
        tri = np.stack([C, ring, np.roll(ring, -1, axis=0)], 1)   # normal +x
        return tri if outward_x > 0 else tri[:, [0, 2, 1]]

    patches = {
        "inlet": cap(P[0], np.array([xs[0], 0.0, zc[0]]), -1),
        "wall": wall,
        "outlet": cap(P[-1], np.array([xs[-1], 0.0, zc[-1]]), +1),
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for name, t in patches.items():
            f.write(f"solid {name}\n")
            for nrm, tri in zip(_normals(t), t):
                f.write(f" facet normal {nrm[0]:.6e} {nrm[1]:.6e} {nrm[2]:.6e}\n  outer loop\n")
                for v in tri:
                    f.write(f"   vertex {v[0]:.9e} {v[1]:.9e} {v[2]:.9e}\n")
                f.write("  endloop\n endfacet\n")
            f.write(f"endsolid {name}\n")
    return out_path