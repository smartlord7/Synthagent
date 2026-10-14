"""Ablation and sensitivity analysis of GHB as released (real_v3.GHB_V5) on the 12 TCGA design networks, with the
folds and privacy noise of dev/v4_dev.py (every variant sees the same folds and the same privacy noise, so differences
are paired). The five held-out networks are not used.

Ablation: one component removed or replaced at a time. Sensitivity: one fixed setting varied at a time, the others at
their released values (half-Cauchy scale of tau 0.5, non-borrowing weight 1/3, budget share of the information release
1/2, Q = 12 quantiles, default prior scales {0.5, 1, 2.5}).
    python3 dev/v5_ablation.py [reps]        then   python3 dev/v5_ablation.py score
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import v4_dev as D  # noqa: E402
import ghb_logit as GL  # noqa: E402
import ghb_v3 as V3  # noqa: E402
import ghb_v4 as V4  # noqa: E402
import real_v3 as RV  # noqa: E402

V5 = dict(RV.GHB_V5)
ABLATION = {
    "GHB": V5,
    "without borrowing across networks": dict(V5, pan=False),
    "without the non-borrowing components": dict(V5, safety=False),
    "without propagating the privacy covariance": dict(V5, vdp=False),
    "without removing the hospital's own data": dict(V5, loo=False),
    "intercept borrowed like the slopes": dict(V5, int_sd=None, free_int=None),
    "loose sensitivity of the information release": dict(V5, tight=False),
    "full information matrix always": dict(V5, hdiag=False),
    "diagonal information always": dict(V5, hdiag=True),
}
SENSITIVITY = {
    **{f"scale of tau {s:g}": dict(V5, scale=s) for s in (0.1, 0.25, 1.0, 2.0)},
    **{f"non-borrowing weight {w:g}": dict(V5, safety_w=w) for w in (0.1, 1.0, 3.0)},
    **{f"information share {h:g}": dict(V5, hshare=h) for h in (0.25, 0.75)},
    **{f"Q = {q}": dict(V5, _Q=q) for q in (1, 3, 6, 24)},
    "default scales {2.5}": dict(V5, _SDS=(2.5,)),
    "default scales {1}": dict(V5, _SDS=(1.0,)),
    "default scales {0.5, 1, 2.5, 5}": dict(V5, _SDS=(0.5, 1.0, 2.5, 5.0)),
}
D.VARIANTS = {**ABLATION, **SENSITIVITY}
D.OUT = D.ghb.RESDIR / "v5_ablation_ckpt"
OUT_JSON = D.ghb.RESDIR / "v5_ablation.json"

_orig = V4.estimate


def _estimate(nets, rng, eps, delta, G, inst=None, cfg=None):
    """V4.estimate with Q and the default scales set per variant (module constants of ghb_v4 / ghb_v3)."""
    cfg = dict(cfg or {})
    q, sds = cfg.pop("_Q", V3.Q), cfg.pop("_SDS", V3.SDS)
    saved = V4.Q, V4.SDS, V3.quantiles.__defaults__
    V4.Q, V4.SDS = q, tuple(sds)
    V3.quantiles.__defaults__ = (GL.GRID, q)
    try:
        return _orig(nets, rng, eps, delta, G, inst=inst, cfg=cfg)
    finally:
        V4.Q, V4.SDS, V3.quantiles.__defaults__ = saved


V4.estimate = _estimate


def score():
    """Pooled held-out log-loss relative to local estimation per network, mean over the 12 networks, and in how many
    networks GHB as released is more accurate than the variant."""
    acc = {}
    files = sorted(D.OUT.glob("*.json"))
    for fn in files:
        for k, (ll, cnt) in json.load(open(fn)).items():
            a = acc.setdefault(k, [0.0, 0.0])
            a[0] += sum(ll); a[1] += sum(cnt)
    nets = sorted({k.split("||")[0] for k in acc})
    L = lambda p, m: acc[f"{p}||{m}"][0] / acc[f"{p}||{m}"][1]
    rel = lambda p, m: L(p, m) / L(p, "Local")
    reps = len(files) // 2
    out = {"reps": reps, "networks": nets, "ablation": {}, "sensitivity": {}}
    for group, names in (("ablation", ABLATION), ("sensitivity", SENSITIVITY)):
        for name in names:
            row = {}
            for e in ("1", "10"):
                v = np.array([rel(p, f"{name}, eps={e}") for p in nets])
                g = np.array([rel(p, f"GHB, eps={e}") for p in nets])
                row[e] = {"mean": float(v.mean()), "ghb_better": int((g < v - 1e-12).sum()), "per_net": v.tolist()}
            out[group][name] = row
    json.dump(out, open(OUT_JSON, "w"), indent=1)
    print(f"{reps} repetitions, {len(nets)} networks; mean relative log-loss (GHB better in n networks)")
    for group in ("ablation", "sensitivity"):
        print(f"--- {group}")
        for name, r in out[group].items():
            print(f"{name:48s} " + "  ".join(f"{r[e]['mean']:.4f} ({r[e]['ghb_better']:2d})" for e in ("1", "10")))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "score":
        score()
        sys.exit()
    D.OUT.mkdir(exist_ok=True)
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    todo = [(w, r) for r in range(reps) for w in ("dev", "fresh") if not (D.OUT / f"{w}_{r}.json").exists()]
    with Pool(4) as pool:
        for it in pool.imap_unordered(D.run, todo):
            print("done", it, flush=True)
