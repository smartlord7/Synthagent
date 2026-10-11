"""Development of the evidence-weighted default prior (revision after the real-data benchmark).

Rule under test: each hospital integrates the scale of the default prior out over {0.5, 1, 2.5} with equal prior
weights, weighted by the Laplace evidence of its own data (ghb_logit.PRIOR_SDS), exactly as tau is integrated out.
Development data only: Simulation 16's eight settings and two settings closer to the real networks (no anchor; no
anchor with hospitals of median size 25), on the development seeds 900-909, which no reported simulation uses.
Paired comparison with GHB as reported, 40 replicates per setting. Run from simulations/: python3 dev/prior_eb.py
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ghb  # noqa: E402
import ghb_logit as GL  # noqa: E402
import ghb_variants as V  # noqa: E402
import logit_world as L  # noqa: E402
from exp7_logistic import P, fit  # noqa: E402

REPS = 40
EB = GL.PRIOR_SDS
SETTINGS = [("tau2=0.01", dict(tau2=0.01, eps=1.0)), ("tau2=0.05", dict(tau2=0.05, eps=1.0)),
            ("tau2=0.2", dict(tau2=0.2, eps=1.0)), ("tau2=0.5", dict(tau2=0.5, eps=1.0)),
            ("eps=0.3", dict(tau2=0.05, eps=0.3)), ("eps=inf", dict(tau2=0.05, eps=np.inf)),
            ("anchor=1.0", dict(tau2=0.05, eps=1.0, anchor_err=1.0)), ("shift=1.0", dict(tau2=0.05, eps=1.0, shift=1.0)),
            ("no anchor", dict(tau2=0.05, eps=1.0, anchor=False)),
            ("no anchor, median 25", dict(tau2=0.05, eps=1.0, anchor=False, median=25))]
VARIANTS = {"GHB (reported)": {}, "GHB, evidence prior": {"prior_sd": EB}}


def run(item):
    i, (label, kw) = item
    rng = np.random.default_rng([900 + i])
    n = ghb.site_sizes(median=kw.get("median", 100))
    avg = ghb.Averager(ghb.metropolis_hastings(ghb.topologies(24)["Ring + 2-hop (degree 4)"]), 60)
    acc = {}
    for _ in range(REPS):
        theta, means, mu0, data = L.world(rng, n, kw["tau2"], kw.get("shift", 0.0), kw.get("anchor_err", 0.3))
        pub = GL.Public(L.features(rng, 20000))
        rel = V.Releases(rng, data, n, mu0)
        base = {} if kw.get("anchor", True) else {"anchor_sd": None, "check": False}
        est = {name: V.estimate(pub, rel, data, n, mu0, kw["eps"], avg, {**base, **cfg}) for name, cfg in VARIANTS.items()}
        est["Local"] = np.array([fit(X, y, mu0 if kw.get("anchor", True) else np.zeros(P), np.full(P, 1.0 if kw.get("anchor", True) else 0.16))
                                 for X, y in data])
        res = [L.evaluate(rng, theta[k], means[k], {m: e[k] for m, e in est.items()}) for k in range(len(n))]
        for m in est:
            acc.setdefault(m, []).append(np.array([res[k][m][0] for k in range(len(n))]))
    acc = {m: np.array(v) for m, v in acc.items()}
    ref = acc["GHB (reported)"].mean(1)
    out = {}
    for m, v in acc.items():
        d = v.mean(1) / ref
        out[m] = dict(kl=float(v.mean()), ratio=float(v.mean() / acc["GHB (reported)"].mean()),
                      ratio_se=float(d.std(ddof=1) / np.sqrt(REPS)),
                      worst_vs_local=float((v.mean(0) / acc["Local"].mean(0)).max()))
    print(label, {m: (round(1000 * o["kl"], 1), round(o["ratio"], 3), round(o["ratio_se"], 3), round(o["worst_vs_local"], 2))
                  for m, o in out.items()}, flush=True)
    return label, out


if __name__ == "__main__":
    with Pool(4) as pool:
        res = dict(pool.map(run, list(enumerate(SETTINGS))))
    json.dump(res, open(Path(__file__).with_name("prior_eb.json"), "w"), indent=1)
