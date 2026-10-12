"""Development check of the evidence-weighted default prior on UCI Heart Disease (the development network).

real_heart.py's protocol (10 repeats of stratified five-fold cross-validation, training sets capped at 20, 40, 80 or
all patients); GHB as reported against GHB with the evidence-weighted prior, at epsilon = 1, 5 and without privacy,
and local MAP with the default prior. Run from simulations/: python3 dev/prior_eb_heart.py
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ghb  # noqa: E402
import ghb_logit as GL  # noqa: E402
import real_heart as RH  # noqa: E402
from exp7_logistic import P, fit  # noqa: E402

REPEATS = 10


def fits(rng, train, n, pub, avg, eps):
    mu0 = np.zeros(P)
    g, sd = GL.release(rng, train, n, mu0, eps, RH.DELTA)
    th, r, drop = GL.invert(pub, g, sd, n, mu0, avg, anchor_sd=GL.PRIOR_SD, check=False)
    return (GL.estimate(train, th, r, drop, mu0, avg, anchor_sd=None),
            GL.estimate(train, th, r, drop, mu0, avg, anchor_sd=None, prior_sd=GL.PRIOR_SDS))


def run(nmax):
    rng = np.random.default_rng([919, 0 if nmax is None else nmax])
    data = RH.load()
    K = len(data)
    pub = GL.Public(RH.public_features(rng))
    avg = ghb.Averager(ghb.metropolis_hastings(ghb.topologies(K)["Ring (degree 2)"]), 60)
    ll = {}
    for _ in range(REPEATS):
        fs = [RH.folds(rng, y, 5) for _, y in data]
        for f in range(5):
            train, test = [], []
            for k, (X, y) in enumerate(data):
                tr = RH.subsample(rng, y, np.flatnonzero(fs[k] != f), nmax)
                te = np.flatnonzero(fs[k] == f)
                train.append((X[tr], y[tr])); test.append((X[te], y[te]))
            n = np.array([len(y) for _, y in train], float)
            est = {"Local": np.array([fit(X, y, np.zeros(P), RH.PRIOR) for X, y in train])}
            for e, t in ((1.0, "eps=1"), (5.0, "eps=5"), (np.inf, "no DP")):
                est[f"GHB {t}"], est[f"GHB evidence prior {t}"] = fits(rng, train, n, pub, avg, e)
            for m, e in est.items():
                ll.setdefault(m, []).append([np.mean(np.logaddexp(0, Xt @ e[k]) - yt * (Xt @ e[k])) for k, (Xt, yt) in enumerate(test)])
    ll = {m: np.array(v) for m, v in ll.items()}
    out = {m: dict(rel=float(v.mean() / ll["Local"].mean()), worst=float((v.mean(0) / ll["Local"].mean(0)).max())) for m, v in ll.items()}
    tag = "all" if nmax is None else str(nmax)
    print(tag, {m: (round(o["rel"], 3), round(o["worst"], 2)) for m, o in out.items()}, flush=True)
    return tag, out


if __name__ == "__main__":
    with Pool(4) as pool:
        res = dict(pool.map(run, RH.NMAX))
    json.dump(res, open(Path(__file__).with_name("prior_eb_heart.json"), "w"), indent=1)
