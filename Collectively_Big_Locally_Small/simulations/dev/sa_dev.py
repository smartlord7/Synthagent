"""Development of GHB with secure aggregation (ghb_sa.py) on simulations, development seeds 930-939.

GHB as reported, GHB-EB, GHB-SA and DP-FedAvg + fine-tuning with secure aggregation (configuration chosen on the
true loss over a small grid, which favours it), in the settings of dev/prior_eb.py. Run from simulations/.
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ghb  # noqa: E402
import ghb_logit as GL  # noqa: E402
import ghb_sa as SA  # noqa: E402
import ghb_variants as V  # noqa: E402
import logit_world as L  # noqa: E402
from exp7_logistic import P, fit  # noqa: E402

REPS = 30
SETTINGS = [("tau2=0.01", dict(tau2=0.01, eps=1.0)), ("tau2=0.05", dict(tau2=0.05, eps=1.0)),
            ("tau2=0.2", dict(tau2=0.2, eps=1.0)), ("tau2=0.5", dict(tau2=0.5, eps=1.0)),
            ("eps=0.3", dict(tau2=0.05, eps=0.3)), ("shift=1.0", dict(tau2=0.05, eps=1.0, shift=1.0)),
            ("no anchor", dict(tau2=0.05, eps=1.0, anchor=False)),
            ("no anchor, median 25", dict(tau2=0.05, eps=1.0, anchor=False, median=25)),
            ("no anchor, median 25, eps=0.3", dict(tau2=0.05, eps=0.3, anchor=False, median=25))]
FT = [(R, eta, E) for R in (3, 10, 30) for eta in (0.5, 2.0) for E in (0, 5, 30)]


def run(item):
    i, (label, kw) = item
    rng = np.random.default_rng([930 + i])
    n = ghb.site_sizes(median=kw.get("median", 100))
    avg = ghb.Averager(ghb.metropolis_hastings(ghb.topologies(24)["Ring + 2-hop (degree 4)"]), 60)
    anchored = kw.get("anchor", True)
    acc = {}
    for _ in range(REPS):
        theta, means, mu0, data = L.world(rng, n, kw["tau2"], kw.get("shift", 0.0), 0.3)
        pub = GL.Public(L.features(rng, 20000))
        rel = V.Releases(rng, data, n, mu0)
        base = {} if anchored else {"anchor_sd": None, "check": False}
        est = {"GHB (reported)": V.estimate(pub, rel, data, n, mu0, kw["eps"], avg, base),
               "GHB-EB": V.estimate(pub, rel, data, n, mu0, kw["eps"], avg, {**base, "prior_sd": GL.PRIOR_SDS}),
               "GHB-SA": SA.estimate(rng, pub, data, n, mu0 if anchored else np.zeros(P), kw["eps"],
                                     anchor_sd=0.3 if anchored else None),
               "Local": np.array([fit(X, y, mu0 if anchored else np.zeros(P), np.full(P, 1.0 if anchored else 0.16)) for X, y in data])}
        rho = GL.rho_total(kw["eps"], 1e-5)
        for R in (3, 10, 30):
            for eta in (0.5, 2.0):
                for E, th in SA.fedavg_sa_ft(rng, data, n, mu0 if anchored else np.zeros(P), rho, R, eta, steps=(0, 5, 30)).items():
                    est[f"FTSA|{R}|{eta}|{E}"] = th
        res = [L.evaluate(rng, theta[k], means[k], {m: e[k] for m, e in est.items()}) for k in range(len(n))]
        for m in est:
            acc.setdefault(m, []).append(np.array([res[k][m][0] for k in range(len(n))]))
    acc = {m: np.array(v) for m, v in acc.items()}
    ft = min([m for m in acc if m.startswith("FTSA")], key=lambda m: acc[m].mean())
    acc["DP-FedAvg + FT, secure aggregation (tuned)"] = acc[ft]
    loc = acc["Local"]
    out = {m: dict(kl=float(v.mean()), rel=float(v.mean() / loc.mean()), worst=float((v.mean(0) / loc.mean(0)).max()))
           for m, v in acc.items() if not m.startswith("FTSA")}
    print(label, ft, {m: (round(1000 * o["kl"], 1), round(o["rel"], 3), round(o["worst"], 2)) for m, o in out.items()}, flush=True)
    return label, out


if __name__ == "__main__":
    with Pool(4) as pool:
        res = dict(pool.map(run, list(enumerate(SETTINGS))))
    json.dump(res, open(Path(__file__).with_name("sa_dev.json"), "w"), indent=1)
