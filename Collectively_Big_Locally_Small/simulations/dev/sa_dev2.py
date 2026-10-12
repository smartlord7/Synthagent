"""Development of GHB-SA version 2 (ghb_sa.estimate_v2) on simulations, development seeds 950-959.

Configurations: clip bound C in {4, 8, 16} x inverse-variance round 3 on/off (default budget split). References on
the same data: local, GHB-EB, GHB-EB without privacy, DP-FedAvg + fine-tuning with secure aggregation, both tuned on
the true loss (oracle) and with one fixed configuration (R = 10, step 0.5, 5 fine-tuning steps; the realistic arm).
All settings without an anchor, as in the real networks, plus Simulation 16's main ones with it. Run from simulations/.
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

REPS = 20
SETTINGS = [("tau2=0.01", dict(tau2=0.01)), ("tau2=0.05", dict(tau2=0.05)), ("tau2=0.2", dict(tau2=0.2)),
            ("tau2=0.5", dict(tau2=0.5)), ("tau2=0.05, eps=0.3", dict(tau2=0.05, eps=0.3)),
            ("tau2=0.05, shift=1", dict(tau2=0.05, shift=1.0)), ("median 25", dict(tau2=0.05, median=25)),
            ("median 25, eps=0.3", dict(tau2=0.05, median=25, eps=0.3)), ("median 25, tau2=0.5", dict(tau2=0.5, median=25))]
CONFIGS = [(C, iv) for C in (4.0, 8.0, 16.0) for iv in (False, True)]


def run(item):
    i, (label, kw) = item
    rng = np.random.default_rng([950 + i])
    eps = kw.get("eps", 1.0)
    n = ghb.site_sizes(median=kw.get("median", 100))
    avg = ghb.Averager(ghb.metropolis_hastings(ghb.topologies(24)["Ring + 2-hop (degree 4)"]), 60)
    acc = {}
    z = np.zeros(P)
    for _ in range(REPS):
        theta, means, mu0, data = L.world(rng, n, kw["tau2"], kw.get("shift", 0.0), 0.3)
        pub = GL.Public(L.features(rng, 20000))
        rel = V.Releases(rng, data, n, z)  # releases at the public point 0, the point GHB-EB is told (no anchor)
        base = {"anchor_sd": None, "check": False, "prior_sd": GL.PRIOR_SDS}
        est = {"Local": np.array([fit(X, y, z, np.full(P, 0.16)) for X, y in data]),
               "GHB-EB": V.estimate(pub, rel, data, n, z, eps, avg, base),
               "GHB-EB, no DP": V.estimate(pub, rel, data, n, z, np.inf, avg, base)}
        for C, iv in CONFIGS:
            est[f"GHB-SA2 C={C:g} iv={int(iv)}"] = SA.estimate_v2(rng, pub, data, n, z, eps, C=C, iv=iv)
        rho = GL.rho_total(eps, 1e-5)
        for R in (3, 10, 30):
            for eta in (0.5, 2.0):
                for E, th in SA.fedavg_sa_ft(rng, data, n, z, rho, R, eta, steps=(0, 5, 30)).items():
                    est[f"FTSA|{R}|{eta}|{E}"] = th
        res = [L.evaluate(rng, theta[k], means[k], {m: e[k] for m, e in est.items()}) for k in range(len(n))]
        for m in est:
            acc.setdefault(m, []).append(np.array([res[k][m][0] for k in range(len(n))]))
    acc = {m: np.array(v) for m, v in acc.items()}
    best = min([m for m in acc if m.startswith("FTSA")], key=lambda m: acc[m].mean())
    acc["FedAvg-SA + FT, oracle-tuned"] = acc[best]
    acc["FedAvg-SA + FT, fixed (R=10, 0.5, E=5)"] = acc["FTSA|10|0.5|5"]
    loc = acc["Local"]
    out = {m: dict(rel=float(v.mean() / loc.mean()), worst=float((v.mean(0) / loc.mean(0)).max()))
           for m, v in acc.items() if not m.startswith("FTSA")}
    print(label, best, {m: (round(o["rel"], 3), round(o["worst"], 2)) for m, o in out.items()}, flush=True)
    return label, out


if __name__ == "__main__":
    with Pool(4) as pool:
        res = dict(pool.map(run, list(enumerate(SETTINGS))))
    json.dump(res, open(Path(__file__).with_name("sa_dev2.json"), "w"), indent=1)
