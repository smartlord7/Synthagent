"""GHB v5 development on the seven small networks and UCI Heart Disease of real_networks.py, with the same folds.

The main random stream of real_networks.run is consumed by its own estimators between folds, so they are rerun (on that
stream only; the secure-aggregation, personalized and v4 streams are skipped). The v5 variants use a new stream
[24, i]. Output: results/v5_networks_ckpt/<network>.json with rel (log-loss relative to Local) per variant, to be
compared with the stored literature rows of real_networks.json (same folds).
    python3 dev/v5_networks.py
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ghb  # noqa: E402
import ghb_v4 as V4  # noqa: E402
import real_networks as RN  # noqa: E402

INT = dict(int_sd=2.5, free_int=2.5)
V5 = {"v4": RN.V4_CFG,
      "v5": dict(RN.V4_CFG, **INT, tight=True, hdiag="auto", noise_aware=True),
      "v5 + loo": dict(RN.V4_CFG, **INT, tight=True, hdiag="auto", noise_aware=True, weights="pbma")}
OUT = ghb.RESDIR / "v5_networks_ckpt"
if "--frozen" in sys.argv:  # the frozen GHB v5 (real_v3.GHB_V5 without the pan-network step: one network per run)
    V5 = {"GHB v5": dict(RN.V4_CFG, **INT, tight=True, hdiag="auto")}
    OUT = ghb.RESDIR / "v5_networks_frozen_ckpt"
_orig = RN.estimates


def run(item):
    i, nw = item
    rng5 = np.random.default_rng([24, i])

    def est(rng, train, pub, avg, W, rng_sa=None, rng_p=None, rng_v4=None):
        e = _orig(rng, train, pub, avg, W, None, None, None)  # main stream exactly as in the original run
        out = {"Local": e["Local"]}
        for e_ in RN.BASE_EPS:
            for name, cfg in V5.items():
                out[f"{name} {e_:g}"] = V4.estimate({"x": train}, rng5, e_, RN.DELTA, np.sqrt(float(RN.P)), cfg=cfg)["x"]
        return out

    RN.estimates = est
    RN.TUNED = {}
    RN.METHODS = ["Local"] + [f"{n} {e:g}" for e in RN.BASE_EPS for n in V5]
    name, res = RN.run(item)
    json.dump({m: res[m] for m in RN.METHODS}, open(OUT / f"{name}.json", "w"))
    return name


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    todo = [(i, nw) for i, nw in enumerate(RN.NETWORKS) if not (OUT / f"{nw[0]}.json").exists()]
    with Pool(2) as pool:
        for n in pool.imap_unordered(run, todo):
            print("done", n, flush=True)
