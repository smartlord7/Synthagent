"""Adds GHB v5 (real_v3.GHB_V5 without the pan-network step; ghb_glm with the same options for TCGA-BRCA) to every
FLamby checkpoint (results/flamby_ckpt/{heart,brca,isic}_{rep}.json) as rows "GHB v5, eps=...", on its own random
stream [2026, rep, dataset, 555]; every other row is unchanged. The pre-registered FLamby predictions concern GHB v4
(rows "GHB, ...") and are not affected.
    python3 dev/flamby_v5.py
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import flamby_bench as FB  # noqa: E402
import ghb_glm as GLM  # noqa: E402
import ghb_logit as GL  # noqa: E402
import ghb_v4 as V4  # noqa: E402
import real_v3 as RV  # noqa: E402

V5 = {k: v for k, v in RV.GHB_V5.items() if k != "pan"}


def task(item):
    ds, rep = item
    f = FB.CKPT / f"{ds}_{rep}.json"
    ck = json.load(open(f))
    if f"GHB v5, eps={FB.etag(np.inf)}" in ck["res"]:
        return item
    rng = np.random.default_rng([FB.PUBLIC_SEED, rep, FB.DSID[ds], 555])
    if ds == "heart":
        train, test, _ = FB.load_heart(); G = np.sqrt(14.0)
        for e in FB.EPS:
            th = V4.estimate({"net": train}, rng, e, FB.DELTA, G, cfg=V5)["net"]
            ck["res"][f"GHB v5, eps={FB.etag(e)}"] = [FB.metrics("logistic", th[k], Xt, yt) for k, (Xt, yt) in enumerate(test)]
    elif ds == "brca":
        train, test, _ = FB.load_brca()
        for e in FB.EPS:
            th = GLM.estimate(GLM.POISSON, train, rng, e, FB.DELTA, FB.BRCA_G, cfg=dict(FB.GHB_CFG, **{k: V5[k] for k in ("int_sd", "free_int", "tight", "hdiag")}))
            ck["res"][f"GHB v5, eps={FB.etag(e)}"] = [FB.metrics("poisson", th[k], Xt, yt) for k, (Xt, yt) in enumerate(test)]
    else:
        train, test, _ = FB.load_isic(); G = np.sqrt(17.0)
        for e in FB.EPS:
            rho = GL.rho_total(e, FB.DELTA) / 8
            heads = []
            for c in range(8):
                tc = [(X, (y == c).astype(float)) for X, y in train]
                heads.append(V4.estimate({"net": tc}, rng, FB.eps_for_rho(rho), FB.DELTA, G, cfg=V5)["net"])
            th = np.array(heads).transpose(1, 0, 2)
            ck["res"][f"GHB v5, eps={FB.etag(e)}"] = [FB.metrics("isic", th[k], Xt, yt) for k, (Xt, yt) in enumerate(test)]
    json.dump(ck, open(f, "w"))
    print("done", item, flush=True)
    return item


if __name__ == "__main__":
    items = [(ds, r) for ds in ("heart", "brca", "isic") for r in range(20)]
    with Pool(2) as p:
        list(p.imap_unordered(task, items))
