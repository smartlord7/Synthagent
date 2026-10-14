"""pFedMe rows of the ISIC 2019 image development (isic_image.py) recomputed with the corrected noise (commit 15e769a).

The other methods' rows are unchanged: they consume the same random draws, so a full rerun would only change pFedMe.
Reads results/isic_image_ckpt/pre_pfedme_fix/d32_{rep}.json, writes results/isic_image_ckpt/d32_{rep}.json with every
"PME|..." row replaced; pFedMe's noise comes from its own stream [2027, rep, 777].
    python3 dev/pfedme_fix_image.py [--reps 10]
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import flamby_bench as FB  # noqa: E402
import ghb_image as GI  # noqa: E402
import isic_image as II  # noqa: E402
import personalized as PZ  # noqa: E402

D32 = 32


def patch(rep):
    old = json.load(open(II.CKPT / "pre_pfedme_fix" / f"d{D32}_{rep}.json"))
    train, test, _ = II.load(D32)
    K, P = II.K, D32 + 1
    G = np.sqrt(P)
    data = [(X, GI.onehot(y, K)) for X, y in train]
    n = np.array([len(y) for _, y in train], float)
    rng = np.random.default_rng([2027, rep, 777])
    new = dict(old)
    for e in II.EPS:
        rho = GI.GL.rho_total(e, FB.DELTA)
        for R, eta, lam in PZ.GRID["pFedMe"]:
            th = GI.pfedme(rng, data, n, rho, R, eta, lam, 2 * np.sqrt(2) * G, P * K)
            row = []
            for k, ((Xt, yt), (X, y)) in enumerate(zip(test, train)):
                sup = np.bincount(y, minlength=K) > 0
                row.append({a: GI.balanced_accuracy(GI.predict(th[k], Xt, X, None if a == "none" else "model",
                                                               sup if a == "support" else None), yt) for a in II.ADJ})
            key = f"{GI.ckey(('PME', R, eta, lam))}, eps={FB.etag(e)}"
            assert key in old, key
            if np.isinf(e):  # no noise: must reproduce the old row exactly
                assert all(abs(row[k][a] - old[key][k][a]) < 1e-9 for k in range(len(row)) for a in II.ADJ), key
            new[key] = row
    json.dump(new, open(II.CKPT / f"d{D32}_{rep}.json", "w"))
    print("patched", rep, flush=True)


if __name__ == "__main__":
    reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 10
    with Pool(2) as pool:
        list(pool.imap_unordered(patch, [r for r in range(reps) if not (II.CKPT / f"d{D32}_{r}.json").exists()]))
