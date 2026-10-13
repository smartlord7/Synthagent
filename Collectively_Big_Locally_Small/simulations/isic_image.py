"""Development of GHB for images on FLamby's ISIC 2019 (declared development; see dev/PREREG_image.md).

    python3 isic_image.py dev [--reps 5] [--d 32]    run (checkpoint per repetition in results/isic_image_ckpt)
    python3 isic_image.py score                      summary (results/isic_image_dev.json)

Data: FLamby's ISIC 2019 centers and splits with FLamby's frozen encoder (flamby_bench.load_isic embeddings); the
projection is the PCA of the encoder features of the public ISIC 2019 Test images (flamby_bench isic-embed-public),
d components; per-center standardization with the center's training statistics, /3, clip, intercept (P = d + 1);
the public rows for GHB's curvature are standardized with their own statistics the same way.
Methods (each repetition r uses seed [2027, r]; epsilon in {1, 10, inf}; softmax head over the 8 classes):
  Local, Local evidence prior, Pooled; GHB variants (ghb_image.estimate); the literature at every configuration of
  real_v3.GRIDS (ghb_image.baselines), scored at the realistic configuration (dev/v4_realistic_configs.json) and at
  the oracle (best mean over repetitions per family and budget).
Balanced accuracy per center (the classes present in its test set), with no adjustment, with the model-implied
adjustment (ghb_image.predict, adjust="model") and with the center's own training class mix (adjust=freq).
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import flamby_bench as FB
import ghb
import ghb_image as GI

CKPT = ghb.RESDIR / "isic_image_ckpt"
OUT = ghb.RESDIR / "isic_image_dev.json"
EPS = (1.0, 10.0, np.inf)
K = 8
PUB = FB.ISIC_DIR / "public_test_embeddings.npz"
VARIANTS = {"GHB": dict(R1=3), "GHB R1=1": dict(R1=1), "GHB free intercepts": dict(R1=3, free_int=True)}
ADJ = ("none", "model", "own")


def _prep(Z, mu, sd):
    return np.hstack([np.ones((len(Z), 1)), np.clip((Z - mu) / sd / 3, -1, 1)])


def load(d):
    from sklearn.decomposition import PCA
    z = np.load(FB.ISIC_DIR / "embeddings.npz", allow_pickle=True)
    pub = np.load(PUB, allow_pickle=True)["emb"]
    split = pd.read_csv(FB.DATA / "fed_isic2019" / "train_test_split.csv").set_index("image").loc[z["images"]]
    pca = PCA(d, random_state=0).fit(pub)
    Z, Zp = pca.transform(z["emb"]), pca.transform(pub)
    train, test = [], []
    for k in range(6):
        tr, te = (split["fold2"] == f"train_{k}").values, (split["fold2"] == f"test_{k}").values
        mu, sd = Z[tr].mean(0), Z[tr].std(0, ddof=1) + 1e-9
        train.append((_prep(Z[tr], mu, sd), split["target"].values[tr].astype(int)))
        test.append((_prep(Z[te], mu, sd), split["target"].values[te].astype(int)))
    return train, test, _prep(Zp, Zp.mean(0), Zp.std(0, ddof=1) + 1e-9)


def realistic_keys():
    return set(FB.REALISTIC.values())


def run_rep(rep, d, full_grid):
    rng = np.random.default_rng([2027, rep])
    train, test, Xpub = load(d)
    P = d + 1
    G = np.sqrt(P)
    D = P * K
    data = [(X, GI.onehot(y, K)) for X, y in train]
    Xp, Yp = np.vstack([X for X, _ in data]), np.vstack([Y for _, Y in data])
    est = {"Local": np.array([GI.fit(X, Y, np.zeros(D), 1 / 2.5 ** 2) for X, Y in data]),
           "Local, evidence prior": np.array([GI.local_eb(X, Y) for X, Y in data]),
           "Pooled": np.repeat(GI.fit(Xp, Yp, np.zeros(D), 1 / 2.5 ** 2)[None], len(data), 0)}
    for e in EPS:
        for name, cfg in VARIANTS.items():
            est[f"{name}, eps={FB.etag(e)}"] = GI.estimate(train, Xpub, rng, e, FB.DELTA, G, {**cfg, "K": K})
        keys = None if full_grid else realistic_keys()
        for k, v in GI.baselines(rng, train, GI.GL.rho_total(e, FB.DELTA), G, K, keys).items():
            est[f"{k}, eps={FB.etag(e)}"] = v
    res = {}
    for m, th in est.items():
        row = []
        for k, ((Xt, yt), (X, y)) in enumerate(zip(test, train)):
            own = (np.bincount(y, minlength=K) + 1.0) / (len(y) + K)
            row.append({a: GI.balanced_accuracy(GI.predict(th[k], Xt, X, None if a == "none" else ("model" if a == "model" else own)), yt)
                        for a in ADJ})
        res[m] = row
    return res


def task(item):
    rep, d, full = item
    res = run_rep(rep, d, full)
    json.dump(res, open(CKPT / f"d{d}_{rep}.json", "w"))
    print("checkpointed", d, rep, flush=True)


def main(reps, d, full):
    CKPT.mkdir(parents=True, exist_ok=True)
    todo = [(r, d, full) for r in range(reps) if not (CKPT / f"d{d}_{r}.json").exists()]
    with Pool(min(4, len(todo) or 1)) as pool:
        list(pool.imap_unordered(task, todo))


def score():
    out = {}
    for d in sorted({int(p.name.split("_")[0][1:]) for p in CKPT.glob("d*_*.json")}):
        runs = [json.load(open(p)) for p in sorted(CKPT.glob(f"d{d}_*.json"))]
        rows = {}
        for m in runs[0]:
            rows[m] = {a: float(np.mean([np.mean([c[a] for c in r[m]]) for r in runs])) for a in ADJ}
        fams = {}
        for lab, code in FB.FAMILIES.items():
            for e in ("1", "10", "inf"):
                ks = [m for m in rows if m.startswith(code + "|") and m.endswith(f", eps={e}")]
                if not ks:
                    continue
                for a in ADJ:
                    best = max(ks, key=lambda m: rows[m][a])
                    fams.setdefault(f"{lab} (oracle), eps={e}", {})[a] = rows[best][a]
                rk = FB.REALISTIC.get(f"{lab}, eps={e}", FB.REALISTIC.get(f"{lab}, eps=10"))
                if f"{rk}, eps={e}" in rows:
                    fams[f"{lab} (realistic), eps={e}"] = rows[f"{rk}, eps={e}"]
        keep = {m: r for m, r in rows.items() if "|" not in m}
        out[f"d={d}"] = dict(reps=len(runs), rows={**keep, **fams})
        print(f"\nd = {d}, {len(runs)} repetitions; balanced accuracy (adjustment: {', '.join(ADJ)})")
        for m, r in {**keep, **fams}.items():
            print(f"{m:48s}" + "".join(f"{r[a]:8.3f}" for a in ADJ))
    json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "score"
    arg = lambda k, v: type(v)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else v
    if cmd == "dev":
        main(arg("--reps", 5), arg("--d", 32), "--full" in sys.argv)
    else:
        score()
