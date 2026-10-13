"""GHB on the FLamby benchmark (Ogier du Terrail et al. 2022), as pre-registered in dev/PREREG_flamby.md.

    python3 flamby_bench.py heart|brca|isic [--reps 20]   run (checkpoint per repetition)
    python3 flamby_bench.py score                         supplement/results/flamby_results.json and summary tables
    python3 flamby_bench.py check                         validation (splits, equivalence of the GLM code, Poisson GHB)
    python3 flamby_bench.py isic-embed                    ISIC2019 embeddings (needs ImageNet weights and the images)
    python3 flamby_bench.py isic-embed-public ZIP OUT     embeddings of the public ISIC 2019 Test images (isic_image.py)

Data and splits reproduce FLamby (commit edacf54, June 2024) in numpy/pandas without importing it:
  Heart  UCI processed.{cleveland,hungarian,switzerland,va}.data; columns 10-12 dropped, rows with missing values dropped,
         label > 0 -> 1, dummy coding of columns 2 (chest pain) and 6 (rest ECG) with drop_first on the pooled frame,
         per-center split by sklearn train_test_split(train_size=0.66, random_state=43, stratified on the label when
         both classes have more than two patients), standardization with each center's training mean and (n - 1)
         standard deviation (+1e-9). Then, as pre-registered, divided by 3 and clipped to [-1, 1]; intercept
         first: P = 14, G = sqrt(14).
  BRCA   FLamby's brca.csv (39 features) and train_test_split.csv (fold2 = train_k / test_k, k = region 0..5).
         Age (the only continuous feature) standardized with each region's training statistics, divided by 3 and
         clipped; the 38 binary one-hot columns kept. P = 40 with the intercept. G = 4: the 38 binary columns are
         the one-hot coding of 14 categorical variables (public schema), so ||x||^2 <= 1 + 1 + 14 = 16.
         Outcome y = (d, t, c): d = event, t = min(days / 3650, 1), c = G / ||x|| (the clipping cap of the
         literature methods, unused by the exact Poisson family).
  ISIC   FLamby's train_test_split (center, fold2); images resized (shorter side 224, bilinear) with color constancy
         as FLamby's resize_images.py, then FLamby's test-time transform (center crop 200, ImageNet normalization) for
         every image (frozen encoder: no augmentation); EfficientNet-B0 (torchvision, ImageNet weights) average-pooled
         features (1280); projection to 16 dimensions by a Gaussian matrix with public seed 2026; per-center training
         standardization, /3, clip; intercept: P = 17, G = sqrt(17). 8 one-vs-rest heads, each with budget rho / 8.

Methods (each repetition r uses rng seed [2026, r, dataset]; epsilon in {1, 10, inf}, delta 1e-5):
  Local (MAP, prior N(0, 2.5^2)); Local, evidence prior (ghb_v3.combine over s in {0.5, 1, 2.5});
  Pooled (MAP on the pooled training data, prior N(0, 2.5^2), no DP) - one model for every hospital;
  GHB = ghb_v4.estimate with dict(f=1, safety=True, R1=1, pan=False) (logistic) or ghb_glm.estimate (Poisson);
  GHB-lite = ghb_v3.estimate with the GHB-EB configuration of real_v3 and a nominal public feature sample (intercept
  and, per feature, clip(N(0, 1) / 3, -1, 1), 20,000 draws, seed 2026, fixed before any data were seen); logistic only;
  the six literature families of real_v3.GRIDS, every configuration (ghb_glm.baselines; logistic family identical to
  real_v3.baselines, Poisson family with per-patient gradients clipped to norm G).
"""
import json
import sys
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

import ghb
import ghb_glm as GLM
import ghb_logit as GL
import ghb_v3 as V3
import ghb_v4 as V4
from exp7_logistic import fit

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ISIC_DIR = Path("/home/user/flamby_data/isic")
CKPT = ghb.RESDIR / "flamby_ckpt"
OUT = ghb.RESDIR / "flamby_results.json"
EPS = (1.0, 10.0, np.inf)
DELTA = 1e-5
GHB_CFG = dict(f=1.0, safety=True, R1=1, pan=False)  # real_v3.GHB_V4 with pan off (one network)
GHB_EB = dict(safety=False, learned_scale=False, fullcov=False, cross=False)  # real_v3.VARIANTS["GHB-EB"]
DSID = {"heart": 0, "brca": 1, "isic": 2}
PUBLIC_SEED = 2026
FAMILIES = {"DP-FedAvg + FT": "FT", "DP-FedAvg-SA + FT": "FTSA", "APFL": "APFL", "Ditto": "PDIT", "pFedMe": "PME",
            "Clustered DP-FedAvg": "CLU"}
REALISTIC = json.load(open(Path(__file__).resolve().parent / "dev" / "v4_realistic_configs.json"))


def etag(e):
    return f"{e:g}"


# ------------------------------------------------------------------------------------------------------------- data
def _scale(Xtr, Xte, cols):
    """Standardize `cols` with the training mean and (n - 1) sd (+1e-9, as FLamby), divide by 3, clip to [-1, 1]."""
    Xtr, Xte = Xtr.copy(), Xte.copy()
    m, s = Xtr[:, cols].mean(0), Xtr[:, cols].std(0, ddof=1)
    for A in (Xtr, Xte):
        A[:, cols] = np.clip((A[:, cols] - m) / (s + 1e-9) / 3, -1, 1)
    return Xtr, Xte


def load_heart():
    from sklearn.model_selection import train_test_split
    feats, labels, centers, sets = [], [], [], []
    for k, name in enumerate(["cleveland", "hungarian", "switzerland", "va"]):
        df = pd.read_csv(DATA / "heart_disease" / f"processed.{name}.data", header=None)
        df = df.replace("?", np.nan).drop([10, 11, 12], axis=1).dropna(axis=0).apply(pd.to_numeric)
        X, y = df.iloc[:, :-1], (df.iloc[:, -1] != 0).astype(int)
        nb = len(df)
        levels = np.unique(y)
        strat = y if len(levels) > 1 and all((y == l).sum() > 2 for l in levels) else None
        _, te = train_test_split(np.arange(nb), test_size=1.0 - 0.66, train_size=0.66, random_state=43, shuffle=True,
                                 stratify=strat)
        feats.append(X); labels.append(y.values); centers += [k] * nb
        sets += ["test" if i in set(te) else "train" for i in range(nb)]
    F = pd.get_dummies(pd.concat(feats, ignore_index=True), columns=[2, 6], drop_first=True).astype(float).values
    yall, centers, sets = np.concatenate(labels).astype(float), np.array(centers), np.array(sets)
    train, test = [], []
    for k in range(4):
        tr, te = (centers == k) & (sets == "train"), (centers == k) & (sets == "test")
        Xtr, Xte = _scale(F[tr], F[te], np.arange(F.shape[1]))
        one = lambda A: np.hstack([np.ones((len(A), 1)), A])
        train.append((one(Xtr), yall[tr])); test.append((one(Xte), yall[te]))
    return train, test, ["Cleveland", "Hungary", "Switzerland", "Long Beach V"]


BRCA_G = 4.0


def load_brca():
    b = pd.read_csv(DATA / "fed_tcga_brca" / "brca.csv")
    s = pd.read_csv(DATA / "fed_tcga_brca" / "train_test_split.csv")
    train, test = [], []
    for k in range(6):
        parts = []
        for f in ("train", "test"):
            d = b[b["pid"].isin(s.loc[s["fold2"] == f"{f}_{k}", "pid"])]
            X = d.iloc[:, 1:40].values.astype(float)
            parts.append((X, d["E"].values.astype(float), d["T"].values.astype(float)))
        (Xtr, dtr, Ttr), (Xte, dte, Tte) = parts
        Xtr, Xte = _scale(Xtr, Xte, [0])
        out = []
        for X, d, T in ((Xtr, dtr, Ttr), (Xte, dte, Tte)):
            X = np.hstack([np.ones((len(X), 1)), X])
            assert np.linalg.norm(X, axis=1).max() <= BRCA_G + 1e-12
            y = np.column_stack([d, np.minimum(T / 3650.0, 1.0), BRCA_G / np.linalg.norm(X, axis=1), T])
            out.append((X, y))
        train.append(out[0]); test.append(out[1])
    return train, test, ["Northeast", "South", "West", "Midwest", "Europe", "Canada"]


def load_isic():
    path = ISIC_DIR / "embeddings.npz"
    if not path.exists():
        raise SystemExit(f"{path} missing: run `python3 flamby_bench.py isic-embed` (needs ImageNet weights)")
    z = np.load(path, allow_pickle=True)
    split = pd.read_csv(DATA / "fed_isic2019" / "train_test_split.csv").set_index("image").loc[z["images"]]
    Z = z["emb"] @ (np.random.default_rng(PUBLIC_SEED).standard_normal((z["emb"].shape[1], 16)) / np.sqrt(z["emb"].shape[1]))
    train, test = [], []
    for k in range(6):
        tr, te = (split["fold2"] == f"train_{k}").values, (split["fold2"] == f"test_{k}").values
        Xtr, Xte = _scale(Z[tr], Z[te], np.arange(16))
        one = lambda A: np.hstack([np.ones((len(A), 1)), A])
        train.append((one(Xtr), split["target"].values[tr])); test.append((one(Xte), split["target"].values[te]))
    return train, test, ["BCN", "HAM_vidir_molemax", "HAM_vidir_modern", "HAM_rosendahl", "MSK", "HAM_vienna_dias"]


def public_sample(P, m=20000):
    rng = np.random.default_rng(PUBLIC_SEED)
    return np.hstack([np.ones((m, 1)), np.clip(rng.standard_normal((m, P - 1)) / 3, -1, 1)])


# ---------------------------------------------------------------------------------------------------------- metrics
def auc(s, y):
    if y.min() == y.max():
        return None
    from scipy.stats import rankdata
    r = rankdata(s)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def harrell_c(T, risk, E):
    """Harrell's C as lifelines.utils.concordance_index(T, -risk, E): pairs with T_i < T_j and E_i = 1 are
    comparable, and so are pairs where j is censored at T_i; ties in risk count 1/2. None without comparable pairs."""
    num = den = 0.0
    for i in np.flatnonzero(E == 1):
        j = (T > T[i]) | ((T == T[i]) & (E == 0))  # a subject censored at the event time counts as longer (lifelines)
        den += j.sum()
        num += (risk[i] > risk[j]).sum() + 0.5 * (risk[i] == risk[j]).sum()
    return None if den == 0 else float(num / den)


def metrics(kind, th, Xt, yt):
    if kind == "logistic":
        s = Xt @ th
        return dict(acc=float(np.mean((expit(s) > 0.5) == (yt == 1))), logloss=float(np.mean(np.logaddexp(0, s) - yt * s)),
                    auc=auc(s, yt))
    if kind == "poisson":
        s = Xt @ th
        d, t, T = yt[:, 0], yt[:, 1], yt[:, 3]
        mu = t * np.exp(np.minimum(s, 50))
        dev = 2 * (mu - d - d * np.log(np.where(d > 0, mu, 1.0)))
        try:  # FLamby's metric: lifelines.utils.concordance_index(T, -eta, E); harrell_c is the same (checked)
            from lifelines.utils import concordance_index
            ci = float(concordance_index(T, -s, d))
        except ZeroDivisionError:  # no comparable pair
            ci = None
        return dict(cindex=ci, deviance=float(np.mean(dev)))
    # isic: th is (8, P), one head per class
    S = Xt @ th.T
    pred = S.argmax(1)
    cls = np.unique(yt)
    bacc = float(np.mean([np.mean(pred[yt == c] == c) for c in cls]))
    p = expit(S)
    p = p / p.sum(1, keepdims=True)
    return dict(bacc=bacc, logloss=float(-np.mean(np.log(np.maximum(p[np.arange(len(yt)), yt.astype(int)], 1e-300)))))


# ------------------------------------------------------------------------------------------------------- estimation
def eps_for_rho(rho):
    """epsilon whose zCDP budget (rho_total) is rho: the conversion rho + 2 sqrt(rho ln(1/delta))."""
    return np.inf if np.isinf(rho) else rho + 2 * np.sqrt(rho * np.log(1 / DELTA))


def fit_logistic_all(rng, train, eps, P, G, pub, rho_share=1.0):
    """GHB, GHB-lite and baselines for one logistic network at one budget (budget rho * rho_share)."""
    rho = GL.rho_total(eps, DELTA) * rho_share
    e = eps if rho_share == 1.0 else eps_for_rho(rho)
    out = {"GHB": V4.estimate({"net": train}, rng, e, DELTA, G, cfg=GHB_CFG)["net"],
           "GHB-lite": V3.estimate({"net": train}, {"net": pub}, rng, e, DELTA, G, cfg=GHB_EB)["net"]}
    out.update(GLM.baselines(GLM.LOGISTIC, rng, train, rho, G))
    return out


def run_rep(dataset, rep):
    rng = np.random.default_rng([PUBLIC_SEED, rep, DSID[dataset]])
    res = {}
    if dataset == "heart":
        train, test, names = load_heart()
        P, G = 14, np.sqrt(14.0)
        pub = V3.Public(public_sample(P))
        Xp, yp = np.vstack([X for X, _ in train]), np.concatenate([y for _, y in train])
        est = {"Local": np.array([fit(X, y, np.zeros(P), np.full(P, 1 / 2.5 ** 2)) for X, y in train]),
               "Local, evidence prior": np.array([V3.combine(X, y, [(0.0, np.zeros(P), np.eye(P) * s ** 2) for s in V3.SDS])
                                                  for X, y in train]),
               "Pooled": np.repeat(fit(Xp, yp, np.zeros(P), np.full(P, 1 / 2.5 ** 2))[None], len(train), 0)}
        for e in EPS:
            for k, v in fit_logistic_all(rng, train, e, P, G, pub).items():
                est[f"{k}, eps={etag(e)}"] = v
        kind = "logistic"
    elif dataset == "brca":
        train, test, names = load_brca()
        P, G = 40, BRCA_G
        Xp, yp = np.vstack([X for X, _ in train]), np.vstack([y for _, y in train])
        fam = GLM.POISSON
        est = {"Local": np.array([GLM.fit(fam, X, y, np.zeros(P), np.full(P, 1 / 2.5 ** 2)) for X, y in train]),
               "Local, evidence prior": np.array([GLM.local_eb(fam, X, y) for X, y in train]),
               "Pooled": np.repeat(GLM.fit(fam, Xp, yp, np.zeros(P), np.full(P, 1 / 2.5 ** 2))[None], len(train), 0)}
        for e in EPS:
            est[f"GHB, eps={etag(e)}"] = GLM.estimate(fam, train, rng, e, DELTA, G, cfg=GHB_CFG)
            for k, v in GLM.baselines(GLM.CLIPPED_POISSON, rng, train, GL.rho_total(e, DELTA), G).items():
                est[f"{k}, eps={etag(e)}"] = v
        kind = "poisson"
    else:
        train, test, names = load_isic()
        P, G = 17, np.sqrt(17.0)
        pub = V3.Public(public_sample(P))
        heads = {}  # name -> (8, K, P)
        for c in range(8):
            tc = [(X, (y == c).astype(float)) for X, y in train]
            Xp, yp = np.vstack([X for X, _ in tc]), np.concatenate([y for _, y in tc])
            h = {"Local": np.array([fit(X, y, np.zeros(P), np.full(P, 1 / 2.5 ** 2)) for X, y in tc]),
                 "Local, evidence prior": np.array([V3.combine(X, y, [(0.0, np.zeros(P), np.eye(P) * s ** 2) for s in V3.SDS])
                                                    for X, y in tc]),
                 "Pooled": np.repeat(fit(Xp, yp, np.zeros(P), np.full(P, 1 / 2.5 ** 2))[None], len(tc), 0)}
            for e in EPS:
                for k, v in fit_logistic_all(rng, tc, e, P, G, pub, rho_share=1 / 8).items():
                    h[f"{k}, eps={etag(e)}"] = v
            for k, v in h.items():
                heads.setdefault(k, [None] * 8)[c] = v
        est = {k: np.array(v).transpose(1, 0, 2) for k, v in heads.items()}  # (K, 8, P)
        kind = "isic"
    for m, th in est.items():
        res[m] = [metrics(kind, th[k], Xt, yt) for k, (Xt, yt) in enumerate(test)]
    info = dict(names=names, n_train=[len(y) for _, y in train], n_test=[len(y) for _, y in test])
    return res, info


def task(item):
    dataset, rep = item
    res, info = run_rep(dataset, rep)
    json.dump(dict(res=res, info=info), open(CKPT / f"{dataset}_{rep}.json", "w"))
    print("checkpointed", dataset, rep, flush=True)
    return rep


def main_run(dataset, reps):
    CKPT.mkdir(parents=True, exist_ok=True)
    todo = [(dataset, r) for r in range(reps) if not (CKPT / f"{dataset}_{r}.json").exists()]
    print("to run:", todo, flush=True)
    with Pool(4) as pool:
        for _ in pool.imap_unordered(task, todo):
            pass


# ------------------------------------------------------------------------------------------------------------ score
METRICS = {"heart": [("acc", 1), ("logloss", -1), ("auc", 1)], "brca": [("cindex", 1), ("deviance", -1)],
           "isic": [("bacc", 1), ("logloss", -1)]}
GRID_KEYS = None


def _grid_keys():
    import real_v3 as RV3
    return {fam: [GLM.ckey(c) for c in RV3.GRIDS[fam]] for fam in FAMILIES}


def predictions(out):
    """Pre-registered predictions 1-3 (dev/PREREG_flamby.md), evaluated mechanically on the realistic arms."""
    res = {}
    lit = lambda t: [f"{f} (realistic), eps={t}" for f in FAMILIES]
    if "heart" in out:
        R = out["heart"]["rows"]
        loc = R["Local"]["logloss"]
        priv = ["GHB, eps=1", "GHB-lite, eps=1"] + lit("1")
        rel = {m: dict(mean_ratio=R[m]["logloss"]["mean"] / loc["mean"],
                       hospital_mean_ratio=float(np.mean(np.array(R[m]["logloss"]["per_hospital"]) / np.array(loc["per_hospital"]))))
               for m in priv}
        res["1"] = dict(statement="Heart eps=1: no private method improves local log-loss by > 3%; GHB within 3% of local",
                        ratios_to_local=rel,
                        held=bool(all(v["mean_ratio"] >= 0.97 for v in rel.values()) and abs(rel["GHB, eps=1"]["mean_ratio"] - 1) <= 0.03))
        d = {}
        for t in ("10", "inf"):
            g = R[f"GHB, eps={t}"]["logloss"]["mean"]
            d[t] = {m: dict(lit=R[m]["logloss"]["mean"], ghb=g, ghb_lower=g < R[m]["logloss"]["mean"]) for m in lit(t)}
        res["2"] = dict(statement="Heart eps=10, inf: GHB mean log-loss below every realistic literature arm", detail=d,
                        held=bool(all(x["ghb_lower"] for v in d.values() for x in v.values())))
    if "brca" in out:
        R = out["brca"]["rows"]
        d = {}
        for t in ("10", "inf"):
            g = R[f"GHB, eps={t}"]["cindex"]["mean"]
            d[t] = {m: dict(other=R[m]["cindex"]["mean"], ghb=g, ghb_at_least=g >= R[m]["cindex"]["mean"]) for m in ["Local"] + lit(t)}
        res["3"] = dict(statement="BRCA eps=10, inf: GHB mean C-index >= local and every realistic literature arm", detail=d,
                        held=bool(all(x["ghb_at_least"] for v in d.values() for x in v.values())))
    for k, v in res.items():
        print(f"prediction {k}: {'HELD' if v['held'] else 'NOT HELD'} - {v['statement']}")
    return res


def score():
    grids = _grid_keys()
    out = {}
    for ds in ("heart", "brca", "isic"):
        files = sorted(CKPT.glob(f"{ds}_*.json"))
        if not files:
            continue
        cks = [json.load(open(f)) for f in files]
        info = cks[0]["info"]
        K = len(info["names"])
        # per method, metric: (reps, K) with nan where undefined
        arr = {}
        for m in cks[0]["res"]:
            for met, _ in METRICS[ds]:
                arr[(m, met)] = np.array([[np.nan if c["res"][m][k][met] is None else c["res"][m][k][met] for k in range(K)]
                                          for c in cks], float)
        hosp = {k: np.nanmean(v, 0) for k, v in arr.items()}  # per-hospital mean over reps
        undefined = {met: [info["names"][k] for k in range(K) if np.isnan(hosp[("Local", met)][k])] for met, _ in METRICS[ds]}
        summ = lambda key: float(np.nanmean(hosp[key]))
        rows = {}

        def add(label, key_by_met):
            rows[label] = {met: dict(mean=summ(key_by_met[met]), per_hospital=hosp[key_by_met[met]].tolist(),
                                     rep_sd=float(np.nanstd(np.nanmean(arr[key_by_met[met]], 1))), config=key_by_met[met][0])
                           for met, _ in METRICS[ds]}
        for m in ("Local", "Local, evidence prior", "Pooled"):
            add(m, {met: (m, met) for met, _ in METRICS[ds]})
        for e in EPS:
            t = etag(e)
            for m in ("GHB", "GHB-lite"):
                if (f"{m}, eps={t}", METRICS[ds][0][0]) in arr:
                    add(f"{m}, eps={t}", {met: (f"{m}, eps={t}", met) for met, _ in METRICS[ds]})
            for fam, code in FAMILIES.items():
                cfg = REALISTIC[f"{fam}, eps={etag(min(e, 10.0))}"]
                add(f"{fam} (realistic), eps={t}", {met: (f"{cfg}, eps={t}", met) for met, _ in METRICS[ds]})
                best = {}
                for met, sign in METRICS[ds]:
                    cand = [(sign * summ((f"{c}, eps={t}", met)), c) for c in grids[fam]]
                    best[met] = (f"{max(cand)[1]}, eps={t}", met)
                add(f"{fam} (oracle), eps={t}", best)
        out[ds] = dict(info=info, reps=len(cks), undefined_metric_hospitals=undefined, rows=rows)
    out["predictions"] = predictions(out)
    json.dump(out, open(OUT, "w"), indent=1)
    for ds, d in out.items():
        if ds not in METRICS:
            continue
        mets = [m for m, _ in METRICS[ds]]
        print(f"\n=== {ds}: {d['reps']} repetitions; hospitals without the metric: {d['undefined_metric_hospitals']}")
        print(f"{'method':48s}" + "".join(f"{m:>10s}" for m in mets))
        for lab, r in d["rows"].items():
            print(f"{lab:48s}" + "".join(f"{r[m]['mean']:10.4f}" for m in mets))


# ------------------------------------------------------------------------------------------------------------- ISIC
def _color_constancy(img, power=6):
    """FLamby's color_constancy (gamma=None), numpy only."""
    dt = img.dtype
    img = img.astype("float32")
    rgb = np.power(np.mean(np.power(img, power), (0, 1)), 1 / power)
    rgb = rgb / np.sqrt(np.sum(np.power(rgb, 2.0)))
    img = np.multiply(img, 1 / (rgb * np.sqrt(3)))
    return img.astype(dt)


def _preprocess(raw):
    """FLamby resize_images.py (shorter side 224, bilinear, color constancy), then its test transform (center crop
    200, ImageNet normalization with max pixel value 255). Returns (3, 200, 200) float32."""
    import io
    from PIL import Image
    img = Image.open(io.BytesIO(raw))
    ratio = 224.0 / min(img.size)
    img = img.resize(tuple(int(x * ratio) for x in img.size), resample=Image.BILINEAR)
    a = _color_constancy(np.array(img))
    h, w = a.shape[:2]
    top, left = (h - 200) // 2, (w - 200) // 2  # albumentations CenterCrop offsets
    a = a[top:top + 200, left:left + 200].astype(np.float32) / 255.0
    a = (a - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    return np.transpose(a, (2, 0, 1))


def isic_embed(batch=64, zpath=None, out=None):
    """Embeds every ISIC 2019 training image (or, with zpath/out, every image of another zip, e.g. the public ISIC 2019
    Test images used for the projection of isic_image.py) with FLamby's baseline encoder: efficientnet_pytorch's EfficientNet-B0 with its
    ImageNet weights (efficientnet-b0-355c32eb.pth, the file EfficientNet.from_pretrained downloads), frozen;
    the 1280-dimensional output of global average pooling."""
    import torch
    from efficientnet_pytorch import EfficientNet
    wpath = ISIC_DIR / "efficientnet-b0-355c32eb.pth"
    if not wpath.exists():  # no substitute: random or untrained weights are not the pre-registered encoder
        raise SystemExit(f"ImageNet weights not found at {wpath}; ISIC2019 not run")
    m = EfficientNet.from_name("efficientnet-b0")
    m.load_state_dict(torch.load(wpath, map_location="cpu"))
    m.eval()
    zf = zipfile.ZipFile(zpath or ISIC_DIR / "ISIC_2019_Training_Input.zip")
    names = {Path(n).stem: n for n in zf.namelist() if n.endswith(".jpg")}
    images = sorted(names) if zpath else list(pd.read_csv(DATA / "fed_isic2019" / "train_test_split.csv")["image"])
    embs = []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            x = torch.from_numpy(np.stack([_preprocess(zf.read(names[im])) for im in images[i:i + batch]]))
            embs.append(torch.flatten(torch.nn.functional.adaptive_avg_pool2d(m.extract_features(x), 1), 1).numpy())
            print(i, flush=True)
    np.savez(out or ISIC_DIR / "embeddings.npz", images=np.array(images), emb=np.vstack(embs))


# ------------------------------------------------------------------------------------------------------------ check
def check():
    import real_v3 as RV3
    import literature as LIT
    import personalized as PZ
    import ghb_sa as SA
    from lifelines.utils import concordance_index
    # 1. FLamby split counts
    tr, te, _ = load_heart()
    print("heart train/test", [(len(a[1]), len(b[1])) for a, b in zip(tr, te)])
    assert [(len(a[1]), len(b[1])) for a, b in zip(tr, te)] == [(199, 104), (172, 89), (30, 16), (85, 45)]
    trb, teb, _ = load_brca()
    print("brca train/test", [(len(a[1]), len(b[1])) for a, b in zip(trb, teb)])
    assert [(len(a[1]), len(b[1])) for a, b in zip(trb, teb)] == [(248, 63), (156, 40), (164, 42), (129, 33), (129, 33), (40, 11)]
    print("heart max ||x||", max(np.linalg.norm(X, axis=1).max() for X, _ in tr + te), "<= sqrt(14)", np.sqrt(14))
    # 2. Harrell's C against lifelines
    for X, y in teb:
        s = X @ np.random.default_rng(0).standard_normal(X.shape[1])
        print("C-index own / lifelines", harrell_c(y[:, 3], s, y[:, 0]), concordance_index(y[:, 3], -s, y[:, 0]))
    # 3. The generic code with the logistic family reproduces the originals (P = 8, real_v3 settings)
    rng = np.random.default_rng(1)
    P, G = 8, np.sqrt(8.0)
    data = []
    for n in (40, 80, 150, 300):
        X = np.hstack([np.ones((n, 1)), np.clip(rng.standard_normal((n, P - 1)) / 2, -1, 1)])
        data.append((X, rng.binomial(1, expit(X @ np.r_[-0.5, rng.standard_normal(P - 1) * 0.5])).astype(float)))
    LIT.configure(P, G); PZ.configure(P, G); SA.G = G; SA.P = P
    for e in (1.0, np.inf):
        a = RV3.baselines(np.random.default_rng(5), data, e)
        b = GLM.baselines(GLM.LOGISTIC, np.random.default_rng(5), data, GL.rho_total(e, DELTA), G)
        print(f"baselines eps={e}: max |diff| over {len(a)} configs", max(np.abs(a[k] - b[k]).max() for k in a))
        a = V4.estimate({"x": data}, np.random.default_rng(6), e, DELTA, G, cfg=GHB_CFG)["x"]
        b = GLM.estimate(GLM.LOGISTIC, data, np.random.default_rng(6), e, DELTA, G, cfg=GHB_CFG)
        print(f"GHB eps={e}: max |diff|", np.abs(a - b).max())
    X, y = data[0]
    print("local fit / local EB diff", np.abs(fit(X, y, np.zeros(P), np.full(P, 0.16)) - GLM.fit(GLM.LOGISTIC, X, y, np.zeros(P), np.full(P, 0.16))).max(),
          np.abs(V3.combine(X, y, [(0.0, np.zeros(P), np.eye(P) * s ** 2) for s in V3.SDS]) - GLM.local_eb(GLM.LOGISTIC, X, y)).max())
    # 4. Heart at eps = inf: GHB is a sane fit between local and pooled; all methods finite
    res, _ = run_rep("heart", 0)
    for m in ("Local", "Local, evidence prior", "Pooled", "GHB, eps=inf", "GHB-lite, eps=inf", "GHB, eps=1", "GHB, eps=10",
              "FT|10|0.2|5, eps=inf", "FT|10|0.2|5, eps=1"):
        print(f"heart rep0 {m:28s} acc {np.mean([r['acc'] for r in res[m]]):.3f} ll {np.mean([r['logloss'] for r in res[m]]):.3f}")
    bad = [m for m, r in res.items() if not all(np.isfinite(x["logloss"]) for x in r)]
    print("heart methods:", len(res), "non-finite:", bad)
    # 5. Poisson GHB on synthetic data: homogeneous and heterogeneous networks
    for tau in (0.0, 1.0, 4.0):
        out = {"Local": [], "Local, evidence prior": [], "GHB": [], "GHB eps=1": [], "GHB worse than local (share)": []}
        for r in range(10):
            rg = np.random.default_rng([77, r])
            P = 8
            mu = np.r_[-1.0, rg.standard_normal(P - 1) * 0.5]
            nets, tests = [], []
            for n in (30, 50, 80, 120, 200, 400):
                th = mu + tau * np.r_[0, rg.standard_normal(P - 1) * 0.5]
                def draw(m):
                    X = np.hstack([np.ones((m, 1)), np.clip(rg.standard_normal((m, P - 1)) / 3, -1, 1)])
                    ev = rg.exponential(1 / np.exp(X @ th))
                    cens = rg.uniform(0, 1.5, m)
                    T = np.minimum(ev, cens)
                    y = np.column_stack([(ev <= cens).astype(float), np.minimum(T, 1.0), np.full(m, 0.0), T])
                    y[:, 2] = np.sqrt(P) / np.linalg.norm(X, axis=1)
                    return X, y, th
                X, y, th_ = draw(n); nets.append((X, y)); tests.append(th_)
            Gs = np.sqrt(P)
            loc = np.array([GLM.fit(GLM.POISSON, X, y, np.zeros(P), np.full(P, 0.16)) for X, y in nets])
            leb = np.array([GLM.local_eb(GLM.POISSON, X, y) for X, y in nets])
            g = GLM.estimate(GLM.POISSON, nets, rg, np.inf, DELTA, Gs, cfg=GHB_CFG)
            g1 = GLM.estimate(GLM.POISSON, nets, rg, 1.0, DELTA, Gs, cfg=GHB_CFG)
            for k, th_ in enumerate(tests):
                out["Local"].append(np.sum((loc[k] - th_) ** 2)); out["GHB"].append(np.sum((g[k] - th_) ** 2))
                out["Local, evidence prior"].append(np.sum((leb[k] - th_) ** 2))
                out["GHB eps=1"].append(np.sum((g1[k] - th_) ** 2))
                out["GHB worse than local (share)"].append(float(out["GHB"][-1] > out["Local"][-1]))
        print(f"synthetic Poisson tau-scale={tau}: mean squared error", {k: round(float(np.mean(v)), 4) for k, v in out.items()})
    # 6. every method at every budget for BRCA (one repetition)
    res, _ = run_rep("brca", 0)
    bad = [m for m, r in res.items() if not all(np.isfinite(x["deviance"]) for x in r)]
    for m in ("Local", "Local, evidence prior", "Pooled", "GHB, eps=1", "GHB, eps=10", "GHB, eps=inf"):
        print(f"brca rep0 {m:24s} C {np.nanmean([np.nan if x['cindex'] is None else x['cindex'] for x in res[m]]):.3f} "
              f"dev {np.mean([x['deviance'] for x in res[m]]):.3f}")
    print("brca methods:", len(res), "non-finite deviance:", bad)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "score":
        score()
    elif cmd == "check":
        check()
    elif cmd == "isic-embed":
        isic_embed()
    elif cmd == "isic-embed-public":
        isic_embed(zpath=Path(sys.argv[2]), out=Path(sys.argv[3]))
    else:
        reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 20
        main_run(cmd, reps)
