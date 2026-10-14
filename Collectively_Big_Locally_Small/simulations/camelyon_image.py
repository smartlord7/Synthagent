"""Pre-registered confirmation of GHB for images on Camelyon17 (dev/PREREG_image.md, "Confirmation on Camelyon17").

    python3 camelyon_image.py coords      WILDS patch grid of the 50 annotated slides (per-slide checkpoint)
    python3 camelyon_image.py pool        WILDS class balancing, then the pre-registered pool (600 + 600 per center)
    python3 camelyon_image.py public      2,000 unlabeled public patches from 10 CAMELYON17 test-set slides
    python3 camelyon_image.py extract     read the pool and public patches from the slides (HTTP range requests)
    python3 camelyon_image.py embed       frozen EfficientNet-B0 features of every extracted patch
    python3 camelyon_image.py run [--reps 10]   GHB, references and the literature grid (checkpoint per repetition)
    python3 camelyon_image.py score       summary (results/camelyon_image.json)

Patches: the WILDS recipe (p-lambda/wilds, dataset_preprocessing/camelyon17, commit 4726775) applied to the official
CAMELYON17 release (camelyon_slides.py); see the pre-registration for why Camelyon17-WILDS itself is not downloaded.
Model and methods as isic_image.py with K = 2 (one softmax head, logit adjustment by the model-implied class mix on the
hospital's training patches, restricted to the classes it has seen).
"""
import hashlib
import json
import os
import sys
import xml.etree.ElementTree as ET
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import camelyon_slides as CS

DATA = Path(os.environ.get("CAMELYON_DIR", "/home/user/camelyon_data"))
PATCH_LEVEL, MASK_LEVEL, CENTER_SIZE, MARGIN = 2, 4, 32, 5
TUMOR_T, NORMAL_T = 0.0, 0.2
N_POOL, N_TRAIN, N_PUBLIC_SLIDES, N_PUBLIC_PER = 600, 100, 10, 200  # per class / per class / slides / per slide
K = 2


# ------------------------------------------------------------------------------------------------- WILDS patch grid
def _s3_keys(prefix):
    keys, token = [], None
    while True:
        url = f"{CS.BASE}/?list-type=2&prefix={prefix}" + (f"&continuation-token={requests.utils.quote(token)}" if token else "")
        root = ET.fromstring(requests.get(url, timeout=60).content)
        ns = {"s": root.tag.split("}")[0][1:]}
        keys += [c.find("s:Key", ns).text for c in root.findall("s:Contents", ns)]
        t = root.find("s:NextContinuationToken", ns)
        if t is None:
            return keys
        token = t.text


def _md5s():
    out = {}
    for line in requests.get(f"{CS.BASE}/CAMELYON17/checksums.md5", timeout=60).text.splitlines():
        if line.strip():
            h, f = line.split(maxsplit=1)
            out[Path(f.strip().lstrip("*")).name] = h
    return out


def _read_xml(text, mask_level):
    """As WILDS _read_xml, with the official group names: Tumor = metastases, Exclusion = normal."""
    tumor, normal = [], []
    for a in ET.fromstring(text).iter("Annotation"):
        g = a.get("PartOfGroup")
        assert g in ("Tumor", "Exclusion", "None", "_0", "_1", "_2", "metastases", "normal"), g
        lst = tumor if g in ("Tumor", "metastases", "_0", "_1") else normal if g in ("Exclusion", "normal", "_2") else None
        if lst is None:
            continue
        for ri, region in enumerate(a.iter("Coordinates")):
            assert ri == 0
            lst.append([[round(float(c.get("X")) / 2 ** mask_level), round(float(c.get("Y")) / 2 ** mask_level)]
                        for c in region])
    return tumor, normal


def _tissue_mask(img):
    import cv2
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # WILDS applies BGR2HSV to RGB input; saturation is order-invariant
    _, m = cv2.threshold(hsv[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return m


def _cell_means(mask, dims_mask, dims_patch, w, h):
    """Mean of `mask` over the central region of every grid cell (i, j), with WILDS's rounded boundaries."""
    ws = CENTER_SIZE * dims_mask[0] / dims_patch[0]
    hs = CENTER_SIZE * dims_mask[1] / dims_patch[1]
    xb = np.array([round(ws * i) for i in range(w + 1)])
    yb = np.array([round(hs * j) for j in range(h + 1)])
    ii = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1))
    ii[1:, 1:] = mask.astype(float).cumsum(0).cumsum(1)
    xb, yb = np.minimum(xb, mask.shape[1]), np.minimum(yb, mask.shape[0])
    S = ii[yb[1:, None], xb[None, 1:]] - ii[yb[:-1, None], xb[None, 1:]] - ii[yb[1:, None], xb[None, :-1]] \
        + ii[yb[:-1, None], xb[None, :-1]]
    cnt = (yb[1:] - yb[:-1])[:, None] * (xb[1:] - xb[:-1])[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        return (S / cnt).T  # (w, h): [i, j]


def slide_grid(key, xml_text=None):
    """WILDS generate_file: (x_coord, y_coord, tumor) of every candidate cell, coordinates of the cell's central region
    at PATCH_LEVEL, cells in WILDS's order (i outer, j inner). Without xml_text: tissue cells only (public slides)."""
    import cv2
    s = CS.Slide(key)
    dm, dp = s.level_dimensions[MASK_LEVEL], s.level_dimensions[PATCH_LEVEL]
    img = s.whole_level(MASK_LEVEL)
    tissue = _tissue_mask(img)
    tumor = np.zeros(dm[::-1])
    if xml_text is not None:
        tc, nc = _read_xml(xml_text, MASK_LEVEL)
        for c in tc:
            cv2.drawContours(tumor, [np.array(c, np.int32)], -1, 255, -1)
        for c in nc:
            cv2.drawContours(tumor, [np.array(c, np.int32)], -1, 0, -1)
    normal = tissue.copy()
    normal[tumor > 127] = 0
    w, h = np.array(dp) // CENTER_SIZE
    tr = _cell_means(tumor, dm, dp, w, h) / 255
    nr = _cell_means(normal, dm, dp, w, h) / 255
    rows = []
    for i in range(MARGIN, w - MARGIN):
        t, n = tr[i, MARGIN:h - MARGIN], nr[i, MARGIN:h - MARGIN]
        js = np.arange(MARGIN, h - MARGIN)
        lab = np.where(t > TUMOR_T, 1, np.where(n > NORMAL_T, 0, -1))
        keep = lab >= 0
        rows.append(np.stack([np.full(keep.sum(), CENTER_SIZE * i), CENTER_SIZE * js[keep], lab[keep]], 1))
    return pd.DataFrame(np.vstack(rows), columns=["x_coord", "y_coord", "tumor"]), dict(dims=s.level_dimensions)


def _coords_one(xml_key):
    name = Path(xml_key).stem
    out = DATA / "coords" / f"{name}.csv"
    if out.exists():
        return name, "cached"
    text = requests.get(f"{CS.BASE}/{xml_key}", timeout=120).content
    md5 = hashlib.md5(text).hexdigest()
    assert md5 == _MD5[Path(xml_key).name], f"MD5 mismatch for {xml_key}"
    df, info = slide_grid(f"CAMELYON17/images/{name}.tif", text)
    _, patient, _, node = name.split("_")
    df["patient"], df["node"] = patient, node
    df.to_csv(out, index=False)
    json.dump(info, open(out.with_suffix(".json"), "w"))
    return name, f"{(df.tumor == 1).sum()} tumor, {(df.tumor == 0).sum()} normal"


_MD5 = {}


def coords():
    global _MD5
    (DATA / "coords").mkdir(parents=True, exist_ok=True)
    _MD5 = _md5s()
    xmls = sorted(k for k in _s3_keys("CAMELYON17/annotations/") if k.endswith(".xml"))
    assert len(xmls) == 50, len(xmls)
    with Pool(3) as pool:
        for name, msg in pool.imap(_coords_one, xmls):
            print(name, msg, flush=True)


# ------------------------------------------------------------------------------------- WILDS balancing and the pool
def pool():
    files = sorted((DATA / "coords").glob("patient_*.csv"))
    assert len(files) == 50, len(files)
    df = pd.concat([pd.read_csv(f, dtype={"patient": str}) for f in files], ignore_index=True)
    # generate_final_metadata.py
    pn = sorted(set(df[["patient", "node"]].itertuples(index=False, name=None)))
    smap = {p: i for i, p in enumerate(pn)}
    df["slide"] = [smap[p] for p in df[["patient", "node"]].itertuples(index=False, name=None)]
    df["center"] = df["patient"].astype(int) // 20
    assert all(c == s // 10 for c, s in set(df[["center", "slide"]].itertuples(index=False, name=None)))
    np.random.seed(0)
    tum = (df["tumor"] == 1).values
    keep = []
    for slide in sorted(set(df["slide"])):
        sm = (df["slide"] == slide).values
        nt, nn = int((sm & tum).sum()), int((sm & ~tum).sum())
        keep += list(np.random.choice(list(df.index[sm & tum]), size=min(nt, nn), replace=False))
    tkeep = np.zeros(len(df), bool)
    tkeep[keep] = True
    for c in range(5):
        cm = (df["center"] == c).values
        nt, nn = int((cm & tkeep).sum()), int((cm & ~tum).sum())
        keep += list(np.random.choice(list(df.index[cm & ~tum]), size=min(nt, nn), replace=False))
    meta = df.loc[keep].reset_index(drop=True)
    stats = {int(c): {"tumor": int(((meta.center == c) & (meta.tumor == 1)).sum()),
                      "normal": int(((meta.center == c) & (meta.tumor == 0)).sum())} for c in range(5)}
    print("WILDS-recipe metadata per center:", stats)
    # the pre-registered pool
    rng = np.random.default_rng(2028)
    rows = []
    for c in range(5):
        for lab in (1, 0):
            idx = np.flatnonzero(((meta.center == c) & (meta.tumor == lab)).values)
            take = rng.choice(idx, size=min(N_POOL, len(idx)), replace=False)
            if len(idx) < N_POOL:
                print(f"center {c} class {lab}: only {len(idx)} patches")
            rows.append(meta.loc[take])
    P = pd.concat(rows, ignore_index=True)
    P.to_csv(DATA / "pool.csv", index=False)
    json.dump({"wilds_recipe_per_center": stats, "all_cells": int(len(df)),
               "pool": {int(c): int((P.center == c).sum()) for c in range(5)}},
              open(DATA / "pool_stats.json", "w"), indent=1)
    print(P.groupby(["center", "tumor"]).size())


# ------------------------------------------------------------------------------------------------- public patches
def public():
    keys = sorted(k for k in _s3_keys("CAMELYON17/images/patient_") if k.endswith(".tif")
                  and 100 <= int(Path(k).stem.split("_")[1]) <= 199)
    rng = np.random.default_rng(2029)
    patients = rng.choice(np.arange(100, 200), size=N_PUBLIC_SLIDES, replace=False)
    rows = []
    for p in patients:
        node = int(rng.integers(5))
        key = f"CAMELYON17/images/patient_{p:03d}_node_{node}.tif"
        assert key in keys, key
        df, _ = slide_grid(key)
        df = df[df.tumor == 0]  # tissue cells (no annotation: "normal" here means tissue, label unknown)
        take = rng.choice(len(df), size=min(N_PUBLIC_PER, len(df)), replace=False)
        sub = df.iloc[take][["x_coord", "y_coord"]].copy()
        sub["patient"], sub["node"] = f"{p:03d}", node
        rows.append(sub)
        print(key, len(df), "tissue cells", flush=True)
    pd.concat(rows, ignore_index=True).to_csv(DATA / "public.csv", index=False)


# ------------------------------------------------------------------------------------------------------- extraction
def _extract_slide(item):
    key, xy = item
    s = CS.Slide(key)
    return np.stack([s.read_level(PATCH_LEVEL, int(x) - CENTER_SIZE, int(y) - CENTER_SIZE, 3 * CENTER_SIZE,
                                  3 * CENTER_SIZE) for x, y in xy])


def extract():
    for name in ("pool", "public"):
        out = DATA / f"{name}_patches.npy"
        if out.exists():
            continue
        df = pd.read_csv(DATA / f"{name}.csv", dtype={"patient": str})
        df["key"] = [f"CAMELYON17/images/patient_{int(p):03d}_node_{int(n)}.tif" for p, n in zip(df.patient, df.node)]
        imgs = np.zeros((len(df), 96, 96, 3), np.uint8)
        groups = list(df.groupby("key").groups.items())
        with Pool(4) as pool:
            res = pool.map(_extract_slide, [(k, df.loc[ix, ["x_coord", "y_coord"]].values) for k, ix in groups])
        for (k, ix), a in zip(groups, res):
            imgs[np.asarray(ix)] = a
        np.save(out, imgs)
        print(name, imgs.shape, flush=True)


# ------------------------------------------------------------------------------------------------------- embedding
WURL = "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth"


def embed(batch=64):
    import torch
    from efficientnet_pytorch import EfficientNet
    from PIL import Image
    wpath = DATA / "efficientnet-b0-355c32eb.pth"
    if not wpath.exists():
        wpath.write_bytes(requests.get(WURL, timeout=300).content)
    assert hashlib.sha256(wpath.read_bytes()).hexdigest().startswith("355c32eb"), "encoder weights checksum"
    m = EfficientNet.from_name("efficientnet-b0")
    m.load_state_dict(torch.load(wpath, map_location="cpu"))
    m.eval()
    mean, sd = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)

    def prep(a):
        a = np.asarray(Image.fromarray(a).resize((224, 224), resample=Image.BILINEAR), np.float32) / 255.0
        return np.transpose((a - mean) / sd, (2, 0, 1))

    for name in ("pool", "public"):
        out = DATA / f"{name}_emb.npy"
        if out.exists():
            continue
        imgs = np.load(DATA / f"{name}_patches.npy")
        embs = []
        with torch.no_grad():
            for i in range(0, len(imgs), batch):
                x = torch.from_numpy(np.stack([prep(a) for a in imgs[i:i + batch]]))
                embs.append(torch.flatten(torch.nn.functional.adaptive_avg_pool2d(m.extract_features(x), 1), 1).numpy())
                print(name, i, flush=True)
        np.save(out, np.vstack(embs))


# -------------------------------------------------------------------------------------------------------- the runs
def _ghb():
    import ghb
    return ghb


def load(rep, d=32):
    """Train/test per hospital for repetition `rep` (pre-registered stratified split) and the public rows."""
    from sklearn.decomposition import PCA
    import isic_image as II
    P = pd.read_csv(DATA / "pool.csv", dtype={"patient": str})
    Z, pub = np.load(DATA / "pool_emb.npy"), np.load(DATA / "public_emb.npy")
    pca = PCA(d, random_state=0).fit(pub)
    Z, Zp = pca.transform(Z), pca.transform(pub)
    rng = np.random.default_rng([2028, rep])
    train, test = [], []
    for c in range(5):
        tr = []
        for lab in (0, 1):
            idx = np.flatnonzero(((P.center == c) & (P.tumor == lab)).values)
            tr += list(rng.choice(idx, size=N_TRAIN, replace=False))
        tr = np.array(sorted(tr))
        te = np.setdiff1d(np.flatnonzero((P.center == c).values), tr)
        mu, sd = Z[tr].mean(0), Z[tr].std(0, ddof=1) + 1e-9
        train.append((II._prep(Z[tr], mu, sd), P.tumor.values[tr].astype(int)))
        test.append((II._prep(Z[te], mu, sd), P.tumor.values[te].astype(int)))
    return train, test, II._prep(Zp, Zp.mean(0), Zp.std(0, ddof=1) + 1e-9)


def _ckpt():
    return _ghb().RESDIR / "camelyon_image_ckpt"


def log_loss(th, X, y, Xtr, support):
    from scipy.special import log_softmax, softmax
    S = X @ th.reshape(X.shape[1], -1)
    pi = softmax(Xtr @ th.reshape(X.shape[1], -1), 1).mean(0)
    S = np.where(support[None, :], S - np.log(np.maximum(pi, 1e-12)), -np.inf)
    lp = log_softmax(S, 1)[np.arange(len(y)), y]
    return float(np.mean(-np.maximum(lp, np.log(1e-15))))


def run_rep(rep):
    import flamby_bench as FB
    import ghb_image as GI
    import isic_image as II
    rng = np.random.default_rng([2027, rep])
    train, test, Xpub = load(rep)
    P = Xpub.shape[1]
    G = np.sqrt(P)
    D = P * K
    data = [(X, GI.onehot(y, K)) for X, y in train]
    Xp, Yp = np.vstack([X for X, _ in data]), np.vstack([Y for _, Y in data])
    est = {"Local": np.array([GI.fit(X, Y, np.zeros(D), 1 / 2.5 ** 2) for X, Y in data]),
           "Local, evidence prior": np.array([GI.local_eb(X, Y) for X, Y in data]),
           "Pooled": np.repeat(GI.fit(Xp, Yp, np.zeros(D), 1 / 2.5 ** 2)[None], len(data), 0)}
    for e in II.EPS:
        est[f"GHB, eps={FB.etag(e)}"] = GI.estimate(train, Xpub, rng, e, FB.DELTA, G,
                                                   {**II.VARIANTS["GHB R1=1"], "K": K})
        for k, v in GI.baselines(rng, train, GI.GL.rho_total(e, FB.DELTA), G, K).items():
            est[f"{k}, eps={FB.etag(e)}"] = v
    res = {}
    for m, th in est.items():
        row = []
        for k, ((Xt, yt), (X, y)) in enumerate(zip(test, train)):
            sup = np.bincount(y, minlength=K) > 0
            row.append({"bacc": GI.balanced_accuracy(GI.predict(th[k], Xt, X, "model", sup), yt),
                        "logloss": log_loss(th[k], Xt, yt, X, sup)})
        res[m] = row
    return res


def task(rep):
    res = run_rep(rep)
    json.dump(res, open(_ckpt() / f"rep_{rep}.json", "w"))
    print("checkpointed", rep, flush=True)


def run(reps):
    _ckpt().mkdir(parents=True, exist_ok=True)
    todo = [r for r in range(reps) if not (_ckpt() / f"rep_{r}.json").exists()]
    with Pool(min(4, len(todo) or 1)) as pool:
        list(pool.imap_unordered(task, todo))


def score():
    import flamby_bench as FB
    runs = [json.load(open(p)) for p in sorted(_ckpt().glob("rep_*.json"))]
    met = ("bacc", "logloss")
    mean = {m: {a: float(np.mean([np.mean([c[a] for c in r[m]]) for r in runs])) for a in met} for m in runs[0]}
    per_rep = {m: {a: [float(np.mean([c[a] for c in r[m]])) for r in runs] for a in met} for m in runs[0]}
    rows = {m: v for m, v in mean.items() if "|" not in m}
    cfgs = {}
    for lab, code in FB.FAMILIES.items():
        for e in ("1", "10", "inf"):
            ks = [m for m in mean if m.startswith(code + "|") and m.endswith(f", eps={e}")]
            best = max(ks, key=lambda m: mean[m]["bacc"])
            rows[f"{lab} (oracle), eps={e}"] = mean[best]
            rk = FB.REALISTIC.get(f"{lab}, eps={e}", FB.REALISTIC.get(f"{lab}, eps=10"))
            rows[f"{lab} (realistic), eps={e}"] = mean[f"{rk}, eps={e}"]
            cfgs[f"{lab}, eps={e}"] = {"oracle": best, "realistic": f"{rk}, eps={e}"}
            per_rep[f"{lab} (realistic), eps={e}"] = per_rep[f"{rk}, eps={e}"]
            per_rep[f"{lab} (oracle), eps={e}"] = per_rep[best]
    per_rep = {m: v for m, v in per_rep.items() if "|" not in m}
    pstats = json.load(open(DATA / "pool_stats.json")) if (DATA / "pool_stats.json").exists() else None
    out = dict(reps=len(runs), rows=rows, per_rep=per_rep, configs=cfgs, pool=pstats)
    json.dump(out, open(_ghb().RESDIR / "camelyon_image.json", "w"), indent=1)
    print(f"{len(runs)} repetitions; balanced accuracy, log-loss")
    for m, r in rows.items():
        print(f"{m:48s}{r['bacc']:8.3f}{r['logloss']:8.3f}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "score"
    arg = lambda k, v: type(v)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else v
    {"coords": coords, "pool": pool, "public": public, "extract": extract, "embed": embed,
     "run": lambda: run(arg("--reps", 10)), "score": score}[cmd]()
