"""Development of GHB v5 on the 12 TCGA development networks, with the folds and privacy noise of dev/v4_dev.py
(declared development; FLamby Heart, TCGA-BRCA, ISIC and the seven small networks are development data too).
    python3 dev/v5_dev.py [reps] [outdir]      then   python3 dev/v4_agg.py <outdir>
"""
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v4_dev as D  # noqa: E402

LOCKED = D.LOCKED
INT = dict(int_sd=2.5, free_int=2.5)  # the intercept keeps its own wide prior and is not borrowed
D.VARIANTS = {
    "v4 locked": LOCKED,
    "v5 intercept": dict(LOCKED, **INT),
    "v5 intercept + loo": dict(LOCKED, **INT, weights="pbma"),
    "v5 loo": dict(LOCKED, weights="pbma"),
}

if __name__ == "__main__":
    D.OUT = D.ghb.RESDIR / (sys.argv[2] if len(sys.argv) > 2 else "v5_dev_ckpt")
    D.OUT.mkdir(exist_ok=True)
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    todo = [(w, r) for r in range(reps) for w in ("dev", "fresh") if not (D.OUT / f"{w}_{r}.json").exists()]
    with Pool(2) as pool:
        for it in pool.imap_unordered(D.run, todo):
            print("done", it, flush=True)
