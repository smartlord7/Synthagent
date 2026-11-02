"""Re-identification risk of GHB's releases, and who gains from borrowing, on the TCGA networks.

1. Bound. GHB's two Gaussian releases, with half of a zero-concentrated budget rho each, form a Gaussian mechanism
   with mu = sqrt(2 rho) in the sense of Gaussian differential privacy (Dong, Roth and Su, 2022): no test can tell
   whether a given patient's record was used (against any replacement) with true-positive rate above
   Phi(Phi^-1(alpha) + mu) at false-positive rate alpha, and its advantage TPR - FPR is at most 2 Phi(mu / 2) - 1.
2. Realized risk. The bound assumes a worst-case patient (score and information at their sensitivity limits). For each
   real TCGA patient, the same computation with the patient's actual contribution, against replacement by the most
   different patient of the same network, gives that patient's attainable advantage.
3. Empirical attack. A loss-threshold membership-inference attack (members: training patients; non-members: held-out
   patients of the same hospital, five-fold cross-validation) against (a) each hospital's local model released without
   privacy, as federated or meta-analytic sharing would, (b) the non-private pooled network estimate, and (c) GHB's
   privatized network estimate (what the releases reveal). AUC 0.5 means the attacker learns nothing.
4. Who gains. GHB's held-out log-loss relative to local estimation by hospital size, on the five held-out networks,
   fitted together and alone (stress_tcga.py checkpoints).
    python3 reid_risk.py        (results/reid_risk.json)
"""
import glob
import json

import numpy as np
from scipy.special import expit
from scipy.stats import norm

import ghb
import ghb_logit as GL
import ghb_v4 as V4
import real_v3 as RV
from exp7_logistic import fit
from real_heart import folds

EPS = (1.0, 10.0)
G = RV.G


def rho_of(eps, delta=RV.DELTA):
    return GL.rho_total(eps, delta)


def bound(eps):
    mu = np.sqrt(2 * rho_of(eps))
    return dict(rho=float(rho_of(eps)), mu=float(mu), advantage=float(2 * norm.cdf(mu / 2) - 1),
                tpr_at_fpr_1pct=float(norm.cdf(norm.ppf(0.01) + mu)), tpr_at_fpr_10pct=float(norm.cdf(norm.ppf(0.10) + mu)))


def noise(eps):
    rho = rho_of(eps)
    return 2 * G / np.sqrt(rho), np.sqrt(2) * G ** 2 / (4 * np.sqrt(rho))  # s_S, s_J (ghb_v4 tight, half rho each)


def realized(eps):
    sS, sJ = noise(eps)
    adv = []
    for which in ("dev", "fresh", "confirm"):
        for p in RV.SETS[which]:
            data, _ = RV.load(p)
            X = np.vstack([x for x, _ in data]); y = np.concatenate([t for _, t in data])
            psi = X * (y - 0.5)[:, None]
            iu = np.triu_indices(X.shape[1])
            phi = np.array([np.outer(x, x)[iu] / 4 for x in X])
            for i in range(len(y)):
                dS = np.linalg.norm(psi - psi[i], axis=1) / sS
                dJ = np.linalg.norm(phi - phi[i], axis=1) / sJ
                mu_i = np.sqrt(dS ** 2 + dJ ** 2).max()  # worst replacement within the network
                adv.append(2 * norm.cdf(mu_i / 2) - 1)
    a = np.array(adv)
    return dict(patients=int(len(a)), median=float(np.median(a)), p95=float(np.quantile(a, 0.95)), max=float(a.max()))


def auc(member, non):
    """AUC of the score -loss for telling members (training) from non-members (held-out); 0.5 = no information."""
    s = np.concatenate([-member, -non]); lab = np.r_[np.ones(len(member)), np.zeros(len(non))]
    order = s.argsort()
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    n1, n0 = lab.sum(), (1 - lab).sum()
    return float((ranks[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def attack(reps=5):
    loss = lambda X, y, th: np.logaddexp(0, X @ th) - y * (X @ th)
    out = {"local": {"small": [], "medium": [], "large": []}, "pooled": [], **{f"ghb eps={e:g}": [] for e in EPS}}
    pool = {b: ([], []) for b in out["local"]}  # members' and non-members' losses, pooled over hospitals of a size band
    band = lambda n: "small" if n <= 15 else "medium" if n <= 40 else "large"
    for rep in range(reps):
        rng = np.random.default_rng([90, rep])
        for which in ("dev", "fresh", "confirm"):
            nets = {p: RV.load(p)[0] for p in RV.SETS[which]}
            for p, data in nets.items():
                fs = [folds(rng, y, RV.FOLDS) for _, y in data]
                f = int(rng.integers(RV.FOLDS))
                tr = [(X[fk != f], y[fk != f]) for (X, y), fk in zip(data, fs)]
                te = [(X[fk == f], y[fk == f]) for (X, y), fk in zip(data, fs)]
                for (Xa, ya), (Xb, yb) in zip(tr, te):  # (a) each hospital's own model, released without privacy
                    if len(yb) and len(ya):
                        th = fit(Xa, ya, np.zeros(RV.P), RV.PRIOR)
                        b = band(len(ya) + len(yb))
                        out["local"][b].append(auc(loss(Xa, ya, th), loss(Xb, yb, th)))
                        pool[b][0].append(loss(Xa, ya, th)); pool[b][1].append(loss(Xb, yb, th))
                Xa = np.vstack([X for X, _ in tr]); ya = np.concatenate([y for _, y in tr])
                Xb = np.vstack([X for X, _ in te]); yb = np.concatenate([y for _, y in te])
                th = fit(Xa, ya, np.zeros(RV.P), RV.PRIOR)  # (b) the pooled network estimate, without privacy
                out["pooled"].append(auc(loss(Xa, ya, th), loss(Xb, yb, th)))
                for e in EPS:  # (c) GHB's privatized network estimate, one Newton step from the two releases
                    st = V4.network_stats(rng, tr, e, RV.DELTA, G, 1.0, 1, None, GL.SCALE, False, True, "auto")
                    out[f"ghb eps={e:g}"].append(auc(loss(Xa, ya, st["w"]), loss(Xb, yb, st["w"])))
    summ = {"local": {b: dict(hospitals=len(v), pooled_auc=auc(np.concatenate(pool[b][0]), np.concatenate(pool[b][1])))
                      for b, v in out["local"].items()}}
    for k in [k for k in out if k != "local"]:
        summ[k] = dict(networks=len(out[k]), mean_auc=float(np.mean(out[k])), max_auc=float(np.max(out[k])))
    return summ


def who_gains():
    nets = RV.SETS["confirm"]
    sizes = {p: [len(y) for _, y in RV.load(p)[0]] for p in nets}
    out = {}
    for cond in ("standard", "alone"):
        acc = {}
        for fn in glob.glob(str(ghb.RESDIR / "stress_tcga_ckpt" / f"{cond}_confirm_*.json")):
            for k, (ll, c) in json.load(open(fn)).items():
                a = acc.setdefault(k, [np.zeros(len(ll)), np.zeros(len(ll))]); a[0] += ll; a[1] += c
        rows = []
        for p in nets:
            loc = acc[f"{p}||Local"]
            for k, n in enumerate(sizes[p]):
                L = loc[0][k] / loc[1][k]
                r = {"n": n, "eb": acc[f"{p}||Local, evidence prior"][0][k] / acc[f"{p}||Local, evidence prior"][1][k] / L}
                for e in ("1", "10"):
                    g = acc[f"{p}||GHB, eps={e}"]
                    r[e] = g[0][k] / g[1][k] / L
                rows.append(r)
        res = {}
        for name, lo, hi in (("5-15", 5, 15), ("16-40", 16, 40), ("over 40", 41, 10 ** 6)):
            sel = [r for r in rows if lo <= r["n"] <= hi]
            res[name] = dict(hospitals=len(sel), patients=int(sum(r["n"] for r in sel)),
                             **{f"ghb_{e}": float(np.mean([r[e] for r in sel])) for e in ("1", "10")},
                             **{f"better_{e}": int(sum(r[e] < 1 for r in sel)) for e in ("1", "10")},
                             eb=float(np.mean([r["eb"] for r in sel])))
        out[cond] = res
    out["networks"] = {p: dict(hospitals=len(s), patients=int(sum(s)), smallest=int(min(s)), largest=int(max(s))) for p, s in sizes.items()}
    return out


if __name__ == "__main__":
    res = {"bound": {f"{e:g}": bound(e) for e in EPS}, "realized": {f"{e:g}": realized(e) for e in EPS},
           "attack": attack(), "who_gains": who_gains()}
    json.dump(res, open(ghb.RESDIR / "reid_risk.json", "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("bound", "realized", "attack")}, indent=1))
