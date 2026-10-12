"""GHB v4: the network mean learned from the hospitals' real data instead of a synthetic public population.

Diagnosis (real_v3.py, fresh networks): GHB-EB inverts each hospital's released score through a synthetic public
population; when the real case mix differs from it, the inversion is biased, the network mean is wrong and, at weak
privacy where the prior is tight, borrowing hurts. At strong privacy the per-hospital releases are so noisy that the
network mean is ignored and GHB reduces to a local model. Both are fixed by giving GHB two pieces of context that cost
little privacy under secure aggregation (the trust model of GHB-SA and DP-FedAvg-SA):

  Stage 1 (budget f rho, secure sums)  R1 rounds of differentially private Newton on the pooled model: every round the
       network releases the sum of the hospitals' gradients (sensitivity 2G) and of their Fisher information matrices
       (sensitivity G^2 / 2 in Frobenius norm), each with Gaussian noise on the sum only. This gives the network mean w
       and the pooled per-patient information F, from the real case mix, with noise of order 1/N rather than 1/n_j.
  Stage 2 (budget (1 - f) rho, per hospital)  each hospital releases its mean score at w (sensitivity 2G / n_j); with F
       it is linearized into theta~_j = w + (F + I / (n_j s0^2))^{-1} g_j, whose spread beyond the privacy and sampling
       noise gives the posterior of the between-hospital spread tau (per coefficient, half-Cauchy prior, Q quantiles).
  Prior of hospital j  N(w_{-j}, diag(tau_q^2) + V_{-j}) for each quantile q, combined with the default N(0, s^2 I) for
       s in {0.5, 1, 2.5}; w_{-j} = w - H_{-j}^{-1} s_j(w) removes hospital j's own data from the network mean (computed
       locally from its exact score s_j and information H_j, no privacy cost), V_{-j} = H_{-j}^{-1}. Optional: the
       no-borrowing components (safety) and the institution effects of GHB v3 (cross).
The final model averages the MAP fits under every component with the Laplace evidence of the hospital's own data.
"""
import numpy as np
from scipy.special import expit

import ghb_logit as GL
import ghb_v3 as V3

S0 = GL.PRIOR_SD
SDS = V3.SDS
Q = V3.Q


def _psd(H, floor):
    H = (H + H.T) / 2
    ev, U = np.linalg.eigh(H)
    return (U * np.maximum(ev, floor)) @ U.T


def stage1(rng, data, rho1, G, R1=2, hshare=None):
    """DP quasi-Newton with secure sums: the pooled information at 0 (sensitivity G^2 / 2, Frobenius) is released
    once, then R1 gradient sums (sensitivity 2G), each release with an equal share of rho1. Returns w, the noisy
    information sum H0, and the covariance of w due to the privacy noise."""
    P = data[0][0].shape[1]
    N = sum(len(y) for _, y in data)
    hs = 1.0 / (R1 + 1) if hshare is None else hshare
    rg, rh = rho1 * (1 - hs) / R1, rho1 * hs
    sg = 0.0 if np.isinf(rg) else 2 * G / np.sqrt(2 * rg)
    sh = 0.0 if np.isinf(rh) else (G ** 2 / 2) / np.sqrt(2 * rh)
    H = sum(X.T @ X / 4 for X, _ in data)
    E = np.triu(rng.standard_normal((P, P))) * sh
    H0 = _psd(H + E + np.triu(E, 1).T, 1e-2 * N)
    A = np.linalg.inv(H0 + np.eye(P) / S0 ** 2)
    w = np.zeros(P)
    for _ in range(R1):
        g = sum(X.T @ (y - expit(X @ w)) for X, y in data) + sg * rng.standard_normal(P)
        w = np.clip(w + A @ (g - w / S0 ** 2), -8, 8)
    return w, H0, sg ** 2 * A @ A * R1


def network_stats(rng, data, eps, delta, G, f, R1=2, hshare=None):
    P = data[0][0].shape[1]
    n = np.array([len(y) for _, y in data], float)
    N = n.sum()
    rho = GL.rho_total(eps, delta)
    w, H, Vdp = stage1(rng, data, f * rho, G, R1, hshare)
    F = H / N
    st = dict(w=w, H=H, n=n, Vdp=Vdp)
    if f < 1:
        rho2 = (1 - f) * rho
        sd = np.zeros(len(n)) if np.isinf(rho2) else 2 * G / n / np.sqrt(2 * rho2)
        g = np.array([X.T @ (y - expit(X @ w)) / len(y) for X, y in data]) + sd[:, None] * rng.standard_normal((len(n), P))
        th, R = np.empty((len(n), P)), np.empty((len(n), P, P))
        for j in range(len(n)):
            A = np.linalg.inv(F + np.eye(P) / (n[j] * S0 ** 2))
            th[j] = w + A @ g[j]
            R[j] = A @ (F / n[j] + sd[j] ** 2 * np.eye(P)) @ A.T
        r = np.array([np.diag(Rj) for Rj in R])
        st.update(th=th, R=R, r=r, tq=np.array([V3.quantiles(V3.tau2_post(th[:, d], r[:, d])) for d in range(P)]).T)
    else:  # no stage 2: tau from its half-Cauchy prior alone
        p = 1.0 / (1 + (GL.GRID / GL.SCALE) ** 2) * np.gradient(GL.GRID)
        st["tq"] = np.repeat(V3.quantiles(p / p.sum())[:, None], P, 1)
    return st


def estimate(nets, rng, eps, delta, G, inst=None, cfg=None):
    """nets: dict name -> list of (X, y); cfg keys f (stage-1 share), safety, cross. Returns dict name -> (K, P)."""
    cfg = {"f": 0.5, "safety": True, "cross": False, "pan": False, "R1": 2, "hshare": None, "loo": True, "vdp": True,
           **(cfg or {})}  # loo, vdp: ablation switches only (leave-one-out mean; propagated privacy covariance)
    stats = {c: network_stats(rng, d, eps, delta, G, cfg["f"], cfg["R1"], cfg["hshare"]) for c, d in nets.items()}
    P = next(iter(nets.values()))[0][0].shape[1]
    names = list(nets)
    if cfg["pan"] and len(names) > 2:  # empirical Bayes across networks: w_c ~ N(omega, diag(psi2)), per coefficient
        W = np.array([stats[c]["w"] for c in names])
        Vt = np.array([np.diag(stats[c]["Vdp"] + np.linalg.inv(stats[c]["H"] + np.eye(P) / S0 ** 2)) for c in names])
        for d in range(P):
            psi2 = V3.quantiles(V3.tau2_post(W[:, d], Vt[:, d]))[Q // 2]
            for i, c in enumerate(names):
                o = np.delete(np.arange(len(names)), i)  # the other networks' mean
                wo = 1.0 / (psi2 + Vt[o, d])
                om, ov = (wo * W[o, d]).sum() / wo.sum(), 1.0 / wo.sum()
                prec = 1.0 / Vt[i, d] + 1.0 / (psi2 + ov)
                stats[c].setdefault("wpan", stats[c]["w"].copy())[d] = (W[i, d] / Vt[i, d] + om / (psi2 + ov)) / prec
                stats[c].setdefault("vpan", np.zeros(P))[d] = 1.0 / prec
        for c in names:  # the shrunken network mean replaces w; its total uncertainty replaces the privacy part
            st = stats[c]
            Vs = np.diag(np.linalg.inv(st["H"] + np.eye(P) / S0 ** 2))
            st["w"], st["Vdp"] = st["wpan"], np.diag(np.maximum(st["vpan"] - Vs, 0.0))
    if cfg.get("anchor") is not None:  # a published model N(mu0, anchor_sd^2) as further context, dropped on conflict
        mu0, asd = cfg["anchor"]
        for c in names:
            st = stats[c]
            Vs = np.diag(np.linalg.inv(st["H"] + np.eye(P) / S0 ** 2))
            Vt = np.diag(st["Vdp"]) + Vs
            if np.abs((mu0 - st["w"]) / np.sqrt(asd ** 2 + Vt)).max() <= GL.Z_CONFLICT:
                prec = 1.0 / Vt + 1.0 / asd ** 2
                st["w"] = (st["w"] / Vt + mu0 / asd ** 2) / prec
                st["Vdp"] = np.diag(np.maximum(1.0 / prec - Vs, 0.0))
    use_cross = cfg["cross"] and cfg["f"] < 1 and inst is not None and len(names) > 1
    if use_cross:  # institution effects from the same institution's deviations in its other cancers
        dev = {}
        for c in names:
            st = stats[c]
            for k, h in enumerate(inst[c]):
                dev.setdefault(h, []).append((c, st["th"][k] - st["w"], st["r"][k]))
        prods = [l[a][1] * l[b][1] for l in dev.values() for a in range(len(l)) for b in range(a + 1, len(l))]
        sb2 = np.maximum(np.mean(prods, axis=0), 0.0) if prods else np.zeros(P)
    out = {}
    for c in names:
        st, data = stats[c], nets[c]
        t2med = st["tq"][Q // 2]
        est = np.empty((len(data), P))
        for j, (X, y) in enumerate(data):
            sj = X.T @ (y - expit(X @ st["w"]))
            Hm = _psd(st["H"] - X.T @ X / 4, 1e-2 * max(st["n"].sum() - len(y), 1)) + np.eye(P) / S0 ** 2
            V = np.linalg.inv(Hm)
            mu = st["w"] - V @ sj if cfg["loo"] else st["w"]
            V = V + st["Vdp"] if cfg["vdp"] else V
            shift, extra = np.zeros(P), np.zeros(P)
            if use_cross:
                others = [o for o in dev.get(inst[c][j], []) if o[0] != c]
                tot = np.minimum(sb2, t2med)
                if others:
                    dm = np.mean([o[1] for o in others], axis=0)
                    dv = np.mean([o[2] for o in others], axis=0) / len(others)
                    k_ = tot / np.maximum(tot + dv, 1e-12)
                    shift, extra = k_ * dm, tot * (1 - k_) - tot
                else:
                    extra = np.zeros(P)
            comps = []
            for q in range(Q):
                t2 = np.maximum(st["tq"][q] + extra, 0.0)
                Cn = np.diag(t2) + V
                Cni = np.linalg.inv(Cn)
                for s in SDS:
                    C = np.linalg.inv(Cni + np.eye(P) / s ** 2)
                    comps.append((np.log(1.0 / (Q * len(SDS))), C @ Cni @ (mu + shift), C))
            if cfg["safety"]:
                for s in SDS:
                    comps.append((np.log(1.0 / (3 * len(SDS))), np.zeros(P), np.eye(P) * s ** 2))
            est[j] = V3.combine(X, y, comps)
        out[c] = est
    return out
