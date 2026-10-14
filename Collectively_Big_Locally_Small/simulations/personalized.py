"""Personalized federated learning baselines under differential privacy, for logistic heads.

Every method communicates one vector per hospital per round; that vector is privatized with the Gaussian mechanism
calibrated to its L2 sensitivity to one patient, and the rounds compose under zero-concentrated DP (budget rho/R per
round), exactly as DP-FedAvg in literature.py. Quantities that never leave a hospital (personal models, local
intercepts, mixing weights) cost no privacy. Full-batch gradients; G = sqrt(5) bounds a patient's score term, and the
logistic loss is L-smooth with L <= max ||x||^2 / 4 = 1.25 per patient.

  Ditto        (Li et al. 2021) global model by DP-FedAvg; each personal model v_k minimizes f_k(v) + lam/2 ||v - w||^2,
               updated locally every round from the current global model.
  pFedMe       (T. Dinh et al. 2020) personal model theta_k = argmin f_k(theta) + lam/2 ||theta - w_k||^2 (exact, local);
               the shared local copy moves w_k <- w_k - beta (w_k - theta_k), beta = min(eta lam, 1), for S steps; the
               network averages the w_k. theta_k moves by at most 2G / (n lam) (strong convexity lam), so the shared
               change has sensitivity S beta 2G / (n lam).
  APFL         (Deng et al. 2020) personalized model alpha v_k + (1 - alpha) w; w by DP-FedAvg, v_k updated locally on the
               personalized loss.
  Per-FedAvg   (Fallah et al. 2020, first-order) shared gradient at the adapted point w - a grad f_k(w), sensitivity
               (1 + a L) 2G / n; each hospital's model is one local step from the final w.
  FedRep/FedPer (Collins et al. 2021; Arivazhagan et al. 2019) the coefficients are shared and trained by DP-FedAvg, the
               intercept is local and refitted exactly each round; optional local fine-tuning at the end.
  FedProx      (Li et al. 2020) E local proximal gradient steps per round (mu/2 ||theta - w||^2), the change shared;
               sensitivity eta E 2G / n (the gradient map is non-expansive for eta <= 2 / (L + mu)); fine-tuning after.
With full-batch gradients and one local step, FedProx and SCAFFOLD both reduce to DP-FedAvg, which is reported
separately; FedProx with several local steps stands for the drift-correcting family.
"""
import numpy as np
from scipy.special import expit

import literature as LIT
from exp7_logistic import P, fit

G = np.sqrt(5.0)
L_SMOOTH = 1.25


def grad(X, y, th):
    return X.T @ (expit(X @ th) - y) / len(y)


def _sd(n, rho, R, factor=1.0):
    n = np.asarray(n, float)
    return np.zeros(len(n)) if np.isinf(rho) else factor * 2 * G / n / np.sqrt(2 * rho / R)


def _avg(rng, vecs, n, sd):
    w = np.asarray(n, float) / np.sum(n)
    noisy = np.array(vecs) + sd[:, None] * rng.standard_normal((len(vecs), P))
    return w @ noisy


def ditto(rng, data, n, rho, R, eta, lam, steps=5, eta_p=0.5):
    sd = _sd(n, rho, R)
    w = np.zeros(P)
    v = np.zeros((len(data), P))
    for _ in range(R):
        g = _avg(rng, [grad(X, y, w) for X, y in data], n, sd)
        for k, (X, y) in enumerate(data):
            for _ in range(steps):
                v[k] = v[k] - eta_p * (grad(X, y, v[k]) + lam * (v[k] - w))
        w = w - eta * g
    return np.array([fit(X, y, w, np.full(P, lam)) for X, y in data])  # final personal models, exact, local


def pfedme(rng, data, n, rho, R, eta, lam, S=5):
    beta = min(eta * lam, 1.0)                    # step of the local copy toward the personal model, kept stable
    # fit() minimizes the SUM of the losses + lam/2 |theta - w_k|^2, so one patient moves theta by at most 2G / lam
    # (not 2G / (n lam): corrected 2026-10-14, the earlier noise was n times too small); the prox is nonexpansive,
    # so each step adds at most beta 2G / lam to the change of w_k, and S steps S beta 2G / lam
    sd = _sd(n, rho, R, factor=S * beta * np.asarray(n, float) / lam)
    w = np.zeros(P)
    for _ in range(R):
        deltas = []
        for X, y in data:
            wk = w.copy()
            for _ in range(S):
                th = fit(X, y, wk, np.full(P, lam))
                wk = wk - beta * (wk - th)
            deltas.append(wk - w)
        w = w + _avg(rng, deltas, n, sd)
    return np.array([fit(X, y, w, np.full(P, lam)) for X, y in data])


def apfl(rng, data, n, rho, R, eta, alpha, steps=5, eta_p=0.5):
    sd = _sd(n, rho, R)
    w = np.zeros(P)
    v = np.zeros((len(data), P))
    for _ in range(R):
        g = _avg(rng, [grad(X, y, w) for X, y in data], n, sd)
        for k, (X, y) in enumerate(data):
            for _ in range(steps):
                mix = alpha * v[k] + (1 - alpha) * w
                v[k] = v[k] - eta_p * alpha * grad(X, y, mix)
        w = w - eta * g
    return alpha * v + (1 - alpha) * w[None, :]


def per_fedavg(rng, data, n, rho, R, eta, a):
    sd = _sd(n, rho, R, factor=1 + a * L_SMOOTH)
    w = np.zeros(P)
    for _ in range(R):
        g = _avg(rng, [grad(X, y, w - a * grad(X, y, w)) for X, y in data], n, sd)
        w = w - eta * g
    return np.array([w - a * grad(X, y, w) for X, y in data])


def fedrep(rng, data, n, rho, R, eta, ft=(0, 5)):
    """Shared coefficients (all but the intercept) by DP-FedAvg; intercepts local and exact."""
    sd = _sd(n, rho, R)
    w = np.zeros(P)
    b = np.zeros(len(data))
    for _ in range(R):
        for k, (X, y) in enumerate(data):  # local intercept given the shared coefficients (Newton, exact)
            for _ in range(10):
                p = expit(b[k] + X[:, 1:] @ w[1:])
                b[k] -= (p - y).mean() / max((p * (1 - p)).mean(), 1e-6)
        gs = []
        for k, (X, y) in enumerate(data):
            th = w.copy(); th[0] = b[k]
            g = grad(X, y, th); g[0] = 0.0
            gs.append(g)
        w = w - eta * _avg(rng, gs, n, sd)
    start = np.repeat(w[None], len(data), 0)
    start[:, 0] = b
    return LIT.fine_tune(data, start, steps=ft)


def fedprox(rng, data, n, rho, R, eta, mu, E=5, ft=(0, 5, 30)):
    sd = _sd(n, rho, R, factor=eta * E)
    w = np.zeros(P)
    for _ in range(R):
        deltas = []
        for X, y in data:
            th = w.copy()
            for _ in range(E):
                th = th - eta * (grad(X, y, th) + mu * (th - w))
            deltas.append(th - w)
        w = w + _avg(rng, deltas, n, sd)
    return LIT.fine_tune(data, np.repeat(w[None], len(data), 0), steps=ft)


GRID = {
    "Ditto": [(R, eta, lam) for R in (3, 10, 30) for eta in (0.5, 2.0) for lam in (0.1, 1.0, 10.0)],
    "pFedMe": [(R, eta, lam) for R in (3, 10, 30) for eta in (0.5, 2.0) for lam in (1.0, 10.0)],
    "APFL": [(R, eta, al) for R in (3, 10, 30) for eta in (0.5, 2.0) for al in (0.25, 0.5, 0.75)],
    "Per-FedAvg": [(R, eta, a) for R in (3, 10, 30) for eta in (0.5, 2.0) for a in (0.1, 0.5)],
    "FedRep": [(R, eta) for R in (3, 10, 30) for eta in (0.5, 2.0)],
    "FedProx": [(R, eta, mu) for R in (3, 10, 30) for eta in (0.2, 0.5) for mu in (0.1, 1.0)],
}


def all_estimates(rng, data, n, eps, delta=1e-5):
    """dict candidate name -> (K, P) for every method and configuration at budget eps."""
    rho = LIT.rho_total(eps, delta)
    t = f"{eps:g}"
    out = {}
    for R, eta, lam in GRID["Ditto"]:
        out[f"PDIT{t}|R={R}|eta={eta}|lam={lam}"] = ditto(rng, data, n, rho, R, eta, lam)
    for R, eta, lam in GRID["pFedMe"]:
        out[f"PME{t}|R={R}|eta={eta}|lam={lam}"] = pfedme(rng, data, n, rho, R, eta, lam)
    for R, eta, al in GRID["APFL"]:
        out[f"APFL{t}|R={R}|eta={eta}|alpha={al}"] = apfl(rng, data, n, rho, R, eta, al)
    for R, eta, a in GRID["Per-FedAvg"]:
        out[f"PFA{t}|R={R}|eta={eta}|a={a}"] = per_fedavg(rng, data, n, rho, R, eta, a)
    for R, eta in GRID["FedRep"]:
        for E, th in fedrep(rng, data, n, rho, R, eta).items():
            out[f"REP{t}|R={R}|eta={eta}|E={E}"] = th
    for R, eta, mu in GRID["FedProx"]:
        for E, th in fedprox(rng, data, n, rho, R, eta, mu).items():
            out[f"PROX{t}|R={R}|eta={eta}|mu={mu}|E={E}"] = th
    return out


def families(eps_list):
    """TUNED-style mapping: family name -> candidate names."""
    fam = {}
    for e in eps_list:
        t = f"{e:g}"
        fam[f"Ditto (DP), eps={t}"] = [f"PDIT{t}|R={R}|eta={eta}|lam={lam}" for R, eta, lam in GRID["Ditto"]]
        fam[f"pFedMe (DP), eps={t}"] = [f"PME{t}|R={R}|eta={eta}|lam={lam}" for R, eta, lam in GRID["pFedMe"]]
        fam[f"APFL (DP), eps={t}"] = [f"APFL{t}|R={R}|eta={eta}|alpha={al}" for R, eta, al in GRID["APFL"]]
        fam[f"Per-FedAvg (DP), eps={t}"] = [f"PFA{t}|R={R}|eta={eta}|a={a}" for R, eta, a in GRID["Per-FedAvg"]]
        fam[f"FedRep (DP), eps={t}"] = [f"REP{t}|R={R}|eta={eta}|E={E}" for R, eta in GRID["FedRep"] for E in (0, 5)]
        fam[f"FedProx (DP), eps={t}"] = [f"PROX{t}|R={R}|eta={eta}|mu={mu}|E={E}" for R, eta, mu in GRID["FedProx"] for E in (0, 5, 30)]
    return fam


def configure(p, g):
    """Number of coefficients and norm bound of x, for models other than the 5-coefficient heads used everywhere else
    (real_v3.py); the defaults reproduce every reported result."""
    global P, G
    P, G = p, g
