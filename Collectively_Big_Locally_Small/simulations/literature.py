"""Iterative baselines from the federated-learning literature, for Simulation 15 (logistic heads).

Every method spends the same total privacy budget (epsilon, delta) per hospital as GHB, accounted
with zero-concentrated DP (Bun and Steinke, 2016): a Gaussian mechanism with sensitivity Delta and
noise s is rho-zCDP with rho = Delta^2 / (2 s^2), rho adds over rounds, and the total rho is chosen
so that rho + 2 sqrt(rho ln(1/delta)) = epsilon. This accounting is tighter than the classical
Gaussian mechanism that GHB's single release uses, so it favours the baselines.

Per-patient gradients of the logistic loss are (p - y) x with ||x|| <= sqrt(5), so they need no
clipping; replacing one patient moves a hospital's mean gradient by at most 2 sqrt(5) / n_k.

  DP-FedAvg            a server holds one model; each round every hospital sends its noisy mean
                       gradient at that model and the server takes an n-weighted gradient step
                       (McMahan et al., 2017/2018, full-batch form).
  + fine-tuning        each hospital then takes E plain gradient steps on its own data from the
                       final global model (no privacy cost: the data stay local).
  Decentralized DP-SGD no server: each hospital steps on its own noisy gradient and averages models
                       with its neighbours (Metropolis-Hastings weights on the agreement graph),
                       adapt-then-combine (Lian et al., 2017; swarm learning, Warnat-Herresthal
                       et al., 2021); optional local fine-tuning as above.
  Local GD from anchor a control without any other hospital: E plain gradient steps from the published
                       anchor mu_0 on the hospital's own data (early stopping as the only regularizer).
  Clustered DP-FedAvg  IFCA-style clustering (Ghosh et al., 2020) made private: choosing a cluster
                       by local loss each round has sensitivity O(1), not O(1/n_k), so hospitals
                       instead make one private ridge release with half the budget, the releases
                       are clustered by weighted k-means (post-processing), and DP-FedAvg runs
                       within each cluster with the other half; optional local fine-tuning.

Hyperparameters (rounds R, step size eta, fine-tuning steps E, clusters C) are tuned by the caller
on separate replicates, with oracle access to the excess log-loss; GHB has no tuning parameter.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit

from exp7_logistic import LAM, P, R_X, fit

G = np.sqrt(5.0)  # bound on the norm of a per-patient gradient
ROUNDS = (1, 2, 3, 5, 10, 30, 100)
DROUNDS = (3, 10, 30, 100)
ETAS = (0.2, 0.5, 2.0)
FT_STEPS = (0, 1, 2, 3, 5, 10, 30, 100)
CLUSTERS = (2, 3, 4)
FT_ETA = 1.0


def rho_total(eps, delta):
    """Total zCDP budget whose (epsilon, delta)-DP conversion equals the target epsilon."""
    if eps is None or np.isinf(eps):
        return np.inf
    return brentq(lambda r: r + 2 * np.sqrt(r * np.log(1 / delta)) - eps, 1e-12, 50)


def grad(X, y, th):
    return X.T @ (expit(X @ th) - y) / len(y)


def noisy_grads(rng, data, thetas, sd):
    return np.array([grad(X, y, thetas[k]) + sd[k] * rng.standard_normal(P) for k, (X, y) in enumerate(data)])


def fine_tune(data, start, steps=FT_STEPS):
    """Snapshots after E local full-batch gradient steps from each hospital's start model."""
    out, th, done = {}, np.array(start, float).copy(), 0
    for E in sorted(steps):
        for _ in range(E - done):
            th = th - FT_ETA * np.array([grad(X, y, th[k]) for k, (X, y) in enumerate(data)])
        done = E
        out[E] = th.copy()
    return out


def fedavg(rng, data, n, init, rho, R, eta, members=None):
    """Full-batch DP-FedAvg over the hospitals in `members` (all by default); returns the global model."""
    idx = np.arange(len(n)) if members is None else np.asarray(members)
    w = n[idx] / n[idx].sum()
    sd = np.zeros(len(idx)) if np.isinf(rho) else 2 * G / n[idx] / np.sqrt(2 * rho / R)
    sub = [data[k] for k in idx]
    th = np.array(init, float).copy()
    for _ in range(R):
        g = noisy_grads(rng, sub, np.repeat(th[None], len(idx), 0), sd)
        th = th - eta * (w @ g)
    return th


def decentralized(rng, data, n, init, rho, R, eta, W):
    sd = np.zeros(len(n)) if np.isinf(rho) else 2 * G / n / np.sqrt(2 * rho / R)
    th = np.repeat(np.asarray(init, float)[None], len(n), 0)
    for _ in range(R):
        th = W @ (th - eta * noisy_grads(rng, data, th, sd))
    return th


def private_clusters(rng, data, n, rho, C, iters=25):
    """Cluster labels from one ridge release per hospital (budget rho), by weighted k-means."""
    raw = np.array([fit(X, y, np.zeros(P), np.full(P, len(y) * LAM)) for X, y in data])
    s2 = np.zeros(len(n)) if np.isinf(rho) else (2 * R_X / (LAM * n)) ** 2 / (2 * rho)
    x = raw + np.sqrt(s2)[:, None] * rng.standard_normal(raw.shape)
    w = 1.0 / (s2 + 1.0 / n)
    cent = [x[np.argmax(w)]]  # weighted k-means++ start from the most precise release
    for _ in range(C - 1):
        d = np.min([((x - c) ** 2).sum(1) for c in cent], axis=0) * w
        cent.append(x[rng.choice(len(n), p=d / d.sum())])
    cent = np.array(cent)
    for _ in range(iters):
        lab = np.argmin(((x[:, None] - cent[None]) ** 2).sum(2), axis=1)
        for c in range(C):
            if (lab == c).any():
                cent[c] = (w[lab == c, None] * x[lab == c]).sum(0) / w[lab == c].sum()
    return lab


def estimates(rng, data, n, mu0, eps, delta, W, keep=None):
    """Per-hospital estimates of every baseline configuration: dict name -> (K, P) array.

    Names are 'family|key=value|...'; `keep` (a set of names) restricts the work to those configurations.
    """
    rho, K, out = rho_total(eps, delta), len(n), {}
    want = (lambda name: True) if keep is None else (lambda name: name in keep)
    prefix_wanted = (lambda pre: keep is None or any(k.startswith(pre) for k in keep))
    if prefix_wanted("Local GD from anchor|"):
        for E, th in fine_tune(data, np.repeat(np.asarray(mu0, float)[None], K, 0)).items():
            if E > 0 and want(f"Local GD from anchor|E={E}"):
                out[f"Local GD from anchor|E={E}"] = th
    for R in ROUNDS:
        for eta in ETAS:
            pre = f"DP-FedAvg|R={R}|eta={eta}|"
            if not prefix_wanted(pre):
                continue
            g = fedavg(rng, data, n, mu0, rho, R, eta)
            for E, th in fine_tune(data, np.repeat(g[None], K, 0)).items():
                if want(pre + f"E={E}"):
                    out[pre + f"E={E}"] = th
    for R in DROUNDS:
        for eta in ETAS:
            pre = f"Decentralized DP-SGD|R={R}|eta={eta}|"
            if not prefix_wanted(pre):
                continue
            th0 = decentralized(rng, data, n, mu0, rho, R, eta, W)
            for E, th in fine_tune(data, th0).items():
                if want(pre + f"E={E}"):
                    out[pre + f"E={E}"] = th
    for C in CLUSTERS:
        if not prefix_wanted(f"Clustered DP-FedAvg|C={C}|"):
            continue
        lab = private_clusters(rng, data, n, rho / 2, C)
        for R in ROUNDS:
            for eta in ETAS:
                pre = f"Clustered DP-FedAvg|C={C}|R={R}|eta={eta}|"
                if not prefix_wanted(pre):
                    continue
                th0 = np.empty((K, P))
                for c in range(C):
                    mem = np.flatnonzero(lab == c)
                    if len(mem):
                        th0[mem] = fedavg(rng, data, n, mu0, rho / 2, R, eta, mem)
                for E, th in fine_tune(data, th0).items():
                    if want(pre + f"E={E}"):
                        out[pre + f"E={E}"] = th
    return out
