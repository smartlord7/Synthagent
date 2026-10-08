"""Closed-form quantities of the blueprint (Section 8): Propositions 8.1, 8.2 and 8.4.

The Gaussian model of Equation (4): theta_k = mu + u_k, u_k ~ N(0, tau2);
theta_hat_k = theta_k + e_k, e_k ~ N(0, v_k), v_k = sigma2 / n_k.
"""
import numpy as np


def dp_noise_var(n, eps, delta=1e-5, clip=1.0):
    """Variance s_j^2 of the Gaussian mechanism for a clipped mean (Proposition 8.2)."""
    if eps is None or np.isinf(eps):
        return np.zeros_like(np.asarray(n, dtype=float))
    sens = 2.0 * clip / np.asarray(n, dtype=float)
    return (sens * np.sqrt(2.0 * np.log(1.25 / delta)) / eps) ** 2


def loo_mse(n_others, tau2, sigma2=1.0, eps=None, delta=1e-5, clip=1.0):
    """M of Equation (6): MSE of the leave-one-out average of privatized releases."""
    n_others = np.asarray(n_others, dtype=float)
    k1 = len(n_others)
    return np.sum(tau2 + sigma2 / n_others + dp_noise_var(n_others, eps, delta, clip)) / k1 ** 2


def risk_local(n, sigma2=1.0):
    return sigma2 / np.asarray(n, dtype=float)


def risk_hb(n, tau2, M, sigma2=1.0):
    """Optimal risk R_k* of Proposition 8.1."""
    v = sigma2 / np.asarray(n, dtype=float)
    return v * (tau2 + M) / (v + tau2 + M)


def alpha_hb(n, tau2, M, sigma2=1.0):
    v = sigma2 / np.asarray(n, dtype=float)
    return (tau2 + M) / (tau2 + M + v)


def borrowed_patients(tau2, M, sigma2=1.0):
    """B = sigma^2 / (tau^2 + M) of Equation (5)."""
    return sigma2 / (tau2 + M)


def risk_full_pooling(n, n_others, tau2, sigma2=1.0):
    """Risk for site k of the sample-size-weighted pooled mean (a single global model)."""
    n = np.asarray(n, dtype=float)
    n_others = np.asarray(n_others, dtype=float)
    total = n + n_others.sum()
    wk = n / total
    w_others_sq = (n_others[None, :] / total[:, None]) ** 2 if n.ndim else (n_others / total) ** 2
    s_others = w_others_sq.sum(axis=-1)
    return tau2 * ((1 - wk) ** 2 + s_others) + sigma2 / total


def metropolis_hastings(adj):
    """Symmetric doubly stochastic gossip matrix W_kl = 1/(1+max(d_k, d_l))."""
    adj = np.asarray(adj, dtype=float)
    deg = adj.sum(axis=1)
    K = len(adj)
    W = np.zeros((K, K))
    for k in range(K):
        for l in range(K):
            if adj[k, l]:
                W[k, l] = 1.0 / (1.0 + max(deg[k], deg[l]))
        W[k, k] = 1.0 - W[k].sum()
    return W


def slem(W):
    """Second largest eigenvalue modulus lambda = max(|lambda_2|, |lambda_K|) (Proposition 8.4)."""
    ev = np.sort(np.linalg.eigvalsh(W))
    return max(abs(ev[-2]), abs(ev[0]))
