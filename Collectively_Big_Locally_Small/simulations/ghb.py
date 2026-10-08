"""Graph Hierarchical Borrowing (GHB) and its baselines, as Monte Carlo estimators.

Arrays of site quantities have shape (R, K): R independent replicates of a network of K hospitals.
Everything a hospital computes from other hospitals uses only their privatized releases and
public quantities (site sizes, the Gaussian-mechanism noise variance, the per-patient variance),
so by the post-processing property of differential privacy it inherits the release guarantee.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from blueprint import dp_noise_var, metropolis_hastings, slem  # noqa: E402,F401

FIGDIR = ROOT / "supplement" / "figures"
RESDIR = ROOT / "supplement" / "results"
DELTA, CLIP, SIGMA2 = 1e-5, 1.0, 1.0


def site_sizes(K=24, median=100, spread=1.0, lo=15, hi=1500, seed=11):
    """A fixed, skewed set of hospital sizes (log-normal, clipped), sorted ascending."""
    rng = np.random.default_rng(seed)
    n = np.exp(np.log(median) + spread * rng.standard_normal(K))
    return np.sort(np.clip(np.round(n), lo, hi))


def draw(rng, n, tau2, R, sigma2=SIGMA2, mu=0.0):
    """Model (4): theta_k = mu + u_k and theta_hat_k = theta_k + e_k."""
    n = np.asarray(n, dtype=float)
    theta = mu + np.sqrt(tau2) * rng.standard_normal((R, len(n)))
    theta_hat = theta + np.sqrt(sigma2 / n) * rng.standard_normal((R, len(n)))
    return theta, theta_hat


def release(rng, theta_hat, n, eps, delta=DELTA, clip=CLIP):
    """Gaussian-mechanism release of each site estimate (Proposition 8.2). Returns (x, s2)."""
    s2 = dp_noise_var(n, eps, delta, clip) * np.ones(np.shape(n))
    return theta_hat + np.sqrt(s2) * rng.standard_normal(theta_hat.shape), s2


class Averager:
    """Network averaging seen by each hospital: exact (a server, for reference) or T gossip rounds."""

    def __init__(self, W=None, T=None):
        self.P = None if W is None else np.linalg.matrix_power(W, T)

    def __call__(self, y):
        if self.P is None:
            return np.repeat(y.mean(axis=1, keepdims=True), y.shape[1], axis=1)
        return y @ self.P.T


def dersimonian_laird(x, r, avg):
    """Method-of-moments tau^2 from releases x with known variances r, using network averages only."""
    K = x.shape[1]
    w = np.broadcast_to(1.0 / r, x.shape)
    S1, Sx, Sxx, S11 = (K * avg(q) for q in (w, w * x, w * x * x, w * w))
    S1 = np.maximum(S1, 1e-12)
    Q = Sxx - Sx ** 2 / S1
    denom = np.maximum(S1 - S11 / S1, 1e-12)
    return np.maximum(0.0, (Q - (K - 1)) / denom)


def tau2_bayes(x, r, scale=0.5, grid=np.r_[0.0, np.logspace(-3.5, 1, 160)]):
    """Posterior mean of tau^2 under a half-Cauchy(scale) prior on tau (Remark 8.3).

    Uses the marginal likelihood of the releases, x_j ~ N(mu, tau^2 + r_j) with mu integrated out
    under a flat prior, on a grid of tau. It needs every release, which each hospital can obtain
    by flooding them over the graph; it is post-processing of privatized releases.
    """
    x = np.atleast_2d(x)
    r = np.broadcast_to(r, x.shape)
    logpost = []
    for t in grid:
        V = t * t + r
        w = 1.0 / V
        mu = (w * x).sum(1, keepdims=True) / w.sum(1, keepdims=True)
        Q = (w * (x - mu) ** 2).sum(1)
        ll = -0.5 * np.log(V).sum(1) - 0.5 * np.log(w.sum(1)) - 0.5 * Q
        logpost.append(ll - np.log1p((t / scale) ** 2))
    lp = np.array(logpost)  # (G, R)
    dt = np.gradient(grid)[:, None]
    p = np.exp(lp - lp.max(0)) * dt
    p /= p.sum(0)
    t2 = (p * (grid ** 2)[:, None]).sum(0)
    return np.repeat(t2[:, None], x.shape[1], axis=1)


def borrow(x, r, avg=None, tau2=None, weights="iv"):
    """What the network gives each hospital: tau^2, the leave-one-out population estimate and its MSE.

    x are the privatized releases and r their (public) variances. tau^2 is estimated by
    DerSimonian-Laird (Remark 8.3) unless an oracle value is given; the leave-one-out mean is
    formed by (gossip) averaging, inverse-variance ("iv") or equal-weighted ("equal").
    """
    avg = avg or Averager()
    K = x.shape[1]
    if tau2 is None:
        t2 = dersimonian_laird(x, r, avg)
    elif isinstance(tau2, str) and tau2 == "bayes":
        t2 = tau2_bayes(x, r)
    else:
        t2 = np.broadcast_to(np.asarray(tau2, dtype=float), x.shape)
    w = 1.0 / (t2 + r) if weights == "iv" else np.ones_like(x)
    Sw = np.maximum(K * avg(w) - w, 1e-12)
    mu_t = (K * avg(w * x) - w * x) / Sw
    if weights == "iv":
        M = 1.0 / Sw
    else:
        Kr = K * avg(np.broadcast_to(t2 + r, x.shape)) - (t2 + r)
        M = Kr / (K - 1) ** 2
    return t2, mu_t, M


def ghb(theta_hat, x, n, s2, sigma2=SIGMA2, avg=None, tau2=None, weights="iv"):
    """GHB estimate of every theta_k (Propositions 8.1, 8.2 and 8.4 combined, all plug-in):
    the noise-free local estimate is shrunk by alpha_k = (tau^2 + M)/(tau^2 + M + v_k)."""
    v = sigma2 / np.asarray(n, dtype=float)
    t2, mu_t, M = borrow(x, v + s2, avg, tau2, weights)
    alpha = (t2 + M) / (t2 + M + v)
    return alpha * theta_hat + (1 - alpha) * mu_t


def oracle_hb(theta_hat, x, n, s2, tau2, sigma2=SIGMA2):
    """Proposition 8.1 exactly: equal-weight leave-one-out mean of releases, true tau^2 and M."""
    K = x.shape[1]
    v = sigma2 / np.asarray(n, dtype=float)
    mu_t = (x.sum(axis=1, keepdims=True) - x) / (K - 1)
    tot = tau2 + v + s2
    M = (tot.sum() - tot) / (K - 1) ** 2
    alpha = (tau2 + M) / (tau2 + M + v)
    return alpha * theta_hat + (1 - alpha) * mu_t


def full_pooling(theta_hat, n):
    """A single global model: the sample-size weighted mean of all (noise-free) site estimates."""
    n = np.asarray(n, dtype=float)
    return np.repeat((theta_hat * n).sum(axis=1, keepdims=True) / n.sum(), theta_hat.shape[1], axis=1)


def risk(est, theta):
    """Per-site Monte Carlo risk and its standard error."""
    se2 = (est - theta) ** 2
    return se2.mean(axis=0), se2.std(axis=0, ddof=1) / np.sqrt(se2.shape[0])


# --- topologies of Figure 9 -------------------------------------------------------------------
def topologies(K=24):
    sys.path.insert(0, str(ROOT / "scripts"))
    import fig_gossip as fg

    return {
        "Ring (degree 2)": fg.ring(K),
        "Regional clusters + bridges": fg.regional(np.random.default_rng(fg.SEED_REGIONAL), k=K),
        "Sparse random (p=0.15)": fg.erdos_renyi(np.random.default_rng(fg.SEED_RANDOM), k=K),
        "Ring + 2-hop (degree 4)": fg.ring(K, hops=2),
    }


def tidy(*axes):
    """No labels on minor log ticks (they collide on short log axes)."""
    from matplotlib.ticker import NullFormatter

    for ax in axes:
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())


def tidy(*axes):
    """No labels on minor log ticks (they collide on short log axes)."""
    from matplotlib.ticker import NullFormatter

    for ax in axes:
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
