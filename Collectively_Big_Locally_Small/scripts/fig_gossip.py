"""Figure 9: gossip consensus error for 24 hospitals on four sparse topologies (Proposition 8.4).

The random graphs and the initial releases of the published figure were drawn with a random
generator whose state is not recoverable from the PDF. This script draws equivalent graphs with
fixed seeds chosen so that the second largest eigenvalue modulus of each Metropolis-Hastings
gossip matrix equals the published value to three decimals (0.977, 0.964, 0.937, 0.933). The
curves therefore have the same rates as the published ones but not the same first few rounds.
It writes figures/regenerated/fig_gossip.pdf; the paper includes the original figure.
"""
import itertools
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import style  # noqa: E402
from blueprint import metropolis_hastings, slem  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "figures" / "regenerated" / "fig_gossip.pdf"
K, ROUNDS = 24, 120
SEED_REGIONAL, SEED_RANDOM, SEED_RELEASES = 5, 577, 0


def ring(k=K, hops=1):
    A = np.zeros((k, k))
    for i in range(k):
        for h in range(1, hops + 1):
            A[i, (i + h) % k] = A[(i + h) % k, i] = 1
    return A


def regional(rng, clusters=4, p_in=0.8, k=K):
    """Dense regional clusters joined in a ring by single bridges."""
    size = k // clusters
    A = np.zeros((k, k))
    for c in range(clusters):
        for i, j in itertools.combinations(range(c * size, (c + 1) * size), 2):
            if rng.random() < p_in:
                A[i, j] = A[j, i] = 1
    for c in range(clusters):
        i = c * size + rng.integers(size)
        j = ((c + 1) % clusters) * size + rng.integers(size)
        A[i, j] = A[j, i] = 1
    return A


def erdos_renyi(rng, p=0.15, k=K):
    A = np.triu((rng.random((k, k)) < p).astype(float), 1)
    return A + A.T


def consensus_error(W, x0, rounds=ROUNDS):
    x, xbar = x0.copy(), x0.mean()
    e0 = np.linalg.norm(x0 - xbar)
    out = []
    for _ in range(rounds):
        out.append(np.linalg.norm(x - xbar) / e0)
        x = W @ x
    return np.array(out)


def topologies():
    return [
        ("Ring (degree 2)", ring(), style.GRAY400),
        ("Regional clusters + bridges", regional(np.random.default_rng(SEED_REGIONAL)), style.AMBER600),
        ("Sparse random (p=0.15)", erdos_renyi(np.random.default_rng(SEED_RANDOM)), style.BLUE600),
        ("Ring + 2-hop (degree 4)", ring(hops=2), style.TEAL700),
    ]


def main():
    style.setup()
    x0 = np.random.default_rng(SEED_RELEASES).normal(size=K)
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    for name, A, color in topologies():
        W = metropolis_hastings(A)
        lam = slem(W)
        ax.plot(range(ROUNDS), consensus_error(W, x0), color=color, linewidth=1.4,
                label=f"{name}, $\\lambda_2$={lam:.3f}")
        print(f"{name}: lambda={lam:.4f}, rounds to 1e-3 ~ {np.log(1e3) / (1 - lam):.0f}")
    ax.set_yscale("log")
    ax.set_xlabel("Gossip rounds $t$")
    ax.set_ylabel("Relative consensus error")
    ax.legend(loc="lower left", frameon=False)
    OUT.parent.mkdir(exist_ok=True)
    style.save(fig, OUT)


if __name__ == "__main__":
    main()
