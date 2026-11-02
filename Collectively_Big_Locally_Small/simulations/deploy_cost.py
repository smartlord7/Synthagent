"""Deployment cost of GHB on the TCGA networks (Section "Deploying GHB in a real network").

Computation: the wall time of GHB as released (real_v3.GHB_V5) on the design and held-out sets, on one processor
core, divided by the number of hospitals (every hospital's local computation is included; the two releases are
simulated by their exact sums, the ideal functionality of secure summation).
Communication, for secure summation along a spanning tree of the agreement graph: each hospital publishes one public
key, relayed to every other hospital so that every pair can derive a pairwise mask (32-byte keys); sends its masked
share, or its subtree's masked subtotal, once to its parent (p + p(p+1)/2 numbers, 8 bytes each); and relays the total
to its children. The bound below takes a tree of maximum degree 4 and the largest network.
    python3 deploy_cost.py        (results/deploy_cost.json)
"""
import json
import time

import numpy as np

import ghb
import ghb_v4 as V4
import real_v3 as RV

KEY_BYTES, NUM_BYTES, TREE_DEGREE = 32, 8, 4


def main():
    res = {}
    kmax = 0
    for which in ("confirm", "dev", "fresh"):
        nets = {p: RV.load(p)[0] for p in RV.SETS[which]}
        kmax = max(kmax, max(len(v) for v in nets.values()))
        K = sum(len(v) for v in nets.values())
        for e in RV.EPS:
            t = time.time()
            V4.estimate(nets, np.random.default_rng(0), e, RV.DELTA, RV.G, cfg=RV.GHB_V5)
            dt = time.time() - t
            res[f"{which} eps={e:g}"] = dict(hospitals=K, seconds=dt, per_hospital=dt / K)
    P = RV.P
    nums = P + P * (P + 1) // 2
    msg = nums * NUM_BYTES
    res["message_numbers"], res["message_bytes"] = nums, msg
    res["largest_network_hospitals"] = kmax
    res["tree"] = dict(max_degree=TREE_DEGREE, key_bytes=KEY_BYTES,
                       bytes_per_hospital=(TREE_DEGREE + 2) * msg + kmax * KEY_BYTES)
    json.dump(res, open(ghb.RESDIR / "deploy_cost.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
