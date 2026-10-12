"""Final real-data comparison: GHB against every literature method in three tuning arms (real_networks.json).

oracle      each baseline tuned on the network's own held-out folds (upper bound)
uci         tuned on the development network (UCI Heart Disease), then frozen
lono        leave-one-network-out: the configuration with the lowest mean relative error over the other seven networks
"""
import json

import numpy as np

import ghb
import real_networks as RN

CONF = ["medpar", "TCGA-LUSC", "TCGA-HNSC", "TCGA-BLCA", "TCGA-STAD", "TCGA-LIHC", "TCGA-KIRC"]
OTHERS = {"DP-FedAvg": "DP-FedAvg, eps={e}", "DP-FedAvg + FT": "DP-FedAvg + fine-tuning, eps={e}",
          "DP-FedAvg + FT, sec. agg.": "DP-FedAvg-SA + fine-tuning, eps={e}", "Decentralized DP-SGD": "Decentralized DP-SGD, eps={e}",
          "Clustered DP-FedAvg": "Clustered DP-FedAvg, eps={e}", "Ditto-style (one-shot)": "Ditto-style (DP), eps={e}",
          "Ditto": "Ditto (DP), eps={e}", "pFedMe": "pFedMe (DP), eps={e}", "APFL": "APFL (DP), eps={e}",
          "Per-FedAvg": "Per-FedAvg (DP), eps={e}", "FedRep": "FedRep (DP), eps={e}", "FedProx": "FedProx (DP), eps={e}"}


def arm_value(R, nm, fam, arm, key="rel"):
    if arm == "oracle":
        return R[nm][fam][key]
    cands = RN.TUNED[fam]
    if arm == "uci":
        best = min(cands, key=lambda c: R["heart"]["cand"][c]["rel"])
    else:
        best = min(cands, key=lambda c: np.mean([R[o]["cand"][c]["rel"] for o in CONF + ["heart"] if o != nm]))
    return R[nm]["cand"][best][key]


def main():
    R = json.load(open(ghb.RESDIR / "real_networks.json"))["res"]
    out = {}
    for e in ("1", "10"):
        for arm in ("oracle", "uci", "lono"):
            rows = {}
            for nm in CONF:
                v = {lab: arm_value(R, nm, f.format(e=e), arm) for lab, f in OTHERS.items()}
                g = {"GHB-EB (no tuning)": R[nm][f"GHB-EB {e}"]["rel"],
                     "GHB, tuned prior": arm_value(R, nm, f"GHB, tuned prior (DP), eps={e}", arm)}
                loc = {"Local, tuned prior": arm_value(R, nm, "Local, tuned prior", arm)}
                best_other = min(v, key=v.get)
                rows[nm] = dict(others=v, ghb=g, local=loc, best_other=best_other)
            out[f"eps={e}|{arm}"] = rows
            wins_eb = sum(rows[nm]["ghb"]["GHB-EB (no tuning)"] < rows[nm]["others"][rows[nm]["best_other"]] for nm in CONF)
            wins_t = sum(rows[nm]["ghb"]["GHB, tuned prior"] < rows[nm]["others"][rows[nm]["best_other"]] for nm in CONF)
            print(f"eps={e} {arm:6s}: GHB-EB beats the best literature method in {wins_eb}/7, GHB tuned prior in {wins_t}/7")
            for nm in CONF:
                r = rows[nm]
                print(f"   {R[nm]['info']['label'][:15]:15s} GHB-EB {r['ghb']['GHB-EB (no tuning)']:.3f}  GHB-tuned {r['ghb']['GHB, tuned prior']:.3f}  "
                      f"best other {r['others'][r['best_other']]:.3f} ({r['best_other']})  local-tuned {r['local']['Local, tuned prior']:.3f}")
    json.dump(out, open(ghb.RESDIR / "final_compare.json", "w"), indent=1)


if __name__ == "__main__":
    main()
