"""Summary of the evidence: every scenario of Simulations 2-15 on one axis.

Each row is a scenario; each marker is a method's error relative to local estimation (Gaussian
simulations: mean risk R_k/v_k; logistic simulations: mean excess log-loss divided by that of local
estimation). Values below 1 mean the hospitals gain. Reads the saved results; no simulation is rerun.
"""
import json

import matplotlib.pyplot as plt
import numpy as np

import ghb
import style

R = ghb.RESDIR
COL = {"GHB": style.TEAL700, "Robust GHB": style.VIOLET600, "GHB + opt-out": style.AMBER600,
       "Ditto-style": style.BLUE600, "Meta-analysis": "#6b7280", "FedAvg": style.SLATE800,
       "Global (no DP)": style.RED700, "GHB, overconfident anchor": style.RED700,
       "GHB + conflict check": style.AMBER600, "DP-FedAvg + fine-tuning": "#92400e",
       "Decentralized DP-SGD": "#be185d", "Clustered DP-FedAvg": style.SLATE800}
MK = {"GHB": "o", "Robust GHB": "s", "GHB + opt-out": "^", "Ditto-style": "D", "Meta-analysis": "v", "FedAvg": "<",
      "Global (no DP)": "x", "GHB, overconfident anchor": "P", "GHB + conflict check": "^",
      "DP-FedAvg + fine-tuning": ">", "Decentralized DP-SGD": "p", "Clustered DP-FedAvg": "^"}


def rows():
    out = []
    s2 = json.load(open(R / "s2_heterogeneity.json"))
    t2 = np.array(s2["tau2"])
    c = s2["curves"]["1.0"]
    for i, lab in ((int(np.argmin(abs(t2 - 0.02))), None), (len(t2) - 1, None)):
        out.append((f"Sim 2: $\\tau^2$={t2[i]:.2g}, $\\epsilon$=1",
                    {"GHB": float(np.mean(c["GHB, plug-in, gossip T=60"][i])),
                     "Global (no DP)": float(np.mean(c["Global model (full pooling)"][i]))}))
    s9 = json.load(open(R / "sim9_atypical.json"))["out"]
    out.append(("Sim 9: heavy-tailed site effects", {m: s9["t3"]["summary"][m] for m in ("GHB", "Robust GHB", "GHB + opt-out")}))
    out.append(("Sim 9: outlier at the smallest hospital (that hospital)",
                {m: s9["outlier_small"]["summary"][f"{m} outlier"] for m in ("GHB", "Robust GHB", "GHB + opt-out")}))
    out.append(("Sim 9: outlier at the largest hospital (that hospital)",
                {m: s9["outlier_large"]["summary"][f"{m} outlier"] for m in ("GHB", "Robust GHB", "GHB + opt-out")}))
    s12 = json.load(open(R / "sim12_byzantine.json"))
    b = s12["b"]
    for bb in (8, 30):
        j = b.index(bb)
        out.append((f"Sim 12: 3 dishonest hospitals, lie of {bb}$\\tau$",
                    {"GHB": s12["out"]["3"]["GHB"][j], "Robust GHB": s12["out"]["3"]["Robust GHB"][j]}))
    s13 = json.load(open(R / "sim13_dropout.json"))
    out.append(("Sim 13: 80% of links fail each round, T=60", {"GHB": s13["risk"]["0.8"]["60"]}))
    s8 = json.load(open(R / "sim8_baselines.json"))
    names = {"GHB (DP)": "GHB", "Ditto-style (DP)": "Ditto-style", "Meta-analysis (DP)": "Meta-analysis",
             "FedAvg (DP)": "FedAvg", "Global (pooled, no DP)": "Global (no DP)"}
    for key, lab in (("het", "0.05"), ("het", "0.5"), ("shift", "1.0")):
        kl = s8[key][lab]["kl"]
        title = (f"Sim 8: logistic, $\\tau^2$={lab}" if key == "het" else "Sim 10: logistic, case-mix shift 1.0")
        out.append((title, {names[m]: kl[m] / kl["Local"] for m in names}))
    s11 = json.load(open(R / "sim11_anchor.json"))["res"]["2.0"]["kl"]
    out.append(("Sim 11: anchor wrong by 2 s.d.",
                {"GHB": s11["GHB, anchor uncertainty stated honestly"] / s11["Local"],
                 "GHB, overconfident anchor": s11["GHB, anchor overconfident (stated 0.3)"] / s11["Local"],
                 "GHB + conflict check": s11["GHB, overconfident + conflict check"] / s11["Local"]}))
    s7 = json.load(open(R / "s7_logistic.json"))["table"]["tau2=0.05|eps=0.3"]
    out.append(("Sim 7: logistic, strict budget $\\epsilon$=0.3", {"GHB": s7["GHB"]["kl"] / s7["Local"]["kl"]}))
    s15 = json.load(open(R / "sim15_literature.json"))["res"]
    for key, title in (("tau2=0.05", "Sim 15: iterative baselines, $\\tau^2$=0.05"), ("eps=0.3", "Sim 15: iterative baselines, $\\epsilon$=0.3"),
                       ("regional", "Sim 15: regional heterogeneity"), ("anchor=1.0", "Sim 15: anchor wrong by 1 s.d.")):
        kl = s15[key]["kl"]
        out.append((title, {m: kl[k] / kl["Local"] for m, k in (("GHB", "GHB (DP)"), ("DP-FedAvg + fine-tuning", "DP-FedAvg + fine-tuning"),
                                                                 ("Decentralized DP-SGD", "Decentralized DP-SGD"),
                                                                 ("Clustered DP-FedAvg", "Clustered DP-FedAvg"))}))
    return out


def main():
    rs = rows()
    style.setup()
    fig, ax = plt.subplots(figsize=(6.8, 0.27 * len(rs) + 0.9))
    seen = set()
    for i, (lab, vals) in enumerate(rs):
        y = len(rs) - 1 - i
        for m, v in vals.items():
            ax.scatter([v], [y], s=22 if MK[m] != "x" else 26, marker=MK[m], color=COL[m], zorder=3,
                       label=m if m not in seen else None, linewidths=1.0)
            seen.add(m)
        ax.hlines(y, min(vals.values()), max(vals.values()), color=style.GRAY400, lw=0.6, zorder=1)
    ax.axvline(1, color=style.SLATE800, lw=0.8)
    ax.text(1.02, len(rs) - 0.4, "local estimation", fontsize=6.3, va="bottom")
    ax.set_yticks(range(len(rs)), [lab for lab, _ in rs][::-1])
    ax.set_xscale("log")
    ax.set_xlabel("Error relative to local estimation (below 1: the hospitals gain)")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.13 * 12 / len(rs)), ncol=5)
    ghb.tidy(ax)
    ax.set_xticks([0.2, 0.5, 1, 2, 5, 20, 100], ["0.2", "0.5", "1", "2", "5", "20", "100"])
    ax.set_xlim(0.2, 200)
    style.save(fig, ghb.FIGDIR / "sim_forest.pdf")
    json.dump(dict(rows=rs), open(R / "sim_forest.json", "w"), indent=1)
    for lab, v in rs:
        print(lab, {m: round(x, 3) for m, x in v.items()})


if __name__ == "__main__":
    main()
