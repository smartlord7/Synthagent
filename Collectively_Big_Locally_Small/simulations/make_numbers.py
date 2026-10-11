"""Write the numbers quoted in the supplement and in Section 8 as LaTeX macros and tables.

Output: supplement/results/sim_numbers.tex (macros \\Sim...) and supplement/results/s7_table.tex.
"""
import json

import ghb

R = ghb.RESDIR


def f2(x):
    return f"{x:.2f}"


def main():
    s1 = json.load(open(R / "s1_validate.json"))["summary"]
    s2 = json.load(open(R / "s2_heterogeneity.json"))["summary"]
    s3 = json.load(open(R / "s3_equity.json"))["summary"]
    s4 = json.load(open(R / "s4_gossip.json"))
    s5 = json.load(open(R / "s5_joint_release.json"))["out"]
    s6 = json.load(open(R / "s6_gmrf.json"))
    s7 = json.load(open(R / "s7_logistic.json"))["table"]
    gname = "GHB, plug-in, gossip T=60"
    n = {
        "OneChecks": str(s1["n_checks"]), "OneWithin": f"{100 * s1['share_within_2se']:.0f}\\%",
        "OneMaxRel": f"{100 * s1['max_rel_err']:.1f}\\%", "OneReps": f"{s1['reps']:,}".replace(",", "{,}"),
        "TwoWorstGHB": f2(max(s2[f"{gname} | eps={e}"]["max_worst"] for e in ("None", "1.0"))),
        "TwoWorstPool": f"{max(s2[f'Global model (full pooling) | eps={e}']['max_worst'] for e in ('None', '1.0')):.0f}",
        "TwoMeanPool": f"{s2['Global model (full pooling) | eps=1.0']['max_mean']:.0f}",
        "TwoMinGHB": f2(s2[f"{gname} | eps=1.0"]["min_mean"]),
        "TwoMinOracle": f2(s2["GHB, oracle $\\tau^2$, exact mean | eps=1.0"]["min_mean"]),
        "TwoWorstFB": f2(max(s2[f"GHB, tau integrated out, gossip T=60 | eps={e}"]["max_worst"] for e in ("None", "1.0"))),
        "TwoMinFB": f2(s2["GHB, tau integrated out, gossip T=60 | eps=1.0"]["min_mean"]),
        "ThreeB": f"{s3['B']:.0f}", "ThreeSmall": f2(s3["gain_small"]), "ThreeLarge": f2(s3["gain_large"]),
        "ThreeMin": f2(s3["min_gain"]), "ThreePoolSmall": f2(s3["pool_gain_small"]),
        "ThreePoolLarge": f2(s3["pool_gain_large"]),
        "ThreeKThreeOracle": f2(s3["curves"]["oracle"][0]), "ThreeKThreeDL": f2(s3["curves"]["plug-in"][0]),
        "ThreeKThreeBayes": f2(s3["curves"]["bayes"][0]),
        "ThreeKFortyEightDL": f2(s3["curves"]["plug-in"][-1]),
        "ThreeKFortyEightOracle": f2(s3["curves"]["oracle"][-1]),
    }
    for name, r in s4["res"].items():
        key = {"Ring (degree 2)": "Ring", "Regional clusters + bridges": "Regional", "Sparse random (p=0.15)": "Random",
               "Ring + 2-hop (degree 4)": "Twohop"}[name]
        n[f"Four{key}Lam"] = f"{r['lam']:.3f}"
        n[f"Four{key}Pred"] = f"{r['t_pred']:.0f}"
        n[f"Four{key}Two"] = str(r["t_within_2pct"])
    for net, key in (("mixed", "Mixed"), ("small", "Small")):
        sm = s5[net]["summary"]
        for g in ("1", "8"):
            n[f"Five{key}G{'One' if g == '1' else 'Eight'}EpsTenth"] = f"{sm[g]['B_eps01']:.1f}"
            n[f"Five{key}G{'One' if g == '1' else 'Eight'}EpsOne"] = f"{sm[g]['B_eps1']:.1f}"
    n["SixGHBLow"], n["SixGMRFLow"] = f2(s6["ghb"][0]), f2(s6["gmrf"][0])
    n["SixGHBHigh"], n["SixGMRFHigh"] = f2(s6["ghb"][-1]), f2(s6["gmrf"][-1])
    n["SixBeta"] = f"{s6['best'][0][0]:.0f}"
    m = s7["tau2=0.05|eps=1.0"]
    n["SevenLocal"], n["SevenGHB"], n["SevenGlobal"] = (f"{m[k]['kl']:.4f}" for k in ("Local", "GHB", "Global"))
    n["SevenReduction"] = f"{100 * (1 - m['GHB']['kl'] / m['Local']['kl']):.0f}\\%"
    n["SevenGHBNoDP"] = f"{s7['tau2=0.05|eps=None']['GHB']['kl']:.4f}"
    n["SevenGHBTenth"] = f"{s7['tau2=0.05|eps=0.3']['GHB']['kl']:.4f}"
    n["SevenLocalTenth"] = f"{s7['tau2=0.05|eps=0.3']['Local']['kl']:.4f}"
    n["SevenWorstRatio"] = f2(m["GHB"]["kl_worst_ratio"])
    n["SevenGlobalWorstRatio"] = f2(m["Global"]["kl_worst_ratio"])
    # Simulations 8-14
    s8 = json.load(open(R / "sim8_baselines.json"))
    for tag, key, lab in (("Mid", "het", "0.05"), ("High", "het", "0.5"), ("Shift", "shift", "1.0")):
        kl = s8[key][lab]["kl"]
        for m, short in (("Local", "Local"), ("GHB (DP)", "GHB"), ("Ditto-style (DP)", "Ditto"), ("Meta-analysis (DP)", "Meta"),
                         ("FedAvg (DP)", "Fedavg"), ("Global (pooled, no DP)", "Pooled")):
            n[f"Eight{tag}{short}"] = f"{kl[m]:.4f}"
        n[f"Eight{tag}GHBvsDitto"] = f"{100 * (1 - kl['GHB (DP)'] / kl['Ditto-style (DP)']):.0f}\\%"
    s9 = json.load(open(R / "sim9_atypical.json"))["out"]
    n["NineGaussGHB"], n["NineGaussRobust"] = f2(s9["gauss"]["summary"]["GHB"]), f2(s9["gauss"]["summary"]["Robust GHB"])
    n["NineTGHB"], n["NineTRobust"] = f2(s9["t3"]["summary"]["GHB"]), f2(s9["t3"]["summary"]["Robust GHB"])
    so, sl = s9["outlier_small"]["summary"], s9["outlier_large"]["summary"]
    n["NineSmallGHB"], n["NineSmallRobust"], n["NineSmallOpt"] = (f"{so[f'{m} outlier']:.1f}" for m in ("GHB", "Robust GHB", "GHB + opt-out"))
    n["NineLargeGHB"], n["NineLargeRobust"] = f2(sl["GHB outlier"]), f2(sl["Robust GHB outlier"])
    s11 = json.load(open(R / "sim11_anchor.json"))["res"]
    n["ElevenHonestMax"] = f"{max(s11[e]['kl']['GHB, anchor uncertainty stated honestly'] for e in s11):.4f}"
    n["ElevenOverTwo"] = f"{s11['2.0']['kl']['GHB, anchor overconfident (stated 0.3)']:.4f}"
    n["ElevenCheckTwo"] = f"{s11['2.0']['kl']['GHB, overconfident + conflict check']:.4f}"
    n["ElevenLocalTwo"] = f"{s11['2.0']['kl']['Local']:.4f}"
    n["ElevenNoAnchorMax"] = f"{max(s11[e]['kl']['GHB, no anchor'] for e in s11):.4f}"
    s12 = json.load(open(R / "sim12_byzantine.json"))
    n["TwelveWorstGHB"] = f2(max(max(s12["out"][f]["GHB"]) for f in ("1", "3")))
    n["TwelveRobustLarge"] = f2(s12["out"]["3"]["Robust GHB"][-1])
    n["TwelveRobustNone"] = f2(s12["out"]["1"]["Robust GHB"][0])
    n["TwelveGHBNone"] = f2(s12["out"]["1"]["GHB"][0])
    s13 = json.load(open(R / "sim13_dropout.json"))
    n["ThirteenEighty"] = f2(s13["risk"]["0.8"]["60"]); n["ThirteenExact"] = f2(s13["exact"])
    n["ThirteenEightyTwenty"] = f2(s13["risk"]["0.8"]["20"])
    s14 = json.load(open(R / "sim14_accounting.json"))
    last = s14["rows"][-1]
    n["FourteenR"] = str(last["r"]); n["FourteenBasic"] = f"{last['B_basic']:.1f}"; n["FourteenZcdp"] = f"{last['B_zcdp']:.1f}"
    n["FourteenOne"] = f"{s14['rows'][0]['B_basic']:.1f}"
    s15 = json.load(open(R / "sim15_literature.json"))["res"]
    short15 = {"Local": "Local", "Local GD from anchor": "AnchorGD", "GHB (DP)": "GHB", "GHB, oracle tau2 (DP)": "Oracle",
               "GHB, earlier version (DP)": "Old",
               "Ditto-style (DP)": "Ditto", "DP-FedAvg": "Fedavg", "DP-FedAvg + fine-tuning": "FT",
               "Decentralized DP-SGD": "Dec", "Clustered DP-FedAvg": "Clus", "Global (pooled, no DP)": "Pooled"}
    tags15 = {"tau2=0.01": "Low", "tau2=0.05": "Mid", "tau2=0.2": "High", "tau2=0.5": "Extreme", "eps=0.3": "Strict",
              "eps=inf": "NoDP", "regional": "Reg", "anchor=1.0": "Anchor", "tau2=0.1,eps=0.5": "Mod", "K=48": "Large"}
    for key, tag in tags15.items():
        r15 = s15[key]
        for m, short in short15.items():
            n[f"Fifteen{tag}{short}"] = f"{r15['kl'][m]:.4f}"
            n[f"Fifteen{tag}{short}Large"] = f"{r15['kl_large'][m]:.4f}"
            n[f"Fifteen{tag}{short}Worst"] = f2(r15["worst_ratio"][m])
    private = [k for k in tags15 if k not in ("eps=inf", "tau2=0.5")]
    n["FifteenGHBWorstMax"] = f2(max(s15[k]["worst_ratio"]["GHB (DP)"] for k in private))
    n["FifteenFTWorstMin"] = f2(min(s15[k]["worst_ratio"]["DP-FedAvg + fine-tuning"] for k in private))
    n["FifteenFTWorstMax"] = f2(max(s15[k]["worst_ratio"]["DP-FedAvg + fine-tuning"] for k in private))
    n["FifteenReps"], n["FifteenTune"] = str(json.load(open(R / "sim15_literature.json"))["reps"]), str(json.load(open(R / "sim15_literature.json"))["tune"])
    lines = ["% Generated by simulations/make_numbers.py. Do not edit."]
    lines += [f"\\newcommand{{\\Sim{k}}}{{{v}}}" for k, v in n.items()]
    (R / "sim_numbers.tex").write_text("\n".join(lines) + "\n")

    # Table: excess log-loss (nats) and AUC of the logistic experiment
    rows = []
    for key, lab in [("tau2=0.01|eps=1.0", "0.01 & 1"), ("tau2=0.05|eps=1.0", "0.05 & 1"),
                     ("tau2=0.2|eps=1.0", "0.2 & 1"), ("tau2=0.5|eps=1.0", "0.5 & 1"),
                     ("tau2=0.05|eps=0.3", "0.05 & 0.3"), ("tau2=0.05|eps=3.0", "0.05 & 3"),
                     ("tau2=0.05|eps=10.0", "0.05 & 10"), ("tau2=0.01|eps=None", "0.01 & $\\infty$"),
                     ("tau2=0.05|eps=None", "0.05 & $\\infty$"), ("tau2=0.2|eps=None", "0.2 & $\\infty$"),
                     ("tau2=0.5|eps=None", "0.5 & $\\infty$")]:
        t = s7[key]
        kls = {k: t[k]["kl"] for k in ("Local", "Global", "GHB")}
        best = min(kls, key=kls.get)
        cells = [(f"\\textbf{{{kls[k]:.4f}}}" if k == best else f"{kls[k]:.4f}") for k in ("Local", "Global", "GHB")]
        aucs = [f"{t[k]['auc']:.3f}" for k in ("Local", "Global", "GHB", "Oracle")]
        rows.append(f"{lab} & " + " & ".join(cells) + " & " + " & ".join(aucs) + r"\\")
    (R / "s7_table.tex").write_text("% Generated by simulations/make_numbers.py. Do not edit.\n" + "\n".join(rows) + "\n")

    # Tables of Simulation 15: excess log-loss per setting, and the tuned hyperparameters
    labels15 = {"tau2=0.01": "$\\tau^2=0.01$", "tau2=0.05": "$\\tau^2=0.05$", "tau2=0.2": "$\\tau^2=0.2$",
                "tau2=0.5": "$\\tau^2=0.5$", "eps=0.3": "$\\epsilon=0.3$", "eps=inf": "no DP", "regional": "regional$^\\ast$",
                "anchor=1.0": "wrong anchor$^\\ast$", "tau2=0.1,eps=0.5": "$\\tau^2=0.1$, $\\epsilon=0.5^\\ast$", "K=48": "$K=48^\\ast$"}
    cols15 = ["Local", "Local GD from anchor", "GHB (DP)", "Ditto-style (DP)", "DP-FedAvg", "DP-FedAvg + fine-tuning",
              "Decentralized DP-SGD", "Clustered DP-FedAvg", "GHB, earlier version (DP)", "GHB, oracle tau2 (DP)", "Global (pooled, no DP)"]
    rows, hrows = [], []
    for key, lab in labels15.items():
        r15 = s15[key]
        comp = [m for m in cols15[:8]]
        best = min(comp, key=lambda m: r15["kl"][m])
        cells = [(f"\\textbf{{{r15['kl'][m]:.4f}}}" if m == best else f"{r15['kl'][m]:.4f}") for m in cols15]
        rows.append(f"{lab} & " + " & ".join(cells) + r"\\")
        hp = []
        for fam in ("DP-FedAvg + fine-tuning", "Decentralized DP-SGD", "Clustered DP-FedAvg", "Local GD from anchor"):
            cfg = r15["chosen"][fam].split("|")[1:]
            hp.append(", ".join(c.replace("eta=", "$\\eta$=") for c in cfg))
        hrows.append(f"{lab} & " + " & ".join(hp) + f" & {r15['ditto_lambda']:g}" + r"\\")
    (R / "s15_table.tex").write_text("% Generated by simulations/make_numbers.py. Do not edit.\n" + "\n".join(rows) + "\n")
    (R / "s15_hyper.tex").write_text("% Generated by simulations/make_numbers.py. Do not edit.\n" + "\n".join(hrows) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
