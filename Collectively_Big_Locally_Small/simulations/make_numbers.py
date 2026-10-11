"""Write the numbers quoted in the supplement and in Section 8 as LaTeX macros and tables.

Output: supplement/results/sim_numbers.tex (macros \\Sim...) and supplement/results/s7_table.tex.
"""
import json

import numpy as np

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
    # Simulation 16 (ablation) and 17 (sensitivity), when present
    p16 = R / "sim16_ablation.json"
    if p16.exists():
        s16 = json.load(open(p16))
        r16 = s16["res"]
        n["SixteenReps"] = str(s16["reps"])
        def rng16(m, keys=None):
            vals = [r16[c][m]["ratio"] for c in (keys or r16)]
            return f2(min(vals)), f2(max(vals))
        tags16 = {"Release: X'y instead of the score": "Xty", "Release: ridge fit instead of the score": "Ridge",
                  "No re-linearization": "NoRelin", "tau: plug-in posterior mean": "TauMean",
                  "tau: plug-in posterior median": "TauMedian", "tau: plug-in DerSimonian-Laird": "TauDL",
                  "tau: components without own evidence": "NoEvidence", "Anchor without conflict check": "NoCheck",
                  "No anchor": "NoAnchor", "No default prior": "NoPrior", "Equal weights in the network mean": "Equal",
                  "Classical Gaussian mechanism": "Classical", "Exact averaging (server; reference)": "Exact",
                  "tau known (oracle; reference)": "Oracle", "GHB, earlier version": "Old", "Local": "Local"}
        set16 = {"tau2=0.01": "Low", "tau2=0.05": "Mid", "tau2=0.2": "High", "tau2=0.5": "Extreme", "eps=0.3": "Strict",
                 "eps=inf": "NoDP", "anchor=1.0": "Anchor", "shift=1.0": "Shift"}
        for m, t in tags16.items():
            lo, hi = rng16(m)
            n[f"Sixteen{t}Min"], n[f"Sixteen{t}Max"] = lo, hi
            for c, ct in set16.items():
                n[f"Sixteen{t}{ct}"] = f2(r16[c][m]["ratio"])
        n["SixteenGHBWorstMax"] = f2(max(r16[c]["GHB (reported)"]["worst"] for c in r16 if c != "tau2=0.5"))
        g = s16["gaussian"]
        for key, tag in (("tau2=0.001|eps=0.3", "TinyStrict"), ("tau2=0.02|eps=0.3", "MidStrict"), ("tau2=0.02|eps=1.0", "Mid")):
            for m, t in (("tau integrated out (reported)", "FB"), ("tau: plug-in posterior mean", "Mean"),
                         ("tau: plug-in DerSimonian-Laird", "DL"), ("Equal weights in the network mean", "Equal"),
                         ("tau known (oracle)", "Oracle"), ("Gossip with 10 rounds", "TenRounds")):
                n[f"SixteenGauss{tag}{t}"] = f2(g[key][m]["mean"])
        rows = []
        lab16 = {"tau2=0.01": "$\\tau^2=0.01$", "tau2=0.05": "$\\tau^2=0.05$", "tau2=0.2": "$\\tau^2=0.2$", "tau2=0.5": "$\\tau^2=0.5$",
                 "eps=0.3": "$\\epsilon=0.3$", "eps=inf": "no DP", "anchor=1.0": "wrong anchor", "shift=1.0": "shift"}
        for m in tags16:
            name = m.replace("tau", "$\\tau$").replace("X'y", "$X^\\top y$")
            cells = []
            for c in lab16:
                o = r16[c][m]
                sig = abs(o["diff"]) > 2 * o["diff_se"] and m != "GHB (reported)"
                cells.append(f"\\textbf{{{o['ratio']:.2f}}}" if sig else f"{o['ratio']:.2f}")
            rows.append(f"{name} & " + " & ".join(cells) + r"\\")
        (R / "s16_table.tex").write_text("% Generated by simulations/make_numbers.py. Do not edit.\n" + "\n".join(rows) + "\n")
        gl = {"tau2=0.001|eps=0.3": "0.001 & 0.3", "tau2=0.02|eps=0.3": "0.02 & 0.3", "tau2=0.2|eps=0.3": "0.2 & 0.3",
              "tau2=0.001|eps=1.0": "0.001 & 1", "tau2=0.02|eps=1.0": "0.02 & 1", "tau2=0.2|eps=1.0": "0.2 & 1",
              "tau2=0.001|eps=None": "0.001 & $\\infty$", "tau2=0.02|eps=None": "0.02 & $\\infty$", "tau2=0.2|eps=None": "0.2 & $\\infty$"}
        gm = ["tau integrated out (reported)", "tau: plug-in posterior mean", "tau: plug-in DerSimonian-Laird",
              "tau: components without own evidence", "Equal weights in the network mean", "Exact averaging (server)",
              "Gossip with 10 rounds", "tau known (oracle)"]
        grows = [f"{lab} & " + " & ".join(f"{g[k][m]['mean']:.3f}" for m in gm) + r"\\" for k, lab in gl.items()]
        (R / "s16_gauss_table.tex").write_text("% Generated by simulations/make_numbers.py. Do not edit.\n" + "\n".join(grows) + "\n")
    p17 = R / "sim17_sensitivity.json"
    if p17.exists():
        s17 = json.load(open(p17))
        A, B = s17["A"], s17["B"]
        n["SeventeenReps"] = str(s17["reps"])
        for par, tag in (("Q", "Q"), ("scale", "Scale"), ("prior_sd", "Prior"), ("z", "Z"), ("anchor_sd", "AnchorSd"), ("T", "T"), ("m", "M")):
            vals = [o["ratio"] for lab in A for k, o in A[lab].items() if k.startswith(par + "|")]
            n[f"Seventeen{tag}Min"], n[f"Seventeen{tag}Max"] = f2(min(vals)), f2(max(vals))
        for key, o in B.items():
            cond, val = key.split("|")
            vt = {"inf": "Inf"}.get(val, val.replace(".", "p").replace("-", "m").replace("e", "E").replace(" ", "").replace("(", "").replace(")", "").replace("+", "").replace("=", ""))
            vt = "".join(ch for ch in vt if ch.isalnum())
            vt = vt.translate(str.maketrans("0123456789", "ABCDEFGHIJ"))
            n[f"Seventeen{cond.capitalize()}{vt}GHB"] = f2(o["GHB"]["rel"])
            n[f"Seventeen{cond.capitalize()}{vt}Worst"] = f2(o["GHB"]["worst"])
        n["SeventeenWorstMax"] = f2(max(o["GHB"]["worst"] for o in B.values()))
        for key, tag in (("K|6", "KSix"), ("K|96", "KNinetySix"), ("K|48", "KFortyEight"), ("median|25", "SizeTwentyFive"),
                         ("median|400", "SizeFourHundred"), ("eps|0.1", "EpsTenth"), ("eps|inf", "EpsInf"),
                         ("delta|0.001", "DeltaThree"), ("delta|1e-09", "DeltaNine"),
                         ("topology|Ring (degree 2)", "TopoRing"), ("topology|Regional clusters + bridges", "TopoRegional")):
            n[f"Seventeen{tag}"] = f2(B[key]["GHB"]["rel"])
            n[f"Seventeen{tag}Oracle"] = f2(B[key]["GHB, tau known"]["rel"])
        n["SeventeenRelMax"] = f2(max(o["GHB"]["rel"] for o in B.values()))
        n["SeventeenRelMin"] = f2(min(o["GHB"]["rel"] for o in B.values()))
    p18 = R / "sim18_fairness.json"
    if p18.exists():
        s18 = json.load(open(p18))
        n["EighteenReps"] = str(s18["reps"])
        tags = {"Local": "Local", "GHB (DP)": "GHB", "GHB, S local (DP)": "GHBS", "Global (pooled, no DP)": "Global",
                "FedAvg (no DP)": "FedAvg", "FAIR-FATE (no DP), F=SP": "FFSP", "FAIR-FATE (no DP), F=EQO": "FFEQO",
                "True model": "True"}
        for sc, res in s18["res"].items():
            S = sc.capitalize()
            for m, t in tags.items():
                o = res[m]
                if "kl" in o:
                    n[f"Eighteen{S}{t}All"] = f"{1000 * o['kl']:.1f}"
                    n[f"Eighteen{S}{t}Min"] = f"{1000 * o['kl1']:.1f}"
                    n[f"Eighteen{S}{t}Dis"] = f2(o["dis"])
                for key in ("SP", "EO", "EQO"):
                    n[f"Eighteen{S}{t}{key}"] = f2(o[key])
    pr = R / "real_heart.json"
    if pr.exists():
        rh = json.load(open(pr))
        n["RealHeartSplits"] = str(rh["repeats"] * rh["folds"])
        mt = {"Local": "Local", "GHB (DP)": "GHB", "GHB (DP, eps=5)": "GHBFive", "GHB (no DP)": "GHBNoDP",
              "GHB, empirical features (no DP)": "GHBEmp", "Global (pooled, no DP)": "Global", "Local GD, tuned stopping": "LGD",
              "DP-FedAvg + fine-tuning": "FT", "Ditto-style (DP)": "Ditto"}
        for tag, T in (("20", "Twenty"), ("40", "Forty"), ("80", "Eighty"), ("all", "All")):
            for m, t in mt.items():
                n[f"RealHeart{T}{t}Rel"] = f2(rh["res"][tag][m]["rel"])
                n[f"RealHeart{T}{t}Worst"] = f2(rh["res"][tag][m]["worst"])
            n[f"RealHeart{T}LocalLL"] = f"{rh['res'][tag]['Local']['ll_mean']:.3f}"
        n["RealHeartTwentyGHBNoDPGain"] = f"{100 * (1 - rh['res']['20']['GHB (no DP)']['rel']):.0f}\\%"
        n["RealHeartGlobalWorstMax"] = f2(max(rh["res"][t]["Global (pooled, no DP)"]["worst"] for t in ("20", "40", "80", "all")))
        n["RealHeartGlobalWorstMin"] = f2(min(rh["res"][t]["Global (pooled, no DP)"]["worst"] for t in ("20", "40", "80", "all")))
        n["RealHeartGHBWorstMax"] = f2(max(rh["res"][t][m]["worst"] for t in ("20", "40", "80", "all") for m in ("GHB (DP)", "GHB (DP, eps=5)")))
    pn = R / "real_networks.json"
    if pn.exists():
        rn = json.load(open(pn))["res"]
        tagn = {"medpar": "Medpar", "TCGA-LUSC": "Lusc", "TCGA-HNSC": "Hnsc", "TCGA-BLCA": "Blca", "TCGA-STAD": "Stad",
                "TCGA-LIHC": "Lihc", "TCGA-KIRC": "Kirc", "heart": "Heart"}
        mt = {"Local": "Local", "GHB (DP, eps=1)": "GHBOne", "GHB (DP, eps=3)": "GHBThree", "GHB (DP, eps=10)": "GHBTen",
              "GHB (no DP)": "GHBNoDP", "Global (pooled, no DP)": "Global", "Local GD, tuned stopping": "LGD"}
        for fam, t in (("DP-FedAvg", "Avg"), ("DP-FedAvg + fine-tuning", "FT"), ("Decentralized DP-SGD", "Dec"),
                       ("Clustered DP-FedAvg", "Clu"), ("Ditto-style (DP)", "Ditto"), ("Meta-analysis (DP)", "Meta")):
            mt[f"{fam}, eps=1"] = t + "One"
            mt[f"{fam}, eps=10"] = t + "Ten"
        g = lambda v: "n/a" if v != v else (f2(v) if v < 9.995 else f"{v:.1f}")
        for nm, T in tagn.items():
            if nm not in rn:
                continue
            r = rn[nm]
            n[f"RealNet{T}K"] = str(r["info"]["K"])
            n[f"RealNet{T}N"] = f"{sum(r['info']['n']):,}".replace(",", "{,}")
            n[f"RealNet{T}Median"] = f"{np.median(r['info']['n']):.0f}"
            for m, t in mt.items():
                n[f"RealNet{T}{t}Rel"] = g(r[m]["rel"])
                n[f"RealNet{T}{t}Small"] = g(r[m]["rel_small"])
                n[f"RealNet{T}{t}Worst"] = g(r[m]["worst"])
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
