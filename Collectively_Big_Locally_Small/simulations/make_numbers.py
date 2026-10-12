"""Write the numbers quoted in the supplement and in Section 8 as LaTeX macros and tables.

Output: supplement/results/sim_numbers.tex (macros \\Sim...) and supplement/results/s7_table.tex.
"""
import json

import numpy as np

import ghb

R = ghb.RESDIR


def f2(x):
    return f"{x:.2f}"


def tcga_numbers():
    """GHB on TCGA with the richer model: pre-registered confirmation (real_v4_confirm.json) and development."""
    c = json.load(open(R / "real_v4_confirm.json"))["rel"]
    fam = {"FT": "DP-FedAvg + FT", "FTSA": "DP-FedAvg-SA + FT", "APFL": "APFL", "Ditto": "Ditto", "PFedMe": "pFedMe",
           "Clu": "Clustered DP-FedAvg"}
    out = {}
    for e, E in (("1", "One"), ("10", "Ten")):
        r = c[e]
        g = np.array(r["GHB v4"])
        out[f"TC{E}GHB"] = f"{g.mean():.3f}"
        out[f"TC{E}Lite"] = f"{np.mean(r['GHB-EB']):.3f}"
        out[f"TC{E}Loc"] = f"{np.mean(r['Local, evidence prior']):.3f}"
        out[f"TC{E}GHBvsLocWins"] = str(int((g < np.array(r["Local, evidence prior"])).sum()))
        for k, fm in fam.items():
            v = np.array(r[fm + " (realistic)"])
            out[f"TC{E}{k}"] = f"{v.mean():.3f}"
            out[f"TC{E}{k}Wins"] = str(int((g < v).sum()))
            out[f"TC{E}{k}Orc"] = f"{np.mean(r[fm + ' (oracle)']):.3f}"
        orc = np.min([r[fm + " (oracle)"] for fm in fam.values()], 0)
        out[f"TC{E}BestOrc"] = f"{orc.mean():.3f}"
        out[f"TC{E}BestOrcWins"] = str(int((g < orc).sum()))
        out[f"TC{E}GHBMin"], out[f"TC{E}GHBMax"] = f"{g.min():.3f}", f"{g.max():.3f}"
    # development: twelve networks (literature tuned in-sample on the same networks)
    import glob
    from collections import defaultdict
    from real_v3 import GRIDS, ckey
    acc = defaultdict(lambda: [0.0, 0.0])
    for fn in glob.glob(str(R / "real_v3_ckpt" / "*.json")):
        if "/confirm_" not in fn:
            for k, (ll, cnt) in json.load(open(fn))["out"].items():
                acc[k][0] += sum(ll); acc[k][1] += sum(cnt)
    for fn in glob.glob(str(R / "v4c_ckpt" / "*.json")):
        for k, (ll, cnt) in json.load(open(fn)).items():
            if "v4c R1=1 +pan," in k:
                k2 = k.replace("v4c R1=1 +pan", "GHB v4")
                acc[k2][0] += sum(ll); acc[k2][1] += sum(cnt)
    dn = sorted({k.split("||")[0] for k in acc})
    L = lambda p, m: (acc[f"{p}||{m}"][0] / acc[f"{p}||{m}"][1]) / (acc[f"{p}||Local"][0] / acc[f"{p}||Local"][1])
    for e, E in (("1", "One"), ("10", "Ten")):
        out[f"TD{E}GHB"] = f"{np.mean([L(p, f'GHB v4, eps={e}') for p in dn]):.3f}"
        out[f"TD{E}Lite"] = f"{np.mean([L(p, f'GHB-EB, eps={e}') for p in dn]):.3f}"
        out[f"TD{E}Loc"] = f"{np.mean([L(p, 'Local, evidence prior') for p in dn]):.3f}"
        best = min(min(np.mean([L(p, f"{ckey(cf)}, eps={e}") for p in dn]) for cf in grid) for grid in GRIDS.values())
        out[f"TD{E}BestLit"] = f"{best:.3f}"
    return out


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
        if "GHB v4 (DP)" in kl:  # GHB (joint releases), added on its own random stream
            n[f"Eight{tag}V"] = f"{kl['GHB v4 (DP)']:.4f}"
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
    if "GHB v4 (DP)" in s11["2.0"]["kl"]:
        n["ElevenVTwo"] = f"{s11['2.0']['kl']['GHB v4 (DP)']:.4f}"
        n["ElevenVMax"] = f"{max(s11[e]['kl']['GHB v4 (DP)'] for e in s11):.4f}"
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
    short15 = {"Local": "Local", "Local GD from anchor": "AnchorGD", "GHB (DP)": "GHB", "GHB v4 (DP)": "V", "GHB, oracle tau2 (DP)": "Oracle",
               "GHB, earlier version (DP)": "Old",
               "Ditto-style (DP)": "Ditto", "DP-FedAvg": "Fedavg", "DP-FedAvg + fine-tuning": "FT",
               "Decentralized DP-SGD": "Dec", "Clustered DP-FedAvg": "Clus", "Global (pooled, no DP)": "Pooled"}
    tags15 = {"tau2=0.01": "Low", "tau2=0.05": "Mid", "tau2=0.2": "High", "tau2=0.5": "Extreme", "eps=0.3": "Strict",
              "eps=inf": "NoDP", "regional": "Reg", "anchor=1.0": "Anchor", "tau2=0.1,eps=0.5": "Mod", "K=48": "Large"}
    for key, tag in tags15.items():
        r15 = s15[key]
        for m, short in short15.items():
            if m not in r15["kl"]:
                continue
            n[f"Fifteen{tag}{short}"] = f"{r15['kl'][m]:.4f}"
            n[f"Fifteen{tag}{short}Large"] = f"{r15['kl_large'][m]:.4f}"
            n[f"Fifteen{tag}{short}Worst"] = f2(r15["worst_ratio"][m])
    private = [k for k in tags15 if k not in ("eps=inf", "tau2=0.5")]
    n["FifteenGHBWorstMax"] = f2(max(s15[k]["worst_ratio"]["GHB (DP)"] for k in private))
    if "GHB v4 (DP)" in s15["tau2=0.05"]["kl"]:
        n["FifteenVWorstMax"] = f2(max(s15[k]["worst_ratio"]["GHB v4 (DP)"] for k in private))
        iters = ("DP-FedAvg + fine-tuning", "Decentralized DP-SGD", "Clustered DP-FedAvg")
        n["FifteenVBeatsIter"] = str(sum(all(s15[k]["kl"]["GHB v4 (DP)"] < s15[k]["kl"][m] for m in iters) for k in tags15))
        n["FifteenVBeatsLite"] = str(sum(s15[k]["kl"]["GHB v4 (DP)"] < s15[k]["kl"]["GHB (DP)"] for k in tags15))
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
              "GHB (no DP)": "GHBNoDP", "Global (pooled, no DP)": "Global", "Local GD, tuned stopping": "LGD",
              "Local, tuned prior": "LocP", "GHB, tuned prior (no DP)": "GHBPNoDP"}
        for fam, t in (("GHB, tuned prior (DP)", "GHBP"), ("DP-FedAvg", "Avg"), ("DP-FedAvg + fine-tuning", "FT"), ("Decentralized DP-SGD", "Dec"),
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
    if (R / "real_networks.json").exists():
        import final_compare as FC
        rn = json.load(open(R / "real_networks.json"))["res"]
        if "cand" in rn.get("heart", {}) and any(k.startswith("pFedMe") for k in rn["heart"]):
            conf = FC.CONF
            tagm = {"Ditto": "Ditto", "pFedMe": "PFedMe", "DP-FedAvg + FT, sec. agg.": "FTSA", "APFL": "APFL",
                    "DP-FedAvg + FT": "FT", "Clustered DP-FedAvg": "Clu", "Decentralized DP-SGD": "Dec", "FedProx": "Prox",
                    "FedRep": "Rep", "DP-FedAvg": "Avg", "Per-FedAvg": "PerFed", "Ditto-style (one-shot)": "DittoOne"}
            for e, E in (("1", "One"), ("10", "Ten")):
                eb = np.array([rn[nm][f"GHB-EB {e}"]["rel"] for nm in conf])
                n[f"RF{E}EB"] = f"{eb.mean():.3f}"
                n[f"RF{E}SA"] = f"{np.mean([rn[nm][f'GHB-SA {e}']['rel'] for nm in conf]):.3f}"
                n[f"RF{E}GHB"] = f"{np.mean([rn[nm][f'GHB (DP, eps={e})']['rel'] for nm in conf]):.3f}"
                for arm, A in (("oracle", "Orc"), ("lono", "Lono"), ("uci", "Uci")):
                    n[f"RF{E}Loc{A}"] = f"{np.mean([FC.arm_value(rn, nm, 'Local, tuned prior', arm) for nm in conf]):.3f}"
                    n[f"RF{E}GHBP{A}"] = f"{np.mean([FC.arm_value(rn, nm, f'GHB, tuned prior (DP), eps={e}', arm) for nm in conf]):.3f}"
                    for lab, t in tagm.items():
                        v = np.array([FC.arm_value(rn, nm, FC.OTHERS[lab].format(e=e), arm) for nm in conf])
                        n[f"RF{E}{t}{A}"] = f"{v.mean():.3f}" if v.mean() < 9.995 else f"{v.mean():.1f}"
                        n[f"RF{E}{t}{A}Wins"] = str(int((eb < v).sum()))
                    best = [min(FC.arm_value(rn, nm, f.format(e=e), arm) for f in FC.OTHERS.values()) for nm in conf]
                    n[f"RF{E}Best{A}Wins"] = str(int((eb < np.array(best)).sum()))
                g0 = np.array([rn[nm][f"GHB (DP, eps={e})"]["rel"] for nm in conf])
                sa = np.array([rn[nm][f"GHB-SA {e}"]["rel"] for nm in conf])
                n[f"RF{E}EBvsGHBWins"] = str(int((eb < g0).sum()))
                n[f"RF{E}SAvsEBWins"] = str(int((sa < eb).sum()))
                n[f"RF{E}LiverWorstGHB"] = f2(rn["TCGA-LIHC"][f"GHB (DP, eps={e})"]["worst"])
                n[f"RF{E}LiverWorstEB"] = f2(rn["TCGA-LIHC"][f"GHB-EB {e}"]["worst"])
                n[f"RF{E}LiverWorstSA"] = f2(rn["TCGA-LIHC"][f"GHB-SA {e}"]["worst"])
                n[f"RF{E}EBMin"], n[f"RF{E}EBMax"] = f"{eb.min():.3f}", f"{eb.max():.3f}"
                locl = np.array([FC.arm_value(rn, nm, "Local, tuned prior", "lono") for nm in conf])
                n[f"RF{E}EBvsLocWins"] = str(int((eb < locl).sum()))
                if all(f"GHB v4 {e}" in rn[nm] for nm in conf):  # GHB (joint releases), on its own random stream
                    v4 = np.array([rn[nm][f"GHB v4 {e}"]["rel"] for nm in conf])
                    n[f"RF{E}V"], n[f"RF{E}VMin"], n[f"RF{E}VMax"] = f"{v4.mean():.3f}", f"{v4.min():.3f}", f"{v4.max():.3f}"
                    n[f"RF{E}VvsEBWins"] = str(int((v4 < eb).sum()))
                    n[f"RF{E}VvsLocWins"] = str(int((v4 < locl).sum()))
                    n[f"RF{E}LiverWorstV"] = f2(rn["TCGA-LIHC"][f"GHB v4 {e}"]["worst"])
                    n[f"RF{E}VWorstMax"] = f2(max(rn[nm][f"GHB v4 {e}"]["worst"] for nm in conf if rn[nm][f"GHB v4 {e}"]["worst"] == rn[nm][f"GHB v4 {e}"]["worst"]))
                    for arm, A in (("oracle", "Orc"), ("lono", "Lono")):
                        for lab, t in tagm.items():
                            v = np.array([FC.arm_value(rn, nm, FC.OTHERS[lab].format(e=e), arm) for nm in conf])
                            n[f"RF{E}V{t}{A}Wins"] = str(int((v4 < v).sum()))
                        best = [min(FC.arm_value(rn, nm, f.format(e=e), arm) for f in FC.OTHERS.values()) for nm in conf]
                        n[f"RF{E}VBest{A}Wins"] = str(int((v4 < np.array(best)).sum()))
    n.update(tcga_numbers())
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
