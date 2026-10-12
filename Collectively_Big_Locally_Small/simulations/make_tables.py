"""Write the result tables of the GHB paper and its supplement as LaTeX table bodies (rows only), from the saved
results of every simulation. Output: supplement/results/tables/*.tex. Each table is the exact-value companion of
a figure, so that no number exists only as a plotted point.

Logistic results are reported in units of 1e-3 nats of excess log-loss, with two Monte Carlo standard errors in
parentheses when the simulation records them; Gaussian results are relative risks R_k/v_k.
"""
import json

import numpy as np

import ghb

R = ghb.RESDIR
OUT = R / "tables"
OUT.mkdir(exist_ok=True)


def load(name):
    return json.load(open(R / name))


# Every literature method is cited wherever a table names it (first cell of a row).
CITES = [("FAIR-FATE", "salazar2023fairfate"), ("Per-FedAvg", "fallah2020personalized"), ("pFedMe", "tdinh2020personalized"),
         ("APFL", "deng2020adaptive"), ("FedRep", "collins2021exploiting"), ("FedProx", "li2020federated"),
         ("Ditto", "li2021ditto"), ("Clustered DP-FedAvg", "ghosh2020efficient,mcmahan2018learning"),
         ("Decentralized DP-SGD", "lian2017can"), ("sec.\\ agg.", "mcmahan2018learning,bonawitz2017practical"),
         ("DP-FedAvg", "mcmahan2018learning"), ("FedAvg", "mcmahan2017communication"), ("Meta-analysis", "dersimonian1986meta")]


def cite_row(row):
    if " & " not in row or "\\cite" in row.split(" & ")[0]:
        return row
    head, rest = row.split(" & ", 1)
    plain = head.lstrip()
    if plain.startswith(("GHB", "\\textbf{GHB", "Local", "Global", "Best", "\\multicolumn", "Pooled")):
        return row
    for name, key in CITES:
        if name in head:
            return f"{head} \\cite{{{key}}} & {rest}"
    return row


# Best values in bold: per table, groups of cell indices (0 = row label) compared along a row ("row") or down a column
# within each block of rows ("col"), lower or higher is better; rows whose label starts with a skip prefix are not
# candidates (references such as local estimation, non-private or oracle rows).
BOLD = {
    "t_real_final.tex": [("colall", [1, 2, 4, 5], "min", ("Local",))],
    "t_tcga_tuned.tex": [("colall", [1, 2, 3, 4], "min", ("GHB, frozen",))],
    "s_tcga_dev.tex": [("colall", [1, 2, 3, 4], "min", ("Local",))],
    "s_tcga_oracle.tex": [("col", [1, 2, 3, 4, 5, 6], "min", ())],
    "s_tcga_ablation.tex": [("col", [1, 3], "min", ())],
    "t_real_heart.tex": [("colall", [1, 3, 5, 7], "min", ("Local", "Global", "dagger")),
                         ("colall", [2, 4, 6, 8], "min", ("Local", "Global", "dagger"))],
    "s_real_heart_sites.tex": [("col", list(range(1, 9)), "min", ("Global", "dagger")),
                               ("col", [9, 10, 11, 12], "max", ("Global", "dagger"))],
    "s_sim11.tex": [("row", [1, 2, 3, 4, 5], "min", ())],
    "s_sim12.tex": [("row", [1, 2], "min", ()), ("row", [4, 5], "min", ())],
    "s_sim1_risk.tex": [("row", [1, 2, 3, 4, 5], "min", ())],
    "s_sim2_eps1.tex": [("row", [1, 3, 5, 7, 9], "min", ()), ("row", [2, 4, 6, 8, 10], "min", ())],
    "s_sim2_nodp.tex": [("row", [1, 3, 5, 7, 9], "min", ()), ("row", [2, 4, 6, 8, 10], "min", ())],
    "s_sim3.tex": [("row", [1, 2, 3], "min", ())],
    "s_sim4.tex": [("row", [1, 2, 3, 4], "min", ())],
    "s_sim6.tex": [("row", [1, 2], "min", ()), ("row", [3, 4], "min", ())],
    "s_sim7.tex": [("row", [2, 3, 4], "min", ()), ("row", [5, 6, 7], "max", ()), ("row", [9, 10], "min", ())],
    "s_sim9.tex": [("row", [1, 2, 3], "min", ()), ("row", [4, 5, 6], "min", ())],
    "s_sim9_bins.tex": [("row", [1, 2, 3], "min", ())],
    "s_sim14.tex": [("row", [1, 2], "max", ())],
}


def _num(cell):
    import re
    m = re.match(r"\s*(?:\\textbf\{)?\s*(-?\d+(?:\.\d+)?)", cell)
    return float(m.group(1)) if m else None


def _embolden(cell):
    import re
    if "\\textbf" in cell:
        return cell
    return re.sub(r"^(\s*)(-?\d+(?:\.\d+)?)", r"\1\\textbf{\2}", cell, count=1)


def bold_rows(name, rows):
    specs = BOLD.get(name)
    if not specs:
        return rows
    split = [r[:-2].split(" & ") if r.endswith("\\\\") and " & " in r else None for r in rows]
    for mode, cols, sense, skip in specs:
        pick = min if sense == "min" else max
        if mode == "row":
            for c in split:
                if c is None or any(k in c[0] for k in skip):
                    continue
                vals = {i: _num(c[i]) for i in cols if i < len(c) and _num(c[i]) is not None}
                if vals:
                    b = pick(vals.values())
                    for i, v in vals.items():
                        if v == b:
                            c[i] = _embolden(c[i])
        else:
            blocks, cur = [], []
            for c in split:
                if c is None and mode == "col":
                    if cur:
                        blocks.append(cur); cur = []
                elif c is not None:
                    cur.append(c)
            if cur:
                blocks.append(cur)
            for blk in blocks:
                cand = [c for c in blk if not any(k in c[0] for k in skip)]
                for i in cols:
                    vals = [(c, _num(c[i])) for c in cand if i < len(c) and _num(c[i]) is not None]
                    if vals:
                        b = pick(v for _, v in vals)
                        for c, v in vals:
                            if v == b:
                                c[i] = _embolden(c[i])
    return [(" & ".join(c) + "\\\\") if c is not None else r for r, c in zip(rows, split)]


V4_TABLES = ("t_real_final", "t_tcga_", "s_tcga_", "s_real_final_lono")  # tables that already use the paper's names


def write(name, rows):
    if not name.startswith(V4_TABLES):  # these tables report the per-hospital protocol: GHB-lite in the paper
        import re
        def ren(r):
            if " & " not in r:
                return r
            h, rest = r.split(" & ", 1)
            return re.sub(r"(?<![A-Za-z\\])GHB(?![A-Za-z-])", "GHB-lite", h) + " & " + rest
        rows = [ren(r) for r in rows]
    rows = bold_rows(name, rows)
    rows = [cite_row(r) for r in rows]
    (OUT / name).write_text("% Generated by simulations/make_tables.py. Do not edit.\n" + "\n".join(rows) + "\n")


def milli(x, se=None, bold=False):
    s = f"{1000 * x:.1f}"
    if bold:
        s = f"\\textbf{{{s}}}"
    if se is not None:
        s += f" \\textcolor{{gray}}{{\\tiny({2000 * se:.1f})}}"
    return s


def f(x, d=2, bold=False):
    s = f"{x:.{d}f}"
    return f"\\textbf{{{s}}}" if bold else s


def tex_tau(lab):
    return lab.replace("tau2=", "$\\tau^2=") + "$" if lab.startswith("tau2") else lab


# ---------------------------------------------------------------- main paper
def sim8():
    s = load("sim8_baselines.json")
    methods = ["Local", "GHB v4 (DP)", "GHB (DP)", "Ditto-style (DP)", "Meta-analysis (DP)", "FedAvg (DP)", "Global (pooled, no DP)"]
    private = methods[:6]
    rows = []
    for key, lab in (("het", "$\\tau^2={}$"), ("shift", "shift ${}$")):
        for x, r in s[key].items():
            if key == "shift" and x == "0.0":
                continue
            best = min(private, key=lambda m: r["kl"][m])
            rows.append(lab.format(x) + " & " + " & ".join(milli(r["kl"][m], r["kl_se"][m], m == best) for m in methods) + r"\\")
        if key == "het":
            rows.append(r"\midrule")
    write("t_sim8.tex", rows)


S15_COLS = ["Local", "Local GD from anchor", "GHB v4 (DP)", "GHB (DP)", "Ditto-style (DP)", "DP-FedAvg + fine-tuning",
            "Decentralized DP-SGD", "Clustered DP-FedAvg", "GHB, oracle tau2 (DP)", "Global (pooled, no DP)"]
S15_PRIVATE = S15_COLS[:8]
S15_LAB = {"tau2=0.01": "$\\tau^2=0.01$", "tau2=0.05": "$\\tau^2=0.05$", "tau2=0.2": "$\\tau^2=0.2$", "tau2=0.5": "$\\tau^2=0.5$",
           "eps=0.3": "$\\epsilon=0.3$", "eps=inf": "no DP", "regional": "regional$^\\ast$", "anchor=1.0": "wrong anchor$^\\ast$",
           "tau2=0.1,eps=0.5": "$\\tau^2=0.1$, $\\epsilon=0.5^\\ast$", "K=48": "$K=48^\\ast$"}


def sim15():
    s = load("sim15_literature.json")["res"]
    if "GHB v4 (DP)" not in s["tau2=0.05"]["kl"]:  # results from before GHB was added: keep the earlier tables
        return
    rows, wrows = [], []
    for key, lab in S15_LAB.items():
        r = s[key]
        best = min(S15_PRIVATE, key=lambda m: r["kl"][m])
        rows.append(lab + " & " + " & ".join(milli(r["kl"][m], r["kl_se"][m], m == best) for m in S15_COLS) + r"\\")
        wbest = min(S15_PRIVATE, key=lambda m: r["worst_ratio"][m])
        wrows.append(lab + " & " + " & ".join(f(r["worst_ratio"][m], 2, m == wbest) for m in S15_PRIVATE) + " & "
                     + " & ".join(f"{100 * r['share_better'][m]:.0f}" for m in ("GHB (DP)", "Ditto-style (DP)", "DP-FedAvg + fine-tuning"))
                     + " & " + " & ".join(milli(r["kl_large"][m]) for m in ("Local", "GHB (DP)", "DP-FedAvg + fine-tuning")) + r"\\")
    write("t_sim15_mean.tex", rows)
    write("t_sim15_worst.tex", wrows)


def sim16():
    s = load("sim16_ablation.json")
    res = s["res"]
    cols = ["tau2=0.01", "tau2=0.05", "tau2=0.2", "tau2=0.5", "eps=0.3", "eps=inf", "anchor=1.0", "shift=1.0"]
    rows = ["Reported GHB ($10^{-3}$ nats) & " + " & ".join(milli(res[c]["GHB (reported)"]["kl"]) for c in cols) + r"\\", r"\midrule"]
    names = [m for m in res[cols[0]] if m != "GHB (reported)"]
    for m in names:
        cells = []
        for c in cols:
            o = res[c][m]
            sig = abs(o["diff"]) > 2 * o["diff_se"]
            cells.append(f"\\textbf{{{o['ratio']:.2f}}}" if sig else f"{o['ratio']:.2f}")
        name = m.replace("tau", "$\\tau$").replace("X'y", "$X^\\top y$")
        rows.append(name + " & " + " & ".join(cells) + r"\\")
        if m in ("Classical Gaussian mechanism",):
            rows.append(r"\midrule")
    write("t_sim16.tex", rows)


PAR_LAB = {"Q": "Quantiles of $\\tau$, $Q$", "scale": "Half-Cauchy scale of $\\tau$", "prior_sd": "Default prior s.d.",
           "z": "Conflict threshold $z$", "anchor_sd": "Stated anchor s.d.", "T": "Gossip rounds $T$", "m": "Public sample size"}
DEF = {"Q": "12", "scale": "0.5", "prior_sd": "2.5", "z": "3.0", "anchor_sd": "0.3", "T": "60", "m": "20000"}


def sim17():
    s = load("sim17_sensitivity.json")
    A, B = s["A"], s["B"]
    bases = list(A)
    rows = []
    order = []
    for k in A[bases[0]]:
        par, val = k.split("|")
        if par == "local":
            continue
        order.append((par, val))
    last = None
    for par, val in order:
        if last is not None and par != last:
            rows.append(r"\addlinespace")
        lab = PAR_LAB[par] if par != last else ""
        last = par
        vlab = "none" if val == "None" else val
        if val == DEF[par]:
            vlab = f"\\textit{{{vlab}}}"
        cells = []
        for b in bases:
            o = A[b][f"{par}|{val}"]
            cells.append("1" if val == DEF[par] else f"{o['ratio']:.3f} \\textcolor{{gray}}{{\\tiny({2 * o['ratio_se']:.3f})}}")
        rows.append(f"{lab} & {vlab} & " + " & ".join(cells) + r"\\")
    write("t_sim17_settings.tex", rows)
    names = {"K": "Hospitals $K$", "median": "Median size", "eps": "Budget $\\epsilon$", "delta": "$\\delta$", "topology": "Topology"}
    topo = {"Ring (degree 2)": "ring", "Regional clusters + bridges": "regional", "Sparse random (p=0.15)": "sparse random", "Ring + 2-hop (degree 4)": "ring, two-hop"}
    rows, last = [], None
    for key, o in B.items():
        cond, val = key.split("|")
        if last is not None and cond != last:
            rows.append(r"\addlinespace")
        lab = names[cond] if cond != last else ""
        last = cond
        v = topo.get(val, "no DP" if val == "inf" else val)
        rows.append(f"{lab} & {v} & " + f"{o['GHB']['rel']:.3f} \\textcolor{{gray}}{{\\tiny({2 * o['GHB']['rel_se']:.3f})}} & "
                    + f"{o['GHB, tau known']['rel']:.3f} & {o['GHB']['worst']:.2f} & {milli(o['GHB']['kl'])} & {milli(o['Local']['kl'])}" + r"\\")
    write("t_sim17_conditions.tex", rows)


# ---------------------------------------------------------------- supplement
def sim1():
    s = load("s1_validate.json")
    eps_s = lambda e: "inf" if e is None else f"{e:g}"
    cell = lambda r, d: f"{r['sim']:.{d}f} \\textcolor{{gray}}{{\\tiny({r['theory']:.{d}f})}}"
    cases = [("local", "inf"), ("pooling", "inf"), ("hb", "inf"), ("hb", "1"), ("hb", "0.1")]
    by = {}
    for r in s["rows"]:
        by.setdefault(int(r["nk"]), {})[(r["method"], eps_s(r["eps"]))] = r
    rows = [f"{nk:,} & ".replace(",", "{,}") + " & ".join(cell(d[c], 4) for c in cases) + r"\\" for nk, d in by.items()]
    write("s_sim1_risk.tex", rows)
    by, eps = {}, []
    for r in s["grid"]:
        by.setdefault(r["nj"], {})[eps_s(r["eps"])] = r
        if eps_s(r["eps"]) not in eps:
            eps.append(eps_s(r["eps"]))
    fmt = lambda r: f"{r['sim']:.3g} \\textcolor{{gray}}{{\\tiny({r['theory']:.3g})}}"
    rows = [f"{nj:,} & ".replace(",", "{,}") + " & ".join(fmt(d[e]) for e in eps) + r"\\" for nj, d in by.items()]
    write("s_sim1_M.tex", rows)


def sim2():
    s = load("s2_heterogeneity.json")
    t2 = s["tau2"]
    methods = ["Global model (full pooling)", "GHB, oracle $\\tau^2$, exact mean", "GHB, plug-in, exact mean",
               "GHB, plug-in, gossip T=60", "GHB, tau integrated out, gossip T=60"]
    for eps, name in (("1.0", "s_sim2_eps1.tex"), ("None", "s_sim2_nodp.tex")):
        rows = []
        for i, t in enumerate(t2):
            cells = []
            for m in methods:
                a = np.array(s["curves"][eps][m][i])
                cells += [f(a.mean(), 3), f(a.max(), 2)]
            rows.append(f"{t:.2g} & " + " & ".join(cells) + r"\\")
        write(name, rows)


def sim3():
    s = load("s3_equity.json")["summary"]
    rows = [f"{K} & " + " & ".join(f(s["curves"][m][i], 3) for m in ("oracle", "plug-in", "bayes")) + r"\\" for i, K in enumerate(s["K"])]
    write("s_sim3.tex", rows)


def sim4():
    s = load("s4_gossip.json")
    names = list(s["res"])
    rows = [f"{T} & " + " & ".join(f(s["res"][nm]["curve"][i], 3) for nm in names) + r"\\" for i, T in enumerate(s["T"])]
    rows.append(r"\midrule")
    rows.append("$\\lambda$ & " + " & ".join(f(s["res"][nm]["lam"], 3) for nm in names) + r"\\")
    rows.append("rounds to 2\\% & " + " & ".join(str(s["res"][nm]["t_within_2pct"]) for nm in names) + r"\\")
    rows.append("bound, $\\gamma=0.01$ & " + " & ".join(f"{s['res'][nm]['t_pred']:.0f}" for nm in names) + r"\\")
    rows.append(f"exact averaging & \\multicolumn{{{len(names)}}}{{c}}{{{s['exact']:.3f}}}" + r"\\")
    write("s_sim4.tex", rows)


def sim5():
    from exp5_joint_release import TAU2 as TAU5
    s = load("s5_joint_release.json")
    rows = []
    for net, lab in (("small", "all small"), ("mixed", "mixed")):
        res = s["out"][net]["res"]
        for i, e in enumerate(s["eps"]):
            cells = [f"{1 / (TAU5 + res[g]['sim'][i]):.1f} ({1 / (TAU5 + res[g]['theory'][i]):.1f})" for g in ("1", "2", "4", "8")]
            rows.append(f"{lab if i == 0 else ''} & {e:.3g} & " + " & ".join(cells) + r"\\")
        rows.append(r"\addlinespace")
    write("s_sim5.tex", rows[:-1])


def sim6():
    s = load("s6_gmrf.json")
    rows = [f"{t:.2g} & {f(s['ghb'][i], 3)} & {f(s['gmrf'][i], 3)} & {f(s['ghb_worst'][i], 2)} & {f(s['gmrf_worst'][i], 2)}" + r"\\"
            for i, t in enumerate(s["tau2_r"])]
    write("s_sim6.tex", rows)


def sim7():
    s = load("s7_logistic.json")["table"]
    rows = []
    for key, r in s.items():
        t2, e = key.split("|")
        t2 = t2.split("=")[1]
        e = e.split("=")[1]
        e = "$\\infty$" if e == "None" else e
        rows.append(f"{t2} & {e} & " + " & ".join(milli(r[m]["kl"]) for m in ("Local", "Global", "GHB")) + " & "
                    + " & ".join(f(r[m]["auc"], 3) for m in ("Local", "Global", "GHB", "Oracle")) + " & "
                    + f(r["GHB"]["kl_worst_ratio"], 2) + " & " + f(r["Global"]["kl_worst_ratio"], 2) + r"\\")
    write("s_sim7.tex", rows)


def sim9():
    s = load("sim9_atypical.json")
    rows = []
    lab = {"gauss": "Gaussian effects", "t3": "Student-$t_3$ effects", "outlier_small": "outlier at the smallest hospital",
           "outlier_large": "outlier at the largest hospital"}
    for k, v in s["out"].items():
        sm = v["summary"]
        if k.startswith("outlier"):
            cells = [f(sm[f"{m} outlier"], 2) for m in ("GHB", "Robust GHB", "GHB + opt-out")] + [f(sm[f"{m} others"], 3) for m in ("GHB", "Robust GHB", "GHB + opt-out")]
        else:
            cells = [f(sm[m], 3) for m in ("GHB", "Robust GHB", "GHB + opt-out")] + ["", "", ""]
        rows.append(lab[k] + " & " + " & ".join(cells) + r"\\")
    write("s_sim9.tex", rows)
    rows = []
    b = s["bins"]
    binned = s["out"]["t3"]["binned"]
    for i in range(len(binned["GHB"]["y"])):
        lo, hi = b[i], (b[i + 1] if i + 1 < len(b) else None)
        rng = f"{lo:g}-{hi:g}" if hi is not None else f"$\\ge{lo:g}$"
        cells = []
        for m in ("GHB", "Robust GHB", "GHB + opt-out"):
            y, e = binned[m]["y"][i], binned[m]["e"][i]
            cells.append("n/a" if y is None or np.isnan(y) else f"{y:.2f} \\textcolor{{gray}}{{\\tiny({e:.2f})}}")
        rows.append(rng + " & " + " & ".join(cells) + f" & {binned['GHB']['n'][i]:,}".replace(",", "{,}") + r"\\")
    write("s_sim9_bins.tex", rows)


def sim11():
    s = load("sim11_anchor.json")["res"]
    ms = ["Local", "GHB, anchor uncertainty stated honestly", "GHB, anchor overconfident (stated 0.3)", "GHB, overconfident + conflict check", "GHB, no anchor"]
    rows = [f"{e} & " + " & ".join(milli(r["kl"][m], r["se"][m]) for m in ms) + f" & {100 * r['dropped']:.0f}" + r"\\" for e, r in s.items()]
    write("s_sim11.tex", rows)


def sim12():
    s = load("sim12_byzantine.json")
    rows = []
    for i, b in enumerate(s["b"]):
        cells = []
        for k in ("1", "3"):
            o = s["out"][k]
            cells += [f(o["GHB"][i], 3), f(o["Robust GHB"][i], 3), f"{100 * o['kept_bad'][i]:.0f}"]
        rows.append(f"{b} & " + " & ".join(cells) + r"\\")
    write("s_sim12.tex", rows)


def sim13():
    s = load("sim13_dropout.json")
    rows = [f"{T} & " + " & ".join(f(s["risk"][p][str(T)], 3) for p in ("0.0", "0.2", "0.5", "0.8")) + r"\\" for T in s["T"]]
    rows.append(f"exact & \\multicolumn{{4}}{{c}}{{{s['exact']:.3f}}}" + r"\\")
    write("s_sim13.tex", rows)


def sim14():
    s = load("sim14_accounting.json")
    rows = [f"{r['r']} & {r['B_basic']:.1f} & {r['B_zcdp']:.1f} & {r['noise_ratio']:.2f}" + r"\\" for r in s["rows"]]
    rows.append(f"no privacy & \\multicolumn{{3}}{{c}}{{$B={s['B_none']:.1f}$}}" + r"\\")
    write("s_sim14.tex", rows)


def sim18():
    s = load("sim18_fairness.json")["res"]
    names = {"Local": "Local", "GHB (DP)": "GHB", "GHB, S local (DP)": "GHB, $S$ local", "Global (pooled, no DP)": "Global$^\\dagger$",
             "FedAvg (no DP)": "FedAvg$^\\dagger$", "FAIR-FATE (no DP), F=SP": "FAIR-FATE, SP$^\\dagger$",
             "FAIR-FATE (no DP), F=EQO": "FAIR-FATE, EQO$^\\dagger$"}
    titles = {"shared": "Shared disparity ($\\tau_S^2=0.05$)", "varies": "Disparity varies ($\\tau_S^2=0.5$)",
              "rare": "Rare minority (share 0.1)", "opposite": "Opposite disparities ($\\mu_S=0$, $\\tau_S^2=0.5$)"}
    rows = []
    for i, (sc, title) in enumerate(titles.items()):
        r = s[sc]
        if i:
            rows.append(r"\midrule")
        rows.append(f"\\multicolumn{{10}}{{@{{}}l}}{{\\emph{{{title}}}}}" + r"\\")
        priv = ["Local", "GHB (DP)", "GHB, S local (DP)"]
        best1 = min(priv, key=lambda m: r[m]["kl1"])
        bestd = min(priv, key=lambda m: r[m]["dis"])
        for m, lab in names.items():
            o = r[m]
            rows.append(f"{lab} & {milli(o['kl'], o['kl_se'])} & {milli(o['kl1'], o['kl1_se'], m == best1)} & {milli(o['kl0'])} & "
                        + (f"\\textbf{{{o['dis']:.3f}}}" if m == bestd else f"{o['dis']:.3f}")
                        + f" & {100 * o['cal']:.1f} & {o['worst']:.2f} & {o['SP']:.2f} & {o['EO']:.2f} & {o['EQO']:.2f}" + r"\\")
        t = r["True model"]
        rows.append(f"True models & & & & & & & {t['SP']:.2f} & {t['EO']:.2f} & {t['EQO']:.2f}" + r"\\")
    write("t_sim18.tex", rows)


def real_heart():
    s = load("real_heart.json")
    res = s["res"]
    tags = ("20", "40", "80", "all")
    names = {"Local": "Local", "GHB (DP)": "GHB, $\\epsilon=1$", "GHB (DP, eps=5)": "GHB, $\\epsilon=5$",
             "DP-FedAvg + fine-tuning": "DP-FedAvg + FT, $\\epsilon=1$", "Ditto-style (DP)": "Ditto-style, $\\epsilon=1$",
             "Local GD, tuned stopping": "Local GD, tuned stopping", "GHB (no DP)": "GHB, no DP$^\\dagger$",
             "GHB, empirical features (no DP)": "GHB, empirical features$^\\dagger$", "Global (pooled, no DP)": "Global$^\\dagger$"}
    rows = ["Local log-loss & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{res[t]['Local']['ll_mean']:.3f}}}" for t in tags) + r"\\",
            r"\midrule"]
    for m, lab in names.items():
        if m == "Local GD, tuned stopping":
            rows.append(r"\midrule")
        if m == "GHB (no DP)":
            rows.append(r"\midrule")
        cells = []
        for t in tags:
            r = res[t][m]
            cells.append(f"{r['rel']:.3f} \\textcolor{{gray}}{{\\tiny({2 * r['rel_paired_se']:.3f})}}")
            cells.append(f"{r['worst']:.2f}")
        rows.append(lab + " & " + " & ".join(cells) + r"\\")
    write("t_real_heart.tex", rows)
    rows = []
    sites = s["info"]["sites"]
    for t, lab in (("40", "at most 40 training patients"), ("all", "all training patients")):
        rows.append(f"\\multicolumn{{13}}{{@{{}}l}}{{\\emph{{{lab[0].upper() + lab[1:]}}}}}" + r"\\")
        for m, name in names.items():
            r = res[t][m]
            cells = [f"{r['ll'][k]:.3f}" for k in range(len(sites))] + [f"{r['brier'][k]:.3f}" for k in range(len(sites))] \
                + [f"{r['auc'][k]:.3f}" for k in range(len(sites))]
            rows.append(name + " & " + " & ".join(cells) + r"\\")
        if t == "40":
            rows.append(r"\midrule")
    write("s_real_heart_sites.tex", rows)
    rows = []
    for t in tags:
        ch = res[t]["chosen"]
        f = lambda c: c.split("|", 1)[1].replace("eta=", "$\\eta$=").replace("lambda=", "$\\lambda$=").replace("|", ", ")
        rows.append(f"{'all' if t == 'all' else t} & {f(ch['Local GD, tuned stopping'])} & {f(ch['DP-FedAvg + fine-tuning'])} & {f(ch['Ditto-style (DP)'])}" + r"\\")
    write("s_real_heart_hyper.tex", rows)


REAL_ORDER = ["medpar", "TCGA-LUSC", "TCGA-HNSC", "TCGA-BLCA", "TCGA-STAD", "TCGA-LIHC", "TCGA-KIRC", "heart"]
REAL_TUNED = [("GHB, tuned prior", "GHB, tuned prior (DP), eps={e}"), ("Local, tuned prior$^\\ddagger$", "Local, tuned prior"),
              ("Local GD, tuned stopping$^\\ddagger$", "Local GD, tuned stopping"), ("DP-FedAvg", "DP-FedAvg, eps={e}"),
              ("DP-FedAvg + FT", "DP-FedAvg + fine-tuning, eps={e}"), ("DP-FedAvg + FT, sec.\\ agg.", "DP-FedAvg-SA + fine-tuning, eps={e}"),
              ("Decentralized DP-SGD", "Decentralized DP-SGD, eps={e}"), ("Clustered DP-FedAvg", "Clustered DP-FedAvg, eps={e}"),
              ("Ditto-style", "Ditto-style (DP), eps={e}")]


def real_realistic(s, nm, fam):
    """The arm tuned on the development network (UCI Heart Disease) and frozen: the candidate best there, evaluated here."""
    import real_networks as RN
    best = min(RN.TUNED[fam], key=lambda c: s["heart"]["cand"][c]["rel"])
    return s[nm]["cand"][best]


def real_networks():
    s = load("real_networks.json")["res"]
    order = [k for k in REAL_ORDER if k in s]
    fmt = lambda v: "n/a" if v != v else (f"{v:.3f}" if v < 9.995 else f"{v:.1f}")
    for e in ("1", "10"):
        for key, suffix in (("rel", ""), ("rel_small", "_small"), ("worst", "_worst")):
            untuned = [("GHB", f"GHB (DP, eps={e})"), ("GHB-EB", f"GHB-EB {e}"), ("GHB-SA (sec.\\ agg.)", f"GHB-SA {e}")]
            vals = {}
            for lab, m in untuned:
                vals[("u", lab)] = [s[nm][m][key] for nm in order]
            for lab, m in REAL_TUNED:
                f = m.format(e=e)
                vals[("r", lab)] = [real_realistic(s, nm, f)[key] for nm in order]
                vals[("o", lab)] = [s[nm][f][key] for nm in order]
            # bold: best private method a hospital could actually run (no tuning, or tuned on the development network)
            best = [min(v[j] for (arm, _), v in vals.items() if arm in ("u", "r") and v[j] == v[j]) for j in range(len(order))]
            rows = ["Hospitals $K$ & " + " & ".join(str(s[nm]["info"]["K"]) for nm in order) + r"\\",
                    "Local log-loss (nats) & " + " & ".join(f"{s[nm]['Local']['ll']:.3f}" for nm in order) + r"\\", r"\midrule"]
            for arm, title in (("u", "No tuning"), ("r", "Tuned on UCI Heart Disease, then frozen (realistic)"),
                               ("o", "Tuned on each network's own held-out folds (oracle; upper bound)")):
                rows.append(f"\\multicolumn{{{len(order) + 1}}}{{@{{}}l}}{{\\emph{{{title}}}}}" + r"\\")
                for (a2, lab), v in vals.items():
                    if a2 != arm:
                        continue
                    cells = [(f"\\textbf{{{fmt(x)}}}" if key != "worst" and arm != "o" and x == best[j] else fmt(x)) for j, x in enumerate(v)]
                    rows.append(lab + " & " + " & ".join(cells) + r"\\")
                rows.append(r"\midrule")
            if e == "1":
                rows.append(f"\\multicolumn{{{len(order) + 1}}}{{@{{}}l}}{{\\emph{{No privacy}}}}" + r"\\")
                for lab, m in (("GHB", "GHB (no DP)"), ("GHB-EB", "GHB-EB inf"), ("Global (pooled)", "Global (pooled, no DP)")):
                    rows.append(lab + " & " + " & ".join(fmt(s[nm][m][key]) for nm in order) + r"\\")
            else:
                rows = rows[:-1]
            name = ("t" if (e == "1" and key == "rel") else "s") + f"_real_networks_eps{e}{suffix}.tex"
            write(name, rows)
    # chosen configurations: oracle per network, and the frozen realistic one
    import real_networks as RN
    rows = []
    for e in ("1", "10"):
        for lab, m in REAL_TUNED:
            f = m.format(e=e)
            g = lambda c: c.split("|", 1)[1].replace("eta=", "$\\eta$=").replace("lambda=", "$\\lambda$=").replace("sd=", "s.d.=").replace("|", ", ")
            real = min(RN.TUNED[f], key=lambda c: s["heart"]["cand"][c]["rel"])
            orc = sorted({s[nm]["chosen"][f] for nm in order if nm != "heart"})
            rows.append(f"{e} & {lab} & {g(real)} & {'; '.join(g(c) for c in orc)}" + r"\\")
        rows.append(r"\addlinespace")
    write("s_real_networks_hyper.tex", rows[:-1])


def real_final():
    """The final real-data comparison (final_compare.py arms): a compact main table and per-network tables."""
    import final_compare as FC
    s = load("real_networks.json")["res"]
    conf = FC.CONF
    eb = {e: np.array([s[nm][f"GHB v4 {e}"]["rel"] for nm in conf]) for e in ("1", "10")}  # wins are GHB's
    rows = []
    head = [(r"\textbf{GHB}, no tuning", lambda e, arm, nm: s[nm][f"GHB v4 {e}"]["rel"]),
            ("GHB-lite, no tuning", lambda e, arm, nm: s[nm][f"GHB (DP, eps={e})"]["rel"]),
            ("GHB-lite, evidence prior, no tuning", lambda e, arm, nm: s[nm][f"GHB-EB {e}"]["rel"]),
            ("GHB-lite with secure aggregation, no tuning", lambda e, arm, nm: s[nm][f"GHB-SA {e}"]["rel"]),
            ("GHB-lite, tuned prior", lambda e, arm, nm: FC.arm_value(s, nm, f"GHB, tuned prior (DP), eps={e}", arm)),
            ("Local, tuned prior$^\\ddagger$", lambda e, arm, nm: FC.arm_value(s, nm, "Local, tuned prior", arm))]
    lit = [(lab, (lambda f: (lambda e, arm, nm: FC.arm_value(s, nm, f.format(e=e), arm)))(f)) for lab, f in FC.OTHERS.items()]
    def line(lab, fun, group):
        cells = []
        for e in ("1", "10"):
            o = np.array([fun(e, "oracle", nm) for nm in conf])
            r = np.array([fun(e, "lono", nm) for nm in conf])
            w = "" if group == "ghb" else f"{int((eb[e] < r).sum())}"
            fmt = lambda x: f"{x:.3f}" if x < 9.995 else f"{x:.1f}"
            cells += [fmt(o.mean()), fmt(r.mean()), w]
        return lab + " & " + " & ".join(cells) + r"\\"
    rows.append(r"\multicolumn{7}{@{}l}{\emph{GHB and the local reference}}\\")
    rows += [line(l, f, "ghb") for l, f in head]
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{7}{@{}l}{\emph{Federated and personalized methods (same budget)}}\\")
    order = sorted(lit, key=lambda t: np.mean([t[1]("1", "lono", nm) for nm in conf]))
    rows += [line(l, f, "lit") for l, f in order]
    write("t_real_final.tex", rows)
    # per network, leave-one-network-out arm, both budgets
    for e in ("1", "10"):
        rows = ["Hospitals $K$ & " + " & ".join(str(s[nm]["info"]["K"]) for nm in conf) + r"\\", r"\midrule"]
        allm = head + lit
        vals = {l: [f(e, "lono", nm) for nm in conf] for l, f in allm}
        best = [min(v[j] for l, v in vals.items() if not l.startswith("Local")) for j in range(len(conf))]
        for l, v in vals.items():
            rows.append(l + " & " + " & ".join((f"\\textbf{{{x:.3f}}}" if x == best[j] else (f"{x:.3f}" if x < 9.995 else f"{x:.1f}")) for j, x in enumerate(v)) + r"\\")
        write(f"s_real_final_lono_eps{e}.tex", rows)


def tcga():
    """TCGA, richer model (real_v3.py, dev/PREREG_v4.md): the pre-registered confirmation on five unseen networks
    (main paper) and the development summary over twelve networks (supplement)."""
    import glob
    from collections import defaultdict
    from real_v3 import GRIDS, ckey
    c = load("real_v4_confirm.json")
    nets, rel = c["nets"], c["rel"]
    fams = list(GRIDS)
    short = {"DP-FedAvg + FT": "DP-FedAvg + FT", "DP-FedAvg-SA + FT": "DP-FedAvg + FT, sec.\\ agg.", "APFL": "APFL", "Ditto": "Ditto",
             "pFedMe": "pFedMe", "Clustered DP-FedAvg": "Clustered DP-FedAvg"}
    rows = []
    for e in ("1", "10"):
        r = rel[e]
        g = np.array(r["GHB v4"])
        lines = [(r"\textbf{GHB}, no tuning", g), ("GHB-lite (evidence prior), no tuning", np.array(r["GHB-EB"])),
                 ("Local, evidence prior$^\\ddagger$", np.array(r["Local, evidence prior"]))]
        lines += [(short[fm] + ", tuned on development", np.array(r[fm + " (realistic)"])) for fm in fams]
        orc = np.min([r[fm + " (oracle)"] for fm in fams], 0)
        lines.append(("Best of the six, oracle per network$^\\star$", orc))
        best = np.min([v for l, v in lines if not l.startswith("Local") and "oracle" not in l], 0)
        rows.append(f"\\multicolumn{{8}}{{@{{}}l}}{{\\emph{{$\\epsilon={e}$}}}}\\\\")
        for l, v in lines:
            cells = [(f"\\textbf{{{x:.3f}}}" if abs(x - b) < 1e-12 else f"{x:.3f}") for x, b in zip(v, best)]
            wins = "" if l.startswith("\\textbf") else f"{int((g < v).sum())}/{len(v)}"
            rows.append(l + " & " + " & ".join(cells) + f" & {v.mean():.3f} & {wins}\\\\")
        if e == "1":
            rows.append("\\midrule")
    write("t_tcga_confirm.tex", rows)
    # oracle arm per method and network, and the configurations frozen for the realistic arm
    sel = json.load(open(ghb.ROOT / "simulations" / "dev" / "v4_realistic_configs.json"))
    rows = []
    for e in ("1", "10"):
        rows.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{$\\epsilon={e}$}}}}\\\\")
        for fm in fams:
            v = np.array(rel[e][fm + " (oracle)"])
            rows.append(short[fm] + " & " + " & ".join(f"{x:.3f}" for x in v) + f" & {v.mean():.3f}\\\\")
        if e == "1":
            rows.append("\\midrule")
    write("s_tcga_oracle.tex", rows)
    names = {"FT": ("rounds $R$", "step $\\eta$", "fine-tuning steps"), "FTSA": ("rounds $R$", "step $\\eta$", "fine-tuning steps"),
             "APFL": ("rounds $R$", "step $\\eta$", "mixing $\\alpha$"), "PDIT": ("rounds $R$", "step $\\eta$", "proximal $\\lambda$"),
             "PME": ("rounds $R$", "step $\\eta$", "proximal $\\lambda$"), "CLU": ("clusters", "rounds $R$", "step $\\eta$", "fine-tuning steps")}
    rows = []
    for fm, grid in GRIDS.items():
        cells = []
        for e in ("1", "10"):
            parts = sel[f"{fm}, eps={e}"].split("|")
            cells.append(", ".join(f"{a}={b}" for a, b in zip(names[parts[0]], parts[1:])))
        rows.append(short[fm] + f" & {len(grid)} & " + " & ".join(cells) + "\\\\")
    write("s_tcga_configs.tex", rows)
    # development: twelve networks, ten repeats (stored baselines and GHB-lite from real_v3_ckpt, GHB from v4c_ckpt)
    acc = defaultdict(lambda: [0.0, 0.0])
    for fn in glob.glob(str(R / "real_v3_ckpt" / "*.json")):
        if "/confirm_" in fn:
            continue
        for k, (ll, cnt) in json.load(open(fn))["out"].items():
            acc[k][0] += sum(ll); acc[k][1] += sum(cnt)
    for fn in glob.glob(str(R / "v4c_ckpt" / "*.json")):
        for k, (ll, cnt) in json.load(open(fn)).items():
            if "v4c R1=1 +pan," in k:
                k2 = k.replace("v4c R1=1 +pan", "GHB v4")
                acc[k2][0] += sum(ll); acc[k2][1] += sum(cnt)
    dn = sorted({k.split("||")[0] for k in acc})
    L = lambda p, m: (acc[f"{p}||{m}"][0] / acc[f"{p}||{m}"][1]) / (acc[f"{p}||Local"][0] / acc[f"{p}||Local"][1])
    rows = []
    def line(lab, fun):
        cells = []
        for e in ("1", "10"):
            cells += [f"{np.mean([fun(p, e, 'a') for p in dn]):.3f}", f"{np.mean([fun(p, e, 'o') for p in dn]):.3f}"]
        return lab + " & " + " & ".join(cells) + r"\\"
    rows.append(line("GHB, no tuning", lambda p, e, a: L(p, f"GHB v4, eps={e}")))
    rows.append(line("GHB-lite (evidence prior), no tuning", lambda p, e, a: L(p, f"GHB-EB, eps={e}")))
    for v in ("+ safety", "+ learned scale", "+ correlated", "+ cross-cancer", "GHB v3"):
        rows.append(line(f"GHB-lite {v.replace('GHB v3', 'with all four additions (v3)').replace('+ ', 'with ')}", (lambda v: lambda p, e, a: L(p, f"{v}, eps={e}"))(v)))
    rows.append(line("Local, evidence prior", lambda p, e, a: L(p, "Local, evidence prior")))
    rows.append(r"\midrule")
    for fm, grid in GRIDS.items():
        def fun(p, e, a, grid=grid):
            M = np.array([[L(q, f"{ckey(cf)}, eps={e}") for q in dn] for cf in grid])
            return M[M.mean(1).argmin(), dn.index(p)] if a == "a" else M[:, dn.index(p)].min()
        rows.append(line(short[fm], fun))
    write("s_tcga_dev.tex", rows)


def tcga_ablation():
    """GHB ablation on the twelve development networks (dev/v4_dev.py, v4_ablation_ckpt): one change at a time."""
    import glob
    from collections import defaultdict
    acc = defaultdict(lambda: [0.0, 0.0])
    files = glob.glob(str(R / "v4_ablation_ckpt" / "*.json"))
    if not files:
        return
    for fn in files:
        for k, (ll, cnt) in json.load(open(fn)).items():
            acc[k][0] += sum(ll); acc[k][1] += sum(cnt)
    dn = sorted({k.split("||")[0] for k in acc})
    L = lambda p, m: (acc[f"{p}||{m}"][0] / acc[f"{p}||{m}"][1]) / (acc[f"{p}||Local"][0] / acc[f"{p}||Local"][1])
    labels = {"v4 locked": "GHB (as frozen)", "v4 - pan-network": "without borrowing across networks",
              "v4 - no-borrowing components": "without the non-borrowing components",
              "v4 - privacy covariance": "without propagating the privacy covariance",
              "v4 - leave-one-out": "without removing the hospital's own data",
              "v4 two Newton rounds": "two rounds of joint releases", "v4 tau from releases (f=.5)": "$\\tau$ from per-hospital releases (half the budget)",
              "v4 + institution effects (f=.5)": "the same, with institution effects across networks"}
    rows = []
    for m, lab in labels.items():
        v = [np.array([L(p, f"{m}, eps={e}") for p in dn]) for e in ("1", "10")]
        base = [np.array([L(p, f"v4 locked, eps={e}") for p in dn]) for e in ("1", "10")]
        cells = []
        for a, b in zip(v, base):
            cells += [f"{a.mean():.3f}", "" if m == "v4 locked" else f"{int((b < a).sum())}/{len(a)}"]
        rows.append(lab + " & " + " & ".join(cells) + r"\\")
        if m == "v4 locked":
            rows.append(r"\midrule")
    write("s_tcga_ablation.tex", rows)


def tcga_tuned():
    """Tuned GHB against tuned literature on the confirmation networks (dev/v4_tune.py, exploratory)."""
    path = R / "real_v4_tuned.json"
    if not path.exists():
        return
    t = json.load(open(path))
    short = {"DP-FedAvg + FT": "DP-FedAvg + FT", "DP-FedAvg-SA + FT": "DP-FedAvg + FT, sec.\\ agg.", "APFL": "APFL",
             "Ditto": "Ditto", "pFedMe": "pFedMe", "Clustered DP-FedAvg": "Clustered DP-FedAvg"}
    mean = lambda v: f"{np.mean(v):.3f}"
    rows = [r"GHB, frozen configuration & " + " & ".join(f"{mean(t[e]['GHB, frozen'])} & {mean(t[e]['GHB, frozen'])}" for e in ("1", "10")) + r"\\",
            r"\textbf{GHB}, tuned like the others & " + " & ".join(f"{mean(t[e]['GHB, tuned on development'])} & {mean(t[e]['GHB, oracle'])}" for e in ("1", "10")) + r"\\",
            r"\midrule"]
    for fm, lab in short.items():
        rows.append(lab + " & " + " & ".join(f"{mean(t[e][fm + ' (realistic)'])} & {mean(t[e][fm + ' (oracle)'])}" for e in ("1", "10")) + r"\\")
    write("t_tcga_tuned.tex", rows)


def main():
    for fn in (tcga, tcga_ablation, tcga_tuned, sim8, sim15, sim16, sim17, sim18, real_heart, real_networks, real_final, sim1, sim2, sim3, sim4, sim5, sim6, sim7, sim9, sim11, sim12, sim13, sim14):
        fn()
    print("tables:", sorted(p.name for p in OUT.glob("*.tex")))


if __name__ == "__main__":
    main()
