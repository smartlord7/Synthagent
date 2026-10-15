"""Statistical tests of the GHB v5 claims, as specified in dev/STATS_PLAN.md (committed before this was run).

    python3 stats_tests.py     writes supplement/results/stats_tests.json and tables/stats_confirm.tex, stats_all.tex

Everything is computed from saved results; nothing is rerun.
"""
import json

import numpy as np
from scipy import stats

import final_compare as FC
import ghb

R = ghb.RESDIR
TAB = R / "tables"
FAMS = ["DP-FedAvg + FT", "DP-FedAvg-SA + FT", "APFL", "Ditto", "pFedMe", "Clustered DP-FedAvg"]  # flamby_bench.FAMILIES
CITE = {"DP-FedAvg + FT": "mcmahan2018learning", "DP-FedAvg-SA + FT": "mcmahan2018learning,bonawitz2017practical",
        "APFL": "deng2020adaptive", "Ditto": "li2021ditto", "pFedMe": "tdinh2020personalized",
        "Clustered DP-FedAvg": "ghosh2020efficient,mcmahan2018learning", "DP-FedAvg": "mcmahan2018learning",
        "Decentralized DP-SGD": "lian2017can", "Ditto-style (DP)": "li2021ditto", "DP-FedAvg + fine-tuning": "mcmahan2018learning",
        "Ditto-style (one-shot)": "li2021ditto", "DP-FedAvg + FT, sec. agg.": "mcmahan2018learning,bonawitz2017practical",
        "Per-FedAvg": "fallah2020personalized", "FedRep": "collins2021exploiting", "FedProx": "li2020federated"}


def holm(ps):
    """Holm-adjusted p-values (same order as ps)."""
    ps = np.asarray(ps, float)
    o, m, out, run = np.argsort(ps), len(ps), np.empty(len(ps)), 0.0
    for i, j in enumerate(o):
        run = max(run, min(1.0, (m - i) * ps[j]))
        out[j] = run
    return out


def wilcoxon_less(d):
    """One-sided exact Wilcoxon signed-rank p for H1: median of d < 0 (d = GHB minus comparator, lower is better)."""
    d = np.asarray(d, float)
    return float(stats.wilcoxon(d, alternative="less", method="exact").pvalue) if np.any(d != 0) else 1.0


def tci(d):
    d = np.asarray(d, float)
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    h = stats.t.ppf(0.975, len(d) - 1) * se
    return float(m), float(m - h), float(m + h)


def name(m):
    return m.replace(" (realistic)", "").replace(" (oracle)", "")


# --------------------------------------------------------------------------------------------- family 1: TCGA confirm
def family1():
    c = json.load(open(R / "v5_confirm.json"))
    out = []
    for e in ("1", "10"):
        rel = c["rel"][e]
        g = np.array(rel["GHB v5"])
        real = {m: np.array(v) for m, v in rel.items() if m.endswith("(realistic)")}
        orc = {m: np.array(v) for m, v in rel.items() if m.endswith("(oracle)")}
        best = min(real, key=lambda m: real[m].mean())
        bo = min(orc, key=lambda m: orc[m].mean())
        for comp, v, test in (("Local", np.ones_like(g), True), (best, real[best], True), (bo, orc[bo], False)):
            d = g - v
            m, lo, hi = tci(d)
            out.append(dict(family=1, data="TCGA confirmation (5 networks)", eps=e, comparator=comp, n=len(d),
                            better=int((d < 0).sum()), mean=m, lo=lo, hi=hi, p=wilcoxon_less(d) if test else None,
                            lower_better=True, tested=test))
    ad = holm([r["p"] for r in out if r["tested"]])
    for r, a in zip([r for r in out if r["tested"]], ad):
        r["p_holm"] = float(a)
    return out


# ---------------------------------------------------------------------------------------------- family 2: Camelyon17
def family2():
    c = json.load(open(R / "camelyon_image.json"))
    pr = {k: np.array(v["bacc"]) for k, v in c["per_rep"].items()}
    out = []

    def add(e, comp, d, margin=0.0, kind="superiority"):
        # higher balanced accuracy is better: test d - margin > 0, i.e. (comparator - GHB - margin) < 0
        dd = d + margin
        m, lo, hi = tci(d)
        out.append(dict(family=2, data="Camelyon17 (10 repetitions)", eps=e, comparator=comp, n=len(d), better=int((d > 0).sum()),
                        mean=m, lo=lo, hi=hi, p=wilcoxon_less(-dd), lower_better=False, tested=True, kind=kind, margin=margin))
    g10 = pr["GHB, eps=10"]
    add("10", "Local", g10 - pr["Local"])
    for f in FAMS:
        k = f"{f} (realistic), eps=10"
        if k in pr:
            add("10", f"{f} (realistic)", g10 - pr[k])
    add("1", "Local", pr["GHB, eps=1"] - pr["Local"], margin=0.005, kind="non-inferiority (margin 0.005)")
    add("inf", "Local", pr["GHB, eps=inf"] - pr["Local"])
    add("inf", "Pooled", pr["GHB, eps=inf"] - pr["Pooled"])
    for r, a in zip(out, holm([r["p"] for r in out])):
        r["p_holm"] = float(a)
    return out


# ------------------------------------------------------------------------------------- family 3: all real data sets
def datasets(e):
    """Per data set: (name, GHB value, Local value, strongest realistic name and value, lower_better)."""
    rows = []
    c = json.load(open(R / "v5_confirm.json"))
    rel = c["rel"][e]
    for i, nm in enumerate(c["nets"]):
        real = {m: rel[m][i] for m in rel if m.endswith("(realistic)")}
        b = min(real, key=real.get)
        rows.append((f"{nm} (confirmation)", rel["GHB v5"][i], 1.0, b, real[b], True))
    rn = json.load(open(R / "real_networks.json"))["res"]
    for nm in FC.CONF:
        real = {lab: FC.arm_value(rn, nm, f.format(e=e), "lono") for lab, f in FC.OTHERS.items()}
        b = min(real, key=real.get)
        rows.append((f"{nm} (small network)", rn[nm][f"GHB v5 {e}"]["rel"], 1.0, b, real[b], True))
    fr = json.load(open(R / "flamby_results.json"))
    for ds, met, low in (("heart", "logloss", True), ("brca", "cindex", False), ("isic", "bacc", False)):
        rw = fr[ds]["rows"]
        v = lambda k: rw[k][met]["mean"]
        real = {f: v(f"{f} (realistic), eps={e}") for f in FAMS if f"{f} (realistic), eps={e}" in rw}
        real = {k: x for k, x in real.items() if np.isfinite(x)}
        b = (min if low else max)(real, key=real.get)
        rows.append((f"FLamby {ds}", v(f"GHB v5, eps={e}"), v("Local"), b, real[b], low))
    cm = json.load(open(R / "camelyon_image.json"))["rows"]
    real = {f: cm[f"{f} (realistic), eps={e}"]["bacc"] for f in FAMS if f"{f} (realistic), eps={e}" in cm}
    b = max(real, key=real.get)
    rows.append(("Camelyon17", cm[f"GHB, eps={e}"]["bacc"], cm["Local"]["bacc"], b, real[b], False))
    return rows


def family3():
    out, detail = [], {}
    for e in ("1", "10"):
        ds = datasets(e)
        detail[e] = [dict(data=d[0], ghb=d[1], local=d[2], best=d[3], best_value=d[4], lower_better=d[5]) for d in ds]
        for comp, idx in (("Local", 2), ("strongest realistic competitor", 4)):
            wins = sum((d[1] < d[idx]) if d[5] else (d[1] > d[idx]) for d in ds)
            p = float(stats.binomtest(wins, len(ds), 0.5, alternative="greater").pvalue)
            out.append(dict(family=3, data=f"All real data sets ({len(ds)})", eps=e, comparator=comp, n=len(ds), better=wins, p=p, tested=True))
    for r, a in zip(out, holm([r["p"] for r in out])):
        r["p_holm"] = float(a)
    return out, detail


# --------------------------------------------------------------------------------------- family 4: Simulation 15
def family4():
    s = json.load(open(R / "sim15_literature.json"))["res"]
    scen = list(s.values()) if isinstance(s, dict) else s
    comps = ["DP-FedAvg", "DP-FedAvg + fine-tuning", "Decentralized DP-SGD", "Clustered DP-FedAvg", "Ditto-style (DP)", "Local"]
    out = []
    for comp in comps:
        zs, labs = [], []
        for r in scen:
            if comp not in r["kl"]:
                continue
            d = r["kl"]["GHB v4 (DP)"] - r["kl"][comp]  # the key "GHB v4 (DP)" holds GHB v5 (logit_world.V4_CFG)
            se = np.hypot(r["kl_se"]["GHB v4 (DP)"], r["kl_se"][comp])
            zs.append(d / se)
            labs.append(r["label"])
        p2 = [2 * stats.norm.sf(abs(z)) for z in zs]
        ph = holm(p2)
        better = sum(1 for z, p in zip(zs, ph) if p < 0.05 and z < 0)
        worse = [lab for z, p, lab in zip(zs, ph, labs) if p < 0.05 and z > 0]
        out.append(dict(family=4, comparator=comp, scenarios=len(zs), better=better, worse=len(worse), worse_in=worse,
                        undecided=len(zs) - better - len(worse)))
    return out


# ------------------------------------------------------------------- family 5: tuned GHB against tuned literature
def family5():
    t = json.load(open(R / "real_v5_tuned.json"))  # every method on the same folds and privacy noise
    out = []
    for e in ("1", "10"):
        v = {m: np.array(x) for m, x in t[e].items() if isinstance(x, list) and len(x) == len(t["nets"])}
        for arm, gk in (("realistic", "GHB, tuned on development"), ("oracle", "GHB, oracle")):
            lit = {m: x for m, x in v.items() if m.endswith(f"({arm})")}
            best = min(lit, key=lambda m: lit[m].mean())
            d = v[gk] - lit[best]
            m, lo, hi = tci(d)
            out.append(dict(family=5, data="TCGA confirmation, both tuned (5 networks)", eps=e, arm=arm,
                            comparator=best, ghb=f"GHB tuned ({'design networks' if arm == 'realistic' else 'test folds'})",
                            n=len(d), better=int((d < 0).sum()), mean=m, lo=lo, hi=hi, p=wilcoxon_less(d), tested=True,
                            ghb_mean=float(v[gk].mean()), comp_mean=float(lit[best].mean())))
    for r, a in zip(out, holm([r["p"] for r in out])):
        r["p_holm"] = float(a)
    return out


# ------------------------------------------------------------------------------------------------------------ tables
def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def comp_tex(c):
    base = name(c)
    arm = " (oracle)" if c.endswith("(oracle)") else (" (realistic)" if c.endswith("(realistic)") else "")
    key = CITE.get(base)
    return f"{base} \\cite{{{key}}}{arm}" if key else base


def tables(f1, f2, f3, f4):
    TAB.mkdir(exist_ok=True)
    L = ["\\begin{tabular}{llllrll}", "\\toprule",
         "Data & $\\varepsilon$ & GHB against & Mean difference [95\\% CI] & Better & $p$ & Holm $p$ \\\\", "\\midrule"]
    for r in f1 + f2:
        e = "$\\infty$" if r["eps"] == "inf" else r["eps"]
        ph = r.get("p_holm")
        sig = ph is not None and ph < 0.05
        pcell = fmt_p(r["p"]) if r["p"] is not None else "not tested"
        hcell = ("\\textbf{" + fmt_p(ph) + "}" if sig else fmt_p(ph)) if ph is not None else "not tested"
        comp = comp_tex(r["comparator"]) + (", non-inferiority" if r.get("kind", "").startswith("non") else "")
        L.append(f"{r['data']} & {e} & {comp} & ${r['mean']:+.3f}$ [${r['lo']:+.3f}$, ${r['hi']:+.3f}$] & {r['better']} of {r['n']} & {pcell} & {hcell} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "stats_confirm.tex").write_text("\n".join(L) + "\n")
    L = ["\\begin{tabular}{llrll}", "\\toprule", "$\\varepsilon$ & GHB against & GHB better in & $p$ (sign test) & Holm $p$ \\\\", "\\midrule"]
    for r in f3:
        h = fmt_p(r["p_holm"])
        L.append(f"{r['eps']} & {r['comparator']} & {r['better']} of {r['n']} & {fmt_p(r['p'])} & {'\\textbf{' + h + '}' if r['p_holm'] < 0.05 else h} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "stats_all.tex").write_text("\n".join(L) + "\n")
    L = ["\\begin{tabular}{lrrrr}", "\\toprule", "GHB against & Scenarios & GHB better & Not distinguishable & GHB worse \\\\", "\\midrule"]
    for r in f4:
        L.append(f"{comp_tex(r['comparator'])} & {r['scenarios']} & {r['better']} & {r['undecided']} & {r['worse']} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "stats_sim15.tex").write_text("\n".join(L) + "\n")


def tuned_table(f5):
    L = ["\\begin{tabular}{llllrll}", "\\toprule",
         "$\\varepsilon$ & Tuned on & GHB against & Mean difference [95\\% CI] & Better & $p$ & Holm $p$ \\\\", "\\midrule"]
    for r in f5:
        h = fmt_p(r["p_holm"])
        L.append(f"{r['eps']} & {'design networks' if r['arm'] == 'realistic' else 'test folds'} & {comp_tex(r['comparator'])} & "
                 f"${r['mean']:+.3f}$ [${r['lo']:+.3f}$, ${r['hi']:+.3f}$] & {r['better']} of {r['n']} & {fmt_p(r['p'])} & "
                 f"{'\\textbf{' + h + '}' if r['p_holm'] < 0.05 else h} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "stats_tuned.tex").write_text("\n".join(L) + "\n")


def detail_table(detail):
    """Per data set, both budgets: GHB, Local, the strongest realistic competitor (name, value); the better of GHB and
    that competitor in bold."""
    dn = lambda d: d.replace(" (confirmation)", "$^{c}$").replace(" (small network)", "").replace("FLamby ", "FLamby ").replace("_", "\\_")
    L = ["\\begin{tabular}{lrrlrrrlr}", "\\toprule",
         " & \\multicolumn{4}{c}{$\\varepsilon=1$} & \\multicolumn{4}{c}{$\\varepsilon=10$} \\\\",
         "\\cmidrule(lr){2-5}\\cmidrule(l){6-9}",
         "Data set & GHB & Local & \\multicolumn{2}{l}{Strongest realistic} & GHB & Local & \\multicolumn{2}{l}{Strongest realistic} \\\\", "\\midrule"]
    for a, b in zip(detail["1"], detail["10"]):
        cells = []
        for r in (a, b):
            g, c = r["ghb"], r["best_value"]
            gw = (g < c) if r["lower_better"] else (g > c)
            fg = f"\\textbf{{{g:.3f}}}" if gw else f"{g:.3f}"
            fc = f"{c:.3f}" if gw else f"\\textbf{{{c:.3f}}}"
            cells += [fg, f"{r['local']:.3f}", comp_tex(r["best"]), fc]
        L.append(f"{dn(a['data'])} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "stats_detail.tex").write_text("\n".join(L) + "\n")


def main():
    f1, f2 = family1(), family2()
    f3, detail = family3()
    f4 = family4()
    f5 = family5()
    json.dump(dict(family1=f1, family2=f2, family3=f3, family3_detail=detail, family4=f4, family5=f5), open(R / "stats_tests.json", "w"), indent=1)
    tables(f1, f2, f3, f4)
    tuned_table(f5)
    detail_table(detail)
    for r in f1 + f2 + f3 + f5:
        extra = f" mean {r['mean']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]" if "mean" in r else ""
        ph = r.get("p_holm")
        print(f"F{r['family']} eps={r['eps']:>3} vs {r['comparator'][:40]:40s} better {r['better']}/{r['n']}{extra} p={r['p'] if r['p'] is None else round(r['p'], 4)} holm={None if ph is None else round(ph, 4)}")
    for r in f4:
        print("F4", r)


if __name__ == "__main__":
    main()
