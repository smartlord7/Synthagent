"""Does the smallness index predict the gain from borrowing better than n, events or EPV? (dev/SIGMA_PLAN.md)

    python3 sigma_validation.py run      per-hospital Local and GHB on the small networks (sigma_networks_ckpt)
    python3 sigma_validation.py score    correlations, bootstrap, tables (results/sigma_validation.json, tables/sigma_*)
"""
import csv
import glob
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import stats

import ghb

R = ghb.RESDIR
TAB = R.parent / "tables"
CK = R / "sigma_networks_ckpt"
SYN = Path(__file__).resolve().parents[2] / "Synthagent__An_Agentic_Framework_for_Synthetic_Data_Generation" / "generated"


# ------------------------------------------------------------------------------------------- Riley et al. criteria
def riley(p, phi, S=0.9, delta=0.05, share=0.15):
    """Minimum sample size for a binary-outcome prediction model with p predictor parameters (max of the three
    criteria of Riley et al.), anticipated R^2_CS = share * its maximum."""
    lnull = phi * np.log(phi) + (1 - phi) * np.log(1 - phi)
    r2max = 1 - np.exp(2 * lnull)
    r2 = share * r2max
    n1 = p / ((S - 1) * np.log(1 - r2 / S))
    s2 = r2 / (r2 + delta * r2max)
    n2 = p / ((s2 - 1) * np.log(1 - r2 / s2))
    n3 = (1.96 / 0.05) ** 2 * phi * (1 - phi)
    return float(max(n1, n2, n3))


def describe(n, ev, p):
    phi = np.clip(ev / n, 0.5 / n, 1 - 0.5 / n)
    events = n * min(phi, 1 - phi)
    ns = riley(p, phi)
    return dict(n=int(n), events=float(events), epv=float(events / p), nstar=ns, sigma=ns / n, p=p, prevalence=float(ev / n))


# ------------------------------------------------------------------------------------- small networks (rerun)
def run_net(item):
    import ghb_v4 as V4
    import real_networks as RN
    from exp7_logistic import P, fit
    i, (name, label) = item
    rng = np.random.default_rng([20, i])
    rng5 = np.random.default_rng([24, i])
    cfg = dict(RN.V4_CFG, int_sd=2.5, free_int=2.5, tight=True, hdiag="auto")  # GHB as released (dev/v5_networks.py --frozen)
    data, ids = RN.load(name)
    K = len(data)
    ll = {m: np.zeros(K) for m in ("Local", "GHB 1", "GHB 10")}
    for rep in range(RN.REPEATS):
        fs = [RN.folds(rng, y, RN.FOLDS) for _, y in data]
        for f in range(RN.FOLDS):
            train, test = [], []
            for k, (X, y) in enumerate(data):
                tr, te = np.flatnonzero(fs[k] != f), np.flatnonzero(fs[k] == f)
                if len(tr) == 0:
                    tr = te
                train.append((X[tr], y[tr])); test.append((X[te], y[te]))
            est = {"Local": np.array([fit(X, y, np.zeros(P), RN.PRIOR) for X, y in train])}
            for e in (1.0, 10.0):
                est[f"GHB {e:g}"] = V4.estimate({"x": train}, rng5, e, RN.DELTA, np.sqrt(float(P)), cfg=cfg)["x"]
            for m, th in est.items():
                for k, (Xt, yt) in enumerate(test):
                    s = Xt @ th[k]
                    ll[m][k] += np.sum(np.logaddexp(0, s) - yt * s)
    out = dict(network=name, p=P - 1, n=[int(len(y)) for _, y in data], events=[float(y.sum()) for _, y in data],
               ll={m: v.tolist() for m, v in ll.items()})
    json.dump(out, open(CK / f"{name}.json", "w"))
    return name


def run():
    import real_networks as RN
    CK.mkdir(exist_ok=True)
    todo = [(i, nw) for i, nw in enumerate(RN.NETWORKS) if not (CK / f"{nw[0]}.json").exists()]
    with Pool(4) as pool:
        for n in pool.imap_unordered(run_net, todo):
            print("done", n, flush=True)


# --------------------------------------------------------------------------------------------- hospital records
def tcga_rows(ckdir, prefix, ghb_key):
    import real_v3 as RV
    files = sorted(glob.glob(str(R / ckdir / f"{prefix}*.json")))
    acc = {}
    for fn in files:
        d = json.load(open(fn))
        for key, (loss, cnt) in d.items():
            net, m = key.split("||")
            a = acc.setdefault(net, {})
            a.setdefault(m, np.zeros(len(loss)))
            a[m] += np.array(loss)
            a.setdefault("_cnt", np.array(cnt))
    rows = []
    for net, a in acc.items():
        data, _ = RV.load(net)
        n = np.array([len(y) for _, y in data])
        assert np.array_equal(n, a["_cnt"]), (net, n, a["_cnt"])
        for e in ("1", "10"):
            g = 1 - a[ghb_key.format(e=e)] / a["Local"]
            for k, (_, y) in enumerate(data):
                rows.append(dict(network=net, set=ckdir, hospital=k, eps=e, gain=float(g[k]), **describe(len(y), y.sum(), 7)))
    return rows


def small_rows():
    rows = []
    for fn in sorted(glob.glob(str(CK / "*.json"))):
        d = json.load(open(fn))
        loc = np.array(d["ll"]["Local"])
        for e in ("1", "10"):
            g = 1 - np.array(d["ll"][f"GHB {e}"]) / loc
            for k, (n, ev) in enumerate(zip(d["n"], d["events"])):
                rows.append(dict(network=d["network"], set="small", hospital=k, eps=e, gain=float(g[k]), **describe(n, ev, d["p"])))
    return rows


# ---------------------------------------------------------------------------------------------------- statistics
PRED = ("sigma", "n", "events", "epv")


def rhos(rows):
    g = np.array([r["gain"] for r in rows])
    return {v: float(stats.spearmanr(g, [r[v] for r in rows]).statistic) for v in PRED}


def D(rh):
    return abs(rh["sigma"]) - max(abs(rh["n"]), abs(rh["events"]), abs(rh["epv"]))


def boot(rows, B=2000, seed=2026):
    rng = np.random.default_rng(seed)
    nets = sorted({r["network"] for r in rows})
    by = {n: [r for r in rows if r["network"] == n] for n in nets}
    ds = []
    for _ in range(B):
        pick = rng.choice(nets, len(nets), replace=True)
        rr = [r for n in pick for r in by[n]]
        ds.append(D(rhos(rr)))
    return [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))]


def within(rows):
    nets = sorted({r["network"] for r in rows})
    out = {v: [] for v in PRED}
    for n in nets:
        rr = [r for r in rows if r["network"] == n]
        if len(rr) >= 5:
            for v, x in rhos(rr).items():
                if np.isfinite(x):
                    out[v].append(x)
    return {v: float(np.mean(x)) for v, x in out.items()}, sum(1 for n in nets if sum(r["network"] == n for r in rows) >= 5)


def synthagent():
    ds = list(csv.DictReader(open(SYN / "datasets.csv")))
    fr = {r["Dataset"]: r for r in csv.DictReader(open(SYN / "full_results.csv"))}
    rows = []
    for d in ds:
        n, p, mino = int(d["Rows"]), int(d["Features"]), float(d["Minority (\\%)"]) / 100
        rows.append(dict(dataset=d["Dataset"], gain=float(fr[d["Dataset"]]["$\\Delta$F1 (pp)"]), **describe(n, n * mino, p)))
    return rows, rhos(rows)


def score():
    prim = (tcga_rows("v5_confirm_ckpt", "confirm_", "GHB v5, eps={e}") + tcga_rows("v5f_dev_ckpt", "", "v5 it + auto, eps={e}")
            + [r for r in small_rows() if r["network"] in ("medpar", "heart")])
    sec = [r for r in small_rows() if r["network"] not in ("medpar", "heart")]
    out = {"plan": "dev/SIGMA_PLAN.md"}
    for e in ("1", "10"):
        P_ = [r for r in prim if r["eps"] == e]
        S_ = [r for r in sec if r["eps"] == e]
        rh = rhos(P_)
        w, nw = within(P_)
        out[e] = dict(hospitals=len(P_), networks=len({r["network"] for r in P_}), rho=rh, D=D(rh), D_ci=boot(P_),
                      within=w, within_networks=nw, secondary=dict(hospitals=len(S_), rho=rhos(S_), D=D(rhos(S_))))
    sa_rows, sa_rho = synthagent()
    out["synthagent"] = dict(rows=sa_rows, rho=sa_rho)
    out["hospitals"] = prim + sec
    out["verdict"] = "better" if all(out[e]["D_ci"][0] > 0 for e in ("1", "10")) else "not shown to be better"
    json.dump(out, open(R / "sigma_validation.json", "w"), indent=1)
    tables(out)
    for e in ("1", "10"):
        o = out[e]
        print(f"eps={e}: {o['hospitals']} hospitals / {o['networks']} networks; rho", {k: round(v, 3) for k, v in o["rho"].items()},
              "D", round(o["D"], 3), "CI", [round(x, 3) for x in o["D_ci"]], "within", {k: round(v, 3) for k, v in o["within"].items()},
              "secondary", {k: round(v, 3) for k, v in o["secondary"]["rho"].items()})
    print("synthagent", {k: round(v, 3) for k, v in sa_rho.items()}, [(r["dataset"], round(r["sigma"], 2), round(r["gain"], 2)) for r in sa_rows])
    print("verdict:", out["verdict"])


def tables(out):
    TAB.mkdir(exist_ok=True)
    lab = {"sigma": "Smallness index $\\sigma$", "n": "Patients $n$", "events": "Events", "epv": "Events per variable"}
    L = ["\\begin{tabular}{@{}lrrrr@{}}", "\\toprule",
         " & \\multicolumn{2}{c}{$\\epsilon=1$} & \\multicolumn{2}{c}{$\\epsilon=10$}\\\\", "\\cmidrule(lr){2-3}\\cmidrule(l){4-5}",
         "Predictor of the gain & pooled & within networks & pooled & within networks\\\\", "\\midrule"]
    for v in PRED:
        cells = []
        for e in ("1", "10"):
            cells += [f"${out[e]['rho'][v]:+.2f}$", f"${out[e]['within'][v]:+.2f}$"]
        L.append(f"{lab[v]} & " + " & ".join(cells) + "\\\\")
    L.append("\\midrule")
    L.append("$D$ [95\\% CI] & \\multicolumn{2}{c}{" + f"${out['1']['D']:+.3f}$ [${out['1']['D_ci'][0]:+.3f}$, ${out['1']['D_ci'][1]:+.3f}$]" + "} & \\multicolumn{2}{c}{"
             + f"${out['10']['D']:+.3f}$ [${out['10']['D_ci'][0]:+.3f}$, ${out['10']['D_ci'][1]:+.3f}$]" + "}\\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "sigma_validation.tex").write_text("\n".join(L) + "\n")
    # per-hospital table for the supplement
    L = ["\\begin{longtable}{@{}llrrrrrrr@{}}",
         "\\caption{Every hospital of the test: size, events, events per variable (EPV), Riley minimum sample size $n^{\\star}$, smallness index $\\sigma$ and gain from GHB over local estimation (one minus the ratio of held-out log-losses). Sets: held out and design TCGA networks, and the networks of the seven-network study (small; Arizona Medicare and UCI Heart Disease are in the primary set, the six TCGA networks in the secondary set).}\\label{tab:sigma-hosp}\\\\", "\\toprule",
         "Network & Set & $n$ & Events & EPV & $n^{\\star}$ & $\\sigma$ & Gain, $\\epsilon=1$ & Gain, $\\epsilon=10$\\\\", "\\midrule", "\\endhead"]
    H = {}
    for r in out["hospitals"]:
        H.setdefault((r["network"], r["set"], r["hospital"]), {})[r["eps"]] = r
    sets = {"v5_confirm_ckpt": "held out", "v5f_dev_ckpt": "design", "small": "small"}
    for (net, st, k), d in sorted(H.items(), key=lambda t: (t[0][1], t[0][0], t[0][2])):
        a = d["1"]
        L.append(f"{net} & {sets[st]} & {a['n']} & {a['events']:.0f} & {a['epv']:.1f} & {a['nstar']:.0f} & {a['sigma']:.2f} & "
                 f"${d['1']['gain']:+.3f}$ & ${d['10']['gain']:+.3f}$\\\\")
    L += ["\\bottomrule", "\\end{longtable}"]
    (TAB / "sigma_hospitals.tex").write_text("\n".join(L) + "\n")
    n = {"SgHospOne": out["1"]["hospitals"], "SgNets": out["1"]["networks"]}
    for e, E in (("1", "One"), ("10", "Ten")):
        for v in PRED:
            n[f"Sg{E}Rho{v.capitalize()}"] = f"{out[e]['rho'][v]:+.2f}"
            n[f"Sg{E}Within{v.capitalize()}"] = f"{out[e]['within'][v]:+.2f}"
        n[f"Sg{E}D"] = f"{out[e]['D']:+.3f}"
        n[f"Sg{E}DLo"], n[f"Sg{E}DHi"] = f"{out[e]['D_ci'][0]:+.3f}", f"{out[e]['D_ci'][1]:+.3f}"
        n[f"Sg{E}SecRhoSigma"] = f"{out[e]['secondary']['rho']['sigma']:+.2f}"
        n[f"Sg{E}SecRhoN"] = f"{out[e]['secondary']['rho']['n']:+.2f}"
        n[f"SgWithinNets"] = out[e]["within_networks"]
    for v in PRED:
        n[f"SgSyn{v.capitalize()}"] = f"{out['synthagent']['rho'][v]:+.2f}"
    sg = sorted(r["sigma"] for r in out["synthagent"]["rows"])
    n["SgSynSigmaMin"], n["SgSynSigmaMax"] = f"{sg[0]:.1f}", f"{sg[-1]:.1f}"
    n["SgVerdict"] = out["verdict"]
    (TAB / "sigma_numbers.tex").write_text("% Generated by simulations/sigma_validation.py. Do not edit.\n"
                                           + "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in n.items()) + "\n")


if __name__ == "__main__":
    run() if sys.argv[1:] == ["run"] else score()
