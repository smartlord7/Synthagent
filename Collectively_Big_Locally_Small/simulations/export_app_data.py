"""Export the real-data results of the GHB paper (Section 7) for the GHB Simulation Lab (app/js/data/realResults.js).

Reads the saved results that make_tables.py turns into Tables 10 to 13, applies the same rules, and writes them as an
ES module of plain objects, so that the app shows exactly the numbers of the paper:

  real_v4_confirm.json    Table 10: pre-registered confirmation on five TCGA networks (realistic and oracle arms)
  real_v4_tuned.json      Table 11: GHB tuned like the literature methods (exploratory, after the confirmation)
  real_networks.json      Table 12: seven real networks, every literature method in the oracle, leave-one-network-out
                          and development (UCI Heart Disease) arms, via final_compare.arm_value / CONF / OTHERS
  real_heart.json         Table 13: UCI Heart Disease by training size
  real_v3_ckpt/confirm_0.json   hospital counts and sizes of the confirmation networks

Run from simulations/:  python3 export_app_data.py
"""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import final_compare as FC  # noqa: E402  (arm_value, CONF, OTHERS; imports real_networks for the tuning grids)

RES = HERE.parent / "supplement" / "results"
OUT = HERE.parent / "app" / "js" / "data" / "realResults.js"

CITE = {"DP-FedAvg": "McMahan et al., 2018", "DP-FedAvg + FT": "McMahan et al., 2018",
        "DP-FedAvg + FT, sec. agg.": "McMahan et al., 2018; Bonawitz et al., 2017", "DP-FedAvg-SA + FT": "McMahan et al., 2018; Bonawitz et al., 2017",
        "Decentralized DP-SGD": "Lian et al., 2017", "Clustered DP-FedAvg": "Ghosh et al., 2020; McMahan et al., 2018",
        "Ditto-style (one-shot)": "Li et al., 2021", "Ditto": "Li et al., 2021", "pFedMe": "T. Dinh et al., 2020", "APFL": "Deng et al., 2020",
        "Per-FedAvg": "Fallah et al., 2020", "FedRep": "Collins et al., 2021", "FedProx": "Li et al., 2020"}
SHORT = {"DP-FedAvg + FT": "DP-FedAvg + FT", "DP-FedAvg-SA + FT": "DP-FedAvg + FT, sec. agg.", "APFL": "APFL", "Ditto": "Ditto",
         "pFedMe": "pFedMe", "Clustered DP-FedAvg": "Clustered DP-FedAvg"}


def r3(x):
    return None if x is None or x != x else round(float(x), 6)


def load(name):
    return json.load(open(RES / name))


def tcga_confirm():
    """Table 10 (make_tables.tcga): per network, mean and wins of GHB against each row."""
    c = load("real_v4_confirm.json")
    info = load("real_v3_ckpt/confirm_0.json")["info"]
    nets = c["nets"]
    out = dict(nets=[dict(id=nm, label=nm.replace("TCGA-", ""), K=info[nm]["K"], n=info[nm]["n"]) for nm in nets],
               repeats=c.get("repeats"), eps={})
    v5c = load("v5_confirm.json")  # GHB as released (v5); its networks are listed in real_v3.SETS order
    order = [v5c["nets"].index(p) for p in nets]
    for e in ("1", "10"):
        r = c["rel"][e]
        g = np.array(v5c["rel"][e]["GHB v5"])[order]
        rows = [dict(key="ghb", label="GHB", note="no tuning", kind="ghb", values=g.tolist()),
                dict(key="v4", label="GHB v4 (earlier version)", note="no tuning", kind="ghb", values=r["GHB v4"]),
                dict(key="lite", label="GHB-lite (evidence prior)", note="no tuning", kind="ghb", values=r["GHB-EB"]),
                dict(key="local", label="Local, evidence prior", note="no other hospital", kind="local", values=r["Local, evidence prior"])]
        for fm in SHORT:
            rows.append(dict(key=fm, label=SHORT[fm], cite=CITE[fm], note="tuned on development", kind="lit", values=r[fm + " (realistic)"],
                             oracle=r[fm + " (oracle)"]))
        orc = np.min([r[fm + " (oracle)"] for fm in SHORT], 0)
        rows.append(dict(key="best_oracle", label="Best of the six, oracle per network", note="tuned on the test folds", kind="oracle", values=orc.tolist()))
        for row in rows:
            v = np.array(row["values"], float)
            row["values"] = [r3(x) for x in v]
            row["mean"] = r3(v.mean())
            row["wins"] = None if row["key"] == "ghb" else int((g < v).sum())
            if "oracle" in row:
                row["oracle"] = [r3(x) for x in row["oracle"]]
                row["oracle_mean"] = r3(np.mean(row["oracle"]))
        out["eps"][e] = rows
    return out


def tcga_tuned():
    """Table 11 (make_tables.tcga_tuned): means over the five confirmation networks."""
    t = load("real_v4_tuned.json")
    m = lambda v: r3(np.mean(v))
    rows = [dict(key="ghb_frozen", label="GHB, frozen configuration", kind="ghb",
                 dev={e: m(t[e]["GHB, frozen"]) for e in ("1", "10")}, oracle={e: m(t[e]["GHB, frozen"]) for e in ("1", "10")}),
            dict(key="ghb_tuned", label="GHB, tuned like the others", kind="ghb",
                 dev={e: m(t[e]["GHB, tuned on development"]) for e in ("1", "10")}, oracle={e: m(t[e]["GHB, oracle"]) for e in ("1", "10")})]
    for fm, lab in SHORT.items():
        rows.append(dict(key=fm, label=lab, cite=CITE[fm], kind="lit", dev={e: m(t[e][fm + " (realistic)"]) for e in ("1", "10")},
                         oracle={e: m(t[e][fm + " (oracle)"]) for e in ("1", "10")}))
    cf = t.get("configs")
    return dict(configs=cf if isinstance(cf, int) else (len(cf) if cf else None), rows=rows)


def real_networks():
    """Table 12 (make_tables.real_final with final_compare's arms) and the per-network values."""
    s = load("real_networks.json")["res"]
    conf = FC.CONF
    nets = [dict(id=nm, label=s[nm]["info"]["label"], K=s[nm]["info"]["K"], n=s[nm]["info"]["n"]) for nm in conf]
    ghb = {e: np.array([s[nm][f"GHB v5 {e}"]["rel"] for nm in conf]) for e in ("1", "10")}
    head = [("ghb", "GHB", "no tuning", "ghb", lambda e, arm, nm: s[nm][f"GHB v5 {e}"]["rel"]),
            ("lite", "GHB-lite", "no tuning", "ghb", lambda e, arm, nm: s[nm][f"GHB (DP, eps={e})"]["rel"]),
            ("liteEB", "GHB-lite, evidence prior", "no tuning", "ghb", lambda e, arm, nm: s[nm][f"GHB-EB {e}"]["rel"]),
            ("liteSA", "GHB-lite with secure aggregation", "no tuning", "ghb", lambda e, arm, nm: s[nm][f"GHB-SA {e}"]["rel"]),
            ("liteTuned", "GHB-lite, tuned prior", "tuned", "ghb", lambda e, arm, nm: FC.arm_value(s, nm, f"GHB, tuned prior (DP), eps={e}", arm)),
            ("local", "Local, tuned prior", "no other hospital", "local", lambda e, arm, nm: FC.arm_value(s, nm, "Local, tuned prior", arm))]
    lit = [(lab, lab, "tuned", "lit", (lambda f: (lambda e, arm, nm: FC.arm_value(s, nm, f.format(e=e), arm)))(f)) for lab, f in FC.OTHERS.items()]
    lit.sort(key=lambda t: np.mean([t[4]("1", "lono", nm) for nm in conf]))
    rows = []
    for key, lab, note, kind, fun in head + lit:
        row = dict(key=key, label=lab, note=note, kind=kind, eps={})
        if kind == "lit":
            row["cite"] = CITE.get(lab)
        for e in ("1", "10"):
            arms = {arm: [r3(fun(e, arm, nm)) for nm in conf] for arm in ("oracle", "lono", "uci")}
            lono = np.array(arms["lono"], float)
            row["eps"][e] = dict(oracle=r3(np.mean(arms["oracle"])), lono=r3(lono.mean()), uci=r3(np.mean(arms["uci"])),
                                 perNetwork=arms, wins=None if kind == "ghb" and key == "ghb" else (int((ghb[e] < lono).sum()) if kind == "lit" else None))
        rows.append(row)
    return dict(nets=nets, rows=rows, repeats=load("real_networks.json").get("repeats"))


def heart():
    """Table 13 (make_tables.real_heart): UCI Heart Disease, relative log-loss by training size."""
    s = load("real_heart.json")
    res = s["res"]
    tags = ("20", "40", "80", "all")
    names = [("Local", "Local", None), ("GHB (DP)", "GHB-lite, ε = 1", None), ("GHB (DP, eps=5)", "GHB-lite, ε = 5", None),
             ("DP-FedAvg + fine-tuning", "DP-FedAvg + FT, ε = 1", "McMahan et al., 2018"), ("Ditto-style (DP)", "Ditto-style, ε = 1", "Li et al., 2021"),
             ("Local GD, tuned stopping", "Local GD, tuned stopping", None), ("GHB (no DP)", "GHB-lite, no DP", None),
             ("GHB, empirical features (no DP)", "GHB-lite, empirical features, no DP", None), ("Global (pooled, no DP)", "Global (pooled), no DP", None)]
    rows = []
    for m, lab, cite in names:
        rows.append(dict(key=m, label=lab, cite=cite, private=("no DP" not in lab and m not in ("Local", "Local GD, tuned stopping")),
                         byTag={t: dict(rel=r3(res[t][m]["rel"]), se=r3(res[t][m]["rel_paired_se"]), worst=r3(res[t][m]["worst"])) for t in tags}))
    return dict(sites=s["info"]["sites"], n=s["info"]["n"], prevalence=[r3(p) for p in s["info"]["prevalence"]],
                localLL={t: r3(res[t]["Local"]["ll_mean"]) for t in tags}, tags=list(tags), rows=rows,
                splits=(s.get("repeats") or 0) * (s.get("folds") or 0))


def main():
    data = dict(meta=dict(generated=str(date.today()), script="simulations/export_app_data.py",
                          sources=["supplement/results/real_v4_confirm.json", "supplement/results/real_v4_tuned.json",
                                   "supplement/results/real_networks.json", "supplement/results/real_heart.json"]),
                confirm=tcga_confirm(), tuned=tcga_tuned(), networks=real_networks(), heart=heart())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    body = ["/**", " * data/realResults.js", " * Responsibility: the real-data results of the GHB paper (Section 7, Tables 10 to 13), generated by",
            " * simulations/export_app_data.py from supplement/results/*.json. Do not edit by hand: rerun the script.", " */"]
    for k, v in data.items():
        name = {"meta": "META", "confirm": "TCGA_CONFIRM", "tuned": "TCGA_TUNED", "networks": "REAL_NETWORKS", "heart": "UCI_HEART"}[k]
        body.append(f"export const {name} = {json.dumps(v, ensure_ascii=False, separators=(',', ':'))};")
    OUT.write_text("\n".join(body) + "\n")
    print("wrote", OUT, OUT.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
