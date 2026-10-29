"""Datasets, sizes and headline results of the corpus works evaluated on health data.

Input:  data/corpus_extraction.json (one record per work: datasets [{name, access, size, sites}], task, result,
        evidence [verbatim quotes], source, status), extracted from each work's abstract or own page.
Checks: every quote of a work whose source is its Europe PMC abstract must occur in that abstract (after whitespace
        normalization); records that fail are listed and their unsupported fields are dropped.
Output: supplement/tables/corpus_works.tex (one longtable per primary channel), supplement/tables/corpus_datasets.tex,
        and the macros \\Cx... in supplement/tables/corpus_numbers.tex.

    python3 scripts/corpus_extract.py
"""
import csv
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TAB = ROOT / "supplement" / "tables"
CH = {"P1": "P1 Pooling", "P2": "P2 Proxy", "P3": "P3 Prior", "P4": "P4 Unlabeled", "P5": "P5 Procurement",
      "P6": "P6 Re-expression", "none": "No remedy (documents a problem)"}
METRICS = {  # macro suffix: regular expression on the reported result (headline result as stated in the abstract)
    "Auc": r"\bAUR?OC\b|\bAUC\b|area under the (receiver|ROC|curve)",
    "Acc": r"\baccura(cy|te)\b",
    "Sens": r"\bsensitivit|\bspecificit|\brecall\b",
    "Prc": r"\bAUPRC\b|\bAUC-?PR\b|\bprecision\b|\bPPV\b",
    "Fone": r"\bF1\b|F-score|F1-score",
    "Dice": r"\bDice\b|\bDSC\b|\bIoU\b|Jaccard",
    "Cal": r"calibrat|\bBrier\b|\bECE\b",
    "Logloss": r"log-?loss|cross-?entropy|deviance|log-?likelihood",
    "Cindex": r"\bC-?index\b|concordance",
    "Err": r"\bR?MSE\b|\bMAE\b|mean (squared|absolute) error",
    "Eps": r"\u03b5|epsilon|differential privacy|privacy budget",
    "Rel": r"%|percent|fold\b",
}


def norm(s):
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def tex(s):
    s = (s or "").replace("\\", "/")
    for a, b in (("&", "\\&"), ("%", "\\%"), ("$", "\\$"), ("#", "\\#"), ("_", "\\_"), ("{", "\\{"), ("}", "\\}"),
                 ("~", "\\textasciitilde{}"), ("^", "\\textasciicircum{}"), ("—", ", "), ("–", "-"), ("≥", "$\\ge$"),
                 ("≤", "$\\le$"), ("±", "$\\pm$"), ("×", "$\\times$"), ("≈", "$\\approx$"), ("·", "$\\cdot$"),
                 ("\u202f", " "), ("\u2009", " "), ("\u00a0", " "), ("’", "'"), ("‘", "'"), ("“", "``"), ("”", "''")):
        s = s.replace(a, b)
    return s


def load():
    corpus = {r["key"]: r for r in csv.DictReader(open(ROOT / "data" / "corpus.csv"))}
    look = {json.loads(l)["key"]: json.loads(l) for l in open(ROOT / "prisma" / "data" / "corpus_lookup.jsonl")}
    ext = json.load(open(ROOT / "data" / "corpus_extraction.json"))
    return corpus, look, ext


def verify(ext, look):
    """Quotes of works extracted from their Europe PMC abstract must occur in that abstract."""
    bad = []
    for r in ext:
        if r.get("source") != "epmc_abstract":
            continue
        ab = norm(look.get(r["key"], {}).get("abstract")) + " " + norm(look.get(r["key"], {}).get("title"))
        miss = [q for q in r.get("evidence", []) if norm(q) not in ab]
        if miss:
            bad.append((r["key"], miss))
    return bad


def main():
    corpus, look, ext = load()
    bad = dict(verify(ext, look))
    for k, miss in bad.items():
        print("unverified quote(s):", k, miss[:2])
    E = {r["key"]: r for r in ext}
    rows = {c: [] for c in CH}
    for k, c in corpus.items():
        if c["modality"] == "General" or k not in E:
            continue
        rows[c["primary_channel"] if c["primary_channel"] in CH else "none"].append((c, E[k]))
    TAB.mkdir(parents=True, exist_ok=True)
    out = []
    for ch, lab in CH.items():
        if not rows[ch]:
            continue
        out += [f"\\subsection*{{{lab} ({len(rows[ch])} works)}}",
                "\\begin{longtable}{@{}p{2.6cm}p{0.7cm}p{1.2cm}p{4.4cm}p{4.6cm}@{}}", "\\toprule",
                "Work & Year & Modality & Data (size) & Reported result\\\\", "\\midrule", "\\endhead"]
        for c, e in sorted(rows[ch], key=lambda t: (t[0]["year"], t[0]["citation"])):
            if e.get("status") == "not_empirical":
                data, res = "no dataset (review, theory, guidance or simulation)", tex(e.get("result")) or "-"
            elif e.get("status") == "no_source":
                data, res = "abstract not accessible", "-"
            else:
                ds = e.get("datasets") or []
                data = "; ".join(tex(d.get("name")) + (f" ({tex(d['size'])})" if d.get("size") else "")
                                 + (f", {d['sites']} site{'' if str(d['sites']) == '1' else 's'}" if d.get("sites") else "") for d in ds) or "not stated"
                res = tex(e.get("result")) or "no number in abstract"
            flag = "$^{\\ast}$" if c["key"] in bad else ""
            out.append(f"\\cite{{{c['key']}}}{flag} & {c['year']} & {c['modality']} & {data} & {res}\\\\")
        out += ["\\bottomrule", "\\end{longtable}", ""]
    (TAB / "corpus_works.tex").write_text("\n".join(out) + "\n")

    # dataset analysis
    emp = [E[k] for k, c in corpus.items() if c["modality"] != "General" and k in E and E[k].get("status") == "ok"]
    names, access, sizes, sites = Counter(), Counter(), [], []
    def is_public(d):
        return d.get("access") == "public" or canon(d.get("name") or "") != (d.get("name") or "").strip()
    for e in emp:
        ds = e.get("datasets") or []
        pub = [d for d in ds if is_public(d)]
        access["none described" if not ds else "public" if len(pub) == len(ds) else
               "institutional only" if not pub else "public and institutional"] += 1
        for nm in {canon(d["name"]) for d in pub}:
            names[nm] += 1
        st = [int(re.sub(r"[^\d]", "", str(d["sites"]))) for d in ds if d.get("sites") and re.sub(r"[^\d]", "", str(d["sites"]))]
        if st:
            sites.append(max(st))
        sz = [first_number(d.get("size")) for d in ds if first_number(d.get("size"))]
        if sz:
            sizes.append(max(sz))
    mets = Counter()
    for e in emp:
        r = e.get("result") or ""
        for m, rx in METRICS.items():
            if re.search(rx, r, re.I):
                mets[m] += 1
    num = {"CxEmpirical": len(emp), "CxHealth": sum(1 for c in corpus.values() if c["modality"] != "General"),
           "CxNotEmp": sum(1 for k, c in corpus.items() if c["modality"] != "General" and E.get(k, {}).get("status") == "not_empirical"),
           "CxNoSource": sum(1 for k, c in corpus.items() if c["modality"] != "General" and E.get(k, {}).get("status") == "no_source"),
           "CxPublic": access["public"], "CxInst": access["institutional only"], "CxBoth": access["public and institutional"],
           "CxNoData": access["none described"], "CxWithResult": sum(1 for e in emp if e.get("result")),
           "CxWithSize": sum(1 for e in emp if any(first_number(d.get("size")) for d in e.get("datasets") or [])),
           "CxMultiSite": sum(1 for s in sites if s >= 2), "CxSitesStated": len(sites)}
    if sizes:
        sizes.sort()
        q = lambda f: f"{sizes[int(f * (len(sizes) - 1))]:,}".replace(",", "{,}")
        num["CxSizeQOne"], num["CxSizeMedian"], num["CxSizeQThree"] = q(0.25), q(0.5), q(0.75)
        num["CxSizeUnderThousand"] = sum(1 for x in sizes if x < 1000)
    if sites:
        num["CxSitesMax"] = max(sites)
    for m in METRICS:
        num[f"CxMet{m}"] = mets[m]
    lines = ["% Generated by scripts/corpus_extract.py. Do not edit."] + [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in num.items()]
    (TAB / "corpus_numbers.tex").write_text("\n".join(lines) + "\n")
    top = names.most_common(15)
    L = ["\\begin{tabular}{@{}lr@{}}", "\\toprule", "Public dataset & Works\\\\", "\\midrule"]
    L += [f"{tex(n)} & {c}\\\\" for n, c in top if c >= 2]
    L += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "corpus_datasets.tex").write_text("\n".join(L) + "\n")
    print(json.dumps(num, indent=1), top, mets.most_common(), sep="\n")


ALIASES = [(r"mimic-cxr", "MIMIC-CXR"), (r"mimic[- ]?iii", "MIMIC-III"), (r"mimic[- ]?iv", "MIMIC-IV"), (r"\beicu", "eICU"),
           (r"chexpert", "CheXpert"), (r"chest-?x-?ray ?(8|14)|nih chest", "ChestX-ray8/14"), (r"\badni\b|alzheimer", "ADNI"),
           (r"messidor", "Messidor"), (r"eyepacs|kaggle diabetic", "EyePACS / Kaggle DR"), (r"\blits\b|liver tumor segmentation", "LiTS"),
           (r"decathlon", "Medical Segmentation Decathlon"), (r"\bi2b2\b", "i2b2/n2c2"), (r"openml", "OpenML"), (r"\buci\b", "UCI repository"), (r"uk biobank", "UK Biobank"),
           (r"\btcga\b|cancer genome atlas", "TCGA"), (r"brats", "BraTS"), (r"isic", "ISIC"), (r"camelyon", "Camelyon"),
           (r"physionet", "PhysioNet"), (r"kaggle", "Kaggle"), (r"ham10000", "HAM10000"),
           (r"luna16|lidc", "LIDC-IDRI / LUNA16"), (r"imagenet", "ImageNet")]


def canon(n):
    for rx, lab in ALIASES:
        if re.search(rx, n, re.I):
            return lab
    return n.strip()


def first_number(s):
    m = re.search(r"\d[\d,\.]*\s*(million|thousand|k\b)?", s or "", re.I)
    if not m:
        return None
    try:
        v = float(m.group(0).split()[0].replace(",", "").rstrip("kK"))
    except ValueError:
        return None
    unit = (m.group(1) or "").lower()
    if not unit and re.match(r"\d[\d,\.]*[kK]\b", m.group(0)):
        unit = "k"
    return int(v * (1e6 if unit == "million" else 1e3 if unit in ("thousand", "k") else 1))


if __name__ == "__main__":
    main()
