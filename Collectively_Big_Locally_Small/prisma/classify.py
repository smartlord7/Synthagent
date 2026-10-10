"""Code the eligible search records with dictionaries, validate the coder on the hand-coded corpus,
and compare the field-level map with the corpus (Figure 7).

Each record is coded from its title and abstract:
  channels   P1-P6, every channel whose dictionary matches; the primary channel is the one with the
             highest score (title matches count twice), ties broken in channel order;
  classes    O1-C2, every class whose dictionary matches (multi-label, as in Table 3);
  modality   the first matching modality, else "General".
The dictionaries encode the definitions of Sections 3 and 4. They are a screening instrument, not a
replacement for full-text coding: the coded synthesis of Section 7 rests on the 290 hand-coded works.

Outputs: data/field_coded.jsonl.gz, data/classify_summary.json.
"""
import csv
import gzip
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE / "data"
sys.path.insert(0, str(ROOT / "scripts"))
from taxonomy import CHANNELS, CLASSES, POTENTIAL  # noqa: E402

CHANNEL_TERMS = {
    "P1": [r"federated", r"multi-?(?:site|cent(?:er|re)|institution)", r"data[- ]sharing", r"swarm learning",
           r"split learning", r"distributed (?:learning|training|deep)", r"collaborative (?:learning|training|model)",
           r"secure aggregation", r"differential(?:ly)? priva", r"homomorphic", r"pool(?:ed|ing) (?:data|analysis|cohort)",
           r"consorti", r"decentrali[sz]ed (?:learning|training|data)", r"privacy[- ]preserv", r"meta-analy",
           r"hierarchical (?:bayesian )?model", r"random[- ]effects", r"partial pooling", r"personali[sz]ed federated"],
    "P2": [r"transfer learning", r"pre-?train", r"fine-?tun", r"domain adaptation", r"multi-?task", r"foundation model",
           r"imagenet", r"large language model", r"\bllms?\b", r"zero-?shot", r"few-?shot", r"meta-?learning",
           r"knowledge distillation", r"vision[- ]language", r"\bclip\b", r"\bbert\b", r"\bgpt"],
    "P3": [r"prior knowledge", r"domain knowledge", r"expert knowledge", r"knowledge[- ]guided", r"knowledge graph",
           r"ontolog", r"physics[- ]informed", r"mechanistic", r"informative prior", r"bayesian prior", r"causal (?:graph|model|structure)",
           r"rule-?based", r"anatomical prior", r"shape prior", r"knowledge[- ]driven", r"theory[- ]guided"],
    "P4": [r"self-?supervised", r"semi-?supervised", r"contrastive", r"pseudo-?label", r"unlabell?ed", r"weakly[- ]supervised",
           r"masked (?:image|auto)", r"consistency regulari", r"unsupervised pre-?train", r"mean teacher"],
    "P5": [r"active learning", r"annotation (?:budget|effort|cost)", r"label(?:l)?ing (?:budget|effort|cost)", r"query strateg",
           r"crowdsourc", r"human-in-the-loop", r"adaptive (?:sampling|design)", r"sample selection", r"data acquisition"],
    "P6": [r"data augmentation", r"augment", r"synthetic (?:data|images?|patients?|records?)", r"generative adversarial",
           r"\bgans?\b", r"diffusion model", r"oversampl", r"\bsmote\b", r"variational auto-?encoder", r"data synthesis",
           r"generative model", r"mixup"],
}
CLASS_TERMS = {
    "O1": [r"rare (?:genetic )?diseases?", r"orphan", r"rare (?:cancers?|tumou?rs?|conditions?|disorders?)", r"small (?:patient )?cohorts?"],
    "O2": [r"rare events?", r"imbalanc", r"low (?:event|incidence|prevalence)", r"minority class", r"adverse events?",
           r"mortality", r"few events"],
    "O3": [r"subgroup", r"underrepresent", r"under-represent", r"minorit(?:y|ies) (?:group|population)", r"fairness",
           r"\bbias(?:es)?\b", r"ethnic", r"pediatric|paediatric"],
    "E1": [r"label", r"annotat", r"few-?shot", r"weakly", r"semi-?supervised", r"self-?supervised", r"active learning"],
    "E2": [r"noisy labels?", r"label noise", r"missing (?:data|values)", r"missingness", r"inter-?rater", r"measurement error",
           r"imputation"],
    "E3": [r"high[- ]dimension", r"small n,? large p", r"\bp\s*>>?\s*n\b", r"genomic", r"transcriptom", r"omics", r"feature selection",
           r"radiomic"],
    "T1": [r"temporal (?:shift|drift|validation)", r"dataset shift over time", r"concept drift", r"model updating",
           r"continual learning", r"obsolesc"],
    "T2": [r"domain (?:shift|adaptation|generali)", r"distribution shift", r"external validation", r"generali[sz]ab",
           r"out-of-distribution", r"site[- ]specific", r"heterogene", r"non-?i\.?i\.?d", r"covariate shift"],
    "C1": [r"federated", r"multi-?(?:site|cent(?:er|re)|institution)", r"data silos?", r"decentrali[sz]ed", r"distributed",
           r"collaborat", r"data[- ]sharing", r"cross-?silo"],
    "C2": [r"privacy", r"consent", r"\bgdpr\b", r"\bhipaa\b", r"confidential", r"de-?identif", r"anonymi[sz]"],
}
MODALITY_TERMS = [
    ("Imaging", r"imag|radiolog|\bmri\b|\bct\b|x-?ray|histopatholog|patholog(?:y|ical) slides?|ultrasound|fundus|dermoscop|mammogra|segmentation"),
    ("Signals", r"\becg\b|\beeg\b|electrocardiog|electroencephalog|wearable|sensor|physiological signal|\bppg\b"),
    ("Text", r"clinical notes?|natural language|\bnlp\b|text|large language model|\bllms?\b"),
    ("Omics", r"genom|transcriptom|proteom|omics|gene expression|single-cell|\bdna\b|\brna\b"),
    ("EHR", r"electronic health record|\behrs?\b|\bemrs?\b|electronic medical record|claims data|intensive care|\bicu\b"),
    ("Tabular", r"tabular|clinical variables|structured data|questionnaire|registry"),
]


def comp(d):
    return {k: [re.compile(p, re.I) for p in v] for k, v in d.items()}


CH, CL = comp(CHANNEL_TERMS), comp(CLASS_TERMS)
MOD = [(m, re.compile(p, re.I)) for m, p in MODALITY_TERMS]


def code(title, abstract):
    t, a = title or "", abstract or ""
    score = {c: sum(2 * bool(p.search(t)) + bool(p.search(a)) for p in pats) for c, pats in CH.items()}
    chans = [c for c, _ in CHANNELS if score[c] > 0]
    primary = max(chans, key=lambda c: (score[c], -int(c[1]))) if chans else "none"
    classes = [c for c, _ in CLASSES if any(p.search(t + " " + a) for p in CL[c])]
    modality = next((m for m, p in MOD if p.search(t + " " + a)), "General")
    return dict(primary=primary, channels=chans, classes=classes, modality=modality)


def norm_title(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())[:120]


def corpus_records():
    """The hand-coded corpus with titles and DOIs from references.bib."""
    bib = open(ROOT / "references.bib").read()
    meta = {}
    for m in re.finditer(r"@\w+\{([^,]+),(.*?)\n\}", bib, re.S):
        body = m.group(2)
        f = lambda k: (re.search(rf"\b{k}\s*=\s*\{{(.*?)\}},?\s*\n", body, re.S) or [None, ""])[1]
        meta[m.group(1)] = dict(title=re.sub(r"[{}\\]", "", f("title")), doi=f("doi").lower(),
                                abstract="", journal=f("journal") + f("booktitle"))
    rows = list(csv.DictReader(open(ROOT / "data" / "corpus.csv")))
    for r in rows:
        r.update(meta.get(r["key"], {}))
        r["classes"] = [c for c in r["scarcity_classes"].split(";") if c]
    return rows


def kappa(a, b):
    labels = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def cells(records, chan_key, class_key):
    c = Counter()
    for r in records:
        for cl in r[class_key]:
            for ch in r[chan_key]:
                if ch != "none":
                    c[(cl, ch)] += 1
    return c


def main():
    recs = [json.loads(l) for l in gzip.open(DATA / "screened.jsonl.gz", "rt")]
    elig = [r for r in recs if r["decision"] == "include"]
    for r in elig:
        r["code"] = code(r["title"], r["abstract"])
    with gzip.open(DATA / "field_coded.jsonl.gz", "wt") as fh:
        for r in elig:
            fh.write(json.dumps(dict(id=r["id"], source=r["source"], doi=r["doi"], year=r["year"], title=r["title"],
                                     **r["code"]), ensure_ascii=False) + "\n")

    # --- recall of the search against the hand-coded corpus -------------------------------------
    corpus = corpus_records()
    by_doi = {r["doi"]: r for r in recs if r.get("doi")}
    by_title = {norm_title(r["title"]): r for r in recs}
    in_scope = [c for c in corpus if int(c["year"]) >= 2010 and c["modality"] != "General"]
    found, eligible_found, matched = [], [], []
    look_pre = {x["key"]: x for x in map(json.loads, open(DATA / "corpus_lookup.jsonl"))}
    by_pmid = {r["id"]: r for r in recs if r["source"] == "PubMed"}
    for c in in_scope:
        r = by_doi.get(c["doi"]) if c["doi"] else None
        r = r or by_title.get(norm_title(c["title"])) or by_pmid.get(look_pre.get(c["key"], {}).get("pmid") or "")
        r = r or by_title.get(norm_title(look_pre.get(c["key"], {}).get("title")))
        if r:
            found.append(c)
            if r["decision"] == "include":
                eligible_found.append(c)
                matched.append((c, r))

    # --- validation of the dictionary coder on every hand-coded work with an abstract in Europe PMC --
    look = {x["key"]: x for x in map(json.loads, open(DATA / "corpus_lookup.jsonl"))}
    val = [(c, look[c["key"]]) for c in corpus if look.get(c["key"], {}).get("abstract")]
    matched = val
    auto = [code(r["title"], r["abstract"]) for _, r in matched]
    hand_p = [c["primary_channel"] for c, _ in matched]
    auto_p = [a["primary"] for a in auto]
    hit_any = sum(h in a["channels"] or (h == "none" and not a["channels"]) for h, a in zip(hand_p, auto))
    class_recall = [len(set(c["classes"]) & set(a["classes"])) / max(1, len(c["classes"])) for (c, _), a in zip(matched, auto)]

    # --- field-level map against the corpus map (Figure 7) ---------------------------------------
    field = cells([dict(ch=r["code"]["channels"], cl=r["code"]["classes"]) for r in elig], "ch", "cl")
    corp = cells([dict(ch=[c["primary_channel"]] + ([c["secondary_channel"]] if c["secondary_channel"] else []),
                       cl=c["classes"]) for c in corpus], "ch", "cl")
    allcells = [(cl, ch) for cl, _ in CLASSES for ch, _ in CHANNELS]
    ranks = {cell: sum(field[c2] < field[cell] for c2 in allcells) / len(allcells) for cell in allcells}
    gaps = [(CLASSES[i][0], CHANNELS[j][0]) for i in range(len(CLASSES)) for j in range(len(CHANNELS))
            if POTENTIAL[i][j] >= 2 and corp[(CLASSES[i][0], CHANNELS[j][0])] <= 2]
    gap_report = [dict(cell=f"{cl}x{ch}", corpus=corp[(cl, ch)], field=field[(cl, ch)],
                       field_share=field[(cl, ch)] / len(elig), field_percentile=ranks[(cl, ch)]) for cl, ch in gaps]
    # rank correlation of cell counts, corpus vs field
    from scipy.stats import spearmanr
    rho = spearmanr([corp[c] for c in allcells], [field[c] for c in allcells]).statistic

    look_all = {x["key"]: x for x in map(json.loads, open(DATA / "corpus_lookup.jsonl"))}
    indexed = [c for c in in_scope if look_all.get(c["key"], {}).get("source") in ("MED", "PPR")]
    found_keys = {c["key"] for c in found}
    prim = Counter(r["code"]["primary"] for r in elig)
    years = Counter(r["year"] for r in elig)
    summary = dict(
        eligible=len(elig),
        recall=dict(corpus_in_scope=len(in_scope), retrieved=len(found), retrieved_and_eligible=len(eligible_found),
                    recall_retrieved=len(found) / len(in_scope), recall_eligible=len(eligible_found) / len(in_scope),
                    indexed_in_searched_sources=len(indexed),
                    retrieved_of_indexed=sum(c["key"] in found_keys for c in indexed),
                    recall_of_indexed=sum(c["key"] in found_keys for c in indexed) / len(indexed)),
        coder_validation=dict(n=len(matched), primary_agreement=sum(x == y for x, y in zip(hand_p, auto_p)) / len(matched),
                              primary_kappa=kappa(hand_p, auto_p), hand_primary_among_auto=hit_any / len(matched),
                              class_recall=sum(class_recall) / len(class_recall)),
        field_primary_channel={k: prim[k] for k in ["P1", "P2", "P3", "P4", "P5", "P6", "none"]},
        field_modality=dict(Counter(r["code"]["modality"] for r in elig).most_common()),
        field_years={str(y): years[y] for y in sorted(y for y in years if y)},
        cell_spearman=rho, gaps=gap_report,
        gaps_in_bottom_third=sum(g["field_percentile"] < 1 / 3 for g in gap_report),
        field_cells={f"{cl}x{ch}": field[(cl, ch)] for cl, ch in allcells},
    )
    json.dump(summary, open(DATA / "classify_summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "field_cells"}, indent=1))


if __name__ == "__main__":
    main()
