"""Deduplicate the search results and screen titles and abstracts against criteria I1, I2 and X1-X3.

Screening is rule-based so that it can be rerun exactly; each rule records the reason for exclusion
used in the PRISMA flow. A single reviewer checked two random samples of decisions (200 records) against
the criteria; verify.py computes precision and negative predictive value of the final rules from
those judgments. Two rules were revised after the checks and are not used: R0 (no scarcity term in
title or abstract), which excluded records the database had matched on author keywords, and a rule
for "limited data" in the sense of a knowledge gap, which also removed small-training-set studies.

Rules, in order (the first that applies decides):
  A1 automation: no abstract, or not in English;
  A2 automation: publication type outside scope by X1 (editorial, letter, comment, erratum, guideline,
     consensus statement, protocol, audio-visual, historical article) or retracted;
  R1 the record's only data-scarcity term refers to a resource-limited setting ("low-resource
     settings / countries / hospitals"), not to scarce data;
  R2 the record's only data-scarcity term is a sample-size limitation statement ("small sample
     size" in a sentence on limitations or future work), so the work does not address scarcity;
  R3 the only decentralization term refers to decentralized clinical trials or care delivery,
     not to learning from data held by several custodians;
  R4 X1 by title: reporting guideline, checklist, standard or policy statement;
  R5 not a learning work: no learning term in the title or abstract (the database matched the
     method block in keywords or MeSH only).
Records passing all rules are eligible (I1 or I2 at title/abstract level).
"""
import gzip
import json
import re
from collections import Counter, OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SOURCES = ["pubmed", "medrxiv", "arxiv_epmc"]

EXCLUDED_TYPES = {"Editorial", "Letter", "Comment", "Published Erratum", "Erratum", "Practice Guideline", "Guideline",
                  "Consensus Statement", "Consensus Development Conference", "Video-Audio Media", "Historical Article",
                  "Clinical Trial Protocol", "News", "Interview", "Retraction of Publication", "Retracted Publication"}

SCARCITY = OrderedDict([
    ("small data", r"\bsmall[- ]data\b"), ("small sample", r"\bsmall[- ]samples?\b"),
    ("limited data", r"\blimited (?:training |labell?ed |annotated )?data\b"),
    ("data scarcity", r"\bdata[- ]scar(?:city|ce)|scarc(?:e|ity of) (?:labell?ed |annotated |training )?data\b"),
    ("few-shot", r"\bfew[- ]shot\b"), ("low-resource", r"\blow[- ]resource\b"),
    ("rare disease", r"\brare (?:genetic )?diseases?\b"), ("label-efficient", r"\blabel[- ]efficien"),
    ("data-efficient", r"\bdata[- ]efficien"), ("federated", r"\bfederated\b"),
    ("decentralized", r"\bdecentrali[sz]ed\b"), ("privacy-preserving", r"\bprivacy[- ]preserv"),
])
LEARNING = (r"machine[- ]learning|deep[- ]learning|transfer[- ]learning|synthetic data|data augmentation|active learning|"
            r"semi[- ]supervised|self[- ]supervised|foundation models?|hierarchical (?:bayesian )?models?|neural network|"
            r"federated learning|few[- ]shot|meta[- ]learning|classif|\blearning\b|artificial intelligence|\bAI\b|"
            r"large language models?|\bLLMs?\b|transformer|convolutional|random forest|boosting|support vector|"
            r"predictive model|prediction model|bayesian")
LOW_RESOURCE_SETTING = re.compile(
    r"low[- ]resource (?:and [\w-]+ )?(?:settings?|countr|regions?|areas?|environments?|health|hospitals?|clinics?|"
    r"contexts?|communities|nations|facilities|populations)", re.I)
LIMITATION = re.compile(r"limitation|limited by|future (?:work|stud|research)|larger (?:sample|cohort|stud|dataset)|"
                        r"further (?:validation|stud|research)|caution|warrant", re.I)
DECENTRAL_TRIAL = re.compile(r"decentrali[sz]ed (?:clinical )?(?:trials?|care|health ?care|delivery|model of care)", re.I)
GUIDELINE_TITLE = re.compile(r"reporting guideline|checklist|\bstatement\b|position paper|policy|consensus", re.I)


def norm_title(t):
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())[:120]


def load():
    recs = []
    for s in SOURCES:
        p = DATA / f"raw_{s}.jsonl.gz"
        if p.exists():
            for line in gzip.open(p, "rt"):
                r = json.loads(line)
                if s == "arxiv_epmc":
                    r["source"] = "arXiv (Europe PMC)"
                recs.append(r)
    return recs


def dedupe(recs):
    """Keep the first record of each DOI or normalized title; sources are in priority order."""
    seen_doi, seen_title, out, dup = {}, {}, [], 0
    for r in recs:
        d, t = r.get("doi") or None, norm_title(r["title"])
        if (d and d in seen_doi) or (t and t in seen_title):
            dup += 1
            keep = seen_doi.get(d) if d and d in seen_doi else seen_title[t]
            keep.setdefault("also_in", []).append(r["source"])
            continue
        if d:
            seen_doi[d] = r
        if t:
            seen_title[t] = r
        out.append(r)
    return out, dup


def sentences(text):
    return re.split(r"(?<=[.;!?])\s+", text)


def scarcity_hits(r):
    """Scarcity terms that refer to scarce data, after removing the R1-R3 uses."""
    title, abstract = r["title"] or "", r["abstract"] or ""
    hits, reasons = set(), set()
    for name, pat in SCARCITY.items():
        for where, txt in (("title", title), ("abstract", abstract)):
            for s in sentences(txt):
                for m in re.finditer(pat, s, re.I):
                    if name == "low-resource" and LOW_RESOURCE_SETTING.match(s[m.start():]):
                        reasons.add("R1"); continue
                    if name == "small sample" and where == "abstract" and LIMITATION.search(s):
                        reasons.add("R2"); continue
                    if name == "decentralized" and DECENTRAL_TRIAL.match(s[m.start():]):
                        reasons.add("R3"); continue
                    hits.add(name)
    return hits, reasons


def screen(r):
    if not (r.get("abstract") or "").strip() or not any(l in ("eng", "en") for l in r.get("language") or ["eng"]):
        return "A1"
    if EXCLUDED_TYPES & set(r.get("pubtypes") or []):
        return "A2"
    hits, reasons = scarcity_hits(r)
    if not hits:
        for code in ("R1", "R2", "R3"):
            if code in reasons:
                return code
        return "include_keyword"  # the term is in the author keywords only: kept, as the database matched it
    if GUIDELINE_TITLE.search(r["title"] or ""):
        return "R4"
    if not re.search(LEARNING, (r["title"] or "") + " " + (r["abstract"] or ""), re.I):
        return "R5"
    r["scarcity_terms"] = sorted(hits)
    return "include"


def screen_record(r):
    d = screen(r)
    if d == "include_keyword":
        if GUIDELINE_TITLE.search(r["title"] or ""):
            return "R4"
        if not re.search(LEARNING, (r["title"] or "") + " " + (r["abstract"] or ""), re.I):
            return "R5"
        return "include"
    return d


def main():
    recs = load()
    by_source = Counter(r["source"] for r in recs)
    uniq, dup = dedupe(recs)
    decisions = Counter()
    for r in uniq:
        r["decision"] = screen_record(r)
        decisions[r["decision"]] += 1
    with gzip.open(DATA / "screened.jsonl.gz", "wt") as fh:
        for r in uniq:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    flow = dict(identified=dict(by_source), identified_total=len(recs), duplicates=dup, screened=len(uniq),
                automation_excluded=dict(A1=decisions["A1"], A2=decisions["A2"]),
                excluded=dict((k, decisions[k]) for k in ("R1", "R2", "R3", "R4", "R5")),
                eligible=decisions["include"])
    json.dump(flow, open(DATA / "flow_search.json", "w"), indent=1)
    print(json.dumps(flow, indent=1))


if __name__ == "__main__":
    main()
