"""Relative recall of the protocol query and of broadened variants against the hand-coded corpus.

The corpus works published since 2010, evaluated on health data and indexed in PubMed are split at
random into two halves (development and test); each query variant is counted on PubMed and its
recall on each half is computed with a PMID filter. Output: data/sensitivity.json.
"""
import json
import random
import time
from pathlib import Path

import requests

from search import HEALTH, METHOD, SCARCITY

DATA = Path(__file__).resolve().parent / "data"
DATE = "2010/01/01:2026/10/08[dp]"
EXTRA_SCARCITY = ['"sample size*"','"small cohort*"','"limited sample*"','"insufficient data"','"scarce data"','"data-limited"','"rare event*"','imbalanc*','multicenter','multicentre','"multi-center"','"multi-centre"','"multi-site"','multisite','"multi-institutional"','"external validation"','"externally validated"','generalizab*','generalisab*','"domain shift"','"dataset shift"','"distribution shift"','"domain adaptation"','"differential privacy"','"data sharing"','"distributed learning"','unlabeled','unlabelled','"weakly supervised"','annotation*','"label noise"','"noisy label*"','overfitting','"spurious correlation*"','"shortcut learning"','"hidden stratification"','underrepresent*','"model updating"','"temporal validation"']
EXTRA_METHOD = ['"artificial intelligence"','"neural network*"','"prediction model*"','"predictive model*"','"risk model*"','"prognostic model*"','classifier*','"convolutional"','transformer*','"large language model*"','"pretrained"','"pre-trained"','"meta-learning"','"contrastive"','"bayesian"','"random effects"']


def block(terms):
    return "(" + " OR ".join(f"{t}[tiab]" for t in terms) + ")"


def count(q):
    for attempt in range(5):
        try:
            r = requests.post("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                              data=dict(db="pubmed", term=q, retmax=0, retmode="json"), timeout=60)
            return int(r.json()["esearchresult"]["count"])
        except (requests.RequestException, ValueError, KeyError):
            time.sleep(2 ** attempt)
    raise RuntimeError("PubMed count failed")


def main():
    look = [json.loads(l) for l in open(DATA / "corpus_lookup.jsonl")]
    ins = [r for r in look if int(r["year"]) >= 2010 and r["modality"] != "General" and r.get("pmid")]
    random.seed(0)
    random.shuffle(ins)
    halves = dict(dev=ins[::2], test=ins[1::2])
    variants = dict(protocol=(SCARCITY, METHOD), scarcity_broadened=(SCARCITY + EXTRA_SCARCITY, METHOD),
                    method_broadened=(SCARCITY, METHOD + EXTRA_METHOD),
                    both_broadened=(SCARCITY + EXTRA_SCARCITY, METHOD + EXTRA_METHOD))
    out = dict(n_extra_scarcity=len(EXTRA_SCARCITY), n_extra_method=len(EXTRA_METHOD), n_pubmed_corpus=len(ins))
    for name, (s, m) in variants.items():
        q = f"{block(s)} AND {block(HEALTH)} AND {block(m)} AND {DATE}"
        res = dict(hits=count(q))
        for h, group in halves.items():
            ids = " OR ".join(f"{r['pmid']}[uid]" for r in group)
            res[f"recall_{h}"] = count(f"({q}) AND ({ids})") / len(group)
        out[name] = res
        print(name, res, flush=True)
    json.dump(out, open(DATA / "sensitivity.json", "w"), indent=1)


if __name__ == "__main__":
    main()
