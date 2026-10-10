"""Look up every hand-coded corpus work in Europe PMC (which covers PubMed and the main preprint
servers) by DOI, then by title, and save its abstract and indexing status.

Used to (i) tell whether a corpus work missed by the search is indexed in the searched databases at
all, and (ii) validate the dictionary coder of classify.py on a larger set of hand-coded works.
Output: data/corpus_lookup.jsonl.
"""
import json
import re
import time
from pathlib import Path

import requests

from classify import corpus_records, norm_title

HERE = Path(__file__).resolve().parent
URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


def query(q):
    for attempt in range(5):
        try:
            r = requests.get(URL, params=dict(query=q, format="json", resultType="core", pageSize=5), timeout=60)
            if r.status_code == 200:
                return r.json().get("resultList", {}).get("result", [])
        except requests.RequestException:
            pass
        time.sleep(2 ** attempt)
    return []


def main():
    out = []
    for c in corpus_records():
        hit = None
        if c["doi"]:
            res = query(f'DOI:"{c["doi"]}"')
            hit = res[0] if res else None
        if hit is None and c["title"]:
            t = re.sub(r'["\\:()\[\]]', " ", c["title"])
            for x in query(f'TITLE:"{t}"'):
                if norm_title(x.get("title")) == norm_title(c["title"]):
                    hit = x
                    break
        rec = dict(key=c["key"], year=c["year"], modality=c["modality"], found=hit is not None)
        if hit:
            rec.update(source=hit.get("source"), pmid=hit.get("pmid"), title=hit.get("title"),
                       abstract=re.sub(r"<[^>]+>", "", hit.get("abstractText", "") or ""),
                       publisher=hit.get("bookOrReportDetails", {}).get("publisher", "") or hit.get("journalInfo", {}).get("journal", {}).get("title", ""))
        out.append(rec)
        time.sleep(0.15)
    with open(HERE / "data" / "corpus_lookup.jsonl", "w") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(sum(r["found"] for r in out), "of", len(out), "found;",
          sum(r["found"] and r.get("source") == "MED" for r in out), "in PubMed")


if __name__ == "__main__":
    main()
