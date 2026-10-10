"""Add exports from subscription databases to the search, so that they are deduplicated and screened
with the records from the open APIs.

Usage: python3 import_exports.py <name> <file> [<file> ...]
  <name>  scopus | wos | ieee | acm (any short name; becomes data/raw_<name>.jsonl.gz)
  <file>  RIS (.ris), BibTeX (.bib) or CSV (.csv) exports of the query of Section 5.4 run in that
          database. CSV columns are matched case-insensitively: title, abstract, doi, year,
          authors, source title / publication title / journal, document type.
Then add <name> to SOURCES in screen.py and rerun screen.py, classify.py, verify.py and flow.py.
"""
import csv
import gzip
import json
import re
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"


def from_ris(text, name):
    out, cur = [], {}
    for line in text.splitlines():
        m = re.match(r"^([A-Z][A-Z0-9])  - ?(.*)$", line)
        if not m:
            continue
        tag, val = m.groups()
        if tag == "ER":
            out.append(cur); cur = {}
        else:
            cur.setdefault(tag, []).append(val.strip())
    recs = []
    for c in out:
        g = lambda *tags: next((c[t][0] for t in tags if t in c), "")
        year = re.search(r"\d{4}", g("PY", "Y1", "DA") or "")
        recs.append(dict(source=name, id=g("AN", "ID") or g("DO"), doi=g("DO").lower(), title=g("TI", "T1"),
                         abstract=g("AB", "N2"), year=int(year.group()) if year else None,
                         venue=g("T2", "JO", "JF"), pubtypes=[g("TY")], language=[g("LA") or "eng"],
                         authors=[a.split(",")[0] for a in c.get("AU", [])]))
    return recs


def from_bib(text, name):
    recs = []
    for m in re.finditer(r"@(\w+)\{([^,]*),(.*?)\n\}", text, re.S):
        body = m.group(3)
        f = lambda k: re.sub(r"[{}]", "", (re.search(rf"\b{k}\s*=\s*[{{\"](.*?)[}}\"],?\s*\n", body, re.S, ) or [0, ""])[1])
        year = re.search(r"\d{4}", f("year"))
        recs.append(dict(source=name, id=m.group(2), doi=f("doi").lower(), title=" ".join(f("title").split()),
                         abstract=" ".join(f("abstract").split()), year=int(year.group()) if year else None,
                         venue=f("journal") or f("booktitle"), pubtypes=[m.group(1)], language=["eng"],
                         authors=[a.split(",")[0].strip() for a in f("author").split(" and ") if a]))
    return recs


def from_csv(path, name):
    recs = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            r = {k.lower().strip(): (v or "") for k, v in row.items() if k}
            g = lambda *keys: next((r[k] for k in keys if r.get(k)), "")
            year = re.search(r"\d{4}", g("year", "publication year", "publication_year"))
            recs.append(dict(source=name, id=g("eid", "accession number", "ut (unique wos id)", "doi"),
                             doi=g("doi").lower(), title=g("title", "document title", "article title"),
                             abstract=g("abstract"), year=int(year.group()) if year else None,
                             venue=g("source title", "publication title", "journal"),
                             pubtypes=[g("document type", "publication type")], language=[g("language of original document", "language") or "eng"],
                             authors=[a.strip() for a in re.split(r";", g("authors", "author full names"))]))
    return recs


def main():
    name, files = sys.argv[1], sys.argv[2:]
    recs = []
    for p in files:
        text = Path(p).read_text(encoding="utf-8", errors="replace")
        if p.lower().endswith(".ris"):
            recs += from_ris(text, name)
        elif p.lower().endswith(".bib"):
            recs += from_bib(text, name)
        else:
            recs += from_csv(p, name)
    with gzip.open(DATA / f"raw_{name}.jsonl.gz", "wt") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(recs)} records written to data/raw_{name}.jsonl.gz")


if __name__ == "__main__":
    main()
