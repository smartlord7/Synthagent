"""Run the search of Section 5.4 on the sources reachable without a licence and save the raw records.

Sources executed:
  PubMed       NCBI E-utilities, title/abstract fields;
  medRxiv      through Europe PMC (preprint source PPR), title/abstract;
  arXiv        the arXiv API (all fields), with truncated terms expanded to their common forms
               because the API has no wildcard; the Europe PMC arXiv subset is also queried and
               merged at deduplication.
Sources not executed here, because they need a subscription or an API key: Scopus, Web of Science,
IEEE Xplore and the ACM Digital Library. Exports from them (RIS or CSV) can be added with
import_exports.py and are then deduplicated and screened with the rest.

Output: prisma/data/raw_<source>.jsonl.gz, one record per line, and prisma/data/search_log.json with
the exact query strings, the date of the search and the hit counts.
"""
import gzip
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SEARCH_DATE = "2026-10-08"
START = "2010-01-01"

SCARCITY = ['"small data"', '"small sample*"', '"limited data"', '"data scarcity"', '"few-shot"', '"low-resource"',
            '"rare disease*"', '"label-efficient"', '"data-efficient"', "federated", "decentralized", "decentralised",
            '"privacy-preserving"']
HEALTH = ["health*", "clinical", "medic*", "hospital*", "patient*"]
METHOD = ['"machine learning"', '"deep learning"', '"transfer learning"', '"synthetic data"', '"data augmentation"',
          '"active learning"', '"semi-supervised"', '"self-supervised"', '"foundation model*"', '"hierarchical model*"']
# The protocol writes decentrali*ed; neither PubMed nor Europe PMC supports an inner wildcard, so both
# spellings are listed.


def pubmed_query():
    block = lambda terms: "(" + " OR ".join(f"{t}[tiab]" for t in terms) + ")"
    return (f"{block(SCARCITY)} AND {block(HEALTH)} AND {block(METHOD)} AND "
            f"{START.replace('-', '/')}:{SEARCH_DATE.replace('-', '/')}[dp]")


def epmc_query(publisher):
    block = lambda terms: "(" + " OR ".join(f"TITLE_ABS:{t}" for t in terms) + ")"
    return (f"{block(SCARCITY)} AND {block(HEALTH)} AND {block(METHOD)} AND SRC:PPR AND PUBLISHER:\"{publisher}\" "
            f"AND FIRST_PDATE:[{START} TO {SEARCH_DATE}]")


ARXIV_EXPAND = {
    '"small sample*"': ['"small sample"', '"small samples"'], '"rare disease*"': ['"rare disease"', '"rare diseases"'],
    "health*": ["health", "healthcare", "health-care"], "medic*": ["medical", "medicine"],
    "hospital*": ["hospital", "hospitals"], "patient*": ["patient", "patients"],
    '"foundation model*"': ['"foundation model"', '"foundation models"'],
    '"hierarchical model*"': ['"hierarchical model"', '"hierarchical models"'],
}


def arxiv_query():
    def block(terms):
        ex = [y for t in terms for y in ARXIV_EXPAND.get(t, [t])]
        return "(" + " OR ".join(f"all:{t}" for t in ex) + ")"
    return (f"{block(SCARCITY)} AND {block(HEALTH)} AND {block(METHOD)} AND "
            f"submittedDate:[{START.replace('-', '')}0000 TO {SEARCH_DATE.replace('-', '')}2359]")


def search_arxiv(page=500):
    q = arxiv_query()
    ns = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom",
          "o": "http://a9.com/-/spec/opensearch/1.1/"}
    out, start, total = [], 0, None
    while total is None or start < total:
        for attempt in range(8):
            try:
                r = requests.get("https://export.arxiv.org/api/query", timeout=180,
                                 params=dict(search_query=q, start=start, max_results=page,
                                             sortBy="submittedDate", sortOrder="ascending"))
                root = ET.fromstring(r.text) if r.status_code == 200 else None
                if root is not None and (root.findall("a:entry", ns) or start >= int(root.find("o:totalResults", ns).text)):
                    break
            except (requests.RequestException, ET.ParseError):
                pass
            time.sleep(5 * 2 ** attempt)
        else:
            raise RuntimeError(f"arXiv failed at start={start}")
        total = int(root.find("o:totalResults", ns).text)
        for e in root.findall("a:entry", ns):
            out.append(dict(
                source="arXiv", id=text(e.find("a:id", ns)).rsplit("/abs/", 1)[-1],
                doi=text(e.find("x:doi", ns)).lower(), title=" ".join(text(e.find("a:title", ns)).split()),
                abstract=" ".join(text(e.find("a:summary", ns)).split()),
                year=int(text(e.find("a:published", ns))[:4]), venue="arXiv",
                pubtypes=["preprint"], language=["eng"],
                authors=[text(a.find("a:name", ns)).split()[-1] for a in e.findall("a:author", ns)],
            ))
        start += page
        time.sleep(3)
    return q, total, out


def get(url, **params):
    for attempt in range(6):
        try:
            r = requests.get(url, params=params, timeout=60)
            if r.status_code == 200:
                return r
        except requests.RequestException:
            pass
        time.sleep(2 ** attempt)
    raise RuntimeError(f"failed: {url}")


def text(el):
    return "".join(el.itertext()).strip() if el is not None else ""


def search_pubmed():
    q = pubmed_query()
    eu = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    r = get(eu + "esearch.fcgi", db="pubmed", term=q, usehistory="y", retmax=0, retmode="json").json()["esearchresult"]
    count, webenv, key = int(r["count"]), r["webenv"], r["querykey"]
    out = []
    for start in range(0, count, 400):
        xml = get(eu + "efetch.fcgi", db="pubmed", WebEnv=webenv, query_key=key, retstart=start, retmax=400,
                  retmode="xml").text
        for art in ET.fromstring(xml).iter("PubmedArticle"):
            mc = art.find("MedlineCitation")
            a = mc.find("Article")
            year = text(a.find("Journal/JournalIssue/PubDate/Year")) or text(
                a.find("Journal/JournalIssue/PubDate/MedlineDate"))[:4]
            ids = {i.get("IdType"): text(i) for i in art.findall("PubmedData/ArticleIdList/ArticleId")}  # not the references
            out.append(dict(
                source="PubMed", id=text(mc.find("PMID")), doi=ids.get("doi", "").lower(),
                title=text(a.find("ArticleTitle")),
                abstract=" ".join(text(x) for x in a.findall("Abstract/AbstractText")),
                year=int(year) if year.isdigit() else None,
                venue=text(a.find("Journal/Title")),
                pubtypes=[text(p) for p in a.findall("PublicationTypeList/PublicationType")],
                language=[text(x) for x in a.findall("Language")],
                authors=[text(x.find("LastName")) for x in a.findall("AuthorList/Author") if x.find("LastName") is not None],
            ))
        time.sleep(0.4)
    return q, count, out


def search_epmc(publisher):
    q = epmc_query(publisher)
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    cursor, out, count = "*", [], None
    while True:
        r = get(url, query=q, format="json", resultType="core", pageSize=1000, cursorMark=cursor).json()
        count = r["hitCount"]
        for x in r.get("resultList", {}).get("result", []):
            out.append(dict(
                source=publisher, id=x.get("id"), doi=(x.get("doi") or "").lower(), title=x.get("title", ""),
                abstract=re.sub(r"<[^>]+>", "", x.get("abstractText", "") or ""),
                year=int(x["pubYear"]) if x.get("pubYear", "").isdigit() else None,
                venue=publisher, pubtypes=(x.get("pubTypeList") or {}).get("pubType", []),
                language=[x.get("language", "eng")],
                authors=[a.get("lastName", "") for a in (x.get("authorList") or {}).get("author", [])],
            ))
        nxt = r.get("nextCursorMark")
        if not nxt or nxt == cursor or not r.get("resultList", {}).get("result"):
            break
        cursor = nxt
    return q, count, out


def main():
    DATA.mkdir(exist_ok=True)
    log = dict(search_date=SEARCH_DATE, run_on=str(date.today()), sources={})
    only = set(sys.argv[1:])
    jobs = [("pubmed", search_pubmed), ("medrxiv", lambda: search_epmc("medRxiv")),
            ("arxiv_epmc", lambda: search_epmc("arXiv")), ("arxiv", search_arxiv)]
    if (DATA / "search_log.json").exists():
        log["sources"] = json.load(open(DATA / "search_log.json"))["sources"]
    for name, f in jobs:
        if only and name not in only:
            continue
        q, count, recs = f()
        with gzip.open(DATA / f"raw_{name}.jsonl.gz", "wt") as fh:
            for rec in recs:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log["sources"][name] = dict(query=q, hits=count, retrieved=len(recs))
        print(f"{name}: {count} hits, {len(recs)} retrieved")
    log["not_executed"] = {
        "Scopus": "requires an Elsevier API key and institutional subscription",
        "Web of Science": "requires a Clarivate subscription (Web of Science Starter or Expanded API key)",
        "IEEE Xplore": "requires an IEEE Xplore API key",
        "ACM Digital Library": "no public search API; export results as BibTeX from dl.acm.org",
    }
    json.dump(log, open(DATA / "search_log.json", "w"), indent=1)


if __name__ == "__main__":
    main()
