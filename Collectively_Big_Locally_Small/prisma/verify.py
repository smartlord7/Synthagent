"""Precision and negative predictive value of the final screening rules, from the single-reviewer
judgments in data/verify_judgments.json (two random samples of 100 records each, drawn from the
included and excluded sets of earlier rule versions; the final rules are re-applied here)."""
import gzip
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"


def main():
    dec = {}
    for line in gzip.open(DATA / "screened.jsonl.gz", "rt"):
        r = json.loads(line)
        dec[r["id"]] = r["decision"] == "include"
    J = json.load(open(DATA / "verify_judgments.json"))["judgments"]
    tp = sum(dec[j["id"]] and j["eligible"] for j in J)
    fp = sum(dec[j["id"]] and not j["eligible"] for j in J)
    tn = sum(not dec[j["id"]] and not j["eligible"] for j in J)
    fn = sum(not dec[j["id"]] and j["eligible"] for j in J)
    out = dict(n=len(J), tp=tp, fp=fp, tn=tn, fn=fn, precision=tp / (tp + fp), npv=tn / (tn + fn),
               sensitivity_in_sample=tp / (tp + fn))
    json.dump(out, open(DATA / "verify_summary.json", "w"), indent=1)
    print(out)


if __name__ == "__main__":
    main()
