"""Download open-access TCGA clinical data (NCI GDC API) for the given projects into data/<dir>/<project>.json, with
the fields used by real_v3.py. Usage: python3 fetch_tcga.py <dir> TCGA-XXXX [...]"""
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

FIELDS = ("submitter_id,tissue_source_site.code,tissue_source_site.name,demographic.vital_status,demographic.days_to_death,"
          "demographic.age_at_index,demographic.sex_at_birth,demographic.race,diagnoses.days_to_last_follow_up,"
          "diagnoses.ajcc_pathologic_stage,diagnoses.ajcc_pathologic_t,diagnoses.ajcc_pathologic_n,diagnoses.ajcc_pathologic_m,diagnoses.figo_stage,"
          "diagnoses.prior_malignancy,diagnoses.tumor_grade,diagnoses.diagnosis_is_primary_disease")


def fetch(project):
    filt = {"op": "in", "content": {"field": "project.project_id", "value": [project]}}
    q = urllib.parse.urlencode({"filters": json.dumps(filt), "fields": FIELDS, "size": 5000, "format": "json"})
    with urllib.request.urlopen("https://api.gdc.cancer.gov/cases?" + q, timeout=120) as r:
        return json.load(r)["data"]["hits"]


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data" / sys.argv[1]
    out.mkdir(exist_ok=True)
    for p in sys.argv[2:]:
        hits = fetch(p)
        json.dump(hits, open(out / f"{p}.json", "w"))
        print(p, len(hits), flush=True)
