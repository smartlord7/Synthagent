"""Adds the frozen GHB v5 rows (results/v5_networks_frozen_ckpt, dev/v5_networks.py --frozen) to
results/real_networks.json as "GHB v5 1" and "GHB v5 10"; the stored rows, GHB v4 included, are unchanged."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ghb  # noqa: E402

p = ghb.RESDIR / "real_networks.json"
d = json.load(open(p))
for nm in d["res"]:
    v5 = json.load(open(ghb.RESDIR / "v5_networks_frozen_ckpt" / f"{nm}.json"))
    for e in ("1", "10"):
        d["res"][nm][f"GHB v5 {e}"] = v5[f"GHB v5 {e}"]
json.dump(d, open(p, "w"), indent=1)
print("merged", list(d["res"]))
