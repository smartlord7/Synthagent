# Spend so far, from the token counts saved in each Claude run's checkpoint.
import json, glob, os
P = {"claude-haiku-5-5": (0.10, 0.50), "claude-haiku-4-5": (1, 5), "claude-sonnet-4-6": (3, 15), "claude-sonnet-5-5": (2, 10)}
M = {"haiku55": "claude-haiku-5-5", "haiku45": "claude-haiku-4-5", "sonnet46": "claude-sonnet-4-6", "sonnet55": "claude-sonnet-5-5"}
tot = 0
for d in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "claude", "*", ""))):
    name = os.path.basename(d.rstrip("/")); pin, pout = P[M[name.split("_")[0]]]
    try: c = json.load(open(d + "checkpoint.json"))
    except Exception: continue
    pt = ct = folds = 0
    for v in c.get("completed", {}).values():
        r = v.get("summary_row", {}); n = r.get("folds") or 0
        pt += (r.get("gen_stats_prompt_tokens_mean") or 0) * n; ct += (r.get("gen_stats_candidates_tokens_mean") or 0) * n; folds += n
    def walk(o):
        global pt, ct, folds
        if isinstance(o, dict):
            if "gen_stats_prompt_tokens" in o:
                pt += o.get("gen_stats_prompt_tokens") or 0; ct += o.get("gen_stats_candidates_tokens") or 0; folds += 1
                return
            for v in o.values(): walk(v)
        elif isinstance(o, list):
            for v in o: walk(v)
    walk(c.get("in_progress", {}))
    cost = pt / 1e6 * pin + ct / 1e6 * pout; tot += cost
    print(f"{name:20s} folds {folds:3d}/60  ${cost:6.2f}  projected ${cost / folds * 60 if folds else 0:6.2f}")
print(f"total so far ${tot:.2f}")
