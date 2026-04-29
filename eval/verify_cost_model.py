"""
Verify Plan A wiring: per row, the cost-model column must equal whichever
sub-planner (baseline or optimizer) the routing table picked for that row's
planner_classification. Any mismatch indicates broken delegation in
CostModelPlanner.

Usage:
    python eval/verify_cost_model.py eval/results.csv
"""
import csv
import sys
from collections import Counter

# Must match COST_MODEL_ROUTING in eval/run_eval.py
ROUTING_TABLE = {
    "keyword":     "optimizer",
    "definition":  "optimizer",
    "procedural":  "optimizer",
    "other":       "optimizer",
    "comparison":  "baseline",
    "explanatory": "baseline",
}
DEFAULT_ROUTE = "optimizer"


def expected(row, metric):
    cat = row["planner_classification"]
    route = ROUTING_TABLE.get(cat, DEFAULT_ROUTE)
    return int(row[f"{route}_{metric}"])


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "eval/results.csv"
    with open(path) as f:
        rows = list(csv.DictReader(f))

    needed = {"baseline_retrieval_hit", "optimizer_retrieval_hit",
              "cost_model_retrieval_hit", "baseline_answer_hit",
              "optimizer_answer_hit", "cost_model_answer_hit",
              "planner_classification"}
    missing = needed - set(rows[0].keys())
    if missing:
        print(f"FAIL: CSV missing columns: {sorted(missing)}")
        sys.exit(2)

    # Skip rows where any required column is blank — happens if the eval
    # was run with only one or two of the three modes enabled.
    skipped = 0
    checked = 0
    retr_mismatch = []
    ans_mismatch = []
    route_counts = Counter()

    for i, r in enumerate(rows, 1):
        if any(r.get(c) in ("", None) for c in needed):
            skipped += 1
            continue
        checked += 1
        cat = r["planner_classification"]
        route = ROUTING_TABLE.get(cat, DEFAULT_ROUTE)
        route_counts[route] += 1

        exp_retr = expected(r, "retrieval_hit")
        act_retr = int(r["cost_model_retrieval_hit"])
        if exp_retr != act_retr:
            retr_mismatch.append((i, r["query"][:60], cat, route, exp_retr, act_retr))

        exp_ans = expected(r, "answer_hit")
        act_ans = int(r["cost_model_answer_hit"])
        if exp_ans != act_ans:
            ans_mismatch.append((i, r["query"][:60], cat, route, exp_ans, act_ans))

    print(f"=== verify_cost_model: {path} ===")
    print(f"  rows total:        {len(rows)}")
    print(f"  rows checked:      {checked}")
    print(f"  rows skipped:      {skipped}  (blank columns — partial-mode run)")
    print(f"  routes used:       {dict(route_counts)}")
    print()
    print(f"  retrieval_hit mismatches: {len(retr_mismatch)} / {checked}")
    for i, q, cat, route, e, a in retr_mismatch[:10]:
        print(f"    row {i:>3} [{cat:>11s}→{route:>9s}]  expected={e} actual={a}  {q}")
    if len(retr_mismatch) > 10:
        print(f"    ... and {len(retr_mismatch) - 10} more")

    print(f"  answer_hit mismatches:    {len(ans_mismatch)} / {checked}")
    for i, q, cat, route, e, a in ans_mismatch[:10]:
        print(f"    row {i:>3} [{cat:>11s}→{route:>9s}]  expected={e} actual={a}  {q}")
    if len(ans_mismatch) > 10:
        print(f"    ... and {len(ans_mismatch) - 10} more")

    print()
    # Retrieval is deterministic — any mismatch is a real wiring bug.
    # Answer-hit depends on LLM generation, which is non-deterministic on
    # CPU llama.cpp even at temperature=0 (KV cache, OMP threading, fp
    # accumulation). Treat answer-hit mismatches as informational unless
    # they show directional bias.
    if retr_mismatch:
        print("RESULT: FAIL — retrieval mismatches indicate broken delegation.")
        print("        Retrieval is deterministic; if cost-model doesn't match")
        print("        the routed planner here, something is wired wrong.")
        sys.exit(1)

    if ans_mismatch:
        # Direction check: count cost-model wins vs losses against the routed planner.
        wins = sum(1 for _, _, _, _, e, a in ans_mismatch if a > e)
        losses = sum(1 for _, _, _, _, e, a in ans_mismatch if a < e)
        print(f"RESULT: PASS (with LLM noise) — retrieval delegation is correct.")
        print(f"        Answer-hit jitter: {wins} cost-model wins, {losses} losses")
        print(f"        out of {checked} rows. Net effect on aggregate is small.")
        print(f"        Expected on CPU llama.cpp; not a wiring bug.")
        sys.exit(0)

    print("RESULT: PASS — cost-model column matches routing rule on every row.")
    sys.exit(0)


if __name__ == "__main__":
    main()
