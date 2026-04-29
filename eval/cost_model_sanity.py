"""
Plan C: project what a cost-model planner *would* score, by post-processing
an existing baseline+optimizer eval CSV. No new eval run required.

For each row, pick baseline_* or optimizer_* hits based on a routing table
keyed on planner_classification (the heuristic's predicted category). Report
the projected hit rate alongside baseline-only and optimizer-only rates.

We also report an "oracle" projection routed by the gold `category` column,
to bound how much of any gap is misclassification vs. table miscalibration.
"""
import csv
import collections
import sys

# Empirical routing table from N=116 run (eval/results_v2_n116.csv):
#   keyword     +10.0  → composite wins → optimizer
#   definition  +10.0  → optimizer
#   procedural   +5.0  → optimizer
#   other       +12.5  → optimizer
#   comparison   -4.2  → baseline
#   explanatory  -4.2  → baseline
ROUTING_TABLE = {
    "keyword":     "optimizer",
    "definition":  "optimizer",
    "procedural":  "optimizer",
    "other":       "optimizer",
    "comparison":  "baseline",
    "explanatory": "baseline",
}
DEFAULT_ROUTE = "optimizer"


def pick(row, mode_col_prefix, route):
    col = f"{route}_{mode_col_prefix}"
    return int(row[col])


def project(rows, route_key):
    """Return per-row picks (retrieval_hit, answer_hit) given a routing key."""
    out = []
    for r in rows:
        cat = r[route_key]
        choice = ROUTING_TABLE.get(cat, DEFAULT_ROUTE)
        out.append({
            **r,
            "cost_model_retrieval_hit": pick(r, "retrieval_hit", choice),
            "cost_model_answer_hit": pick(r, "answer_hit", choice),
            "cost_model_choice": choice,
        })
    return out


def rate(rs, c):
    return sum(int(r[c]) for r in rs) / len(rs) * 100 if rs else 0.0


def report(label, rows):
    print(f"=== {label} (N={len(rows)}) ===")
    print(f"  baseline_retrieval_hit    {rate(rows, 'baseline_retrieval_hit'):6.2f}%")
    print(f"  optimizer_retrieval_hit   {rate(rows, 'optimizer_retrieval_hit'):6.2f}%")
    print(f"  cost_model_retrieval_hit  {rate(rows, 'cost_model_retrieval_hit'):6.2f}%")
    print(f"  baseline_answer_hit       {rate(rows, 'baseline_answer_hit'):6.2f}%")
    print(f"  optimizer_answer_hit      {rate(rows, 'optimizer_answer_hit'):6.2f}%")
    print(f"  cost_model_answer_hit     {rate(rows, 'cost_model_answer_hit'):6.2f}%")
    print(f"  delta vs baseline (ans)   {rate(rows,'cost_model_answer_hit')-rate(rows,'baseline_answer_hit'):+6.2f}")
    print(f"  delta vs optimizer (ans)  {rate(rows,'cost_model_answer_hit')-rate(rows,'optimizer_answer_hit'):+6.2f}")
    print()


def per_category(rows):
    by_cat = collections.defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    print(f"  {'cat':12s} {'n':>3}  {'b_ans':>7} {'o_ans':>7} {'cm_ans':>7}  {'choice':>10}")
    for cat, rs in sorted(by_cat.items()):
        ba = rate(rs, "baseline_answer_hit")
        oa = rate(rs, "optimizer_answer_hit")
        cma = rate(rs, "cost_model_answer_hit")
        # Most-common routing choice in this gold category
        choices = collections.Counter(r["cost_model_choice"] for r in rs)
        common_choice = choices.most_common(1)[0][0]
        print(f"  {cat:12s} {len(rs):>3}  {ba:6.1f}% {oa:6.1f}% {cma:6.1f}%  {common_choice:>10}")
    print()


def confusion(rows):
    """Show how often the heuristic classifier disagrees with the gold label."""
    print("  classifier confusion (gold → predicted):")
    pairs = collections.Counter((r["category"], r["planner_classification"]) for r in rows)
    misses = [(g, p, n) for (g, p), n in pairs.items() if g != p]
    if not misses:
        print("    (no mismatches)")
        return
    for g, p, n in sorted(misses, key=lambda x: -x[2]):
        print(f"    {g:12s} → {p:12s}  n={n}")
    print()


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "eval/results_v2_n116.csv"
    with open(path) as f:
        rows = list(csv.DictReader(f))

    # Production cost model: route by heuristic classification (the planner
    # has no access to the gold category at inference time).
    prod = project(rows, "planner_classification")
    report(f"Cost model (route by planner_classification) — {path}", prod)
    per_category(prod)

    # Oracle: route by gold category. Upper bound; gap to prod = classifier loss.
    oracle = project(rows, "category")
    report(f"Oracle cost model (route by gold category) — {path}", oracle)

    confusion(rows)


if __name__ == "__main__":
    main()
