"""
Compare two eval CSVs side-by-side per category.

Helps spot which per-category numbers are stable across runs vs. which are
swinging due to LLM generation noise. The routing table baked into the cost
model was derived from a specific run; if a category's baseline/optimizer
gap flips sign in a later run, the routing decision for that category is
noise, not signal.

Usage:
    python eval/compare_runs.py eval/results_v2_n116.csv eval/results_v3_n116_with_cost_model.csv
"""
import csv
import collections
import sys


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def rate(rs, c):
    return sum(int(r[c] or 0) for r in rs) / len(rs) * 100 if rs else 0.0


def by_cat(rows):
    d = collections.defaultdict(list)
    for r in rows:
        d[r["category"]].append(r)
    return d


def main():
    if len(sys.argv) < 3:
        print("usage: compare_runs.py RUN_A.csv RUN_B.csv")
        sys.exit(2)
    path_a, path_b = sys.argv[1], sys.argv[2]
    rows_a = load(path_a)
    rows_b = load(path_b)
    cats_a, cats_b = by_cat(rows_a), by_cat(rows_b)
    cats = sorted(set(cats_a) | set(cats_b))

    label_a = path_a.split("/")[-1]
    label_b = path_b.split("/")[-1]

    print(f"A = {label_a}  (N={len(rows_a)})")
    print(f"B = {label_b}  (N={len(rows_b)})")
    print()
    print(f"  {'cat':12s} {'n':>3}  {'A b_ans':>8} {'B b_ans':>8}  {'A o_ans':>8} {'B o_ans':>8}  {'A gap':>7} {'B gap':>7}")
    print(f"  {'-'*12} {'-'*3}  {'-'*8} {'-'*8}  {'-'*8} {'-'*8}  {'-'*7} {'-'*7}")
    for cat in cats:
        ra = cats_a.get(cat, [])
        rb = cats_b.get(cat, [])
        n = max(len(ra), len(rb))
        a_b = rate(ra, "baseline_answer_hit")  if ra else 0.0
        b_b = rate(rb, "baseline_answer_hit")  if rb else 0.0
        a_o = rate(ra, "optimizer_answer_hit") if ra else 0.0
        b_o = rate(rb, "optimizer_answer_hit") if rb else 0.0
        a_gap = a_o - a_b
        b_gap = b_o - b_b
        print(f"  {cat:12s} {n:>3}  {a_b:7.1f}% {b_b:7.1f}%  {a_o:7.1f}% {b_o:7.1f}%  {a_gap:+6.1f} {b_gap:+6.1f}")

    print()
    print("  Legend: gap = optimizer − baseline. If a category's gap sign flips between A and B,")
    print("  the routing decision for that category is within LLM-noise range, not robust signal.")
    print()
    if "cost_model_answer_hit" in (rows_b[0] if rows_b else {}):
        cm = rate(rows_b, "cost_model_answer_hit")
        bb = rate(rows_b, "baseline_answer_hit")
        bo = rate(rows_b, "optimizer_answer_hit")
        print(f"  B has cost_model column: cm_ans={cm:.2f}%  (vs B baseline {bb:.2f}%, B optimizer {bo:.2f}%)")
        print(f"  Δ cost-model − baseline:  {cm-bb:+.2f}")
        print(f"  Δ cost-model − optimizer: {cm-bo:+.2f}")


if __name__ == "__main__":
    main()
