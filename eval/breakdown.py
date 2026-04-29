import csv
import collections
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "eval/results_v2_n116.csv"
rows = list(csv.DictReader(open(path)))

def rate(rs, c):
    return sum(int(r[c]) for r in rs) / len(rs) * 100 if rs else 0.0

cols = [
    "baseline_retrieval_hit",
    "optimizer_retrieval_hit",
    "baseline_answer_hit",
    "optimizer_answer_hit",
]

print(f"=== N={len(rows)} ({path}) ===")
for c in cols:
    print(f"  {c:32s} {rate(rows, c):6.2f}%")
print(f"  delta retrieval: {rate(rows,'optimizer_retrieval_hit')-rate(rows,'baseline_retrieval_hit'):+.2f}")
print(f"  delta answer:    {rate(rows,'optimizer_answer_hit')-rate(rows,'baseline_answer_hit'):+.2f}")
print()

by_cat = collections.defaultdict(list)
for r in rows:
    by_cat[r["category"]].append(r)

print(f"  {'cat':12s} {'n':>3}  {'b_ret':>7} {'o_ret':>7} {'b_ans':>7} {'o_ans':>7}  {'dret':>6} {'dans':>6}")
for cat, rs in sorted(by_cat.items()):
    br = rate(rs, "baseline_retrieval_hit")
    orr = rate(rs, "optimizer_retrieval_hit")
    ba = rate(rs, "baseline_answer_hit")
    oa = rate(rs, "optimizer_answer_hit")
    print(f"  {cat:12s} {len(rs):>3}  {br:6.1f}% {orr:6.1f}% {ba:6.1f}% {oa:6.1f}%  {orr-br:+5.1f} {oa-ba:+5.1f}")
