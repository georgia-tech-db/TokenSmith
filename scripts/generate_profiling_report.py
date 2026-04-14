"""
generate_profiling_report.py

Generates charts and a summary report from profiling data produced by
profile_retrieval_plans.py.

Outputs:
  - latency_comparison.png:   Bar chart of avg latency per plan
  - quality_vs_latency.png:   Scatter plot of hit_rate vs latency per plan
  - per_query_type_heatmap.png: Heatmap of plan quality by query type
  - profiling_report.txt:     Text summary with lookup table

Usage:
  python -m scripts.generate_profiling_report
  python -m scripts.generate_profiling_report --input results/profiling --output results/profiling/charts
"""

import argparse
import json
import pathlib
import sys
from collections import defaultdict

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Warning: matplotlib not installed. Generating text report only.")


def load_data(input_dir: pathlib.Path):
    results_path = input_dir / "profiling_results.json"
    summary_path = input_dir / "plan_summary.json"
    per_type_path = input_dir / "per_query_type_summary.json"
    lookup_path = input_dir / "lookup_table.json"

    with open(results_path) as f:
        results = json.load(f)
    with open(summary_path) as f:
        summary = json.load(f)
    with open(per_type_path) as f:
        per_type = json.load(f)
    with open(lookup_path) as f:
        lookup = json.load(f)

    return results, summary, per_type, lookup


# ── Chart 1: Latency comparison bar chart ──────────────────────────────────

def plot_latency_comparison(summary, output_dir):
    if not HAS_MPL:
        return

    plans = list(summary.keys())
    labels = [summary[p]["label"] for p in plans]
    avg_lat = [summary[p]["avg_latency_ms"] for p in plans]
    p50_lat = [summary[p]["p50_latency_ms"] for p in plans]
    p95_lat = [summary[p]["p95_latency_ms"] for p in plans]

    # Color: no-rerank plans in blue, rerank plans in orange
    colors = ["#4C72B0" if "rerank" not in p else "#DD8452" for p in plans]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(plans))
    bars = ax.bar(x, avg_lat, color=colors, edgecolor="white", linewidth=0.5)

    # Add p95 as error bars (whiskers)
    for i, (avg, p95) in enumerate(zip(avg_lat, p95_lat)):
        ax.plot([i, i], [avg, p95], color="black", linewidth=1.5)
        ax.plot([i - 0.1, i + 0.1], [p95, p95], color="black", linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Average Retrieval Latency by Plan (whiskers = P95)")

    # Legend
    blue_patch = mpatches.Patch(color="#4C72B0", label="Without Reranking")
    orange_patch = mpatches.Patch(color="#DD8452", label="With Reranking")
    ax.legend(handles=[blue_patch, orange_patch], loc="upper left")

    # Add value labels on bars
    for bar, val in zip(bars, avg_lat):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{val:.0f}ms", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = output_dir / "latency_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Chart 2: Quality vs Latency scatter ───────────────────────────────────

def plot_quality_vs_latency(summary, output_dir):
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    for plan_name, stats in summary.items():
        is_rerank = "rerank" in plan_name
        color = "#DD8452" if is_rerank else "#4C72B0"
        marker = "D" if is_rerank else "o"
        ax.scatter(
            stats["avg_latency_ms"], stats["avg_hit_rate"],
            s=120, c=color, marker=marker, edgecolors="black", linewidths=0.5,
            zorder=3
        )
        # Label each point
        ax.annotate(
            stats["label"],
            (stats["avg_latency_ms"], stats["avg_hit_rate"]),
            textcoords="offset points", xytext=(8, 5),
            fontsize=8, alpha=0.85,
        )

    ax.set_xlabel("Average Latency (ms)")
    ax.set_ylabel("Average Hit Rate @ 10")
    ax.set_title("Retrieval Quality vs Latency Tradeoff")
    ax.grid(True, alpha=0.3)

    # Pareto frontier (connect points that aren't dominated)
    points = sorted(
        [(summary[p]["avg_latency_ms"], summary[p]["avg_hit_rate"]) for p in summary],
        key=lambda x: x[0],
    )
    pareto = []
    best_hr = -1
    for lat, hr in points:
        if hr > best_hr:
            pareto.append((lat, hr))
            best_hr = hr
    if len(pareto) > 1:
        px, py = zip(*pareto)
        ax.plot(px, py, "--", color="green", alpha=0.5, linewidth=1.5, label="Pareto frontier")
        ax.legend()

    plt.tight_layout()
    path = output_dir / "quality_vs_latency.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Chart 3: Per-query-type heatmap ───────────────────────────────────────

def plot_per_query_type_heatmap(per_type, output_dir):
    if not HAS_MPL:
        return

    # Build matrix: rows = query types, cols = plans (no-rerank only for clarity)
    base_plans = ["faiss_only", "bm25_only", "faiss_bm25", "full_ensemble"]
    query_types = sorted(per_type.keys())

    matrix = []
    plan_labels = []
    for plan in base_plans:
        label = None
        col = []
        for qtype in query_types:
            if plan in per_type[qtype]:
                col.append(per_type[qtype][plan]["avg_hit_rate"])
                if label is None:
                    label = per_type[qtype][plan]["label"]
            else:
                col.append(0.0)
        matrix.append(col)
        plan_labels.append(label or plan)

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=0.3, vmax=0.85)

    ax.set_xticks(range(len(query_types)))
    ax.set_xticklabels(query_types, fontsize=10)
    ax.set_yticks(range(len(plan_labels)))
    ax.set_yticklabels(plan_labels, fontsize=10)

    # Add text annotations
    for i in range(len(plan_labels)):
        for j in range(len(query_types)):
            val = matrix[i][j]
            text_color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=text_color)

    ax.set_title("Hit Rate @ 10 by Plan and Query Type (no reranking)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Hit Rate")

    plt.tight_layout()
    path = output_dir / "per_query_type_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Text report ───────────────────────────────────────────────────────────

def generate_text_report(summary, per_type, lookup, results, output_dir):
    lines = []
    lines.append("=" * 72)
    lines.append("  TokenSmith Retrieval Plan Profiling Report")
    lines.append("=" * 72)
    lines.append("")

    # Overall plan comparison
    lines.append("1. PLAN COMPARISON (averaged across all benchmark queries)")
    lines.append("-" * 72)
    lines.append(f"  {'Plan':<30s}  {'Avg Lat':>8s}  {'P50':>6s}  {'P95':>6s}  {'HR@10':>7s}  {'NDCG':>7s}")
    lines.append("-" * 72)
    for plan_name, s in summary.items():
        lines.append(
            f"  {s['label']:<30s}  {s['avg_latency_ms']:>6.1f}ms  "
            f"{s['p50_latency_ms']:>4.1f}ms  {s['p95_latency_ms']:>4.1f}ms  "
            f"{s['avg_hit_rate']:>6.4f}  {s['avg_ndcg']:>6.4f}"
        )
    lines.append("")

    # Per-query-type breakdown
    lines.append("2. PER-QUERY-TYPE BREAKDOWN")
    lines.append("-" * 72)
    for qtype, plans in per_type.items():
        lines.append(f"\n  Query Type: {qtype}")
        lines.append(f"  {'Plan':<30s}  {'Avg Lat':>8s}  {'HR@10':>7s}  {'NDCG':>7s}")
        for pname, ps in plans.items():
            lines.append(
                f"  {ps['label']:<30s}  {ps['avg_latency_ms']:>6.1f}ms  "
                f"{ps['avg_hit_rate']:>6.4f}  {ps['avg_ndcg']:>6.4f}"
            )
    lines.append("")

    # Lookup table
    lines.append("3. RECOMMENDED PLAN LOOKUP TABLE")
    lines.append("-" * 72)
    lines.append(f"  {'Query Type':<20s}  {'Recommended Plan':<30s}")
    lines.append(f"  {'─' * 50}")
    from scripts.profile_retrieval_plans import PLANS
    for qtype, plan_name in lookup.items():
        label = PLANS.get(plan_name, {}).get("label", plan_name)
        lines.append(f"  {qtype:<20s}  {label:<30s}")
    lines.append("")

    # Estimated speedup
    lines.append("4. ESTIMATED LATENCY SAVINGS (--fast mode)")
    lines.append("-" * 72)
    # The default config.yaml uses FAISS-only + cross-encoder reranking.
    # Compare that baseline against the optimizer-selected plans.
    baseline_plan = "faiss_only_rerank"
    baseline_avg = summary.get(baseline_plan, {}).get("avg_latency_ms", 0)
    if baseline_avg == 0:
        baseline_plan = "full_ensemble"
        baseline_avg = summary.get(baseline_plan, {}).get("avg_latency_ms", 0)
    baseline_label = summary.get(baseline_plan, {}).get("label", baseline_plan)

    # Count queries per type (each query appears once per plan = 8 times)
    query_type_counts = defaultdict(int)
    seen_queries = set()
    for r in results:
        key = r["query_id"]
        if key not in seen_queries:
            seen_queries.add(key)
            qt = r.get("query_type", "semantic")
            query_type_counts[qt] += 1
    total_queries = sum(query_type_counts.values())
    query_type_fracs = {qt: count / max(1, total_queries) for qt, count in query_type_counts.items()}

    fast_avg = 0.0
    for qtype, plan_name in lookup.items():
        frac = query_type_fracs.get(qtype, 0)
        plan_lat = per_type.get(qtype, {}).get(plan_name, {}).get("avg_latency_ms", baseline_avg)
        fast_avg += frac * plan_lat

    if baseline_avg > 0:
        speedup = (1 - fast_avg / baseline_avg) * 100
        lines.append(f"  Baseline ({baseline_label}):  {baseline_avg:.1f}ms")
        lines.append(f"  --fast mode estimated avg:       {fast_avg:.1f}ms")
        lines.append(f"  Estimated latency reduction:     {speedup:.1f}%")
    lines.append("")
    lines.append("=" * 72)

    report = "\n".join(lines)
    path = output_dir / "profiling_report.txt"
    with open(path, "w") as f:
        f.write(report)
    print(f"  Saved {path}")
    print()
    print(report)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate profiling report and charts")
    parser.add_argument("--input", default="results/profiling",
                        help="Directory with profiling results")
    parser.add_argument("--output", default="results/profiling/charts",
                        help="Output directory for charts")
    args = parser.parse_args()

    import os
    os.chdir(PROJECT_ROOT)

    input_dir = pathlib.Path(args.input)
    output_dir = pathlib.Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading profiling data...")
    results, summary, per_type, lookup = load_data(input_dir)
    print(f"  Loaded {len(results)} result entries across {len(summary)} plans.\n")

    print("Generating charts...")
    plot_latency_comparison(summary, output_dir)
    plot_quality_vs_latency(summary, output_dir)
    plot_per_query_type_heatmap(per_type, output_dir)

    print("\nGenerating text report...")
    generate_text_report(summary, per_type, lookup, results, output_dir)


if __name__ == "__main__":
    main()
