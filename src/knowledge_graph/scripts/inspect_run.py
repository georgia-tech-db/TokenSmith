import argparse

import networkx as nx

from src.knowledge_graph.io import RUNS_DIR, load_graph_and_chunks, resolve_run_dir
from src.knowledge_graph.section_tree import SectionTree, load_section_tree

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEP = "─" * 60


def _percentiles(values: list[float], qs=(10, 25, 50, 75, 90, 95, 99)) -> dict[int, float]:
    if not values:
        return {q: 0.0 for q in qs}
    s = sorted(values)
    n = len(s)
    result = {}
    for q in qs:
        idx = (q / 100) * (n - 1)
        lo, frac = int(idx), idx % 1
        result[q] = s[lo] + frac * (s[min(lo + 1, n - 1)] - s[lo])
    return result


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _bar(value: float, max_value: float, width: int = 30) -> str:
    filled = int(round(value / max_value * width)) if max_value else 0
    return "█" * filled + "░" * (width - filled)


# ---------------------------------------------------------------------------
# Graph section
# ---------------------------------------------------------------------------

def _print_graph_stats(graph: nx.Graph, chunks_json: dict[int, str]) -> None:
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    density = nx.density(graph)

    degrees = [d for _, d in graph.degree()]
    isolated = sum(1 for d in degrees if d == 0)

    comp_list = list(nx.connected_components(graph))
    n_components = len(comp_list)
    largest_comp = max(len(c) for c in comp_list) if comp_list else 0

    print(f"\n{'GRAPH':^60}")
    print(SEP)
    print(f"  Nodes              {n_nodes:>8,}")
    print(f"  Edges              {n_edges:>8,}")
    print(f"  Density            {density:>12.6f}")
    print(f"  Connected comps    {n_components:>8,}  (largest: {largest_comp:,} nodes)")
    print(f"  Isolated nodes     {isolated:>8,}")

    # Degree distribution with percentiles
    print(f"\n  Degree distribution")
    print(f"  {'Stat':<10} {'Value':>8}")
    print(f"  {'─'*20}")
    print(f"  {'mean':<10} {_mean(degrees):>8.2f}")
    print(f"  {'max':<10} {max(degrees, default=0):>8}")
    pcts = _percentiles(degrees)
    for q, v in pcts.items():
        print(f"  {'p' + str(q):<10} {v:>8.2f}")

    # Top-10 hubs
    top10 = sorted(graph.degree(), key=lambda x: x[1], reverse=True)[:10]
    max_deg = top10[0][1] if top10 else 1
    print(f"\n  Top-10 hub keywords (by degree)")
    print(f"  {'Keyword':<35} {'Deg':>4}  Distribution")
    print(f"  {'─'*60}")
    for kw, deg in top10:
        chunks_count = len(graph.nodes[kw].get("chunk_ids", []))
        bar = _bar(deg, max_deg)
        print(f"  {kw:<35} {deg:>4}  {bar}  ({chunks_count} chunks)")

    # Coverage: chunks with ≥1 keyword vs zero
    chunk_ids_in_graph: set[int] = set()
    for _, data in graph.nodes(data=True):
        chunk_ids_in_graph.update(data.get("chunk_ids", []))
    total_chunks = len(chunks_json)
    covered = len(chunk_ids_in_graph & set(chunks_json.keys()))
    uncovered = total_chunks - covered
    pct = covered / total_chunks * 100 if total_chunks else 0
    print(f"\n  Chunk coverage")
    print(f"  {'Total chunks':<30} {total_chunks:>6,}")
    print(f"  {'Covered by ≥1 keyword':<30} {covered:>6,}  ({pct:.1f}%)")
    print(f"  {'No keywords (invisible)':<30} {uncovered:>6,}")

    # Keywords-per-chunk distribution
    kw_per_chunk: dict[int, int] = {}
    for _, data in graph.nodes(data=True):
        for cid in data.get("chunk_ids", []):
            kw_per_chunk[cid] = kw_per_chunk.get(cid, 0) + 1
    kpc_values = list(kw_per_chunk.values())
    if kpc_values:
        pcts_kpc = _percentiles(kpc_values)
        print(f"\n  Keywords per chunk (covered chunks only)")
        print(f"  {'Stat':<10} {'Value':>8}")
        print(f"  {'─'*20}")
        print(f"  {'mean':<10} {_mean(kpc_values):>8.2f}")
        print(f"  {'max':<10} {max(kpc_values):>8}")
        for q, v in pcts_kpc.items():
            print(f"  {'p' + str(q):<10} {v:>8.2f}")


# ---------------------------------------------------------------------------
# Section tree section
# ---------------------------------------------------------------------------

def _print_tree_stats(tree: SectionTree) -> None:
    level_labels = {1: "chapters", 2: "sections", 3: "subsections"}

    all_nodes = list(tree.node_index.values())
    level_2_nodes = [n for n in all_nodes if n.level == 2]

    print(f"\n{'SECTION TREE':^60}")
    print(SEP)

    # Count per level
    level_counts: dict[int, int] = {}
    for node in all_nodes:
        level_counts[node.level] = level_counts.get(node.level, 0) + 1
    for lvl, count in sorted(level_counts.items()):
        label = level_labels.get(lvl, f"level-{lvl} nodes")
        print(f"  {count:>4} {label}")

    # Keyword set sizes per level
    print(f"\n  Keyword set size by level")
    print(f"  {'Level':<14} {'mean':>6}  {'p50':>6}  {'p90':>6}  {'max':>6}")
    print(f"  {'─'*45}")
    for lvl in sorted(level_counts.keys()):
        nodes_at_lvl = [n for n in all_nodes if n.level == lvl]
        sizes = [len(n.keyword_set) for n in nodes_at_lvl]
        pcts = _percentiles(sizes)
        label = level_labels.get(lvl, f"level-{lvl}")
        print(
            f"  {label:<14} {_mean(sizes):>6.1f}  {pcts[50]:>6.1f}  {pcts[90]:>6.1f}  {max(sizes, default=0):>6}"
        )

    # Top-5 and bottom-5 sections by keyword set size (level 2)
    if level_2_nodes:
        by_kw = sorted(level_2_nodes, key=lambda n: len(n.keyword_set), reverse=True)
        max_kw = len(by_kw[0].keyword_set) if by_kw else 1

        print(f"\n  Top-5 sections by keyword richness")
        print(f"  {'Section':<40} {'KWs':>5}  Distribution")
        print(f"  {'─'*60}")
        for node in by_kw[:5]:
            bar = _bar(len(node.keyword_set), max_kw)
            print(f"  {node.heading:<40} {len(node.keyword_set):>5}  {bar}")

        print(f"\n  Bottom-5 sections (potential retrieval blind spots)")
        print(f"  {'Section':<40} {'KWs':>5}  Distribution")
        print(f"  {'─'*60}")
        for node in by_kw[-5:]:
            bar = _bar(len(node.keyword_set), max_kw)
            print(f"  {node.heading:<40} {len(node.keyword_set):>5}  {bar}")

    # Sibling overlap: top-5 most similar section pairs
    if len(level_2_nodes) >= 2:
        overlaps = []
        for i, a in enumerate(level_2_nodes):
            for b in level_2_nodes[i + 1:]:
                if a.chapter != b.chapter:
                    continue  # only compare within same chapter
                shared = len(a.keyword_set & b.keyword_set)
                union = len(a.keyword_set | b.keyword_set)
                if union > 0:
                    overlaps.append((shared / union, shared, a, b))
        overlaps.sort(key=lambda x: (x[0], x[1]), reverse=True)
        if overlaps:
            print(f"\n  Top-5 most similar sibling sections (within chapter, by Jaccard)")
            print(f"  {'Jaccard':>7}  {'Shared':>6}  Pair")
            print(f"  {'─'*60}")
            for jaccard, shared, a, b in overlaps[:5]:
                print(f"  {jaccard:>7.3f}  {shared:>6}  {a.heading}  ↔  {b.heading}")


# ---------------------------------------------------------------------------
# Cross-signal coverage
# ---------------------------------------------------------------------------

def _print_cross_signal(graph: nx.Graph, tree: SectionTree, chunks_json: dict[int, str]) -> None:
    all_chunk_ids = set(chunks_json.keys())

    graph_covered: set[int] = set()
    for _, data in graph.nodes(data=True):
        graph_covered.update(data.get("chunk_ids", []))
    graph_covered &= all_chunk_ids

    tree_covered: set[int] = set(tree.chunk_to_sections.keys()) & all_chunk_ids

    both = graph_covered & tree_covered
    graph_only = graph_covered - tree_covered
    tree_only = tree_covered - graph_covered
    neither = all_chunk_ids - graph_covered - tree_covered

    total = len(all_chunk_ids)

    def pct(n):
        return n / total * 100 if total else 0

    print(f"\n{'CROSS-SIGNAL COVERAGE':^60}")
    print(SEP)
    print(f"  {'Chunk set':<35} {'Count':>6}  {'%':>6}")
    print(f"  {'─'*50}")
    print(f"  {'Graph + section tree':<35} {len(both):>6,}  {pct(len(both)):>5.1f}%")
    print(f"  {'Graph only':<35} {len(graph_only):>6,}  {pct(len(graph_only)):>5.1f}%")
    print(f"  {'Section tree only':<35} {len(tree_only):>6,}  {pct(len(tree_only)):>5.1f}%")
    print(f"  {'Neither (invisible)':<35} {len(neither):>6,}  {pct(len(neither)):>5.1f}%")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a KG run's graph and section tree.")
    parser.add_argument(
        "--run",
        default=None,
        help="Path to a specific run directory. Defaults to the latest run.",
    )
    args = parser.parse_args()

    run_path = args.run or RUNS_DIR
    run_dir = resolve_run_dir(run_path)
    print(f"Run: {run_dir}")

    graph, chunks_json = load_graph_and_chunks(run_dir)
    tree = load_section_tree(run_dir)

    _print_graph_stats(graph, chunks_json)
    _print_tree_stats(tree)
    _print_cross_signal(graph, tree, chunks_json)
    print()


if __name__ == "__main__":
    main()
