#!/usr/bin/env bash
set -Eeuo pipefail

# Full local evaluation runner for TokenSmith retrieval-quality work.
#
# Usage:
#   bash scripts/run_evals.sh
#
# Optional environment overrides:
#   LOG_ROOT=eval_runs
#   CONFIG_PATH=config/config.yaml
#   BENCHMARKS_PATH=tests/benchmarks.yaml
#   ARTIFACTS_DIR=index/sections
#   INDEX_PREFIX=textbook_index
#   RUN_EXTRACT=0
#   RUN_INDEX=1
#   RUN_TESTS=1
#   RUN_BENCHMARK=1
#   RUN_BASELINE=0
#   MULTIPROC_INDEXING=0
#   EXTRA_INDEX_ARGS="--keep_tables"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

timestamp() {
  date +"%Y-%m-%dT%H:%M:%S%z"
}

LOG_ROOT="${LOG_ROOT:-eval_runs}"
RUN_ID="${RUN_ID:-$(date +"%Y%m%d_%H%M%S")}"
RUN_DIR="${LOG_ROOT}/full_textbook_eval_${RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
RESULTS_DIR="${RUN_DIR}/results"
SNAPSHOT_DIR="${RUN_DIR}/snapshot"

CONFIG_PATH="${CONFIG_PATH:-config/config.yaml}"
BENCHMARKS_PATH="${BENCHMARKS_PATH:-tests/benchmarks.yaml}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-index/sections}"
INDEX_PREFIX="${INDEX_PREFIX:-textbook_index}"

RUN_EXTRACT="${RUN_EXTRACT:-0}"
RUN_INDEX="${RUN_INDEX:-1}"
RUN_TESTS="${RUN_TESTS:-1}"
RUN_BENCHMARK="${RUN_BENCHMARK:-1}"
RUN_BASELINE="${RUN_BASELINE:-0}"
MULTIPROC_INDEXING="${MULTIPROC_INDEXING:-0}"
EXTRA_INDEX_ARGS="${EXTRA_INDEX_ARGS:-}"

mkdir -p "${LOG_DIR}" "${RESULTS_DIR}" "${SNAPSHOT_DIR}"

MASTER_LOG="${LOG_DIR}/run.log"
SUMMARY_JSON="${RESULTS_DIR}/summary.json"
SUMMARY_MD="${RESULTS_DIR}/summary.md"
IMPROVED_JSONL="${RESULTS_DIR}/retrieval_improved.jsonl"
BASELINE_JSONL="${RESULTS_DIR}/retrieval_baseline.jsonl"

exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "=== TokenSmith full textbook evaluation ==="
echo "Run directory: ${RUN_DIR}"
echo "Started at: $(timestamp)"
echo "Repo root: ${REPO_ROOT}"

if [[ -x ".conda-envs/tokensmith/bin/python" ]]; then
  PYTHON_CMD=("${REPO_ROOT}/.conda-envs/tokensmith/bin/python")
elif command -v conda >/dev/null 2>&1; then
  PYTHON_CMD=(conda run --no-capture-output -n tokensmith python)
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD=(python3)
else
  echo "ERROR: Could not find Python. Activate/create the TokenSmith environment first." >&2
  exit 1
fi

run_step() {
  local name="$1"
  shift
  local step_log="${LOG_DIR}/${name}.log"
  echo
  echo "=== [${name}] START $(timestamp) ==="
  echo "Command: $*"
  set +e
  "$@" > >(tee -a "${step_log}") 2> >(tee -a "${step_log}" >&2)
  local status=$?
  set -e
  echo "=== [${name}] END status=${status} $(timestamp) ==="
  if [[ "${status}" -ne 0 ]]; then
    echo "ERROR: Step '${name}' failed. See ${step_log}" >&2
    exit "${status}"
  fi
}

run_python_step() {
  local name="$1"
  shift
  run_step "${name}" "${PYTHON_CMD[@]}" "$@"
}

write_failure_summary() {
  local status=$?
  {
    echo "# TokenSmith Full Textbook Evaluation"
    echo
    echo "- Status: FAILED"
    echo "- Exit code: ${status}"
    echo "- Finished: $(timestamp)"
    echo "- Logs: ${LOG_DIR}"
  } > "${SUMMARY_MD}"
  exit "${status}"
}
trap write_failure_summary ERR

echo
echo "=== Configuration ==="
echo "CONFIG_PATH=${CONFIG_PATH}"
echo "BENCHMARKS_PATH=${BENCHMARKS_PATH}"
echo "ARTIFACTS_DIR=${ARTIFACTS_DIR}"
echo "INDEX_PREFIX=${INDEX_PREFIX}"
echo "RUN_EXTRACT=${RUN_EXTRACT}"
echo "RUN_INDEX=${RUN_INDEX}"
echo "RUN_TESTS=${RUN_TESTS}"
echo "RUN_BENCHMARK=${RUN_BENCHMARK}"
echo "RUN_BASELINE=${RUN_BASELINE}"
echo "MULTIPROC_INDEXING=${MULTIPROC_INDEXING}"
echo "EXTRA_INDEX_ARGS=${EXTRA_INDEX_ARGS}"
echo "PYTHON=${PYTHON_CMD[*]}"

run_step "git_status" git status --short
run_step "git_rev" git rev-parse --show-toplevel
run_step "git_head" git log -1 --oneline

cp "${CONFIG_PATH}" "${SNAPSHOT_DIR}/config.yaml"
cp "${BENCHMARKS_PATH}" "${SNAPSHOT_DIR}/benchmarks.yaml"

run_python_step "preflight" - "${CONFIG_PATH}" <<'PY'
from pathlib import Path
import sys
from src.config import RAGConfig

cfg = RAGConfig.from_yaml(sys.argv[1])
required = {
    "embedding model": Path(cfg.embed_model),
    "generation model": Path(cfg.gen_model),
}
missing = [f"{label}: {path}" for label, path in required.items() if not path.exists()]
if missing:
    raise SystemExit("Missing required model file(s):\n" + "\n".join(missing))

data_dir = Path("data")
if not any(data_dir.glob("*.md")) and not (data_dir / "extracted_sections.json").exists():
    raise SystemExit("No data/*.md or data/extracted_sections.json found. Set RUN_EXTRACT=1 if PDFs are in data/chapters/.")

print(f"embed_model={cfg.embed_model}")
print(f"gen_model={cfg.gen_model}")
print(f"top_k={cfg.top_k}")
print(f"num_candidates={cfg.num_candidates}")
print(f"section_top_k={cfg.section_top_k}")
print(f"adaptive={cfg.enable_adaptive_routing}")
print(f"hierarchical={cfg.enable_hierarchical_retrieval}")
PY

if [[ "${RUN_EXTRACT}" == "1" ]]; then
  run_python_step "extract_textbook" -m src.preprocessing.extraction
fi

if [[ "${RUN_INDEX}" == "1" ]]; then
  INDEX_ARGS=( -m src.main index --config "${CONFIG_PATH}" --index_prefix "${INDEX_PREFIX}" )
  if [[ "${MULTIPROC_INDEXING}" == "1" ]]; then
    INDEX_ARGS+=( --multiproc_indexing )
  fi
  if [[ -n "${EXTRA_INDEX_ARGS}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS_ARRAY=( ${EXTRA_INDEX_ARGS} )
    INDEX_ARGS+=( "${EXTRA_ARGS_ARRAY[@]}" )
  fi
  run_python_step "build_full_index" "${INDEX_ARGS[@]}"
fi

run_python_step "validate_artifacts" - <<PY
from pathlib import Path
from src.retriever import load_artifact_bundle

bundle = load_artifact_bundle(Path("${ARTIFACTS_DIR}"), "${INDEX_PREFIX}")
print(f"chunks={len(bundle.chunks)}")
print(f"sections={len(bundle.sections)}")
print(f"hierarchical_artifacts={bundle.has_hierarchical_artifacts}")
print(f"manifest_version={bundle.manifest.get('artifact_version') if bundle.manifest else None}")
print(f"page_map_pages={len(bundle.page_to_chunk_map)}")
if not bundle.has_hierarchical_artifacts:
    raise SystemExit("Built artifacts do not include hierarchical section artifacts.")
PY

if [[ "${RUN_BENCHMARK}" == "1" ]]; then
  run_step "retrieval_benchmark_improved" env PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_CMD[@]}" scripts/run_retrieval_benchmark.py \
    --repo-root "${REPO_ROOT}" \
    --config "${CONFIG_PATH}" \
    --benchmarks "${BENCHMARKS_PATH}" \
    --artifacts-dir "${ARTIFACTS_DIR}" \
    --index-prefix "${INDEX_PREFIX}" \
    --mode improved \
    --output "${IMPROVED_JSONL}"

  if [[ "${RUN_BASELINE}" == "1" ]]; then
    run_step "retrieval_benchmark_baseline" env PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_CMD[@]}" scripts/run_retrieval_benchmark.py \
      --repo-root "${REPO_ROOT}" \
      --config "${CONFIG_PATH}" \
      --benchmarks "${BENCHMARKS_PATH}" \
      --artifacts-dir "${ARTIFACTS_DIR}" \
      --index-prefix "${INDEX_PREFIX}" \
      --mode baseline \
      --output "${BASELINE_JSONL}"
  fi
fi

if [[ "${RUN_TESTS}" == "1" ]]; then
  # Run the full suite intentionally so this local evaluation remains aligned with CI.
  run_python_step "pytest_all" -m pytest tests/ -q
  run_step "make_lint" make lint
fi

run_python_step "summarize_results" - <<PY
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

run_dir = Path("${RUN_DIR}")
results_dir = Path("${RESULTS_DIR}")
improved_path = Path("${IMPROVED_JSONL}")
summary_json = Path("${SUMMARY_JSON}")
summary_md = Path("${SUMMARY_MD}")

metric_keys = [
    "chunk_ndcg_10_similarity",
    "chunk_recall_5_similarity",
    "chunk_recall_10_similarity",
    "chunk_mrr_10_similarity",
    "chunk_map_10_similarity",
    "page_hit_5_similarity",
    "page_hit_10_similarity",
    "direct_page_hit_10_similarity",
    "final_score",
]

def load_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows

rows = load_jsonl(improved_path)
aggregates = {}
for key in metric_keys:
    values = [float(row.get("scores", {}).get(key, 0.0)) for row in rows]
    if values:
        aggregates[key] = mean(values)

by_query_type = defaultdict(list)
confidence_widened = 0
for row in rows:
    query_type = row.get("retrieval_trace", {}).get("resolved_query_type") or row.get("expected_query_type") or "unknown"
    by_query_type[query_type].append(float(row.get("scores", {}).get("final_score", 0.0)))
    if row.get("retrieval_trace", {}).get("confidence_widening_used"):
        confidence_widened += 1

query_type_scores = {
    query_type: mean(values)
    for query_type, values in sorted(by_query_type.items())
    if values
}

summary = {
    "status": "success",
    "run_dir": str(run_dir),
    "num_benchmark_rows": len(rows),
    "aggregates": aggregates,
    "query_type_final_score": query_type_scores,
    "confidence_widening_used_count": confidence_widened,
    "improved_results": str(improved_path) if improved_path.exists() else None,
}

summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

lines = [
    "# TokenSmith Full Textbook Evaluation",
    "",
    "- Status: SUCCESS",
    f"- Run directory: {run_dir}",
    f"- Benchmark rows: {len(rows)}",
    f"- Confidence widening used: {confidence_widened}",
    "",
    "## Aggregate Retrieval Metrics",
]
for key, value in aggregates.items():
    lines.append(f"- {key}: {value:.4f}")
lines.extend(["", "## Final Score By Query Type"])
for key, value in query_type_scores.items():
    lines.append(f"- {key}: {value:.4f}")

summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(summary_md.read_text(encoding="utf-8"))
PY

echo
echo "=== Completed successfully at $(timestamp) ==="
echo "Run directory: ${RUN_DIR}"
echo "Summary: ${SUMMARY_MD}"
echo "Send this folder back for review: ${RUN_DIR}"
