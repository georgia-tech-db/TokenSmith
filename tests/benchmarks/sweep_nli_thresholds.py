"""Sweeps the NLI confidence thresholds over the BuzzDB NLI cases.

The model runs once (buzzdb_nli_verify.py); every threshold setting is then
replayed in Python, mirroring claimVerdict and classifyAnswerStatus in
src/shared/confidence.ts. Settings are ranked on the "dev" questions only and
reported on the "heldout" ones, so a winner that only fits dev shows up.

usage: python tests/benchmarks/sweep_nli_thresholds.py [verify_output.json]
"""

import itertools
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1].parent
GRID = {
    "supported": [0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    "entailed": [0.4, 0.5, 0.6, 0.7, 0.8],
    "contradicted": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "minEntailedShare": [0.5, 0.67, 0.75, 1.0],
    "contradictionWins": [False, True],
}
CURRENT = {"supported": 0.85, "entailed": 0.8, "contradicted": 0.5, "minEntailedShare": 0.5, "contradictionWins": True}
PREVIOUS = {"supported": 0.8, "entailed": 0.6, "contradicted": 0.5, "minEntailedShare": 0.67, "contradictionWins": False}


def verdict(entailment, contradiction, t):
    if t["contradictionWins"] and contradiction >= t["contradicted"] and contradiction >= entailment:
        return "contradicted"
    if entailment >= t["supported"]:
        return "supported"
    if entailment >= t["entailed"]:
        return "likely"
    return "contradicted" if contradiction >= t["contradicted"] else "unsupported"


def status(claims, t):
    if not claims:
        return "unverified"
    verdicts = [verdict(e, c, t) for e, c in claims]
    entailed = sum(v in ("supported", "likely") for v in verdicts)
    if "contradicted" in verdicts or entailed / len(verdicts) < t["minEntailedShare"]:
        return "unsupported"
    return "supported" if all(v == "supported" for v in verdicts) else "partial"


def errors(rows, t):
    counts = {"correct": 0, "total": 0, "missed": 0, "wrongLow": 0, "falseSupported": 0}
    for expect, claims in rows:
        s = status(claims, t)
        missed = expect == "abstain" and s != "unsupported"
        wrong = expect == "answer" and s in ("unsupported", "unverified")
        false_supported = expect != "answer" and s == "supported"
        counts["missed"] += missed
        counts["wrongLow"] += wrong
        counts["falseSupported"] += false_supported
        counts["correct"] += not (missed or wrong or false_supported)
        counts["total"] += 1
    return counts


def cost(counts):
    # A false Supported is the failure that misleads a student most.
    return counts["missed"] + counts["wrongLow"] + 3 * counts["falseSupported"]


def describe(counts):
    return (f"{counts['correct']}/{counts['total']} missed={counts['missed']} "
            f"wrongLow={counts['wrongLow']} falseSupported={counts['falseSupported']}")


def main():
    if len(sys.argv) > 1:
        results = json.loads(Path(sys.argv[1]).read_text())["results"]
    else:
        run = subprocess.run([sys.executable, str(ROOT / "tests/benchmarks/buzzdb_nli_verify.py")],
                             check=True, capture_output=True, text=True, cwd=ROOT)
        results = json.loads(run.stdout)["results"]

    splits = {"dev": [], "heldout": []}
    for result in results:
        verification = result["verification"]
        claims = [
            (max((p["entailment"] for p in c["passages"]), default=0.0),
             max((p["contradiction"] for p in c["passages"]), default=0.0))
            for c in (verification["claims"] if verification.get("available", True) else [])
        ]
        splits[result.get("split", "dev")].append((result["expect"], claims))

    ranked = []
    for values in itertools.product(*GRID.values()):
        t = dict(zip(GRID, values))
        if t["entailed"] <= t["supported"]:
            dev = errors(splits["dev"], t)
            ranked.append((cost(dev), dev["wrongLow"], t, dev))
    ranked.sort(key=lambda row: row[:2])

    for name, t in (("previous", PREVIOUS), ("current", CURRENT)):
        print(f"{name:9s} dev {describe(errors(splits['dev'], t))} | heldout {describe(errors(splits['heldout'], t))}")
    print("best on dev:")
    for _, _, t, dev in ranked[:10]:
        print(f"  {t} dev {describe(dev)} | heldout {describe(errors(splits['heldout'], t))}")


if __name__ == "__main__":
    main()
