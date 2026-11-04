"""Sweeps the NLI confidence thresholds over the BuzzDB NLI cases.

The model runs once (buzzdb_nli_verify.py); every threshold setting is then
replayed in Python, mirroring claimVerdict and classifyAnswerStatus in
src/shared/confidence.ts. Settings are ranked on the "dev" questions only and
reported on the "heldout" ones, so a winner that only fits dev shows up.

usage: python tests/benchmarks/sweep_nli_thresholds.py [verify_output.json]
"""

import itertools
import re
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1].parent
GRID = {
    "supported": [0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    "entailed": [0.4, 0.5, 0.6, 0.7, 0.8],
    "contradicted": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "highShare": [0.5, 0.66, 0.75, 0.9, 1.0],
    "mediumShare": [0.25, 0.34, 0.5],
}
CURRENT = {"supported": 0.8, "entailed": 0.7, "contradicted": 0.5, "highShare": 0.66, "mediumShare": 0.34}
PREVIOUS = {"supported": 0.85, "entailed": 0.8, "contradicted": 0.5, "highShare": 1.0, "mediumShare": 0.5}


NEGATION = re.compile(r"\b(?:not|no|never|none|neither|nor|cannot|without|unable)\b|n't\b", re.IGNORECASE)


def polarity_differs(claim, evidence):
    return bool(NEGATION.search(claim)) != bool(NEGATION.search(evidence or ""))


def verdict(entailment, contradiction, t, flipped=False):
    if contradiction >= t["contradicted"] and contradiction >= entailment:
        return "contradicted"
    if entailment >= t["supported"]:
        return "supported"
    if entailment >= t["entailed"] and not flipped:
        return "likely"
    return "unsupported"


def status(claims, t):
    if not claims:
        return "unverified"
    verdicts = [verdict(e, c, t, flipped) for e, c, flipped in claims]
    if "contradicted" in verdicts:
        return "unsupported"
    share = sum(1.0 if v == "supported" else 0.5 if v == "likely" else 0.0 for v in verdicts) / len(verdicts)
    if share >= t["highShare"]:
        return "supported"
    return "partial" if share >= t["mediumShare"] else "unsupported"


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
             max((p["contradiction"] for p in c["passages"]), default=0.0),
             polarity_differs(c["text"], c.get("evidence")))
            for c in (verification["claims"] if verification.get("available", True) else [])
        ]
        splits[result.get("split", "dev")].append((result["expect"], claims))

    ranked = []
    for values in itertools.product(*GRID.values()):
        t = dict(zip(GRID, values))
        if t["entailed"] <= t["supported"] and t["mediumShare"] < t["highShare"]:
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
