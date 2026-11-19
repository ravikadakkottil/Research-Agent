"""
Compare two eval results: did a change really help?

    python evals/compare.py evals/results/OLD.json evals/results/NEW.json

A difference is only meaningful if it's bigger than the natural run-to-run
variance ("spread"). Smaller deltas are marked "~noise".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from research_agent.evals.runner import METRICS


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fmt(v: float | None) -> str:
    return "  —  " if v is None else f"{v:5.2f}"


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    old, new = load(sys.argv[1]), load(sys.argv[2])
    print(f"OLD: {old['meta']['label']} ({old['meta']['timestamp']})")
    print(f"NEW: {new['meta']['label']} ({new['meta']['timestamp']})\n")

    print(f"{'metric':<22}{'old':>8}{'new':>8}{'delta':>9}   verdict")
    for m in METRICS:
        # .get: older reports may not have newer metrics (e.g. shortlist_recall)
        o_m, n_m = old["overall"].get(m, {}), new["overall"].get(m, {})
        a, b = o_m.get("mean"), n_m.get("mean")
        if a is None or b is None:
            print(f"{m:<22}{fmt(a):>8}{fmt(b):>8}{'':>9}")
            continue
        delta = b - a
        noise = max(o_m.get("spread") or 0, n_m.get("spread") or 0)
        verdict = "~noise" if abs(delta) <= noise else ("better" if delta > 0 else "WORSE")
        print(f"{m:<22}{fmt(a):>8}{fmt(b):>8}{delta:>+9.2f}   {verdict}")

    print(f"\n{'topic':<26}{'recall old→new':>18}{'precision old→new':>22}")
    for tid in new["per_topic"]:
        o, n = old["per_topic"].get(tid, {}), new["per_topic"][tid]
        rec = f"{fmt(o.get('recall'))} → {fmt(n['recall'])}"
        prec = f"{fmt(o.get('precision'))} → {fmt(n['precision'])}"
        print(f"{tid:<26}{rec:>18}{prec:>22}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
