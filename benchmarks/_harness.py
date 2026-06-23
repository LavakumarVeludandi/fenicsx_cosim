"""Lightweight benchmark harness: run cases, print a table, save CSV.

A *benchmark* here is a validated artifact: it compares library output to a
closed-form / analytic answer and reports the error as a number anyone can
reproduce and cite. This is deliberately stronger than a buried pass/fail test
assert — adoption of a numerical coupling library hinges on visible, repeatable
correctness evidence.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"


@dataclass
class BenchmarkResult:
    """One measured number compared against a tolerance."""

    name: str
    metric: str           # what the number means, e.g. "rel L2 error"
    value: float
    tolerance: float
    extra: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.value <= self.tolerance


def report(results, title: str, save: bool = True) -> bool:
    """Print a result table, optionally persist a CSV, return overall pass."""
    print(f"\n=== {title} ===")
    print(f"{'benchmark':<42}{'metric':<22}{'value':>12}{'tol':>12}  status")
    print("-" * 100)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"{r.name:<42}{r.metric:<22}{r.value:>12.3e}"
              f"{r.tolerance:>12.3e}  {status}")
    ok = all(r.passed for r in results)
    if save:
        RESULTS_DIR.mkdir(exist_ok=True)
        out = RESULTS_DIR / (title.lower().replace(" ", "_") + ".csv")
        with out.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name", "metric", "value", "tolerance", "passed"])
            for r in results:
                w.writerow([r.name, r.metric, r.value, r.tolerance, r.passed])
        print(f"\nsaved: {out}")
    print(f"overall: {'PASS' if ok else 'FAIL'}")
    return ok
