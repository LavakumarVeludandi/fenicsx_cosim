"""Benchmark: NearestNeighborMapper accuracy vs an analytic interface field.

Validates the core coupling primitive (``DataMapper``). A smooth field sampled
on a source point cloud is mapped to a *non-conforming* target cloud, and the
mapped values are compared to the closed-form field at the target points.

Reports the relative L2 error under source refinement and the observed
convergence rate. Nearest-neighbor mapping is first order, so the error should
fall ~linearly with source spacing and the rate should approach 1.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from _harness import BenchmarkResult, report  # noqa: E402
from fenicsx_cosim.data_mapper import NearestNeighborMapper  # noqa: E402


def _field(x: np.ndarray) -> np.ndarray:
    """Smooth analytic interface field."""
    return np.sin(2.0 * np.pi * x)


def _line_cloud(x: np.ndarray) -> np.ndarray:
    """(N, 3) coordinates along the y = z = 0 interface line."""
    c = np.zeros((len(x), 3))
    c[:, 0] = x
    return c


def run(resolutions=(20, 40, 80, 160)):
    # Fixed non-conforming target cloud: different count, interior, never
    # coincident with the uniform source nodes.
    xt = np.linspace(0.02, 0.98, 137)
    target = _line_cloud(xt)
    exact = _field(xt)

    errs = []
    for n in resolutions:
        xs = np.linspace(0.0, 1.0, n)
        mapper = NearestNeighborMapper()
        mapper.build(_line_cloud(xs), target)
        mapped = mapper.map(_field(xs))
        errs.append(float(np.linalg.norm(mapped - exact)
                          / np.linalg.norm(exact)))

    rate = math.log(errs[-2] / errs[-1]) / math.log(
        resolutions[-1] / resolutions[-2]
    )

    results = [
        BenchmarkResult(f"NN map  n={n:<4d}", "rel L2 error", e, 0.6)
        for n, e in zip(resolutions, errs)
    ]
    results.append(
        BenchmarkResult("NN finest resolution", "rel L2 error",
                        errs[-1], 0.05)
    )
    results.append(
        BenchmarkResult("NN convergence rate", "|order - 1|",
                        abs(rate - 1.0), 0.5, extra={"rate": rate})
    )
    return results, errs, resolutions


if __name__ == "__main__":
    res, _, _ = run()
    ok = report(res, "Mapping Accuracy")
    sys.exit(0 if ok else 1)
