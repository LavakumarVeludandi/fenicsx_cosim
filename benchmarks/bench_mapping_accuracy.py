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
from fenicsx_cosim.data_mapper import (  # noqa: E402
    InverseDistanceMapper, NearestNeighborMapper,
)


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

    nn_errs, idw_errs = [], []
    for n in resolutions:
        xs = np.linspace(0.0, 1.0, n)
        src = _line_cloud(xs)
        fs = _field(xs)

        nn = NearestNeighborMapper()
        nn.build(src, target)
        nn_errs.append(float(np.linalg.norm(nn.map(fs) - exact)
                             / np.linalg.norm(exact)))

        idw = InverseDistanceMapper(k=4, power=2.0)
        idw.build(src, target)
        idw_errs.append(float(np.linalg.norm(idw.map(fs) - exact)
                              / np.linalg.norm(exact)))

    rate = math.log(nn_errs[-2] / nn_errs[-1]) / math.log(
        resolutions[-1] / resolutions[-2]
    )

    # Consistency: inverse-distance must reproduce a constant field exactly.
    idw_c = InverseDistanceMapper(k=4)
    idw_c.build(_line_cloud(np.linspace(0.0, 1.0, 50)), target)
    const_err = float(np.max(np.abs(idw_c.map(np.ones(50)) - 1.0)))

    results = [
        BenchmarkResult(f"NN map  n={n:<4d}", "rel L2 error", e, 0.6)
        for n, e in zip(resolutions, nn_errs)
    ]
    results += [
        BenchmarkResult("NN finest resolution", "rel L2 error",
                        nn_errs[-1], 0.05),
        BenchmarkResult("NN convergence rate", "|order - 1|",
                        abs(rate - 1.0), 0.5, extra={"rate": rate}),
        BenchmarkResult("IDW finest resolution", "rel L2 error",
                        idw_errs[-1], 0.05),
        BenchmarkResult("IDW beats NN (finest)", "idw/nn ratio",
                        idw_errs[-1] / nn_errs[-1], 1.0),
        BenchmarkResult("IDW consistency (constant field)", "max abs error",
                        const_err, 1e-12),
    ]
    return results, {"nn": nn_errs, "idw": idw_errs}, resolutions


if __name__ == "__main__":
    res, _, _ = run()
    ok = report(res, "Mapping Accuracy")
    sys.exit(0 if ok else 1)
