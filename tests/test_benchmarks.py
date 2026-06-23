"""CI gate for the validated benchmarks.

Turns the benchmark accuracy numbers into a pass/fail gate so a regression in
the coupling primitives (mapping, homogenization) is caught automatically. The
mapping benchmark is pure NumPy/SciPy and runs in plain CI; the FE2 benchmark
needs DOLFINx and is marked slow/integration.
"""

import sys
from pathlib import Path

import pytest

_BENCH = Path(__file__).resolve().parents[1] / "benchmarks"
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))


def test_mapping_accuracy_benchmark():
    import bench_mapping_accuracy as bench

    results, _, _ = bench.run()
    for r in results:
        assert r.passed, f"{r.name}: {r.value:.3e} > tol {r.tolerance:.3e}"


def test_implicit_coupling_benchmark():
    import bench_implicit_coupling as bench

    for r in bench.run():
        assert r.passed, f"{r.name}: {r.value:.3e} > tol {r.tolerance:.3e}"


@pytest.mark.slow
@pytest.mark.integration
def test_fe2_homogenization_benchmark():
    pytest.importorskip("dolfinx")
    import bench_fe2_homogenization as bench

    for r in bench.run():
        assert r.passed, f"{r.name}: {r.value:.3e} > tol {r.tolerance:.3e}"
