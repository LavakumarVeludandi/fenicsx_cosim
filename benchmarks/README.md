# Validated Benchmarks

Each benchmark compares `fenicsx-cosim` output to a **closed-form analytic
answer** and reports the error as a reproducible number. They are the
correctness evidence behind the library — not demos. The pass/fail tolerances
are enforced in CI via `tests/test_benchmarks.py`.

| Benchmark | Validates | Analytic reference | Tolerance |
|---|---|---|---|
| `bench_mapping_accuracy.py` | `DataMapper` — nearest-neighbor **and** inverse-distance (consistent) field transfer | smooth field `sin(2πx)` mapped to a non-conforming target cloud | NN: rel L2 < 0.05 finest, first-order (rate ≈ 1). IDW: beats NN; constant-field error < 1e-12 (consistency) |
| `bench_implicit_coupling.py` | strong-coupling accelerators (`Aitken`, `IQNILS`) | affine fixed point `S(x)=Ax+b`, `x*=(I-A)^{-1}b`, ρ(A)>1 (added-mass: Gauss-Seidel diverges) | accelerated `‖x-x*‖` < 1e-7; IQN-ILS converges in ≤ n+2 sub-iterations |
| `bench_fe2_homogenization.py` | RVE homogenization correctness (KUBC solve + volume averaging) — *not* the dispatch fabric | homogeneous elastic RVE → `σ̄ = C : ε̄` (plane-strain Hooke); Hill-Mandel | rel error < 1e-7; Hill-Mandel residual < 1e-8 |

> The FE² **dispatch fabric** (scatter→worker→gather over ZeroMQ) is validated
> separately by `tests/test_fe2_e2e.py` (real subprocess worker, path-dependent
> round-trip), not by a benchmark.

The FE² case is exact by construction: a homogeneous linear-elastic RVE under
KUBC has the affine solution `u = ε̄·x`, exactly representable by the finite
element space, so the homogenized stress equals closed-form Hooke to solver
precision — there is no discretization error to hide a bug behind.

## Run

```bash
# Mapping benchmark — pure NumPy/SciPy, runs anywhere
python benchmarks/bench_mapping_accuracy.py

# FE² benchmark — requires DOLFINx
conda run -n fenicsx-env python benchmarks/bench_fe2_homogenization.py
```

Each script prints a result table and writes a CSV to `benchmarks/results/`
(gitignored). As the CI gate:

```bash
pytest tests/test_benchmarks.py -v                 # mapping only (no DOLFINx)
conda run -n fenicsx-env pytest tests/test_benchmarks.py -v   # + FE²
```
