# Validated Benchmarks

Each benchmark compares `fenicsx-cosim` output to a **closed-form analytic
answer** and reports the error as a reproducible number. They are the
correctness evidence behind the library — not demos. The pass/fail tolerances
are enforced in CI via `tests/test_benchmarks.py`.

| Benchmark | Validates | Analytic reference | Tolerance |
|---|---|---|---|
| `bench_mapping_accuracy.py` | `DataMapper` (nearest-neighbor) — the core field-transfer primitive | smooth field `sin(2πx)` mapped to a non-conforming target cloud | rel L2 error < 0.05 at finest; first-order convergence (rate ≈ 1) |
| `bench_fe2_homogenization.py` | FE² homogenization (KUBC RVE + volume averaging) | homogeneous elastic RVE → `σ̄ = C : ε̄` (plane-strain Hooke); Hill-Mandel | rel error < 1e-7; Hill-Mandel residual < 1e-8 |

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
