"""Partitioned thermo-mechanical coupling — subdomain A (heat, left strip).

Real overlapping-Schwarz heat conduction over fenicsx-cosim. A owns [0, 1.2]
with a hot left edge (T=100) and exchanges interface temperatures with B.

HONEST NOTE: coupling two *FEniCSx* subdomains over a socket is only sensible
as a connector demo. For a single shared mesh, solve monolithically — the
partitioning here exists to exercise the coupling machinery, not because it is
the right way to solve one heat equation. The real justification for
fenicsx-cosim is coupling FEniCSx to a *different* code (Abaqus/Kratos/OpenFOAM).

Run::

    python -m thermomech.thermal_a       # terminal 1 (binds first)
    python -m thermomech.thermal_b       # terminal 2
"""

import numpy as np

from fenicsx_cosim import CouplingInterface

from thermomech.subdomains import HeatSubdomain

N_SCHWARZ = 25


def main() -> None:
    A = HeatSubdomain(x0=0.0, x1=1.2, n=16)
    cosim = CouplingInterface(name="DomainA", partner_name="DomainB",
                              role="bind")

    a_right = 0.0
    for it in range(N_SCHWARZ):
        A.solve(left_val=100.0, right_val=a_right)
        cosim.export_raw("b_left", np.array([A.eval_at_x(0.8)]))
        a_right = float(cosim.import_raw("a_right")[0])
        print(f"[A] iter {it:02d}  interface T(1.0)={A.eval_at_x(1.0):.3f}",
              flush=True)

    print(f"[A] converged interface T(1.0)={A.eval_at_x(1.0):.3f} "
          f"(analytic 50.0)", flush=True)
    cosim.disconnect()


if __name__ == "__main__":
    main()
