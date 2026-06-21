"""Partitioned thermo-mechanical coupling — subdomain B (heat + structure).

B owns [0.8, 2.0] with a cold right edge (T=0). It exchanges interface
temperatures with A (overlapping Schwarz), then runs a genuine thermoelastic
solve on its converged temperature field: the bar is clamped at the cold end
(x=2) and expands toward the hot overlap, so the interface end moves in -x.

See ``thermal_a.py`` for the honesty note on partitioning two FEniCSx domains.

Run after ``python -m thermomech.thermal_a``::

    python -m thermomech.thermal_b
"""

import numpy as np

from fenicsx_cosim import CouplingInterface

from thermomech.subdomains import HeatSubdomain, ThermoElasticSubdomain

N_SCHWARZ = 25


def main() -> None:
    B = HeatSubdomain(x0=0.8, x1=2.0, n=16)
    cosim = CouplingInterface(name="DomainB", partner_name="DomainA",
                              role="connect")

    for it in range(N_SCHWARZ):
        b_left = float(cosim.import_raw("b_left")[0])
        TB = B.solve(left_val=b_left, right_val=0.0)
        cosim.export_raw("a_right", np.array([B.eval_at_x(1.2)]))

    print(f"[B] thermal done, interface T(1.0)={B.eval_at_x(1.0):.3f}",
          flush=True)

    # Downstream thermoelastic solve on the converged temperature field.
    mech = ThermoElasticSubdomain(x0=0.8, x1=2.0, n=16,
                                  E=70e3, nu=0.3, alpha=1e-4, T_ref=0.0)
    mech.solve(temperature_fn=TB, clamp_x=2.0)
    print(f"[B] thermoelastic free-end u_x(0.8)={mech.eval_ux_at_x(0.8):.3e} "
          f"(expansion => negative)", flush=True)
    cosim.disconnect()


if __name__ == "__main__":
    main()
