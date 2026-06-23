"""Tests for the real partitioned thermo-mechanical example.

Replaces the placeholder thermal/mechanical scripts (which faked the PDE solve
with ``interpolate(lambda ...)``). Two genuine physics solves are coupled:

1. Overlapping-Schwarz partitioned steady heat conduction across two
   subdomains exchanging only interface temperature converges to the *exact*
   linear profile — interface temperature is analytically 50 for T(0)=100,
   T(2)=0, equal conductivity.
2. A downstream thermoelastic solve on the heated subdomain produces a
   physically correct thermal expansion (free end displaces outward).

These are transport-agnostic (the Schwarz exchange is a plain Python loop in
the test); the ZeroMQ scripts wrap the same solves. The example carries an
honest header: partitioning two FEniCSx subdomains is only justified when they
live in different codes — otherwise solve monolithically.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

pytest.importorskip("dolfinx")

from thermomech.subdomains import HeatSubdomain, ThermoElasticSubdomain  # noqa: E402


@pytest.mark.slow
@pytest.mark.integration
def test_schwarz_heat_converges_to_analytic_interface_temperature():
    # Overlapping subdomains: A=[0,1.2], B=[0.8,2.0], overlap [0.8,1.2].
    A = HeatSubdomain(x0=0.0, x1=1.2, n=12)
    B = HeatSubdomain(x0=0.8, x1=2.0, n=12)

    a_right = 0.0  # B's temperature at x=1.2, fed to A
    for _ in range(50):
        TA = A.solve(left_val=100.0, right_val=a_right)
        b_left = A.eval_at_x(0.8)        # A's temperature at B's left edge
        TB = B.solve(left_val=b_left, right_val=0.0)
        a_right_new = B.eval_at_x(1.2)   # B's temperature at A's right edge
        if abs(a_right_new - a_right) < 1e-8:
            a_right = a_right_new
            break
        a_right = a_right_new

    # Analytic steady solution T(x) = 100 (2 - x) / 2  =>  T(1) = 50.
    assert abs(A.eval_at_x(1.0) - 50.0) < 0.5
    assert abs(B.eval_at_x(1.0) - 50.0) < 0.5
    # Linearity check at another overlap point.
    assert abs(A.eval_at_x(0.5) - 75.0) < 0.5


@pytest.mark.slow
@pytest.mark.integration
def test_thermoelastic_expands_outward():
    """Heated bar clamped at x=2 expands; its free end at x=0.8 moves in -x."""
    T = lambda x: 50.0 + 50.0 * (1.2 - x[0]) / 0.4  # hot at left of [0.8,1.2]
    mech = ThermoElasticSubdomain(x0=0.8, x1=1.2, n=12,
                                  E=70e3, nu=0.3, alpha=1e-4, T_ref=0.0)
    u = mech.solve(temperature_fn=T, clamp_x=1.2)
    ux_free = mech.eval_ux_at_x(0.8)
    assert ux_free < -1e-6  # expansion pushes the free end outward (-x)
