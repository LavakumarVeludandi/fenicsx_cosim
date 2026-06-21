"""DEPRECATED placeholder — superseded by the real example in ``thermomech/``.

The previous version faked the elasticity solve with ``interpolate(lambda ...)``.
The real partitioned thermo-mechanical coupling (genuine FEniCSx heat +
thermoelasticity, overlapping Schwarz over fenicsx-cosim) lives in
``thermomech/``. This shim runs subdomain B (heat + thermoelastic solve). See
the honesty note in ``thermomech/thermal_a.py``.
"""
import runpy

print("[mechanical_solver] superseded -> running thermomech.thermal_b (real physics).")
runpy.run_module("thermomech.thermal_b", run_name="__main__")
