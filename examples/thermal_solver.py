"""DEPRECATED placeholder — superseded by the real example in ``thermomech/``.

The previous version faked the heat solve with ``interpolate(lambda ...)``.
The real partitioned thermo-mechanical coupling (genuine FEniCSx heat +
thermoelasticity, overlapping Schwarz over fenicsx-cosim) lives in
``thermomech/``. This shim runs subdomain A. See ``docs/fe2_design.md`` and the
honesty note in ``thermomech/thermal_a.py``.
"""
import runpy

print("[thermal_solver] superseded -> running thermomech.thermal_a (real physics).")
runpy.run_module("thermomech.thermal_a", run_name="__main__")
