"""DEPRECATED placeholder — superseded by the real FE² example in ``fe2/``.

The previous version faked the macro strains and stresses. The real macro
solver (genuine 2D elasticity driven by remote J2 RVE workers) now lives in
``fe2/macro_solver.py``. This shim runs it so existing commands keep working.
See ``docs/fe2_design.md``.
"""
import runpy

print("[fe2_macro_solver] superseded -> running fe2.macro_solver (real physics).")
runpy.run_module("fe2.macro_solver", run_name="__main__")
