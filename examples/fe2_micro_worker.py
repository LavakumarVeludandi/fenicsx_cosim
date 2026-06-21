"""DEPRECATED placeholder — superseded by the real FE² example in ``fe2/``.

The previous version faked the RVE with a closed-form damage formula. The real
micro worker (a stateful pool of genuine two-phase J2 RVE solves) now lives in
``fe2/micro_worker.py``. This shim runs it so existing commands keep working.
See ``docs/fe2_design.md``.
"""
import runpy
import sys

print("[fe2_micro_worker] superseded -> running fe2.micro_worker (real physics).")
sys.argv = [sys.argv[0]] + sys.argv[1:]
runpy.run_module("fe2.micro_worker", run_name="__main__")
