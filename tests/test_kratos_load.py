"""Proof test: Kratos actually *loads* what the middleware writes.

meshio's mdpa output passed string-shaped checks yet Kratos rejected it (geometry
type name instead of a solvable element, no SubModelParts, no Properties linkage).
So substring assertions on the .mdpa are not proof. The real gate lives in
``_kratos_load_check.py``: Kratos ``ModelPartIO`` reads the written mesh,
``ReadMaterialsUtility`` reads the written materials.json, and the SubModelParts,
the solvable element, and the constitutive law all materialize.

That check runs in a clean child process because Kratos segfaults pytest's
interpreter on teardown (it deregisters its kernel atexit, which crashes once
pytest is imported into the same process). This test only orchestrates the child
and asserts it exited 0 with the success sentinel. Skips where Kratos is absent.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

_CHECK = Path(__file__).resolve().parent / "_kratos_load_check.py"
_OK = "KRATOS_LOAD_OK"

# Kratos lives in the machine's system python, not the FEniCSx conda env — the
# two-interpreter (conda FEniCSx <-> system Kratos) split that fenicsx-cosim is
# built to connect. So the interpreter that runs the Kratos check is independent
# of the one running pytest. Override with KRATOS_PYTHON to run the proof from a
# pytest launched in an env that lacks Kratos, e.g.:
#     KRATOS_PYTHON=/usr/bin/python3 conda run -n fenicsx-env pytest tests/test_kratos_load.py
_KRATOS_PY = os.environ.get("KRATOS_PYTHON", sys.executable)


def _kratos_usable() -> bool:
    """Probe the interpreter that will actually run the check (``_KRATOS_PY``).

    ``importorskip`` is wrong here: it probes the *orchestrator*, but the check
    runs in a separate-interpreter subprocess. They differ — e.g. in fenicsx-env
    the KratosMultiphysics *package* leaks in from another env yet importing it
    only warns "Unable to find KratosCore" with rc=0, so importorskip would NOT
    skip while the compiled core is unusable. Probe ``_KRATOS_PY`` with the exact
    imports ``_kratos_load_check.py`` needs and require a clean success.
    """
    probe = (
        "import KratosMultiphysics, "
        "KratosMultiphysics.StructuralMechanicsApplication; print('USABLE')"
    )
    r = subprocess.run([_KRATOS_PY, "-c", probe],
                       capture_output=True, text=True)
    return r.returncode == 0 and "USABLE" in r.stdout


pytestmark = pytest.mark.skipif(
    not _kratos_usable(),
    reason=f"KratosMultiphysics (+ StructuralMechanicsApplication) not usable "
           f"by {_KRATOS_PY} (set KRATOS_PYTHON to a Kratos-capable interpreter)",
)


def test_kratos_reads_middleware_mesh_and_materials():
    proc = subprocess.run([_KRATOS_PY, str(_CHECK)],
                          capture_output=True, text=True)
    assert _OK in proc.stdout, (
        f"Kratos rejected the middleware output (rc={proc.returncode}).\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
