"""FE² micro worker — real two-phase J2 RVE solves over scatter-gather.

Replaces the placeholder closed-form damage formula of the old
``fe2_micro_worker.py``. A single stateful worker owns all RVE states (keyed by
the macro quadrature-point index carried in each message), which is what makes
a *path-dependent* J2 FE² correct over the round-robin socket (no affinity).
See ``docs/fe2_design.md`` for the milestone-2 plan to parallelise.

Run (single worker, milestone 1)::

    python -m fe2.micro_worker [master_host]

alongside ``python -m fe2.macro_solver``.
"""

import os
import sys

from fenicsx_cosim.scatter_gather_communicator import ScatterGatherCommunicator

from fe2.stateful_pool import StatefulRVEPool

# Micro material model lives entirely on the worker side; the macro solver only
# sends strains + a commit flag. RVE mesh resolution is env-tunable so tests
# can shrink it (FE2_RVE_N).
RVE_KWARGS = dict(
    n=int(os.environ.get("FE2_RVE_N", "8")),
    E_matrix=70.0e3, nu_matrix=0.3, sigma_y=200.0, H=1.0e3,
    E_incl=210.0e3, nu_incl=0.2, incl_radius=0.25,
)


def main() -> None:
    master_host = sys.argv[1] if len(sys.argv) > 1 else "localhost"
    pool = StatefulRVEPool(rve_kwargs=RVE_KWARGS)

    sg = ScatterGatherCommunicator(
        role="worker",
        push_endpoint=f"tcp://{master_host}:5557",  # -> master collector
        pull_endpoint=f"tcp://{master_host}:5556",  # <- master ventilator
    )
    print("[micro_worker] connected; waiting for macro strains...", flush=True)

    def solve_fn(index, strain, meta):
        # meta carries the trial/commit flag from the macro Newton loop.
        commit = bool(meta.get("commit", True))
        return pool.solve(index, strain, commit=commit)

    n = sg.work_loop(solve_fn)
    print(f"[micro_worker] shutdown after {n} RVE solves "
          f"({pool.num_rves} distinct RVEs).", flush=True)
    sg.close()


if __name__ == "__main__":
    main()
