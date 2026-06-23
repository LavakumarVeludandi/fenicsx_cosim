"""FE² macro solver — real 2D elasticity driven by remote RVE workers.

Replaces the placeholder ``fe2_macro_solver.py`` (which faked strains and
stresses). The macro physics lives in :class:`~fe2.macro_driver.MacroFE2Problem`;
here the ``stress_callback`` is wired to the ``fenicsx_cosim`` scatter-gather
socket, so each per-cell macro strain is solved by a genuine micro RVE on a
worker process. The exact same driver runs in-process in the test suite by
swapping the callback for a local :class:`StatefulRVEPool` — that is the point
of the dependency-injected callback (rapid-prototyper / connector).

Run::

    python -m fe2.micro_worker          # terminal 1 (start first)
    python -m fe2.macro_solver          # terminal 2
"""

from fenicsx_cosim.scatter_gather_communicator import ScatterGatherCommunicator

from fe2.macro_driver import MacroFE2Problem


def main() -> None:
    macro = MacroFE2Problem(
        n=4, E_eff=85.0e3, nu_eff=0.3, applied_strain_xx=4.0e-3, n_steps=4
    )

    sg = ScatterGatherCommunicator(
        role="master",
        push_endpoint="tcp://*:5556",   # ventilator -> workers
        pull_endpoint="tcp://*:5557",   # collector  <- workers
    )
    sg.slow_start(0.5)  # let workers connect before the first scatter

    def stress_callback(strains, commit):
        meta = [{"commit": bool(commit)} for _ in strains]
        sg.scatter(strains, meta)
        return sg.gather(len(strains), ordered=True)

    sxx = macro.solve(stress_callback)
    print(f"[macro_solver] done. converged={macro.converged_flags} "
          f"volume-avg sigma_xx={sxx:.3f}", flush=True)

    sg.broadcast_shutdown(n_workers=1)
    sg.close()


if __name__ == "__main__":
    main()
