"""
AbaqusFileAdapter — File-based staggered coupling for Abaqus.

Abaqus does not provide a live Python API that can easily integrate with
ZeroMQ communication during an implicit timestep. Instead, this adapter
facilitates a file-based staggered coupling approach.

The FEniCSx solver (or a Python wrapper running alongside Abaqus) uses
this adapter to read/write Numpy ``.npy`` files in a designated exchange
directory.

Workflow
--------
1. A pre-processing step writes boundary coordinates to ``<exch_dir>/boundary_coords.npy``.
2. During the time loop, Abaqus writes its output (e.g. TEMPERATURE) to
   ``<exch_dir>/TEMPERATURE_out.npy``.
3. The coupling interface extracts this parameter and sends it to FEniCSx.
4. FEniCSx sends back its output (e.g. DISPLACEMENT).
5. The adapter injects (writes) this parameter to ``<exch_dir>/DISPLACEMENT_in.npy``.
6. The Abaqus script reads the ``_in.npy`` file to apply boundary conditions
   for the next step.

Attributes
----------
exchange_dir : Path
    Directory where data files are exchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fenicsx_cosim.adapters.base import SolverAdapter
from fenicsx_cosim.utils import get_logger

logger = get_logger(__name__)


class AbaqusFileAdapter(SolverAdapter):
    """Adapter for file-based coupling with Abaqus.

    Parameters
    ----------
    exchange_dir : str | Path
        Path to the shared directory where .npy files are written/read.
    timeout_s : float, optional
        Maximum time to wait for expected files to appear. Currently unused
        as exact file presence is checked instantly for simplicity, but
        could be extended for polling file locks.
    """

    def __init__(
        self,
        exchange_dir: str | Path,
        timeout_s: float = 60.0,
    ) -> None:
        self.exchange_dir = Path(exchange_dir)
        self.timeout_s = timeout_s

        if not self.exchange_dir.exists():
            self.exchange_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                "Created Abaqus exchange directory: %s", self.exchange_dir
            )

    # ------------------------------------------------------------------
    # SolverAdapter interface
    # ------------------------------------------------------------------

    def get_boundary_coordinates(self) -> np.ndarray:
        """Read coordinates from 'boundary_coords.npy'.

        Returns
        -------
        np.ndarray
            Shape ``(N, 3)``.

        Raises
        ------
        FileNotFoundError
            If the coordinates file does not exist.
        """
        coord_file = self.exchange_dir / "boundary_coords.npy"
        if not coord_file.exists():
            raise FileNotFoundError(
                f"Missing coordinate file: {coord_file}. "
                "Ensure the Abaqus pre-processing step writes this file."
            )
        coords = np.load(coord_file)
        logger.info("Loaded %d coordinates from %s", len(coords), coord_file)
        return coords

    def extract_field(self, field_name: str) -> np.ndarray:
        """Read field from '<field_name>_out.npy'.

        Parameters
        ----------
        field_name : str
            Name of the field.

        Returns
        -------
        np.ndarray
            Shape ``(N,)`` or ``(N, 3)``.
        """
        filename = self.exchange_dir / f"{field_name}_out.npy"
        if not filename.exists():
            raise FileNotFoundError(f"Missing output field file: {filename}")
        data = np.load(filename)
        logger.debug("Read %s from %s", field_name, filename.name)
        return data

    def inject_field(self, field_name: str, values: np.ndarray) -> None:
        """Write field to '<field_name>_in.npy'.

        Parameters
        ----------
        field_name : str
            Name of the field.
        values : np.ndarray
            Values to write.
        """
        filename = self.exchange_dir / f"{field_name}_in.npy"
        np.save(filename, values)
        logger.debug("Wrote %s to %s", field_name, filename.name)

    def advance(self) -> None:
        """No-op — File synchronization handles step progression implicitly."""
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_metadata(self) -> dict[str, str]:
        return {
            "solver": "Abaqus (File-based)",
            "exchange_dir": str(self.exchange_dir.absolute()),
        }
