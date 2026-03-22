"""
Tests for AbaqusFileAdapter.

These tests use a temporary directory to mock the shared filesystem
between Abaqus and FEniCSx. They verify that the adapter correctly
reads and writes NumPy .npy files representing the coupling boundary and fields.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from fenicsx_cosim.adapters.abaqus_adapter import AbaqusFileAdapter

# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

@pytest.fixture
def exchange_dir():
    """Create a temporary directory for file exchange."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def setup_abaqus_mock(exchange_dir):
    """Write mock Abaqus pre-processing files."""
    # 1. Coordinates: 4 nodes in a square
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0]
    ])
    np.save(exchange_dir / "boundary_coords.npy", coords)
    
    # 2. Mock some Abaqus output (e.g. TEMPERATURE)
    temp_out = np.array([300.0, 310.0, 320.0, 315.0])
    np.save(exchange_dir / "TEMPERATURE_out.npy", temp_out)
    
    # 3. Mock vector output (e.g. REACTION_FORCE)
    rf_out = np.array([
        [10.0, 0.0, 0.0],
        [20.0, 0.0, 0.0],
        [30.0, 0.0, 0.0],
        [40.0, 0.0, 0.0]
    ])
    np.save(exchange_dir / "REACTION_FORCE_out.npy", rf_out)

    return exchange_dir, coords

class TestAbaqusFileAdapter:

    def test_initialization(self, setup_abaqus_mock):
        """Adapter initializes and creates the directory if needed."""
        exch_dir, coords = setup_abaqus_mock
        adapter = AbaqusFileAdapter(exch_dir)
        
        # Verify it finds the directory correctly
        assert adapter.exchange_dir == exch_dir
        assert adapter.exchange_dir.exists()

    def test_get_boundary_coordinates(self, setup_abaqus_mock):
        """Adapter correctly reads 'boundary_coords.npy'."""
        exch_dir, expected_coords = setup_abaqus_mock
        adapter = AbaqusFileAdapter(exch_dir)
        
        coords = adapter.get_boundary_coordinates()
        assert coords.shape == (4, 3)
        np.testing.assert_array_equal(coords, expected_coords)
        
    def test_missing_coordinates(self, exchange_dir):
        """Adapter raises an error if coordinates file is missing."""
        adapter = AbaqusFileAdapter(exchange_dir)
        
        with pytest.raises(FileNotFoundError, match="Missing coordinate file"):
            adapter.get_boundary_coordinates()

    def test_extract_field_scalar(self, setup_abaqus_mock):
        """Adapter reads '<FIELD>_out.npy' written by Abaqus."""
        exch_dir, _ = setup_abaqus_mock
        adapter = AbaqusFileAdapter(exch_dir)
        
        temp = adapter.extract_field("TEMPERATURE")
        assert temp.shape == (4,)
        np.testing.assert_array_equal(temp, [300.0, 310.0, 320.0, 315.0])
        
    def test_extract_field_vector(self, setup_abaqus_mock):
        """Adapter reads vector '<FIELD>_out.npy' written by Abaqus."""
        exch_dir, _ = setup_abaqus_mock
        adapter = AbaqusFileAdapter(exch_dir)
        
        rf = adapter.extract_field("REACTION_FORCE")
        assert rf.shape == (4, 3)
        assert rf[0, 0] == 10.0
        assert rf[3, 0] == 40.0

    def test_inject_field_scalar(self, exchange_dir):
        """Adapter writes to '<FIELD>_in.npy' for Abaqus to read."""
        adapter = AbaqusFileAdapter(exchange_dir)
        
        # Inject pressure
        pressure = np.array([10.0, 20.0, 30.0, 40.0])
        adapter.inject_field("PRESSURE", pressure)
        
        # Verify file was written
        file_path = exchange_dir / "PRESSURE_in.npy"
        assert file_path.exists()
        
        # Verify contents
        written_data = np.load(file_path)
        np.testing.assert_array_equal(written_data, pressure)

    def test_inject_field_vector(self, exchange_dir):
        """Adapter writes vector data to '<FIELD>_in.npy'."""
        adapter = AbaqusFileAdapter(exchange_dir)
        
        # Inject displacements
        disp = np.array([
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.3, 0.0, 0.0],
            [0.4, 0.0, 0.0]
        ])
        adapter.inject_field("DISPLACEMENT", disp)
        
        # Verify file was written
        file_path = exchange_dir / "DISPLACEMENT_in.npy"
        assert file_path.exists()
        
        # Verify contents
        written_data = np.load(file_path)
        np.testing.assert_array_equal(written_data, disp)

    def test_missing_field(self, setup_abaqus_mock):
        """Adapter raises error if requesting a missing mapped file."""
        exch_dir, _ = setup_abaqus_mock
        adapter = AbaqusFileAdapter(exch_dir)
        
        with pytest.raises(FileNotFoundError, match="Missing output field file"):
            adapter.extract_field("NONEXISTENT_VAR")

    def test_metadata(self, exchange_dir):
        """Adapter provides correct metadata."""
        adapter = AbaqusFileAdapter(exchange_dir)
        meta = adapter.get_metadata()
        
        assert meta["solver"] == "Abaqus (File-based)"
        assert meta["exchange_dir"] == str(exchange_dir.absolute())
