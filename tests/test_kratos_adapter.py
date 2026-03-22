"""
Tests for KratosAdapter using mocked Kratos objects.

These tests run without requiring the KratosMultiphysics package to be
installed, allowing CI to pass on environments where building Kratos
is not feasible.
"""

from __future__ import annotations

import sys
import threading
from unittest.mock import MagicMock

import numpy as np
import pytest

from fenicsx_cosim.adapters.base import SolverAdapter


# ------------------------------------------------------------------
# Mocking KratosMultiphysics
# ------------------------------------------------------------------

class MockArray3:
    """Mocks KratosMultiphysics.Array3."""
    def __init__(self):
        self._data = [0.0, 0.0, 0.0]

    def __setitem__(self, i, val):
        self._data[i] = val

    def __getitem__(self, i):
        return self._data[i]

    def __eq__(self, other):
        if isinstance(other, MockArray3):
            return self._data == other._data
        return self._data == list(other)

    def __repr__(self):
        return f"Array3{self._data}"


class MockNode:
    """Mocks Kratos Node."""
    def __init__(self, node_id, x, y, z):
        self.Id = node_id
        self.X = x
        self.Y = y
        self.Z = z
        self._values = {}

    def SetSolutionStepValue(self, variable, step, value):
        self._values[variable] = value

    def GetSolutionStepValue(self, variable, step=0):
        return self._values.get(variable, 0.0)


class MockSubModelPart:
    """Mocks Kratos SubModelPart."""
    def __init__(self, name, nodes):
        self.name = name
        self.Nodes = nodes


class MockModelPart:
    """Mocks Kratos ModelPart."""
    def __init__(self):
        self._sub_parts = {}

    def add_sub_model_part(self, sub_part):
        self._sub_parts[sub_part.name] = sub_part

    def GetSubModelPart(self, name):
        if name not in self._sub_parts:
            raise RuntimeError(f"SubModelPart {name} not found")
        return self._sub_parts[name]


# Apply the mock to sys.modules BEFORE importing KratosAdapter
mock_kratos = MagicMock()
mock_kratos.Array3 = MockArray3
class MockVariable:
    def __init__(self, name):
        self.name = name

class MockGlobals:
    @staticmethod
    def GetVariable(name):
        if name == "INVALID":
            raise ValueError("Invalid variable")
        return MockVariable(name)

mock_kratos.KratosGlobals = MockGlobals()

sys.modules["KratosMultiphysics"] = mock_kratos

# Now we can safely import the adapter and force the HAS_KRATOS flag
import fenicsx_cosim.adapters.kratos_adapter as ka
ka._HAS_KRATOS = True
ka._MOCK_MODE = True  # Add a mock mode flag to bypass Kratos-specific type checks if necessary
from fenicsx_cosim.adapters.kratos_adapter import KratosAdapter
import KratosMultiphysics


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

@pytest.fixture
def mock_kratos_model():
    """Create a mock Kratos model with 3 nodes on an interface."""
    nodes = [
        MockNode(1, 0.0, 0.0, 0.0),
        MockNode(2, 1.0, 0.0, 0.0),
        MockNode(3, 0.0, 1.0, 0.0),
    ]
    # Preheat some scalar and vector values
    nodes[0].SetSolutionStepValue("VAR_TEMPERATURE", 0, 100.0)
    nodes[1].SetSolutionStepValue("VAR_TEMPERATURE", 0, 110.0)
    nodes[2].SetSolutionStepValue("VAR_TEMPERATURE", 0, 120.0)

    vec0 = MockArray3(); vec0[0]=0.1; vec0[1]=0.0; vec0[2]=0.0
    vec1 = MockArray3(); vec1[0]=0.2; vec1[1]=0.0; vec1[2]=0.0
    vec2 = MockArray3(); vec2[0]=0.3; vec2[1]=0.0; vec2[2]=0.0
    
    nodes[0].SetSolutionStepValue("VAR_VELOCITY", 0, vec0)
    nodes[1].SetSolutionStepValue("VAR_VELOCITY", 0, vec1)
    nodes[2].SetSolutionStepValue("VAR_VELOCITY", 0, vec2)

    sub_part = MockSubModelPart("coupling_interface", nodes)
    model_part = MockModelPart()
    model_part.add_sub_model_part(sub_part)
    
    return model_part

@pytest.fixture
def adapter(mock_kratos_model):
    from fenicsx_cosim.adapters.kratos_adapter import KratosAdapter
    return KratosAdapter(mock_kratos_model, "coupling_interface")


class TestKratosAdapter:

    def test_initialization(self, adapter):
        """Adapter binds to the correct sub-model-part."""
        assert adapter.num_nodes == 3
        assert adapter.node_ids == [1, 2, 3]
        
    def test_get_boundary_coordinates(self, adapter):
        """Coordinates are extracted as (N,3) array."""
        coords = adapter.get_boundary_coordinates()
        
        assert coords.shape == (3, 3)
        np.testing.assert_array_equal(coords[0], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(coords[1], [1.0, 0.0, 0.0])
        np.testing.assert_array_equal(coords[2], [0.0, 1.0, 0.0])

    def test_extract_scalar_field(self, adapter):
        """Scalar extraction uses GetSolutionStepValue."""
        # Using VAR_ prefix because our mock nodes were pre-heated with VAR_
        adapter._resolve_variable = lambda name: f"VAR_{name}"
        temp = adapter.extract_field("TEMPERATURE")
        
        assert temp.shape == (3,)
        np.testing.assert_array_equal(temp, [100.0, 110.0, 120.0])

    def test_extract_vector_field(self, adapter):
        """Vector extraction converts Array3 to (N,3) ndarray."""
        adapter._resolve_variable = lambda name: f"VAR_{name}"
        vel = adapter.extract_vector_field("VELOCITY")
        
        assert vel.shape == (3, 3)
        np.testing.assert_array_equal(vel[0], [0.1, 0.0, 0.0])
        np.testing.assert_array_equal(vel[1], [0.2, 0.0, 0.0])
        np.testing.assert_array_equal(vel[2], [0.3, 0.0, 0.0])

    def test_inject_scalar_field(self, adapter, mock_kratos_model):
        """Scalar injection uses SetSolutionStepValue."""
        adapter._resolve_variable = lambda name: f"VAR_{name}"
        
        new_pressure = np.array([10.0, 20.0, 30.0])
        adapter.inject_field("PRESSURE", new_pressure)
        
        # Verify it went into the mock nodes
        nodes = mock_kratos_model.GetSubModelPart("coupling_interface").Nodes
        assert nodes[0]._values["VAR_PRESSURE"] == 10.0
        assert nodes[1]._values["VAR_PRESSURE"] == 20.0
        assert nodes[2]._values["VAR_PRESSURE"] == 30.0

    def test_inject_vector_field(self, adapter, mock_kratos_model):
        """Vector injection converts (N,3) ndarray to Array3."""
        adapter._resolve_variable = lambda name: f"VAR_{name}"
        
        new_disp = np.array([
            [0.1, 0.1, 0.0],
            [0.2, 0.2, 0.0],
            [0.3, 0.3, 0.0]
        ])
        adapter.inject_vector_field("DISPLACEMENT", new_disp)
        
        nodes = mock_kratos_model.GetSubModelPart("coupling_interface").Nodes
        # Mocks implement __eq__ against list
        assert nodes[0]._values["VAR_DISPLACEMENT"] == [0.1, 0.1, 0.0]
        assert nodes[1]._values["VAR_DISPLACEMENT"] == [0.2, 0.2, 0.0]
        assert nodes[2]._values["VAR_DISPLACEMENT"] == [0.3, 0.3, 0.0]

    def test_invalid_field_shape(self, adapter):
        with pytest.raises(ValueError, match="Expected 3 values"):
            adapter.inject_field("PRESSURE", np.array([10.0, 20.0]))
            
        with pytest.raises(ValueError, match="Expected shape"):
            adapter.inject_vector_field("DISPLACEMENT", np.array([[1.0, 2.0]]))

    def test_variable_resolution(self, mock_kratos_model):
        """Checks variable_map vs KratosGlobals fallback."""
        from fenicsx_cosim.adapters.kratos_adapter import KratosAdapter
        vmap = {"CUSTOM_VAR": "MAPPED_VAR"}
        adapter = KratosAdapter(
            mock_kratos_model, "coupling_interface", variable_map=vmap
        )
        
        # Un-mock for this specific test
        # Uses mapping
        adapter.inject_field("CUSTOM_VAR", np.array([1., 2., 3.]))
        nodes = mock_kratos_model.GetSubModelPart("coupling_interface").Nodes
        assert nodes[0]._values["MAPPED_VAR"] == 1.0
        
        # Valid fallback
        adapter.inject_field("FALLBACK", np.array([4., 5., 6.]))
        assert any(k.name == "FALLBACK" for k in nodes[0]._values.keys() if hasattr(k, "name"))
        
        # Invalid fallback
        with pytest.raises(KeyError, match="Cannot resolve"):
            adapter.extract_field("INVALID")

