"""
Integration tests for KratosAdapter using a real local Kratos installation.

These tests actually instantiate Kratos ModelParts and test the adapter logic
with real C++ backed objects. If Kratos is not installed, these tests are skipped.
"""

import numpy as np
import pytest

try:
    import KratosMultiphysics as KM
    from KratosMultiphysics import StructuralMechanicsApplication as SMA
    HAS_KRATOS = True
except ImportError:
    HAS_KRATOS = False

from fenicsx_cosim.adapters.kratos_adapter import KratosAdapter

pytestmark = pytest.mark.skip(
    reason="Manual-only integration tests requiring a local Kratos installation."
)

@pytest.fixture
def kratos_model_part():
    """Create a real Kratos ModelPart with a coupling interface."""
    model = KM.Model()
    model_part = model.CreateModelPart("Main")
    
    # Add variables we want to test
    model_part.AddNodalSolutionStepVariable(KM.DISPLACEMENT)
    model_part.AddNodalSolutionStepVariable(KM.TEMPERATURE)
    model_part.AddNodalSolutionStepVariable(KM.PRESSURE)
    
    interface = model_part.CreateSubModelPart("coupling_interface")
    
    # Add some nodes
    node1 = model_part.CreateNewNode(1, 0.0, 0.0, 0.0)
    node2 = model_part.CreateNewNode(2, 1.0, 0.0, 0.0)
    node3 = model_part.CreateNewNode(3, 0.0, 1.0, 0.0)
    
    # Add nodes to interface
    interface.AddNodes([1, 2, 3])
    
    # Set some initial values
    node1.SetSolutionStepValue(KM.TEMPERATURE, 0, 100.0)
    node2.SetSolutionStepValue(KM.TEMPERATURE, 0, 110.0)
    node3.SetSolutionStepValue(KM.TEMPERATURE, 0, 120.0)
    
    node1.SetSolutionStepValue(KM.DISPLACEMENT, 0, [0.1, 0.0, 0.0])
    node2.SetSolutionStepValue(KM.DISPLACEMENT, 0, [0.2, 0.0, 0.0])
    node3.SetSolutionStepValue(KM.DISPLACEMENT, 0, [0.3, 0.0, 0.0])
    
    return model_part

class TestRealKratosAdapter:
    def test_coordinates_extraction(self, kratos_model_part):
        adapter = KratosAdapter(kratos_model_part, "coupling_interface")
        coords = adapter.get_boundary_coordinates()
        
        assert coords.shape == (3, 3)
        np.testing.assert_array_equal(coords[0], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(coords[1], [1.0, 0.0, 0.0])
        np.testing.assert_array_equal(coords[2], [0.0, 1.0, 0.0])

    def test_scalar_field_extraction(self, kratos_model_part):
        adapter = KratosAdapter(kratos_model_part, "coupling_interface")
        temp = adapter.extract_field("TEMPERATURE")
        
        assert temp.shape == (3,)
        np.testing.assert_array_equal(temp, [100.0, 110.0, 120.0])

    def test_vector_field_extraction(self, kratos_model_part):
        adapter = KratosAdapter(kratos_model_part, "coupling_interface")
        disp = adapter.extract_vector_field("DISPLACEMENT")
        
        assert disp.shape == (3, 3)
        assert np.allclose(disp[0], [0.1, 0.0, 0.0])
        assert np.allclose(disp[1], [0.2, 0.0, 0.0])
        assert np.allclose(disp[2], [0.3, 0.0, 0.0])

    def test_scalar_field_injection(self, kratos_model_part):
        adapter = KratosAdapter(kratos_model_part, "coupling_interface")
        
        new_pressure = np.array([10.0, 20.0, 30.0])
        adapter.inject_field("PRESSURE", new_pressure)
        
        # Read back via Kratos API to verify
        interface = kratos_model_part.GetSubModelPart("coupling_interface")
        nodes = list(interface.Nodes)
        assert nodes[0].GetSolutionStepValue(KM.PRESSURE, 0) == 10.0
        assert nodes[1].GetSolutionStepValue(KM.PRESSURE, 0) == 20.0
        assert nodes[2].GetSolutionStepValue(KM.PRESSURE, 0) == 30.0

    def test_vector_field_injection(self, kratos_model_part):
        adapter = KratosAdapter(kratos_model_part, "coupling_interface")
        
        new_disp = np.array([
            [1.1, 1.2, 1.3],
            [2.1, 2.2, 2.3],
            [3.1, 3.2, 3.3]
        ])
        adapter.inject_vector_field("DISPLACEMENT", new_disp)
        
        # Read back via Kratos API
        interface = kratos_model_part.GetSubModelPart("coupling_interface")
        nodes = list(interface.Nodes)
        
        d1 = nodes[0].GetSolutionStepValue(KM.DISPLACEMENT, 0)
        assert np.allclose([d1[0], d1[1], d1[2]], [1.1, 1.2, 1.3])
        
        d2 = nodes[1].GetSolutionStepValue(KM.DISPLACEMENT, 0)
        assert np.allclose([d2[0], d2[1], d2[2]], [2.1, 2.2, 2.3])

if __name__ == "__main__":
    if not HAS_KRATOS:
        print("SKIPPED: KratosMultiphysics is not installed.")
        import sys
        sys.exit(0)
    
    print("Running Real Kratos Adapter Tests...")
    part = kratos_model_part()
    tester = TestRealKratosAdapter()
    
    tester.test_coordinates_extraction(part)
    print("✓ test_coordinates_extraction")
    tester.test_scalar_field_extraction(part)
    print("✓ test_scalar_field_extraction")
    tester.test_vector_field_extraction(part)
    print("✓ test_vector_field_extraction")
    tester.test_scalar_field_injection(part)
    print("✓ test_scalar_field_injection")
    tester.test_vector_field_injection(part)
    print("✓ test_vector_field_injection")
    
    print("ALL TESTS PASSED WITH NATIVE KRATOS MULTIPHYSICS!")
