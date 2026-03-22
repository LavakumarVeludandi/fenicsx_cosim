"""
Example: Kratos Thermal Solver communicating with FEniCSx.

This script demonstrates the usage of the KratosAdapter. It simulates
a thermal solve using Kratos Native API, exchanging coupling data with a
partner mechanical solver (e.g. fenicsx_kratos_mechanical.py).

Usage
-----
    Terminal 1:  python kratos_thermal_solver.py
    Terminal 2:  python fenicsx_kratos_mechanical.py
"""

import sys
import numpy as np

try:
    import KratosMultiphysics
except ImportError:
    print("This example requires KratosMultiphysics. Please install it.")
    sys.exit(0)

from fenicsx_cosim.adapters import KratosAdapter
from fenicsx_cosim import CouplingInterface


def create_mock_kratos_model():
    """Sets up a minimal Kratos model with an interface."""
    model = KratosMultiphysics.Model()
    model_part = model.CreateModelPart("ThermalDomain")
    
    # Add variables to the ModelPart (required by Kratos before adding nodes)
    model_part.AddNodalSolutionStepVariable(KratosMultiphysics.TEMPERATURE)
    model_part.AddNodalSolutionStepVariable(KratosMultiphysics.DISPLACEMENT)

    # Let's say we have a 1D interface with 5 nodes
    for i in range(1, 6):
        x = (i - 1) * 0.25
        # Kratos nodes: ID, X, Y, Z
        node = model_part.CreateNewNode(i, x, 1.0, 0.0)
        
    # Create the SubModelPart for coupling
    interface = model_part.CreateSubModelPart("coupling_interface")
    interface.AddNodes([1, 2, 3, 4, 5])
    
    # Initial temperature gradient
    for node in interface.Nodes:
        node.SetSolutionStepValue(KratosMultiphysics.TEMPERATURE, 0, 300.0 + x * 50.0)
        
    return model_part

def main():
    print("[KratosSolver] Initializing ModelPart...")
    model_part = create_mock_kratos_model()
    
    # Initialize the adapter
    print("[KratosSolver] Initializing Adapter...")
    adapter = KratosAdapter(model_part, "coupling_interface")
    
    # Initialize the coupling interface
    print("[KratosSolver] Waiting for FEniCSx solver...")
    cosim = CouplingInterface.from_adapter(
        adapter=adapter,
        name="KratosThermal",
        partner_name="FEniCSxMechanical",
        role="bind",  # Kratos will act as the server
        endpoint="tcp://*:5555"
    )
    
    # Register the interface (exchanges coordinates for mapping)
    cosim.register_adapter_interface()
    
    T_final = 0.5
    dt = 0.1
    t = 0.0
    step = 0
    
    while t < T_final - 1e-10:
        t += dt
        step += 1
        print(f"\n[KratosSolver] === Step {step}, t={t:.2f} ===")
        
        # 1. Update Temperature (mocked solve)
        interface = model_part.GetSubModelPart("coupling_interface")
        for node in interface.Nodes:
            old_temp = node.GetSolutionStepValue(KratosMultiphysics.TEMPERATURE)
            new_temp = old_temp + 5.0 * np.sin(2 * np.pi * t)
            node.SetSolutionStepValue(KratosMultiphysics.TEMPERATURE, 0, new_temp)
            
        print(f"  Max Temperature: {max(n.GetSolutionStepValue(KratosMultiphysics.TEMPERATURE) for n in interface.Nodes):.2f}")
            
        # 2. Export Temperature
        cosim.export_via_adapter("TEMPERATURE")
        print("  Exported TEMPERATURE")
        
        # 3. Import Displacement (vector field needs vector extraction)
        # Note: Kratos extracts/injects vectors slightly differently, so we 
        # extract explicitly as a vector via the adapter's custom method or 
        # map it before using the general import_via_adapter
        
        # The generic import_via_adapter will just dump the NumPy array into inject_field,
        # which expects (N,) for scalars. For vectors, we need a slight adjustment.
        # Since DISPLACEMENT is (N, 3), we manually receive it and call inject_vector_field
        name, disp_array = cosim._communicator.receive_array()
        if cosim.mapper is not None:
            disp_array = cosim.mapper.map(disp_array)
            
        adapter.inject_vector_field("DISPLACEMENT", disp_array)
        print("  Imported DISPLACEMENT")
        
        # 4. Sync
        cosim.advance_adapter()
        print(f"  Synchronized (step {cosim.step_count})")

    cosim.disconnect()
    print("\n[KratosSolver] Co-simulation complete.")

if __name__ == "__main__":
    main()
