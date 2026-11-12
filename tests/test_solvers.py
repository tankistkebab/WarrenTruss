import pytest
import numpy as np
from truss import generate_bridge_system
from solvers import gauss_partial_pivoting, gauss_no_pivoting

def test_solver_accuracy():
    """Verify that GPP solver matches NumPy's solution."""
    A, b, _, _ = generate_bridge_system(10)
    x_ref = np.linalg.solve(A, b)
    x_gpp = gauss_partial_pivoting(A, b)
    
    np.testing.assert_allclose(x_gpp, x_ref, atol=1e-8)

def test_vertical_reactions():
    """Verify that the sum of vertical reactions equals the total applied load."""
    load_val = 10000.0
    num_sections = 10
    A, b, nodes, members = generate_bridge_system(num_sections, load=load_val)
    x_sol = gauss_partial_pivoting(A, b)
    
    # Reactions are at the end of the solution vector
    num_members = len(members)
    # x_sol = [member_forces..., Rx1, Ry1, Ry2]
    Ry1 = x_sol[num_members + 1]
    Ry2 = x_sol[num_members + 2]
    
    total_reaction = Ry1 + Ry2
    total_load = np.sum(b[1::2]) # All vertical loads are in b[odd_indices]
    
    assert total_load > 0
    np.testing.assert_allclose(total_reaction, total_load, atol=1e-8)

def test_symmetry():
    """Verify that for a symmetric bridge and load, reactions are equal."""
    # Test with even number of lower nodes (num_sections = 9 -> 10 nodes)
    # and odd number of lower nodes (num_sections = 10 -> 11 nodes)
    for n in [9, 10]:
        A, b, nodes, members = generate_bridge_system(n)
        x_sol = gauss_partial_pivoting(A, b)
        
        num_members = len(members)
        Ry1 = x_sol[num_members + 1]
        Ry2 = x_sol[num_members + 2]
        
        np.testing.assert_allclose(Ry1, Ry2, atol=1e-8, err_msg=f"Failed symmetry for n={n}")

def test_naive_gauss_failure():
    """Naive Gauss should raise ValueError if it hits a zero pivot."""
    A, b, _, _ = generate_bridge_system(50)
    with pytest.raises(ValueError):
        gauss_no_pivoting(A, b)
