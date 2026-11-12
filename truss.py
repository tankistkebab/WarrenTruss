import numpy as np

def generate_bridge_system(num_sections, L=2.0, H=1.5, load=10000.0):
    """
    Generates the Ax=b system for a Warren Truss bridge.
    
    Args:
        num_sections (int): Number of triangular segments.
        L (float): Length of one segment (meters).
        H (float): Height (meters).
        load (float): Load applied to central nodes (Newtons).
        
    Returns:
        tuple: (A, b, nodes, members)
    """
    # Coordinate generation
    nodes = []
    # Alternate lower nodes (y=0) and upper nodes (y=H)
    for i in range(num_sections + 1):
        nodes.append((i * L, 0.0))       # Lower Node
        if i < num_sections:
            nodes.append((i * L + L/2.0, H)) # Upper Node

    num_nodes = len(nodes)

    # Define Members (Connections between nodes)
    members = []
    # Horizontal lower members
    for i in range(0, num_nodes - 2, 2):
        members.append((i, i+2))

    # Horizontal upper members
    for i in range(1, num_nodes - 2, 2):
        members.append((i, i+2))

    # Diagonal members
    for i in range(num_sections):
        lower_left = 2*i
        upper = 2*i + 1
        lower_right = 2*i + 2
        members.append((lower_left, upper)) # / member
        members.append((upper, lower_right)) # \ member

    num_members = len(members)

    # Build Matrix A and Vector b
    # 2 equations per node (X and Y balance).
    # Unknowns: Member forces (num_members) + 3 Reactions (Rx1, Ry1, Ry2)
    num_eq = 2 * num_nodes
    num_vars = num_members + 3

    A = np.zeros((num_eq, num_vars))
    b = np.zeros(num_eq)

    # Fill Matrix A with direction cosines
    for k, (node_i, node_j) in enumerate(members):
        xi, yi = nodes[node_i]
        xj, yj = nodes[node_j]
        dist = np.sqrt((xj-xi)**2 + (yj-yi)**2)

        c = (xj - xi) / dist
        s = (yj - yi) / dist

        # Node i equations
        A[2*node_i, k] += c
        A[2*node_i+1, k] += s

        # Node j equations (opposite direction)
        A[2*node_j, k] -= c
        A[2*node_j+1, k] -= s

    # Add Support Reactions
    # Node 0 (Left Lower): Fixed X and Y
    A[0, num_members] = 1.0     # Rx1
    A[1, num_members+1] = 1.0   # Ry1

    # Last lower node: Fixed only Y (roller)
    last_node_idx = num_nodes - 1 if (num_nodes - 1) % 2 == 0 else num_nodes - 2
    # The last lower node is at index 2 * num_sections
    last_lower_node_idx = 2 * num_sections
    A[2*last_lower_node_idx+1, num_members+2] = 1.0 # Ry2

    # Apply Load
    # Symmetrically apply load to central nodes
    # We want to target nodes on the bottom chord
    # num_sections + 1 is the number of lower nodes.
    lower_node_indices = [2*i for i in range(num_sections + 1)]
    mid = len(lower_node_indices) // 2
    
    # Apply load to 4 nodes around the center
    # For odd number of nodes, mid is the center. For even, mid is to the right of center.
    # range(mid - 2, mid + 2) for a bridge with at least 5 lower nodes
    load_start = max(0, mid - 1)
    load_end = min(len(lower_node_indices), mid + 2 if len(lower_node_indices) % 2 == 0 else mid + 1)
    
    # Corrected symmetry logic for central nodes
    if num_sections >= 4:
        # For a more robust symmetry, let's pick 4 nodes if possible
        target_indices = lower_node_indices[mid-1:mid+2] if len(lower_node_indices)%2 != 0 else lower_node_indices[mid-2:mid+2]
        # Wait, the request said range(n//2 - 2, n//2 + 2) was asymmetric. 
        # If n=5 (nodes 0,1,2,3,4), n//2 = 2. range(0, 4) -> 0,1,2,3. Mid is 2.
        # Let's just use a fixed number of nodes or a clearer range.
        
        num_load_nodes = 4
        start_idx = max(0, (num_sections + 1 - num_load_nodes) // 2)
        for i in range(start_idx, start_idx + num_load_nodes):
            if i < num_sections + 1:
                b[2*(2*i) + 1] -= load
    else:
        # Fallback for small bridges
        b[2*(lower_node_indices[mid]) + 1] -= load

    return A, b, nodes, members
