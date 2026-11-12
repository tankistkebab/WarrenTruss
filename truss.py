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
    last_lower_node_idx = 2 * num_sections
    A[2*last_lower_node_idx+1, num_members+2] = 1.0 # Ry2

    # Apply Load to central lower nodes
    num_lower = num_sections + 1
    num_load_nodes = 4 if num_lower % 2 == 0 else 3
    start_idx = (num_lower - num_load_nodes) // 2
    for i in range(start_idx, start_idx + num_load_nodes):
        node_idx = 2 * i
        b[2*node_idx + 1] += load

    return A, b, nodes, members
