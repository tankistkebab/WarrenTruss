import numpy as np

def utris(U, b):
    """
    Solves the upper triangular system Ux = b using back substitution.
    Vectorized version.
    """
    n = len(b)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if abs(U[i, i]) < 1e-15:
             raise ValueError(f"Zero pivot encountered at index {i} during back substitution.")
        x[i] = (b[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
    return x

def gauss_no_pivoting(A_in, b_in):
    """
    Gaussian elimination without pivoting.
    Vectorized version.
    """
    A = A_in.copy().astype(float)
    b = b_in.copy().astype(float)
    n = len(b)
    n_vars = A.shape[1]
    limit = min(n, n_vars)

    for k in range(limit):
        pivot = A[k, k]
        if abs(pivot) < 1e-15:
            raise ValueError(f"Zero pivot encountered at row {k}. Naive Gauss fails.")

        # Vectorized elimination
        factors = A[k + 1:, k] / pivot
        A[k + 1:, k:] -= np.outer(factors, A[k, k:])
        b[k + 1:] -= factors * b[k]

    return utris(A[:n_vars, :n_vars], b[:n_vars])

def gauss_partial_pivoting(A_in, b_in):
    """
    Gaussian elimination with partial pivoting.
    Vectorized version.
    """
    A = A_in.copy().astype(float)
    b = b_in.copy().astype(float)
    n = len(b)
    n_vars = A.shape[1]
    limit = min(n, n_vars)

    for k in range(limit):
        # Pivot selection
        pivot_index = k + np.argmax(np.abs(A[k:, k]))
        
        if abs(A[pivot_index, k]) < 1e-15:
            raise ValueError(f"Singular matrix encountered at column {k}.")

        # Swap rows
        if pivot_index != k:
            A[[k, pivot_index]] = A[[pivot_index, k]]
            b[[k, pivot_index]] = b[[pivot_index, k]]

        # Vectorized elimination
        factors = A[k + 1:, k] / A[k, k]
        A[k + 1:, k:] -= np.outer(factors, A[k, k:])
        b[k + 1:] -= factors * b[k]

    return utris(A[:n_vars, :n_vars], b[:n_vars])

def solve_sparse(A, b):
    """
    Solves Ax = b using scipy.sparse.spsolve.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import spsolve
    A_sparse = csr_matrix(A)
    return spsolve(A_sparse, b)
