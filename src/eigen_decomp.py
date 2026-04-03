import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import igl

def compute_laplacian_eigenmodes(points, elements, k=10):
    """
    Computes the first k non-trivial eigenmodes of the cotangent Laplacian for a given mesh.
    
    Args:
        points: (N, 3) numpy array of vertex coordinates.
        elements: (M, 3) or (M, 4) numpy array of faces or tetrahedra.
        k: number of eigenmodes to return.
        
    Returns:
        vals: (k,) eigenvalues.
        vecs: (N, k) eigenvectors.
    """
    if elements.shape[1] not in [3, 4]:
        raise ValueError("Elements must be triangles (3 vertices) or tetrahedra (4 vertices).")
        
    # 1. Compute cotangent Laplacian matrix (sparse)
    # igl.cotmatrix returns a negative semi-definite matrix, so we negate it to make it positive semi-definite.
    L = -igl.cotmatrix(points, elements)
    
    # 2. Compute mass matrix (sparse) for the generalized eigenvalue problem
    # Using Voronoi areas (default) for the mass matrix.
    M = igl.massmatrix(points, elements, igl.MASSMATRIX_TYPE_VORONOI)
    
    # 3. Solve generalized eigenvalue problem: L * x = lambda * M * x
    # We need k+1 modes because the first one is the constant mode (eigenvalue ~ 0)
    # Using shift-invert mode (sigma=-1e-8) is highly recommended and much faster for finding smallest eigenvalues.
    try:
        vals, vecs = eigsh(L, k=k+1, M=M, sigma=-1e-8, which='LM')
    except Exception as e:
        print(f"Shift-invert eigsh failed: {e}. Falling back to standard SM mode...")
        vals, vecs = eigsh(L, k=k+1, M=M, which='SM')
    
    # 4. Sort the eigenvalues and eigenvectors (eigsh doesn't strictly guarantee order)
    idx = np.argsort(vals)
    vals = vals[idx]
    vecs = vecs[:, idx]
    
    # 5. Skip the first mode (constant mode with eigenvalue 0)
    return vals[1:k+1], vecs[:, 1:k+1]

