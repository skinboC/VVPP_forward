import os
import sys
import numpy as np
import trimesh
import igl
from scipy.sparse.linalg import eigsh
from tqdm import tqdm

def compute_laplacian_eigenmodes(points, elements, k=10):
    """
    Computes the first k non-trivial eigenmodes of the cotangent Laplacian for a given mesh.
    """
    if elements.shape[1] not in [3, 4]:
        raise ValueError("Elements must be triangles (3 vertices) or tetrahedra (4 vertices).")
        
    # 1. Compute cotangent Laplacian matrix (sparse)
    L = -igl.cotmatrix(points, elements)
    
    # 2. Compute mass matrix (sparse) for the generalized eigenvalue problem
    M = igl.massmatrix(points, elements, igl.MASSMATRIX_TYPE_VORONOI)
    
    # 3. Solve generalized eigenvalue problem: L * x = lambda * M * x
    try:
        vals, vecs = eigsh(L, k=k+1, M=M, sigma=-1e-8, which='LM')
    except Exception as e:
        print(f"Shift-invert eigsh failed: {e}. Falling back to standard SM mode...")
        vals, vecs = eigsh(L, k=k+1, M=M, which='SM')
    
    # 4. Sort the eigenvalues and eigenvectors
    idx = np.argsort(vals)
    vals = vals[idx]
    vecs = vecs[:, idx]
    
    # 5. Skip the first mode (constant mode with eigenvalue 0)
    return vals[1:k+1], vecs[:, 1:k+1]

def main():
    print("="*50)
    print("Precomputing Laplacian Eigenmodes (Standalone)")
    print("="*50)
    
    # Configure paths
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_root, "data", "coarse_eigen_mesh")
    cache_dir = os.path.join(project_root, "data", "cache")
    n_eigenmodes = 64
    
    print(f"Data Directory: {data_dir}")
    print(f"Cache Directory: {cache_dir}")
    print(f"Number of Eigenmodes (k): {n_eigenmodes}")
    
    # Ensure directories exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        print(f"Created directory {data_dir}. Please place .obj files here.")
        return
        
    os.makedirs(cache_dir, exist_ok=True)
    
    # Gather all .obj files
    mesh_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.obj')]
    mesh_files.sort()
    
    if not mesh_files:
        print(f"No .obj files found in {data_dir}")
        return
        
    print(f"Checking and preparing cache for {len(mesh_files)} meshes...")
    
    # Use tqdm for progress tracking
    pbar = tqdm(mesh_files, desc="Precomputing Eigenmodes")
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for mesh_path in pbar:
        mesh_name = os.path.basename(mesh_path)
        cache_name = mesh_name.replace('.obj', f'_eigen_{n_eigenmodes}.npz')
        cache_path = os.path.join(cache_dir, cache_name)
        
        pbar.set_postfix({"current": mesh_name})
        
        if os.path.exists(cache_path):
            skip_count += 1
            continue
            
        try:
            # Load mesh using trimesh
            mesh = trimesh.load(mesh_path, force='mesh')
            vertices = mesh.vertices
            faces = mesh.faces
            
            # Check if mesh is valid
            if len(vertices) == 0 or len(faces) == 0:
                raise ValueError(f"Mesh {mesh_name} has 0 vertices or faces.")
            
            # Compute eigenmodes
            vals, vecs = compute_laplacian_eigenmodes(vertices, faces, k=n_eigenmodes)
            
            # Save to cache
            np.savez(cache_path, eigenvals=vals, eigenvecs=vecs)
            success_count += 1
            
        except Exception as e:
            error_count += 1
            tqdm.write(f"Error processing {mesh_name}: {e}")
            
    print(f"Preparation complete. Success: {success_count}, Skipped: {skip_count}, Errors: {error_count}")
    
    print("="*50)
    print("Precomputation finished successfully.")
    print("You can now run `python main.py` for training.")

if __name__ == "__main__":
    main()