import os
import torch
import numpy as np
import trimesh
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from src.eigen_decomp import compute_laplacian_eigenmodes

class EigenMeshDataset(Dataset):
    """
    A Dataset for loading 3D meshes and their precomputed Laplacian eigenmodes.
    """
    def __init__(self, data_dir="data/eigen_mesh", cache_dir="data/cache", k=64, num_points=1024):
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.k = k
        self.num_points = num_points
        self.mesh_files = []
        
        # Ensure directories exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
            print(f"Created directory {self.data_dir}. Please place .obj files here.")
            
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Gather all .obj files
        for f in os.listdir(self.data_dir):
            if f.endswith('.obj'):
                self.mesh_files.append(os.path.join(self.data_dir, f))
                
        self.mesh_files.sort()
        
    def prepare_cache(self):
        """
        Precomputes and caches the Laplacian eigenmodes for all meshes in the dataset.
        """
        if not self.mesh_files:
            print(f"No .obj files found in {self.data_dir}")
            return
            
        print(f"Checking and preparing cache for {len(self.mesh_files)} meshes...")
        
        # Use tqdm for progress tracking
        pbar = tqdm(self.mesh_files, desc="Precomputing Eigenmodes")
        success_count = 0
        skip_count = 0
        error_count = 0
        
        for mesh_path in pbar:
            mesh_name = os.path.basename(mesh_path)
            cache_name = mesh_name.replace('.obj', f'_eigen_{self.k}.npz')
            cache_path = os.path.join(self.cache_dir, cache_name)
            
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
                vals, vecs = compute_laplacian_eigenmodes(vertices, faces, k=self.k)
                
                # Save to cache
                np.savez(cache_path, eigenvals=vals, eigenvecs=vecs)
                success_count += 1
                
            except Exception as e:
                error_count += 1
                tqdm.write(f"Error processing {mesh_name}: {e}")
                
        print(f"Preparation complete. Success: {success_count}, Skipped: {skip_count}, Errors: {error_count}")

    def __len__(self):
        return len(self.mesh_files)
        
    def __getitem__(self, idx):
        mesh_path = self.mesh_files[idx]
        mesh_name = os.path.basename(mesh_path)
        cache_name = mesh_name.replace('.obj', f'_eigen_{self.k}.npz')
        cache_path = os.path.join(self.cache_dir, cache_name)
        
        # If cache is missing, raise an error instead of computing on the fly
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Cache missing for {mesh_name}. Please run precompute_eigenmodes.py first.")
        
        # Load from cache
        mesh = trimesh.load(mesh_path, force='mesh')
        vertices = mesh.vertices
        data = np.load(cache_path)
        vals = data['eigenvals']
        vecs = data['eigenvecs']
            
        # Normalize vertices to [-1, 1] for network input
        v_min = vertices.min(axis=0)
        v_max = vertices.max(axis=0)
        center = (v_max + v_min) / 2
        scale = np.max(v_max - v_min) / 2
        if scale == 0:
            scale = 1.0
        vertices_norm = (vertices - center) / scale
        
        # Create PyTorch Geometric Data object
        pos = torch.tensor(vertices_norm, dtype=torch.float32)
        # For PointNet, x is often just the coordinates themselves if no other features exist,
        # or we can use normal vectors. We'll just use pos.
        x = pos.clone() 
        
        y_vecs = torch.tensor(vecs, dtype=torch.float32) # [N, k]
        
        # Subsample points for training efficiency if mesh is too large (optional)
        # Here we just use all points, but you might want to randomly sample N points
        # in a real scenario.
        if self.num_points is not None and len(pos) > self.num_points:
            # Randomly sample points
            indices = np.random.choice(len(pos), self.num_points, replace=False)
            pos = pos[indices]
            x = x[indices]
            y_vecs = y_vecs[indices]
            
        data_obj = Data(x=x, pos=pos, y=y_vecs)
        data_obj.mesh_path = mesh_path
            
        return data_obj

if __name__ == "__main__":
    # Setup paths relative to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data", "coarse_eigen_mesh")
    cache_dir = os.path.join(project_root, "data", "cache")
    
    # Initialize dataset
    dataset = EigenMeshDataset(data_dir=data_dir, cache_dir=cache_dir, k=64)
    
    # Run the cache preparation step
    dataset.prepare_cache()
    
    # Test the DataLoader
    if len(dataset) > 0:
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        print(f"\nSuccessfully loaded {len(dataset)} items into DataLoader.")
        
        for batch in dataloader:
            print(f"Batch item loaded: {batch['mesh_path'][0]}")
            print(f"Eigenvalues shape: {batch['eigenvals'].shape}")
            print(f"Eigenvectors shape: {batch['eigenvecs'].shape}")
            break
