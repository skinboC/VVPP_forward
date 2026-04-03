import os
import sys
import numpy as np
import trimesh
import polyscope as ps
import polyscope.imgui as psim

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

class EigenViewer:
    def __init__(self, data_dir, cache_dir, n_eigenmodes=64):
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.n_eigenmodes = n_eigenmodes
        
        self.mesh_files = []
        self.obj_names = []
        self.current_obj_idx = 0
        
        self.current_mesh = None
        self.current_ps_mesh = None
        
        self.eigenvals = None
        self.eigenvecs = None
        self.current_eigenmode_idx = 0
        
        self._load_dataset_info()
        
        # Initialize polyscope
        ps.init()
        ps.set_program_name("Eigenmodes Viewer")
        ps.set_up_dir("y_up")
        
        # Register UI callback
        ps.set_user_callback(self.ui_callback)
        
        # Load initial object
        if self.obj_names:
            self.load_object(self.current_obj_idx)
            
    def _load_dataset_info(self):
        if not os.path.exists(self.data_dir):
            print(f"Data directory not found: {self.data_dir}")
            return
            
        for f in os.listdir(self.data_dir):
            if f.endswith('.obj'):
                self.mesh_files.append(os.path.join(self.data_dir, f))
                self.obj_names.append(os.path.basename(f))
                
        # Sort to keep consistent order
        zipped = sorted(zip(self.obj_names, self.mesh_files))
        if zipped:
            self.obj_names, self.mesh_files = zip(*zipped)
            self.obj_names = list(self.obj_names)
            self.mesh_files = list(self.mesh_files)

    def load_object(self, obj_idx):
        if obj_idx < 0 or obj_idx >= len(self.mesh_files):
            return
            
        ps.remove_all_structures()
        
        mesh_path = self.mesh_files[obj_idx]
        obj_name = self.obj_names[obj_idx]
        
        print(f"Loading object: {obj_name}")
        
        # 1. Load Mesh
        try:
            self.current_mesh = trimesh.load(mesh_path, force='mesh')
            self.current_ps_mesh = ps.register_surface_mesh(
                f"Mesh_{obj_name}", 
                self.current_mesh.vertices, 
                self.current_mesh.faces,
                color=[0.8, 0.8, 0.8],
                edge_width=1.0
            )
        except Exception as e:
            print(f"Failed to load mesh {obj_name}: {e}")
            self.current_ps_mesh = None
            return
            
        # 2. Load precomputed eigenmodes from cache
        # Note: we need to handle potential extension issues if obj_name already has .obj
        if obj_name.endswith('.obj'):
            base_name = obj_name[:-4]
        else:
            base_name = obj_name
            
        cache_name = f"{base_name}_eigen_{self.n_eigenmodes}.npz"
        cache_path = os.path.join(self.cache_dir, cache_name)
        
        # Debugging info
        print(f"Trying to load cache from: {cache_path}")
        
        if not os.path.exists(cache_path):
            print(f"Cache not found: {cache_path}. Please run precompute_eigenmodes.py first.")
            self.eigenvals = None
            self.eigenvecs = None
            return
            
        try:
            data = np.load(cache_path)
            self.eigenvals = data['eigenvals']
            self.eigenvecs = data['eigenvecs']
            
            # Check if vertex count matches
            n_vertices = self.current_mesh.vertices.shape[0]
            if self.eigenvecs.shape[0] != n_vertices:
                print(f"Warning: Vertex count mismatch! Mesh has {n_vertices} vertices, but cache has {self.eigenvecs.shape[0]}.")
                print("This usually happens if the mesh was modified after computing the cache. The visualization will fail.")
                self.eigenvals = None
                self.eigenvecs = None
                return
            
            # Ensure index is within bounds
            self.current_eigenmode_idx = min(self.current_eigenmode_idx, len(self.eigenvals) - 1)
            
            self.update_eigenmode_visualization()
            print("Eigenmodes loaded from cache.")
        except Exception as e:
            print(f"Failed to load cache {cache_name}: {e}")
            self.eigenvals = None
            self.eigenvecs = None

    def update_eigenmode_visualization(self):
        if self.eigenvecs is not None and self.current_ps_mesh is not None:
            mode_data = self.eigenvecs[:, self.current_eigenmode_idx]
            
            # Add scalar quantity to polyscope mesh
            self.current_ps_mesh.add_scalar_quantity(
                "Eigenmode", 
                mode_data, 
                defined_on='vertices', 
                cmap='coolwarm', 
                enabled=True
            )

    def ui_callback(self):
        psim.PushItemWidth(200)
        
        # 1. Object Selection
        psim.TextUnformatted("Object Selection:")
        if self.obj_names:
            changed, new_obj_idx = psim.Combo(
                "Mesh", 
                self.current_obj_idx, 
                self.obj_names
            )
            if changed:
                self.current_obj_idx = new_obj_idx
                self.load_object(self.current_obj_idx)
                
            # Previous / Next Object Buttons
            if psim.Button("< Prev Obj") and self.current_obj_idx > 0:
                self.current_obj_idx -= 1
                self.load_object(self.current_obj_idx)
            psim.SameLine()
            if psim.Button("Next Obj >") and self.current_obj_idx < len(self.obj_names) - 1:
                self.current_obj_idx += 1
                self.load_object(self.current_obj_idx)
        else:
            psim.TextUnformatted("No objects found in data directory.")
            
        psim.Separator()
        
        # 2. Laplacian Eigenmodes Visualization
        psim.TextUnformatted("Laplacian Eigenmodes:")
        if self.eigenvals is not None:
            num_modes = len(self.eigenvals)
            
            changed, new_idx = psim.SliderInt("Mode Index", self.current_eigenmode_idx, 0, num_modes - 1)
            if changed:
                self.current_eigenmode_idx = new_idx
                self.update_eigenmode_visualization()
                
            # Previous / Next Mode Buttons
            if psim.Button("< Prev Mode") and self.current_eigenmode_idx > 0:
                self.current_eigenmode_idx -= 1
                self.update_eigenmode_visualization()
            psim.SameLine()
            if psim.Button("Next Mode >") and self.current_eigenmode_idx < num_modes - 1:
                self.current_eigenmode_idx += 1
                self.update_eigenmode_visualization()
                
            psim.TextUnformatted(f"Current Mode: {self.current_eigenmode_idx}")
            psim.TextUnformatted(f"Eigenvalue: {self.eigenvals[self.current_eigenmode_idx]:.6f}")
        else:
            psim.TextUnformatted("Eigenmodes not available.")
            psim.TextColored((1.0, 0.0, 0.0, 1.0), "Please precompute cache first.")
            
        psim.PopItemWidth()

    def run(self):
        if not self.obj_names:
            print("No valid samples found. Exiting.")
            return
        ps.show()

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Target directories
    data_dir = os.path.join(project_root, "data", "eigen_mesh")
    cache_dir = os.path.join(project_root, "data", "cache")
    n_eigenmodes = 64
    
    print("="*50)
    print("Starting Eigenmodes Interactive Viewer")
    print(f"Data Dir: {data_dir}")
    print(f"Cache Dir: {cache_dir}")
    print("="*50)
    
    viewer = EigenViewer(data_dir=data_dir, cache_dir=cache_dir, n_eigenmodes=n_eigenmodes)
    viewer.run()
