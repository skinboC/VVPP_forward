import os
import sys
import multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import sounddevice as sd
import torch
from scipy.spatial import KDTree

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.dataset_loader import VVImpactDataset


def play_audio_process(waveform, sample_rate):
    sd.play(waveform, sample_rate)
    sd.wait()


class PolyscopeViewer:
    def __init__(self, dataset):
        self.dataset = dataset
        self.obj_ids = [sample["obj_id"] for sample in dataset.samples]
        self.current_obj_idx = 0
        self.current_impact_idx = 0
        self.current_sample = None
        self.audio_process = None
        self.current_ps_mesh = None
        self.current_ps_cloud = None
        ps.init()
        ps.set_program_name("VV-Impact Viewer")
        ps.set_up_dir("y_up")
        ps.set_user_callback(self.ui_callback)
        if self.obj_ids:
            self.load_object(0)

    def load_object(self, sample_idx):
        self.current_sample = self.dataset[sample_idx]
        self.current_impact_idx = 0
        ps.remove_all_structures()
        self.current_ps_mesh = ps.register_volume_mesh(
            f"Mesh_{self.current_sample['obj_id']}",
            self.current_sample["mesh_vertices"].numpy(),
            tets=self.current_sample["mesh_tetra"].numpy(),
            color=[0.8, 0.8, 0.8],
            edge_width=1.0,
        )
        self.current_ps_cloud = ps.register_point_cloud(
            "Impact_Points",
            self.current_sample["impact_point"].numpy(),
            radius=0.01,
            color=[1.0, 0.2, 0.2],
        )
        self.add_pca_coloring()
        self.highlight_selected_impact()

    def highlight_selected_impact(self):
        impact_point = self.current_sample["impact_point"][self.current_impact_idx].numpy()
        ps.register_point_cloud(
            "Selected_Impact",
            np.array([impact_point]),
            radius=0.02,
            color=[0.0, 1.0, 0.0],
        )

    def add_pca_coloring(self):
        features = self.current_sample["mel_spectrogram"].float().max(dim=-1).values
        n_components = min(3, features.shape[0], features.shape[1])
        centered = features - features.mean(dim=0, keepdim=True)
        _, _, basis = torch.pca_lowrank(features, q=n_components, center=True)
        colors = torch.matmul(centered, basis[:, :n_components]).cpu().numpy()
        min_vals = colors.min(axis=0)
        max_vals = colors.max(axis=0)
        denom = np.where(max_vals > min_vals, max_vals - min_vals, 1.0)
        colors = (colors - min_vals) / denom
        if colors.shape[1] < 3:
            colors = np.pad(colors, ((0, 0), (0, 3 - colors.shape[1])))
        self.current_ps_cloud.add_color_quantity("PCA_RGB", colors, enabled=True)

        mesh_vertices = self.current_sample["mesh_vertices"].numpy()
        impact_points = self.current_sample["impact_point"].numpy()
        neighbor_count = min(10, len(impact_points))
        distances, indices = KDTree(impact_points).query(mesh_vertices, k=neighbor_count)
        if neighbor_count == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        weights = 1.0 / (distances + 1e-8) ** 2
        weights /= weights.sum(axis=1, keepdims=True)
        interpolated_colors = (colors[indices] * weights[..., None]).sum(axis=1)
        self.current_ps_mesh.add_color_quantity(
            "Interpolated_PCA_RGB",
            interpolated_colors,
            defined_on="vertices",
            enabled=True,
        )

    def show_current_spec(self):
        spec = self.current_sample["mel_spectrogram"][self.current_impact_idx].numpy()
        vertex_id = self.current_sample["vertex_id"][self.current_impact_idx].item()
        plt.figure(figsize=(8, 4))
        plt.imshow(spec, aspect="auto", origin="lower", cmap="magma")
        plt.title(f"{self.current_sample['obj_id']} | Vertex {vertex_id}")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show(block=False)

    def play_audio_and_show_spec(self):
        waveform_length = int(self.current_sample["waveform_length"][self.current_impact_idx].item())
        waveform = self.current_sample["waveform"][self.current_impact_idx, :waveform_length].numpy()
        if self.audio_process and self.audio_process.is_alive():
            self.audio_process.terminate()
        self.audio_process = mp.Process(
            target=play_audio_process,
            args=(waveform, int(self.current_sample["sample_rate"])),
        )
        self.audio_process.start()
        self.show_current_spec()

    def ui_callback(self):
        psim.PushItemWidth(200)
        changed, new_idx = psim.Combo("Object", self.current_obj_idx, self.obj_ids)
        if changed:
            self.current_obj_idx = new_idx
            self.load_object(new_idx)

        max_impact_idx = max(0, int(self.current_sample["num_impacts"].item()) - 1)
        changed, new_impact_idx = psim.SliderInt("Impact Index", self.current_impact_idx, 0, max_impact_idx)
        if changed:
            self.current_impact_idx = new_impact_idx
            self.highlight_selected_impact()

        vertex_id = self.current_sample["vertex_id"][self.current_impact_idx].item()
        psim.TextUnformatted(f"Vertex ID: {vertex_id}")
        psim.TextUnformatted(f"Impacts: {int(self.current_sample['num_impacts'].item())}")
        if psim.Button("Play Audio & Show Impact Spec"):
            self.play_audio_and_show_spec()
        psim.PopItemWidth()

    def run(self):
        ps.show()
        if self.audio_process and self.audio_process.is_alive():
            self.audio_process.terminate()


if __name__ == "__main__":
    dataset = VVImpactDataset(train_only=False)
    if len(dataset) > 0:
        viewer = PolyscopeViewer(dataset)
        viewer.run()
