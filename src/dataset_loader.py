import os

import matplotlib.pyplot as plt
import meshio
import numpy as np
import scipy.io.wavfile as wavfile
import torch
import torchaudio
import trimesh
import torchvision.transforms as T
from ocnn.octree import Octree, Points, merge_octrees, merge_points
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from config.config import cfg



"""

这是当前 [VVImpactDataset.__getitem__] 返回的一个“单个 mesh 样本”的字典。  
也就是说，`raw_data` 不是一个 batch，而是：

- 一个物体 / 一个 `.msh` 体网格
- 以及这个物体上所有敲击点的全部信息

下面我按字段解释。

**整体约定**
- 设 `K` = 当前 mesh 上的敲击点数量
- 设 `V` = 当前 mesh 的顶点数
- 设 `E` = 当前 `.msh` 中四面体单元数
- 设 `H, W` = 频谱图 PNG 的高宽
- 你当前真实样本里大致是：
  - `K = 200`
  - `V = 2380`
  - `E = 7271`
  - `mel_spectrogram` 约为 `(200, 257, 250)`

**字段说明**
- `mel_spectrogram`
  - 形状：`[K, H, W]`，你当前大致是 `[200, 257, 250]`
  - 含义：当前 mesh 的所有 impact 对应的频谱图张量
  - 来源：逐个读取 `impact_specs/.../audio_xxx.png`，转灰度后 `ToTensor()`，再去掉通道维 [load_spec](file:///Users/bobo/Codes/vv-impact/src/dataset_loader.py#L83-L87) 和 [__getitem__](file:///Users/bobo/Codes/vv-impact/src/dataset_loader.py#L97-L110)
  - 说明：这里其实不是从 waveform 现算 mel，而是直接读已经生成好的频谱图 PNG

- `impact_image`
  - 形状：`[K, 3, 224, 224]`
  - 含义：同一批 impact 对应的可视化预览图
  - 来源：同样来自 `impact_specs/.../audio_xxx.png`，但转成 RGB 后再做 `Resize(224,224) + ToTensor()` [dataset_loader.py:L19-L23](file:///Users/bobo/Codes/vv-impact/src/dataset_loader.py#L19-L23) [load_spec](file:///Users/bobo/Codes/vv-impact/src/dataset_loader.py#L83-L87)
  - 用途：主要给 viewer 展示，不是训练主输入

- `waveform`
  - 形状：`[K, Lmax]`
  - 含义：当前 mesh 所有 impact 的音频波形，已做 padding
  - 来源：逐个读取 `impact_audio/.../audio_xxx.wav`，转 float，必要时重采样到 `sample_rate`，然后 `pad_sequence` 拼成同长度 [load_waveform](file:///Users/bobo/Codes/vv-impact/src/dataset_loader.py#L89-L102) [__getitem__](file:///Users/bobo/Codes/vv-impact/src/dataset_loader.py#L97-L112)
  - 说明：`Lmax` 是这个 mesh 内所有 impact 里最长的音频长度；你现在样本里大致是 `[200, 16000]`

- `waveform_length`
  - 形状：`[K]`
  - 含义：每条 waveform 的真实长度
  - 来源：在 padding 前统计每条波形长度 [dataset_loader.py:L111-L112](file:///Users/bobo/Codes/vv-impact/src/dataset_loader.py#L111-L112)
  - 用途：播放音频或做时域模型时裁掉 padding

- `sample_rate`
  - 形状：标量 `int`
  - 含义：音频采样率
  - 来源：数据集初始化参数 `sample_rate`，默认 16000 [dataset_loader.py:L15-L18](file:///Users/bobo/Codes/vv-impact/src/dataset_loader.py#L15-L18)
  - 用途：播放音频、时域处理

- `mesh_vertices`
  - 形状：`[V, 3]`
  - 含义：当前 `.msh` 体网格的全部顶点坐标
  - 来源：用 `meshio.read(msh_path)` 读取 `.msh` 后取 `msh.points` [load_mesh](file:///Users/bobo/Codes/vv-impact/src/dataset_loader.py#L69-L81)
  - 用途：这是 [pipeline.py](file:///Users/bobo/Codes/vv-impact/src/pipeline.py#L112-L140) 真正喂给 PointNet++ 的几何输入

- `mesh_tetra`
  - 形状：`[E, 4]`
  - 含义：体网格的四面体单元索引
  - 来源：从 `.msh` 的 `cells_dict["tetra"]` 或 `cells` 中取出 `tetra` [load_mesh](file:///Users/bobo/Codes/vv-impact/src/dataset_loader.py#L72-L79)
  - 用途：主要给 Polyscope 可视化体网格；当前训练没直接用

- `mesh`
  - 结构：`{"vertices": mesh_vertices, "tetra": mesh_tetra}`
  - 含义：把 mesh 再封装成一个字典
  - 来源：在 `__getitem__` 里直接组装 [dataset_loader.py:L113-L115](file:///Users/bobo/Codes/vv-impact/src/dataset_loader.py#L113-L115)
  - 用途：便于下游统一传递 mesh 对象

- `impact_point`
  - 形状：`[K, 3]`
  - 含义：每个 impact 对应的 3D 敲击点坐标
  - 来源：用 `mesh["vertices"][impact_vertex_index]` 直接索引得到 [dataset_loader.py:L104-L106](file:///Users/bobo/Codes/vv-impact/src/dataset_loader.py#L104-L106)
  - 用途：viewer 里显示红色敲击点；也可做几何监督

- `impact_vertex_index`
  - 形状：`[K]`
  - 含义：当前 mesh 上所有敲击顶点的索引
  - 来源：从文件名 `audio_<vertex_id>.png/.wav` 里解析出顶点编号 [dataset_loader.py:L43-L50](file:///Users/bobo/Codes/vv-impact/src/dataset_loader.py#L43-L50) ，然后在 `__getitem__` 里组装成张量 [dataset_loader.py:L104](file:///Users/bobo/Codes/vv-impact/src/dataset_loader.py#L104)
  - 用途：这是 [pipeline.py](file:///Users/bobo/Codes/vv-impact/src/pipeline.py#L115-L118) 构造 `hit_indices` 的关键字段

- `num_impacts`
  - 形状：标量张量 `[]`
  - 含义：当前 mesh 上有多少个 impact
  - 来源：`impact_vertex_index.numel()` [dataset_loader.py:L117](file:///Users/bobo/Codes/vv-impact/src/dataset_loader.py#L117)
  - 用途：batch 内把每个 mesh 的 global feature 重复到所有 impact 上时会用到 [pipeline.py:L113-L132](file:///Users/bobo/Codes/vv-impact/src/pipeline.py#L113-L132)

- `mesh_path`
  - 形状：字符串
  - 含义：当前 mesh 文件路径
  - 现在实际内容：就是 `.msh` 路径
  - 来源：`sample["msh_path"]` [dataset_loader.py:L118-L119](file:///Users/bobo/Codes/vv-impact/src/dataset_loader.py#L118-L119)
  - 说明：这里名字叫 `mesh_path`，但现在指向的是体网格 `.msh`

- `msh_path`
  - 形状：字符串
  - 含义：当前 `.msh` 文件路径
  - 来源：扫描 `vv++test/msh/<group>/<obj_id>.obj_.msh` 时记录 [dataset_loader.py:L37-L57](file:///Users/bobo/Codes/vv-impact/src/dataset_loader.py#L37-L57)
  - 说明：和 `mesh_path` 当前是重复信息

- `obj_id`
  - 形状：字符串
  - 含义：当前物体 ID
  - 来源：目录名，比如 `21002` [dataset_loader.py:L37-L57](file:///Users/bobo/Codes/vv-impact/src/dataset_loader.py#L37-L57)

- `group`
  - 形状：字符串
  - 含义：上层分组目录，比如 `21`
  - 来源：扫描 `impact_specs/<group>/<obj_id>` 时得到 [dataset_loader.py:L32-L38](file:///Users/bobo/Codes/vv-impact/src/dataset_loader.py#L32-L38)

- `vertex_id`
  - 形状：`[K]`
  - 含义：本质上和 `impact_vertex_index` 一样，是同一组顶点编号
  - 来源：`impact_vertex_index.clone()` [dataset_loader.py:L120-L123](file:///Users/bobo/Codes/vv-impact/src/dataset_loader.py#L120-L123)
  - 说明：这是保留一份更贴近原始命名的副本

- `impact_spec_path`
  - 形状：长度为 `K` 的字符串列表
  - 含义：每个 impact 对应的频谱图路径
  - 来源：扫描 `impact_specs` 时保存 [dataset_loader.py:L47-L50](file:///Users/bobo/Codes/vv-impact/src/dataset_loader.py#L47-L50) ，在 `__getitem__` 里收集 [dataset_loader.py:L95-L103](file:///Users/bobo/Codes/vv-impact/src/dataset_loader.py#L95-L103)
  - 用途：调试、追溯样本来源

- `impact_audio_path`
  - 形状：长度为 `K` 的字符串列表
  - 含义：每个 impact 对应的 wav 路径
  - 来源：扫描 `impact_audio` 时构造 [dataset_loader.py:L46-L50](file:///Users/bobo/Codes/vv-impact/src/dataset_loader.py#L46-L50) ，在 `__getitem__` 里收集 [dataset_loader.py:L95-L103](file:///Users/bobo/Codes/vv-impact/src/dataset_loader.py#L95-L103)
  - 用途：viewer 播放声音、调试音频来源

**这些数据是怎么组织出来的**
- 第一步：扫描目录，构建“一个 mesh 对应一组 impacts”的样本列表 [VVImpactDataset.__init__](file:///Users/bobo/Codes/vv-impact/src/dataset_loader.py#L14-L57)
  - `impact_specs/<group>/<obj_id>/audio_xxx.png`
  - `impact_audio/<group>/<obj_id>/audio_xxx.wav`
  - `msh/<group>/<obj_id>.obj_.msh`
- 第二步：读取 `.msh` 得到体网格 [load_mesh](file:///Users/bobo/Codes/vv-impact/src/dataset_loader.py#L69-L81)
- 第三步：对该 mesh 的每个 impact：
  - 读 PNG 频谱图 [load_spec](file:///Users/bobo/Codes/vv-impact/src/dataset_loader.py#L83-L87)
  - 读 WAV 波形 [load_waveform](file:///Users/bobo/Codes/vv-impact/src/dataset_loader.py#L89-L102)
- 第四步：在 [__getitem__](file:///Users/bobo/Codes/vv-impact/src/dataset_loader.py#L104-L124) 里把这些 impact 信息聚合成一个 mesh 级样本

**和 pipeline 的关系**
当前 [forward](file:///Users/bobo/Codes/vv-impact/src/pipeline.py#L108-L145) 真正用到的主要是：
- `mesh_vertices`
- `impact_vertex_index`
- `num_impacts`
- `mel_spectrogram`

其中：
- `mesh_vertices` → 几何输入
- `impact_vertex_index` → 敲击点位置
- `mel_spectrogram` → 构造监督目标 `targets`
- `num_impacts` → 把每个 mesh 的全局特征扩展到所有 impact

而这些字段：
- `impact_image`
- `waveform`
- `waveform_length`
- `mesh_tetra`
- `impact_point`
- `impact_spec_path`
- `impact_audio_path`

主要是给 [interactive_viewer.py] 做验证性可视化和音频播放用的。

"""


class VVImpactDataset(Dataset):
    def __init__(self, data_dir=None, sample_rate=16000, transform_image=None, train_only=False):
        self.data_dir = self.resolve_data_dir(data_dir or cfg.DATA_DIR)
        self.sample_rate = sample_rate
        self.train_only = train_only
        self.preview_transform = transform_image or T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
        ])
        self.spec_transform = T.ToTensor()
        self.samples = []
        self.mesh_cache = {}
        self.remesh_cache = {}
        self.interp_cache = {}
        self.octree_cache = {}
        self.resampler_cache = {}

        specs_dir = os.path.join(self.data_dir, "impact_specs")
        audio_dir = os.path.join(self.data_dir, "impact_audio")
        msh_dir = os.path.join(self.data_dir, "msh")
        remesh_dir = os.path.join(self.data_dir, "remesh")
        if not os.path.isdir(specs_dir) or not os.path.isdir(audio_dir) or not os.path.isdir(msh_dir) or not os.path.isdir(remesh_dir):
            return

        for group in sorted(os.listdir(specs_dir)):
            group_specs_dir = os.path.join(specs_dir, group)
            group_audio_dir = os.path.join(audio_dir, group)
            group_msh_dir = os.path.join(msh_dir, group)
            group_remesh_dir = os.path.join(remesh_dir, group)
            if not os.path.isdir(group_specs_dir) or not os.path.isdir(group_audio_dir) or not os.path.isdir(group_msh_dir) or not os.path.isdir(group_remesh_dir):
                continue
            for obj_id in sorted(os.listdir(group_specs_dir)):
                obj_specs_dir = os.path.join(group_specs_dir, obj_id)
                obj_audio_dir = os.path.join(group_audio_dir, obj_id)
                msh_path = os.path.join(group_msh_dir, f"{obj_id}.obj_.msh")
                remesh_path = os.path.join(group_remesh_dir, f"{obj_id}.obj")
                if not os.path.isdir(obj_specs_dir) or not os.path.isdir(obj_audio_dir) or not os.path.exists(msh_path) or not os.path.exists(remesh_path):
                    continue
                impacts = []
                for spec_name in sorted(os.listdir(obj_specs_dir)):
                    if not spec_name.startswith("audio_") or not spec_name.endswith(".png"):
                        continue
                    vertex_id = int(spec_name.split("_")[1].split(".")[0])
                    wav_path = os.path.join(obj_audio_dir, f"audio_{vertex_id}.wav")
                    if not os.path.exists(wav_path):
                        continue
                    impacts.append({
                        "vertex_id": vertex_id,
                        "spec_path": os.path.join(obj_specs_dir, spec_name),
                        "wav_path": wav_path,
                    })
                if impacts:
                    self.samples.append({
                        "group": group,
                        "obj_id": obj_id,
                        "msh_path": msh_path,
                        "remesh_path": remesh_path,
                        "samples": impacts,
                    })

    def resolve_data_dir(self, data_dir):
        candidates = [data_dir, os.path.join(data_dir, "vv++test")]
        for candidate in candidates:
            if candidate and os.path.isdir(os.path.join(candidate, "impact_specs")) and os.path.isdir(os.path.join(candidate, "msh")):
                return candidate
        return data_dir

    def __len__(self):
        return len(self.samples)

    def load_mesh(self, msh_path):
        mesh = self.mesh_cache.get(msh_path)
        if mesh is None:
            msh = meshio.read(msh_path)
            tetra = msh.cells_dict.get("tetra")
            if tetra is None:
                tetra = next((cell_block.data for cell_block in msh.cells if cell_block.type == "tetra"), [])
            mesh = {
                "vertices": torch.tensor(msh.points, dtype=torch.float32),
                "tetra": torch.tensor(tetra, dtype=torch.long),
            }
            self.mesh_cache[msh_path] = mesh
        return mesh

    def load_remesh(self, remesh_path):
        remesh = self.remesh_cache.get(remesh_path)
        if remesh is None:
            obj = trimesh.load(remesh_path, force="mesh", process=False)
            remesh = {
                "mesh": obj,
                "vertices": torch.tensor(obj.vertices, dtype=torch.float32),
                "faces": torch.tensor(obj.faces, dtype=torch.long),
                "normals": torch.tensor(obj.vertex_normals, dtype=torch.float32),
            }
            self.remesh_cache[remesh_path] = remesh
        return remesh

    def load_octree(self, remesh_path):
        octree_data = self.octree_cache.get(remesh_path)
        if octree_data is None:
            remesh = self.load_remesh(remesh_path)
            points = Points(remesh["vertices"], remesh["normals"])
            octree = Octree(depth=cfg.OCTREE_DEPTH, full_depth=cfg.OCTREE_FULL_DEPTH)
            octree.build_octree(points)
            octree.construct_all_neigh()
            octree_data = {
                "points": points,
                "octree": octree,
            }
            self.octree_cache[remesh_path] = octree_data
        return octree_data

    def load_interpolation(self, msh_path, remesh_path, impact_point):
        cache_key = (msh_path, remesh_path)
        cached = self.interp_cache.get(cache_key)
        if cached is None:
            remesh = self.load_remesh(remesh_path)
            closest, _, face_id = trimesh.proximity.closest_point(remesh["mesh"], impact_point.numpy())
            face_index = remesh["faces"][torch.as_tensor(face_id, dtype=torch.long)]
            triangles = remesh["vertices"][face_index].numpy()
            barycentric = trimesh.triangles.points_to_barycentric(triangles, closest)
            cached = {
                "gnn_face_index": face_index,
                "gnn_barycentric": torch.tensor(barycentric, dtype=torch.float32),
            }
            self.interp_cache[cache_key] = cached
        return cached

    def load_spec(self, spec_path):
        spec_image = Image.open(spec_path).convert("L")
        spec_tensor = self.spec_transform(spec_image).squeeze(0)
        preview_tensor = self.preview_transform(spec_image.convert("RGB"))
        return spec_tensor, preview_tensor

    def load_waveform(self, wav_path):
        sample_rate, waveform_np = wavfile.read(wav_path)
        if np.issubdtype(waveform_np.dtype, np.integer):
            waveform_np = waveform_np.astype(np.float32) / np.iinfo(waveform_np.dtype).max
        elif waveform_np.dtype != np.float32:
            waveform_np = waveform_np.astype(np.float32)
        waveform = torch.from_numpy(waveform_np).float()
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=-1)
        if sample_rate != self.sample_rate:
            resampler = self.resampler_cache.get(sample_rate)
            if resampler is None:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
                self.resampler_cache[sample_rate] = resampler
            waveform = resampler(waveform.unsqueeze(0)).squeeze(0)
        return waveform

    def __getitem__(self, idx):
        sample = self.samples[idx]
        mesh = self.load_mesh(sample["msh_path"])
        remesh = self.load_remesh(sample["remesh_path"])
        octree_data = self.load_octree(sample["remesh_path"])
        impact_specs = []
        impact_vertex_index = []
        if not self.train_only:
            impact_images = []
            waveforms = []
            impact_spec_path = []
            impact_audio_path = []

        for impact in sample["samples"]:
            spec_tensor, preview_tensor = self.load_spec(impact["spec_path"])
            impact_specs.append(spec_tensor)
            impact_vertex_index.append(impact["vertex_id"])
            if not self.train_only:
                impact_images.append(preview_tensor)
                waveforms.append(self.load_waveform(impact["wav_path"]))
                impact_spec_path.append(impact["spec_path"])
                impact_audio_path.append(impact["wav_path"])

        impact_vertex_index = torch.tensor(impact_vertex_index, dtype=torch.long)
        impact_point = mesh["vertices"][impact_vertex_index]
        interpolation = self.load_interpolation(sample["msh_path"], sample["remesh_path"], impact_point)
        mel_spectrogram = torch.stack(impact_specs)
        data = {
            "mel_spectrogram": mel_spectrogram,
            "mesh_vertices": mesh["vertices"],
            "mesh_tetra": mesh["tetra"],
            "mesh": {"vertices": mesh["vertices"], "tetra": mesh["tetra"]},
            "gnn_vertices": remesh["vertices"],
            "gnn_face_index": interpolation["gnn_face_index"],
            "gnn_barycentric": interpolation["gnn_barycentric"],
            "gnn_normals": remesh["normals"],
            "octree_points": octree_data["points"],
            "octree": octree_data["octree"],
            "impact_point": impact_point,
            "impact_vertex_index": impact_vertex_index,
            "num_impacts": torch.tensor(impact_vertex_index.numel(), dtype=torch.long),
            "mesh_path": sample["msh_path"],
            "msh_path": sample["msh_path"],
            "remesh_path": sample["remesh_path"],
            "obj_id": sample["obj_id"],
            "group": sample["group"],
            "vertex_id": impact_vertex_index.clone(),
        }
        if not self.train_only:
            data["impact_image"] = torch.stack(impact_images)
            data["waveform"] = pad_sequence(waveforms, batch_first=True)
            data["waveform_length"] = torch.tensor([wave.size(0) for wave in waveforms], dtype=torch.long)
            data["sample_rate"] = self.sample_rate
            data["impact_spec_path"] = impact_spec_path
            data["impact_audio_path"] = impact_audio_path
        return data


def collate_vvimpact_batch(batch):
    collated = {
        "mel_spectrogram": [item["mel_spectrogram"] for item in batch],
        "mesh_vertices": [item["mesh_vertices"] for item in batch],
        "mesh_tetra": [item["mesh_tetra"] for item in batch],
        "mesh": [item["mesh"] for item in batch],
        "gnn_vertices": [item["gnn_vertices"] for item in batch],
        "gnn_face_index": [item["gnn_face_index"] for item in batch],
        "gnn_barycentric": [item["gnn_barycentric"] for item in batch],
        "gnn_normals": [item["gnn_normals"] for item in batch],
        "octree_points": merge_points([item["octree_points"] for item in batch]),
        "octree": merge_octrees([item["octree"] for item in batch]),
        "impact_point": [item["impact_point"] for item in batch],
        "impact_vertex_index": [item["impact_vertex_index"] for item in batch],
        "num_impacts": torch.stack([item["num_impacts"] for item in batch]),
        "mesh_path": [item["mesh_path"] for item in batch],
        "msh_path": [item["msh_path"] for item in batch],
        "remesh_path": [item["remesh_path"] for item in batch],
        "obj_id": [item["obj_id"] for item in batch],
        "group": [item["group"] for item in batch],
        "vertex_id": [item["vertex_id"] for item in batch],
    }
    collated["octree"].construct_all_neigh()
    if "impact_image" in batch[0]:
        collated["impact_image"] = [item["impact_image"] for item in batch]
        collated["waveform"] = [item["waveform"] for item in batch]
        collated["waveform_length"] = [item["waveform_length"] for item in batch]
        collated["sample_rate"] = batch[0]["sample_rate"]
        collated["impact_spec_path"] = [item["impact_spec_path"] for item in batch]
        collated["impact_audio_path"] = [item["impact_audio_path"] for item in batch]
    return collated


def visualize_sample(batch, save_path="sample_visualization.png"):
    spec = batch["mel_spectrogram"][0][0].numpy()
    impact_image = batch["impact_image"][0][0].permute(1, 2, 0).numpy()
    vertices = batch["mesh_vertices"][0].numpy()
    impact_points = batch["impact_point"][0].numpy()
    obj_id = batch["obj_id"][0]
    vertex_id = batch["vertex_id"][0][0].item()

    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(impact_image)
    ax1.set_title("Impact Spec Image")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 3, 2)
    cax = ax2.imshow(spec, aspect="auto", origin="lower", cmap="magma")
    ax2.set_title("Impact Spec Tensor")
    fig.colorbar(cax, ax=ax2)

    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    ax3.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=1, alpha=0.08)
    ax3.scatter(impact_points[:, 0], impact_points[:, 1], impact_points[:, 2], color="red", s=20)
    ax3.set_title(f"Mesh {obj_id} | Vertex {vertex_id}")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")

    plt.tight_layout()
    plt.savefig(save_path)


if __name__ == "__main__":
    dataset = VVImpactDataset()
    print(f"Total valid meshes found: {len(dataset)}")
    if len(dataset) > 0:
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_vvimpact_batch)
        batch = next(iter(dataloader))
        print("Batch keys:", batch.keys())
        print("Meshes per batch:", len(batch["mesh_vertices"]))
        print("Impacts per mesh:", batch["num_impacts"].tolist())
        visualize_sample(batch)
