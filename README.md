# VV-Impact: Spatially-Aware Sound Synthesis & Impact Localization

## 📖 项目简介 (Project Introduction)

**VV-Impact** 是一个基于 PyTorch 和 PyTorch Lightning 构建的 3D 深度学习项目。该项目致力于解决现有工作（如 VibraVerse 和 DiffSound）中存在的物理学谬误——即在同一个物体的不同位置（如波节与波腹、薄壁与厚底）敲击时，产生的声音没有物理差异的问题。

本项目通过引入**空间感知的声学建模 (Spatially-Aware Acoustic Modeling)**，深入理解物体局部的几何特征（模态形变）与敲击声音之间的物理因果映射。项目不仅支持基于 3D 几何和敲击点生成符合物理规律的真实声音频谱，还支持从真实敲击声音中反向定位敲击点的三维坐标。

## ✨ 核心特性 (Core Features)

- **物理真实的声学特征建模**：使用拉普拉斯特征模态 (Laplacian Eigenmodes) 表征物体的共振特性，支持离线预计算与缓存，极大加速训练。
- **先进的 3D 深度学习架构**：
  - **Encoder (特征提取)**：采用基于 PyTorch Geometric (PyG) 的定制化 `DeepPointNet2`，从 3D Mesh 提取全局和局部的几何形状嵌入 (Shape Embedding)。
  - **Decoder (特征解码)**：采用轻量级、支持二阶梯度的 **Tri-Plane (三平面)** 隐式神经表示网络。将 3D 空间压缩至三个 2D 特征平面 (XY, XZ, YZ)，支持对任意连续 3D 坐标的高效特征查询和声场插值。
- **稳健的工程管线**：基于 PyTorch Lightning 搭建训练 Pipeline，内置 ModelCheckpoint、EarlyStopping，代码结构清晰，易于扩展。

## 📂 项目结构 (Project Structure)

```text
vv-impact/
├── main.py                     # 模型训练主入口 (PyTorch Lightning)
├── precompute_eigenmodes.py    # 数据预处理：预计算并缓存 3D 模型的拉普拉斯特征模态
├── requirements.txt            # 项目 Python 依赖列表
├── Algorithm.md                # 核心算法理论、实验设计与 TODO 列表
├── config/
│   └── config.py               # 全局超参数配置文件 (学习率、特征维度、路径等)
├── src/
│   ├── pipeline.py             # PyTorch Lightning 模块 (MyPipeline)，定义前向传播与训练逻辑
│   ├── eigen_dataset.py        # 特征模态数据集加载与处理逻辑
│   ├── dataset_loader.py       # (备用) VVImpact 数据集加载器
│   ├── eigen_decomp.py         # 拉普拉斯矩阵特征值与特征向量分解算法
│   ├── interactive_viewer.py   # 3D 交互式可视化工具 (基于 Polyscope 等)
│   └── models/
│       ├── pointnet2.py        # DeepPointNet2 编码器实现
│       └── triplane.py         # 基于三平面的 ModulatedNetwork 解码器实现
├── external/
│   └── remeshing.py            # 外部依赖：网格重划分工具
├── data/                       # 数据集目录 (需用户自行准备)
│   ├── coarse_eigen_mesh/      # 原始 3D Mesh 数据
│   └── cache/                  # 预计算特征模态的缓存目录 (.npz)
└── checkpoints/                # 训练好的模型权重保存目录
```

## 🛠️ 环境依赖与安装 (Installation)

1. 克隆本项目：
   ```bash
   git clone <your-repo-url>
   cd vv-impact
   ```

2. 安装基础依赖：
   ```bash
   pip install -r requirements.txt
   ```
   *依赖库包括：`torch`, `torchaudio`, `torchvision`, `torch_geometric`, `pytorch_lightning`, `trimesh`, `meshio`, `polyscope` 等。*

3. **注意**：本项目高度依赖 `torch_geometric` 及其 C++ 扩展（如 `torch-cluster`, `torch-scatter`）。在 Mac (MPS) 上可能存在兼容性问题，目前代码已在 `main.py` 中默认 fallback 到 CPU 进行稳定训练。如果使用 CUDA 环境，请确保正确安装匹配版本的 PyG 扩展。

## 🚀 快速开始 (Quick Start)

### 1. 数据预处理 (预计算 Eigenmodes)
在开始训练之前，需要对 3D 网格数据集提取拉普拉斯特征模态。
请确保数据放置在 `config.DATA_DIR` 指向的路径下（默认 `../data/coarse_eigen_mesh`）。
```bash
python precompute_eigenmodes.py
```
预计算的结果将会缓存到 `cache` 目录下，以加速后续的 DataLoader 读取。

### 2. 模型训练
启动 PyTorch Lightning 训练管线：
```bash
python main.py
```
训练过程中的日志将保存在 `logs/lightning_logs/` 目录下，可通过 TensorBoard 查看：
```bash
tensorboard --logdir logs/lightning_logs
```

## 🧪 核心实验与研究任务 (Research Tasks)

本项目围绕以下几大核心实验展开（详见 `Algorithm.md`）：

1. **空间感知的音频合成 (Spatially-Aware Sound Synthesis)**：输入 3D Mesh + 敲击点坐标 + 材质参数，预测特定位置产生的梅尔频谱，证明模型能精准捕获敲击点（波节/波腹）带来的物理声音差异。
2. **声音引导的敲击点精准定位 (Sound-Guided Impact Localization)**：输入物体 3D Mesh 和一段真实敲击频谱，网络反向预测敲击点的 3D 坐标。
3. **位置条件下的 3D 形状重建 (Position-Aware Shape Reconstruction)**：结合敲击声音与敲击点坐标，消除声音推导几何的歧义性，高精度重建 3D 形状。
4. **稳健的材料与内部实心识别 (Material & Solid Identification)**：利用局部敲击产生的不完全激励频谱，判断物体材质及内部是否挖空。
5. **连续物体表面声场插值 (Continuous Acoustic Field Interpolation)**：证明模型拟合了底层的连续特征向量场，能够对物体表面未见过的全新敲击点进行平滑的物理推导。

## 📅 TODO (Future Work)

- [ ] 在 ObjectFolder 数据集上进行微调，在核心 Metric 上超越原始论文。
- [ ] 采购相同形状、不同材质的实物进行 sim2real 实验。
- [ ] 采购相同形状、不同空心程度的实物进行对比实验。
- [ ] 探索 Sim2Real 的域适应方法（如环境噪声数据增强，解决实际敲击中带噪的问题）。
