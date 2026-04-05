import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_max_pool

# 确保项目根目录在 sys.path 中
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from config.config import cfg
import src.models.pointnet2 as pointnet2


class FiLMResidualBlock(nn.Module):
    """
    FiLM 残差块：利用局部激振特征，有选择性地调制全局共振基底。
    遵循 L2G-FiLM-Net 阶段 B 的设计。
    """
    def __init__(self, hidden_dim, condition_dim):
        super().__init__()
        # 1. 参数投影层：预测缩放系数 (gamma) 和平移系数 (beta)
        self.gamma_layer = nn.Linear(condition_dim, hidden_dim)
        self.beta_layer = nn.Linear(condition_dim, hidden_dim)
        
        # 特征映射层
        self.res_layer = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.ReLU()

    def forward(self, main_feature, condition_feature):
        """
        main_feature: 主干特征 (最初为全局特征 E_global)
        condition_feature: 条件特征 (局部激振特征 E_hit)
        """
        # 计算缩放系数和平移系数
        gamma = self.gamma_layer(condition_feature)
        beta = self.beta_layer(condition_feature)
        
        # 2. 特征调制 (Feature-wise Modulation)
        # 用生成的系数对主干特征进行逐元素仿射变换
        modulated = gamma * main_feature + beta
        
        # 3. 非线性激活与残差连接
        output = self.activation(self.res_layer(modulated)) + main_feature
        return output


class L2G_FiLM_Decoder(nn.Module):
    """
    级联特征调制解码器 (Cascaded FiLM Decoder)
    包含 K 个 FiLM-ResBlocks 和最终的降维输出多层感知机。
    """
    def __init__(self, hidden_dim, output_dim, num_blocks=4):
        super().__init__()
        # 串联的 K 个 FiLM 残差块
        self.blocks = nn.ModuleList([
            FiLMResidualBlock(hidden_dim, hidden_dim) for _ in range(num_blocks)
        ])
        
        # 阶段 C: 降维与声学输出 (Acoustic Readout)
        # 两层 MLP，将隐藏层维度压降至目标声音特征维度 (如 64 维)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, global_feature, hit_feature):
        # 初始化解码器的主干输入为全局特征
        hidden = global_feature
        
        # 逐层进行特征调制
        for block in self.blocks:
            hidden = block(hidden, hit_feature)
            
        # 最终输出目标声音特征
        output = self.readout(hidden)
        return output


class MyPipeline(pl.LightningModule):
    """
    L2G-FiLM-Net 核心管道类。
    实现“局部特征调制全局特征”且“纯离散顶点、无插值、纯前向推理”的核心共识。
    """
    def __init__(self, learning_rate=None):
        super().__init__()
        self.learning_rate = learning_rate if learning_rate is not None else getattr(cfg, "LEARNING_RATE", 1e-3)
        self.hidden_dim = getattr(cfg, "HIDDEN_DIM", 256)
        self.output_dim = getattr(cfg, "OUTPUT_DIM", 64)

        # ====== 阶段 A: 几何声学基底提取 (GNN) ======
        # 使用 PointNet++ 将 3D 网格的物理空间结构映射到高维特征空间
        self.gnn = pointnet2.DeepPointNet2(in_channels=3, out_channels=self.hidden_dim)
        
        # ====== 阶段 B & C: 级联特征调制解码器 ======
        self.decoder = L2G_FiLM_Decoder(
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            num_blocks=4
        )

    def build_validation_report(self, targets, output, loss):
        gt = targets.detach().cpu()
        pred = output.detach().cpu()
        diff = (pred - gt).abs()
        sample_idx = int(diff.mean(dim=1).argmax().item())
        gt_sample = gt[sample_idx]
        pred_sample = pred[sample_idx]
        diff_sample = diff[sample_idx]
        sample_count = min(8, gt.size(0))
        gt_panel = gt[:sample_count]
        pred_panel = pred[:sample_count]
        mae = diff.mean().item()
        rmse = torch.sqrt(((pred - gt) ** 2).mean()).item()
        corr = torch.corrcoef(torch.stack([gt_sample, pred_sample]))[0, 1].item() if gt_sample.numel() > 1 else 0.0
        worst_dims = torch.topk(diff_sample, k=min(8, diff_sample.numel())).indices.tolist()

        fig = plt.figure(figsize=(16, 10), dpi=160)
        gs = fig.add_gridspec(3, 3, height_ratios=[1.1, 1.6, 1.6])

        ax_text = fig.add_subplot(gs[0, :])
        ax_text.axis("off")
        text = "\n".join([
            f"epoch={self.current_epoch}  val_loss={float(loss.item()):.6f}  samples={gt.size(0)}  dims={gt.size(1)}",
            f"global_mae={mae:.6f}  global_rmse={rmse:.6f}  sample_idx={sample_idx}  sample_corr={corr:.6f}",
            f"gt(mean/std/min/max)=({gt_sample.mean():.4f}, {gt_sample.std():.4f}, {gt_sample.min():.4f}, {gt_sample.max():.4f})",
            f"pred(mean/std/min/max)=({pred_sample.mean():.4f}, {pred_sample.std():.4f}, {pred_sample.min():.4f}, {pred_sample.max():.4f})",
            f"worst_dims={worst_dims}",
        ])
        ax_text.text(0.01, 0.98, text, va="top", ha="left", family="monospace", fontsize=11)

        ax_line = fig.add_subplot(gs[1, 0])
        ax_line.plot(gt_sample.numpy(), label="GT", linewidth=2)
        ax_line.plot(pred_sample.numpy(), label="Pred", linewidth=2)
        ax_line.set_title("GT vs Pred")
        ax_line.legend()

        ax_diff = fig.add_subplot(gs[1, 1])
        ax_diff.bar(range(diff_sample.numel()), diff_sample.numpy(), color="tab:red")
        ax_diff.set_title("Absolute Error")

        ax_scatter = fig.add_subplot(gs[1, 2])
        ax_scatter.scatter(gt_sample.numpy(), pred_sample.numpy(), s=18, alpha=0.8)
        lo = min(gt_sample.min().item(), pred_sample.min().item())
        hi = max(gt_sample.max().item(), pred_sample.max().item())
        ax_scatter.plot([lo, hi], [lo, hi], color="black", linewidth=1)
        ax_scatter.set_title("GT-Pred Scatter")
        ax_scatter.set_xlabel("GT")
        ax_scatter.set_ylabel("Pred")

        ax_gt = fig.add_subplot(gs[2, 0])
        im_gt = ax_gt.imshow(gt_panel.numpy(), aspect="auto", cmap="viridis")
        ax_gt.set_title("GT Heatmap")
        fig.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)

        ax_pred = fig.add_subplot(gs[2, 1])
        im_pred = ax_pred.imshow(pred_panel.numpy(), aspect="auto", cmap="viridis")
        ax_pred.set_title("Pred Heatmap")
        fig.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)

        ax_err = fig.add_subplot(gs[2, 2])
        im_err = ax_err.imshow((pred_panel - gt_panel).abs().numpy(), aspect="auto", cmap="magma")
        ax_err.set_title("AbsDiff Heatmap")
        fig.colorbar(im_err, ax=ax_err, fraction=0.046, pad=0.04)

        fig.tight_layout()
        fig.canvas.draw()
        image = torch.from_numpy(np.asarray(fig.canvas.buffer_rgba()).copy()[..., :3]).permute(2, 0, 1)
        plt.close(fig)
        return image

    def forward(self, batch_data):
        """
        前向推理函数：将输入 3D 网格和敲击点转化为声音特征向量
        """
        positions = [pos.to(self.device) for pos in batch_data["mesh_vertices"]]
        offsets = torch.tensor([0] + [pos.size(0) for pos in positions[:-1]], dtype=torch.long, device=self.device).cumsum(0)
        num_impacts = batch_data["num_impacts"].to(self.device)
        hit_indices = torch.cat([
            impact_vertex_index.to(self.device).long() + offset
            for impact_vertex_index, offset in zip(batch_data["impact_vertex_index"], offsets)
        ])  # hit_indices: [Q_total]
        #targets = torch.cat([mel.float().to(self.device).mean(dim=-1) for mel in batch_data["mel_spectrogram"]], dim=0)
        targets = torch.cat([mel.float().to(self.device).max(dim=-1).values for mel in batch_data["mel_spectrogram"]], dim=0)
        targets = F.adaptive_avg_pool1d(targets.unsqueeze(1), self.output_dim).squeeze(1)  # 维度转换：将可能200+的mel谱图压缩至 self.output_dim 维
        batch_data = Batch.from_data_list([Data(pos=pos, x=pos) for pos in positions])
        batch_data.hit_idx = hit_indices
        batch_data.y = targets

        # 1. 逐顶点特征计算 (E_vertices)
        vertex_features = self.gnn(batch_data)   # vertex_features: [V_total, hidden_dim]
        
        # 2. 提取全局共振基底 (E_global)
        # 对顶点维度进行全局最大池化，提取物体拓扑和固有频率字典
        global_features = global_max_pool(vertex_features, batch_data.batch)
        global_features = torch.repeat_interleave(global_features, num_impacts, dim=0)   # global_features: [Q_total, hidden_dim]
        
        # 3. 离散提取局部激振特征 (E_hit)
        # 以 O(1) 的时间复杂度进行数组切片，获取专属特征，摒弃复杂的空间插值
        hit_features = vertex_features[hit_indices]
        
        # 4. 特征调制与声学读出
        # 局部激振特征作为条件，调制全局共振基底，并回归至目标 64 维声音特征
        output = self.decoder(global_features, hit_features)
        loss = F.smooth_l1_loss(output, targets)
        return loss, output

    def training_step(self, batch, batch_idx):
        loss, _ = self(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch["num_impacts"].sum().item())
        return loss

    def validation_step(self, batch, batch_idx):
        loss, output = self(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch["num_impacts"].sum().item())
        if batch_idx == 0 and getattr(self.logger, "experiment", None) is not None:
            targets = torch.cat([mel.float().to(self.device).max(dim=-1).values for mel in batch["mel_spectrogram"]], dim=0)
            targets = F.adaptive_avg_pool1d(targets.unsqueeze(1), self.output_dim).squeeze(1)
            report = self.build_validation_report(targets, output, loss)
            self.logger.experiment.add_image("val/gt_pred_absdiff", report, self.current_epoch)
        return loss

    def test_step(self, batch, batch_idx):
        loss, _ = self(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch["num_impacts"].sum().item())
        return loss

    def configure_optimizers(self):
        """
        配置优化器和学习率调度器
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=getattr(cfg, "WEIGHT_DECAY", 0.0),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
