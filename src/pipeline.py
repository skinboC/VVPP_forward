import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import ocnn
import pytorch_lightning as pl
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.nn.functional as F

# 确保项目根目录在 sys.path 中
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from config.config import cfg
import src.models.ocnn_model_ref.my_ocnn as ocnn_unet


class AcousticFieldHead(nn.Module):
    def __init__(self, hidden_dim, output_dim, pe_frequencies=6, attention_heads=4, num_peaks=12, use_modal_bins=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.pe_frequencies = pe_frequencies
        self.num_peaks = num_peaks
        self.use_modal_bins = use_modal_bins
        self.register_buffer(
            "frequency_bands",
            (2.0 ** torch.arange(pe_frequencies, dtype=torch.float32)) * torch.pi,
            persistent=False,
        )
        position_dim = 3 + 3 * 2 * pe_frequencies
        local_input_dim = hidden_dim + position_dim
        global_input_dim = hidden_dim * 2
        self.local_encoder = nn.Sequential(
            nn.Linear(local_input_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(global_input_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.cross_attention = nn.MultiheadAttention(hidden_dim, attention_heads, batch_first=True)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        
        # 1. 直接预测方案 (Direct)
        self.predictor_direct = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
        # 2. 匈牙利匹配方案 (Bipartite) - 预测 12 个峰 (freq, amp, width)
        self.predictor_bipartite = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_peaks * 3),
        )
        
        # 3. 锚点分箱方案 (Anchor) - 预测 12 个 bin (base_val, offset, amp, width)
        self.predictor_anchor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_peaks * 4),
        )
        
        # 4. 模态锚点方案 (Modal Anchor) - 在 64 个区间预测 (prob, offset, amp)
        self.num_modal_bins = 64
        self.predictor_modal_anchor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_modal_bins * 3),
        )

    def positional_encoding(self, xyz):
        scaled_xyz = xyz.unsqueeze(-1) * self.frequency_bands.view(1, 1, -1)
        pe = torch.cat([scaled_xyz.sin(), scaled_xyz.cos()], dim=-1).reshape(xyz.size(0), -1)
        return torch.cat([xyz, pe], dim=-1)

    def render_spectrum(self, peaks, base_vals=None):
        """将预测的参数化峰 (freq, amp, width) 渲染回 64 维的连续频谱，用于兼容现有可视化并计算辅助 Loss"""
        B = peaks.size(0)
        x = torch.linspace(0, 1, self.output_dim, device=peaks.device).view(1, 1, self.output_dim)
        freqs = peaks[..., 0].unsqueeze(-1)
        amps = peaks[..., 1].unsqueeze(-1)
        widths = peaks[..., 2].unsqueeze(-1)
        
        gaussians = amps * torch.exp(-0.5 * ((x - freqs) / widths) ** 2)
        spectrum = gaussians.sum(dim=1)
        
        if base_vals is not None:
            # base_vals: [B, num_peaks]
            # 将输出空间等分为 num_peaks 个区域，将对应的 base_val 加到相应的区域上
            bin_size = self.output_dim // self.num_peaks
            base_spectrum = torch.zeros_like(spectrum)
            for i in range(self.num_peaks):
                start_idx = i * bin_size
                # 最后一个区域包含剩余的部分
                end_idx = (i + 1) * bin_size if i < self.num_peaks - 1 else self.output_dim
                base_spectrum[:, start_idx:end_idx] = base_vals[:, i:i+1]
            spectrum = spectrum + base_spectrum
            
        return spectrum

    def forward_direct(self, features):
        return self.predictor_direct(features), None

    def forward_bipartite(self, features):
        out = self.predictor_bipartite(features).view(-1, self.num_peaks, 3)
        freqs = torch.sigmoid(out[..., 0])  # 限制频率在 [0, 1] 之间
        # 加上初始偏置，避免初始 amp 接近 0 导致梯度消失。真实数据一般最大值在 1 左右
        amps = F.softplus(out[..., 1] + 1.0) 
        # 给宽度一个合理的初始值，比如占据整个频段的 1/20
        widths = F.softplus(out[..., 2] - 2.0) + 0.05 
        peaks = torch.stack([freqs, amps, widths], dim=-1)
        rendered = self.render_spectrum(peaks)
        return rendered, peaks

    def forward_anchor(self, features):
        out = self.predictor_anchor(features).view(-1, self.num_peaks, 4)
        # 给 base_val 一个较小的初始偏置，避免负数域梯度消失
        base_val = F.softplus(out[..., 0] - 1.0)  
        offset = torch.sigmoid(out[..., 1])  # 限制在 [0, 1] 表示在 bin 内部的相对偏移
        # 加上初始偏置
        amps = F.softplus(out[..., 2] + 1.0)      
        # 给宽度一个合理的初始值，大概覆盖一个 bin
        widths = F.softplus(out[..., 3] - 2.0) + (1.0 / self.num_peaks) 
        
        # 计算绝对频率
        bin_centers = torch.arange(self.num_peaks, device=features.device).float() / self.num_peaks
        freqs = bin_centers.unsqueeze(0) + (offset - 0.5) / self.num_peaks
        freqs = freqs.clamp(0, 1)
        
        peaks = torch.stack([freqs, amps, widths], dim=-1)
        rendered = self.render_spectrum(peaks, base_val)
        anchors = torch.stack([base_val, freqs, amps, widths], dim=-1)
        return rendered, anchors

    def render_modal_spectrum(self, prob, freqs, amps):
        """将预测的模态参数渲染回连续频谱。使用极小的固定宽度以保证可导性，并结合概率 prob 控制激活"""
        x = torch.linspace(0, 1, self.output_dim, device=freqs.device).view(1, 1, self.output_dim)
        
        freqs = freqs.unsqueeze(-1)
        amps = amps.unsqueeze(-1)
        prob = prob.unsqueeze(-1)
        
        # 使用极小的固定宽度（例如一个输出像素的宽度），保证梯度平滑传递
        fixed_width = 1.0 / self.output_dim
        
        # 有效振幅 = 存在概率 * 原始振幅
        effective_amps = prob * amps
        
        gaussians = effective_amps * torch.exp(-0.5 * ((x - freqs) / fixed_width) ** 2)
        spectrum = gaussians.sum(dim=1)
        
        return spectrum

    def forward_modal_anchor(self, features):
        out = self.predictor_modal_anchor(features).view(-1, self.num_modal_bins, 3)
        
        # prob: 存在概率，限制在 [0, 1]
        prob = torch.sigmoid(out[..., 0])
        # amp: 振幅
        amps = F.softplus(out[..., 2])
        
        if self.use_modal_bins:
            # 方案一：强行划分区间 (Fixed Bins)
            # offset: 在 bin 内部的相对偏移，限制在 [0, 1]
            offset = torch.sigmoid(out[..., 1])
            # 计算绝对频率
            bin_centers = (torch.arange(self.num_modal_bins, device=features.device).float() + 0.5) / self.num_modal_bins
            freqs = bin_centers.unsqueeze(0) + (offset - 0.5) / self.num_modal_bins
            freqs = freqs.clamp(0, 1)
        else:
            # 路线 B：无匹配的集合预测 (Free Continuous Set Prediction)
            # 不划分区间，直接把预测值当作全局的绝对频率，用 sigmoid 限制在 [0, 1] 之间
            freqs = torch.sigmoid(out[..., 1])
        
        # 渲染频谱
        rendered = self.render_modal_spectrum(prob, freqs, amps)
        
        # 将 prob, freqs, amps 打包作为锚点输出，供后续计算辅助 loss 或可视化使用
        anchors = torch.stack([prob, freqs, amps], dim=-1)
        
        return rendered, anchors

    def forward(self, point_features, global_features, xyz, mode="direct"):
        local_token = self.local_encoder(torch.cat([point_features, self.positional_encoding(xyz)], dim=-1))
        global_token = self.global_encoder(global_features)
        attention_input = torch.stack([local_token, global_token], dim=1)
        attention_output, _ = self.cross_attention(
            local_token.unsqueeze(1),
            attention_input,
            attention_input,
            need_weights=False,
        )
        fused_local_token = self.attention_norm(local_token + attention_output.squeeze(1))
        features = torch.cat([fused_local_token, global_token], dim=-1)
        
        if mode == "bipartite":
            return self.forward_bipartite(features)
        elif mode == "anchor":
            return self.forward_anchor(features)
        elif mode == "modal_anchor":
            return self.forward_modal_anchor(features)
        else:
            return self.forward_direct(features)


class MyPipeline(pl.LightningModule):
    def __init__(self, learning_rate=None):
        super().__init__()
        # 读取配置，决定使用哪种预测模式："direct", "bipartite", "anchor", "modal_anchor"
        self.prediction_mode = getattr(cfg, "PREDICTION_MODE", "direct")
        self.use_modal_bins = getattr(cfg, "USE_MODAL_BINS", True)  # 新增开关，默认True(使用方案一)
        self.sparse_penalty_weight = getattr(cfg, "SPARSE_PENALTY_WEIGHT", 0.01)  # 稀疏损失权重
        
        self.learning_rate = learning_rate if learning_rate is not None else getattr(cfg, "LEARNING_RATE", 1e-3)
        self.hidden_dim = getattr(cfg, "HIDDEN_DIM", 256)
        self.output_dim = getattr(cfg, "OUTPUT_DIM", 256)
        self.global_context_points = max(1, int(getattr(cfg, "GLOBAL_CONTEXT_POINTS", 512)))
        self.train_vis_every_n_epochs = max(1, int(getattr(cfg, "TRAIN_VIS_EVERY_N_EPOCHS", 1)))
        self.input_feature = ocnn.modules.InputFeature("NPD", nempty=cfg.OCTREE_NEMPTY)
        self.backbone_network = ocnn_unet.UNet(in_channels=7, out_channels=self.hidden_dim, nempty=cfg.OCTREE_NEMPTY)
        self.acoustic_head = AcousticFieldHead(
            self.hidden_dim,
            self.output_dim,
            attention_heads=max(1, int(getattr(cfg, "HEAD_ATTENTION_HEADS", 4))),
            num_peaks=12,
            use_modal_bins=self.use_modal_bins,
        )

    def build_targets(self, batch_data):
        targets = torch.cat([mel.float().to(self.device).max(dim=-1).values for mel in batch_data["mel_spectrogram"]], dim=0)
        return F.adaptive_avg_pool1d(targets.unsqueeze(1), self.output_dim).squeeze(1)



    def compute_loss_terms(self, output, targets, aux_data=None):
        # 1. 基础波形回归损失
        smooth_l1_loss = F.smooth_l1_loss(output, targets)
        
        # 2. 能量分布损失 (EMD 替代方案，更稳定)
        # 避免除以 0，并且强制在正数域内
        pred_dist = output.clamp(min=1e-6)
        target_dist = targets.clamp(min=1e-6)
        
        pred_dist = pred_dist / pred_dist.sum(dim=-1, keepdim=True)
        target_dist = target_dist / target_dist.sum(dim=-1, keepdim=True)
        
        pred_cdf = torch.cumsum(pred_dist, dim=-1)
        target_cdf = torch.cumsum(target_dist, dim=-1)
        emd_loss = F.l1_loss(pred_cdf, target_cdf)
        
        # 3. 能量守恒约束 (强制网络输出能量，打破全0坍塌)
        # 让预测频谱的总能量，尽量接近真实频谱的总能量
        pred_energy = output.sum(dim=-1)
        target_energy = targets.sum(dim=-1)
        energy_loss = F.l1_loss(pred_energy, target_energy)
        
        total_loss = smooth_l1_loss * 10.0 + emd_loss * 1.0 + energy_loss * 0.1
        mode_loss = torch.tensor(0.0, device=self.device)
        
        # 4. 模态稀疏惩罚 (针对 modal_anchor 方案及无匹配集合预测方案)
        if self.prediction_mode == "modal_anchor" and aux_data is not None:
            # aux_data 为 anchors: [..., prob, freqs, amps]
            prob = aux_data[..., 0]
            # 施加 L1 稀疏惩罚，强迫网络用尽量少的模态，抑制无效的 prob
            mode_loss = prob.mean()
            total_loss = total_loss + mode_loss * self.sparse_penalty_weight
                
        return total_loss, smooth_l1_loss, emd_loss, mode_loss, energy_loss

    def select_global_context_points(self, vertices):
        if vertices.size(0) <= self.global_context_points:
            return vertices
        step = max(1, (vertices.size(0) + self.global_context_points - 1) // self.global_context_points)
        return vertices[::step][:self.global_context_points]

    def build_batched_query_points(self, point_groups):
        point_xyz = torch.cat(point_groups, dim=0)
        point_counts = torch.tensor([points.size(0) for points in point_groups], dtype=torch.long, device=self.device)
        query_batch_index = torch.repeat_interleave(
            torch.arange(len(point_groups), device=self.device, dtype=torch.long),
            point_counts,
        )
        query_pts = torch.cat([point_xyz, query_batch_index[:, None].float()], dim=1)
        return point_xyz, query_pts, point_counts

    def build_prediction_report(self, batch_data, targets, output, loss, stage):
        gt = targets.detach().cpu()
        pred = output.detach().cpu()
        diff = (pred - gt).abs()
        object_impact_count = int(batch_data["num_impacts"][0].item())
        gt_object = gt[:object_impact_count]
        pred_object = pred[:object_impact_count]
        diff_object = diff[:object_impact_count]
        sample_idx = 0
        gt_sample = gt_object[sample_idx]
        pred_sample = pred_object[sample_idx]
        diff_sample = diff_object[sample_idx]
        sample_count = min(8, gt_object.size(0))
        gt_panel = gt_object[:sample_count]
        pred_panel = pred_object[:sample_count]
        mae = diff_object.mean().item()
        rmse = torch.sqrt(((pred_object - gt_object) ** 2).mean()).item()
        corr = torch.corrcoef(torch.stack([gt_sample, pred_sample]))[0, 1].item() if gt_sample.numel() > 1 else 0.0
        worst_dims = torch.topk(diff_sample, k=min(8, diff_sample.numel())).indices.tolist()
        impact_points = batch_data["impact_point"][0].detach().cpu()
        highlighted_point = impact_points[sample_idx]
        axis_variance = impact_points.var(dim=0)
        axis_order = torch.argsort(axis_variance, descending=True).tolist()
        axis_x = axis_order[0]
        axis_y = axis_order[1]
        axis_names = ["x", "y", "z"]
        obj_id = batch_data["obj_id"][0]

        fig = plt.figure(figsize=(16, 10), dpi=160)
        gs = fig.add_gridspec(3, 3, height_ratios=[1.1, 1.6, 1.6])

        ax_text = fig.add_subplot(gs[0, :])
        ax_text.axis("off")
        text = "\n".join([
            f"epoch={self.current_epoch}  {stage}_loss={float(loss.item()):.6f}  obj_id={obj_id}  obj_impacts={object_impact_count}  dims={gt_object.size(1)}",
            f"object_mae={mae:.6f}  object_rmse={rmse:.6f}  impact_idx={sample_idx}  sample_corr={corr:.6f}",
            f"gt(mean/std/min/max)=({gt_sample.mean():.4f}, {gt_sample.std():.4f}, {gt_sample.min():.4f}, {gt_sample.max():.4f})",
            f"pred(mean/std/min/max)=({pred_sample.mean():.4f}, {pred_sample.std():.4f}, {pred_sample.min():.4f}, {pred_sample.max():.4f})",
            f"impact_xyz=({highlighted_point[0]:.4f}, {highlighted_point[1]:.4f}, {highlighted_point[2]:.4f})",
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
        ax_scatter.scatter(
            impact_points[:, axis_x].numpy(),
            impact_points[:, axis_y].numpy(),
            s=24,
            alpha=0.7,
            label="Impacts",
        )
        ax_scatter.scatter(
            highlighted_point[axis_x].item(),
            highlighted_point[axis_y].item(),
            s=70,
            color="tab:red",
            label="Selected",
        )
        ax_scatter.set_title("Impact Positions")
        ax_scatter.set_xlabel(axis_names[axis_x])
        ax_scatter.set_ylabel(axis_names[axis_y])
        ax_scatter.legend()

        vmin = min(gt_panel.min().item(), pred_panel.min().item())
        vmax = max(gt_panel.max().item(), pred_panel.max().item())

        ax_gt = fig.add_subplot(gs[2, 0])
        im_gt = ax_gt.imshow(gt_panel.numpy().T, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax_gt.set_title("GT Heatmap")
        fig.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)

        ax_pred = fig.add_subplot(gs[2, 1])
        im_pred = ax_pred.imshow(pred_panel.numpy().T, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax_pred.set_title("Pred Heatmap")
        fig.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)

        ax_err = fig.add_subplot(gs[2, 2])
        im_err = ax_err.imshow((pred_panel - gt_panel).abs().numpy().T, aspect="auto", cmap="magma")
        ax_err.set_title("AbsDiff Heatmap")
        fig.colorbar(im_err, ax=ax_err, fraction=0.046, pad=0.04)

        fig.tight_layout()
        fig.canvas.draw()
        image = torch.from_numpy(np.asarray(fig.canvas.buffer_rgba()).copy()[..., :3]).permute(2, 0, 1)
        plt.close(fig)
        return image

    def forward(self, batch_data):
        impact_points = [points.to(self.device) for points in batch_data["impact_point"]]
        global_context_points = [
            self.select_global_context_points(vertices.to(self.device))
            for vertices in batch_data["gnn_vertices"]
        ]
        octree = batch_data["octree"].to(self.device)
        data = self.input_feature(octree)
        targets = self.build_targets(batch_data) # target: [n_points, n_freqs] 每个频率的能量峰值
        combined_query_groups = [
            torch.cat([impact_point, context_point], dim=0)
            for impact_point, context_point in zip(impact_points, global_context_points)
        ]
        point_xyz, query_pts, _ = self.build_batched_query_points(combined_query_groups)
        combined_features = self.backbone_network(data=data, octree=octree, depth=octree.depth, query_pts=query_pts) # (n_points, n_features)
        local_feature_groups = []
        global_feature_groups = []
        split_sizes = [points.size(0) for points in combined_query_groups]
        for impact_point, context_point, feature_group in zip(
            impact_points,
            global_context_points,
            combined_features.split(split_sizes),
        ):
            impact_count = impact_point.size(0)
            local_features = feature_group[:impact_count]
            context_features = feature_group[impact_count:impact_count + context_point.size(0)]
            pooled_context = torch.cat(
                [context_features.mean(dim=0), context_features.max(dim=0).values],
                dim=0,
            ) # (2*n_features) 为上下文特征的均值和最大值
            global_features = pooled_context.unsqueeze(0).expand(impact_count, -1)
            local_feature_groups.append(local_features)
            global_feature_groups.append(global_features)
        point_xyz = torch.cat(impact_points, dim=0)
        local_features = torch.cat(local_feature_groups, dim=0)
        global_features = torch.cat(global_feature_groups, dim=0)
        
        output, aux_data = self.acoustic_head(local_features, global_features, point_xyz, mode=self.prediction_mode)
        loss, smooth_l1_loss, emd_loss, mode_loss, energy_loss = self.compute_loss_terms(output, targets, aux_data)
        return loss, output, smooth_l1_loss, emd_loss, mode_loss, energy_loss

    def training_step(self, batch, batch_idx):
        loss, output, smooth_l1_loss, emd_loss, mode_loss, energy_loss = self(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch["num_impacts"].sum().item())
        self.log("train_smooth_l1_loss", smooth_l1_loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch["num_impacts"].sum().item())
        self.log("train_emd_loss", emd_loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch["num_impacts"].sum().item())
        self.log("train_energy_loss", energy_loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch["num_impacts"].sum().item())
        if self.prediction_mode in ["bipartite", "anchor", "modal_anchor"]:
            self.log("train_mode_loss", mode_loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch["num_impacts"].sum().item())
        
        opt = self.optimizers()
        if opt:
            lr = opt.param_groups[0]["lr"]
            self.log("lr", lr, on_step=False, on_epoch=True, prog_bar=True)
            
        should_log_train_visualization = (self.current_epoch) % self.train_vis_every_n_epochs == 0
        if batch_idx == 0 and should_log_train_visualization and getattr(self.logger, "experiment", None) is not None:
            targets = self.build_targets(batch)
            report = self.build_prediction_report(batch, targets, output, loss, stage="train")
            self.logger.experiment.add_image("train/gt_pred_absdiff", report, self.current_epoch)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, output, smooth_l1_loss, emd_loss, mode_loss = self(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch["num_impacts"].sum().item())
        self.log("val_smooth_l1_loss", smooth_l1_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch["num_impacts"].sum().item())
        self.log("val_emd_loss", emd_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch["num_impacts"].sum().item())
        self.log("val_energy_loss", energy_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch["num_impacts"].sum().item())
        if self.prediction_mode in ["bipartite", "anchor", "modal_anchor"]:
            self.log("val_mode_loss", mode_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch["num_impacts"].sum().item())
            
        if batch_idx == 0 and getattr(self.logger, "experiment", None) is not None:
            targets = self.build_targets(batch)
            report = self.build_prediction_report(batch, targets, output, loss, stage="val")
            self.logger.experiment.add_image("val/gt_pred_absdiff", report, self.current_epoch)
        return loss

    def test_step(self, batch, batch_idx):
        loss, _, smooth_l1_loss, emd_loss, mode_loss = self(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch["num_impacts"].sum().item())
        self.log("test_smooth_l1_loss", smooth_l1_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch["num_impacts"].sum().item())
        self.log("test_emd_loss", emd_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch["num_impacts"].sum().item())
        self.log("test_energy_loss", energy_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch["num_impacts"].sum().item())
        if self.prediction_mode in ["bipartite", "anchor", "modal_anchor"]:
            self.log("test_mode_loss", mode_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch["num_impacts"].sum().item())
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
        total_epochs = max(1, int(getattr(cfg, "MAX_EPOCHS", 1)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: max(0.0, 1.0 - epoch / total_epochs),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
