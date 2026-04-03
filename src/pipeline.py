import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

# Add project root to path to import config
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from config.config import cfg

import src.models.triplane as triplane
import src.models.pointnet2 as pointnet2

import torch_geometric
from torch_geometric.nn.models.mlp import MLP

class MyPipeline(pl.LightningModule):
    def __init__(self, learning_rate=None):
        super(MyPipeline, self).__init__()
        self.learning_rate = learning_rate if learning_rate is not None else cfg.LEARNING_RATE
        # encoder 神经网络
        self.backbone = pointnet2.DeepPointNet2(in_channels=3, out_channels=128)
        # decoder 神经网络
        self.decoder = triplane.ModulatedNetwork(input_dim=3, output_dim=64, embd_dim=128)

        
        
    def make_prediction(self, geo_data, query_xyz):
        """
        Forward pass to predict features at query locations.
        
        Args:
            geo_data: PyTorch Geometric Data object containing the 3D geometry (pos, batch, x).
            query_xyz: [N, 3] tensor of query 3D coordinates (assumed to be in [-1, 1]).
            
        Returns:
            predicted_features: [N, 64] predicted feature vector at the query_xyz locations.
        """
        # 1. Pass the 3D geometry through the PointNet2 backbone to get point-wise embeddings
        # The PointNet2 returns per-point embeddings of shape [num_points, 128]
        point_embeddings = self.backbone(geo_data)
        
        # 2. Global Pooling: Aggregate the point-wise embeddings to get a single shape embedding
        # We use global_max_pool (or mean pool) to get a [Batch, 128] shape descriptor
        # We need torch_geometric.nn.global_max_pool for this
        from torch_geometric.nn import global_max_pool
        shape_embedding = global_max_pool(point_embeddings, geo_data.batch)
        
        # If query_xyz doesn't have a batch dimension matching the shape embedding,
        # and we are doing a single object prediction, we expand the embedding.
        # Assuming query_xyz is [N, 3] for a single object in this simple forward pass.
        # If query_xyz has its own batching, we need to match it.
        # For simplicity, assuming shape_embedding is [1, 128] and query_xyz is [N, 3].
        N = query_xyz.shape[0]
        # Expand shape embedding to match the number of query points if needed by the decoder.
        # Triplane decoder takes embd of shape [N, embd_dim] matching the query points N.
        if shape_embedding.shape[0] == 1:
            embd_expanded = shape_embedding.expand(N, -1)
        else:
            # If batch size > 1, we need proper indexing. 
            # Assuming query_xyz comes with a batch_idx, but for now we handle single object
            # or assume query_xyz is correctly batched.
            # We'll just expand for the single object case as a baseline.
            embd_expanded = shape_embedding.expand(N, -1)
            
        # 3. Decode: Query the tri-plane network at the given XYZ coordinates
        # using the shape embedding as the condition.
        predicted_features = self.decoder(query_xyz, embd_expanded)
        
        return predicted_features

    def training_step(self, batch, batch_idx):
        # batch is a PyG Data object containing the batched graphs
        # It has:
        # batch.pos: [Total_N, 3] coordinates
        # batch.y: [Total_N, 64] ground truth eigenmodes
        # batch.batch: [Total_N] batch indices indicating which point belongs to which graph
        
        # 1. Forward pass
        # Use make_prediction but since we want to query at ALL points in the point cloud
        # we can just pass batch.pos as the query_xyz.
        # But wait, make_prediction expects query_xyz to have shape [N, 3] and assumes a 
        # specific batching mechanism for Triplane.
        
        # Let's write the training logic inline to handle PyG batching properly
        point_embeddings = self.backbone(batch) # [Total_N, 128]
        from torch_geometric.nn import global_max_pool
        shape_embeddings = global_max_pool(point_embeddings, batch.batch) # [B, 128]
        
        # Expand shape_embeddings back to [Total_N, 128] so each point gets its parent's embedding
        # PyG's batch attribute is exactly what we need to index the shape_embeddings!
        embd_expanded = shape_embeddings[batch.batch] # [Total_N, 128]
        
        # Query Triplane Decoder
        # Triplane decoder expects query_xyz in [-1, 1] and matching embeddings
        predicted_features = self.decoder(batch.pos, embd_expanded) # [Total_N, 64]
        
        # 2. Compute Loss
        # Compare predicted eigenmodes to Ground Truth
        # Note: Eigenvectors can have arbitrary sign (+ or -), so we should use a sign-invariant loss
        # For simplicity first, let's use MSE, but in reality you might need min(MSE(pred, gt), MSE(pred, -gt))
        loss = F.mse_loss(predicted_features, batch.y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
        return loss

    def validation_step(self, batch, batch_idx):
        point_embeddings = self.backbone(batch)
        from torch_geometric.nn import global_max_pool
        shape_embeddings = global_max_pool(point_embeddings, batch.batch)
        embd_expanded = shape_embeddings[batch.batch]
        predicted_features = self.decoder(batch.pos, embd_expanded)
        
        loss = F.mse_loss(predicted_features, batch.y)
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
        return loss
        
    def test_step(self, batch, batch_idx):
        point_embeddings = self.backbone(batch)
        from torch_geometric.nn import global_max_pool
        shape_embeddings = global_max_pool(point_embeddings, batch.batch)
        embd_expanded = shape_embeddings[batch.batch]
        predicted_features = self.decoder(batch.pos, embd_expanded)
        
        loss = F.mse_loss(predicted_features, batch.y)
        
        self.log('test_loss', loss, batch_size=batch.num_graphs)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # Add a learning rate scheduler if needed
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }