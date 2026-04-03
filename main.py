import os
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import random_split

# Add project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from config.config import cfg
from src.pipeline import MyPipeline
# Assuming you will use EigenMeshDataset or VVImpactDataset for training
from src.eigen_dataset import EigenMeshDataset
# from src.dataset_loader import VVImpactDataset

def main():
    print("="*50)
    print("Starting Training Pipeline")
    print("="*50)
    
    # 1. Setup seed for reproducibility
    pl.seed_everything(42)
    
    # 2. Load Dataset
    print(f"Loading dataset from {cfg.DATA_DIR}...")
    
    # You can change this to VVImpactDataset depending on your task
    dataset = EigenMeshDataset(
        data_dir=os.path.join(cfg.DATA_DIR, "coarse_eigen_mesh"),
        cache_dir=os.path.join(cfg.DATA_DIR, "cache"),
        k=cfg.N_EIGENMODES
    )
    
    if len(dataset) == 0:
        print("Warning: Dataset is empty. Cannot start training.")
        return
        
    # Split dataset (80% train, 20% val)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create DataLoaders
    # Using PyTorch Geometric DataLoader to properly batch graph data (handles pos, x, batch_idx correctly)
    train_loader = PyGDataLoader(
        train_dataset, 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=True, 
        num_workers=cfg.NUM_WORKERS,
        persistent_workers=True if cfg.NUM_WORKERS > 0 else False
    )
    
    val_loader = PyGDataLoader(
        val_dataset, 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=False, 
        num_workers=cfg.NUM_WORKERS,
        persistent_workers=True if cfg.NUM_WORKERS > 0 else False
    )
    
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    
    # 3. Initialize Model Pipeline
    print("Initializing Model Pipeline...")
    model = MyPipeline(learning_rate=cfg.LEARNING_RATE)
    
    # 4. Setup Callbacks
    # Save the best model based on validation loss
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(project_root, "checkpoints"),
        filename="best-checkpoint-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )
    
    # Early stopping
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=10,
        verbose=True,
        mode="min"
    )
    
    # 5. Initialize Trainer
    # Since torch-cluster's fps op fails on MPS (Mac GPU) and only works on CPU/CUDA,
    # we explicitly set accelerator="cpu" for now to avoid the RuntimeError.
    trainer = pl.Trainer(
        max_epochs=cfg.MAX_EPOCHS, 
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator="cpu",  # Force CPU to avoid torch-cluster MPS bugs
        devices=1,
        log_every_n_steps=1, # Adjust this based on your dataset size
        default_root_dir=os.path.join(project_root, "logs")
    )
    
    # 6. Start Training
    print("Starting training loop...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    # 7. (Optional) Test the model
    # print("Running testing...")
    # trainer.test(model, dataloaders=val_loader) # Replace with test_loader if you have one
    
    print("Training finished.")

if __name__ == "__main__":
    main()
