import math
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Subset

torch.multiprocessing.set_sharing_strategy('file_system')

# Add project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from config.config import cfg
from src.pipeline import MyPipeline
from src.dataset_loader import VVImpactDataset, collate_vvimpact_batch


def build_train_val_subsets(dataset):
    dataset_percent = float(getattr(cfg, "DATASET_PERCENT", 100.0))
    if not 0 < dataset_percent <= 100:
        raise ValueError(f"DATASET_PERCENT must be in (0, 100], got {dataset_percent}.")

    total_size = len(dataset)
    subset_size = math.ceil(total_size * dataset_percent / 100.0)
    if total_size >= 2:
        subset_size = max(2, subset_size)
    subset_size = min(total_size, subset_size)

    indices = torch.randperm(total_size).tolist()[:subset_size]
    if subset_size == 1:
        train_indices = [indices[0]]
        val_indices = [indices[0]]
    else:
        train_size = max(1, int(subset_size * 0.8))
        if train_size >= subset_size:
            train_size = subset_size - 1
        val_size = subset_size - train_size
        if val_size == 0:
            val_size = 1
            train_size -= 1
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

    return Subset(dataset, train_indices), Subset(dataset, val_indices), subset_size


def main():
    print("="*50)
    print("Starting Training Pipeline")
    print("="*50)
    
    # 1. Setup seed for reproducibility
    pl.seed_everything(42)
    
    # 2. Load Dataset
    print(f"Loading dataset from {cfg.DATA_DIR}...")
    
    dataset = VVImpactDataset(
        data_dir=cfg.DATA_DIR,
        sample_rate=cfg.SAMPLE_RATE,
        train_only=True,
        obj_limit=cfg.OBJ_LIMIT,
    )
    
    if len(dataset) == 0:
        print("Warning: Dataset is empty. Cannot start training.")
        return
        
    train_dataset, val_dataset, subset_size = build_train_val_subsets(dataset)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=True, 
        num_workers=cfg.NUM_WORKERS,
        persistent_workers=True if cfg.NUM_WORKERS > 0 else False,
        collate_fn=collate_vvimpact_batch,
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=False, 
        num_workers=cfg.NUM_WORKERS,
        persistent_workers=True if cfg.NUM_WORKERS > 0 else False,
        collate_fn=collate_vvimpact_batch,
    )
    
    print(
        f"Using {subset_size}/{len(dataset)} samples "
        f"({cfg.DATASET_PERCENT:.2f}%) | Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}"
    )
    
    # 3. Initialize Model Pipeline
    print("Initializing Model Pipeline...")
    model = MyPipeline(learning_rate=cfg.LEARNING_RATE)
    
    # 4. Setup Callbacks
    # Save the best model based on validation loss
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(project_root, "checkpoints"),
        filename="best-checkpoint-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        verbose=False,
        monitor="val_loss",
        mode="min"
    )
    
    # Early stopping (通过配置文件控制耐心值)
    early_stop_patience = getattr(cfg, "EARLY_STOP_PATIENCE", 50)
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=early_stop_patience,
        verbose=False,
        mode="min"
    )
    
    accelerator = cfg.DEVICE.lower()
    devices = cfg.DEVICES
    if accelerator == "cuda" and not torch.cuda.is_available():
        accelerator = "cpu"
        devices = 1
    if accelerator == "mps" and not torch.backends.mps.is_available():
        accelerator = "cpu"
        devices = 1

    tensorboard_logger = TensorBoardLogger(
        save_dir=os.path.join(project_root, "logs"),
        name="tensorboard",
    )

    trainer = pl.Trainer(
        max_epochs=cfg.MAX_EPOCHS, 
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator=accelerator,
        devices=devices,
        check_val_every_n_epoch=max(1, int(getattr(cfg, "VAL_EVERY_N_EPOCHS", 1))),
        log_every_n_steps=5,
        logger=tensorboard_logger,
        enable_progress_bar=True,
        enable_model_summary=True,
        num_sanity_val_steps=0,
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
