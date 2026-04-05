import os
import sys
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, random_split

# Add project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from config.config import cfg
from src.pipeline import MyPipeline
from src.dataset_loader import VVImpactDataset, collate_vvimpact_batch


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
    )
    
    if len(dataset) == 0:
        print("Warning: Dataset is empty. Cannot start training.")
        return
        
    # Split dataset (80% train, 20% val)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
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
        verbose=False,
        monitor="val_loss",
        mode="min"
    )
    
    # Early stopping
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=10,
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
        log_every_n_steps=1,
        logger=tensorboard_logger,
        enable_progress_bar=True,
        enable_model_summary=False,
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
