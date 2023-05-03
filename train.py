import pytorch_lightning as pl
from torch.utils.data import DataLoader
from object_detection_dataset import ObjectDetectionDataset
from models import MultiClassJetNet
from utils import Encoder

# In this file, we will train a multi-class object detection model using PyTorch Lightning.
if __name__  == "__main__":
    # Load data
    train_data = ObjectDetectionDataset("data/train")
    val_data = ObjectDetectionDataset("data/val")
    # Create model
    model = MultiClassJetNet(learning_rate=1e-3)
    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=4)
    # Create trainer
    trainer = pl.Trainer(limit_predict_batches=100, max_epochs=20)
    # Fit model
    trainer.fit(
        model=model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )
    # Train model
    # Save model
