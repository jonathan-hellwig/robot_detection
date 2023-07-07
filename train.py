from utils import Encoder
import torch
from datasets import RoboEireanDataModule
import lightning.pytorch as pl

from models import JetNet, SingleShotDetectorLoss, ObjectDetectionTask
from lightning.pytorch.loggers import TensorBoardLogger

if __name__ == "__main__":
    LEARNING_RATE = 1e-2
    ALPHA = 2.0
    NUM_CLASSES = 1
    DEFAULT_SCALINGS = torch.tensor(
        [
            [0.06549374, 0.12928654],
            [0.11965626, 0.26605093],
            [0.20708716, 0.38876095],
            [0.31018215, 0.47485098],
            [0.415882, 0.8048184],
            [0.7293086, 0.8216225],
        ]
    )
    encoder = Encoder(DEFAULT_SCALINGS, NUM_CLASSES)
    model = JetNet(NUM_CLASSES, DEFAULT_SCALINGS.shape[0])
    loss = SingleShotDetectorLoss(ALPHA)
    data_module = RoboEireanDataModule("data/raw/", encoder, 128)
    data_module.setup("fit")
    task = ObjectDetectionTask(model, loss, encoder, LEARNING_RATE)
    logger = TensorBoardLogger(save_dir="new_logs")
    trainer = pl.Trainer(max_epochs=10, logger=logger)
    trainer.fit(model=task, datamodule=data_module)
