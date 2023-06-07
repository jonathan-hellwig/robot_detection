from utils import Encoder
import torch
from datasets import RoboEireanDataModule
import pytorch_lightning as pl

from models import JetNet, SingleShotDetectorLoss, ObjectDetectionTask
from lightning.pytorch.loggers import TensorBoardLogger

if __name__ == "__main__":
    default_scalings = torch.tensor(
        [
            [0.06549374, 0.12928654],
            [0.11965626, 0.26605093],
            [0.20708716, 0.38876095],
            [0.31018215, 0.47485098],
            [0.415882, 0.8048184],
            [0.7293086, 0.8216225],
        ]
    )
    encoder = Encoder(default_scalings, 1)
    model = JetNet(1, default_scalings.shape[0], 0.01)
    loss = SingleShotDetectorLoss()
    data_module = RoboEireanDataModule("data/raw/", ["robot"], encoder)
    data_module.setup()
    task = ObjectDetectionTask(model, loss, encoder)
    logger = TensorBoardLogger(save_dir="new_logs")
    trainer = pl.Trainer(max_epochs=10, logger=logger)
    trainer.fit(model=task, datamodule=data_module)
