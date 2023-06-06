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
    # for batch in data_module.train_dataloader():
    #     images, target_bounding_boxes, target_classes = batch
    #     loss_value = task.training_step(batch, 0)
    #     print(loss_value)
    #     break
    # data_module.setup()
    # num_robots = []
    # for batch in data_module.train_dataloader():
    #     predicted_boxes, predicted_classes = model(images)
    #     loss = task.training_step(batch, 0)
    #     print(loss)
    #     # print(predicted_boxes.shape, predicted_classes.shape)
    #     break
