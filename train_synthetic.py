import torch
from utils import Encoder
from pytorch_lightning.callbacks import RichProgressBar
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from models import JetNet, SingleShotDetectorLoss, ObjectDetectionTask
from datasets import SyntheticDataModule

# Hyperparameters
num_classes = 1
learning_rate = 1e-2
alpha = 1.0
batch_size = 64
default_scalings = torch.tensor([[0.25, 0.25]])

torch.manual_seed(2)
encoder = Encoder(default_scalings, num_classes)
data_module = SyntheticDataModule(encoder, batch_size)
model = JetNet(num_classes, default_scalings.size(0))
loss = SingleShotDetectorLoss(alpha=alpha)
task = ObjectDetectionTask(model, loss, encoder, learning_rate)

tb_logger = pl_loggers.TensorBoardLogger(save_dir="synthetic_data_logs/")

data_module.setup()
trainer = pl.Trainer(
    limit_predict_batches=100,
    max_epochs=200,
    callbacks=[RichProgressBar()],
    logger=tb_logger,
    log_every_n_steps=2,
)
trainer.fit(model=task, datamodule=data_module)
