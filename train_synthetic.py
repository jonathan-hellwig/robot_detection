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
batch_size = 16
default_scalings = torch.tensor([[0.25, 0.25]])

torch.manual_seed(2)
encoder = Encoder(default_scalings, num_classes)
data_module = SyntheticDataModule(encoder, batch_size)
model = JetNet(num_classes, default_scalings.size(0), learning_rate)
loss = SingleShotDetectorLoss(alpha=alpha)
task = ObjectDetectionTask(model, loss, encoder)

tb_logger = pl_loggers.TensorBoardLogger(save_dir="synthetic_data_logs/")

data_module.setup()
# trainer = pl.Trainer(
#     limit_predict_batches=100,
#     max_epochs=200,
#     callbacks=[RichProgressBar()],
#     logger=tb_logger,
# )
# trainer.fit(model=task, datamodule=data_module)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for _ in range(1000):
    for batch in data_module.train_dataloader():
        images, encoded_target_bounding_boxes, encoded_target_classes = batch
        (
            encoded_predicted_bounding_boxes,
            encoded_predicted_class_logits,
        ) = model.forward(images)
        total_loss, location_loss, classification_loss = loss(
            encoded_target_bounding_boxes,
            encoded_target_classes,
            encoded_predicted_bounding_boxes,
            encoded_predicted_class_logits,
        )
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(total_loss)
