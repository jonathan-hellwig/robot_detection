#!/bin/env python
import os
from datetime import timedelta

# there is a bug somewhere, this fixes things for progress bars
from tqdm import tqdm as _

import torch
import lightning
from datasets import SyntheticDataModule
from models import JetNet, ObjectDetectionTask, SingleShotDetectorLoss
import optuna
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

# from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning_pruning_callback import PyTorchLightningPruningCallback
from utils import Encoder


def compose_hyperparameters(trial: optuna.trial.Trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-1, log=True)
    alpha = trial.suggest_float("alpha", 0.1, 3.0, log=True)
    batch_size = 2 ** trial.suggest_int("batch_size", 4, 8, log=True)
    default_scalings = torch.tensor([[0.25, 0.25]])

    return {
        "learning_rate": learning_rate,
        "alpha": alpha,
        "batch_size": batch_size,
        "default_scalings": default_scalings,
    }


def objective(trial: optuna.trial.Trial):
    study_name = trial.study.study_name
    hyperparameters = compose_hyperparameters(trial)
    num_classes = 1
    encoder = Encoder(hyperparameters["default_scalings"], num_classes)
    data_module = SyntheticDataModule(encoder, hyperparameters["batch_size"])
    model = JetNet(num_classes, hyperparameters["default_scalings"].size(0))
    loss = SingleShotDetectorLoss(hyperparameters["alpha"])
    task = ObjectDetectionTask(model, loss, encoder, hyperparameters["learning_rate"])

    early_stopping_callback = EarlyStopping(
        monitor="validation/mean_average_precision",
        min_delta=0.01,
        patience=3,
        mode="max",
    )
    pruning_callback = PyTorchLightningPruningCallback(
        trial, monitor="validation/mean_average_precision"
    )
    checkpoint_callback = ModelCheckpoint(
        f"checkpoints/{study_name}/trial_{trial.number}",
        monitor="validation/mean_average_precision",
        mode="max",
    )
    tensorboard_logger = TensorBoardLogger(
        save_dir="logs",
        name=study_name,
        version=f"trial_{trial.number}",
        default_hp_metric=False,
    )
    trainer = lightning.Trainer(
        max_epochs=30,
        callbacks=[
            early_stopping_callback,
            checkpoint_callback,
            pruning_callback,
        ],
        logger=tensorboard_logger,
        enable_model_summary=False,
        accelerator="cpu",
    )
    initial_performance = trainer.validate(task, data_module, verbose=False)[0][
        "validation/mean_average_precision"
    ]
    trainer.logger.log_hyperparams(
        hyperparameters,
        metrics={"validation/mean_average_precision": initial_performance},
    )
    trainer.fit(task, data_module)
    mean_average_precision = trainer.callback_metrics[
        "validation/mean_average_precision"
    ].item()
    return mean_average_precision


if __name__ == "__main__":
    sampler = optuna.samplers.TPESampler()
    pruner = optuna.pruners.HyperbandPruner()
    study = optuna.create_study(
        study_name="mnist",
        direction="maximize",
        storage=None,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    timeout = timedelta(hours=4)
    study.optimize(objective, timeout=timeout.total_seconds())
