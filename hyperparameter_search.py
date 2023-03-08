from smac.configspace import ConfigurationSpace
from ConfigSpace import Integer, Float, Categorical
from smac.scenario.scenario import Scenario
from smac.facade.smac_bb_facade import SMAC4BB

import torch
from torch.utils.data import DataLoader
from object_detection_data import TransformedDataset
from utils import Encoder
import pytorch_lightning as pl
from object_detection_models import MultiClassJetNet


def scalings(
    initial_quadratic_scaling,
    initial_rectangular_scaling,
    number_of_quadratic,
    number_of_rectangular,
):
    return torch.row_stack(
        (
            initial_quadratic_scaling
            / 1.4 ** torch.arange(0, number_of_quadratic).reshape((-1, 1)),
            initial_rectangular_scaling
            / 1.4 ** torch.arange(0, number_of_rectangular).reshape((-1, 1)),
        )
    )


def train_network(config, seed=0):
    # TODO: Parametrize by aspect ratio
    default_scalings = scalings(
        torch.tensor([1.0, 1.0]),
        torch.tensor([0.5, 1.0]),
        config["number_of_quadratic"],
        config["number_of_rectangular"],
    )
    num_classes = 4
    encoder = Encoder(default_scalings, num_classes)
    transformed_train_data = TransformedDataset("data/train", encoder)
    transformed_val_data = TransformedDataset("data/val", encoder)

    train_loader = DataLoader(
        transformed_train_data,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        transformed_val_data,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    pl_model = MultiClassJetNet(encoder, config["learning_rate"])
    trainer = pl.Trainer(limit_predict_batches=100, max_epochs=20)
    trainer.fit(
        model=pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )
    return pl_model.mean_average_precisions[-1].item()


if __name__ == "__main__":
    torch.manual_seed(2)

    configspace = ConfigurationSpace(seed=0)
    configspace.add_hyperparameters(
        [
            Float("learning_rate", (0.00001, 0.1), log=True),
            Categorical("batch_size", [4, 8, 16, 32, 64, 128, 256]),
            Integer("number_of_quadratic", (1, 8)),
            Integer("number_of_rectangular", (1, 8)),
        ]
    )
    scenario = Scenario({"run_obj": "quality", "runcount-limit": 10, "cs": configspace})
    optimizer = SMAC4BB(scenario=scenario, tae_runner=train_network)
    result = optimizer.optimize()
    print(result)
