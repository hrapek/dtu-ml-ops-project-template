import click
import torch
import logging
import hydra
from omegaconf import OmegaConf
from hydra.utils import to_absolute_path

# import os
# import time
from torch import nn

from models.model import define_model

log = logging.getLogger(__name__)

@hydra.main(config_path="config", config_name="experiment/exp1.yaml")

# @click.group()
# def cli():
#     """Command line interface."""
#     pass


# @click.command()
# @click.option("--data-path", default="data/processed", help="Path to load data from")
# @click.option("--lr", default=1e-3, help="learning rate to use for training")
# @click.option("--batch_size", default=256, help="batch size to use for training")
# @click.option("--n_epochs", default=20, help="number of epochs to train for")
def train(config):
    """Train a model on MNIST."""
    log.info("Training day and night")

    print(f"configuration: \n {OmegaConf.to_yaml(config)}")
    hparams = config.experiment
    torch.manual_seed(hparams["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Creating directory based on time
    # current_time = time.localtime()
    # folder_name = time.strftime("%Y/%m/%d", current_time)  # Create folder structure based on current date
    # model_path = f"models/{folder_name}"
    # os.makedirs(model_path, exist_ok=True)
    # model_time = time.strftime("%H%M%S", current_time)
    # model_name = f'model_{model_time}'
    # model_file = f"{model_path}/{model_name}.pt"  # Create the complete file path

    # Training
    model_config = OmegaConf.load(to_absolute_path("some_project/models/config/experiment/exp1.yaml"))
    model = define_model(model_config).to(device)
    print(f"configuration: \n {OmegaConf.to_yaml(model_config)}")

    train_set = torch.load(to_absolute_path(f'{hparams["data_path"]}/train_data.pt'), map_location=device)
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=hparams["batch_size"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=hparams["lr"])
    loss_fn = nn.CrossEntropyLoss()

    for e in range(hparams["n_epochs"]):
        for batch in train_dataloader:
            optimizer.zero_grad()
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
        log.info(f"Epoch {e} Loss {loss}")

    torch.save(model, to_absolute_path("models/model.pt"))  # Save model state dictionary with the unique name


if __name__ == "__main__":
    train()
