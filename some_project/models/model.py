from torch import nn
import hydra


@hydra.main(config_path="config", config_name="experiment/exp1.yaml")
def define_model(config):
    hparams = config

    # Model architecture
    MyAwesomeModel = nn.Sequential(
        nn.Conv2d(1, hparams['conv1_dim'], 3),  # [B, 1, 28, 28] -> [B, 32, 26, 26]
        nn.ReLU(),
        nn.Conv2d(hparams['conv1_dim'], hparams['conv2_dim'], 3),  # [B, 32, 26, 26] -> [B, 64, 24, 24]
        nn.ReLU(),
        nn.Conv2d(hparams['conv2_dim'], hparams['conv3_dim'], 3),  # [B, 64, 24, 24] -> [B, 128, 22, 22]
        nn.ReLU(),
        nn.MaxPool2d(2),  # [B, 128, 22, 22] -> [B, 128, 11, 11]
        nn.Flatten(),  # [B, 128, 11, 11] -> [B, 64 * 12 * 12]
        nn.Linear(hparams['conv3_dim'] * 11 * 11, 10),
    )
    return MyAwesomeModel
