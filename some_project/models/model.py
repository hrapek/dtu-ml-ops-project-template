from torch import nn

MyAwesomeModel = nn.Sequential(
    nn.Conv2d(1, 32, 3),  # [B, 1, 28, 28] -> [B, 32, 26, 26]
    nn.ReLU(),
    nn.Conv2d(32, 64, 3),  # [B, 32, 26, 26] -> [B, 64, 24, 24]
    nn.ReLU(),
    nn.Conv2d(64, 128, 3),  # [B, 64, 24, 24] -> [B, 128, 22, 22]
    nn.ReLU(),
    nn.MaxPool2d(2),  # [B, 128, 22, 22] -> [B, 128, 11, 11]
    nn.Flatten(),  # [B, 128, 11, 11] -> [B, 64 * 12 * 12]
    nn.Linear(128 * 11 * 11, 10),
)
