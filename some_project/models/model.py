from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch import nn, optim
from pytorch_lightning import LightningModule


class MyAwesomeModel(LightningModule):
    def __init__(self):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3),  # [B, 1, 28, 28] -> [B, 32, 26, 26]
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),  # [B, 32, 26, 26] -> [B, 64, 24, 24]
            nn.ReLU(),
            nn.Conv2d(64, 128, 3),  # [B, 64, 24, 24] -> [B, 128, 22, 22]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 128, 22, 22] -> [B, 128, 11, 11]
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),  # [B, 128, 11, 11] -> [B, 64 * 12 * 12]
            nn.Linear(128 * 11 * 11, 10)
        )

        self.criterium = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.classifier(self.backbone(x))
    
    def training_step(self, batch):
        data, target = batch
        preds = self(data)
        loss = self.criterium(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.03)
    
    def test_step(self, batch):
        data, target = batch
        preds = self(data)
        loss = self.criterium(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics)
        return metrics   
