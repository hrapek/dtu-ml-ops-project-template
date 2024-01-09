import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from data.make_dataset import MNISTDataModule
from models.model import MyAwesomeModel

# initiate data and model instances
data = MNISTDataModule()
model = MyAwesomeModel()  # LightningModule

# monitor model checkpoints
checkpoint_callback = ModelCheckpoint(dirpath='./models', monitor='train_loss', mode='min')

# trainer wiht wandb logger
trainer = pl.Trainer(
    accelerator='gpu', max_epochs=5, callbacks=[checkpoint_callback], logger=pl.loggers.WandbLogger(project='dtu_mlops')
)

if __name__ == '__main__':
    trainer.fit(model, data)
