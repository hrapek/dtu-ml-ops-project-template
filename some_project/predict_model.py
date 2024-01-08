import pytorch_lightning as pl

from models.model import MyAwesomeModel
from data.make_dataset import MNISTDataModule
from pytorch_lightning.callbacks import ModelCheckpoint

#data
data = MNISTDataModule()

#checkpoint location
checkpoint_name = 'epoch=2-step=2346.ckpt'
checkpoint_path = f'models/{checkpoint_name}'

#model abd trainer
model = MyAwesomeModel.load_from_checkpoint(checkpoint_path)  #LightningModule
trainer = pl.Trainer()

if __name__ == "__main__":
    trainer.test(model, data)
