import glob
import os

import pytorch_lightning as pl
import torch


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, load_path='data/raw/corruptmnist/', save_path='data/processed/'):
        super().__init__()
        self.load_path = load_path
        self.save_path = save_path

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            train, test = self.generate_mnist_dataset(self.load_path)
            torch.save(train, f'{self.save_path}/train_data.pt')
            torch.save(test, f'{self.save_path}/test_data.pt')

    def generate_mnist_dataset(self, load_path):
        """Return train and test dataloaders for MNIST."""

        # check how many files there are and load them all
        train_images = glob.glob(os.path.join(load_path, 'train_images_*.pt'))
        train_targets = glob.glob(os.path.join(load_path, 'train_target_*.pt'))

        train_images.sort()
        train_targets.sort()

        x_train_raw = {i: torch.load(image_file) for i, image_file in enumerate(train_images)}
        y_train_raw = {i: torch.load(target_file) for i, target_file in enumerate(train_targets)}

        # Combine train data dynamically
        x_train = torch.cat([x_train_raw[i] for i in x_train_raw], dim=0).unsqueeze(1)
        y_train = torch.cat([y_train_raw[i] for i in y_train_raw], dim=0)

        # load test data
        x_test = torch.load(f'{load_path}/test_images.pt').unsqueeze(1)
        y_test = torch.load(f'{load_path}/test_target.pt')

        # calulate mean and sd
        mean_train, std_train = torch.mean(x_train), torch.std(x_train)
        mean_test, std_test = torch.mean(x_test), torch.std(x_test)

        # normalize
        x_train = (x_train - mean_train) / std_train
        x_test = (x_test - mean_test) / std_test

        print(x_train.shape)
        print(y_train.shape)
        print(x_test.shape)
        print(y_test.shape)

        return (torch.utils.data.TensorDataset(x_train, y_train), torch.utils.data.TensorDataset(x_test, y_test))

    def train_dataloader(self):
        train = torch.load(f'{self.save_path}/train_data.pt')
        return torch.utils.data.DataLoader(train, batch_size=64, shuffle=True, num_workers=11)

    def test_dataloader(self):
        test = torch.load(f'{self.save_path}/test_data.pt')
        return torch.utils.data.DataLoader(test, batch_size=64, num_workers=11)
