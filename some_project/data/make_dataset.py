import click
import torch
import glob
import os


@click.group()
def cli():
    """Command line interface."""
    pass


def generate_mnist_dataset(load_path):
    """Return train and test dataloaders for MNIST."""

    #check how many files there are and load them all
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
    x_test = torch.load(f"{load_path}/test_images.pt").unsqueeze(1)
    y_test = torch.load(f"{load_path}/test_target.pt")

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


@click.command()
@click.option("--load-path", default="data/raw/corruptmnist/", help="Path to load data from")
@click.option("--save-path", default="data/processed/", help="Path to save processed data")
def process_files(load_path, save_path):
    """Process files and save it in desired directory"""

    train, test = generate_mnist_dataset(load_path=load_path)
    torch.save(train, f"{save_path}/train_data.pt")
    torch.save(test, f"{save_path}/test_data.pt")


if __name__ == "__main__":
    process_files()
