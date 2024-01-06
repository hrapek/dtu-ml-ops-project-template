import torch
import click

@click.group()
def cli():
    """Command line interface."""
    pass


def generate_mnist_dataset(load_path):
    """Return train and test dataloaders for MNIST."""
    
    #load train data into dict
    x_train_raw = {i: torch.load(f'{load_path}/train_images_{i}.pt') for i in range(6)}
    y_train_raw = {i: torch.load(f'{load_path}/train_target_{i}.pt') for i in range(6)}

    #combine train data
    x_train = torch.cat([x_train_raw[i] for i in range(6)], dim=0).unsqueeze(1)
    y_train = torch.cat([y_train_raw[i] for i in range(6)], dim=0)

    #load test data
    x_test = torch.load(f'{load_path}/test_images.pt').unsqueeze(1)
    y_test = torch.load(f'{load_path}/test_target.pt')

    #calulate mean and sd
    mean_train, std_train = torch.mean(x_train), torch.std(x_train)
    mean_test, std_test = torch.mean(x_test), torch.std(x_test)

    #normalize
    x_train = (x_train-mean_train)/std_train
    x_test = (x_test-mean_test)/std_test

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    return (
        torch.utils.data.TensorDataset(x_train, y_train), 
        torch.utils.data.TensorDataset(x_test, y_test)
    )


@click.command()
@click.option('--load-path', default='data/raw/corruptmnist/', help='Path to load data from')
@click.option('--save-path', default='data/processed/', help='Path to save processed data')


def process_files(load_path, save_path):
    """Process files and save it in desired directory"""

    train, test = generate_mnist_dataset(load_path=load_path)
    torch.save(train, f'{save_path}/train_data.pt')
    torch.save(test, f'{save_path}/test_data.pt')


if __name__ == '__main__':
    process_files()
