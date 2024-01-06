import torch
import click

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@click.group()
def cli():
    """Command line interface."""
    pass

@click.command()
@click.argument("model")
@click.option('--data-path', default='data/processed', help='Path to load data from')

def evaluate(model, data_path):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    print(model)

    model = torch.load(f'models/{model}')

    test_set = torch.load(f'{data_path}/test_data.pt', map_location=device)
    test_dataloader = torch.utils.data.DataLoader(
        test_set, batch_size=64, shuffle=False
    )

    model.eval()

    test_preds = [ ]
    test_labels = [ ]

    with torch.no_grad():
        for batch in test_dataloader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            test_preds.append(y_pred.argmax(dim=1).cpu())
            test_labels.append(y.cpu())

    test_preds = torch.cat(test_preds, dim=0)
    test_labels = torch.cat(test_labels, dim=0)

    print((test_preds == test_labels).float().mean())

if __name__ == '__main__':
    evaluate()
