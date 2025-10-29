import typer

from dataloader import dataset_orchestration
from utils import configure_logging
from config import classifier_config

from classifier.train import train

app = typer.Typer()

@app.command()
def load_data():
    configure_logging()
    dataset_orchestration()

@app.command()
def hello():
    print('Hello World')

@app.command()
def train_classifier(
    data_set: str = 'celeba',
    model_name: str = 'VGG16',

    batch_size: int = None,
    learning_rate: float = None,
    optimizer_name: str = None,
    momentum: float = None,
    weight_decay: float = None,
    epoch: int = None,
    patience: int = None,
    ):

    dataset_alias = {
        'celeba': 'dataset/private/celeba',
        'facescrub-full': 'dataset/private/facescrub-full',
        'pubfig83': 'dataset/private/pubfig83',
    }

    data_root = dataset_alias.get(data_set.lower(), data_set)

    batch_size = batch_size or classifier_config['batch_size']
    learning_rate = learning_rate or classifier_config['learning_rate']
    optimizer_name = optimizer_name or classifier_config['optimizer']
    momentum = momentum if momentum is not None else classifier_config.get('momentum', 0.9)
    weight_decay = weight_decay if weight_decay is not None else classifier_config.get('weight_decay', 0.0)
    epoch = epoch or classifier_config['epoch']
    patience = patience if patience is not None else classifier_config.get('patience', 5)

    results = train(
        data_root=data_root,
        model_name=model_name,
        batch_size=batch_size,
        learning_rate=learning_rate,
        optimizer_name=optimizer_name,
        momentum=momentum,
        weight_decay=weight_decay,
        epochs=epoch,
        patience=patience,
    )

    typer.echo(f"Training complete. Best val loss {results['best_val_loss']:.4f} at epoch {results['best_epoch']}")
    typer.echo(f"Test loss {results['test_loss']:.4f}, test acc {results['test_acc']:.2f}%")
    typer.echo(f"Checkpoint saved to {results['checkpoint_path']}")

if __name__ == "__main__":
    app()
