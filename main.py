import typer

from dataloader import dataset_orchestration
from utils import configure_logging
from config import classifier_config

from classifier.train import train
from generator.train import train as train_generator_model

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
    aug_level: str | None = typer.Option(
        None,
        "--aug-level",
        help="Select augmentation intensity: weak, normal, or strong.",
        case_sensitive=False,
    ),
    saving_option: str | None = typer.Option(
        None,
        "--saving-option",
        help="Checkpoint selection metric: top1, top3, top5, or loss.",
        case_sensitive=False,
    ),
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
        augmentation_level=aug_level,
        saving_option=saving_option,
    )

    eval_label = "validation" if results["eval_split"] == "valid" else "test"
    save_opt = results.get("saving_option", "top5")
    metric_value = results.get("best_metric_value", results["best_eval_loss"])
    if save_opt == "loss":
        crit_summary = f"loss {metric_value:.4f}"
    else:
        crit_summary = f"top-{save_opt[-1]} {metric_value:.2f}%"
    typer.echo(
        f"Training complete (augmentation: {results['augmentation_level']}). "
        f"Best {eval_label} {crit_summary} at epoch {results['best_epoch']} "
        f"(top-1/top-3/top-5: {results['best_eval_top1']:.2f}% / "
        f"{results['best_eval_top3']:.2f}% / {results['best_eval_top5']:.2f}%, loss {results['best_eval_loss']:.4f})"
    )
    typer.echo(
        f"Test top-1/top-3/top-5: {results['test_top1']:.2f}% / "
        f"{results['test_top3']:.2f}% / {results['test_top5']:.2f}% "
        f"(loss {results['test_loss']:.4f})"
    )
    typer.echo(f"Checkpoint saved to {results['checkpoint_path']}")

@app.command()
def train_generator(
    data_root: str = typer.Option(
        "dataset/public/flickrfaceshq-dataset-ffhq",
        "--data-root",
        help="Root directory containing image data. If it has splits, --split will select one.",
    ),
    output_dir: str = typer.Option(
        "checkpoints/generator",
        "--output-dir",
        help="Directory where generator checkpoints and samples are stored.",
    ),
    epochs: int = typer.Option(50, min=1, help="Number of training epochs."),
    batch_size: int = typer.Option(128, min=1, help="Batch size for training."),
    latent_dim: int = typer.Option(100, min=1, help="Dimension of the latent noise vector."),
    learning_rate: float = typer.Option(
        2e-4,
        "--lr",
        "--learning-rate",
        help="Learning rate for both generator and discriminator.",
    ),
    beta1: float = typer.Option(0.5, help="Beta1 for Adam optimizer."),
    beta2: float = typer.Option(0.999, help="Beta2 for Adam optimizer."),
    num_workers: int = typer.Option(4, min=0, help="Number of data loading workers."),
    seed: int = typer.Option(42, help="Random seed. Set to a negative value to disable seeding."),
    sample_every: int = typer.Option(5, min=1, help="Save sample images every N epochs."),
    split: str = typer.Option(
        "train",
        "--split",
        help="Dataset split subdirectory to use. Leave blank if data_root already points to images.",
    ),
    device: str | None = typer.Option(
        None,
        "--device",
        help="Device override, e.g., 'cuda', 'mps', or 'cpu'. Defaults to auto-detection.",
    ),
    base_dim: int = typer.Option(
        64,
        "--base-dim",
        min=8,
        help="Base channel dimension for generator and discriminator.",
    ),
):
    configure_logging()

    seed_value = seed if seed >= 0 else None
    results = train_generator_model(
        data_root=data_root,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        latent_dim=latent_dim,
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        num_workers=num_workers,
        seed=seed_value,
        sample_every=sample_every,
        split=split or None,
        device=device,
        base_dim=base_dim,
    )

    typer.echo(
        f"Generator training complete on {results['device']} for {results['epochs']} epochs. "
        f"Checkpoint saved to {results['last_checkpoint']}."
    )
    if results.get("last_sample"):
        typer.echo(f"Latest sample grid saved to {results['last_sample']}.")
    typer.echo(
        f"Final losses — D: {results['final_d_loss']:.4f}, G: {results['final_g_loss']:.4f}"
    )

if __name__ == "__main__":
    app()
