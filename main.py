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

    dataset_key = data_set.lower()
    data_root = dataset_alias.get(dataset_key, data_set)
    checkpoint_suffix_map = {
        'celeba': 'celeba',
        'facescrub-full': 'facescrub',
        'pubfig83': 'pubfig83',
    }
    checkpoint_suffix = checkpoint_suffix_map.get(dataset_key)

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
        checkpoint_suffix=checkpoint_suffix,
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
        "checkpoints",
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
    sample_every: int = typer.Option(1, min=1, help="Save sample images every N epochs."),
    sample_count: int = typer.Option(10, min=1, help="Number of images to export whenever samples are saved."),
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
    critic_steps: int = typer.Option(3, min=1, help="Number of discriminator updates per generator step."),
    gp_weight: float = typer.Option(5.0, help="Gradient penalty weight (WGAN-GP). Set to 0 to disable."),
    drift: float = typer.Option(0.001, help="Drift regularization strength."),
    instance_noise: float = typer.Option(
        0.0,
        help="Stddev of Gaussian instance noise applied to real/fake images. Use 0 to disable.",
    ),
    instance_noise_decay: float = typer.Option(
        0.98,
        help="Per-epoch multiplicative decay for instance noise (1.0 keeps it constant).",
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
        sample_count=sample_count,
        split=split or None,
        device=device,
        base_dim=base_dim,
        critic_steps=critic_steps,
        gp_weight=gp_weight,
        drift=drift,
        instance_noise=instance_noise,
        instance_noise_decay=instance_noise_decay,
    )

    typer.echo(
        f"Generator training complete on {results['device']} for {results['epochs']} epochs. "
        f"Checkpoint saved to {results['last_checkpoint']}."
    )
    if results.get("last_sample"):
        typer.echo(f"Latest samples saved under {results['last_sample']}.")
    typer.echo(
        f"Final losses — D: {results['final_d_loss']:.4f}, G: {results['final_g_loss']:.4f}"
    )

@app.command()
def train_attack(
    generator_path: str = typer.Option(..., help="Path to generator checkpoint."),
    classifier_path: str = typer.Option(..., help="Path to classifier checkpoint."),
    target_class: int = typer.Option(..., help="Target class to attack."),
    generator_dim: int = typer.Option(64, help="Generator latent dimension (64 or 128)."),
    max_episodes: int = typer.Option(40000, help="Maximum number of episodes (paper default: 40000)."),
    max_step: int = typer.Option(1, help="Maximum steps per episode (paper default: 1)."),
    z_dim: int = typer.Option(100, help="Latent vector dimension (paper default: 100)."),
    alpha: float = typer.Option(0.0, help="Diversity factor alpha (paper default: 0.0)."),
    w1: float = typer.Option(2.0, help="Weight for state score (paper default: 2.0)."),
    w2: float = typer.Option(2.0, help="Weight for action score (paper default: 2.0)."),
    w3: float = typer.Option(8.0, help="Weight for distinction score (paper default: 8.0)."),

    confidence_threshold: float = typer.Option(0.95, help="Confidence threshold for early stopping (default: 0.95)."),
    seed: int = typer.Option(42, help="Random seed."),
    device: str = typer.Option("cuda", help="Device to use (cuda/cpu/mps)."),
):
    from attack.train_rl import train_attack as train_rl_attack
    
    train_rl_attack(
        generator_path=generator_path,
        classifier_path=classifier_path,
        target_class=target_class,
        generator_dim=generator_dim,
        max_episodes=max_episodes,
        max_step=max_step,
        z_dim=z_dim,
        alpha=alpha,
        w1=w1,
        w2=w2,
        w3=w3,
        confidence_threshold=confidence_threshold,
        seed=seed,
        device=device,
    )

@app.command()
def measure_accuracy(
    generator_path: str = typer.Option(..., help="Path to generator checkpoint."),
    target_classifier_path: str = typer.Option(..., help="Path to target classifier checkpoint."),
    eval_classifier_path: list[str] = typer.Option(..., help="Path(s) to evaluation classifier checkpoint(s). Can be specified multiple times for ensemble."),
    num_labels: int = typer.Option(10, help="Number of target classes to attack."),
    generator_dim: int = typer.Option(64, help="Generator latent dimension (64 or 128)."),
    max_episodes: int = typer.Option(10000, help="Episodes per target class (default: 10000 for speed)."),
    max_step: int = typer.Option(1, help="Maximum steps per episode (paper default: 1)."),
    z_dim: int = typer.Option(100, help="Latent vector dimension (paper default: 100)."),
    alpha: float = typer.Option(0.0, help="Diversity factor alpha (paper default: 0.0)."),
    w1: float = typer.Option(2.0, help="Weight for state score (paper default: 2.0)."),
    w2: float = typer.Option(2.0, help="Weight for action score (paper default: 2.0)."),
    w3: float = typer.Option(8.0, help="Weight for distinction score (paper default: 8.0)."),

    confidence_threshold: float = typer.Option(0.95, help="Confidence threshold for early stopping (default: 0.95)."),
    seed: int = typer.Option(42, help="Random seed."),
    device: str = typer.Option("cuda", help="Device to use (cuda/cpu/mps)."),
):
    """
    Measure attack accuracy by attacking multiple classes and evaluating with independent classifier(s).
    Supports ensemble evaluation when multiple eval classifier paths are provided.
    """
    from attack.evaluate import measure_attack_accuracy
    
    results = measure_attack_accuracy(
        generator_path=generator_path,
        target_classifier_path=target_classifier_path,
        eval_classifier_paths=eval_classifier_path if isinstance(eval_classifier_path, list) else [eval_classifier_path],
        num_labels=num_labels,
        generator_dim=generator_dim,
        z_dim=z_dim,
        alpha=alpha,
        max_episodes=max_episodes,
        max_step=max_step,
        w1=w1,
        w2=w2,
        w3=w3,
        confidence_threshold=confidence_threshold,
        seed=seed,
        device=device,
    )
    
    print(f"\n✅ Attack Accuracy Measurement Complete!")
    print(f"Top-1 Accuracy: {results['top1_accuracy']:.2f}%")
    print(f"Top-5 Accuracy: {results['top5_accuracy']:.2f}%")

if __name__ == "__main__":
    app()
