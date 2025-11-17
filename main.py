import typer

from dataloader import dataset_orchestration
from utils import configure_logging
from config import classifier_config

from classifier.train import train
from generator.train import train as train_generator_model
from attack.attack import RLB_MI_Attack

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
def run_rlb_mi_attack(
    generator_checkpoint: str = typer.Option(
        ...,
        "--generator",
        help="Path to trained generator checkpoint (e.g., checkpoints/generator_last.pt)",
    ),
    target_model_checkpoint: str = typer.Option(
        ...,
        "--target-model",
        help="Path to target classifier checkpoint (e.g., checkpoints/vgg16_celeba_best.pt)",
    ),
    model_name: str = typer.Option(
        "VGG16",
        "--model-name",
        help="Target model architecture: VGG16, ResNet152, or FaceNet",
    ),
    target_class: int = typer.Option(
        ...,
        "--target-class",
        help="Target class ID to reconstruct",
    ),
    num_classes: int = typer.Option(
        1000,
        "--num-classes",
        help="Number of classes in target model",
    ),
    max_episodes: int = typer.Option(
        40000,
        "--episodes",
        help="Number of training episodes for SAC agent",
    ),
    diversity_factor: float = typer.Option(
        0.0,
        "--alpha",
        "--diversity",
        help="Diversity factor (0.0 for accuracy, 0.97 for diversity)",
    ),
    latent_dim: int = typer.Option(
        100,
        "--latent-dim",
        help="Dimension of GAN latent space",
    ),
    num_images: int = typer.Option(
        1000,
        "--num-images",
        help="Number of images to generate after training",
    ),
    top_k: int = typer.Option(
        10,
        "--top-k",
        help="Number of best images to select",
    ),
    output_dir: str = typer.Option(
        "attack_results",
        "--output-dir",
        help="Directory to save attack results",
    ),
    device: str | None = typer.Option(
        None,
        "--device",
        help="Device override (cuda, mps, or cpu)",
    ),
):
    """Run RLB-MI (Reinforcement Learning-Based Black-box Model Inversion) attack."""
    import torch
    from pathlib import Path
    from generator.model import Generator
    from classifier.models import VGG16, ResNet152, FaceNet
    from torchvision import utils as vutils

    configure_logging()

    # Determine device
    if device:
        device_obj = torch.device(device)
    elif torch.cuda.is_available():
        device_obj = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device_obj = torch.device("mps")
    else:
        device_obj = torch.device("cpu")

    typer.echo(f"Using device: {device_obj}")

    # Load generator
    typer.echo(f"Loading generator from {generator_checkpoint}...")
    generator = Generator(in_dim=latent_dim, dim=64).to(device_obj)
    gen_ckpt = torch.load(generator_checkpoint, map_location=device_obj)
    generator.load_state_dict(gen_ckpt['generator'])
    generator.eval()
    typer.echo("Generator loaded successfully.")

    # Load target model
    typer.echo(f"Loading target model ({model_name}) from {target_model_checkpoint}...")
    if model_name == "VGG16":
        target_model = VGG16(num_classes)
    elif model_name == "ResNet152":
        target_model = ResNet152(num_classes)
    elif model_name in {"FaceNet", "Face.evoLVe"}:
        target_model = FaceNet(num_classes)
    else:
        typer.echo(f"Error: Unknown model name '{model_name}'", err=True)
        raise typer.Exit(1)

    target_ckpt = torch.load(target_model_checkpoint, map_location=device_obj)
    target_model.load_state_dict(target_ckpt['model'])
    target_model.to(device_obj)
    target_model.eval()
    typer.echo("Target model loaded successfully.")

    # Initialize RLB-MI attack
    typer.echo(f"Initializing RLB-MI attack for class {target_class}...")
    attack = RLB_MI_Attack(
        generator=generator,
        target_model=target_model,
        target_class=target_class,
        latent_dim=latent_dim,
        device=device_obj,
        diversity_factor=diversity_factor,
    )

    # Train agent
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    agent_path = output_path / f"agent_class_{target_class}.pt"

    typer.echo(f"\nTraining SAC agent for {max_episodes} episodes...")
    metrics = attack.train_agent(
        max_episodes=max_episodes,
        verbose=True,
        log_interval=1000,
        save_path=str(agent_path),
    )

    # Generate reconstructed images
    typer.echo(f"\nGenerating {num_images} images and selecting top {top_k}...")
    reconstructed_images, confidences, latents = attack.generate_reconstructed_images(
        num_images=num_images,
        select_best=True,
        top_k=top_k,
    )

    # Save reconstructed images
    image_grid_path = output_path / f"reconstructed_class_{target_class}.png"
    vutils.save_image(
        reconstructed_images,
        image_grid_path,
        normalize=True,
        value_range=(-1, 1),
        nrow=min(5, top_k),
    )

    # Save individual images
    images_dir = output_path / f"class_{target_class}_images"
    images_dir.mkdir(exist_ok=True)
    for idx, img in enumerate(reconstructed_images):
        vutils.save_image(
            img,
            images_dir / f"image_{idx:03d}_conf_{confidences[idx]:.4f}.png",
            normalize=True,
            value_range=(-1, 1),
        )

    # Print results
    typer.echo(f"\n{'='*60}")
    typer.echo("RLB-MI Attack Complete!")
    typer.echo(f"{'='*60}")
    typer.echo(f"Target Class: {target_class}")
    typer.echo(f"Images Generated: {num_images}")
    typer.echo(f"Top-K Selected: {top_k}")
    typer.echo(f"Average Confidence (top-{top_k}): {confidences.mean():.4f}")
    typer.echo(f"Max Confidence: {confidences.max():.4f}")
    typer.echo(f"Min Confidence: {confidences.min():.4f}")
    typer.echo(f"\nResults saved to:")
    typer.echo(f"  - Agent: {agent_path}")
    typer.echo(f"  - Image Grid: {image_grid_path}")
    typer.echo(f"  - Individual Images: {images_dir}")
    typer.echo(f"{'='*60}")

if __name__ == "__main__":
    app()
