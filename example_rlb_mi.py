"""
Example script for running RLB-MI attack.

This script demonstrates how to use the RLB-MI (Reinforcement Learning-Based
Black-box Model Inversion) attack to reconstruct private training data from
a black-box classifier.

Usage:
    python example_rlb_mi.py

Requirements:
    1. Trained generator checkpoint (from train_generator command)
    2. Trained target classifier checkpoint (from train_classifier command)
"""

import torch
import numpy as np
from pathlib import Path
from torchvision import utils as vutils

from generator.model import Generator
from classifier.models import VGG16, ResNet152, FaceNet
from attack.attack import RLB_MI_Attack


def main():
    # ===========================
    # Configuration
    # ===========================

    # Paths to checkpoints
    GENERATOR_CHECKPOINT = "checkpoints/generator_last.pt"
    TARGET_MODEL_CHECKPOINT = "checkpoints/vgg16_celeba_best.pt"

    # Model configuration
    MODEL_NAME = "VGG16"  # Options: VGG16, ResNet152, FaceNet
    NUM_CLASSES = 1000
    LATENT_DIM = 100

    # Attack configuration
    TARGET_CLASS = 0  # Class to reconstruct
    MAX_EPISODES = 40000  # Training episodes (paper uses 40,000)
    DIVERSITY_FACTOR = 0.0  # 0.0 for accuracy, 0.97 for diversity

    # Generation configuration
    NUM_IMAGES = 1000  # Number of images to generate
    TOP_K = 10  # Select top K best images

    # Output
    OUTPUT_DIR = "attack_results"

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # ===========================
    # Load Models
    # ===========================

    print(f"\nLoading generator from {GENERATOR_CHECKPOINT}...")
    generator = Generator(in_dim=LATENT_DIM, dim=64).to(device)
    gen_ckpt = torch.load(GENERATOR_CHECKPOINT, map_location=device)
    generator.load_state_dict(gen_ckpt['generator'])
    generator.eval()
    print("✓ Generator loaded successfully.")

    print(f"\nLoading target model ({MODEL_NAME}) from {TARGET_MODEL_CHECKPOINT}...")
    if MODEL_NAME == "VGG16":
        target_model = VGG16(NUM_CLASSES)
    elif MODEL_NAME == "ResNet152":
        target_model = ResNet152(NUM_CLASSES)
    elif MODEL_NAME in {"FaceNet", "Face.evoLVe"}:
        target_model = FaceNet(NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model name: {MODEL_NAME}")

    target_ckpt = torch.load(TARGET_MODEL_CHECKPOINT, map_location=device)
    target_model.load_state_dict(target_ckpt['model'])
    target_model.to(device)
    target_model.eval()
    print("✓ Target model loaded successfully.")

    # ===========================
    # Initialize Attack
    # ===========================

    print(f"\nInitializing RLB-MI attack for class {TARGET_CLASS}...")
    print(f"Parameters:")
    print(f"  - Diversity factor (α): {DIVERSITY_FACTOR}")
    print(f"  - Max episodes: {MAX_EPISODES}")
    print(f"  - Latent dimension: {LATENT_DIM}")

    attack = RLB_MI_Attack(
        generator=generator,
        target_model=target_model,
        target_class=TARGET_CLASS,
        latent_dim=LATENT_DIM,
        device=device,
        diversity_factor=DIVERSITY_FACTOR,
        # Default reward weights from paper: w1=2, w2=2, w3=8
        reward_w1=2.0,
        reward_w2=2.0,
        reward_w3=8.0,
        epsilon=1e-7,
    )

    # ===========================
    # Train Agent
    # ===========================

    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    agent_path = output_path / f"agent_class_{TARGET_CLASS}.pt"

    print(f"\n{'='*60}")
    print(f"Training SAC agent for {MAX_EPISODES} episodes...")
    print(f"{'='*60}\n")

    metrics = attack.train_agent(
        max_episodes=MAX_EPISODES,
        verbose=True,
        log_interval=1000,
        save_path=str(agent_path),
    )

    print(f"\n✓ Agent training complete!")
    print(f"  - Final avg reward (last 1000): {np.mean(metrics['episode_rewards'][-1000:]):.4f}")
    print(f"  - Agent saved to: {agent_path}")

    # ===========================
    # Generate Reconstructed Images
    # ===========================

    print(f"\n{'='*60}")
    print(f"Generating {NUM_IMAGES} images and selecting top {TOP_K}...")
    print(f"{'='*60}\n")

    reconstructed_images, confidences, latents = attack.generate_reconstructed_images(
        num_images=NUM_IMAGES,
        select_best=True,
        top_k=TOP_K,
    )

    # ===========================
    # Save Results
    # ===========================

    # Save image grid
    image_grid_path = output_path / f"reconstructed_class_{TARGET_CLASS}.png"
    vutils.save_image(
        reconstructed_images,
        image_grid_path,
        normalize=True,
        value_range=(-1, 1),
        nrow=min(5, TOP_K),
    )

    # Save individual images
    images_dir = output_path / f"class_{TARGET_CLASS}_images"
    images_dir.mkdir(exist_ok=True)
    for idx, img in enumerate(reconstructed_images):
        vutils.save_image(
            img,
            images_dir / f"image_{idx:03d}_conf_{confidences[idx]:.4f}.png",
            normalize=True,
            value_range=(-1, 1),
        )

    # Save latent vectors
    latents_path = output_path / f"latents_class_{TARGET_CLASS}.npy"
    np.save(latents_path, latents)

    # ===========================
    # Print Summary
    # ===========================

    print(f"\n{'='*60}")
    print("RLB-MI Attack Complete!")
    print(f"{'='*60}")
    print(f"\nAttack Configuration:")
    print(f"  - Target Class: {TARGET_CLASS}")
    print(f"  - Model: {MODEL_NAME}")
    print(f"  - Episodes Trained: {MAX_EPISODES}")
    print(f"  - Diversity Factor (α): {DIVERSITY_FACTOR}")

    print(f"\nGeneration Results:")
    print(f"  - Images Generated: {NUM_IMAGES}")
    print(f"  - Top-K Selected: {TOP_K}")
    print(f"  - Average Confidence: {confidences.mean():.4f}")
    print(f"  - Max Confidence: {confidences.max():.4f}")
    print(f"  - Min Confidence: {confidences.min():.4f}")

    print(f"\nResults Saved:")
    print(f"  - Agent: {agent_path}")
    print(f"  - Image Grid: {image_grid_path}")
    print(f"  - Individual Images: {images_dir}")
    print(f"  - Latent Vectors: {latents_path}")
    print(f"{'='*60}\n")

    # ===========================
    # Optional: Evaluation Metrics
    # ===========================

    print("To compute evaluation metrics (Attack Accuracy, KNN Dist, Feat Dist),")
    print("you need an evaluation classifier and target class images.")
    print("See attack/attack.py for metric computation functions:")
    print("  - compute_attack_accuracy()")
    print("  - compute_knn_distance()")
    print("  - compute_feature_distance()")


if __name__ == "__main__":
    main()
