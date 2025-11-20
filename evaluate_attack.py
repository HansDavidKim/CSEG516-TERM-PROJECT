import typer
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from typing import List, Optional

from generator.model import Generator
from classifier.models import VGG16, ResNet152, FaceNet
from attack.attack import RLB_MI_Attack, compute_attack_accuracy, compute_knn_distance, compute_feature_distance
from utils import configure_logging
from dataloader import get_dataloader
from torchvision import utils as vutils

app = typer.Typer()

@app.command()
def evaluate_attack(
    target_model_checkpoint: str = typer.Option(..., "--target-model", help="Path to target classifier checkpoint"),
    eval_model_checkpoint: str = typer.Option(..., "--eval-model", help="Path to evaluation classifier checkpoint"),
    generator_checkpoint: str = typer.Option(..., "--generator", help="Path to trained generator checkpoint"),
    data_root: str = typer.Option("dataset/private/celeba", "--data-root", help="Path to dataset root for real images"),
    target_model_name: str = typer.Option("VGG16", "--target-model-name", help="Target model architecture"),
    eval_model_name: str = typer.Option("ResNet152", "--eval-model-name", help="Evaluation model architecture"),
    classes: str = typer.Option("0-9", "--classes", help="Range of classes to evaluate (e.g., '0-9' or '0,1,5')"),
    num_classes: int = typer.Option(1000, "--num-classes", help="Number of classes in models"),
    max_episodes: int = typer.Option(1000, "--episodes", help="Number of training episodes per class"),
    num_images: int = typer.Option(1000, "--num-images", help="Number of images to generate per class"),
    top_k: int = typer.Option(10, "--top-k", help="Number of best images to select for evaluation"),
    output_dir: str = typer.Option("evaluation_results", "--output-dir", help="Directory to save results"),
    device: str = typer.Option(None, "--device", help="Device override"),
    batch_size: int = typer.Option(32, "--batch-size", help="Batch size for dataloader"),
):
    """
    Evaluate RLB-MI attack on multiple classes.
    Computes Attack Accuracy, KNN Distance, and Feature Distance.
    """
    configure_logging()
    
    # Device setup
    if device:
        device_obj = torch.device(device)
    elif torch.cuda.is_available():
        device_obj = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device_obj = torch.device("mps")
    else:
        device_obj = torch.device("cpu")
    
    print(f"Using device: {device_obj}")

    # Parse classes
    target_classes = []
    if '-' in classes:
        start, end = map(int, classes.split('-'))
        target_classes = list(range(start, end + 1))
    else:
        target_classes = [int(c) for c in classes.split(',')]
    
    print(f"Evaluating on {len(target_classes)} classes: {target_classes}")

    # Load Models
    print("Loading models...")
    
    # Generator
    generator = Generator(in_dim=100, dim=64).to(device_obj)
    gen_ckpt = torch.load(generator_checkpoint, map_location=device_obj)
    generator.load_state_dict(gen_ckpt['generator'])
    generator.eval()

    # Target Model
    def load_classifier(name, checkpoint, num_classes):
        if name == "VGG16":
            model = VGG16(num_classes)
        elif name == "ResNet152":
            model = ResNet152(num_classes)
        elif name in {"FaceNet", "Face.evoLVe"}:
            model = FaceNet(num_classes)
        else:
            raise ValueError(f"Unknown model: {name}")
        
        ckpt = torch.load(checkpoint, map_location=device_obj)
        model.load_state_dict(ckpt['model'])
        model.to(device_obj)
        model.eval()
        return model

    target_model = load_classifier(target_model_name, target_model_checkpoint, num_classes)
    eval_model = load_classifier(eval_model_name, eval_model_checkpoint, num_classes)
    
    print("Models loaded.")

    # Prepare Dataset for Real Images (for KNN/Feature Dist)
    # We need a way to get images for specific classes. 
    # Assuming standard folder structure or using the dataloader.
    # For simplicity, we'll use the dataloader and filter.
    # NOTE: This might be slow if scanning the whole dataset. 
    # Ideally, we should have a way to get class-specific images efficiently.
    # For now, we will load the test set.
    
    print("Loading dataset for evaluation metrics...")
    # Using a small batch size to avoid OOM, we will iterate to find class images
    # This part assumes the dataset is organized or we can iterate through it.
    # Let's try to use the existing dataloader but we might need to be careful.
    # The existing dataloader returns a wrapper.
    
    # Actually, for KNN and Feature Dist, we need REAL images of the target class.
    # If we can't easily get them, we might skip those metrics or warn.
    # Let's assume we can get them via a helper or just iterating.
    
    # To avoid complexity in this script, let's define a helper to get images for a class
    # by iterating through the validation/test loader.
    
    from dataloader import get_dataloader
    # We use 'valid' or 'test' split
    loader = get_dataloader(data_root, batch_size=batch_size, resize_size=64, split='test') 
    # Note: resize_size=64 matches standard CelebA/GAN setup usually. 
    # Check generator output size? Generator usually outputs 64x64 or 128x128.
    # The attack.py doesn't specify size, but VGG16/ResNet usually expect 224 or similar, 
    # but for these research datasets (CelebA), 64x64 is common for GANs.
    # Let's assume 64x64 for now as per Generator default.
    
    # We'll cache class images to avoid re-reading for every class if possible, 
    # or just read on demand. Since we have a list of classes, let's pre-collect if memory allows,
    # or just iterate for each class (slow).
    # Better: Iterate once through the loader and collect images for the target classes.
    
    class_images = {c: [] for c in target_classes}
    print("Collecting real images for target classes from test set...")
    
    max_real_images = 100 # Limit real images for KNN to save time/memory
    
    for batch_idx, (images, labels) in enumerate(tqdm(loader)):
        # labels might be one-hot or indices. Assuming indices based on typical PyTorch.
        # If one-hot, convert.
        if labels.dim() > 1 and labels.shape[1] > 1:
            labels = labels.argmax(dim=1)
            
        for img, label in zip(images, labels):
            lbl = label.item()
            if lbl in class_images and len(class_images[lbl]) < max_real_images:
                class_images[lbl].append(img)
        
        # Check if we have enough for all classes
        if all(len(imgs) >= max_real_images for imgs in class_images.values()):
            break
            
    # Convert lists to tensors
    for c in class_images:
        if len(class_images[c]) > 0:
            class_images[c] = torch.stack(class_images[c]).to(device_obj)
        else:
            print(f"Warning: No real images found for class {c}. KNN/Feat Dist will be skipped.")

    # Results storage
    results = []
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Main Loop
    for target_class in tqdm(target_classes, desc="Evaluating Classes"):
        print(f"\n--- Class {target_class} ---")
        
        # Initialize Attack
        attack = RLB_MI_Attack(
            generator=generator,
            target_model=target_model,
            target_class=target_class,
            latent_dim=100,
            device=device_obj,
            diversity_factor=0.0 # Focus on accuracy for evaluation? Or diversity? Paper uses 0.0 for Acc.
        )
        
        # Train Agent
        # We can save the agent if needed, but for eval we might just want the images.
        agent_path = output_path / f"agent_class_{target_class}.pt"
        attack.train_agent(
            max_episodes=max_episodes,
            verbose=False,
            save_path=str(agent_path)
        )
        
        # Generate Images
        recon_images, confidences, _ = attack.generate_reconstructed_images(
            num_images=num_images,
            select_best=True,
            top_k=top_k
        )
        
        # Compute Metrics
        
        # 1. Attack Accuracy
        acc = compute_attack_accuracy(recon_images, target_class, eval_model, device_obj)
        print(f"Attack Accuracy: {acc:.2f}%")
        
        # 2. KNN & Feature Distance (if real images available)
        knn_dist = float('nan')
        feat_dist = float('nan')
        
        if target_class in class_images and isinstance(class_images[target_class], torch.Tensor):
            real_imgs = class_images[target_class]
            knn_dist = compute_knn_distance(recon_images, real_imgs, eval_model, device_obj)
            feat_dist = compute_feature_distance(recon_images, real_imgs, eval_model, device_obj)
            print(f"KNN Dist: {knn_dist:.4f}")
            print(f"Feat Dist: {feat_dist:.4f}")
        
        results.append({
            "class_id": target_class,
            "attack_acc": acc,
            "knn_dist": knn_dist,
            "feat_dist": feat_dist,
            "avg_confidence": confidences.mean().item()
        })
        
        # Save images for this class
        images_dir = output_path / f"class_{target_class}_images"
        images_dir.mkdir(exist_ok=True)
        for idx, img in enumerate(recon_images):
            vutils.save_image(
                img,
                images_dir / f"image_{idx:03d}_conf_{confidences[idx]:.4f}.png",
                normalize=True,
                value_range=(-1, 1),
            )
            
    # Save CSV
    df = pd.DataFrame(results)
    csv_path = output_path / "evaluation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nEvaluation complete. Results saved to {csv_path}")
    
    # Print Summary
    print("\nSummary:")
    print(df.mean())

if __name__ == "__main__":
    app()
