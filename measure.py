#!/usr/bin/env python3
"""
Comprehensive Evaluation Script for Model Inversion Attack
Measures three key metrics from RLB-MI paper:
1. Attack Accuracy (Top-1, Top-5)
2. KNN Distance (feature space similarity to private training data)
3. FID (Fréchet Inception Distance - perceptual quality of generated images)

Usage:
    python measure.py --generator-path <path> --target-classifier <path> --eval-classifier <path> --private-data <path>
"""

import os
import csv
import time
import argparse
import random
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import numpy as np
from scipy import linalg

from attack.sac_agent import Agent
from attack.utils import load_generator, load_classifier
from attack.attack_utils import compute_reward, evaluate_confidence


class PrivateDataset(Dataset):
    """Dataset loader for private training data organized by identity folders."""
    
    def __init__(self, data_root: str, transform=None, max_per_class: int = None):
        """
        Args:
            data_root: Root directory containing train/test splits with identity folders
            transform: Image transformations
            max_per_class: Maximum images to load per class (None = load all)
        """
        self.data_root = Path(data_root)
        self.transform = transform
        self.samples = []
        self.class_to_samples = {}  # Map class_id -> list of (image_path, class_id)
        
        # Load from train split (private training data)
        train_dir = self.data_root / "train"
        if not train_dir.exists():
            raise ValueError(f"Train directory not found: {train_dir}")
        
        # Iterate through identity folders
        for identity_dir in sorted(train_dir.iterdir()):
            if not identity_dir.is_dir():
                continue
            
            class_id = int(identity_dir.name)
            class_samples = []
            
            # Load images from this identity folder
            for img_path in sorted(identity_dir.glob("*.jpg")):
                class_samples.append((str(img_path), class_id))
            
            # Apply max_per_class limit if specified
            if max_per_class and len(class_samples) > max_per_class:
                class_samples = class_samples[:max_per_class]
            
            self.class_to_samples[class_id] = class_samples
            self.samples.extend(class_samples)
        
        print(f"Loaded {len(self.samples)} private training images from {len(self.class_to_samples)} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, class_id = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, class_id
    
    def get_class_samples(self, class_id: int) -> List[Tuple[str, int]]:
        """Get all samples for a specific class."""
        return self.class_to_samples.get(class_id, [])


def extract_features(model, images, device):
    """
    Extract feature embeddings from classifier.
    
    Args:
        model: Classifier model
        images: Batch of images (B, C, H, W)
        device: Device to run on
    
    Returns:
        features: Feature embeddings (B, feature_dim)
    """
    with torch.no_grad():
        features, _ = model(images.to(device))
    return features


def compute_knn_distance(
    generated_features: torch.Tensor,
    private_features: torch.Tensor,
    k: int = 5
) -> float:
    """
    Compute average K-Nearest Neighbor distance in feature space.
    
    Args:
        generated_features: Features of generated images (N, feature_dim)
        private_features: Features of private training images (M, feature_dim)
        k: Number of nearest neighbors
    
    Returns:
        avg_knn_dist: Average distance to k nearest neighbors
    """
    # Normalize features
    generated_features = F.normalize(generated_features, p=2, dim=1)
    private_features = F.normalize(private_features, p=2, dim=1)
    
    # Compute pairwise distances (N, M)
    distances = torch.cdist(generated_features, private_features, p=2)
    
    # Get k smallest distances for each generated image
    knn_distances, _ = torch.topk(distances, k=min(k, distances.size(1)), largest=False, dim=1)
    
    # Average over k neighbors and all generated images
    avg_knn_dist = knn_distances.mean().item()
    
    return avg_knn_dist


class InceptionV3FeatureExtractor(nn.Module):
    """InceptionV3 feature extractor for FID calculation."""
    
    def __init__(self, device):
        super().__init__()
        try:
            from torchvision.models import inception_v3, Inception_V3_Weights
            # Load pretrained InceptionV3
            self.inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
            self.inception.eval()
            
            # Use CUDA if available, otherwise CPU (MPS causes segfault with InceptionV3)
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                print(f"  (InceptionV3 using CUDA)")
            else:
                self.device = torch.device('cpu')
                print(f"  (InceptionV3 using CPU for stability)")
            self.inception.to(self.device)
            
            # Remove the final fc layer to get 2048-dim features
            self.inception.fc = nn.Identity()
            
        except ImportError:
            raise ImportError("torchvision is required for FID calculation. Install with: pip install torchvision")
    
    def forward(self, x):
        """
        Extract features from images.
        Args:
            x: Images tensor (B, 3, H, W) in range [0, 1]
        Returns:
            features: Feature vectors (B, 2048)
        """
        # Move input to CPU (InceptionV3 runs on CPU for stability)
        x = x.to(self.device)
        
        # Resize to 299x299 for InceptionV3
        if x.shape[-1] != 299 or x.shape[-2] != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Normalize to [-1, 1] as expected by InceptionV3
        x = 2 * x - 1
        
        with torch.no_grad():
            # InceptionV3 with fc=Identity will output (B, 2048)
            features = self.inception(x)
        
        return features


def compute_fid(
    generated_features: torch.Tensor,
    private_features: torch.Tensor
) -> float:
    """
    Compute Fréchet Inception Distance (FID) between generated and private images.
    
    FID measures the distance between two distributions of images in feature space.
    Lower FID indicates better quality and more realistic generated images.
    
    Args:
        generated_features: Features of generated images (N, feature_dim)
        private_features: Features of private training images (M, feature_dim)
    
    Returns:
        fid_score: FID score (lower is better)
    """
    # Convert to numpy
    gen_features = generated_features.cpu().numpy()
    real_features = private_features.cpu().numpy()
    
    # Calculate mean and covariance
    mu_gen = np.mean(gen_features, axis=0)
    mu_real = np.mean(real_features, axis=0)
    
    sigma_gen = np.cov(gen_features, rowvar=False)
    sigma_real = np.cov(real_features, rowvar=False)
    
    # Calculate FID
    diff = mu_gen - mu_real
    
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma_gen @ sigma_real, disp=False)
    
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff @ diff + np.trace(sigma_gen + sigma_real - 2 * covmean)
    
    return float(fid)


def attack_single_class(
    target_class: int,
    generator,
    target_classifier,
    target_arc_head,
    eval_classifier,
    eval_arc_head,
    z_dim: int,
    max_episodes: int,
    max_step: int,
    alpha: float,
    w1: float,
    w2: float,
    w3: float,
    confidence_threshold: float,
    seed: int,
    device,
    mean,
    std
) -> Tuple[torch.Tensor, float]:
    """
    Attack a single target class and return best generated image.
    Best image is selected based on TARGET classifier confidence (goal is to fool target).
    Eval classifier is used to measure transferability of the attack.
    
    Returns:
        best_image: Best generated image tensor (selected by target_classifier)
        best_score: Best target confidence score achieved
    """
    # Initialize RL agent
    agent = Agent(
        state_size=z_dim,
        action_size=z_dim,
        random_seed=seed,
        hidden_size=256,
        action_prior="uniform",
        device=str(device)
    )
    
    best_target_score = 0
    best_image = None
    
    # Train attack
    pbar = tqdm(range(1, max_episodes + 1), desc=f"Class {target_class}", leave=False, mininterval=1.0)
    for i_episode in pbar:
        # Generate initial latent
        z = torch.randn(1, z_dim, device=device)
        state = z.cpu().numpy().copy()
        
        for t in range(max_step):
            # Get action
            action = agent.act(state)
            
            # Update latent
            z = alpha * z + (1 - alpha) * action.clone().detach().reshape((1, len(action))).to(device)
            next_state = z.cpu().numpy().copy()
            
            # Generate images
            state_image = generator(z).detach()
            action_image = generator(action.clone().detach().reshape((1, len(action))).to(device)).detach()
            
            # Normalize
            state_image_norm = (state_image + 1) / 2
            state_image_norm = (state_image_norm - mean) / std
            action_image_norm = (action_image + 1) / 2
            action_image_norm = (action_image_norm - mean) / std
            
            # Get logits
            features_z, _ = target_classifier(state_image_norm)
            features_a, _ = target_classifier(action_image_norm)
            
            if target_arc_head is not None:
                state_output = target_arc_head.inference_logits(features_z)
                action_output = target_arc_head.inference_logits(features_a)
            else:
                _, state_output = target_classifier(state_image_norm)
                _, action_output = target_classifier(action_image_norm)
            
            # Calculate reward
            reward = compute_reward(
                state_output, action_output, target_class,
                w1=w1, w2=w2, w3=w3
            )
            
            done = (t == max_step - 1)
            agent.step(state, action, reward, next_state, done, t)
            state = next_state
        
        # Evaluate current policy with target classifier only
        # (eval classifier is used only for final measurement, not during training)
        with torch.no_grad():
            z_test = torch.randn(1, z_dim, device=device)
            
            # Generate image through latent path (max_step iterations)
            for t in range(max_step):
                state_test = z_test.cpu().numpy()
                action_test = agent.act(state_test)
                z_test = alpha * z_test + (1 - alpha) * action_test.clone().detach().reshape((1, len(action_test))).to(device)
            
            test_image = generator(z_test).detach()
            test_image_norm = (test_image + 1) / 2
            test_image_norm = (test_image_norm - mean) / std
            
            # Evaluate with target classifier only (for training signal)
            if target_arc_head is not None:
                test_features, _ = target_classifier(test_image_norm)
                test_output = target_arc_head.inference_logits(test_features)
            else:
                _, test_output = target_classifier(test_image_norm)
            
            target_score = evaluate_confidence(test_output, target_class)
            
            # Get top-1 predictions for early stopping
            target_probs = F.softmax(test_output, dim=-1)
            
            # Select best image based on TARGET classifier (goal is to fool target!)
            if target_score > best_target_score:
                best_target_score = target_score
                best_image = test_image
            
            pbar.set_postfix({
                'target': f'{target_score:.4f}',
                'best': f'{best_target_score:.4f}'
            })
            
            # Early stopping: when target classifier has >= 80% confidence on target class
            target_conf = target_probs[0, target_class].item()
            if target_conf >= 0.80:
                print(f"\n✓ Early stop! Target classifier confidence on target class: {target_conf:.2%}")
                break
    
    return best_image, best_target_score


def measure_all_metrics(
    generator_path: str,
    target_classifier_path: str,
    eval_classifier_path: str,
    private_data_path: str,
    num_labels: int = 10,
    generator_dim: int = 64,
    z_dim: int = 100,
    alpha: float = 0.0,
    max_episodes: int = 10000,
    max_step: int = 1,
    w1: float = 2.0,
    w2: float = 2.0,
    w3: float = 8.0,
    confidence_threshold: float = 0.95,
    knn_k: int = 5,
    seed: int = 42,
    device: str = "cuda",
    arcface_scale: float = 16.0,
):
    """
    Measure all three evaluation metrics for model inversion attack.
    
    Metrics (from RLB-MI paper):
    1. Attack Accuracy: Top-1 and Top-5 classification accuracy on evaluation network
    2. KNN Distance: Average distance to K nearest neighbors in private training set
    3. FID: Fréchet Inception Distance - perceptual quality of generated images
    """
    # Setup device (CUDA priority)
    if device == "cuda" and torch.cuda.is_available():
        device_obj = torch.device("cuda")
    elif device == "mps" and torch.backends.mps.is_available():
        device_obj = torch.device("mps")
    else:
        device_obj = torch.device("cpu")
    
    print(f"Using {device_obj} device")
    
    # Extract dataset and model names for organized output
    import re
    dataset_name = Path(private_data_path).name  # e.g., "celeba" or "facescrub-full"
    target_model_match = re.search(r'(vgg16|resnet152|facenet)', target_classifier_path.lower())
    target_model_name = target_model_match.group(1) if target_model_match else "unknown"
    
    # Create output directory
    output_dir = Path(f"generated_images/{dataset_name}_{target_model_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    print(f"\n{'='*80}")
    print("Model Inversion Attack - Comprehensive Evaluation (RLB-MI Metrics)")
    print(f"{'='*80}")
    print(f"Target Classes: {num_labels}")
    print(f"Episodes per class: {max_episodes}")
    print(f"KNN K: {knn_k}")
    print(f"{'='*80}\n")
    
    # Fix random seeds
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Load models
    print("Loading models...")
    generator = load_generator(generator_path, device_obj, dim=generator_dim)
    generator.eval()
    
    target_classifier, target_arc_head = load_classifier(target_classifier_path, device_obj, arcface_scale=arcface_scale)
    target_classifier.eval()
    
    eval_classifier, eval_arc_head = load_classifier(eval_classifier_path, device_obj, arcface_scale=arcface_scale)
    eval_classifier.eval()
    
    # Load InceptionV3 for FID calculation
    print("Loading InceptionV3 for FID calculation...")
    inception_extractor = InceptionV3FeatureExtractor(device_obj)
    print("✓ Models loaded\n")
    
    # Load private training data
    print("Loading private training data...")
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    
    private_dataset = PrivateDataset(
        data_root=private_data_path,
        transform=transform,
        max_per_class=100  # Limit for memory efficiency
    )
    print("✓ Private data loaded\n")
    
    # Normalization for classifier
    mean = torch.tensor([0.5177433, 0.4284404, 0.3802497], device=device_obj).view(1, 3, 1, 1)
    std = torch.tensor([0.3042383, 0.2845056, 0.2826854], device=device_obj).view(1, 3, 1, 1)
    
    
    # Results tracking
    results = []
    generated_images_all = []
    generated_eval_features_all = []  # For KNN distance (using eval classifier features)
    private_eval_features_by_class = {}  # Eval classifier features for KNN
    
    # Metric accumulators
    top1_correct = 0
    top5_correct = 0
    total = 0
    
    # Attack each target class
    all_classes = list(range(num_labels))
    
    for target_class in tqdm(all_classes, desc="Attacking classes"):
        print(f"\n{'='*60}")
        print(f"Target Class: {target_class}")
        print(f"{'='*60}")
        
        # Attack this class
        best_image, best_score = attack_single_class(
            target_class=target_class,
            generator=generator,
            target_classifier=target_classifier,
            target_arc_head=target_arc_head,
            eval_classifier=eval_classifier,
            eval_arc_head=eval_arc_head,
            z_dim=z_dim,
            max_episodes=max_episodes,
            max_step=max_step,
            alpha=alpha,
            w1=w1,
            w2=w2,
            w3=w3,
            confidence_threshold=confidence_threshold,
            seed=seed,
            device=device_obj,
            mean=mean,
            std=std
        )
        
        # Normalize generated image for evaluation
        generated_norm = (best_image + 1) / 2
        generated_norm = (generated_norm - mean) / std
        
        # Extract features from generated image (using eval classifier for KNN)
        with torch.no_grad():
            gen_eval_features = extract_features(eval_classifier, generated_norm, device_obj)
            generated_eval_features_all.append(gen_eval_features.cpu())
            
            # Note: InceptionV3 features for FID will be computed at the end from saved images
            generated_images_all.append(best_image.cpu())
            
            # Get predictions from BOTH target and eval classifiers
            # Target classifier
            if target_arc_head is not None:
                target_features, _ = target_classifier(generated_norm)
                target_output = target_arc_head.inference_logits(target_features)
            else:
                _, target_output = target_classifier(generated_norm)
            
            target_probs = F.softmax(target_output, dim=-1)[0]
            target_top1_pred = torch.argmax(target_probs).item()
            
            # Eval classifier
            if eval_arc_head is not None:
                eval_output = eval_arc_head.inference_logits(gen_eval_features)
            else:
                _, eval_output = eval_classifier(generated_norm)
            
            probs = F.softmax(eval_output, dim=-1)[0]
            
            # Get target class probability for logging
            target_prob = probs[target_class].item()
            
            # Get top-1 prediction
            top1_pred = torch.argmax(probs).item()
            top5_probs, top5_indices = torch.topk(probs, min(5, len(probs)))
            top5_preds = top5_indices.tolist()
            
            # Success = BOTH classifiers predict target class as top-1
            target_fooled = (target_top1_pred == target_class)
            eval_fooled = (top1_pred == target_class)
            
            is_top1_correct = target_fooled and eval_fooled
            is_top5_correct = (target_class in top5_preds)
        
        # Load and extract features from private training data for this class
        if target_class not in private_eval_features_by_class:
            class_samples = private_dataset.get_class_samples(target_class)
            if class_samples:
                class_images = []
                for img_path, _ in class_samples:
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = transform(img)  # This gives [0, 1] range
                    class_images.append(img_tensor)
                
                class_images = torch.stack(class_images).to(device_obj)  # Shape: (N, 3, 64, 64), range [0, 1]
                
                # Extract eval classifier features for KNN (needs normalization)
                class_images_norm = (class_images - mean) / std
                with torch.no_grad():
                    class_eval_features = extract_features(eval_classifier, class_images_norm, device_obj)
                    private_eval_features_by_class[target_class] = class_eval_features.cpu()
            else:
                print(f"Warning: No private training data found for class {target_class}")
                private_eval_features_by_class[target_class] = None
        
        # Compute KNN distance for this class (using eval classifier features)
        if private_eval_features_by_class[target_class] is not None:
            knn_dist = compute_knn_distance(
                gen_eval_features.cpu(),
                private_eval_features_by_class[target_class],
                k=knn_k
            )
        else:
            knn_dist = float('inf')
        
        # Note: FID will be computed at the end using all generated images
        
        # Update counters
        if is_top1_correct:
            top1_correct += 1
        if is_top5_correct:
            top5_correct += 1
        total += 1
        
        # Store result
        result = {
            'target_class': target_class,
            'target_confidence': best_score,
            'eval_top1_pred': top1_pred,
            'eval_top1_prob': probs[top1_pred].item(),
            'eval_top5_preds': top5_preds,
            'is_top1_correct': is_top1_correct,
            'is_top5_correct': is_top5_correct,
            'knn_distance': knn_dist,
        }
        results.append(result)
        
        # Save generated image to organized directory
        # (output_dir was set earlier based on dataset and model names)
        
        # Convert image from [-1, 1] to [0, 255]
        img_to_save = (best_image[0].cpu() + 1) / 2  # [0, 1]
        img_to_save = (img_to_save * 255).clamp(0, 255).byte()
        img_to_save = img_to_save.permute(1, 2, 0).numpy()  # CHW -> HWC
        
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(img_to_save.astype('uint8'), 'RGB')
        
        # Save with descriptive filename
        img_filename = output_dir / f"class_{target_class:03d}_target_{best_score:.4f}_eval_{probs[top1_pred].item():.4f}.png"
        pil_img.save(img_filename)
        
        print(f"\n--- Results for Target Class {target_class} ---")
        print(f"Target Top-1 Pred: {target_top1_pred} (target: {target_class}) - Fooled: {target_fooled}")
        print(f"Eval Top-1 Pred: {top1_pred} (target: {target_class}) - Fooled: {eval_fooled}")
        print(f"Overall Success: {is_top1_correct}")
        print(f"Eval Top-5: {top5_preds} - Correct: {is_top5_correct}")
        print(f"KNN Distance (k={knn_k}): {knn_dist:.4f}")
        print(f"Saved image: {img_filename}")
        print(f"\nCurrent Success Rate: Top-1: {top1_correct}/{total} = {100*top1_correct/total:.2f}%, Top-5: {top5_correct}/{total} = {100*top5_correct/total:.2f}%")
    
    # Compute final metrics
    top1_accuracy = 100 * top1_correct / total
    top5_accuracy = 100 * top5_correct / total
    
    # Average KNN distance
    valid_knn_dists = [r['knn_distance'] for r in results if r['knn_distance'] != float('inf')]
    avg_knn_dist = np.mean(valid_knn_dists) if valid_knn_dists else float('inf')
    
    
    # Compute FID using saved generated images (loaded from disk for stability)
    print(f"\nComputing FID (Fréchet Inception Distance)...")
    print("  Loading saved images from disk...")
    fid_score = float('inf')
    try:
        from PIL import Image as PILImage
        from torchvision import transforms as T
        
        # Load generated images from disk (from output_dir)
        gen_transform = T.ToTensor()
        gen_images_for_fid = []
        
        for img_path in sorted(output_dir.glob("class_*.png")):
            img = PILImage.open(img_path).convert('RGB')
            gen_images_for_fid.append(gen_transform(img))
        
        if gen_images_for_fid:
            gen_images_tensor = torch.stack(gen_images_for_fid)
            print(f"  Loaded {len(gen_images_tensor)} generated images")
            
            # Load private images for comparison
            private_images_for_fid = []
            for class_id in range(num_labels):
                class_samples = private_dataset.get_class_samples(class_id)
                if class_samples:
                    for img_path, _ in class_samples[:2]:  # 2 per class
                        img = PILImage.open(img_path).convert('RGB')
                        img = img.resize((64, 64))
                        private_images_for_fid.append(gen_transform(img))
            
            if private_images_for_fid:
                private_images_tensor = torch.stack(private_images_for_fid)
                print(f"  Loaded {len(private_images_tensor)} private images")
                
                # Extract features and compute FID (InceptionV3 on CPU)
                print("  Extracting InceptionV3 features...")
                gen_features = inception_extractor(gen_images_tensor)
                private_features = inception_extractor(private_images_tensor)
                
                fid_score = compute_fid(gen_features, private_features)
                print(f"  FID Score: {fid_score:.4f}")
    except Exception as e:
        print(f"  FID computation failed: {e}")
        fid_score = float('inf')
    
    # Print final results
    print(f"\n{'='*80}")
    print("FINAL EVALUATION RESULTS (RLB-MI Metrics)")
    print(f"{'='*80}")
    print(f"Total Classes Attacked: {total}")
    print(f"\n[1] Attack Accuracy (Evaluation Classifier)")
    print(f"    Top-1 Success: {top1_correct}/{total} = {top1_accuracy:.2f}%")
    print(f"    Top-5 Success: {top5_correct}/{total} = {top5_accuracy:.2f}%")
    print(f"\n[2] KNN Distance (k={knn_k})")
    print(f"    Average Distance: {avg_knn_dist:.4f}")
    print(f"    (Lower is better - indicates similarity to private training data)")
    print(f"\n[3] FID (Fréchet Inception Distance)")
    print(f"    FID Score: {fid_score:.4f}")
    print(f"    (Lower is better - indicates perceptual quality and realism)")
    print(f"{'='*80}\n")
    
    # Save results to CSV
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_dir = Path("metric_report")
    report_dir.mkdir(exist_ok=True)
    report_path = report_dir / f"comprehensive_metrics_{timestamp}.csv"
    
    print(f"Saving detailed results to {report_path}...")
    
    with open(report_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        # Header
        writer.writerow([
            "Target Class",
            "Target Confidence",
            "Eval Top-1 Pred", "Eval Top-1 Prob", "Eval Top-1 Correct",
            "Eval Top-5 Preds", "Eval Top-5 Correct",
            "KNN Distance",
            "Generator Path", "Target Classifier Path", "Eval Classifier Path", "Private Data Path"
        ])
        
        # Rows
        for res in results:
            writer.writerow([
                res['target_class'],
                f"{res['target_confidence']:.4f}",
                res['eval_top1_pred'], f"{res['eval_top1_prob']:.4f}", res['is_top1_correct'],
                res['eval_top5_preds'], res['is_top5_correct'],
                f"{res['knn_distance']:.4f}",
                generator_path, target_classifier_path, eval_classifier_path, private_data_path
            ])
        
        # Summary row
        writer.writerow([])
        writer.writerow(["SUMMARY"])
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Attack Accuracy (Top-1)", f"{top1_accuracy:.2f}%"])
        writer.writerow(["Attack Accuracy (Top-5)", f"{top5_accuracy:.2f}%"])
        writer.writerow(["Average KNN Distance", f"{avg_knn_dist:.4f}"])
        writer.writerow(["FID Score", f"{fid_score:.4f}"])
    
    print(f"✓ Report saved successfully!\n")
    
    return {
        'top1_accuracy': top1_accuracy,
        'top5_accuracy': top5_accuracy,
        'avg_knn_distance': avg_knn_dist,
        'fid_score': fid_score,
        'results': results,
        'report_path': str(report_path)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Model Inversion Attack Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("--generator-path", type=str, required=True,
                        help="Path to generator checkpoint")
    parser.add_argument("--target-classifier", type=str, required=True,
                        help="Path to target classifier checkpoint")
    parser.add_argument("--eval-classifier", type=str, required=True,
                        help="Path to evaluation classifier checkpoint (e.g., evoLVe)")
    parser.add_argument("--private-data", type=str, required=True,
                        help="Path to private training data directory")
    
    # Optional arguments
    parser.add_argument("--num-labels", type=int, default=10,
                        help="Number of target classes to attack")
    parser.add_argument("--generator-dim", type=int, default=64,
                        help="Generator base dimension")
    parser.add_argument("--z-dim", type=int, default=100,
                        help="Latent vector dimension")
    parser.add_argument("--alpha", type=float, default=0.0,
                        help="Diversity factor")
    parser.add_argument("--max-episodes", type=int, default=10000,
                        help="Maximum episodes per class")
    parser.add_argument("--max-step", type=int, default=1,
                        help="Maximum steps per episode")
    parser.add_argument("--w1", type=float, default=2.0,
                        help="Weight for state score")
    parser.add_argument("--w2", type=float, default=2.0,
                        help="Weight for action score")
    parser.add_argument("--w3", type=float, default=8.0,
                        help="Weight for distinction score")
    parser.add_argument("--confidence-threshold", type=float, default=0.95,
                        help="Confidence threshold for early stopping")
    parser.add_argument("--knn-k", type=int, default=5,
                        help="K for KNN distance calculation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "mps", "cpu"],
                        help="Device to use")
    parser.add_argument("--arcface-scale", type=float, default=16.0,
                        help="ArcFace scale factor for temperature scaling (default 16.0, original 64.0)")
    
    args = parser.parse_args()
    
    # Run evaluation
    results = measure_all_metrics(
        generator_path=args.generator_path,
        target_classifier_path=args.target_classifier,
        eval_classifier_path=args.eval_classifier,
        private_data_path=args.private_data,
        num_labels=args.num_labels,
        generator_dim=args.generator_dim,
        z_dim=args.z_dim,
        alpha=args.alpha,
        max_episodes=args.max_episodes,
        max_step=args.max_step,
        w1=args.w1,
        w2=args.w2,
        w3=args.w3,
        confidence_threshold=args.confidence_threshold,
        knn_k=args.knn_k,
        seed=args.seed,
        device=args.device,
        arcface_scale=args.arcface_scale,
    )
    
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE!")
    print("="*80)
    print(f"Attack Accuracy: {results['top1_accuracy']:.2f}% (Top-1), {results['top5_accuracy']:.2f}% (Top-5)")
    print(f"KNN Distance: {results['avg_knn_distance']:.4f}")
    print(f"FID Score: {results['fid_score']:.4f}")
    print(f"Report: {results['report_path']}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
