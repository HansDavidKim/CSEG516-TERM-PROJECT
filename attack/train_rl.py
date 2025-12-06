import os
import time
import heapq
import torch
import torch.nn.functional as F
from pathlib import Path
from copy import deepcopy
from torchvision.utils import save_image
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms

from attack.sac_agent import Agent
from attack.utils import load_generator, load_classifier
from attack.attack_utils import compute_reward, evaluate_confidence


def train_attack(
    generator_path: str,
    classifier_path: str,
    target_class: int,
    generator_dim: int = 64,
    max_episodes: int = 40000,
    max_step: int = 1,
    z_dim: int = 100,
    alpha: float = 0.0,
    w1: float = 2.0,
    w2: float = 2.0,
    w3: float = 8.0,

    confidence_threshold: float = 0.95,
    seed: int = 42,
    device: str = "cuda",
):
    """
    Train RL attack matching original RLB-MI inversion() function.
    
    This implements the exact training loop from the original paper's code.
    """
    run_name = f"attack_cls{target_class}_{int(time.time())}"
    output_dir = Path(f"attack_results/{run_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device (CUDA priority)
    if device == "cuda" and torch.cuda.is_available():
        device_obj = torch.device("cuda")
    elif device == "mps" and torch.backends.mps.is_available():
        device_obj = torch.device("mps")
    else:
        device_obj = torch.device("cpu")
    
    print(f"Using {device_obj} device")
    
    # Load models
    print("Loading generator...")
    generator = load_generator(generator_path, device_obj, dim=generator_dim)
    generator.eval()
    
    print("Loading classifier...")
    classifier, arc_head = load_classifier(classifier_path, device_obj)
    classifier.eval()
    

    
    # Initialize SAC agent
    print("Initializing agent...")
    agent = Agent(
        state_size=z_dim,
        action_size=z_dim,
        random_seed=seed,
        hidden_size=256,  # Match original HIDDEN_SIZE
        action_prior="uniform",
        device=str(device_obj)
    )
    
    # Normalization parameters for classifier input (from classifier/train.py)
    mean = torch.tensor([0.5177433, 0.4284404, 0.3802497], device=device_obj).view(1, 3, 1, 1)
    std = torch.tensor([0.3042383, 0.2845056, 0.2826854], device=device_obj).view(1, 3, 1, 1)
    
    print(f"Target Label: {target_class}")
    print(f"Starting training for {max_episodes} episodes...")
    
    # Track top 10 images with min-heap: (score, episode_num, image_tensor)
    top_k = 10
    top_images = []  # Min-heap
    
    # Main training loop - matches original inversion() function
    pbar = tqdm(range(1, max_episodes + 1), desc=f"Training Class {target_class}")
    for i_episode in pbar:
        # Initialize the state at the beginning of each episode
        z = torch.randn(1, z_dim, device=device_obj)
        state = deepcopy(z.cpu().numpy())
        
        for t in range(max_step):
            # Get action from agent
            action = agent.act(state)
            
            # Update latent vector: z = alpha * z + (1-alpha) * action
            z = alpha * z + (1 - alpha) * action.clone().detach().reshape((1, len(action))).to(device_obj)
            next_state = deepcopy(z.cpu().numpy())
            
            # Generate images from state and action
            state_image = generator(z).detach()
            action_image = generator(action.clone().detach().reshape((1, len(action))).to(device_obj)).detach()
            
            # Normalize images for classifier
            state_image_norm = (state_image + 1) / 2  # [-1,1] -> [0,1]
            state_image_norm = (state_image_norm - mean) / std
            
            action_image_norm = (action_image + 1) / 2
            action_image_norm = (action_image_norm - mean) / std
            
            # Get classifier outputs (features, logits)
            features_z, _ = classifier(state_image_norm)
            features_a, _ = classifier(action_image_norm)
            
            # Use ArcFace head for logits if available, otherwise use fc_layer
            if arc_head is not None:
                state_output = arc_head.inference_logits(features_z)
                action_output = arc_head.inference_logits(features_a)
            else:
                _, state_output = classifier(state_image_norm)
                _, action_output = classifier(action_image_norm)
            
            # Calculate reward using classifier outputs
            reward = compute_reward(
                state_output, action_output, target_class,
                w1=w1, w2=w2, w3=w3, epsilon=1e-7
            )
            
            # Determine if episode is done
            done = (t == max_step - 1)
            
            # Update policy
            agent.step(state, action, reward, next_state, done, t)
            state = next_state
        
        # Evaluate current episode
        test_images = []
        test_scores = []
        
        with torch.no_grad():
            z_test = torch.randn(1, z_dim, device=device_obj)
            for t in range(max_step):
                state_test = z_test.cpu().numpy()
                action_test = agent.act(state_test)
                z_test = alpha * z_test + (1 - alpha) * action_test.clone().detach().reshape((1, len(action_test))).to(device_obj)
            
            test_image = generator(z_test).detach()
            
            # Normalize and get classifier output
            test_image_norm = (test_image + 1) / 2
            test_image_norm = (test_image_norm - mean) / std
            
            # Use ArcFace head if available
            if arc_head is not None:
                test_features, _ = classifier(test_image_norm)
                test_output = arc_head.inference_logits(test_features)
            else:
                _, test_output = classifier(test_image_norm)
            
            # Evaluate confidence score
            test_score = evaluate_confidence(test_output, target_class)
            test_images.append(test_image.cpu())
            test_scores.append(test_score)
        
        mean_score = sum(test_scores) / len(test_scores)
        
        # Update top-K tracking with min-heap
        if len(top_images) < top_k:
            # Still filling up the heap
            heapq.heappush(top_images, (mean_score, i_episode, test_images[0]))
        elif mean_score > top_images[0][0]:  # Better than worst in top-K
            heapq.heapreplace(top_images, (mean_score, i_episode, test_images[0]))
        
        # Get current best score for display
        best_score = max(top_images, key=lambda x: x[0])[0] if top_images else 0.0
        
        # Update progress bar with current best score
        pbar.set_postfix({'best_conf': f'{best_score:.4f}', 'curr_conf': f'{mean_score:.4f}'})
        
        # Early stopping: confidence threshold to prevent overfitting
        if best_score >= confidence_threshold:
            print(f'\n✅ Confidence threshold ({confidence_threshold}) reached at episode {i_episode}! Stopping to prevent overfitting.')
            # Save top-K images before stopping
            save_top_k_images(top_images, output_dir, target_class, top_k)
            break
        
        # Save top-K images periodically (every 10k episodes) and at the end
        if i_episode % 10000 == 0 or i_episode == max_episodes:
            print(f'\nEpisodes {i_episode}/{max_episodes}, Best confidence: {best_score:.4f}')
            save_top_k_images(top_images, output_dir, target_class, alpha)
    
    # Final save of top-K images
    print(f"Training finished. Best score: {best_score:.4f}")
    print(f"Saving top {len(top_images)} images...")
    save_top_k_images(top_images, output_dir, target_class, alpha)
    
    print(f"Results saved to {output_dir}")
    
    return test_images[0] if test_images else None


def save_top_k_images(top_images, output_dir, target_class, alpha):
    """Save top-K images without annotation (pure images only)."""
    if not top_images:
        return
    
    # Sort by score (descending)
    sorted_images = sorted(top_images, key=lambda x: x[0], reverse=True)
    
    image_dir = output_dir / "images"
    image_dir.mkdir(exist_ok=True)
    
    for rank, (score, episode, image_tensor) in enumerate(sorted_images, 1):
        # Save image directly without annotation
        img_tensor = (image_tensor[0] + 1) / 2  # [-1,1] -> [0,1]
        img_tensor = torch.clamp(img_tensor, 0, 1)
        
        # Save with metadata in filename only
        filename = f"rank{rank:02d}_cls{target_class}_ep{episode}_conf{score:.4f}.png"
        save_image(img_tensor, image_dir / filename)
    
    
    print(f"Saved top {len(sorted_images)} images to {image_dir}")
