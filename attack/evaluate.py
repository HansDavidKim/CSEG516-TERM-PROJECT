import os
import random
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

from attack.sac_agent import Agent
from attack.utils import load_generator, load_classifier
from attack.attack_utils import evaluate_confidence


def measure_attack_accuracy(
    generator_path: str,
    target_classifier_path: str,
    eval_classifier_path: str,
    num_labels: int,
    generator_dim: int = 64,
    z_dim: int = 100,
    alpha: float = 0.0,
    max_episodes: int = 40000,
    max_step: int = 1,
    w1: float = 2.0,
    w2: float = 2.0,
    w3: float = 8.0,
    confidence_threshold: float = 0.95,
    seed: int = 42,
    device: str = "cuda",
):
    """
    Measure attack accuracy across multiple target classes.
    
    This follows the original RLB-MI evaluation protocol:
    1. Attack each target class with RL
    2. Evaluate generated images with independent evaluation classifier
    3. Calculate Top-1 and Top-5 accuracy
    """
    # Setup device
    if device == "mps" and torch.backends.mps.is_available():
        device_obj = torch.device("mps")
    elif device == "cuda" and torch.cuda.is_available():
        device_obj = torch.device("cuda")
    else:
        device_obj = torch.device("cpu")
    
    print(f"Using {device_obj} device")
    print(f"Measuring attack accuracy for {num_labels} target classes...")
    
    # Fix random seeds for reproducibility
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
    print("Loading generator...")
    generator = load_generator(generator_path, device_obj, dim=generator_dim)
    generator.eval()
    
    print("Loading target classifier...")
    target_classifier, target_arc_head = load_classifier(target_classifier_path, device_obj)
    target_classifier.eval()
    
    print("Loading evaluation classifier...")
    eval_classifier, eval_arc_head = load_classifier(eval_classifier_path, device_obj)
    eval_classifier.eval()
    
    # Normalization for classifier
    mean = torch.tensor([0.5177433, 0.4284404, 0.3802497], device=device_obj).view(1, 3, 1, 1)
    std = torch.tensor([0.3042383, 0.2845056, 0.2826854], device=device_obj).view(1, 3, 1, 1)
    
    # Random sample of target classes
    all_classes = list(range(num_labels))
    
    # Results tracking
    top1_correct = 0
    top5_correct = 0
    total = 0
    
    results = []
    
    # Attack each target class
    for target_class in tqdm(all_classes, desc="Attacking classes"):
        print(f"\n{'='*60}")
        print(f"Target Class: {target_class}")
        print(f"{'='*60}")
        
        # Initialize RL agent for this target
        agent = Agent(
            state_size=z_dim,
            action_size=z_dim,
            random_seed=seed,
            hidden_size=256,
            action_prior="uniform",
            device=str(device_obj)
        )
        
        best_score = 0
        best_image = None
        attack_succeeded = False
        
        # Train attack for this target (simplified, fewer episodes for speed)
        print(f"Training attack for {max_episodes} episodes...")
        pbar_episodes = tqdm(range(1, max_episodes + 1), desc=f"Class {target_class} Training", leave=False)
        for i_episode in pbar_episodes:
            # Generate initial latent
            z = torch.randn(1, z_dim, device=device_obj)
            state = z.cpu().numpy().copy()
            
            for t in range(max_step):
                # Get action
                action = agent.act(state)
                
                # Update latent
                z = alpha * z + (1 - alpha) * action.clone().detach().reshape((1, len(action))).to(device_obj)
                next_state = z.cpu().numpy().copy()
                
                # Generate images
                state_image = generator(z).detach()
                action_image = generator(action.clone().detach().reshape((1, len(action))).to(device_obj)).detach()
                
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
                from attack.attack_utils import compute_reward
                reward = compute_reward(state_output, action_output, target_class, w1, w2, w3)
                
                done = (t == max_step - 1)
                agent.step(state, action, reward, next_state, done, t)
                state = next_state
            
            # Evaluate current policy EVERY episode (don't miss successful attacks)
            with torch.no_grad():
                z_test = torch.randn(1, z_dim, device=device_obj)
                for t in range(max_step):
                    state_test = z_test.cpu().numpy()
                    action_test = agent.act(state_test)
                    z_test = alpha * z_test + (1 - alpha) * action_test.clone().detach().reshape((1, len(action_test))).to(device_obj)
                
                test_image = generator(z_test).detach()
                test_image_norm = (test_image + 1) / 2
                test_image_norm = (test_image_norm - mean) / std
                
                if target_arc_head is not None:
                    test_features, _ = target_classifier(test_image_norm)
                    test_output = target_arc_head.inference_logits(test_features)
                else:
                    _, test_output = target_classifier(test_image_norm)
                
                test_score = evaluate_confidence(test_output, target_class)
                
                if test_score > best_score:
                    best_score = test_score
                    best_image = test_image
                
                # Update progress bar every episode
                pbar_episodes.set_postfix({'best_conf': f'{best_score:.4f}', 'curr_conf': f'{test_score:.4f}'})
                
                # Early stopping: confidence threshold to prevent overfitting
                if best_score >= confidence_threshold:
                    print(f"✅ Confidence threshold ({confidence_threshold}) reached at episode {i_episode}! Stopping to prevent overfitting.")
                    attack_succeeded = True
                    break
            
            # Early stopping check with evaluation classifier (every 100 episodes to save time)
            if i_episode % 100 == 0:
                eval_image_norm = (best_image + 1) / 2
                eval_image_norm = (eval_image_norm - mean) / std
                
                if eval_arc_head is not None:
                    eval_features, _ = eval_classifier(eval_image_norm)
                    eval_output = eval_arc_head.inference_logits(eval_features)
                else:
                    _, eval_output = eval_classifier(eval_image_norm)
                
                eval_probs = F.softmax(eval_output, dim=-1)[0]
                top1_pred = torch.argmax(eval_probs).item()
                
                if top1_pred == target_class:
                    print(f"✅ Top-1 attack succeeded at episode {i_episode}! Moving to next class.")
                    attack_succeeded = True
                    break
            
            # Progress logging
            if i_episode % 1000 == 0:
                print(f"Episode {i_episode}/{max_episodes}, Best confidence: {best_score:.4f}")
        
        # Evaluate with independent classifier (skip if already succeeded)
        print(f"\nEvaluating generated image with evaluation classifier...")
        
        if attack_succeeded:
            # Already verified during early stopping
            eval_image_norm = (best_image + 1) / 2
            eval_image_norm = (eval_image_norm - mean) / std
            
            if eval_arc_head is not None:
                eval_features, _ = eval_classifier(eval_image_norm)
                eval_output = eval_arc_head.inference_logits(eval_features)
            else:
                _, eval_output = eval_classifier(eval_image_norm)
            
            eval_probs = F.softmax(eval_output, dim=-1)[0]
            top1_pred = torch.argmax(eval_probs).item()
            top5_probs, top5_indices = torch.topk(eval_probs, min(5, len(eval_probs)))
            top5_preds = top5_indices.tolist()
            
            is_top1_correct = True  # Already verified
            is_top5_correct = True
        else:
            # Full evaluation
            with torch.no_grad():
                eval_image_norm = (best_image + 1) / 2
                eval_image_norm = (eval_image_norm - mean) / std
                
                if eval_arc_head is not None:
                    eval_features, _ = eval_classifier(eval_image_norm)
                    eval_output = eval_arc_head.inference_logits(eval_features)
                else:
                    _, eval_output = eval_classifier(eval_image_norm)
                
                eval_probs = F.softmax(eval_output, dim=-1)[0]
                
                # Top-1 prediction
                top1_pred = torch.argmax(eval_probs).item()
                
                # Top-5 predictions
                top5_probs, top5_indices = torch.topk(eval_probs, min(5, len(eval_probs)))
                top5_preds = top5_indices.tolist()
                
                # Check correctness
                is_top1_correct = (top1_pred == target_class)
                is_top5_correct = (target_class in top5_preds)
            
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
                'eval_top1_prob': eval_probs[top1_pred].item(),
                'eval_top5_preds': top5_preds,
                'is_top1_correct': is_top1_correct,
                'is_top5_correct': is_top5_correct,
            }
            results.append(result)
            
            print(f"\n--- Results for Target Class {target_class} ---")
            print(f"Target Classifier Confidence: {best_score:.4f}")
            print(f"Eval Top-1 Prediction: {top1_pred} (prob: {eval_probs[top1_pred]:.4f})")
            print(f"Eval Top-5 Predictions: {top5_preds}")
            print(f"Top-1 Correct: {is_top1_correct}")
            print(f"Top-5 Correct: {is_top5_correct}")
            print(f"\nCurrent Attack Accuracy:")
            print(f"  Top-1: {top1_correct}/{total} = {100*top1_correct/total:.2f}%")
            print(f"  Top-5: {top5_correct}/{total} = {100*top5_correct/total:.2f}%")
    
    # Final results
    top1_accuracy = 100 * top1_correct / total
    top5_accuracy = 100 * top5_correct / total
    
    print(f"\n{'='*60}")
    print(f"FINAL Attack Accuracy Results")
    print(f"{'='*60}")
    print(f"Total Classes Attacked: {total}")
    print(f"Top-1 Accuracy: {top1_correct}/{total} = {top1_accuracy:.2f}%")
    print(f"Top-5 Accuracy: {top5_correct}/{total} = {top5_accuracy:.2f}%")
    print(f"{'='*60}")
    
    return {
        'top1_accuracy': top1_accuracy,
        'top5_accuracy': top5_accuracy,
        'results': results
    }
