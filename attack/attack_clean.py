"""
RLB-MI (Reinforcement Learning-Based Black-box Model Inversion) Attack
Based on: Han et al., "Reinforcement Learning-Based Black-Box Model Inversion Attacks" (2023)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from tqdm.auto import tqdm

from .models import SACAgent


class RLB_MI_Attack:
    """
    Reinforcement Learning-Based Black-box Model Inversion Attack.

    This attack uses SAC to search the latent space of a GAN to reconstruct
    private training data from a black-box target classifier.
    """

    def __init__(
        self,
        generator: nn.Module,
        target_model: nn.Module,
        target_class: int,
        latent_dim: int = 100,
        device: torch.device = torch.device('cpu'),
        # SAC hyperparameters (from paper: Section 4.1)
        learning_rate: float = 5e-4,
        gamma: float = 0.99,
        tau: float = 0.01,
        batch_size: int = 256,
        replay_buffer_capacity: int = 1_000_000,
        # MDP hyperparameters (from paper)
        diversity_factor: float = 0.0,  #  in paper (default 0 for accuracy, 0.97 for diversity)
        max_steps_per_episode: int = 1,  # Maximum steps per episode (paper uses 1)
        # Reward hyperparameters (from paper: Section 4.1)
        reward_w1: float = 2.0,  # Weight for state score r1
        reward_w2: float = 2.0,  # Weight for action score r2
        reward_w3: float = 8.0,  # Weight for class separation r3
        epsilon: float = 1e-7,   # Small constant to avoid log(0)
    ):
        """
        Initialize RLB-MI attack.

        Args:
            generator: Pre-trained GAN generator G
            target_model: Target classifier T to attack
            target_class: Target class y to reconstruct
            latent_dim: Dimension of GAN latent space
            device: Device to run on
            learning_rate: Learning rate for SAC
            gamma: Discount factor for SAC
            tau: Soft update parameter for SAC
            batch_size: Batch size for SAC updates
            replay_buffer_capacity: Replay buffer capacity
            diversity_factor:  for state transition (0 = accuracy focus, higher = diversity)
            max_steps_per_episode: Maximum steps per episode
            reward_w1: Weight for state score r1
            reward_w2: Weight for action score r2
            reward_w3: Weight for class separation r3
            epsilon: Small constant for numerical stability
        """
        self.generator = generator.to(device).eval()
        self.target_model = target_model.to(device).eval()
        self.target_class = target_class
        self.latent_dim = latent_dim
        self.device = device

        # MDP parameters
        self.diversity_factor = diversity_factor  #  in paper
        self.max_steps = max_steps_per_episode

        # Reward parameters
        self.w1 = reward_w1
        self.w2 = reward_w2
        self.w3 = reward_w3
        self.epsilon = epsilon

        # Initialize SAC agent
        self.agent = SACAgent(
            state_dim=latent_dim,
            action_dim=latent_dim,
            device=device,
            learning_rate=learning_rate,
            gamma=gamma,
            tau=tau,
            replay_buffer_capacity=replay_buffer_capacity,
        )

        self.batch_size = batch_size

    def _compute_confidence_scores(self, latent_vector: torch.Tensor) -> torch.Tensor:
        """
        Generate image from latent vector and get confidence scores from target model.

        Args:
            latent_vector: Latent vector (can be batched or single)

        Returns:
            Confidence scores from target model (softmax output)
        """
        with torch.no_grad():
            # Generate image
            if latent_vector.dim() == 1:
                latent_vector = latent_vector.unsqueeze(0)

            generated_image = self.generator(latent_vector)

            # Get confidence scores from target model
            _, logits = self.target_model(generated_image)
            confidence_scores = F.softmax(logits, dim=1)

        return confidence_scores

    def _compute_reward(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray
    ) -> float:
        """
        Compute reward based on paper equation (4), (5), (6).

        Reward components:
        - r1 = log[T_y(G(s_{t+1}))]  : State score
        - r2 = log[T_y(G(a_t))]       : Action score
        - r3 = log[max{, T_y(G(s_{t+1})) - max_{i`y}T_i(G(s_{t+1}))}] : Class separation

        R_t = w1r1 + w2r2 + w3r3

        Args:
            state: Current state s_t
            action: Action a_t
            next_state: Next state s_{t+1}

        Returns:
            Reward value
        """
        # Convert to tensors
        next_state_tensor = torch.FloatTensor(next_state).to(self.device)
        action_tensor = torch.FloatTensor(action).to(self.device)

        # Get confidence scores
        next_state_scores = self._compute_confidence_scores(next_state_tensor)[0]
        action_scores = self._compute_confidence_scores(action_tensor)[0]

        # r1: State score - log of target class confidence for next state
        target_conf_next = next_state_scores[self.target_class].item()
        r1 = np.log(max(self.epsilon, target_conf_next))

        # r2: Action score - log of target class confidence for action
        target_conf_action = action_scores[self.target_class].item()
        r2 = np.log(max(self.epsilon, target_conf_action))

        # r3: Class separation score
        # Find max confidence for other classes
        other_classes_mask = torch.ones_like(next_state_scores, dtype=torch.bool)
        other_classes_mask[self.target_class] = False
        max_other_conf = next_state_scores[other_classes_mask].max().item()

        # Compute separation: target_conf - max_other_conf
        separation = target_conf_next - max_other_conf
        r3 = np.log(max(self.epsilon, separation))

        # Total reward
        reward = self.w1 * r1 + self.w2 * r2 + self.w3 * r3

        return reward

    def _state_transition(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        State transition function from paper equation (3):
        s_{t+1} = s_t + (1-)a_t

        Args:
            state: Current state s_t
            action: Action a_t

        Returns:
            Next state s_{t+1}
        """
        next_state = self.diversity_factor * state + (1 - self.diversity_factor) * action
        return next_state

    def train_agent(
        self,
        max_episodes: int = 40000,
        verbose: bool = True,
        log_interval: int = 1000,
        save_path: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """
        Train SAC agent using Algorithm 1 from the paper.

        Algorithm 1: Agent training
        Input: GAN G, Target classifier T, Target class y
        Output: Learned agent A

        Args:
            max_episodes: Maximum number of episodes to train
            verbose: Whether to print training progress
            log_interval: How often to log progress
            save_path: Path to save trained agent

        Returns:
            Training metrics history
        """
        metrics_history = {
            'episode_rewards': [],
            'critic1_loss': [],
            'critic2_loss': [],
            'actor_loss': [],
            'alpha': [],
        }

        progress_bar = tqdm(range(1, max_episodes + 1), desc="Training Agent", disable=not verbose)

        for episode in progress_bar:
            # Initialize state as random latent vector: s_0 ~ N(0, 1)
            state = np.random.randn(self.latent_dim).astype(np.float32)
            episode_reward = 0.0

            for step in range(self.max_steps):
                # Select action from agent: a_t = A(s_t)
                action = self.agent.select_action(state, evaluate=False)

                # Compute next state: s_{t+1} = s_t + (1-)a_t
                next_state = self._state_transition(state, action)

                # Compute reward
                reward = self._compute_reward(state, action, next_state)
                episode_reward += reward

                # Check if episode is done (always true for max_steps=1)
                done = (step == self.max_steps - 1)

                # Store experience in replay buffer
                self.agent.observe(state, action, reward, next_state, done)

                # Update policy
                update_info = self.agent.update_policy(self.batch_size)

                # Move to next state
                state = next_state

            # Log metrics
            metrics_history['episode_rewards'].append(episode_reward)

            if update_info is not None:
                metrics_history['critic1_loss'].append(update_info['critic1_loss'])
                metrics_history['critic2_loss'].append(update_info['critic2_loss'])
                metrics_history['actor_loss'].append(update_info['actor_loss'])
                metrics_history['alpha'].append(update_info['alpha'])

            # Update progress bar
            if episode % log_interval == 0 or episode == 1:
                avg_reward = np.mean(metrics_history['episode_rewards'][-log_interval:])
                progress_bar.set_postfix({
                    'avg_reward': f'{avg_reward:.4f}',
                    'episode': episode
                })

            # Log periodically
            if verbose and episode % log_interval == 0:
                avg_reward = np.mean(metrics_history['episode_rewards'][-log_interval:])
                if update_info is not None:
                    tqdm.write(
                        f"Episode {episode}/{max_episodes} | "
                        f"Avg Reward: {avg_reward:.4f} | "
                        f"Actor Loss: {update_info['actor_loss']:.4f} | "
                        f"Critic Loss: {(update_info['critic1_loss'] + update_info['critic2_loss']) / 2:.4f} | "
                        f"Alpha: {update_info['alpha']:.4f}"
                    )

        # Save trained agent
        if save_path:
            save_dir = Path(save_path).parent
            save_dir.mkdir(parents=True, exist_ok=True)
            self.agent.save(save_path)
            if verbose:
                print(f"Agent saved to {save_path}")

        return metrics_history

    def generate_reconstructed_images(
        self,
        num_images: int = 1000,
        select_best: bool = True,
        top_k: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """
        Generate reconstructed images using trained agent.

        Args:
            num_images: Number of images to generate (episodes)
            select_best: Whether to select best images based on confidence
            top_k: Number of best images to return (if select_best=True)

        Returns:
            Tuple of:
            - Generated images (top_k or num_images)
            - Confidence scores for target class
            - Terminal states (latent vectors)
        """
        generated_latents = []
        confidences = []

        self.agent.actor.eval()

        for _ in tqdm(range(num_images), desc="Generating Images"):
            # Initialize random state
            state = np.random.randn(self.latent_dim).astype(np.float32)

            # Run episode
            for step in range(self.max_steps):
                action = self.agent.select_action(state, evaluate=True)
                next_state = self._state_transition(state, action)
                state = next_state

            # Store terminal state
            generated_latents.append(state)

            # Get confidence score
            state_tensor = torch.FloatTensor(state).to(self.device)
            scores = self._compute_confidence_scores(state_tensor)[0]
            target_conf = scores[self.target_class].item()
            confidences.append(target_conf)

        generated_latents = np.array(generated_latents)
        confidences = np.array(confidences)

        # Select best images
        if select_best:
            top_indices = np.argsort(confidences)[-top_k:][::-1]
            selected_latents = generated_latents[top_indices]
            selected_confidences = confidences[top_indices]
        else:
            selected_latents = generated_latents
            selected_confidences = confidences

        # Generate images from latents
        with torch.no_grad():
            latent_tensors = torch.FloatTensor(selected_latents).to(self.device)
            generated_images = self.generator(latent_tensors)

        return generated_images, torch.FloatTensor(selected_confidences), selected_latents

    def load_agent(self, path: str):
        """Load trained agent from checkpoint."""
        self.agent.load(path)
        print(f"Agent loaded from {path}")


def compute_attack_accuracy(
    reconstructed_images: torch.Tensor,
    target_class: int,
    eval_classifier: nn.Module,
    device: torch.device,
) -> float:
    """
    Compute attack accuracy using evaluation classifier.

    Args:
        reconstructed_images: Reconstructed images
        target_class: Target class
        eval_classifier: Evaluation classifier (different from target model)
        device: Device to run on

    Returns:
        Attack accuracy (percentage)
    """
    eval_classifier.eval()
    with torch.no_grad():
        images = reconstructed_images.to(device)
        _, logits = eval_classifier(images)
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == target_class).float().mean().item() * 100.0
    return accuracy


def compute_knn_distance(
    reconstructed_images: torch.Tensor,
    target_class_images: torch.Tensor,
    eval_classifier: nn.Module,
    device: torch.device,
    k: int = 1,
) -> float:
    """
    Compute K-nearest neighbor distance in feature space.

    Args:
        reconstructed_images: Reconstructed images
        target_class_images: Real images from target class
        eval_classifier: Evaluation classifier for feature extraction
        device: Device to run on
        k: Number of nearest neighbors

    Returns:
        Average L2 distance to k-nearest neighbors
    """
    eval_classifier.eval()
    with torch.no_grad():
        # Extract features
        recon_features, _ = eval_classifier(reconstructed_images.to(device))
        target_features, _ = eval_classifier(target_class_images.to(device))

        # Compute pairwise L2 distances
        recon_features = recon_features.cpu()
        target_features = target_features.cpu()

        distances = []
        for recon_feat in recon_features:
            # Compute L2 distance to all target features
            dists = torch.norm(target_features - recon_feat.unsqueeze(0), p=2, dim=1)
            # Get k smallest distances
            knn_dists = torch.topk(dists, k, largest=False).values
            distances.append(knn_dists.mean().item())

        avg_knn_dist = np.mean(distances)

    return avg_knn_dist


def compute_feature_distance(
    reconstructed_images: torch.Tensor,
    target_class_images: torch.Tensor,
    eval_classifier: nn.Module,
    device: torch.device,
) -> float:
    """
    Compute L2 distance between reconstructed features and centroid of target class features.

    Args:
        reconstructed_images: Reconstructed images
        target_class_images: Real images from target class
        eval_classifier: Evaluation classifier for feature extraction
        device: Device to run on

    Returns:
        Average L2 distance to feature centroid
    """
    eval_classifier.eval()
    with torch.no_grad():
        # Extract features
        recon_features, _ = eval_classifier(reconstructed_images.to(device))
        target_features, _ = eval_classifier(target_class_images.to(device))

        # Compute centroid of target features
        target_centroid = target_features.mean(dim=0)

        # Compute L2 distances to centroid
        distances = torch.norm(recon_features.cpu() - target_centroid.cpu().unsqueeze(0), p=2, dim=1)
        avg_feat_dist = distances.mean().item()

    return avg_feat_dist
