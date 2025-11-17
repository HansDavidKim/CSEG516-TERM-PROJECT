"""
SAC (Soft Actor-Critic) Agent Implementation for RLB-MI Attack
Based on: Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple, Optional
import random
from collections import deque


class ReplayBuffer:
    """Experience replay buffer for SAC agent."""

    def __init__(self, capacity: int = 1_000_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: np.ndarray, reward: float,
             next_state: np.ndarray, done: bool):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample a batch of experiences."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)).unsqueeze(1)
        )

    def __len__(self) -> int:
        return len(self.buffer)


class Actor(nn.Module):
    """Gaussian policy network (actor) for SAC."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

        # Action bounds (for latent space, we use no bounds initially)
        self.action_scale = 1.0
        self.action_bias = 0.0

        # Clamp log_std to avoid numerical instability
        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass to get mean and log_std."""
        x = self.net(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state: torch.Tensor, epsilon: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from the policy."""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        # Reparameterization trick
        x_t = normal.rsample()
        action = torch.tanh(x_t) * self.action_scale + self.action_bias

        # Calculate log probability with change of variables
        log_prob = normal.log_prob(x_t)
        # Enforcing action bounds with tanh
        log_prob -= torch.log(self.action_scale * (1 - action.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)

        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob


class Critic(nn.Module):
    """Q-function network (critic) for SAC."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass to get Q-value."""
        x = torch.cat([state, action], dim=1)
        return self.net(x)


class SACAgent:
    """Soft Actor-Critic Agent for latent space search."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: torch.device,
        hidden_dim: int = 256,
        learning_rate: float = 5e-4,
        gamma: float = 0.99,
        tau: float = 0.01,
        alpha: float = 0.2,
        auto_entropy_tuning: bool = True,
        replay_buffer_capacity: int = 1_000_000,
    ):
        """
        Initialize SAC agent.

        Args:
            state_dim: Dimension of state space (latent dimension)
            action_dim: Dimension of action space (latent dimension)
            device: Device to run on
            hidden_dim: Hidden layer dimension
            learning_rate: Learning rate for all networks
            gamma: Discount factor
            tau: Soft update parameter
            alpha: Entropy regularization coefficient
            auto_entropy_tuning: Whether to automatically tune alpha
            replay_buffer_capacity: Capacity of replay buffer
        """
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        # Actor network
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)

        # Twin critics
        self.critic1 = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=learning_rate)

        # Target critics
        self.critic1_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic2_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Automatic entropy tuning
        self.auto_entropy_tuning = auto_entropy_tuning
        if auto_entropy_tuning:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=learning_rate)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(replay_buffer_capacity)

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """Select action from policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if evaluate:
            with torch.no_grad():
                mean, _ = self.actor(state_tensor)
                action = torch.tanh(mean) * self.actor.action_scale + self.actor.action_bias
        else:
            with torch.no_grad():
                action, _ = self.actor.sample(state_tensor)

        return action.cpu().numpy()[0]

    def observe(self, state: np.ndarray, action: np.ndarray, reward: float,
                next_state: np.ndarray, done: bool):
        """Store experience in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update_policy(self, batch_size: int = 256) -> Optional[dict]:
        """Update policy using SAC algorithm."""
        if len(self.replay_buffer) < batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Update critics
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_next = self.critic1_target(next_states, next_actions)
            q2_next = self.critic2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            q_target = rewards + (1 - dones) * self.gamma * q_next

        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)

        critic1_loss = F.mse_loss(q1, q_target)
        critic2_loss = F.mse_loss(q2, q_target)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Update actor
        new_actions, log_probs = self.actor.sample(states)
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update alpha (entropy coefficient)
        alpha_loss = None
        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp().item()

        # Soft update target networks
        self._soft_update(self.critic1_target, self.critic1)
        self._soft_update(self.critic2_target, self.critic2)

        return {
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha,
            'alpha_loss': alpha_loss.item() if alpha_loss is not None else 0.0,
        }

    def _soft_update(self, target: nn.Module, source: nn.Module):
        """Soft update target network parameters."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def save(self, path: str):
        """Save agent parameters."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'critic1_target': self.critic1_target.state_dict(),
            'critic2_target': self.critic2_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.auto_entropy_tuning else None,
            'alpha_optimizer': self.alpha_optimizer.state_dict() if self.auto_entropy_tuning else None,
        }, path)

    def load(self, path: str):
        """Load agent parameters."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])

        if self.auto_entropy_tuning and checkpoint['log_alpha'] is not None:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
