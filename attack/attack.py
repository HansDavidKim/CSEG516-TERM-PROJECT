### attack/attack.py
### Custom Gymnasium Environment for Adversarial Attack
### This will be combined with the stable-baseline3 library

import gymnasium as gym
from gymnasium import spaces

import numpy as np
import torch

from attack.utils import load_generator, load_classifier

class ModelInversionAttackEnv(gym.Env):
    def __init__(self,
        generator_path: str,
        classifier_path: str,
        target_class: int,
        generator_dim: int = 64,
        alpha: float = 0.9,
        w1: float = 2.0,
        w2: float = 2.0,
        w3: float = 8.0,
    ):
        super().__init__()
        
        self.generator_dim = generator_dim
        self.target_class = target_class
        self.alpha = alpha
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

        ### action space is 100-dim vector
        ### which follows Normal distribution : N(0, I)
        ### We limit the range to [-3, 3] to stay within reasonable latent space bounds.
        self.action_space = spaces.Box(low=-3, high=3, shape=(100,), dtype=np.float32)

        ### observation here is latent vector z in 100-dim real vector space.
        ### Its upper bound and lower bound is indeed negative infinity and positive infinity.
        ### However, for simplicity, we clamp it to [-100, 100].
        self.observation_space = spaces.Box(low=-100, high=100, shape=(100,), dtype=np.float32)       
        
        self.z = np.random.normal(0, 1, 100)
        self.z = np.clip(self.z, -3, 3)
        
        ### We load the generator and classifier so that we can define action and reward with them.
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        # Load Generator
        self.generator = load_generator(generator_path, self.device, dim=self.generator_dim)
        
        # Load Classifier and ArcFace head
        self.classifier, self.arc_head = load_classifier(classifier_path, self.device)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.z = np.random.normal(0, 1, 100)
        self.z = np.clip(self.z, -3, 3)
        return self.z.astype(np.float32), {}

    def step(self, action):
        # State transition: s_{t+1} = alpha * s_t + (1 - alpha) * a_t
        # Paper uses alpha=0, so s_{t+1} = a_t (action becomes next state directly)
        action = np.clip(action, -3, 3)
        
        if self.alpha == 0:
            self.z = action
        else:
            next_z = self.alpha * self.z + (1 - self.alpha) * action
            self.z = next_z
        
        # Calculate Reward
        # We need tensor versions for model inference
        z_tensor = torch.tensor(self.z, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_tensor = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            # Generate images (Range: [-1, 1])
            gen_img_z = self.generator(z_tensor)
            gen_img_a = self.generator(action_tensor)
            
            # Normalize for Classifier
            # 1. [-1, 1] -> [0, 1]
            gen_img_z_norm = (gen_img_z + 1) / 2
            gen_img_a_norm = (gen_img_a + 1) / 2
            
            # 2. [0, 1] -> Normalized (using stats from classifier/train.py)
            mean = torch.tensor([0.5177433, 0.4284404, 0.3802497], device=self.device).view(1, 3, 1, 1)
            std = torch.tensor([0.3042383, 0.2845056, 0.2826854], device=self.device).view(1, 3, 1, 1)
            
            gen_img_z_norm = (gen_img_z_norm - mean) / std
            gen_img_a_norm = (gen_img_a_norm - mean) / std
            
            # Get features from classifier
            features_z, _ = self.classifier(gen_img_z_norm)
            features_a, _ = self.classifier(gen_img_a_norm)
            
            # Use ArcFace head for inference (if available)
            if self.arc_head is not None:
                logits_z = self.arc_head.inference_logits(features_z)
                logits_a = self.arc_head.inference_logits(features_a)
            else:
                # Fallback to fc_layer if no arc_head
                _, logits_z = self.classifier(gen_img_z_norm)
                _, logits_a = self.classifier(gen_img_a_norm)
            
            probs_z = torch.softmax(logits_z, dim=1)
            probs_a = torch.softmax(logits_a, dim=1)
            
            # Get target class probability
            prob_target_z = probs_z[0, self.target_class].item()
            prob_target_a = probs_a[0, self.target_class].item()
            
            # Find max probability among other classes
            probs_z_clone = probs_z[0].clone()
            probs_z_clone[self.target_class] = -1.0  # Mask target
            max_other_prob = torch.max(probs_z_clone).item()
            
            # Paper's reward function with log
            # r1 = log(T_y(G(s_{t+1})))
            r1 = np.log(prob_target_z + 1e-10)
            
            # r2 = log(T_y(G(a_t)))
            r2 = np.log(prob_target_a + 1e-10)
            
            # r3 = log(max(epsilon, T_y(G(s_{t+1})) - max_{i != y} T_i(G(s_{t+1}))))
            epsilon = 1e-7
            diff = prob_target_z - max_other_prob
            r3 = np.log(max(epsilon, diff))
            
            reward = self.w1 * r1 + self.w2 * r2 + self.w3 * r3
            
        terminated = False # Infinite horizon in paper context, or handled by max_steps wrapper
        truncated = False
        
        return self.z.astype(np.float32), reward, terminated, truncated, {"target_prob": prob_target_z}

    