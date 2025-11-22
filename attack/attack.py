### attack/attack.py
### Custom Gymnasium Environment for Adversarial Attack
### This will be combined with the stable-baseline3 library

import gymnasium as gym
from gymnasium import spaces

import numpy as np
import torch

from attack.utils import load_generator, load_classifier

class ModelInversionAttackEnv(gym.Env):
    def __init__(self, generator_path: str, classifier_path: str):
        super().__init__()
        
        ### action space is 100-dim vector
        ### which follows Normal distribution : N(0, I)
        self.action_space = spaces.Box(low=-100, high=100, shape=(100,), dtype=np.float32)

        ### observation here is latent vector z in 100-dim real vector space.
        ### Its upper bound and lower bound is indeed negative infinity and positive infinity.
        ### However, for simplicity, we clamp it to [-100, 100].
        self.observation_space = spaces.Box(low=-100, high=100, shape=(100,), dtype=np.float32)       
        
        self.z = np.random.normal(0, 1, 100)
        self.z = np.clip(self.z, -100, 100)
        
        ### We load the generator and classifier so that we can define action and reward with them.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Generator
        self.generator = load_generator(generator_path, self.device)
        
        # Load Classifier
        self.classifier = load_classifier(classifier_path, self.device)

    def reset(self):
        self.z = np.random.normal(0, 1, 100)
        self.z = np.clip(self.z, -100, 100)
        return self.z

    