#!/usr/bin/env python3
"""
rl_policy.py
============
Shared SB3 feature extractor for the orchard RecurrentPPO policy (RGBD image +
9D state, as produced by OrchardEnv). Kept in its own module — rather than
inline in train_rl.py — so that any script loading a saved checkpoint
(run_rl.py, future eval/deploy scripts) can explicitly import the exact class
its policy_kwargs reference, instead of depending on cloudpickle's by-value
serialization of a class defined in __main__.
"""

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from orchard_env import IMG_C, IMG_H, IMG_W, N_IMG, STATE_DIM


class ImageStateFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(IMG_C, 32, 5, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2), nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        dummy = torch.zeros(1, IMG_C, IMG_H, IMG_W)
        conv_out_dim = self.conv(dummy).view(1, -1).shape[1]
        self.image_bottleneck = nn.Sequential(nn.Linear(conv_out_dim, 96), nn.ReLU())
        self.state_encoder = nn.Sequential(
            nn.Linear(STATE_DIM, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        img_flat = obs[:, :N_IMG]
        state    = obs[:, N_IMG:]
        img = img_flat.view(-1, IMG_C, IMG_H, IMG_W)
        img_feat = self.conv(img).view(img.size(0), -1)
        img_enc  = self.image_bottleneck(img_feat)
        st_enc   = self.state_encoder(state)
        return torch.cat([img_enc, st_enc], dim=1)
