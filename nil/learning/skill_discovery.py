"""Automatic skill discovery through diversity-seeking."""

from calendar import c

import numpy as np
import torch
import torch.nn as nn


class SkillDiscovery:
    """
    Discover diverse skills automatically.
    No Human tells it what skill to learn!
    """

    def __init__(self, state_dim, action_dim, num_skills=8, hidden_dim=64):
        self.num_skills = num_skills

        # skill-conditioned policy: n(a|s, z) where z is skill
        self.policy = nn.Sequential(
            nn.Linear(state_dim + num_skills, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        self.disciminator = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_skills),
        )
