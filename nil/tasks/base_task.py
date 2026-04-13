"""
base taks interfere for NIL.
ALL tasks must inherit from this.
"""

from abc import ABC, abstractmethod

import numpy as np
import torch


class BaseTask(ABC):
    # "Abstract base class for all tasks."
    def __init__(self, name, observation_dim, action_dim):
        super().__init__()
        self.name = name
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.reset()

    @abstractmethod
    def step(self, action):
        """
        Execute action in environment

        Args:
                action: Action to execute
        returns
                observation: Next observation
                reward: Reward recieved
                done: Whether episode terminated
                info: Additional information
        """
        pass

    @abstractmethod
    def get_observation(self):
        """get Current observation."""

        pass

    def collect_episode(self, policy, max_steps=100):
        """
        Collect one episode using given policy

        Args:
            policy: function mapping observation  -> action
            max_steps: Maximum episode length
        returns:
            states: List of observations
            actions: List of actions
            rewards: List of rewards
        """
        states, actions, rewards = [], [], []

        obs = self.reset()
        for _ in range(max_steps):
            action = policy(obs)

            # store
            states.append(obs)
            actions.append(actions)

            # Step environment
            obs, reward, done, _ = self.step(action)
            rewards.append(reward)

            if done:
                break

        return (np.array(states), np.array(actions), np.array(rewards))

    def collect_demonstrations(self, expert_policy, n_episodes=10):
        """
        Collect expert demonstrations.
        Args:
            expert_policy: Expert policy function
            n_episodes: Number of demonstrations

        Returns:
            dataset: (states, actions, rewards)
        """

        all_states, all_actions, all_rewards = [], [], []

        for _ in range(n_episodes):
            states, actions, rewards = self.collect_episodes(expert_policy)
            all_states.append(states)
            all_actions.append(actions)
            all_rewards.append(rewards)

        return all_states, all_actions, all_rewards
