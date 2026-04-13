"""
Simple reaching task for testing meta-learning.
Robot must reach a randomly positioned target.
"""

import numpy as np


class ReachTask:
    """2D reaching task."""

    def __init__(self, workspace_size=1.0):
        self.workspace_size = workspace_size
        self.max_steps = 50
        self.observation_dim = 4  # [x, y, goal_x, goal_y]
        self.action_dim = 2  # [dx, dy]
        self.step_count = 0

        # Initialize as None first
        self.position = None
        self.goal = None

        # Then call reset to set them properly
        self.reset()

    def reset(self):
        """Reset with random initial position and goal."""
        # Random initial position - MUST be size=2
        self.position = np.random.uniform(
            low=-self.workspace_size / 2,
            high=self.workspace_size / 2,
            size=2,  # Important: creates array of shape (2,)
        )

        # Random goal position - MUST be size=2
        self.goal = np.random.uniform(
            low=-self.workspace_size / 2,
            high=self.workspace_size / 2,
            size=2,  # Important: creates array of shape (2,)
        )

        self.step_count = 0

        # Debug: uncomment these to verify shapes
        # print(f"position shape: {self.position.shape}, goal shape: {self.goal.shape}")
        # print(f"position: {self.position}, goal: {self.goal}")

        return self.get_observation()

    def get_observation(self):
        """Get current state."""
        # Both should be shape (2,), result should be shape (4,)
        obs = np.concatenate([self.position, self.goal])
        return obs

    def step(self, action):
        """Execute action."""
        # Ensure action is numpy array
        action = np.asarray(action)

        # Apply action (with bounds)
        self.position = self.position + np.clip(action, -0.1, 0.1)
        self.position = np.clip(
            self.position, -self.workspace_size / 2, self.workspace_size / 2
        )

        # Compute reward (negative distance to goal)
        distance = np.linalg.norm(self.position - self.goal)
        reward = -distance

        # Check if reached goal
        done = (distance < 0.05) or (self.step_count >= self.max_steps)
        self.step_count += 1

        obs = self.get_observation()
        info = {"distance": distance, "success": distance < 0.05}

        return obs, reward, done, info

    def expert_policy(self, obs):
        """Simple expert: move towards goal."""
        obs = np.asarray(obs)  # Ensure it's numpy array
        pos = obs[:2]
        goal = obs[2:]

        direction = goal - pos

        # Normalize and scale
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm

        return direction * 0.05  # Small steps

    def collect_episode(self, policy, max_steps=100):
        """Collect one episode using given policy."""
        states, actions, rewards = [], [], []

        obs = self.reset()

        for _ in range(max_steps):
            # Get action from policy
            action = policy(obs)

            # Ensure action is numpy array
            action = np.asarray(action)

            # Store
            states.append(obs.copy())
            actions.append(action.copy())

            # Step environment
            obs, reward, done, _ = self.step(action)
            rewards.append(reward)

            if done:
                break

        return (np.array(states), np.array(actions), np.array(rewards))


if __name__ == "__main__":
    # Test task
    print("Testing ReachTask...")
    task = ReachTask()

    print(f"Observation dim: {task.observation_dim}")
    print(f"Action dim: {task.action_dim}")

    # Test observation
    obs = task.get_observation()
    print(f"Observation shape: {obs.shape}, expected: (4,)")
    print(f"Observation: {obs}")

    # Collect expert demonstration
    print("\nCollecting expert episode...")
    states, actions, rewards = task.collect_episode(
        policy=task.expert_policy, max_steps=50
    )

    print(f"Episode length: {len(states)}")
    print(f"States shape: {states.shape}")
    print(f"Actions shape: {actions.shape}")
    print(f"Total reward: {sum(rewards):.3f}")
    print(f"Final distance: {np.linalg.norm(states[-1][:2] - states[-1][2:]):.3f}")
    print("✓ ReachTask works!")
