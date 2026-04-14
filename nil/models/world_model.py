from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from your_module import LiquidCell  # replace with actual path


class WorldModel(nn.Module):
    """Predictive world model using Liquid Neural Networks.

    Learns environment dynamics: given current state and action,
    predict next state. Enables planning in imagination without
    real-world interaction.

    Architecture:
        state → encoder → latent
        (latent, action) → dynamics → next_latent
        next_latent → decoder → predicted_next_state

    Args:
        state_dim: Dimension of state observations
        action_dim: Dimension of actions
        latent_dim: Dimension of latent representation
        hidden_dim: Hidden layer size for LNN components
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int = 32,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        # Encoder: state → latent
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Dynamics: (latent, action) → next_latent
        # Uses a LiquidCell for adaptive dynamics
        self.dynamics = LiquidCell(
            input_dim=latent_dim + action_dim,
            hidden_dim=latent_dim,
            tau_min=0.1,
            tau_max=5.0,
        )

        # Decoder: latent → predicted state
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

        # Reward predictor: latent → predicted reward
        self.reward_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        """Encode state to latent representation."""
        return self.encoder(state)

    def predict_next(
        self, latent: torch.Tensor, action: torch.Tensor, dt: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict next latent state and reward.

        Args:
            latent: Current latent state [batch, latent_dim]
            action: Action [batch, action_dim]
            dt: Time step

        Returns:
            (next_latent, predicted_reward)
        """
        combined = torch.cat([latent, action], dim=-1)
        next_latent = self.dynamics(combined, latent, dt)
        reward = self.reward_head(next_latent)
        return next_latent, reward.squeeze(-1)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to predicted state."""
        return self.decoder(latent)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor, dt: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full forward: state + action → predicted_next_state + reward."""
        latent = self.encode(state)
        next_latent, reward = self.predict_next(latent, action, dt)
        next_state = self.decode(next_latent)
        return next_state, reward

    def imagine(
        self,
        initial_state: torch.Tensor,
        action_sequence: torch.Tensor,
        dt: float = 0.1,
    ) -> Tuple[List[torch.Tensor], List[float]]:
        """Imagine a trajectory by rolling out the world model.

        Args:
            initial_state: Starting state [state_dim] or [batch, state_dim]
            action_sequence: Sequence of actions [seq_len, action_dim]
            dt: Time step

        Returns:
            (predicted_states, predicted_rewards)
        """
        if initial_state.dim() == 1:
            initial_state = initial_state.unsqueeze(0)

        latent = self.encode(initial_state)
        states = [initial_state]
        rewards = []

        for t in range(len(action_sequence)):
            action = (
                action_sequence[t : t + 1]
                if action_sequence[t].dim() == 1
                else action_sequence[t : t + 1]
            )
            if action.dim() == 1:
                action = action.unsqueeze(0)

            latent, reward = self.predict_next(latent, action, dt)
            state = self.decode(latent)
            states.append(state)
            rewards.append(float(reward.item()))

        return states, rewards

    def learn_from_experience(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        rewards: Optional[torch.Tensor] = None,
        epochs: int = 10,
        lr: float = 0.001,
    ) -> Dict[str, float]:
        """Train world model on collected experience.

        Args:
            states: [N, state_dim]
            actions: [N, action_dim]
            next_states: [N, state_dim]
            rewards: [N] (optional)
            epochs: Training epochs
            lr: Learning rate

        Returns:
            Training metrics
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        state_loss_fn = nn.MSELoss()
        reward_loss_fn = nn.MSELoss()

        total_loss = 0.0

        for epoch in range(epochs):
            optimizer.zero_grad()

            pred_states, pred_rewards = self(states, actions)
            loss = state_loss_fn(pred_states, next_states)

            if rewards is not None:
                loss += 0.1 * reward_loss_fn(pred_rewards, rewards)

            loss.backward()
            optimizer.step()
            total_loss = float(loss.item())

        return {"state_loss": total_loss, "epochs": epochs}


if __name__ == "__main__":
    import numpy as np

    from nil.tasks.reach import ReachTask

    env = ReachTask()

    # Create self-learning agent
    agent = SelfLearningWorldModel(
        state_dim=env.observation_dim, action_dim=env.action_dim
    )

    # Step: 1 Explore randomly and collect data
    agent.collect_random_expirence(env, num_episodes=50)

    # Step: 2 Learn physics from experience (no labels!)
    agent.learn_physics()

    # Step: 3 Use learned model to plan
    print("\nTesting learned world model ")
    state = env.reset()
    action = agent.plan_in_imagination(state, num_plans=20)
    print(f"planned action: {action}")
    print("Agent can now Imagine and plan! ")
