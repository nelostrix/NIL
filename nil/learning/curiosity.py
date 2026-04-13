"""curiosity driven learning module.
Robot learns by exploring what surprises it."""

import numpy as np
import torch
import torch.nn as nn


class CuriosityModule(nn.Module):
    """
    Generate intrinsic rewards based on  prediction error.
    the more surprising something is, the more reward.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # forward model: predicts next state given current_state + action
        self.forward_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

        self.inverse_model = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def compute_intrinstic_reward(self, state, action, next_state):
        """
        compute curiosity reward = prediction error.
        High error = surprising = High reward = explore more!!

        Args:
            state: Current state
            action: Action taken
            next_state: resulting state
        Returns:
            intinsic_reward: How surprising this transition was
        """

        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        next_state = torch.FloatTensor(next_state)

        # Predict what next state Should be
        predicted_next_state = self.forward_model(torch.cat([state, action], dim=-1))

        # Compute prediction error (surprise)
        prediction_error = torch.mean((predicted_next_state - next_state) ** 2)

        return prediction_error.item()

    def update(self, state, action, next_state):
        """
        Train the curiosity module to get better at predictions.
        (As it gets better, it gets less surprised by familiar things)
        """
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        next_state = torch.FloatTensor(next_state)

        # Forward model loss (predict next state)
        pred_next_state = self.forward_model(torch.cat([state, action], dim=-1))
        forward_loss = nn.MSELoss()(pred_next_state, next_state)

        # Inverse model loss (predict action)
        pred_action = self.inverse_model(torch.cat([state, next_state], dim=-1))
        inverse_loss = nn.MSELoss()(pred_action, action)

        # total_loss
        loss = forward_loss + inverse_loss

        # update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


class SelfLearningAgent:
    """Agent that learns entirely from curiosity.
    No Human-provided rewards needed!"""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.curiosity = CuriosityModule(state_dim, action_dim, hidden_dim)

        # Simple policy network
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # Actions in [-1, 1]
        )
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=0.001)

    def explore(self, env, num_episodes=100):
        """
        Explore environment purely driven by curiosity.
        No task, no goal, just exploring:"""

        all_rewards = []

        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            done = False

            while not done:
                # Get action from policy
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action = self.policy(state_tensor).detach().numpy()[0]

                # ADD exploration noise
                action += np.random.normal(0, 0.1, size=action.shape)
                action = np.clip(action, -1, 1)

                # take action
                next_state, _, done, _ = env.step(action)

                # compute intristic reward (curiosity)
                intrinsic_reward = self.curiosity.compute_intrinstic_reward(
                    state, action, next_state
                )

                episode_reward += intrinsic_reward
                #
                #
                # Update curisity module
                self.curiosity.update(state, action, next_state)

                state = next_state

            all_rewards.append(episode_reward)

            if episode % 10 == 0:
                print(f"Episode {episode}: Curiosity reward = {episode_reward:.3f}")
        return all_rewards


if __name__ == "__main__":
    from nil.tasks.reach import ReachTask

    # Create environment
    env = ReachTask()

    # create self-learning agent
    agent = SelfLearningAgent(state_dim=env.observation_dim, action_dim=env.action_dim)

    # Let it explore on its own!
    print("Agent exploring without any task or reward....")
    rewards = agent.explore(env, num_episodes=50)

    print(f"\nAverage curiosity: {np.mean(rewards):.3f}")
    print("Agent learned by pure curiosity!")
