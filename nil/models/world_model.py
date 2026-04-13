"""World model that learns physics by predicting the future."""

import numpy as np
import torch
import torch.nn as nn


class WorldModel(nn.Module):
    """
    Learns to predict: state(t+1) = f(state(t), action(t))
    Once trained, can plan in imagination!
    """

    def __init__(self, state_dim, action_dim, latent_dim=32):
        super().__init__()

        # Encoder: compress state to latent representation
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, latent_dim)
        )

        # Dynamics model: predict next latent state
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )

        # self.decoder: reconst4ruct state from latent
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, state_dim)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, state, action):
        """Predict next state given current state and action."""
        # Encode current state
        z = self.encoder(state)

        # predict next latent state
        z_next = self.dynamics(torch.cat([z, action], dim=-1))

        # Decode to next state
        next_state_pred = self.decoder(z_next)
        #
        return next_state_pred

    def learn_from_experience(self, states, actions, next_states, epochs=10):
        """
        Learn world model from observed transitions.
        No labels needed - just raw experience!
        """
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        next_states = torch.FloatTensor(next_states)

        for epoch in range(epochs):
            # predict next states
            pred_next_states = self(states, actions)

            # Loss: how well did we predict the future?
            loss = nn.MSELoss()(pred_next_states, next_states)

            # update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if epoch % 2 == 0:
                print(f"Epoch {epoch}: Prediction Loss = {loss.item():.6f}")

        return loss.item()

    def imagine(self, initial_state, action_sequence):
        """Imagine the future without interacting with environment.
        This is mental simulation - key to AGI
        """
        state = torch.FloatTensor(initial_state)
        imagined_trajectory = [state.numpy()]

        for action in action_sequence:
            action_tensor = torch.FloatTensor(action).unsqueeze(0)
            state = self(state.unsqueeze(0), action_tensor).squeeze(0)
            imagined_trajectory.append(state.detach().numpy())

        return imagined_trajectory


class SelfLearningWorldModel:
    """Agent that learns physics and plans in imagination."""

    def __init__(self, state_dim, action_dim):
        self.world_model = WorldModel(state_dim, action_dim)
        self.experience_buffer = []

    def collect_random_expirence(self, env, num_episodes=50):
        """Collect experience by random exploration."""
        print("collecting random experience....")

        for episode in range(num_episodes):
            state = env.reset()
            done = False

            while not done:
                # Random action
                action = np.random.uniform(-0.1, 0.1, size=env.action_dim)
                next_state, _, done, _ = env.step(action)

                # Store tranisition
                self.experience_buffer.append(
                    {"state": state, "action": action, "next_state": next_state}
                )
                state = next_state

        print(f"collected {len(self.experience_buffer)} transitions")

    def learn_physics(self):
        """Learn world model from collected experience."""
        print("\nLearning physics from experience...")

        # Extract data
        states = np.array([t["state"] for t in self.experience_buffer])
        actions = np.array([t["action"] for t in self.experience_buffer])
        next_states = np.array([t["next_state"] for t in self.experience_buffer])

        # Train world model
        self.world_model.learn_from_experience(states, actions, next_states, epochs=20)

    def plan_in_imagination(self, initial_state, num_plans=10):
        """Generate random action sequence and imagine their outcomes.
        Pick th best one based on imagined reward."""

        best_plan = None
        best_value = -float("inf")

        for _ in range(num_plans):
            # Random action sequence
            action_seq = [np.random.uniform(-0.1, 0.1, size=2) for _ in range(10)]
            # imagine the future
            trajectory = self.world_model.imagine(initial_state, action_seq)

            # Evalutate imagined trajectory
            # (here we use distance to goal as heuristic, but could be learned)
            final_state = trajectory[-1]
            value = -np.linalg.norm(final_state[:2] - final_state[2:])

            if value > best_value:
                best_value = value
                best_plan = action_seq
        return best_plan[0]  # Return first action


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
