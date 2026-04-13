"""Automatic skill discovery through diversity-seeking."""

from calendar import c

import numpy as np
from numpy.random.mtrand import f
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
        self.discriminator = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_skills),
        )
        
        self.policy_optimizer  = torch.optim.Adam(
            self.policy.parameters(), lr=0.001
        )
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=0.001
        )
        
    def get_action(self, state, skill_id):
        """get action for a specific skill."""
        
        skill_onehot = np.zeros(self.num_skills)
        skill_onehot[skill_id] = 1
        
        #concatenate state and skill
        input_vec = torch.FloatTensor(
            np.concatenate([state, skill_onehot])
        ).unsqueeze(0)
        
        action = self.policy(input_vec).detach().numpy()[0]
        return action
    def discover_skills(self, env, num_episodes=200):
        """
        Discover skills by maximizing diversity.
        Each skilll should lead to different behaviours!
        """
        print("Discovering skills automatically")
        
        for episode in range(num_episodes):
            #Pick random skill to practice
            skill_id = np.random.randint(0, self.num_skills)
            
            state = env.reset()
            states_visited = []
            done = False
            
            while not done:
                #Get action from skill
                action = self.get_action(state, skill_id)
                action = np.random.normal(0, 0.05, size=action.shape)
                
                next_state, _, done, _ = env.step(action)
                states_visited.append(state)
                state = next_state
                
            #Train discriminator to recognize which skill was used
            if len(states_visited) > 0:
                self._train_discriminator(states_visited, skill_id)
                
            if episode % 20 == 0:
                print(f"Episode {episode}: Practical skill {skill_id}")
        print(f"Discovered {self.num_skills} diverse skills!")
        
    def _train_discriminator(self, states, skill_id):
        """Train discriminator to identify skills."""
        states_tensor = torch.FloatTensor(np.array(states))
        skill_tensor = torch.LongTensor([skill_id] * len(states))
        
        #Predict which skill was used
        pred_skill = self.discriminator(states_tensor)
        
        #Loss: can we tell which sill we used?
        loss = nn.CrossEntropyLoss()(pred_skill, skill_tensor)
        
        self.discriminator_optimizer.zero_grad()
        loss.backward()
        self.discriminator_optimizer.step()
        
    def visualize_skills(self, env, num_steps=50):
        """Visualize what each discovered skill does."""
        print(
                "\nVisualizing discovered skills:"
        )
        
        for skill_id in range(self.num_skills):
            state = env.reset()
            trajectory = []
            
            
            for _ in range(num_steps):
                action = self.get_action(state, skill_id)
                state, _, _, _ = env.step(action)
                trajectory.append(state[:2]) #Just x, y position
                
                
            trajectory = np.array(trajectory)
            print(f"skill {skill_id}: Explored from {trajectory[0]} to {trajectory[-1]}")
        
        
if __name__ == "__main__":
    from nil.tasks.reach import ReachTask
    
    env = ReachTask()
    
    #Discover skills automatically
    discoverer = SkillDiscovery(
        state_dim = env.observation_dim,
        action_dim = env.action_dim,
        num_skills = 8
    )
    
    #let it discover skills on its own!
    discoverer.discover_skills(env, num_episodes=200)
    
    #See what it Learned
    discoverer.visualize_skills(env)