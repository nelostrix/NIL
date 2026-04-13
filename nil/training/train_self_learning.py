"""Train a completely self learning AI"""

import numpy as np
from nil.learning.curiosity import SelfLearningAgent
from nil.learning.skill_discovery import SkillDiscovery
from nil.models.world_model import SelfLearningWorldModel
from nil.tasks.reach import ReachTask


def main():
    env = ReachTask()

    print("=" * 60)
    print("self Learning AI Experiment")
    print("=" * 60)

    # Experiment 1: Curiosity-driven exploration
    print("\n[1/3] Curiosity-Driven Learning")
    print("-" * 60)
    curiosity_agent = SelfLearningAgent(
        state_dim=env.observation_dim, action_dim=env.action_dim
    )
    curiosity_rewards = curiosity_agent.explore(env, num_episodes=100)
    print(f"Learned through Curiosity! Avg reward: {np.mean(curiosity_rewards):.3f}")

    # Experiment 2: World model learning
    print("\n[2/3] world model learning")
    print("=" * 60)
    world_model_agent = SelfLearningWorldModel(
        state_dim=env.observation_dim, action_dim=env.action_dim
    )
    world_model_agent.collect_random_expirence(env, num_episodes=50)
    world_model_agent.learn_physics()
    print("Learned Physics of the World: ")
    # Experiment 3: skill discovery
    print("[3/3] skill Discovery")
    print("-" * 60)
    skill_agent = SkillDiscovery(
        state_dim=env.observation_dim, action_dim=env.action_dim, num_skills=8
    )
    skill_agent.discover_skills(env, num_episodes=200)
    skill_agent.visualize_skills(env)
    print("✓ Discovered reusable skills!")

    print("\n" + "=" * 60)
    print("ALL SELF-LEARNING EXPERIMENTS COMPLETE!")
    print("=" * 60)
    print("\nThe AI learned:")
    print("  ✓ To explore driven by curiosity")
    print("  ✓ Physics of its world")
    print("  ✓ Diverse reusable skills")
    print("\nAll without ANY human-provided labels or rewards!")


if __name__ == "__main__":
    main()
