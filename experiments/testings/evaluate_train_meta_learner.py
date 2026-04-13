"""Evaluate trained meta learner"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from nil.models.liquid_nn import LiquidNN
from nil.models.meta_learner import MAMLWrapper
from nil.tasks.reach import ReachTask


def evaluate_adaptation(meta_learner, n_trials=10):
    """Test few-shot adaptation."""

    results = {"pre_adaptation": [], "post_adaptation": []}
    for trial in range(n_trials):
        # create new task
        task = ReachTask()

        # taks before adaptation
        states, _, rewards = task.collect_episode(
            policy=lambda obs: (
                meta_learner.model(torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0))[0]
                .squeeze()
                .detach()
                .numpy()
            ),
            max_steps=50,
        )
        results["pre_adaptation"].append(sum(rewards))

        # collect 10 demonstrations
        support_states, support_actions = [], []
        for _ in range(10):
            s, a, _ = task.collect_episode(task.expert_policy, max_steps=50)
            support_states.append(s)
            support_actions.append(a)

        # Convert to tensors (simplified)
        support_x = torch.FloatTensor(np.array(support_states[:1]))
        support_y = torch.FloatTensor(np.array(support_actions[:1]))

        # Adapt
        adapted_model = meta_learner.adapt(support_x, support_y)

        # test after adaptation
        task.reset()  # same task, new episode
        states, _, rewards = task.collect_episode(
            policy=lambda obs: (
                adapted_model(torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0))[0]
                .squeeze()
                .detach()
                .numpy()
            ),
            max_steps=50,
        )
        results["post_adaptation"].append(sum(rewards))

    return results


def main():
    # Load Trained_model
    print("loading Trained Model....")
    base_model = LiquidNN(input_dim=4, hidden_dim=32, output_dim=2, num_layers=5)
    base_model.load_state_dict(
        torch.load("experiments/runs/meta_reach_v1/final_model.pt")
    )
    meta_learner = MAMLWrapper(base_model, inner_lr=0.01, num_inner_steps=5)

    # Evaluate
    print("Running evaluation...")
    results = evaluate_adaptation(meta_learner, n_trials=20)

    # plot results
    fig, ax = plt.subplots(figsize=(10, 6))

    positions = [1, 2]
    data = [results["pre_adaptation"], results["post_adaptation"]]
    labels = ["before Adaptation", "After 10 Demos"]

    bp = ax.boxplot(data, positions=positions, label=labels, patch_artist=True)
    bp["boxes"][0].set_facecolor("lightcoral")
    bp["boxed"][1].set_facecolor("lightgreen")

    ax.set_ylabel("episode Return")
    ax.set_title("Meta-learning Evaluation: Few-shot adaptation")
    ax.grid(True, alpha=0.3, axis="y")

    # add statistics
    pre_mean = np.mean(results["pre_adaptation"])
    post_mean = np.mean(results["post_adaptation"])
    improvement = ((post_mean - pre_mean) / abs(pre_mean)) * 100

    ax.text(
        1.5,
        max(max(data[0]), max(data[1])) * 0.9,
        f"Improvement: {improvement: .1f}%",
        ha="center",
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor="wheat"),
    )

    plt.tight_layout()
    plt.savefig("experiments/figures/meta_learning_eval.png", dpi=300)
    print("\nResults saved to experiments/figures/meta_learning_eval.png")

    print(
        f"\nPre-adaptation: {pre_mean: .2f} +- {np.std(results['pre_adaptation']):.2f}"
    )
    print(
        f"Post-adaptation: {post_mean: .2f} +- {np.std(results['post_adaptation']):.2f}"
    )
    print(f"Improvement: {improvement: .1f}%")


if __name__ == "__main__":
    main()
