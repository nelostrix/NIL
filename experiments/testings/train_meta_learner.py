"""Meta-training script for NIL"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from nil.models.liquid_nn import LiquidNN
from nil.models.meta_learner import MAMLWrapper
from nil.tasks.reach import ReachTask
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def load_config(config_path):
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def collect_task_data(task, policy, n_episodes, max_steps):
    """Collect data from task using policy."""
    all_states, all_actions = [], []

    for _ in range(n_episodes):
        states, actions, _ = task.collect_episode(policy, max_steps)  # Fixed: singular
        all_states.append(states)
        all_actions.append(actions)

    # Convert to tensors
    # Pad sequences to same length
    max_len = max(len(s) for s in all_states)

    states_tensor = torch.zeros(n_episodes, max_len, task.observation_dim)
    actions_tensor = torch.zeros(n_episodes, max_len, task.action_dim)

    for i, (s, a) in enumerate(zip(all_states, all_actions)):
        states_tensor[i, : len(s)] = torch.FloatTensor(s)
        actions_tensor[i, : len(a)] = torch.FloatTensor(a)  # Fixed: capital T

    return states_tensor, actions_tensor


def main():
    # Load config
    config = load_config(
        "/home/shobhit/NELO/Intelligence-layer/experiments/configs/meta_reach.yaml"
    )  # Fixed: no leading slash
    print("Configuration loaded:")
    print(yaml.dump(config, default_flow_style=False))

    # Set seed
    set_seed(config["experiment"]["seed"])

    # Setup logging
    log_dir = Path("experiments/runs") / config["experiment"]["name"]
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # Create model
    print("\nInitializing model...")
    base_model = LiquidNN(
        input_dim=config["model"]["input_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        output_dim=config["model"]["output_dim"],
        num_layers=config["model"]["num_layers"],
    )

    meta_learner = MAMLWrapper(
        model=base_model,
        inner_lr=config["meta_learning"]["inner_lr"],
        num_inner_steps=config["meta_learning"][
            "num_inner_steps"
        ],  # Fixed: meta_learning
    )

    num_params = sum(p.numel() for p in base_model.parameters())
    print(
        f"Model created with {num_params:,} parameters"
    )  # Fixed: space before 'parameters'

    # Optimizer
    optimizer = optim.Adam(
        meta_learner.model.parameters(), lr=config["meta_learning"]["outer_lr"]
    )

    # Training loop
    print("\nStarting meta-training...")
    loss_fn = nn.MSELoss()

    for epoch in tqdm(range(config["meta_learning"]["num_meta_epochs"])):
        # Sample batch of tasks
        task_batch = []

        for _ in range(config["meta_learning"]["tasks_per_batch"]):  # Fixed: plural
            task = ReachTask(workspace_size=config["task"]["workspace_size"])

            # Collect support set (for adaptation)
            support_x, support_y = collect_task_data(
                task,
                task.expert_policy,
                config["meta_learning"]["support_episodes"],
                config["task"]["max_episode_steps"],
            )

            # Collect query set (for meta-loss)
            query_x, query_y = collect_task_data(
                task,
                task.expert_policy,
                config["meta_learning"]["query_episodes"],  # Fixed: meta_learning
                config["task"]["max_episode_steps"],
            )

            task_batch.append((support_x, support_y, query_x, query_y))

        # Meta-update
        optimizer.zero_grad()
        meta_loss = meta_learner.meta_loss(task_batch, loss_fn)
        meta_loss.backward()
        optimizer.step()

        # Logging
        if epoch % config["logging"]["log_interval"] == 0:
            writer.add_scalar("Loss/meta_train", meta_loss.item(), epoch)
            tqdm.write(f"Epoch {epoch}: Meta-loss = {meta_loss.item():.6f}")

        # Checkpoint
        if epoch % config["logging"]["checkpoint_interval"] == 0:
            checkpoint_path = log_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": meta_learner.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "meta_loss": meta_loss.item(),
                },
                checkpoint_path,
            )

    # Save final model
    final_path = log_dir / "final_model.pt"
    torch.save(meta_learner.model.state_dict(), final_path)
    print(f"\nTraining complete! Model saved to {final_path}")

    writer.close()


if __name__ == "__main__":
    main()
