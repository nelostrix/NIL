"""Native policy construction and offline imitation learning; no actuation."""

from nil.nn import mlp


def policy(observation_dim, action_dim, hidden=(128, 128)):
    """Continuous-action policy. Units, bounds and deployment are explicit caller contracts."""
    return mlp(observation_dim, action_dim, hidden=hidden)


def demonstrations(observations, actions):
    from nil.data import tensors
    from nil.runtime import require

    torch = require("torch", "torch")
    observations = torch.as_tensor(observations, dtype=torch.float32)
    actions = torch.as_tensor(actions, dtype=torch.float32)
    if observations.ndim != 2 or actions.ndim != 2 or len(observations) != len(actions):
        raise ValueError("Demonstrations require matching [samples, features/actions] tensors")
    if (
        not len(observations)
        or not torch.isfinite(observations).all()
        or not torch.isfinite(actions).all()
    ):
        raise ValueError("Demonstrations must be nonempty and finite")
    return tensors(observations, actions)
