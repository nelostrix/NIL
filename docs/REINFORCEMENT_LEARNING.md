# Reinforcement Learning Signals

> **Status: Implemented** — The full RL signal layer exists in `nil/rl/`. `DenseRewardModel`, `ShapedRewardModel`, `InverseRLRewardModel`, `TrajectoryScorer`, `EloTrajectoryRating`, `DirectPreferenceOptimizer`, `Policy` / `BehavioralCloning`, `ValueFunction` / `AdvantageEstimator`, `ExplorationScheduler`, `ReplayBuffer`, and `RLEnabledAgent` are all implemented.

---

## Background

The evolution engine originally operated through textual gradients — LLM-generated critiques of failed traces that get converted into prompt modifications. This was prompt optimization dressed as evolution. It worked initially but plateaued because:

1. **Textual gradients are high-variance** — an LLM's critique of a trace is non-deterministic and may miss the actual failure mode
2. **No credit assignment** — a 10-step trace with 1 bad step gets critiqued as a whole; the good 9 steps are not distinguished
3. **No exploration** — text gradients only exploit patterns seen in failures; they never suggest trying something genuinely new
4. **No reward modeling** — success/failure is binary; there's no notion of *how much* better one trajectory is over another
5. **No policy representation** — the "policy" is a text prompt, not a parameterized function that can be optimized with gradient descent

Formal RL signals transform Nil from a prompt-optimization system into a true policy-optimization system.

---

## Before RL Signals

```
Textual Gradient Evolution:

  Failed Trace ──→ LLM Critique ──→ Prompt Edit ──→ New Strategy

  Limitations:
    • No numerical reward signal
    • No per-step credit assignment
    • No exploration policy
    • No value function
    • No regret minimization
    • No sample efficiency tracking
    • No online/offline distinction
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RL SIGNAL LAYER                             │
│                                                                     │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────────┐ │
│  │   Reward   │  │  Trajectory│  │ Preference  │  │  Value       │ │
│  │   Model    │  │  Scorer    │  │  Optimizer  │  │  Function    │ │
│  │            │  │            │  │            │  │              │ │
│  │ R(s,a) from│  │ Score full │  │ Pairwise   │  │ V(s) =       │ │
│  │ outcomes   │  │ trajectories│ │ comparison  │  │ expected    │ │
│  │ + cost     │  │ for quality │  │ of runs    │  │ future      │ │
│  │ + progress │  │ ranking     │  │ via Elo    │  │ reward      │ │
│  └─────┬──────┘  └──────┬─────┘  └──────┬─────┘  └──────┬───────┘ │
│        │                │               │               │          │
│        └────────────────┴───────────────┴───────────────┘          │
│                                 │                                   │
│                                 v                                   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Policy Optimizer                          │   │
│  │                                                              │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌───────┐  │   │
│  │  │ Offline RL │  │  Online RL │  │ Imitation  │  │ MCTS  │  │   │
│  │  │ (CQL/IQL)  │  │  (PPO/REIN-│  │ Learning   │  │       │  │   │
│  │  │            │  │  FORCE)    │  │ (BC)       │  │       │  │   │
│  │  └────────────┘  └────────────┘  └────────────┘  └───────┘  │   │
│  │                                                              │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                 │                                   │
│                                 v                                   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Exploration Layer                         │   │
│  │                                                              │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌───────┐  │   │
│  │  │ Thompson   │  │ Upper      │  │ Epsilon-   │  │ Info- │  │   │
│  │  │ Sampling   │  │ Confidence │  │ Greedy     │  │ gain  │  │   │
│  │  │            │  │ Bound (UCB)│  │            │  │ bonus │  │   │
│  │  └────────────┘  └────────────┘  └────────────┘  └───────┘  │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 1. Reward Model

The first step is moving from binary success/failure to a continuous, multi-factor reward signal.

### Dense Per-Step Rewards

```python
class DenseRewardModel:
    """
    Computes reward at every step of a trajectory, not just at the end.

    This enables credit assignment: which steps contributed to success?
    """

    def step_reward(
        self,
        step: StepRecord,
        state_before: EnvironmentState,
        state_after: EnvironmentState,
        trajectory: list[StepRecord],
    ) -> float:
        """
        Multi-factor reward for a single step.

        Components:
          Progress:   Did we move toward goal? (+0.1 to +1.0)
          Cost:       How expensive was this step? (-0.01 to -0.5)
          Error:      Did we encounter an error? (-0.2 to -1.0)
          Novelty:    Did we try something new? (+0.05 exploration bonus)
          Efficiency: Did we make progress without wasted steps? (+0.1)
        """
        reward = 0.0

        # Progress reward
        reward += self._progress_reward(state_before, state_after)

        # Information gain reward (reduced uncertainty)
        reward += self._info_gain_reward(state_before, state_after)

        # Cost penalty
        reward -= self._cost_penalty(step)

        # Error penalty
        if step.error:
            reward -= self._error_penalty(step)

        # Exploration bonus
        if self._is_novel_action(state_before, step):
            reward += self.config.exploration_bonus

        # Efficiency bonus (progress per token spent)
        if step.tokens_in + step.tokens_out > 0:
            progress_per_token = (
                self._progress_reward(state_before, state_after)
                / (step.tokens_in + step.tokens_out)
            )
            if progress_per_token > self.config.efficiency_threshold:
                reward += self.config.efficiency_bonus

        return reward

    def trajectory_return(self, trajectory: list[StepRecord]) -> float:
        """Sum of discounted step rewards over the full trajectory."""
        returns = 0.0
        gamma = self.config.discount_factor
        for t, step in enumerate(trajectory):
            returns += (gamma ** t) * self.step_reward(...)
        return returns
```

### Reward Shaping from Domain Knowledge

```python
class ShapedRewardModel(DenseRewardModel):
    """
    Reward shaping using domain knowledge.

    Principles:
      - Potential-based shaping: F(s, s') = γΦ(s') - Φ(s)
      - Preserves optimal policy while speeding learning
      - Φ(s) is the domain's "progress toward goal" potential
    """

    def potential(self, state: EnvironmentState) -> float:
        """Domain-specific potential function."""
        # Example for incident-response domain
        potential = 0.0

        # Diagnosis progress
        if state.investigation_stage == "diagnosed":
            potential += 5.0
        elif state.investigation_stage == "resolved":
            potential += 10.0

        # Service health
        if state.service_health == "healthy":
            potential += 8.0
        elif state.service_health == "degraded":
            potential += 2.0

        # Penalize high uncertainty
        potential -= state.uncertainty * 3.0

        return potential

    def shaping_reward(self, state: State, next_state: State) -> float:
        """Potential-based shaping: γΦ(s') - Φ(s)."""
        gamma = self.config.discount_factor
        return gamma * self.potential(next_state) - self.potential(state)
```

### Learned Reward Model (Inverse RL)

When the domain's `success_signal` is too coarse, learn the reward function from expert demonstrations:

```python
class InverseRLRewardModel:
    """
    Learn reward function from successful trajectories using
    Maximum Entropy Inverse Reinforcement Learning.

    Principle:
      Expert trajectories maximize reward, but with bounded rationality.
      P(τ | R) ∝ exp(∑ R(s, a)) — trajectory probability under reward R.

    Uses:
      - Successful traces as "expert demonstrations"
      - Learns feature weights that explain expert behavior
      - Generalizes reward beyond the training traces
    """

    def learn_from_demonstrations(
        self,
        expert_traces: list[Trace],
        feature_extractor: Callable,
        n_iterations: int = 100,
    ) -> RewardFunction:
        """
        MaxEnt IRL:
          1. Compute feature expectations of expert
          2. Initialize reward as linear in features: R = w·φ(s,a)
          3. For each iteration:
             a. Compute policy under current reward
             b. Sample trajectories from policy
             c. Compute feature expectations of policy
             d. Update weights: w ← w + η * (μ_expert - μ_policy)
          4. Return learned reward function
        """
        ...
```

---

## 2. Trajectory Scoring & Ranking

Not all successful trajectories are equal. Some are elegant, some are clumsy. Some took 2 steps, some took 20. Trajectory scoring enables fine-grained learning.

```python
class TrajectoryScorer:
    """
    Scores full trajectories on multiple quality dimensions.

    Used for:
      - Ranking trajectories for preference optimization
      - Identifying exemplar trajectories for imitation
      - Filtering low-quality successes from the training set
    """

    def score(self, trace: Trace) -> TrajectoryScore:
        return TrajectoryScore(
            # Efficiency metrics
            steps_efficiency=self._step_efficiency(trace),
            token_efficiency=self._token_efficiency(trace),
            cost_efficiency=self._cost_efficiency(trace),

            # Quality metrics
            correctness=self._correctness(trace),
            completeness=self._completeness(trace),
            robustness=self._robustness(trace),

            # Learning metrics
            novelty=self._novelty(trace),
            information_gain=self._information_gain(trace),
            exploration_value=self._exploration_value(trace),

            # Composite
            composite=self._compute_composite(),
        )

    def rank(
        self, trajectories: list[Trace]
    ) -> list[tuple[Trace, TrajectoryScore]]:
        """Rank trajectories by quality for preference optimization."""
        scored = [(t, self.score(t)) for t in trajectories]
        return sorted(scored, key=lambda x: x[1].composite, reverse=True)
```

---

## 3. Preference Optimization

Instead of optimizing from binary rewards, learn from pairwise comparisons between trajectories — which is more robust and scalable.

### Elo-Based Trajectory Rating

```python
class EloTrajectoryRating:
    """
    Rates trajectories using an Elo system.

    Each successful trajectory is a "player."
    When two trajectories are compared, the better one wins Elo points.

    Benefits:
      - Continuous improvement signal from many weak comparisons
      - Naturally handles multi-dimensional quality
      - Can incorporate human feedback when available
    """

    def __init__(self, k_factor: int = 32, initial_rating: int = 1500):
        self.ratings: dict[str, float] = {}  # trace_id → Elo
        self.k = k_factor

    def compare(self, winner_id: str, loser_id: str) -> tuple[float, float]:
        """
        Update Elo ratings after comparing two trajectories.

        Expected score:
          E_A = 1 / (1 + 10^((R_B - R_A) / 400))

        Update:
          R_A ← R_A + K * (1 - E_A)
          R_B ← R_B + K * (0 - E_B)
        """
        R_A = self.ratings.get(winner_id, 1500)
        R_B = self.ratings.get(loser_id, 1500)

        E_A = 1.0 / (1.0 + 10 ** ((R_B - R_A) / 400.0))
        E_B = 1.0 - E_A

        self.ratings[winner_id] = R_A + self.k * (1.0 - E_A)
        self.ratings[loser_id] = R_B + self.k * (0.0 - E_B)

        return self.ratings[winner_id], self.ratings[loser_id]
```

### Direct Preference Optimization (DPO)

For policy learning from preferences (no explicit reward model needed):

```python
class DirectPreferenceOptimizer:
    """
    Optimizes the agent's policy directly from pairwise
    trajectory preferences, without learning an intermediate
    reward function.

    DPO loss:
      L = -E[ log σ(β * (log π(y_w | x) - log π(y_l | x))) ]

    Where:
      y_w: preferred (winning) trajectory
      y_l: dispreferred (losing) trajectory
      π:   policy (agent)
      β:   temperature parameter
      σ:   logistic sigmoid
    """

    def optimize(
        self,
        preferences: list[tuple[Trace, Trace, float]],
        # (winner, loser, preference_strength)
        policy: Policy,
        n_epochs: int = 10,
    ) -> Policy:
        """
        Update policy using DPO loss on trajectory preferences.

        The policy learns to assign higher probability to
        preferred trajectories over dispreferred ones.
        """
        for epoch in range(n_epochs):
            for winner, loser, strength in preferences:
                log_prob_winner = policy.log_probability(winner)
                log_prob_loser = policy.log_probability(loser)

                # DPO loss
                loss = -torch.log(
                    torch.sigmoid(
                        self.beta * (log_prob_winner - log_prob_loser)
                    )
                )

                # Weight by preference strength
                loss = loss * strength

                # Gradient step
                loss.backward()
                self.optimizer.step()

        return policy
```

---

## 4. Policy Learning

The core of the RL approach: learn a parameterized policy that maps states to actions, and optimize it with RL algorithms.

### Policy Representation

```python
@dataclass
class Policy:
    """
    A parameterized decision-making function.

    π(a | s, task) → probability of taking action a in state s.

    Representation:
      - For discrete actions: neural network over state features
      - For LLM actions: learned prompt + preference-weighted in-context examples
      - Hybrid: neural router that selects which LLM/config to use per state
    """

    policy_type: Literal["neural", "llm", "hybrid"]

    # Neural policy parameters
    network_weights: Optional[dict] = None

    # LLM policy parameters
    system_prompt_template: Optional[str] = None
    in_context_examples: Optional[list[dict]] = None
    temperature: float = 0.0

    # Hybrid router
    router_weights: Optional[dict] = None     # state → model selector

    def act(self, state: EnvironmentState, task: str) -> Action:
        """Sample action from the policy."""
        ...

    def log_probability(self, trajectory: Trace) -> float:
        """Log probability of a trajectory under this policy."""
        ...
```

### Offline RL: Conservative Q-Learning (CQL)

Learn from the existing trace database without online interaction:

```python
class ConservativeQLearning:
    """
    Offline RL algorithm that learns from static trace data.

    Key insight: standard Q-learning overestimates values for
    out-of-distribution actions. CQL adds a conservative penalty
    that prevents the Q-function from assigning high value to
    actions not seen in the data.

    Loss:
      L = α * E[log Σ_a exp(Q(s, a)) - E[Q(s, a)]] + standard Bellman error

    This enables learning from all historical traces (including
    failed ones) to produce a better policy — without ever
    interacting with the environment during training.
    """

    def learn(
        self,
        traces: list[Trace],
        state_space: StateSpace,
        action_space: ActionSpace,
        alpha: float = 0.1,            # conservatism penalty weight
        gamma: float = 0.99,           # discount factor
    ) -> QFunction:
        """
        Train a Q-function from offline trace data.

        Usage:
          After training, the Q-function can be used to select
          actions: π(s) = argmax_a Q(s, a)

        CQL ensures we don't recommend actions we've never tried.
        """
        ...
```

### Online RL: PPO (Proximal Policy Optimization)

For agents that can still interact with the environment (at cost):

```python
class ProximalPolicyOptimization:
    """
    Online RL for agents that can interact with the environment.

    PPO is stable, sample-efficient, and widely used.

    Loss:
      L_CLIP = min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)

    Where:
      ratio = π_new(a|s) / π_old(a|s)
      A = advantage (how much better than expected)
      ε = clip range (0.2 typical)
    """

    def __init__(self, policy: Policy, learning_rate: float = 3e-4):
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
        self.clip_epsilon = 0.2

    def update(
        self,
        trajectories: list[Trace],
        advantages: list[float],
    ) -> PolicyUpdate:
        """
        Single PPO update step.

        Steps:
          1. Compute probability ratio: π_new / π_old
          2. Clipped surrogate objective
          3. Gradient step on policy parameters
          4. Optional: KL penalty to prevent destructive updates
        """
        total_loss = 0.0

        for trajectory, advantage in zip(trajectories, advantages):
            old_log_prob = trajectory.policy_log_prob     # stored at collection
            new_log_prob = self.policy.log_probability(trajectory)

            ratio = torch.exp(new_log_prob - old_log_prob)
            clipped_ratio = torch.clamp(
                ratio,
                1.0 - self.clip_epsilon,
                1.0 + self.clip_epsilon,
            )

            loss = -torch.min(
                ratio * advantage,
                clipped_ratio * advantage,
            ).mean()

            total_loss += loss

        # Gradient step
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        self.optimizer.step()

        return PolicyUpdate(loss=total_loss.item())
```

### Imitation Learning: Behavioral Cloning

Simple but effective: learn to imitate successful trajectories:

```python
class BehavioralCloning:
    """
    Supervised learning from successful trajectories.

    Loss: -Σ log π(a_t | s_t)  for all (s_t, a_t) in successful traces.

    Fast, stable, but limited by the quality of demonstrations.
    Good for bootstrapping; bad for surpassing the demonstrator.
    """

    def clone(
        self,
        demonstrations: list[Trace],
        policy: Policy,
    ) -> Policy:
        """
        Train policy via maximum likelihood on demonstration data.

        Policy learns to mimic the actions taken in successful traces,
        conditioned on the state at each step.
        """
        for epoch in range(self.n_epochs):
            for trace in demonstrations:
                for step in trace.steps:
                    state = step.state_before
                    action = step.action
                    log_prob = policy.log_probability(action, state)
                    loss = -log_prob.mean()
                    loss.backward()
                    self.optimizer.step()

        return policy
```

---

## 5. Advantage & Value Estimation

For policy gradient methods, we need to know whether an action was better or worse than expected:

```python
class AdvantageEstimator:
    """
    Estimates the advantage A(s, a) = Q(s, a) - V(s).

    Positive advantage: action was better than average
    Negative advantage: action was worse than average

    Methods:
      - Monte Carlo: A = G - V(s) where G is actual return
      - TD(λ): A = δ_t + γλδ_{t+1} + (γλ)²δ_{t+2} + ...
      - GAE: Generalized Advantage Estimation
    """

    def compute_gae(
        self,
        rewards: list[float],
        values: list[float],
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> list[float]:
        """
        Generalized Advantage Estimation.

        Balances bias and variance through λ parameter:
          λ=0: high bias, low variance (TD(0))
          λ=1: low bias, high variance (Monte Carlo)

        GAE(λ) = Σ (γλ)^t * δ_{t+1}
        where δ_t = r_t + γV(s_{t+1}) - V(s_t)
        """
        advantages = []
        gae = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                delta = rewards[t] - values[t]
            else:
                delta = rewards[t] + gamma * values[t + 1] - values[t]

            gae = delta + gamma * lam * gae
            advantages.insert(0, gae)

        return advantages
```

### Value Function Learning

```python
class ValueFunction:
    """
    V(s) = expected return from state s under current policy.

    Used for:
      - Advantage estimation
      - Baseline for variance reduction
      - Evaluating state quality without acting
    """

    def __init__(self, state_dim: int, hidden_dim: int = 64):
        self.network = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-3)

    def predict(self, state: EnvironmentState) -> float:
        """Estimate value of state."""
        features = self._extract_features(state)
        return self.network(features).item()

    def update(self, states: list[State], returns: list[float]) -> float:
        """Train V(s) to predict actual returns."""
        losses = []
        for state, G in zip(states, returns):
            features = self._extract_features(state)
            predicted = self.network(features)
            loss = torch.nn.functional.mse_loss(predicted, torch.tensor([[G]]))
            losses.append(loss)

        total_loss = torch.stack(losses).mean()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()
```

---

## 6. Exploration via Bandit Algorithms

Evolution needs exploration, not just exploitation of known patterns:

```python
class ExplorationScheduler:
    """
    Manages the explore/exploit trade-off for the agent.

    When to explore:
      - New domain or task type (high uncertainty)
      - Recent performance degradation (drift detected)
      - Known strategies failing repeatedly
      - Information gain would be high

    When to exploit:
      - High-confidence strategy available
      - Cost budget is tight
      - Task is routine and well-understood
    """

    # --- Bandit Algorithms ---

    def ucb1(self, action: str, n_total: int) -> float:
        """
        Upper Confidence Bound.

        score(a) = Q(a) + sqrt(2 * log(N) / n(a))

        Actions with fewer tries get an optimism bonus.
        Guarantees logarithmic regret.
        """
        Q = self.action_values.get(action, 0.0)
        n = self.action_counts.get(action, 1)
        return Q + math.sqrt(2 * math.log(n_total + 1) / n)

    def thompson_sampling(self, action: str) -> float:
        """
        Thompson Sampling (Bayesian bandit).

        Maintains a Beta distribution over each action's success rate.
        Samples from the posterior to decide.

        Naturally balances exploration and exploitation.
        """
        alpha = self.success_counts.get(action, 1)
        beta = self.failure_counts.get(action, 1)
        return numpy.random.beta(alpha, beta)

    # --- Exploration Strategy ---

    def should_explore(self, state: EnvironmentState) -> bool:
        """Decide whether to explore in the current state."""
        # Rule 1: High uncertainty → explore
        if state.uncertainty > self.config.uncertainty_threshold:
            return True

        # Rule 2: Novel state → explore
        if self.state_visits.get(state.key, 0) < self.config.min_visits:
            return True

        # Rule 3: Drift detected → explore alternatives
        if self.drift_detector.is_degrading(state.domain):
            return True

        # Rule 4: Budget-dependent epsilon-greedy
        epsilon = self._adaptive_epsilon(state)
        return random.random() < epsilon

    def _adaptive_epsilon(self, state: EnvironmentState) -> float:
        """
        Epsilon decays with experience, rises with uncertainty.

        ε = max(0.01, min(0.5, base_ε * uncertainty / max_uncertainty))
        """
        base = self.config.base_exploration_rate      # e.g. 0.1
        uncertainty_ratio = state.uncertainty / self.config.max_uncertainty
        decay_factor = 1.0 / (1.0 + state.total_runs * 0.001)

        return max(0.01, min(0.5, base * uncertainty_ratio * decay_factor))
```

---

## 7. Monte Carlo Tree Search (MCTS)

For high-stakes decisions, MCTS simulates many possible futures to select the best action:

```python
class MCTSNode:
    """Node in the Monte Carlo search tree."""

    def __init__(self, state: State, parent: Optional['MCTSNode'] = None, action: Optional[Action] = None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: dict[Action, MCTSNode] = {}
        self.visits = 0
        self.total_value = 0.0
        self.untried_actions: list[Action] = state.available_actions()

    def ucb1(self, exploration_constant: float = 1.41) -> float:
        """Upper Confidence Bound for Trees."""
        if self.visits == 0:
            return float('inf')
        exploitation = self.total_value / self.visits
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration


class MCTSPlanner:
    """
    Monte Carlo Tree Search for action selection.

    Four phases per iteration:
      1. SELECT: Traverse tree using UCB1 until a leaf
      2. EXPAND: Add child nodes for all untried actions
      3. SIMULATE: Roll out from leaf using default policy
      4. BACKUP: Propagate results up the tree
    """

    def __init__(self, transition_model, reward_model, n_simulations: int = 100):
        self.transition_model = transition_model
        self.reward_model = reward_model
        self.n_simulations = n_simulations

    def best_action(self, state: State) -> Action:
        root = MCTSNode(state)

        for _ in range(self.n_simulations):
            # SELECT
            node = root
            while node.untried_actions == [] and node.children:
                node = max(node.children.values(), key=lambda n: n.ucb1())

            # EXPAND
            if node.untried_actions:
                action = node.untried_actions.pop()
                next_state = self.transition_model.sample(node.state, action)
                child = MCTSNode(next_state, parent=node, action=action)
                node.children[action] = child
                node = child

            # SIMULATE
            value = self._simulate(node.state)

            # BACKUP
            while node is not None:
                node.visits += 1
                node.total_value += value
                node = node.parent

        # Return most visited action
        return max(root.children.items(), key=lambda x: x[1].visits)[0]

    def _simulate(self, state: State, depth: int = 10) -> float:
        """Roll out a random/default policy from the state."""
        total_reward = 0.0
        gamma = self.reward_model.config.discount_factor

        for t in range(depth):
            if state.is_terminal:
                break
            action = self._default_policy(state)
            next_state = self.transition_model.sample(state, action)
            reward = self.reward_model.reward(state, action, next_state)
            total_reward += (gamma ** t) * reward
            state = next_state

        return total_reward
```

---

## 8. Integration with Nil

### Training Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│                    RL Training Pipeline                       │
│                                                               │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌────────┐ │
│  │ Collect   │───→│ Compute  │───→│ Compute  │───→│ Policy │ │
│  │ Trajectory│    │ Reward   │    │ Advantage│    │ Update │ │
│  └──────────┘    └──────────┘    └──────────┘    └────────┘ │
│       │                               ▲                      │
│       │                               │                      │
│       v                               │                      │
│  ┌──────────┐    ┌──────────┐         │                      │
│  │ Store in │───→│ Sample   │─────────┘                      │
│  │ Replay   │    │ Batch    │                                │
│  │ Buffer   │    │          │                                │
│  └──────────┘    └──────────┘                                │
│                                                               │
│  Background consolidation (offline RL):                       │
│    Every N runs, run CQL/BC on all accumulated traces         │
│    Extract improved policy, deploy to agent                   │
└──────────────────────────────────────────────────────────────┘
```

### Agent Integration

```python
class RLEnabledAgent(Agent):
    def __init__(self, ...):
        super().__init__(...)
        self.replay_buffer = ReplayBuffer(max_size=10000)
        self.reward_model = DenseRewardModel(domain=self.domain)
        self.value_function = ValueFunction(state_dim=self._state_dim())
        self.exploration = ExplorationScheduler()
        self.policy = self._init_policy()

        # Online RL config
        self.online_rl_config = RLConfig(
            algorithm="ppo",          # ppo, cql, bc, dpo
            batch_size=64,
            learning_rate=3e-4,
            discount_factor=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            update_frequency=5,       # update policy every N steps
        )

    def _select_action(
        self,
        state: EnvironmentState,
        available_actions: list[Action],
    ) -> Action:
        """RL-informed action selection."""
        if self.exploration.should_explore(state):
            # Explore: sample from bandit or random
            return self.exploration.sample_action(available_actions)

        # Exploit: use learned policy
        if self.config.use_mcts and state.uncertainty > 0.3:
            # High uncertainty → plan with MCTS
            return self.mcts_planner.best_action(state)

        # Default: policy → highest-value action
        return self.policy.act(state)

    def _on_step_complete(
        self,
        step: StepRecord,
        state_before: State,
        state_after: State,
    ):
        """Record step for RL training."""
        reward = self.reward_model.step_reward(
            step, state_before, state_after, self._current_trajectory
        )
        self.replay_buffer.push(
            state=state_before,
            action=step.action,
            reward=reward,
            next_state=state_after,
            done=state_after.is_terminal,
        )

        # Periodic policy update
        if len(self.replay_buffer) >= self.online_rl_config.batch_size:
            self._update_policy()

    def _update_policy(self):
        """RL policy update step."""
        batch = self.replay_buffer.sample(self.online_rl_config.batch_size)

        if self.online_rl_config.algorithm == "ppo":
            # Compute advantages using GAE
            values = self.value_function.predict_batch(batch.states)
            next_values = self.value_function.predict_batch(batch.next_states)
            advantages = compute_gae(
                batch.rewards, values, next_values, batch.done,
                self.online_rl_config.gamma,
                self.online_rl_config.gae_lambda,
            )
            # PPO update
            self.ppo.update(batch.trajectories, advantages)

        elif self.online_rl_config.algorithm == "dpo":
            # Preference optimization from ranked trajectories
            ranked = self.trajectory_scorer.rank(self.recent_trajectories)
            self.dpo_optimizer.optimize(ranked, self.policy)
```

### Offline Consolidation (Background)

```python
class RLBackgroundConsolidation:
    """
    Runs in background between agent runs.

    Uses all accumulated traces for offline RL:
      - CQL: learn Q-function from all data
      - BC: clone best trajectories
      - Preference optimization: learn from rankings
    """

    def consolidate(
        self,
        domain: str,
        config: RLConsolidationConfig,
    ) -> ConsolidationReport:
        traces = self.memory.get_traces(domain, limit=10000)

        # Score and rank all trajectories
        scored = self.trajectory_scorer.rank(traces)

        # Top trajectories for behavioral cloning
        top_demonstrations = [t for t, s in scored[:50] if s.composite > 0.8]

        if config.use_bc and top_demonstrations:
            self.bc.clone(top_demonstrations, self.policy)

        # All trajectories for offline RL
        if config.use_cql:
            self.cql.learn(
                traces,
                self.state_space,
                self.action_space,
            )

        # Preference pairs for DPO
        if config.use_dpo:
            preferences = self._build_preference_pairs(scored[:200])
            self.dpo.optimize(preferences, self.policy)

        # Update value function
        returns = [self.reward_model.trajectory_return(t) for t in traces]
        self.value_function.update(traces, returns)

        return ConsolidationReport(
            bc_loss=...,
            cql_loss=...,
            dpo_accuracy=...,
            value_mse=...,
        )
```

---

## Comparison: Textual Gradients vs. RL Signals

| Dimension | Textual Gradients | RL Signals |
|---|---|---|
| **Feedback** | Natural language critique | Numerical reward (dense, per-step) |
| **Credit assignment** | Whole-trajectory critique | Per-step reward → which step helped/hurt |
| **Exploration** | None (critiques known patterns) | Bandit algorithms, MCTS, epsilon-greedy |
| **Policy representation** | Text prompt edits | Learned function (neural, tabular, hybrid) |
| **Sample efficiency** | Low (one critique per failure) | High (every step is a training example) |
| **Optimality** | Limited by LLM critique quality | Converges to optimal policy (given enough data) |
| **Scalability** | Degrades with more skills | Improves with more data |
| **Uncertainty** | Not modeled | Explicit in value function, belief state |
| **Regret** | Not tracked | Minimized by design |
| **Mathematical foundation** | Heuristic | Well-studied convergence guarantees |

---

## When to Use What

```
┌────────────────────────────────────────────────────────────────┐
│                    DECISION MATRIX                              │
├────────────────────────────────────────────────────────────────┤
│                                                               │
│  Use textual gradients when:                                   │
│    • Fewer than 50 traces available (cold start)               │
│    • No numerical reward signal is defined                     │
│    • Quick iteration needed (RL training takes compute)        │
│    • The domain is fundamentally linguistic (writing, editing) │
│                                                               │
│  Use RL signals when:                                          │
│    • 100+ traces accumulated                                   │
│    • Clear success/failure signal exists                       │
│    • Multi-step workflows with many possible action sequences  │
│    • Robotics, automation, control tasks                       │
│    • Cost optimization matters (trading off quality for price) │
│                                                               │
│  Use BOTH (recommended):                                       │
│    • Textual gradients for understanding WHY something failed  │
│    • RL signals for learning WHAT to do differently            │
│    • Textual critiques shape the reward model                  │
│    • RL optimization finds the optimal policy                  │
│                                                               │
└────────────────────────────────────────────────────────────────┘
```

---

## Implementation Status

| Phase | What | Status |
|---|---|---|
| 1 | Dense reward model | **Done** — per-step multi-factor rewards (progress, cost, info-gain) |
| 2 | Trajectory scoring & ranking | **Done** — score every trajectory on efficiency, quality, novelty |
| 3 | Behavioral cloning | **Done** — learn policy from top-ranked trajectories |
| 4 | Bandit exploration | **Done** — UCB1 + Thompson sampling for action selection |
| 5 | Advantage estimation | **Done** — GAE for credit assignment, value function baseline |
| 6 | Offline RL (CQL) | **Done** — Conservative Q-Learning from all historical traces |
| 7 | Preference optimization (DPO) | **Done** — pairwise trajectory comparison with Elo ratings |
| 8 | Online RL (PPO) | **Done** — active policy improvement, replay buffer, periodic updates |
| 9 | MCTS planning | **Done** — for high-stakes decisions |
| 10 | Full integration | **Done** — `RLEnabledAgent` integrates all signals |

---

## Key Differentiator

**Textual gradients only:**
The agent's improvement is limited by the LLM's ability to self-critique. It sees a failed trace, guesses at the failure mode, and edits a prompt. There is no learning curve — every failure is a one-off critique, not a training point.

**With RL signals:**
Every step of every run — successful or failed — becomes a training datum. The agent builds a value function that knows which states are promising. It explores systematically when uncertain. It exploits optimally when confident. Improvement follows a learning curve, not a series of one-shot guesses.

Prompt optimization asks: "What should I tell the LLM?"
Policy optimization asks: "What should the agent do?"
