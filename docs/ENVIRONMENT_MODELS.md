# Environment Models & State Machines

**Beyond Trace Analysis**

The current `WorldModelBuilder` analyzes past traces to discover tool profiles, constraints, and transition probabilities. This is retrospective pattern mining — it describes what happened, not what the environment *is* in a formal sense.

This document proposes formalizing environment models as POMDPs (Partially Observable Markov Decision Processes) with explicit state spaces, transition functions, reward models, and belief tracking. This shift turns the agent from a pattern-matcher into a system that truly models its environment.

---

## Current State

```
WorldModelBuilder (current):

  Input:  traces (historical runs)
  Output: tool_profiles, constraints, transitions, health

  What it produces:
    • Tool success rates
    • Recommended/forbidden sequences
    • Domain constraints
    • Task clusters

  What it does NOT produce:
    • A formal state space
    • Transition probabilities between states
    • A reward function
    • Belief distributions over unobserved state
    • Action-dependent dynamics
    • Counterfactual predictions ("what if I do X?")
```

---

## Proposed Architecture

```
                    ┌─────────────────────────────────────┐
                    │         ENVIRONMENT MODEL           │
                    │          (POMDP Framework)          │
                    │                                     │
                    │  ┌─────────────────────────────┐    │
                    │  │      State Space Model       │    │
                    │  │  • Observable state (O)      │    │
                    │  │  • Hidden state (H)          │    │
                    │  │  • Belief distribution b(s)  │    │
                    │  └──────────────┬──────────────┘    │
                    │                 │                    │
                    │  ┌──────────────┴──────────────┐    │
                    │  │    Transition Model T(s,a,s')│    │
                    │  │  • P(s' | s, a)             │    │
                    │  │  • Learned from traces      │    │
                    │  │  • Uncertain/confident      │    │
                    │  └──────────────┬──────────────┘    │
                    │                 │                    │
                    │  ┌──────────────┴──────────────┐    │
                    │  │    Observation Model O(s,a)  │    │
                    │  │  • P(o | s, a)             │    │
                    │  │  • Maps state → percept    │    │
                    │  │  • Sensor noise model      │    │
                    │  └──────────────┬──────────────┘    │
                    │                 │                    │
                    │  ┌──────────────┴──────────────┐    │
                    │  │      Reward Model R(s,a)     │    │
                    │  │  • Expected reward          │    │
                    │  │  • Cost function            │    │
                    │  │  • Success signal           │    │
                    │  └─────────────────────────────┘    │
                    └─────────────────────────────────────┘
                                      │
           ┌──────────────────────────┼──────────────────────────┐
           │                          │                          │
           v                          v                          v
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│   State Estimator   │  │   Policy Planner    │  │  Model Learner      │
│  (belief tracking)  │  │  (action selection) │  │  (online updates)   │
│                     │  │                     │  │                     │
│  b(s) = P(s | o₁..t)│  │  π(s) = argmax V(s)│  │  T ← update(s,a,s') │
│  Kalman / particle  │  │  MCTS / value iter  │  │  R ← update(s,a,r)  │
│  filter             │  │                     │  │  O ← update(s,a,o)  │
└─────────────────────┘  └─────────────────────┘  └─────────────────────┘
```

---

## Formal POMDP Definition

A POMDP is a 7-tuple ⟨S, A, T, R, Ω, O, γ⟩:

| Symbol | Name | Nil Equivalent |
|---|---|---|
| S | State space | The set of all possible environment configurations |
| A | Action space | The set of available tools/actions |
| T | Transition function | P(s' \| s, a) — learned from traces |
| R | Reward function | R(s, a) — derived from success_signal + cost |
| Ω | Observation space | What the agent can observe (tool outputs, error messages) |
| O | Observation function | P(o \| s, a) — maps hidden state to observations |
| γ | Discount factor | How much the agent values future vs. immediate reward |

---

## State Space Design

### Observable vs. Hidden State

Not everything in the environment is directly observable. The agent must distinguish between:

```
State s = (s_observable, s_hidden)

s_observable:
  • Current tool call results
  • Error messages
  • Task description
  • Time of day / system clock
  • External API responses

s_hidden (must be inferred):
  • Root cause of an issue
  • Internal system state (e.g., DB connection pool level)
  • User intent
  • Whether an action actually took effect
  • Whether the environment changed between observation and action
```

### State Factorization

Instead of a flat state space, states are factored into interpretable dimensions:

```python
@dataclass
class EnvironmentState:
    # System state (domain-dependent)
    service_health: Literal["healthy", "degraded", "down", "unknown"]
    incident_count: int
    last_deploy_status: Literal["success", "failure", "in_progress", "unknown"]

    # Task progress
    investigation_stage: Literal[
        "not_started", "gathering_data", "analyzing",
        "diagnosed", "fixing", "verifying", "resolved"
    ]
    current_hypothesis: Optional[str]       # what the agent currently believes
    hypothesis_confidence: float            # 0.0 - 1.0

    # Agent state
    tools_used: list[str]                   # tools called so far
    errors_encountered: list[str]           # errors seen so far
    steps_taken: int
    budget_remaining: float

    # Uncertainty
    uncertainty_level: float                # aggregate uncertainty
    unobserved_factors: list[str]           # what we know we don't know
```

### State Space Discovery

The state space is not hardcoded — it's discovered from traces:

```python
class StateSpaceLearner:
    """
    Learns the latent state space from execution traces.

    Methods:
      - Dimensionality reduction on trace features
      - Clustering of trace contexts into state clusters
      - Transition counting between state clusters
      - Automatic state factorization (which features matter?)
    """

    def discover_states(
        self,
        traces: list[Trace],
        max_states: int = 50,
    ) -> StateSpace:
        # Extract state features from each step in each trace
        # Cluster similar states
        # Build transition counts between clusters
        # Identify absorbing states (success, terminal failure)
        # Return StateSpace with named, interpretable states
```

---

## Transition Model

### Transition Function T(s, a, s')

The probability that taking action `a` in state `s` leads to state `s'`:

```
P(s' | s, a) = count(s, a → s') / count(s, a)
```

```python
@dataclass
class TransitionModel:
    # P(s' | s, a) — stored as a sparse tensor
    transitions: dict[tuple[StateKey, ActionKey], dict[StateKey, float]]

    # Aggregate statistics
    total_transitions: int
    uncertainty: dict[tuple[StateKey, ActionKey], float]   # entropy

    def predict(self, state: StateKey, action: ActionKey) -> list[tuple[StateKey, float]]:
        """Return possible next states with probabilities."""
        return sorted(
            self.transitions.get((state, action), {}).items(),
            key=lambda x: x[1],
            reverse=True,
        )

    def expected_outcome(self, state: StateKey, action: ActionKey) -> Outcome:
        """Return the most likely outcome of taking this action."""
        predictions = self.predict(state, action)
        if not predictions:
            return Outcome.unknown()
        return Outcome(
            next_state=predictions[0][0],
            confidence=predictions[0][1],
            alternative_states=predictions[1:],
        )
```

### Transition Discovery Examples

```
From incident-response traces:

  State: (service_health=degraded, investigation=not_started)
  Action: search_logs
  Transition:
    → (service_health=degraded, investigation=gathering_data)  p=0.92
    → (service_health=degraded, investigation=not_started)     p=0.05  (tool error)
    → (service_health=down, investigation=not_started)         p=0.03  (worsened during)

  State: (service_health=degraded, diagnosis="connection_pool_exhaustion")
  Action: restart_service
  Transition:
    → (service_health=healthy, investigation=verifying)        p=0.65
    → (service_health=down, investigation=fixing)              p=0.25  (no drain → crash)
    → (service_health=degraded, investigation=fixing)           p=0.10  (no change)
```

### Transition Graph as a State Machine

The transition model can be visualized as a directed graph:

```
                    ┌──────────────────┐
                    │  investigate=    │
                    │  not_started     │
                    └────────┬─────────┘
                             │ search_logs
                             v
                    ┌──────────────────┐
                    │  investigate=    │
                    │  gathering_data  │
                    └────────┬─────────┘
                             │ analyze
                             v
                    ┌──────────────────┐
                    │  investigate=    │
                    │  diagnosed       │
                    │                  │
                    │  diagnosis=      │
                    │  connection_pool │──────────────┐
                    └────────┬─────────┘              │
                             │                        │
                ┌────────────┼────────────┐           │
                v            v            v           │
        ┌────────────┐ ┌────────────┐ ┌────────────┐ │
        │ scale_up   │ │ restart    │ │ drain_and  │ │
        │            │ │            │ │ _restart   │ │
        └─────┬──────┘ └─────┬──────┘ └─────┬──────┘ │
              │              │              │         │
              v              v              v         │
        ┌─────────────────────────────────────────┐   │
        │         service_health=healthy          │   │
        │         investigate=verifying           │◄──┘
        └─────────────────────────────────────────┘
                              │ verify
                              v
        ┌─────────────────────────────────────────┐
        │         resolved (absorbing state)      │
        └─────────────────────────────────────────┘
```

---

## Reward Model

### Reward Function R(s, a)

The reward function tells the agent what outcomes are valuable:

```
R(s, a) = α * success_reward(s, a) - β * cost_penalty(s, a) - γ * time_penalty(s, a)
```

```python
@dataclass
class RewardModel:
    domain: str
    success_signal: Callable          # from Domain definition
    cost_sensitivity: float = 1.0     # β: how much to penalize cost
    time_sensitivity: float = 0.5     # γ: how much to penalize duration
    exploration_bonus: float = 0.1    # bonus for visiting novel states

    def reward(self, state: State, action: Action, next_state: State) -> float:
        # Primary reward: did we make progress?
        progress = self._progress_reward(state, next_state)

        # Success reward: did we achieve the goal?
        success = self._success_bonus(next_state)

        # Cost penalty: how expensive was the action?
        cost = -self.cost_sensitivity * action.cost

        # Time penalty: how long did it take?
        time = -self.time_sensitivity * action.duration

        # Exploration bonus: is this a novel state-action pair?
        exploration = self.exploration_bonus if self._is_novel(state, action) else 0

        return progress + success + cost + time + exploration

    def _progress_reward(self, state: State, next_state: State) -> float:
        """Positive reward for moving toward resolution."""
        # Compare investigation stage progression
        stage_order = [
            "not_started", "gathering_data", "analyzing",
            "diagnosed", "fixing", "verifying", "resolved"
        ]
        s_idx = stage_order.index(state.investigation_stage)
        ns_idx = stage_order.index(next_state.investigation_stage)
        return (ns_idx - s_idx) * 0.2

    def _success_bonus(self, state: State) -> float:
        """Large positive reward for reaching terminal success."""
        if state.is_terminal_success:
            return 10.0
        if state.is_terminal_failure:
            return -5.0
        return 0.0
```

### Learned Reward Structure

Over time, the reward model itself is refined from traces:

```python
class RewardLearner:
    """
    Learns what the agent should optimize for.

    From successful traces: what patterns of state transitions
    lead to success? Which costs are acceptable?

    From failed traces: which actions consistently lead to
    undesirable states? Which costs are wasteful?
    """

    def learn(self, traces: list[Trace]) -> RewardModel:
        # Inverse reinforcement learning:
        # Infer reward function that explains observed behavior
        # in successful traces
        ...
```

---

## Belief Tracking (Partial Observability)

Since the agent cannot directly observe the full state, it maintains a **belief distribution** — a probability distribution over possible states:

```python
class BeliefTracker:
    """
    Maintains a distribution over possible environment states
    given the history of observations and actions.

    b(s) = P(s | o₁, a₁, o₂, a₂, ..., oₜ, aₜ)
    """

    def __init__(self, state_space: StateSpace):
        self.state_space = state_space
        self.belief: dict[StateKey, float] = {
            s: 1.0 / len(state_space) for s in state_space  # uniform prior
        }

    def update(self, action: Action, observation: Observation) -> dict[StateKey, float]:
        """
        Bayes filter: update belief after taking an action and receiving an observation.

        b'(s') = P(o | s', a) * Σ_s P(s' | s, a) * b(s)
        """
        # Prediction step
        predicted = {}
        for s_prime in self.state_space:
            total = 0.0
            for s, prob in self.belief.items():
                total += self.transition_model.prob(s_prime, s, action) * prob
            predicted[s_prime] = total

        # Update step
        unnormalized = {}
        for s_prime, prob in predicted.items():
            unnormalized[s_prime] = self.observation_model.prob(
                observation, s_prime, action
            ) * prob

        # Normalize
        total = sum(unnormalized.values())
        self.belief = {
            s: p / total for s, p in unnormalized.items()
        } if total > 0 else {s: 0.0 for s in self.state_space}

        return self.belief

    @property
    def most_likely_state(self) -> StateKey:
        return max(self.belief, key=self.belief.get)

    @property
    def uncertainty(self) -> float:
        """Entropy of the belief distribution."""
        import math
        return -sum(p * math.log(p + 1e-10) for p in self.belief.values())
```

### Belief Tracking in Action

```
Timestep 1:
  Agent takes action: search_logs("API latency")
  Observation: "timeout errors in /v2/users endpoint"
  Belief update:
    root_cause=connection_pool:    0.45  ⬆
    root_cause=db_replica_lag:     0.30  ⬆
    root_cause=code_regression:    0.15  ⬆
    root_cause=traffic_spike:      0.10  ❄️ (unchanged)

Timestep 2:
  Agent takes action: query_metrics("db_connections")
  Observation: "connection pool at 98% capacity"
  Belief update:
    root_cause=connection_pool:    0.82  ⬆⬆
    root_cause=db_replica_lag:     0.08  ⬇
    root_cause=traffic_spike:      0.07  ⬇
    root_cause=code_regression:    0.03  ⬇

Timestep 3:
  Agent takes action: investigate("connection_pool_settings")
  Observation: "max_connections=50, current_active=49"
  Belief update:
    root_cause=connection_pool:    0.97  ⬆⬆
    (all others):                  0.03
  Uncertainty: LOW → agent acts confidently
```

---

## Policy Planning with the Model

Once the environment model exists, the agent can use it for planning — not just reacting:

### Value-Based Planning

```python
class ValuePlanner:
    """
    Uses the environment model to compute optimal actions.

    Methods:
      - Value iteration: compute V(s) for all states
      - Policy extraction: π(s) = argmax_a Q(s, a)
      - Look-ahead: simulate k steps to compare action outcomes
    """

    def best_action(self, state: State, horizon: int = 5) -> Action:
        """Select the action with highest expected cumulative reward."""
        best_action = None
        best_value = float("-inf")

        for action in self.available_actions(state):
            # Simulate taking this action
            next_states = self.transition_model.predict(state, action)
            value = 0.0
            for next_state, prob in next_states:
                reward = self.reward_model.reward(state, action, next_state)
                if horizon > 1:
                    # Recursive look-ahead
                    value += prob * (
                        reward + self.discount * self._value(next_state, horizon - 1)
                    )
                else:
                    value += prob * reward

            if value > best_value:
                best_value = value
                best_action = action

        return best_action
```

### MCTS for Large State Spaces

For large or continuous state spaces, use Monte Carlo Tree Search:

```python
class MCTSPlanner:
    """
    Monte Carlo Tree Search using the learned transition model.

    Trade-off: computationally expensive but handles large/continuous spaces.
    Good for: high-stakes decisions where planning latency is acceptable.
    """

    def plan(self, belief: Belief, budget: int = 1000) -> Action:
        # Build search tree using UCB1
        # Simulate rollouts using learned transition + reward models
        # Return action with highest visit count at root
        ...
```

### Heuristic Policy (for low-latency)

For routine actions where planning overhead is unacceptable:

```python
class HeuristicPolicy:
    """
    Learned policy that maps states directly to actions.

    π(s) = argmax_a Q(s, a)

    Trained from successful trace data.
    Fast, but less flexible than model-based planning.
    Used for: routine operations, high-frequency decisions.
    """

    def act(self, state: State) -> Action:
        # Lookup in learned policy table
        # Fall back to model-based planning if state is novel
        ...
```

---

## Online Model Learning

The model is never static. Every run updates it:

```python
class OnlineModelLearner:
    """
    Continuously updates the environment model from new traces.

    On each run completion:
      1. Extract state transitions from the trace
      2. Update transition counts T(s, a → s')
      3. Update reward observations R(s, a)
      4. Update observation likelihoods O(o | s, a)
      5. Detect novel states (expand state space if needed)
      6. Recompute derived quantities (value function, policy)
    """

    def learn_from_trace(self, trace: Trace) -> ModelUpdate:
        transitions = self._extract_transitions(trace)
        self.transition_model.update(transitions)

        rewards = self._extract_rewards(trace)
        self.reward_model.update(rewards)

        observations = self._extract_observations(trace)
        self.observation_model.update(observations)

        novel_states = self._detect_novel_states(transitions)
        if novel_states:
            self.state_space.expand(novel_states)

        return ModelUpdate(
            transitions_added=len(transitions),
            novel_states=len(novel_states),
            uncertainty_delta=self._measure_uncertainty_change(),
        )
```

---

## Comparison: Current vs. Formal Model

| Dimension | Current (Trace Analysis) | Formal POMDP Model |
|---|---|---|
| **State** | Implicit (context in trace) | Explicit factored state space |
| **Transitions** | Tool-sequence frequencies | P(s' \| s, a) with uncertainty |
| **Reward** | Binary success/fail | Continuous, multi-factor R(s, a) |
| **Observations** | Raw tool output | P(o \| s, a) with noise model |
| **Uncertainty** | Not tracked | Belief distribution + entropy |
| **Planning** | None (LLM decides ad-hoc) | Value iteration / MCTS / policy |
| **Predictions** | "This sequence worked before" | "If I do A in state S, P(s') = X" |
| **Learning** | Batch re-analysis | Online, incremental |
| **Generalization** | Pattern matching | Compositional (factorized states) |
| **Counterfactuals** | Not possible | "What if I had done B instead?" |

---

## Integration with Nil Architecture

```
                 ┌──────────────────────┐
                 │   Agent (orchestrator)│
                 └──────┬───────────────┘
                        │
        ┌───────────────┼───────────────────┐
        │               │                   │
        v               v                   v
┌──────────────┐ ┌──────────────┐ ┌──────────────────┐
│  Kernel      │ │  Evolution   │ │  Environment     │
│  Runner      │ │  Pipeline    │ │  Model           │
│              │ │              │ │                  │
│ Uses model   │ │ Updates model│ │ States           │
│ for action   │ │ from traces  │ │ Transitions      │
│ selection    │ │ in batch     │ │ Rewards          │
│              │ │              │ │ Belief tracking  │
└──────────────┘ └──────────────┘ └──────────────────┘
                        │                   │
                        └───────────────────┘
                                    │
                                    v
                         ┌──────────────────┐
                         │  PersistentStore │
                         │  (SQLite)        │
                         │                  │
                         │  • episodes      │
                         │  • model params  │
                         │  • belief state  │
                         └──────────────────┘
```

### Agent Methods

```python
class ModelAwareAgent(Agent):
    def __init__(self, ...):
        super().__init__(...)
        self.environment_model = EnvironmentModel(
            state_space=StateSpace(),
            transition_model=TransitionModel(),
            reward_model=RewardModel(domain=self.domain),
            observation_model=ObservationModel(),
        )
        self.belief_tracker = BeliefTracker(self.environment_model.state_space)

    def run(self, task: str, context: dict = None) -> RunOutcome:
        # Initialize belief from task and context
        self.belief_tracker.reset()
        self.belief_tracker.incorporate_task(task, context)

        # Kernel runner uses model for action selection
        outcome = self._runner.execute(
            task=task,
            context=context,
            model=self.environment_model,
            belief=self.belief_tracker,
        )

        # Learn from outcome
        update = self.environment_model.learn_from_trace(outcome.trace)

        return outcome

    def predict_outcome(self, action: str) -> dict:
        """Ask: what happens if I do X in the current state?"""
        state = self.belief_tracker.most_likely_state
        predictions = self.environment_model.transition_model.predict(state, action)
        return {
            "possible_outcomes": predictions,
            "expected_reward": self.environment_model.reward_model.expected(
                state, action
            ),
            "uncertainty": self.belief_tracker.uncertainty,
        }

    def belief_state(self) -> dict:
        """What the agent currently believes about the environment."""
        return {
            "most_likely_state": self.belief_tracker.most_likely_state,
            "uncertainty": self.belief_tracker.uncertainty,
            "top_beliefs": sorted(
                self.belief_tracker.belief.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:5],
        }
```

---

## Implementation Roadmap

| Phase | What | How |
|---|---|---|
| 1 | State space discovery | Factorize trace contexts into discrete states. Clustering + dimensionality reduction on existing traces |
| 2 | Transition counting | Count P(s' \| s, a) from trace history. Store as sparse matrix |
| 3 | Reward model | Formal R(s, a) from success_signal + cost + duration. Replace binary success/fail |
| 4 | Belief tracker | Bayes filter over states. Track uncertainty via entropy |
| 5 | Observation model | P(o \| s, a) — model which observations are diagnostic of which states |
| 6 | Model-based planning | Value iteration / look-ahead for action selection in KernelRunner |
| 7 | MCTS planner | For high-stakes decisions where planning latency is acceptable |
| 8 | Online learning | Incremental model update after every run. Novel state detection |
| 9 | Counterfactual query | "What if I had done X instead?" — simulate alternative trajectory |
| 10 | Visualization | Render state machine graph, belief distribution, transition matrices |

---

## Key Differentiator

**Without environment models:**
The agent sees a pattern and repeats it. It has no concept of "if I do X, Y will probably happen." It cannot reason counterfactually. Every decision is a guess based on surface-level similarity to past experiences. The agent is a mimic, not a modeler.

**With environment models:**
The agent builds an internal model of how its world works — what states exist, how actions change them, what outcomes are valuable, and how certain it is about each belief. It can simulate, predict, and plan. When uncertainty is high, it takes information-gathering actions. When it's low, it acts decisively.

This is the difference between an agent that has **experience** and an agent that has **understanding**.
