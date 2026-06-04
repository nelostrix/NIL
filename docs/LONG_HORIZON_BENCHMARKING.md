# Long-Horizon Benchmarking

> **Status: Implemented** — The full benchmark suite exists in `nil/benchmarks/`. `BenchmarkSuite` / `BenchmarkTask` / `BenchmarkResult` (base framework), `LearningCurveBenchmark` / `LearningAnalyzer`, `HorizonBenchmark`, `RecoveryBenchmark`, `MemoryBenchmark`, `TransferBenchmark`, `SystematicVariationBenchmark`, `BenchmarkRunner` (unified runner with structured JSON export), and `BenchmarkVisualizer` are all implemented.

---

## Background

Most agent benchmarks measure one thing: did the agent complete the task correctly in a single run? This measures accuracy at a point in time — it does not measure learning, adaptation, memory retention, or robustness.

Nil is a *self-improving* system. Its core value proposition is that agents get better over time. To validate this, the benchmark suite measures not just *what an agent does* but *how it improves across runs* — cross-run adaptation curves, long-horizon tasks, recovery from failure, and system-level learning dynamics.

---

## Before Long-Horizon Benchmarking

```
Existing benchmarks measure:

  single_run_accuracy = correct(task) ? 1 : 0

  Limitations:
    • No learning curve (single run, no improvement measurement)
    • No horizon length (all tasks are short, 1-5 steps)
    • No recovery measurement (failure = end, no retry)
    • No memory retention (each run is independent)
    • No transfer learning (same task type, no variation)
    • No robustness (no noise, no tool degradation, no distribution shift)
```

---

## Benchmark Suite

```
┌───────────────────────────────────────────────────────────────────────┐
│                    LONG-HORIZON BENCHMARK SUITE                       │
│                                                                       │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐  │
│  │ Learning Curve  │  │ Long-Horizon   │  │ Recovery & Robustness  │  │
│  │ Benchmarks     │  │ Benchmarks     │  │ Benchmarks             │  │
│  │                │  │                │  │                        │  │
│  │ • Cold start   │  │ • 10-step tasks│  │ • Failure recovery     │  │
│  │ • Learning     │  │ • 25-step tasks│  │ • Tool degradation     │  │
│  │   velocity     │  │ • 50-step tasks│  │ • Distribution shift   │  │
│  │ • Plateau      │  │ • Credit       │  │ • Noisy observations   │  │
│  │   detection    │  │   assignment   │  │ • Adversarial inputs   │  │
│  │ • Sample       │  │   distance     │  │ • Delayed rewards      │  │
│  │   efficiency   │  │                │  │                        │  │
│  └────────────────┘  └────────────────┘  └────────────────────────┘  │
│                                                                       │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐  │
│  │ Memory &        │  │ Transfer &     │  │ System Benchmarks      │  │
│  │ Retention      │  │ Adaptation     │  │                        │  │
│  │                │  │                │  │                        │  │
│  │ • Episodic     │  │ • Same domain  │  │ • Cost per run         │  │
│  │   retention    │  │   new task     │  │ • Latency per run      │  │
│  │ • Skill        │  │ • Similar      │  │ • Memory growth        │  │
│  │   retention    │  │   domain       │  │ • Evolution overhead   │  │
│  │ • Forgetting   │  │ • Novel domain │  │ • Retrieval quality    │  │
│  │   curve        │  │ • Few-shot     │  │ • Compilation ratio    │  │
│  │ • Interference │  │   adaptation   │  │                        │  │
│  └────────────────┘  └────────────────┘  └────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 1. Learning Curve Benchmarks

These are the most important benchmarks for a self-improving system. They measure how the agent improves across runs.

### Core Metric: Learning Velocity

```python
class LearningVelocity:
    """
    Measures how fast the agent improves over runs.

    Metrics:
      - Initial accuracy: score at run 0 (cold start)
      - Learning velocity: slope of accuracy over first N runs
      - Plateau accuracy: asymptotic performance
      - Runs to plateau: how many runs to reach stable performance
      - Sample efficiency: improvement per run (accuracy delta / run)
    """

    def compute(self, run_scores: list[float]) -> LearningReport:
        n = len(run_scores)
        if n < 3:
            return LearningReport(insufficient_data=True)

        # Initial accuracy
        initial = run_scores[0]

        # Learning velocity (linear regression on first 30 runs or all)
        window = min(30, n)
        early_scores = run_scores[:window]
        x = list(range(window))
        slope, intercept = self._linear_regression(x, early_scores)

        # Plateau accuracy (average of last 10 runs or last 20%)
        plateau_window = min(10, max(1, n // 5))
        plateau = sum(run_scores[-plateau_window:]) / plateau_window

        # Runs to plateau (first run where score >= 90% of plateau)
        threshold = 0.9 * plateau
        runs_to_plateau = n
        for i, score in enumerate(run_scores):
            if score >= threshold:
                runs_to_plateau = i + 1
                break

        # Sample efficiency
        total_improvement = plateau - initial
        sample_efficiency = total_improvement / max(n, 1)

        # Forgetting (did accuracy ever significantly drop?)
        drops = self._detect_forgetting_events(run_scores)

        return LearningReport(
            initial_accuracy=round(initial, 4),
            final_accuracy=round(plateau, 4),
            learning_velocity=round(slope, 6),
            runs_to_plateau=runs_to_plateau,
            sample_efficiency=round(sample_efficiency, 6),
            plateau_accuracy=round(plateau, 4),
            forgetting_events=drops,
            total_runs=n,
        )

    def _detect_forgetting_events(self, scores: list[float]) -> list[ForgettingEvent]:
        """
        Detect significant drops in accuracy that indicate forgetting.

        A forgetting event is: accuracy drops by >15% and stays
        low for at least 3 runs before recovering.
        """
        events = []
        i = 0
        while i < len(scores) - 3:
            if scores[i + 1] < scores[i] * 0.85:
                drop_start = i
                drop_end = i + 1
                while drop_end < len(scores) and scores[drop_end] < scores[i] * 0.85:
                    drop_end += 1
                if drop_end - drop_start >= 3:
                    events.append(ForgettingEvent(
                        run=drop_start,
                        drop=scores[i] - min(scores[drop_start:drop_end]),
                        duration=drop_end - drop_start,
                    ))
                i = drop_end
            else:
                i += 1
        return events
```

### Learning Curve Test Suite

```python
class LearningCurveBenchmarks:
    """
    Benchmark tasks designed to measure learning over repeated runs.

    Each task family has:
      - A training set (runs 1-50): learn the pattern
      - A test set (runs 51-60): measure generalization
      - Held-out variations: measure transfer
    """

    benchmarks = {
        "tool_selection": ToolSelectionBenchmark(
            name="learn_which_tool_works",
            description="Agent must learn which of 5 tools solves each task type",
            n_tasks=50,
            n_tools=5,
            n_adaptation_runs=30,
            metrics=["accuracy", "exploration_rate", "regret"],
        ),
        "sequential_reasoning": SequentialReasoningBenchmark(
            name="learn_multi_step_sequence",
            description="Agent must learn the correct 5-step tool sequence",
            n_tasks=50,
            sequence_length=5,
            n_alternatives=3,   # 3 plausible but wrong sequences
            n_adaptation_runs=40,
            metrics=["sequence_accuracy", "step_efficiency"],
        ),
        "context_dependence": ContextDependenceBenchmark(
            name="learn_context_dependent_action",
            description="Correct action depends on context the agent must learn",
            n_tasks=50,
            n_contexts=4,
            n_adaptation_runs=30,
            metrics=["context_accuracy", "adaptation_speed"],
        ),
    }
```

### Example Learning Curve Output

```
Learning Curve — "learn_which_tool_works"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Run    Accuracy    Tool Used    Regret
──────────────────────────────────────────────────────────
  1    0.20       search_logs   0.80   (cold start)
  5    0.45       search_logs   0.55   (early learning)
 10    0.70       analyze       0.30   (learning which tool)
 15    0.85       analyze       0.15   (converging)
 20    0.90       analyze       0.10   (stable)
 25    0.92       analyze       0.08
 30    0.91       analyze       0.09
 35    0.93       analyze       0.07   (plateau reached)
 40    0.92       analyze       0.08
 45    0.94       analyze       0.06
 50    0.93       analyze       0.07

Learning Velocity:   0.031  (accuracy gain per run, first 30 runs)
Plateau Accuracy:    0.93   (average of last 10 runs)
Runs to Plateau:     22     (reached 90% of plateau)
Sample Efficiency:   0.019  (total improvement / total runs)
Forgetting Events:   0      (no significant drops)
```

---

## 2. Long-Horizon Task Benchmarks

These measure the agent's ability to execute extended sequences without error accumulation.

```python
class LongHorizonBenchmark:
    """
    Tasks that require many sequential steps to complete.

    Key challenge: error propagation. A mistake at step 3
    must be recovered from at step 4+, not just start over.
    """

    task_lengths = [10, 25, 50, 100]
    n_tasks_per_length = 20

    def run(self, agent: Agent) -> dict[int, HorizonResult]:
        results = {}
        for length in self.task_lengths:
            scores = []
            error_recovery = []
            for _ in range(self.n_tasks_per_length):
                task = self._generate_horizon_task(length)
                outcome = agent.run(task.description)

                # Score: proportion of steps correct
                step_correctness = self._score_step_accuracy(outcome, task)

                # Error recovery: if a step failed, did later steps compensate?
                recovery = self._score_error_recovery(outcome, task)

                scores.append(step_correctness)
                error_recovery.append(recovery)

            results[length] = HorizonResult(
                task_length=length,
                mean_accuracy=sum(scores) / len(scores),
                std_accuracy=self._std(scores),
                error_recovery_rate=sum(error_recovery) / len(error_recovery),
                completion_rate=sum(1 for s in scores if s > 0.8) / len(scores),
            )
        return results

    def _score_step_accuracy(self, outcome, task) -> float:
        """What fraction of required steps were executed correctly?"""
        correct = 0
        for required_step in task.required_steps:
            if self._step_matches(outcome, required_step):
                correct += 1
        return correct / len(task.required_steps)

    def _score_error_recovery(self, outcome, task) -> float:
        """
        If the agent made a mistake, did it recover?

        Recovery score:
          0.0: error → dead end
          0.5: error → retry same thing
          1.0: error → alternative approach → success
        """
        if not outcome.had_errors:
            return 1.0  # no errors to recover from

        recovery_score = 0.0
        for error_step in outcome.error_steps:
            subsequent = outcome.steps_after(error_step)
            if self._reached_goal(subsequent):
                recovery_score += 1.0
            elif self._changed_strategy(subsequent, error_step):
                recovery_score += 0.7
            elif self._retried_same(subsequent, error_step):
                recovery_score += 0.3

        return recovery_score / len(outcome.error_steps)
```

---

## 3. Recovery & Robustness Benchmarks

These measure the agent's ability to handle failure, noise, and changing conditions.

```python
class RecoveryBenchmark:
    """
    Benchmarks that deliberately inject failures and measure recovery.

    Injection types:
      - Tool failure: a tool returns an error unexpectedly
      - Tool degradation: a tool becomes slower or less accurate
      - Observation noise: tool outputs are corrupted
      - Distribution shift: task type changes mid-benchmark
      - Adversarial inputs: deliberately misleading observations
      - Delayed rewards: success signal arrives 5+ steps after correct action
    """

    def run(self, agent: Agent) -> RecoveryReport:
        results = {}

        # Test 1: Tool failure recovery
        results["tool_failure"] = self._test_tool_failure(agent)

        # Test 2: Tool degradation
        results["tool_degradation"] = self._test_tool_degradation(agent)

        # Test 3: Distribution shift
        results["distribution_shift"] = self._test_distribution_shift(agent)

        # Test 4: Delayed rewards
        results["delayed_rewards"] = self._test_delayed_rewards(agent)

        return RecoveryReport(
            overall_recovery_score=self._composite(results),
            per_test=results,
        )

    def _test_tool_failure(self, agent: Agent) -> TestResult:
        """
        Run tasks where a specific tool fails 50% of the time.

        Good agent: detects failure, tries alternative tool.
        Bad agent: retries failing tool repeatedly, or fails entirely.
        """
        with self._inject_tool_failure(tool_name="search_logs", failure_rate=0.5):
            outcomes = []
            for _ in range(20):
                outcome = agent.run("Investigate API error")
                outcomes.append(outcome)

        # Metrics:
        #   - Did agent switch tools after failure? (adaptation)
        #   - How many retries before switching? (efficiency)
        #   - Did agent still complete the task? (robustness)
        return TestResult(
            completion_rate=sum(o.success for o in outcomes) / len(outcomes),
            tool_switch_latency=self._avg_steps_before_tool_switch(outcomes),
            unnecessary_retries=sum(self._count_unnecessary_retries(o) for o in outcomes),
        )

    def _test_distribution_shift(self, agent: Agent) -> TestResult:
        """
        Train on task type A for 20 runs, then switch to task type B.

        Good agent: detects shift, adapts strategy, maintains performance.
        Bad agent: keeps using old strategy, performance collapses.
        """
        phase1_tasks = self._generate_tasks("type_a", n=20)
        phase2_tasks = self._generate_tasks("type_b", n=20)

        phase1_scores = []
        for task in phase1_tasks:
            phase1_scores.append(agent.run(task).success)

        phase2_scores = []
        for task in phase2_tasks:
            phase2_scores.append(agent.run(task).success)

        # Metrics:
        #   - Phase 1 plateau accuracy
        #   - Accuracy drop at shift point
        #   - Runs to recover to 80% of phase 1 plateau
        plateau_1 = sum(phase1_scores[-5:]) / 5
        drop = plateau_1 - phase2_scores[0]
        recovery_runs = self._runs_to_recover(phase2_scores, 0.8 * plateau_1)

        return TestResult(
            pre_shift_accuracy=plateau_1,
            shift_drop=drop,
            recovery_runs=recovery_runs,
            post_shift_accuracy=sum(phase2_scores[-5:]) / 5,
        )
```

---

## 4. Memory & Retention Benchmarks

These measure whether the agent actually retains what it learns across runs.

```python
class MemoryRetentionBenchmark:
    """
    Measures how well the agent retains learned knowledge over time.

    Protocols:
      - Immediate recall: test right after learning
      - Delayed recall: test after N intervening tasks
      - Interference: test after learning conflicting patterns
      - Forgetting curve: accuracy as function of elapsed runs
    """

    def forgetting_curve(
        self,
        agent: Agent,
        learn_task: Callable,
        test_task: Callable,
        delays: list[int] = [1, 5, 10, 25, 50, 100],
    ) -> ForgettingCurve:
        """
        Measure how accuracy decays with delay between learning and testing.

        The agent learns a pattern, then is tested on it after
        varying numbers of intervening tasks.
        """
        # Train on the learning task
        for _ in range(5):
            agent.run(learn_task())

        results = []
        for delay in delays:
            # Run intervening tasks (different domain)
            for _ in range(delay):
                agent.run(self._distractor_task())

            # Test retention
            test_outcomes = [agent.run(test_task()) for _ in range(3)]
            accuracy = sum(o.success for o in test_outcomes) / len(test_outcomes)
            results.append((delay, accuracy))

        return ForgettingCurve(
            task=learn_task.__name__,
            points=results,
            half_life=self._estimate_half_life(results),
            retention_at_100=results[-1][1] if len(results) >= 6 else None,
        )

    def interference_test(
        self,
        agent: Agent,
        pattern_a: Callable,
        pattern_b: Callable,  # conflicts with pattern A
        test: Callable,
    ) -> InterferenceResult:
        """
        Measure catastrophic forgetting.

        Learn A → Learn B (which conflicts) → Test A.

        If accuracy on A drops significantly after learning B,
        the system suffers from interference.
        """
        # Phase 1: Learn A
        for _ in range(10):
            agent.run(pattern_a())
        accuracy_a_before = self._measure_accuracy(agent, test, n=5)

        # Phase 2: Learn B (conflicting)
        for _ in range(10):
            agent.run(pattern_b())
        accuracy_b = self._measure_accuracy(agent, lambda: pattern_b(), n=5)

        # Phase 3: Test A
        accuracy_a_after = self._measure_accuracy(agent, test, n=5)

        interference = accuracy_a_before - accuracy_a_after

        return InterferenceResult(
            accuracy_a_before=accuracy_a_before,
            accuracy_a_after=accuracy_a_after,
            interference_drop=interference,
            interference_ratio=interference / accuracy_a_before if accuracy_a_before > 0 else 1.0,
        )
```

---

## 5. Transfer & Adaptation Benchmarks

These measure whether the agent can apply what it learned in one domain to related domains.

```python
class TransferBenchmark:
    """
    Measures transfer learning across domains.

    Levels:
      - Same domain, new task (e.g., new type of incident)
      - Similar domain (e.g., incident-response → monitoring)
      - Different domain (e.g., incident-response → data-pipeline)
      - Zero-shot: test on unseen domain without any training
      - Few-shot: 1, 3, 5, 10 demonstrations in new domain
    """

    def run(self, agent: Agent, source_domain: Domain) -> TransferReport:
        # Train on source domain
        for _ in range(20):
            agent.run(self._random_task(source_domain))

        results = {}
        for target_domain, similarity in [
            ("same_new_task", 1.0),
            ("similar_domain", 0.6),
            ("different_domain", 0.2),
        ]:
            domain = self._get_domain(target_domain)

            # Zero-shot
            zero_shot = self._measure_accuracy(agent, domain, n_shots=0)

            # Few-shot (1, 3, 5, 10)
            few_shot_results = {}
            for n in [1, 3, 5, 10]:
                # Reset agent's working memory (keep compiled skills)
                agent.reset_working_memory()
                for shot in range(n):
                    task = self._random_task(domain)
                    agent.run(task)
                accuracy = self._measure_accuracy(agent, domain, n=10)
                few_shot_results[n] = accuracy

            results[target_domain] = TransferResult(
                domain=target_domain,
                similarity=similarity,
                zero_shot_accuracy=zero_shot,
                few_shot_accuracy=few_shot_results,
                transfer_efficiency=self._transfer_efficiency(
                    source_domain, domain, zero_shot
                ),
            )

        return TransferReport(
            source_domain=source_domain.name,
            transfer_results=results,
            overall_transfer_score=self._composite_transfer(results),
        )
```

---

## 6. System Benchmarks

These measure the system's operational characteristics, not just task accuracy.

```python
class SystemBenchmark:
    """
    Measures operational characteristics of the agent system.

    These matter for production deployment.
    """

    def run(self, agent: Agent, n_runs: int = 100) -> SystemReport:
        runs = [agent.run(self._task()) for _ in range(n_runs)]

        # Cost metrics
        costs = [r.total_cost for r in runs]

        # Latency metrics
        latencies = [r.duration for r in runs]

        # Memory growth
        memory_before = self._measure_memory_size(agent)
        # Run more tasks
        for _ in range(n_runs):
            agent.run(self._task())
        memory_after = self._measure_memory_size(agent)
        memory_growth = (memory_after - memory_before) / n_runs

        # Evolution overhead
        evolution_times = [
            r.evolution_duration for r in runs if r.evolution_triggered
        ]

        # Retrieval quality (if RAC is active)
        retrieval_stats = self._measure_retrieval_quality(agent)

        return SystemReport(
            # Cost
            cost_per_run=sum(costs) / len(costs),
            cost_p50=self._percentile(costs, 0.5),
            cost_p95=self._percentile(costs, 0.95),
            # Latency
            latency_per_run=sum(latencies) / len(latencies),
            latency_p50=self._percentile(latencies, 0.5),
            latency_p95=self._percentile(latencies, 0.95),
            # Memory
            memory_growth_per_run=memory_growth,
            total_memory_after=memory_after,
            # Evolution
            avg_evolution_time=sum(evolution_times) / len(evolution_times)
            if evolution_times else 0,
            evolution_overhead_pct=(
                sum(evolution_times) / sum(latencies) * 100
                if sum(latencies) > 0 else 0
            ),
            # Retrieval
            retrieval_recall=retrieval_stats.get("recall@5", None),
            retrieval_latency=retrieval_stats.get("avg_latency_ms", None),
        )
```

---

## 7. Unified Benchmark Runner

```python
class NilBenchmarkSuite:
    """
    Runs the complete long-horizon benchmark suite.

    Usage:
        suite = NilBenchmarkSuite()
        report = suite.run(agent, output_dir="./benchmark_results")

        print(report.summary())

        # Plot learning curves
        report.plot_learning_curves()
        report.plot_forgetting_curves()
        report.plot_transfer_matrix()
    """

    def __init__(self):
        self.learning = LearningCurveBenchmarks()
        self.horizon = LongHorizonBenchmark()
        self.recovery = RecoveryBenchmark()
        self.memory = MemoryRetentionBenchmark()
        self.transfer = TransferBenchmark()
        self.system = SystemBenchmark()

    def run(
        self,
        agent: Agent,
        output_dir: str = "./benchmark_results",
    ) -> BenchmarkReport:
        report = BenchmarkReport(agent_name=str(agent))

        # Phase 1: Learning curves
        report.learning_curves = self.learning.run(agent)

        # Phase 2: Long-horizon tasks
        report.horizon_results = self.horizon.run(agent)

        # Phase 3: Recovery & robustness
        report.recovery_results = self.recovery.run(agent)

        # Phase 4: Memory retention
        report.memory_results = self.memory.run(agent)

        # Phase 5: Transfer learning
        report.transfer_results = self.transfer.run(agent)

        # Phase 6: System metrics
        report.system_metrics = self.system.run(agent, n_runs=100)

        # Phase 7: Composite scores
        report.composite = self._compute_composite(report)

        # Save and export
        self._save_report(report, output_dir)

        return report

    def _compute_composite(self, report: BenchmarkReport) -> CompositeScore:
        """
        Weighted composite across all benchmark dimensions.

        Weights are configurable depending on which aspects
        are most important for the use case.
        """
        return CompositeScore(
            learning_score=report.learning_curves.composite * 0.25,
            horizon_score=report.horizon_results.composite * 0.20,
            recovery_score=report.recovery_results.composite * 0.20,
            memory_score=report.memory_results.composite * 0.15,
            transfer_score=report.transfer_results.composite * 0.10,
            system_score=report.system_metrics.composite * 0.10,
        )
```

---

## 8. Visualization

```python
class BenchmarkVisualizer:
    """Generate visualizations from benchmark results."""

    def plot_learning_curve(
        self, curve: LearningReport, output_path: str
    ):
        """
        Standard learning curve plot:
          X-axis: run number
          Y-axis: accuracy
          Overlay: learning velocity (regression line)
          Annotate: runs to plateau, forgetting events
        """
        ...

    def plot_forgetting_curve(
        self, curve: ForgettingCurve, output_path: str
    ):
        """
        Forgetting curve plot:
          X-axis: delay (number of intervening runs)
          Y-axis: retention accuracy
          Overlay: exponential decay fit with half-life annotation
        """
        ...

    def plot_transfer_matrix(
        self, report: TransferReport, output_path: str
    ):
        """
        Transfer matrix heatmap:
          Rows: source domains
          Columns: target domains
          Color: zero-shot accuracy
          Annotate: few-shot improvement
        """
        ...

    def plot_comparison(
        self, reports: list[BenchmarkReport], labels: list[str], output_path: str
    ):
        """
        Side-by-side comparison of multiple agents/configurations.

        Radar chart with composite dimensions:
          - Learning velocity
          - Horizon capability
          - Recovery
          - Memory retention
          - Transfer
          - System efficiency
        """
        ...
```

---

## Implementation Status

| Phase | What | Status |
|---|---|---|
| 1 | Learning curve benchmarks | **Done** — repeated runs on same task family, accuracy vs. run number |
| 2 | Long-horizon tasks | **Done** — 10/25/50/100-step tasks with step-level scoring |
| 3 | Error recovery measurement | **Done** — inject tool failures, measure strategy switching and completion |
| 4 | Distribution shift | **Done** — task type changes mid-benchmark, drop and recovery speed |
| 5 | Forgetting curves | **Done** — delayed recall at [1, 5, 10, 25, 50, 100] intervening runs |
| 6 | Interference testing | **Done** — Learn A → Learn conflicting B → Test A, catastrophic forgetting measurement |
| 7 | Transfer benchmarks | **Done** — zero-shot and few-shot (1/3/5/10) transfer |
| 8 | System benchmarks | **Done** — cost, latency, memory growth, evolution overhead, retrieval quality |
| 9 | Unified runner | **Done** — single entry point, structured report, JSON export |
| 10 | Visualizer | **Done** — learning curves, forgetting curves, transfer matrices, radar charts |
| 11 | CI integration | **Not implemented** — no nightly regression tracking yet |

---

## Key Differentiator

**Single-run benchmarks:**
"Did the agent complete this one task?" This tells you if the agent works at a point in time. It tells you nothing about whether the agent is improving, forgetting, or adapting.

**Long-horizon benchmarks:**
"How does the agent improve across 50 runs? Does it remember what it learned 100 runs ago? Can it transfer knowledge to a new domain? Does it recover when tools fail?" These are the questions that matter for a self-improving system. They measure not just *performance* but *learning dynamics* — and learning dynamics are the entire value proposition of the Nil framework.
