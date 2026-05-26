# Runtime Observability

**Beyond Logs**

Nil agents execute autonomously, evolve their own strategies, and modify their own prompts. When something goes wrong, you cannot just look at the code — the agent may have changed its own behavior in ways the original developer never anticipated.

This is the observability problem unique to self-modifying systems. You need visibility not just into *what the agent did* but *why it decided to do it*, *what knowledge it used*, *how its strategy has evolved*, and *what changed between versions*.

This document proposes a comprehensive observability stack: structured traces, flame graphs, token timelines, reasoning trees, memory inspection, evolution diffs, strategy lineage, and a visual debugger.

---

## Current State

```
Current observability:

  Log lines like:
    INFO:nil:Starting run incident-response-run-3-...
    INFO:nil:Run completed: success=True, duration=12.5s, steps=4
    INFO:nil.evolution:Evolution triggered after 10 runs
    INFO:nil.evolution:Applied 2 deltas → strategy v3

  What's missing:
    • No reasoning trace (why did the LLM choose that tool?)
    • No token visualization (where did tokens go?)
    • No memory inspection (what knowledge was retrieved?)
    • No evolution diff (what exactly changed in the strategy?)
    • No strategy lineage (how did we get from v1 to v7?)
    • No flame graph (where is time being spent?)
    • No live debugging (pause, inspect, resume)
```

---

## Proposed Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      OBSERVABILITY STACK                          │
│                                                                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────────┐  │
│  │  Trace   │ │  Flame   │ │  Token   │ │  Reasoning Tree    │  │
│  │  Viewer  │ │  Graphs  │ │  Timeline│ │  Visualizer        │  │
│  └──────────┘ └──────────┘ └──────────┘ └────────────────────┘  │
│                                                                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────────┐  │
│  │  Memory  │ │ Evolution│ │ Strategy │ │  Live Debugger     │  │
│  │  Browser │ │  Diff    │ │ Lineage  │ │  (pause/inspect)   │  │
│  └──────────┘ └──────────┘ └──────────┘ └────────────────────┘  │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                    DATA LAYER                                │  │
│  │                                                              │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐  │  │
│  │  │ Run      │ │ Step     │ │ LLM      │ │ Evolution    │  │  │
│  │  │ Store    │ │ Store    │ │ Call     │ │ Event Store  │  │  │
│  │  │          │ │          │ │ Store    │ │              │  │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────────┘  │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 1. Structured Event Store

Every significant event in the agent's lifecycle is recorded as a structured event:

```python
@dataclass
class ObservabilityEvent:
    """Every event in the agent's runtime, stored for query and visualization."""

    event_id: UUID
    event_type: EventType
    timestamp: float
    agent_id: str
    domain: str
    run_id: str

    # Event-specific payload
    payload: dict

    # Parent-child relationships for tree views
    parent_event_id: Optional[UUID] = None
    span_id: str = ""                   # for flame graph grouping
    trace_id: str = ""                  # for distributed tracing


class EventType(Enum):
    # Run lifecycle
    RUN_STARTED            = "run_started"
    RUN_COMPLETED          = "run_completed"
    RUN_FAILED             = "run_failed"

    # LLM interactions
    LLM_CALL_STARTED       = "llm_call_started"
    LLM_CALL_COMPLETED     = "llm_call_completed"
    LLM_CALL_FAILED        = "llm_call_failed"
    LLM_TOKEN_USAGE        = "llm_token_usage"

    # Tool calls
    TOOL_CALL_STARTED      = "tool_call_started"
    TOOL_CALL_COMPLETED    = "tool_call_completed"
    TOOL_CALL_FAILED       = "tool_call_failed"

    # Memory operations
    MEMORY_READ            = "memory_read"
    MEMORY_WRITE           = "memory_write"
    MEMORY_RETRIEVAL       = "memory_retrieval"
    MEMORY_CONSOLIDATION   = "memory_consolidation"

    # Retrieval (RAC)
    RETRIEVAL_QUERY        = "retrieval_query"
    RETRIEVAL_RESULT       = "retrieval_result"

    # Evolution
    EVOLUTION_TRIGGERED    = "evolution_triggered"
    ANALYSIS_COMPLETED     = "analysis_completed"
    SKILL_EXTRACTED        = "skill_extracted"
    GRADIENT_COMPUTED      = "gradient_computed"
    DELTA_APPLIED          = "delta_applied"
    STRATEGY_PROMOTED      = "strategy_promoted"
    DRIFT_DETECTED         = "drift_detected"
    RECOVERY_TRIGGERED     = "recovery_triggered"
```


### Event Recorder

```python
class EventRecorder:
    """
    Records all observability events to a structured store.

    Usage:
        recorder = EventRecorder()
        recorder.record(LLM_CALL_STARTED, run_id="...", payload={...})
        recorder.record(TOOL_CALL_COMPLETED, run_id="...", payload={...})
    """

    def __init__(self, store: ObservabilityStore):
        self.store = store
        self._current_span_stack: list[str] = []

    def record(
        self,
        event_type: EventType,
        run_id: str,
        payload: dict,
        parent_id: Optional[UUID] = None,
    ) -> ObservabilityEvent:
        event = ObservabilityEvent(
            event_id=uuid4(),
            event_type=event_type,
            timestamp=time.time(),
            agent_id=self.agent_id,
            domain=self.domain,
            run_id=run_id,
            payload=payload,
            parent_event_id=parent_id,
            span_id=self._current_span_stack[-1] if self._current_span_stack else "",
            trace_id=run_id,
        )
        self.store.write(event)
        return event

    def start_span(self, span_name: str) -> str:
        """Start a new span (for flame graph nesting)."""
        span_id = f"{span_name}-{uuid4().hex[:8]}"
        self._current_span_stack.append(span_id)
        return span_id

    def end_span(self):
        """End the current span."""
        if self._current_span_stack:
            self._current_span_stack.pop()
```

---

## 2. Trace Viewer

A rich trace viewer showing the full execution of a run:

```
Run: incident-response-run-3-1685029200
Task: "API latency spike on /v2/users"
Status: ✅ Success    Duration: 12.4s    Cost: $0.042

Timeline:
  ──────────────────────────────────────────────────────────────►
  [0.0s]  LLM: Analyze task → decompose into steps
  [1.2s]  Tool: search_logs(query="API latency /v2/users", timerange="30m")
  [2.8s]  LLM: logs show timeout errors → query metrics
  [3.1s]  Tool: query_metrics(endpoint="/v2/users", metric="latency_p99")
  [4.5s]  LLM: metrics confirm latency spike → check DB connections
  [4.8s]  Tool: check_db(metric="connection_pool_usage")
  [6.2s]  LLM: DB connection pool at 98% → root cause identified
  [6.5s]  LLM: Plan fix → drain connections → restart service
  [7.0s]  Tool: drain_connections(service="api-gateway")
  [8.3s]  Tool: restart_service(service="api-gateway")
  [9.1s]  LLM: verify recovery
  [9.4s]  Tool: check_endpoint(path="/v2/users/health")
  [10.2s] LLM: endpoint healthy → success

Reasoning (expandable):
  ┌────────────────────────────────────────────────────────────────┐
  │ Step 3 — LLM Reasoning:                                       │
  │ "The logs show timeout errors with 'connection refused'       │
  │  messages. This suggests the connection pool is exhausted.    │
  │  I should check DB metrics to confirm before acting."         │
  │                                                               │
  │ Tools considered: analyze_logs, query_metrics, check_db       │
  │ Retrieval used: failure_traces/connection_pool_exhaustion     │
  │ Skill applied: search_logs_then_analyze (92% confidence)      │
  └────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
class TraceViewer:
    """
    Generates rich trace views from stored events.

    Supports:
      - Tree view (parent-child nesting)
      - Timeline view (chronological with duration bars)
      - Flame view (nested span visualization)
      - JSON export (for external tools)
    """

    def render_timeline(self, run_id: str) -> TimelineView:
        events = self.store.get_events(run_id=run_id)

        # Group by span
        spans = self._build_spans(events)

        # Calculate timing
        start_time = min(e.timestamp for e in events)
        max_time = max(e.timestamp for e in events)
        total_duration = max_time - start_time

        rows = []
        for span in spans:
            row = TimelineRow(
                span_name=span.name,
                depth=span.depth,
                start_offset=(span.start - start_time) / total_duration * 100,
                duration_pct=(span.end - span.start) / total_duration * 100,
                duration_s=round(span.end - span.start, 2),
                status=span.status,
                # Expandable details
                reasoning=span.reasoning,
                tokens_used=span.tokens,
                cost=span.cost,
            )
            rows.append(row)

        return TimelineView(
            run_id=run_id,
            total_duration=round(total_duration, 2),
            total_cost=sum(r.cost for r in rows),
            rows=rows,
        )
```

---

## 3. Flame Graphs

Flame graphs show where time is spent during a run — critical for identifying bottlenecks:

```
Flame Graph — Run: incident-response-run-3
─────────────────────────────────────────────────────────────────

  Total: 12.4s

  ┌──────────────────────────────────────────────────────────┐
  │ LLM Think (4.2s)                                         │
  │ ┌──────────────┐ ┌──────────────┐ ┌────────────────────┐ │
  │ │ search_logs  │ │ query_metrics│ │ check_db           │ │
  │ │ (1.6s)       │ │ (1.4s)       │ │ (1.4s)             │ │
  │ └──────────────┘ └──────────────┘ └────────────────────┘ │
  │ ┌──────────────┐ ┌──────────────┐ ┌────────────────────┐ │
  │ │ drain_conns  │ │ restart_svc  │ │ check_endpoint     │ │
  │ │ (1.3s)       │ │ (1.1s)       │ │ (0.8s)             │ │
  │ └──────────────┘ └──────────────┘ └────────────────────┘ │
  │ ┌──────────────────────────────────────────────────────┐ │
  │ │ Evolution (0.8s)                                     │ │
  │ │ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────────────┐ │ │
  │ │ │Analyze │ │Extract │ │Gradient│ │ Evaluate       │ │ │
  │ │ │(0.3s)  │ │(0.2s)  │ │(0.2s)  │ │ (0.1s)         │ │ │
  │ │ └────────┘ └────────┘ └────────┘ └────────────────┘ │ │
  │ └──────────────────────────────────────────────────────┘ │
  └──────────────────────────────────────────────────────────┘

  Hot spots:
    • LLM calls: 4.2s (34%) — consider cheaper model for routine steps
    • Tool: restart_service: 1.1s — slowest individual tool
    • Retrieval: 0.15s — acceptable latency
```

### Implementation

```python
class FlameGraphGenerator:
    """
    Generates flame graph data from observability events.

    Output formats:
      - Interactive HTML (d3-flame-graph)
      - SVG (static)
      - JSON data (for embedding in dashboards)
    """

    def generate(self, run_id: str) -> FlameGraph:
        events = self.store.get_events(run_id=run_id)
        spans = self._build_span_tree(events)

        root = FlameNode(
            name=f"Run: {run_id[:16]}",
            value=spans.total_duration * 1000,  # ms
            children=[],
        )

        for span in spans.top_level:
            root.children.append(self._span_to_node(span))

        return FlameGraph(
            run_id=run_id,
            root=root,
            total_ms=spans.total_duration * 1000,
            metadata={
                "total_cost": spans.total_cost,
                "n_steps": spans.n_steps,
                "n_tool_calls": spans.n_tool_calls,
                "n_llm_calls": spans.n_llm_calls,
            },
        )

    def _span_to_node(self, span: Span) -> FlameNode:
        """Convert a span to a flame graph node with children."""
        node = FlameNode(
            name=f"{span.name} ({span.duration_s:.2f}s)",
            value=int(span.duration_ms),
            metadata={
                "status": span.status,
                "cost": span.cost,
                "tokens": span.tokens,
            },
        )
        for child in span.children:
            node.children.append(self._span_to_node(child))
        return node
```

---

## 4. Token Timeline

Shows exactly how tokens are consumed across a run — critical for cost optimization:

```
Token Timeline — Run: incident-response-run-3
─────────────────────────────────────────────────────────────────

  Total tokens: 12,450 (8,200 in / 4,250 out)
  Total cost:   $0.042

  Step  Token Usage                    In      Out    Cost
  ───────────────────────────────────────────────────────────
    1   System prompt                   500      0    $0.0015
        + Skill injection               850      0    $0.0026
        + Retrieved context           1,200      0    $0.0036
        ───────────────────────────────────
        Total:                        2,550

    2   LLM: decompose task           2,550    620    $0.0105
    3   Tool: search_logs                 0      0    $0.0000  (no LLM)
    4   LLM: analyze logs             3,170    480    $0.0122
    5   Tool: query_metrics               0      0    $0.0000
    6   Tool: check_db                    0      0    $0.0000
    7   LLM: diagnose                 3,650    890    $0.0154
    8   LLM: plan fix                 4,540    760    $0.0168
    9   Tool: drain_connections           0      0    $0.0000
   10   Tool: restart_service             0      0    $0.0000
   11   LLM: verify                   5,300    1,100  $0.0205
   12   LLM: final answer             6,400    400    $0.0192
   13   Evolution pipeline               300    700    $0.0030

  Token allocation:
    ████████████████████████████████░░░ System prompt + context
    ██████████████████████████████████████ LLM reasoning
    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ Tool I/O (zero)
    ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ Evolution

  Recommendations:
    • Retrieved context (1,200 tokens) — consider compact mode
    • Skill injection (850 tokens) — could be trimmed
    • Evolution overhead is minimal (0.3% of tokens)
```

### Implementation

```python
class TokenTimeline:
    """
    Tracks token usage per step and visualizes allocation.
    """

    def compute(self, events: list[ObservabilityEvent]) -> TokenReport:
        steps = []
        total_in = 0
        total_out = 0

        for event in events:
            if event.event_type == EventType.LLM_CALL_COMPLETED:
                tokens_in = event.payload.get("tokens_in", 0)
                tokens_out = event.payload.get("tokens_out", 0)
                step = event.payload.get("step_number", 0)
                step_type = "llm"
                cost = self._estimate_cost(tokens_in, tokens_out)

                steps.append(TokenStep(
                    step_number=step,
                    step_type=step_type,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    cost=cost,
                    description=event.payload.get("description", ""),
                ))

                total_in += tokens_in
                total_out += tokens_out

            elif event.event_type == EventType.MEMORY_RETRIEVAL:
                # Retrieved context tokens
                context_tokens = event.payload.get("context_tokens", 0)
                steps.append(TokenStep(
                    step_number=event.payload.get("step_number", 0),
                    step_type="retrieval",
                    tokens_in=context_tokens,
                    tokens_out=0,
                    cost=0,
                    description=f"Retrieved: {event.payload.get('source', 'unknown')}",
                ))
                total_in += context_tokens

        return TokenReport(
            total_tokens_in=total_in,
            total_tokens_out=total_out,
            total_cost=self._estimate_cost(total_in, total_out),
            steps=sorted(steps, key=lambda s: s.step_number),
            breakdown={
                "system_prompt": self._sum_by_type(steps, "system"),
                "llm_reasoning": self._sum_by_type(steps, "llm"),
                "retrieved_context": self._sum_by_type(steps, "retrieval"),
                "evolution": self._sum_by_type(steps, "evolution"),
            },
        )
```

---

## 5. Reasoning Tree Visualizer

Shows the LLM's reasoning process — what it considered, what it chose, and why:

```
Reasoning Tree — Step 3: Diagnostic Decision
────────────────────────────────────────────────────────────────

  ❓ What is causing the API latency spike?

  ├── Hypothesis A: Connection pool exhaustion
  │     Evidence: timeout errors in logs
  │     Confidence: 0.65
  │     Action: check_db(metric="connection_pool_usage")
  │
  ├── Hypothesis B: Database replica lag
  │     Evidence: no direct evidence yet
  │     Confidence: 0.25
  │     Action: query_metrics(metric="replica_lag")
  │
  └── Hypothesis C: Traffic spike
        Evidence: no direct evidence yet
        Confidence: 0.10
        Action: query_metrics(metric="request_rate")

  🏆 Chosen: Hypothesis A
     Rationale: "Timeout errors in logs strongly suggest
                connection pool exhaustion. Checking pool
                usage will confirm or rule out the most
                likely cause first."
     Influence: Skill "investigate_timeout" triggered
     Retrieved: failure_trace/connection_pool_exhaustion/run-42
```

### Implementation

```python
class ReasoningTree:
    """
    Captures and visualizes the LLM's reasoning process.

    Each LLM call records:
      - The question/decision being made
      - Options considered (tools, hypotheses, actions)
      - Evidence for each option
      - Chosen option and rationale
      - Influences (retrieved memories, applied skills)
    """

    def capture(
        self,
        run_id: str,
        step_number: int,
        prompt: str,           # The LLM prompt
        response: str,         # The LLM response
        options: list[str],    # What was considered
        chosen: str,           # What was chosen
        rationale: str,        # Why it was chosen
        influences: list[str], # Skills/memories that influenced this
    ) -> ReasoningNode:
        node = ReasoningNode(
            run_id=run_id,
            step_number=step_number,
            prompt_preview=prompt[:200],
            response_preview=response[:200],
            options=options,
            chosen=chosen,
            rationale=rationale,
            influences=influences,
        )
        self.store.write_reasoning(node)
        return node

    def render_tree(self, run_id: str) -> str:
        """Render the reasoning tree as an interactive HTML view."""
        nodes = self.store.get_reasoning_nodes(run_id)
        # Generate collapsible tree view
        ...
```

---

## 6. Memory Browser

Inspect what the agent currently knows — skills, strategies, facts, constraints:

```
Memory Browser — Domain: incident-response
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Skills (7 active, 2 deprecated)
  ──────────────────────────────────────────────────────────────
  ✓ search_logs_then_analyze        92%     24 uses   active
  ✓ drain_then_restart             88%     12 uses   active
  ✓ diagnose_connection_pool        85%      8 uses   active
  ✓ check_logs_first                80%     18 uses   active
  ⚠ escalate_if_critical            72%      5 uses   active
  ✗ restart_then_verify            34%      3 uses   deprecated
  ✗ restart_without_draining       12%      6 uses   deprecated

  Click for details:
  ┌────────────────────────────────────────────────────────────┐
  │ Skill: drain_then_restart                                  │
  │ Sequence: drain_connections → restart_service → verify     │
  │ Trigger: Connection pool exhaustion diagnosed              │
  │ Success rate: 88% (12 uses)                                │
  │ Avg cost: $0.018                                           │
  │ Avg duration: 4.2s                                         │
  │ Source runs: run-42, run-57, run-83, ...                   │
  │ Compiled: Yes (Native Python)                              │
  └────────────────────────────────────────────────────────────┘

  Strategy (version 7, active)
  ──────────────────────────────────────────────────────────────
  Version history:
    v7 ← current (10 min ago) — added error handling for restart
    v6 (1 hour ago) — promoted drain_then_restart skill
    v5 (3 hours ago) — added cost efficiency hint
    v4 (8 hours ago) — rollback from v5 (performance regression)
    v3 (12 hours ago) — promoted search_logs_then_analyze
    ...

  Diff v6 → v7:
    + reasoning_hints: [
    +   "If restart fails, wait 5 seconds and retry once"
    + ]
    + error_handling_hints: [
    +   "connection refused after restart → check if
    +    service is still starting up"
    + ]
```

### Implementation

```python
class MemoryBrowser:
    """
    Inspect the agent's current memory contents.

    Sections:
      - Skills (active, deprecated, candidate)
      - Strategy (current + version history with diffs)
      - World model (tool profiles, constraints, transitions)
      - Semantic facts
      - Retrieved context cache
    """

    def render_html(self, domain: str) -> str:
        """Generate an interactive HTML memory browser."""
        sections = []

        # Skills section
        skills = self._build_skills_section(domain)
        sections.append(skills)

        # Strategy section
        strategy = self._build_strategy_section(domain)
        sections.append(strategy)

        # World model
        world = self._build_world_model_section(domain)
        sections.append(world)

        # Facts
        facts = self._build_facts_section(domain)
        sections.append(facts)

        return self._assemble_html(sections)

    def _build_strategy_section(self, domain: str) -> HtmlSection:
        """Show strategy version history with diffs."""
        versions = self.memory.get_strategy_versions(domain)

        # Compute diffs between consecutive versions
        version_diffs = []
        for i in range(len(versions) - 1):
            diff = self._compute_strategy_diff(
                versions[i + 1]["strategy_data"],
                versions[i]["strategy_data"],
            )
            version_diffs.append({
                "from_v": versions[i]["version"],
                "to_v": versions[i + 1]["version"],
                "diff": diff,
                "promoted_at": versions[i + 1]["created_at"],
            })

        return HtmlSection(
            title=f"Strategy ({len(versions)} versions)",
            content=self._render_version_tree(version_diffs),
        )
```

---

## 7. Evolution Diff Viewer

Shows exactly what changed during an evolution cycle:

```
Evolution Run #12 — Domain: incident-response
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Trigger: Periodic (10 runs since last evolution)
  Duration: 0.8s
  Traces analyzed: 47

  Stages:
    ✅ Analyze: 3 causal patterns found, 1 significant
    ✅ Extract: 1 new skill extracted
    ✅ Gradients: 2 critiques → 2 deltas
    ✅ Evaluate: 1 delta promoted (p=0.032)
    ✅ World model: rebuilt (47 traces)
    ✅ Drift: no drift detected

  New Skill Extracted:
    ┌────────────────────────────────────────────────────────┐
    │ Name: drain_then_restart                                │
    │ Sequence: drain_connections → restart_service → verify  │
    │ Confidence: 88%                                         │
    │ Source: 7 successful traces matching this pattern       │
    │ Effect: Reduces post-restart failures by 60%            │
    └────────────────────────────────────────────────────────┘

  Delta Applied:
    ┌────────────────────────────────────────────────────────┐
    │ Type: failure_avoidance                                 │
    │ Change: +error_handling_hints[1]                        │
    │ Content: "If restart fails, wait 5s and retry once"    │
    │ Expected impact: 0.4                                    │
    │ Evidence: 3 runs where restart failed on first try     │
    └────────────────────────────────────────────────────────┘

  Strategy: v6 → v7
```

### Implementation

```python
class EvolutionDiffViewer:
    """
    Shows detailed diffs of evolution pipeline runs.

    For each evolution event, shows:
      - What triggered it
      - What was analyzed
      - What skills were extracted
      - What deltas were applied
      - How the strategy changed
      - Drift status
    """

    def render(self, evolution_id: str) -> EvolutionDiff:
        events = self.store.get_events(event_type="evolution", filter=evolution_id)

        pipeline_result = self._extract_pipeline_result(events)

        return EvolutionDiff(
            evolution_id=evolution_id,
            trigger=pipeline_result.get("trigger", "unknown"),
            duration=pipeline_result.get("duration", 0),
            traces_analyzed=pipeline_result.get("traces_analyzed", 0),

            # Stage results
            analysis=self._render_analysis_stage(events),
            extraction=self._render_extraction_stage(events),
            gradients=self._render_gradients_stage(events),
            evaluation=self._render_evaluation_stage(events),
            world_model=self._render_world_model_stage(events),
            drift=self._render_drift_stage(events),

            # Strategy diff
            strategy_diff=self._compute_strategy_diff(
                pipeline_result.get("strategy_before", {}),
                pipeline_result.get("strategy_after", {}),
            ),
        )
```

---

## 8. Strategy Lineage

Shows the complete history of strategy changes as an interactive tree:

```
Strategy Lineage — Domain: incident-response
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                          v1 (initial)
                             │
                    ┌────────┼────────┐
                    │        │        │
                    v        v        v
                  v2(skill) v3(hint) v4(rollback)
                    │        │        │
                    └────────┼────────┘
                             │
                             v
                           v5 (merge)
                             │
                    ┌────────┼────────┐
                    │        │        │
                    v        v        v
                  v6(drift) v7(skill) v8(cost)
                    │        │
                    │        v
                    │      v9 (merge)
                    │        │
                    └────────┘
                             │
                             v
                          v10 (current) ✓

  Legend:
    skill   = skill extraction
    hint    = reasoning hint added
    rollback = reverted to previous version
    merge   = combined multiple deltas
    drift   = drift-triggered update
    cost    = cost optimization
```

### Implementation

```python
class StrategyLineage:
    """
    Tracks and visualizes the full lineage of strategy versions.

    Key features:
      - Tree view of version history
      - Branching (parallel strategies tested)
      - Rollback visualization
      - Promotion path (which versions led to current)
    """

    def build(self, domain: str) -> LineageGraph:
        versions = self.memory.get_strategy_versions(domain)

        graph = LineageGraph()
        for v in versions:
            node = LineageNode(
                version=v["version"],
                mutation_type=v.get("mutation_type", "unknown"),
                score=v.get("score", 0),
                parent_version=v.get("parent_version"),
                is_active=v.get("is_active", False),
                timestamp=v["created_at"],
            )
            graph.add_node(node)

            # Auto-detect rollbacks
            if v.get("mutation_type") == "rollback":
                graph.mark_rollback(v["version"], v.get("rollback_target"))

        return graph

    def render_html(self, domain: str) -> str:
        """Render interactive tree visualization."""
        graph = self.build(domain)
        return self._generate_tree_html(graph)
```

---

## 9. Live Debugger

An interactive debugger that can pause, inspect, and resume agent execution:

```python
class LiveDebugger:
    """
    Interactive debugger for agent execution.

    Commands:
      pause         — pause execution at next LLM call
      continue      — resume execution
      step          — execute one step
      inspect       — view current state (prompt, context, memory)
      edit_prompt   — modify the prompt before sending
      set_breakpoint — pause at specific step/tool/condition
      snapshot      — save full state for later replay
      rollback_step — undo last step
    """

    def __init__(self, agent: Agent):
        self.agent = agent
        self.paused = False
        self.breakpoints: list[Breakpoint] = []
        self.snapshot_history: list[AgentSnapshot] = []

    async def pause_at_next_llm_call(self, context: RunContext):
        """Pause execution before an LLM call."""
        self.paused = True

        # Capture snapshot
        snapshot = AgentSnapshot(
            run_id=context.run_id,
            step_number=context.current_step,
            system_prompt=context.system_prompt,
            messages=context.messages,
            memory_state=context.memory.snapshot(),
            pending_tool_calls=context.pending_tool_calls,
        )
        self.snapshot_history.append(snapshot)

        # Send to debugger UI
        await self._notify_ui("paused", snapshot)

        # Wait for user command
        command = await self._wait_for_command()
        await self._execute_command(command, context)

    async def _execute_command(self, command: DebugCommand, context: RunContext):
        if command.type == "continue":
            self.paused = False

        elif command.type == "step":
            # Execute one step then pause again
            pass  # let one iteration through

        elif command.type == "edit_prompt":
            context.system_prompt = command.new_prompt

        elif command.type == "inspect":
            await self._send_inspection(command.query)

        elif command.type == "snapshot":
            self._save_snapshot(context)

        elif command.type == "rollback":
            # Restore previous snapshot
            prev = self.snapshot_history[-2]
            context.restore(prev)
```

---

## 10. Dashboard

A real-time dashboard showing all active agents, their performance, and system health:

```
┌─────────────────────────────────────────────────────────────────────┐
│  NIL OBSERVABILITY DASHBOARD                   ● live  ● warnings  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Active Agents: 3               Runs Today: 142     Cost Today: $2.47 │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ Agent: incident-response    Status: ● running                   │ │
│  │  ┌────────────────────────────────────────────────────────────┐ │ │
│  │  │ Success Rate (last 50 runs)                                │ │ │
│  │  │ 100% ┤                                                  ●  │ │ │
│  │  │  80% ┤                           ●────●────●────●────●     │ │ │
│  │  │  60% ┤     ●────●────●────●────●                            │ │ │
│  │  │  40% ┤  ●                                                  │ │ │
│  │  │  20% ┤ ●                                                   │ │ │
│  │  │   0% ┤────────────────────────────────────────────────────  │ │ │
│  │  │      1    10    20    30    40    50                        │ │ │
│  │  └────────────────────────────────────────────────────────────┘ │ │
│  │                                                                  │ │
│  │  Evolution: v10 active (7 deltas applied)    Drift: none        │ │
│  │  Skills: 7 active, 3 deprecated              Cost/run: $0.038   │ │
│  │  Latency: 8.4s avg                           Tokens/run: 12.4K  │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ Memory Stats  │  │ Evolution    │  │ Recent Events            │  │
│  │              │  │ Activity     │  │                          │  │
│  │ Skills: 7    │  │              │  │ 12:30  Run #142 success  │  │
│  │ Facts: 124   │  │ v10 ── 2min  │  │ 12:28  Run #141 success  │  │
│  │ Traces: 847  │  │  │           │  │ 12:25  Skill extracted   │  │
│  │ Constraints: │  │ v9 ── 1hr   │  │ 12:20  Evals run #12     │  │
│  │   18         │  │  │           │  │ 12:15  Strategy v10 prom │  │
│  │ Storage:     │  │ v8 ── 3hr   │  │ 12:10  Drift check: none │  │
│  │   4.2 MB     │  │  │           │  │                          │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Roadmap

| Phase | What | How |
|---|---|---|
| 1 | Structured event store | Replace log lines with typed events. SQLite or file-based event log |
| 2 | Trace viewer | Timeline + span visualization. Group events by run_id |
| 3 | Token timeline | Track token usage per step. Collect from provider responses |
| 4 | Memory browser | Read and display all memory tables. Show skills, strategies, facts |
| 5 | Strategy lineage | Version tree with rollback detection. Diff between versions |
| 6 | Evolution diff viewer | Per-evolution-run breakdown. Before/after strategy comparison |
| 7 | Flame graphs | Span-based timing visualization. Identify bottlenecks |
| 8 | Reasoning tree | Capture LLM options, evidence, chosen action, rationale per step |
| 9 | Live debugger | Pause/resume/step/inspect/edit. Integrated into agent loop |
| 10 | Web dashboard | Real-time aggregation. Multi-agent view. Historical charts |

---

## Key Differentiator

**Without observability:**
The agent is a black box. It runs, it evolves, it changes its own behavior — and you have no idea why. Debugging is reading log lines and guessing. When performance degrades, you do not know if it was a bad evolution, a corrupted memory, or a change in the task distribution. The system is autonomous but opaque.

**With observability:**
Every decision is recorded and inspectable. You can replay any run step-by-step, see what the LLM was thinking, which memories it retrieved, which skills it used, how it changed its own strategy. You can pause a running agent, inspect its state, modify its prompt, and resume. You can trace performance degradation back to the specific evolution cycle that caused it.

This is the difference between a self-driving car with a black box and one with a transparent cockpit — when the system is autonomous, observability is not a nice-to-have, it is the only way to maintain trust and control.
