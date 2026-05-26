# Tool Embeddings & Affordance Learning

**Beyond Static Function Schemas**

Tools in Nil are currently defined as static schemas — a name, description, parameter JSON Schema, and a Python function. The agent calls tools by name, selected by the LLM's text generation. There is no notion of tool *similarity*, tool *composition*, tool *affordance* (what a tool makes possible), or latent tool *capabilities* that emerge from usage patterns.

This document proposes a representation where tools are embedded in a continuous vector space, their affordances are learned from usage data, compositions are discovered automatically, and selection happens in a latent space rather than by explicit name-matching.

---

## Current State

```
Current Tool Representation:

  Tool(name="search_logs",
       description="Search application logs for a query",
       parameters={"query": {"type": "string", ...}},
       fn=search_logs_impl)

  Selection: LLM generates text "I should use search_logs"
  → text match → tool name → invoke

  Limitations:
    • Tool names are opaque strings — no notion of similarity
    • No capability representation beyond the description text
    • Composition must be explicitly chained by the LLM
    • No learning which tools work together
    • No learning which tools substitute for each other
    • No learning which contexts make a tool useful or dangerous
    • Every tool is equally "present" — no activation gating
```

---

## Proposed Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    TOOL REPRESENTATION LAYER                      │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                    Tool Embedding Space                     │  │
│  │                                                             │  │
│  │       search_logs ●                                         │  │
│  │                    \                                        │  │
│  │                     ● query_metrics                         │  │
│  │                    /                                        │  │
│  │       analyze_logs ●                                        │  │
│  │                                                             │  │
│  │       restart_service ●─────● drain_connections              │  │
│  │                                                             │  │
│  │       deploy ●──────────────● rollback                      │  │
│  │                                                             │  │
│  │       ● pagerduty_alert        ● slack_notify               │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │ Affordance   │  │ Composition  │  │ Latent Selection       │ │
│  │ Learner      │  │ Synthesizer  │  │ Router                 │ │
│  │              │  │              │  │                        │ │
│  │ Learns what  │  │ Discovers    │  │ Embed query →          │ │
│  │ each tool    │  │ valid tool   │  │ nearest tools →        │ │
│  │ enables      │  │ combinations │  │ gated activation       │ │
│  └──────────────┘  └──────────────┘  └────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

---

## 1. Tool Embeddings

### Embedding Sources

Tool embeddings are constructed from multiple signals:

```python
@dataclass
class ToolEmbedding:
    tool_name: str

    # From static schema
    name_embedding: np.ndarray          # from tool name (e.g., sentence-transformer)
    description_embedding: np.ndarray   # from tool docstring
    parameter_embedding: np.ndarray     # from parameter names + types

    # Learned from usage (updated after every run)
    usage_embedding: np.ndarray         # from co-occurrence patterns
    outcome_embedding: np.ndarray       # from contexts where tool succeeded/failed
    transition_embedding: np.ndarray    # from predecessor/successor tool patterns

    # Composite
    full_embedding: np.ndarray          # weighted combination of all sources

    def similarity(self, other: 'ToolEmbedding') -> float:
        """Cosine similarity between composite embeddings."""
        return cosine_similarity(self.full_embedding, other.full_embedding)
```

### Embedding Generation

```python
class ToolEmbeddingGenerator:
    """
    Generates and updates tool embeddings.

    Embeddings are initialized from static schema and refined
    through usage statistics after every run.
    """

    embedding_dim: int = 128

    def initialize(self, tool: Tool) -> ToolEmbedding:
        """Initial embedding from tool definition alone."""
        # Schema-based embeddings
        name_emb = self._encode_text(tool.name)
        desc_emb = self._encode_text(tool.description)
        param_emb = self._encode_parameters(tool.parameters)

        # Weighted combination
        full = self._combine([name_emb, desc_emb, param_emb], weights=[0.3, 0.5, 0.2])

        return ToolEmbedding(
            tool_name=tool.name,
            name_embedding=name_emb,
            description_embedding=desc_emb,
            parameter_embedding=param_emb,
            full_embedding=full,
        )

    def update_from_usage(
        self,
        embedding: ToolEmbedding,
        traces: list[Trace],
    ) -> ToolEmbedding:
        """Refine embedding based on observed usage patterns."""
        # Co-occurrence → similar tools move closer
        cooccurrence_vector = self._compute_cooccurrence_vector(tool.name, traces)
        # Outcome conditioning → tools with similar success profiles cluster
        outcome_vector = self._compute_outcome_vector(tool.name, traces)
        # Transitions → tools that follow similar predecessors cluster
        transition_vector = self._compute_transition_vector(tool.name, traces)

        # Update usage embedding
        embedding.usage_embedding = self._fuse(
            cooccurrence_vector, outcome_vector, transition_vector
        )

        # Recompute composite
        embedding.full_embedding = self._combine([
            embedding.name_embedding,
            embedding.description_embedding,
            embedding.parameter_embedding,
            embedding.usage_embedding,
        ], weights=[0.15, 0.25, 0.10, 0.50])  # usage dominates as data accumulates

        return embedding
```

### Tool Space Structure

As embeddings are learned, the tool space reveals structure:

```
Example learned tool space (2D projection):

  Investigation cluster:          Communication cluster:
    search_logs  ●                  slack_notify  ●
    query_metrics ●                 pagerduty_alert ●
    analyze_trace ●                 email_team    ●
    check_db      ●                 escalate      ●

  Action cluster:                 Recovery cluster:
    restart_service ●              drain_connections ●
    scale_up       ●              rollback_deploy   ●
    deploy         ●              revert_migration  ●
    failover       ●              restore_backup    ●

  Substitute pairs (high similarity):
    search_logs  ↔ query_metrics    (both investigate)
    restart_service ↔ scale_up      (both fix degraded)
    deploy ↔ rollback               (inverse operations)
```

---

## 2. Affordance Learning

An *affordance* is what a tool makes possible or what it enables the agent to do. This goes beyond the tool's description to capture emergent properties.

```python
@dataclass
class ToolAffordance:
    """
    What a tool enables in the space of states and outcomes.

    Not "what the tool does" — but "what the tool makes possible."
    """
    tool_name: str

    # State transitions the tool consistently enables
    enabled_transitions: list[tuple[str, str, float]]
    # (from_state → to_state, confidence)

    # Goals this tool is useful for
    associated_goals: list[tuple[str, float]]
    # (goal_description, relevance)

    # Preconditions (what must be true for this tool to be useful)
    preconditions: list[tuple[str, float]]

    # Postconditions (what becomes possible after using this tool)
    postconditions: list[tuple[str, float]]

    # Substitute tools (other tools that can replace this one)
    substitutes: list[tuple[str, float]]

    # Complementary tools (tools that work well after/before this one)
    complements: list[tuple[str, float]]

    # Failure contexts (conditions under which this tool fails)
    failure_contexts: list[tuple[str, float]]
```

### Affordance Discovery

```python
class AffordanceLearner:
    """
    Discovers tool affordances from execution traces.

    This is the bridge between "tool description" and
    "tool capability" — learning what tools actually
    do in practice, not just what their docstrings say.
    """

    def learn_affordances(
        self,
        tool: Tool,
        traces: list[Trace],
        state_space: StateSpace,
    ) -> ToolAffordance:
        # 1. State transitions enabled by this tool
        transitions = self._extract_state_transitions(tool, traces, state_space)

        # 2. Goals associated with this tool
        goals = self._extract_associated_goals(tool, traces)

        # 3. Preconditions (what state features precede successful use)
        preconditions = self._extract_preconditions(tool, traces, state_space)

        # 4. Postconditions (what state features follow successful use)
        postconditions = self._extract_postconditions(tool, traces, state_space)

        # 5. Substitutes (tools used in similar contexts with similar outcomes)
        substitutes = self._discover_substitutes(tool, traces)

        # 6. Complements (tools that frequently precede/follow this one in successful runs)
        complements = self._discover_complements(tool, traces)

        # 7. Failure contexts
        failure_contexts = self._extract_failure_contexts(tool, traces, state_space)

        return ToolAffordance(
            tool_name=tool.name,
            enabled_transitions=transitions,
            associated_goals=goals,
            preconditions=preconditions,
            postconditions=postconditions,
            substitutes=substitutes,
            complements=complements,
            failure_contexts=failure_contexts,
        )

    def _discover_substitutes(self, tool: Tool, traces: list[Trace]) -> list[tuple[str, float]]:
        """
        Find tools that can substitute for this one.

        A tool B is a substitute for tool A if:
          1. B is used in similar states as A
          2. B produces similar state transitions as A
          3. B achieves similar success rates as A in those states

        Measured by: embedding similarity + outcome similarity + transition similarity
        """
        candidates = []
        for other_tool in self.all_tools:
            if other_tool.name == tool.name:
                continue

            # Context similarity: are they used in similar states?
            context_sim = self._context_similarity(tool, other_tool, traces)

            # Outcome similarity: do they produce similar results?
            outcome_sim = self._outcome_similarity(tool, other_tool, traces)

            # Transition similarity: do they enable similar next states?
            transition_sim = self._transition_similarity(tool, other_tool, traces)

            # Composite
            similarity = 0.4 * context_sim + 0.3 * outcome_sim + 0.3 * transition_sim
            if similarity > 0.6:  # threshold
                candidates.append((other_tool.name, similarity))

        return sorted(candidates, key=lambda x: x[1], reverse=True)

    def _discover_complements(self, tool: Tool, traces: list[Trace]) -> list[tuple[str, float]]:
        """
        Find tools that work well with this one.

        Tool B is complementary to tool A if:
          - A→B sequence has higher success rate than A alone
          - B→A sequence has higher success rate than B alone
          - A and B co-occur in successful trajectories more than expected
        """
        # Lift-based co-occurrence analysis
        # P(B | A) / P(B) — how much more likely B is when A is used
        complements = []
        for other_tool in self.all_tools:
            if other_tool.name == tool.name:
                continue
            lift = self._cooccurrence_lift(tool, other_tool, traces)
            if lift > 1.5:  # 50% more likely than random
                complements.append((other_tool.name, lift))

        return sorted(complements, key=lambda x: x[1], reverse=True)
```

### Affordance Example

```
Tool: search_logs

  Affordances:
    Enabled transitions:
      (unknown → investigating)  conf=0.95
      (investigating → diagnosing) conf=0.72  (when combined with analysis)

    Associated goals:
      "find root cause"         relevance=0.91
      "understand failure"      relevance=0.88
      "check for errors"        relevance=0.85

    Preconditions:
      "service is accessible"   conf=0.90
      "logs are being written"  conf=0.85

    Postconditions:
      "can identify error type"    conf=0.88
      "can narrow time window"     conf=0.82

    Substitutes:
      query_metrics          0.81
      analyze_trace          0.72
      check_db               0.45  (weak substitute)

    Complements:
      analyze_logs           0.89  (search_logs → analyze_logs is highly successful)
      restart_service        0.42  (sometimes useful after, but not strongly coupled)

    Failure contexts:
      "logs are rotated/empty"    conf=0.82
      "service is down"           conf=0.71
      "permission denied"         conf=0.90
```

---

## 3. Compositional Tool Synthesis

The agent should not just call individual tools — it should synthesize composite tools from primitives:

```python
class ToolCompositionSynthesizer:
    """
    Discovers and synthesizes composite tools from frequently
    occurring tool sequences.

    A composite tool is a named, compressed sequence of tool calls
    that can be invoked as a single unit.
    """

    def discover_compositions(
        self,
        traces: list[Trace],
        min_occurrences: int = 5,
        min_success_rate: float = 0.75,
    ) -> list[CompositeTool]:
        """
        Find frequent tool sequences that could be composed.

        Uses sequential pattern mining (GSP / SPADE / PrefixSpan)
        to find subsequences that:
          - Appear frequently in successful trajectories
          - Have high mutual information between steps
          - Produce consistent state transitions
        """
        candidates = []

        # Mine frequent sequential patterns
        patterns = self._mine_frequent_sequences(
            traces, min_support=min_occurrences
        )

        for pattern in patterns:
            sequence = pattern.sequence  # e.g., ["search_logs", "analyze", "fix"]
            success_rate = pattern.success_rate

            # Check if composition is beneficial
            # (reduces token cost, reduces errors, faster)
            benefit = self._estimate_composition_benefit(sequence, traces)

            if success_rate >= min_success_rate and benefit > 0:
                composite = CompositeTool(
                    name=self._generate_composite_name(sequence),
                    tools=sequence,
                    success_rate=success_rate,
                    expected_cost_savings=benefit.cost_savings,
                    expected_step_savings=benefit.step_savings,
                    affordance=self._derive_composite_affordance(sequence, traces),
                )
                candidates.append(composite)

        return candidates

    def synthesize_composite_tool(
        self, sequence: list[str], traces: list[Trace]
    ) -> CompositeTool:
        """
        Create an executable composite tool from a tool sequence.

        The composite stores:
          - The tool sequence (for execution)
          - Preconditions (from first tool)
          - Postconditions (from last tool)
          - Trigger conditions (when to use this composite)
          - Cost/duration estimates
        """
        first_tool = sequence[0]
        last_tool = sequence[-1]

        return CompositeTool(
            name=self._generate_name(sequence),
            tools=sequence,
            preconditions=self._compose_preconditions(sequence, traces),
            postconditions=self._compose_postconditions(sequence, traces),
            trigger=self._infer_trigger_conditions(sequence, traces),
            avg_cost=sum(self.tool_costs[t] for t in sequence),
            avg_duration=sum(self.tool_durations[t] for t in sequence),
            success_rate=self._sequence_success_rate(sequence, traces),
        )


@dataclass
class CompositeTool:
    """A synthesized tool composed of multiple primitive tool calls."""

    name: str
    tools: list[str]
    preconditions: list[str]
    postconditions: list[str]
    trigger: TriggerCondition
    avg_cost: float
    avg_duration: float
    success_rate: float

    def to_tool(self) -> Tool:
        """Convert to a standard Tool object for registration."""
        return Tool(
            name=self.name,
            description=f"Composite: {' → '.join(self.tools)}. "
                        f"Success rate: {self.success_rate:.0%}",
            parameters={},  # synthesized tools handle their own params
            fn=self._execute,
        )

    def _execute(self, **kwargs) -> Any:
        """Execute the composite sequence as a single callable."""
        results = []
        for tool_name in self.tools:
            tool = self.tool_manager.get(tool_name)
            result = tool.invoke(kwargs)
            results.append(result)
        return results[-1]  # return last tool's output
```

---

## 4. Latent Tool Selection

Instead of the LLM generating tool names as text, tools are selected via a latent embedding space:

```python
class LatentToolSelector:
    """
    Selects tools via embedding similarity rather than text generation.

    Process:
      1. Encode the current context (task + state + recent actions) into
         a query embedding
      2. Find nearest tool embeddings in the tool space
      3. Gate: filter out tools whose preconditions are not met
      4. Rank: score by relevance + affordance match + exploration bonus
      5. Present top-K to the LLM or auto-execute top-1
    """

    def select_tools(
        self,
        context: RetrievalQuery,
        top_k: int = 3,
    ) -> list[ScoredTool]:
        """
        Select the most relevant tools for the current context.

        This replaces "which tool should I use?" text generation
        with a learned retrieval process.
        """
        # 1. Encode context into query embedding
        query_emb = self._encode_context(context)

        # 2. Find nearest tools
        candidates = self.tool_index.search(query_emb, top_k=top_k * 2)

        # 3. Gate by precondition satisfaction
        gated = []
        for tool_name, similarity in candidates:
            affordance = self.affordances.get(tool_name)
            if affordance and self._preconditions_met(affordance, context):
                gated.append((tool_name, similarity))

        # 4. Rank with exploration bonus
        ranked = []
        for tool_name, similarity in gated[:top_k]:
            exploration_bonus = self.exploration_scheduler.get_bonus(
                tool_name, context
            )
            affordance_score = self._compute_affordance_match(
                tool_name, context
            )
            total_score = (
                0.5 * similarity +
                0.3 * affordance_score +
                0.2 * exploration_bonus
            )
            ranked.append(ScoredTool(
                tool_name=tool_name,
                score=total_score,
                similarity=similarity,
                affordance_score=affordance_score,
                exploration_bonus=exploration_bonus,
            ))

        return sorted(ranked, key=lambda x: x.score, reverse=True)[:top_k]

    def _encode_context(self, context: RetrievalQuery) -> np.ndarray:
        """Encode the current context into the tool embedding space."""
        parts = [
            self._encode_text(context.task),
            self._encode_text(" ".join(context.current_tool_state)),
            self._encode_text(context.last_error or ""),
            self._encode_text(context.domain),
        ]
        return np.mean(parts, axis=0)

    def _preconditions_met(
        self, affordance: ToolAffordance, context: RetrievalQuery
    ) -> bool:
        """Check if preconditions for this tool are satisfied."""
        for precondition, confidence in affordance.preconditions:
            if confidence > 0.8 and not self._check_precondition(
                precondition, context
            ):
                return False
        return True
```

### Integration with LLM

```python
class LatentAwarePromptBuilder:
    """
    Builds prompts where tool selection is guided by
    embeddings rather than left entirely to LLM text generation.

    Instead of listing all tools, inject only the top-K
    selected by the latent router.
    """

    def build_tool_section(
        self,
        context: RetrievalQuery,
        all_tools: list[Tool],
    ) -> str:
        selected = self.latent_selector.select_tools(context, top_k=5)

        # Build prioritized tool list
        tools_section = ["Available tools (priority order):"]
        for scored_tool in selected:
            tool = self.tool_manager.get(scored_tool.tool_name)
            tools_section.append(
                f"  [{scored_tool.score:.2f}] {tool.name}: {tool.description[:100]}"
            )

        return "\n".join(tools_section)
```

---

## 5. Tool Affordance Graph

Tools and their affordances form a graph that the agent can query:

```python
class ToolAffordanceGraph:
    """
    A graph where:
      - Nodes are tools
      - Edges are affordance relationships
      - Edge types: substitutes_for, complements, precedes, enables, conflicts_with

    Enables reasoning like:
      "What can I use instead of X?"
      "What should I do after Y?"
      "What tools help me achieve goal Z?"
    """

    def __init__(self):
        self.graph = nx.DiGraph()

    def add_tool(self, tool: Tool, affordance: ToolAffordance):
        self.graph.add_node(tool.name, tool=tool, affordance=affordance)

        for sub_name, weight in affordance.substitutes:
            self.graph.add_edge(tool.name, sub_name,
                              type="substitutes_for", weight=weight)

        for comp_name, weight in affordance.complements:
            self.graph.add_edge(tool.name, comp_name,
                              type="complements", weight=weight)

    def substitutes(self, tool_name: str) -> list[tuple[str, float]]:
        """What can I use instead of this tool?"""
        edges = self.graph.edges(tool_name, data=True)
        return [
            (e[1], e[2]["weight"])
            for e in edges
            if e[2].get("type") == "substitutes_for"
        ]

    def downstream_tools(self, tool_name: str) -> list[tuple[str, float]]:
        """What tools work well after this one?"""
        edges = self.graph.edges(tool_name, data=True)
        return [
            (e[1], e[2]["weight"])
            for e in edges
            if e[2].get("type") == "complements"
        ]

    def tools_for_goal(self, goal: str, top_k: int = 5) -> list[str]:
        """Find tools associated with a specific goal."""
        scored = []
        for node, data in self.graph.nodes(data=True):
            affordance = data.get("affordance")
            if affordance:
                for g, relevance in affordance.associated_goals:
                    if goal.lower() in g.lower():
                        scored.append((node, relevance))
        return sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]
```

---

## 6. Learning Loop

```
Every run completion:

  1. Tool Embeddings Update
     - Update usage embeddings from new trace
     - Recompute co-occurrence, outcome, transition vectors
     - Refresh composite embeddings

  2. Affordance Refresh
     - Update state transition probabilities per tool
     - Update substitute and complement relationships
     - Update failure contexts

  3. Composite Discovery
     - Mine frequent sequential patterns
     - Evaluate composition benefits (cost, step savings)
     - Register high-value composites as new tools

  4. Graph Update
     - Add/update nodes and edges
     - Recompute graph metrics (centrality, communities)
     - Identify tool clusters (investigation, action, communication)
```

---

## Implementation Roadmap

| Phase | What | How |
|---|---|---|
| 1 | Tool name + description embeddings | Static embedding per tool from schema. Sentence-transformers or text-embedding-3-small |
| 2 | Usage-based embedding refinement | Update embeddings from co-occurrence + outcome patterns after each run |
| 3 | Substitution discovery | Embedding-space nearest neighbors → identify substitute tools |
| 4 | Complement discovery | Lift-based co-occurrence analysis → identify tool pairs that work well together |
| 5 | Affordance extraction | Full ToolAffordance from traces: preconditions, postconditions, failure contexts, enabled transitions |
| 6 | Latent tool selection | Encode context → nearest tools in embedding space → gate by preconditions → present top-K |
| 7 | Composite tool synthesis | Frequent sequence mining → evaluate benefit → register as callable composite |
| 8 | Affordance graph | NetworkX graph with typed edges. Query: substitutes, complements, tools-for-goal |

---

## Key Differentiator

**Static schemas:**
The agent knows only what the tool's author wrote in the docstring. It cannot discover that `search_logs` and `query_metrics` serve similar purposes, that `restart_service` works best after `drain_connections`, or that `deploy` and `rollback` are inverses. Every tool is an island.

**With embeddings and affordances:**
The tool space has *structure*. The agent knows which tools cluster, which substitute, which compose. It discovers emergent properties — tools that are good for investigating versus acting versus communicating. It synthesizes new composite tools from frequently observed sequences. Tool selection becomes a learned retrieval process, not a text-generation lottery.
