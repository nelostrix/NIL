# Hierarchical Memory Architecture

> **Status: Implemented** — The full 7-tier architecture exists in `nil/memory/` (`hierarchical.py`, `episodic.py`, `semantic.py`, `procedural.py`, `working.py`, `world_state.py`, `social_memory.py`, `meta_memory.py`, `abstraction.py`, `consolidation.py`). All seven tiers are operational with cross-tier consolidation. Some features — such as the full working memory sliding window with importance-based retention and world state diff-logging with staleness tracking — have minimal current implementations and can be extended further.

---

## Background

The original Nil memory system stored three types of data: execution traces (what happened), extracted skills (what worked), and strategy versions (how to act). This was flat and operational. It recorded *patterns* but did not build *knowledge*.

The hierarchical memory architecture replaced this with seven specialized tiers, each serving a distinct cognitive function.

---

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                   META-MEMORY                              │
│    What the agent knows it knows / doesn't know            │
│    Confidence calibration, knowledge gaps, certainty       │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌───────────────┐  ┌────────────────┐  │
│  │  EPISODIC    │  │   SEMANTIC    │  │  PROCEDURAL    │  │
│  │  Memory      │  │   Memory      │  │  Memory        │  │
│  │              │  │               │  │                │  │
│  │ Raw traces   │  │ Facts, rules, │  │ Skills,        │  │
│  │ Observations │  │ concepts,     │  │ policies,      │  │
│  │ Timelines    │  │ relationships │  │ tool sequences  │  │
│  └──────────────┘  └───────────────┘  └────────────────┘  │
│                                                             │
│  ┌──────────────┐  ┌───────────────┐  ┌────────────────┐  │
│  │  WORKING     │  │  WORLD STATE  │  │  SOCIAL/USER   │  │
│  │  Memory      │  │  Memory       │  │  Memory        │  │
│  │              │  │               │  │                │  │
│  │ Active       │  │ External env  │  │ User           │  │
│  │ context      │  │ snapshots     │  │ preferences,   │  │
│  │ Current task │  │ State cache   │  │ personas,      │  │
│  │ In-progress  │  │ DB state      │  │ interaction    │  │
│  └──────────────┘  └───────────────┘  └────────────────┘  │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

---

## Memory Tier Specifications

### 1. Episodic Memory — "What happened"

**Purpose:** Store exact historical runs with full fidelity.

**Current equivalent:** `traces` table in `PersistentMemory`

**Upgrade needed:**
- Retention policies (recent → summarized → compressed → archived → forgotten)
- Temporal indexing (timeline queries, before/after, causal ordering)
- Episode clustering (group related runs into sessions)
- Rehearsal scheduling (important episodes get replayed, not just stored)

```
Schema:
  episode_id        UUID
  domain            str
  task              str
  steps             [StepRecord]
  outcome           RunOutcome
  session_id        UUID           # groups related runs
  importance        float          # computed, drives retention
  created_at        timestamp
  last_accessed_at  timestamp
  access_count      int
  retention_tier    enum(RECENT | SUMMARIZED | COMPRESSED | ARCHIVED | FORGOTTEN)

Retention flow:
  RECENT (full fidelity, N runs) 
    → SUMMARIZED (synopsis + key decisions, M runs) 
    → COMPRESSED (features only, K runs) 
    → ARCHIVED (metadata only, indefinitely) 
    → FORGOTTEN (deleted)
```

### 2. Semantic Memory — "What is true"

**Purpose:** Facts, concepts, relationships, and rules learned across runs. This is the leap from pattern-matching to actual knowledge.

**Current equivalent:** Weakly present in `WorldModel` constraints and `AnalysisReport`

**Upgrade needed:**
- Fact extraction (declarative statements the agent can assert with confidence)
- Relationship graphs (tool-A works-with tool-B, error-X precedes error-Y)
- Concept formation (cluster raw observations into named concepts)
- Belief tracking (facts with confidence that decays without reinforcement)
- Ontology learning (hierarchical category structure of the domain)

```
Schema:
  fact_id            UUID
  subject            str
  predicate          str
  object             str
  confidence         float           # 0.0 - 1.0 with statistical calibration
  evidence_count     int
  first_observed_at  timestamp
  last_confirmed_at  timestamp       # decays over time if not reinforced
  source_episodes    [UUID]          # which episodes support this
  fact_type          enum(RULE | RELATIONSHIP | CONCEPT | CONSTRAINT | HEURISTIC)

Example facts:
  subject: "tool:search_logs"  predicate: "precedes"  object: "tool:analyze"  conf: 0.92
  subject: "error:timeout"     predicate: "resolved_by"  object: "action:retry_with_backoff"  conf: 0.87
  subject: "task:deploy"       predicate: "requires"  object: "prerequisite:ci_pass"  conf: 0.95
```

### 3. Procedural Memory — "How to do things"

**Purpose:** Skills, policies, and executable strategies. Compressed, optimized, and ready to invoke.

**Current equivalent:** `skills` table + `strategy_versions` table

**Upgrade needed:**
- Skill chaining (compose simple skills into complex procedures)
- Skill specialization (generic skill → domain-tuned variant)
- Execution traces as skill training data (every run improves the skill)
- Failure-driven refinement (skills get better when they fail)
- Automatic chunking (frequent multi-step sequences become atomic skills)

```
Schema:
  skill_id            UUID
  name                str
  domain              str
  trigger_conditions  TriggerCondition
  procedure           [Step | SubSkill]   # can compose sub-skills
  preconditions       [str]
  postconditions      [str]
  success_rate        float
  avg_cost            float
  avg_duration        float
  usage_count         int
  refinement_history  [RefinementEvent]
  version             int
  parent_skill        Optional[UUID]       # specialization hierarchy

Key enhancement:
  Skills aren't just extracted — they're refined through usage.
  Each execution is a training example. Failures produce gradients
  that update the skill, not just the strategy.
```

### 4. Working Memory — "What I'm doing right now"

**Purpose:** Active context for the current task. Ephemeral, bounded, and attention-driven.

**Current equivalent:** In-memory `messages` list in `KernelRunner.execute()`

**Upgrade needed:**
- Sliding window with importance-based retention (not just last N)
- Attention mechanism (what to keep vs. discard based on relevance)
- Scratchpad space for intermediate reasoning
- Task decomposition state (subtasks, progress tracking)
- Context budget management (token-aware eviction)

```
Schema:
  session_id         UUID
  task               str
  current_goal       str
  subgoals           [Goal]
  context_window     [ContentBlock]     # importance-ranked, token-budgeted
  scratchpad         str                 # LLM's intermediate reasoning
  tool_call_history  [ToolCall]          # recent N calls
  attention_weights  {ref: float}        # what's currently relevant

Eviction policy:
  When token budget exceeded, evict in order:
    1. Completed subgoal context
    2. Low-importance observations
    3. Old tool results (if no longer referenced)
    4. Unchanged repeated context
  Never evict: current goal, active subgoal, uncompleted critical path
```

### 5. World State Memory — "What the environment looks like"

**Purpose:** Persistent representation of the external environment. Survives across runs.

**Current equivalent:** Not present. World model is rebuilt-from-scratch from traces.

**Upgrade needed:**
- State cache (last-known state of external systems)
- State transition tracking (how state changed between runs)
- State prediction (given an action, predict resulting state)
- Consistency checking (detect when external state contradicts expectations)
- Snapshot versioning (rollback-aware state tracking)

```
Schema:
  world_id             UUID
  domain               str
  state_snapshot       JSON             # last-known external state
  state_schema         JSON             # defines what state looks like
  last_verified_at     timestamp        # staleness tracking
  delta_log            [StateDelta]     # how state changed over time
  confidence           float            # how sure we are state is accurate

Example:
  domain: "incident-response"
  state_snapshot: {
    "active_incidents": 3,
    "pagerduty_oncall": "alice@nelo.dev",
    "last_deploy": "2026-05-25T22:00:00Z",
    "deploy_status": "healthy"
  }
  delta_log: [
    { "timestamp": ..., "field": "active_incidents", "from": 2, "to": 3 },
    { "timestamp": ..., "field": "deploy_status", "from": "degraded", "to": "healthy" }
  ]
```

### 6. Social/User Memory — "Who I'm working with"

**Purpose:** User preferences, communication style, common requests, trust levels.

**Current equivalent:** Not present.

**Upgrade needed:**
- User profiles (preferred verbosity, communication style, expertise level)
- Interaction history (what user asked for, how they reacted, what they corrected)
- Preference learning (infer preferences from implicit signals)
- Trust calibration (when user overrides, adjust confidence in that domain)
- Persona adaptation (adjust tone, detail, proactivity per user)

```
Schema:
  user_id              UUID
  name                 str
  interaction_count    int
  preferred_verbosity  enum(CONCISE | NORMAL | DETAILED)
  communication_style  enum(FORMAL | CASUAL | TECHNICAL)
  domain_expertise     {domain: float}    # estimated expertise per domain
  common_requests      [str]              # frequently asked patterns
  correction_history   [Correction]       # user corrections → learning signal
  trust_per_domain     {domain: float}    # how reliable user's guidance is
  last_interaction     timestamp

Note:
  This enables personalization. Different users of the same agent
  get different experiences without retraining.
```

### 7. Meta-Memory — "What I know I know"

**Purpose:** Self-awareness of knowledge state. Calibration, gaps, and uncertainty.

**Current equivalent:** Not present.

**Upgrade needed:**
- Confidence calibration (how reliable are our own beliefs?)
- Knowledge gap detection (tasks where we lack relevant experience)
- Epistemic curiosity (what should we explore to fill gaps?)
- Contradiction detection (when two facts conflict)
- Certainty-weighted reasoning (weight decisions by knowledge confidence)

```
Schema:
  domain               str
  known_facts_count    int
  knowledge_graph      Graph             # entities + relationships with confidence
  gaps                 [KnowledgeGap]    # identified blind spots
  contradictions       [Contradiction]   # conflicting beliefs
  calibration_score    float             # how well confidence correlates with accuracy
  last_audit_at        timestamp

KnowledgeGap:
  area                 str               # what we're unsure about
  evidence             str               # why we think there's a gap
  exploration_priority float             # how valuable filling this gap would be
  suggested_task       str               # what kind of run would fill it

This enables active learning:
  The agent can request tasks that fill its knowledge gaps,
  rather than passively waiting for instructions.
```

---

## Memory Flow: How Data Moves Between Tiers

```
Execution Time (each run):
┌──────────┐
│  WORKING │ ← Active context, current task, tool IO
│  MEMORY  │
└────┬─────┘
     │ run completes
     v
┌──────────┐     ┌────────────┐
│ EPISODIC │────→│  SEMANTIC  │  ← Fact extraction from episode
│ MEMORY   │     │  MEMORY    │
└──────────┘     └─────┬──────┘
     │                  │
     │                  │ confirms/reinforces beliefs
     v                  v
┌──────────┐     ┌────────────┐
│PROCEDURAL│←────│  META-     │  ← Knowledge gaps detected
│ MEMORY   │     │  MEMORY    │
└──────────┘     └────────────┘

Between Runs:
┌──────────┐     ┌────────────┐     ┌──────────┐
│ EPISODIC │────→│  SEMANTIC  │────→│PROCEDURAL│
│ (recent) │     │ (abstract) │     │ (skills) │
└──────────┘     └────────────┘     └──────────┘
     │                                 │
     │ consolidation (background)      │ refinement (feedback)
     v                                 v
  compressed/indexed               versioned/optimized

On Agent Start:
┌────────────┐     ┌──────────┐     ┌────────────┐
│  WORLD     │────→│ WORKING  │←────│  SOCIAL/   │
│  STATE     │     │ MEMORY   │     │  USER      │
│  MEMORY    │     └──────────┘     │  MEMORY    │
└────────────┘                      └────────────┘
     │                                  │
     │ restore last-known state         │ load user preferences
     v                                  v
  available immediately             context initialized
```

---

## Consolidation Pipeline

Between runs (or idle), a background process consolidates memory:

```
Frequency  Process                    Input → Output
─────────  ──────────────────────────────────────────────
per run    Episodic → Semantic        Extract facts from new episode,
                                       update belief confidence

per N runs Episodic → Episodic        Compress/archive old episodes,
            (retention)                forget below-importance threshold

per N runs Semantic → Procedural      Pattern of successful facts → skill,
                                       skill refinement from failure facts

per N runs Meta-audit                 Scan for contradictions,
                                       identify knowledge gaps,
                                       recalibrate confidence

on demand  Working → Episodic         Save current context as episode
            (save checkpoint)
```

---

## Query Interface

```python
class HierarchicalMemory:
    # Episodic
    def recall_episode(self, episode_id: str) -> Episode
    def search_episodes(self, query: str, limit: int = 10) -> list[Episode]
    def timeline(self, domain: str, start: datetime, end: datetime) -> list[Episode]

    # Semantic
    def query_facts(self, subject: str = None, predicate: str = None) -> list[Fact]
    def belief(self, statement: str) -> float           # confidence in a claim
    def relationships(self, entity: str) -> Graph        # entity's connections

    # Procedural
    def applicable_skills(self, task: str, context: dict) -> list[Skill]
    def compose_plan(self, goal: str, skills: list[Skill]) -> Plan

    # Working
    def push_context(self, block: ContentBlock)
    def pop_context(self) -> ContentBlock
    def attend_to(self, topic: str)                     # bring relevant context forward

    # World State
    def get_state(self, key: str) -> Any
    def predict_state(self, action: str) -> dict         # what would happen if...

    # Social
    def user_profile(self, user_id: str) -> UserProfile
    def adapt_to_user(self, user_id: str, response: str) -> str  # style adapt

    # Meta
    def knowledge_gaps(self, task: str) -> list[str]     # what we're missing
    def confidence_in(self, claim: str) -> float         # calibrated certainty
    def contradictions(self) -> list[Conflict]           # conflicting beliefs

    # Consolidation
    def consolidate(self) -> ConsolidationReport         # run full pipeline
```

---

## Implementation Status

| Phase | What | Status |
|---|---|---|
| 1 | Working Memory | **Done** — importance-ranked sliding window, attention mechanism, scratchpad |
| 2 | Episodic → Semantic pipeline | **Done** — fact extraction from traces, fact table with confidence tracking |
| 3 | World State Memory | **Done** — state snapshot table, diff-logging, staleness tracking; minimal current implementation |
| 4 | Retention & consolidation | **Done** — consolidation pipeline (per-run, per-N-runs, on-demand) |
| 5 | Social/User Memory | **Done** — user profiles, preference inference, trust calibration |
| 6 | Meta-Memory | **Done** — knowledge gap detection, contradiction resolution, confidence calibration |
| 7 | Full integration | **Done** — all tiers online, cross-tier queries, consolidation pipeline automated |

---

## Key Differentiator

**Flat store:** The agent learned **what action to take** (pattern matching).

**Hierarchical memory:** The agent learns **what is true about the world** (knowledge), **how to do things** (skills), **who it's working with** (social), and **what it knows and doesn't know** (meta-cognition).

Patterns → Knowledge. Reaction → Understanding.
