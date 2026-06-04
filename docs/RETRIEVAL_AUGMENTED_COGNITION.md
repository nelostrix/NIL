# Retrieval-Augmented Cognition (RAC)

> **Status: Implemented** — The full RAC system exists in `nil/cognition/`. `RetrievalQuery`, `RetrievalIndex` (dense + sparse + constraint-based), `HybridRetriever` (fused dense/sparse/structure retrieval with cross-encoder re-ranking), `ContextAssembler` (token-budget-optimized prompt assembly), `RetrievalFeedback` (outcome-driven score updates), `RetrievalStrategySelector`, and `RACEngine` are all implemented. The `CognitiveLoop` (10-phase unified cognitive loop) integrates RAC as the retrieval phase.

---

## Background

Nil originally injected all learned skills into the system prompt as a flat block of text. Every run, every task, every domain context — the agent saw the complete skill list regardless of relevance.

Retrieval-Augmented Cognition (RAC) replaces static injection with a dynamic retrieval system inspired by retrieval-augmented generation (RAG), extended to cover all memory types and optimized for agentic cognition rather than document Q&A.

---

## Before RAC

```
┌─────────────────────────────────────────────────────┐
│              Static Prompt Injection                │
├─────────────────────────────────────────────────────┤
│                                                      │
│  System Prompt:                                       │
│    You are an agent specialized in incident-response │
│    Objectives: ...                                    │
│                                                      │
│    Learned Skills (from past experience):             │
│      • search_then_analyze_then_fix [92%]            │
│      • check_logs_skill [85%]                        │
│      • restart_service_then_verify [78%]             │
│      • search_logs_x2_then_deploy [74%]              │
│      • diagnose_network [71%]                        │
│      • escalate_then_document [65%]                  │
│      • ... 15 more skills ...                        │
│                                                      │
│    Available tools: ...                               │
│                                                      │
└─────────────────────────────────────────────────────┘
```

**Problems that emerge at scale:**

| Problem | Consequence |
|---|---|
| Token bloat | 20 skills × 200 tokens = 4K tokens before the task even starts |
| Context distraction | Irrelevant skills confuse the LLM, dilute attention |
| Retrieval noise floor | Below ~50% confidence, skills are more distracting than helpful |
| No task-awareness | Injection is identical for "API latency spike" and "deploy new release" |
| Skill conflict | Two skills may contradict — static injection provides no resolution |
| No failure awareness | Skills from success are injected, but relevant failure patterns are absent |
| Static ordering | The LLM sees skills in insertion order, not relevance order |

---

## Architecture

```
                Task: "API latency spike on /v2/users"
                              │
                              v
                    ┌─────────────────┐
                    │  Query Encoder  │
                    │  (task + state) │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────────┐
              │              │                   │
              v              v                   v
   ┌────────────────┐ ┌──────────────┐ ┌──────────────────┐
   │ Sparse Retriever│ │Dense Retriever│ │ Hybrid Fuser    │
   │ (BM25/SPLADE)  │ │ (embedding)  │ │ (re-rank + mix) │
   └────────────────┘ └──────────────┘ └────────┬─────────┘
              │              │                   │
              └──────────────┴───────────────────┘
                                                 │
                                                 v
                                      ┌────────────────────┐
                                      │  Multi-Source      │
                                      │  Retriever         │
                                      │                    │
                                      │  ┌──────────────┐  │
                                      │  │ Skill Memory │  │
                                      │  └──────────────┘  │
                                      │  ┌──────────────┐  │
                                      │  │ Failure      │  │
                                      │  │ Episodes     │  │
                                      │  └──────────────┘  │
                                      │  ┌──────────────┐  │
                                      │  │ Successful   │  │
                                      │  │ Trajectories │  │
                                      │  └──────────────┘  │
                                      │  ┌──────────────┐  │
                                      │  │ Domain       │  │
                                      │  │ Constraints  │  │
                                      │  └──────────────┘  │
                                      │  ┌──────────────┐  │
                                      │  │ Semantic     │  │
                                      │  │ Facts        │  │
                                      │  └──────────────┘  │
                                      └────────┬───────────┘
                                               │
                                               v
                                   ┌───────────────────────┐
                                   │  Context Assembler    │
                                   │  (format, rank, trim) │
                                   │                       │
                                   │  Output: token-budget │
                                   │  optimized prompt     │
                                   │  block                │
                                   └───────────────────────┘
                                               │
                                               v
                                   ┌───────────────────────┐
                                   │    LLM (Agent Loop)   │
                                   └───────────────────────┘
```

---

## Retrieval Sources

Each source has a dedicated index and retrieval strategy:

| Source | Index Type | Content | Retrieval Trigger |
|---|---|---|---|
| **Skill Memory** | Dense (embedding) | Skill name + description + trigger conditions | Task description, current tool state |
| **Failure Episodes** | Sparse + Dense hybrid | Failed trace summaries + error signatures | Task description, errors seen so far |
| **Successful Trajectories** | Dense (embedding) | Full successful traces | Task description, current partial trajectory |
| **Domain Constraints** | Sparse (keyword/rule) | Constraint descriptions | Tool calls, state transitions |
| **Semantic Facts** | Dense (embedding) | Declarative facts about domain | Task description, current hypothesis |
| **Working Memory Context** | Cross-attention | Recent steps, current state | Every turn (not retrieval, but context window) |

---

## Retrieval Pipeline

### Step 1: Query Construction

The query is not just the raw task string — it's a structured composite:

```python
class RetrievalQuery:
    task: str                             # original task
    current_tool_state: list[str]         # tools called so far
    last_error: Optional[str]             # most recent error
    current_step: int                     # where we are in the loop
    domain: str                           # domain namespace
    recent_outputs: list[str]             # last few LLM outputs
    active_hypothesis: Optional[str]      # what the LLM is currently investigating
```

### Step 2: Multi-Vector Encoding

The query is encoded multiple ways for different retrievers:

```python
# Dense embedding (semantic similarity)
query_embedding = embedder.encode(query.task)

# Sparse encoding (keyword overlap)
query_bow = tokenizer.bow(query.task + " " + " ".join(query.current_tool_state))

# Structured encoding (for constraints)
query_features = {
    "tools_used": set(query.current_tool_state),
    "error_type": classify_error(query.last_error),
    "domain": query.domain,
}
```

### Step 3: Hybrid Retrieval (Sparse + Dense)

```python
def retrieve(query: RetrievalQuery, top_k: int = 5) -> list[ScoredResult]:
    # Dense retrieval — captures semantic similarity
    dense_results = dense_index.search(
        query_embedding,
        sources=["skills", "trajectories", "facts"],
        top_k=top_k * 2,
    )

    # Sparse retrieval — captures exact keyword matches
    sparse_results = sparse_index.search(
        query_bow,
        sources=["failures", "constraints"],
        top_k=top_k * 2,
    )

    # Structure-based retrieval — captures rule matches
    structure_results = constraint_index.match(
        query_features,
        top_k=top_k,
    )

    # Fuse and re-rank
    all_results = fuse(
        dense_results,
        sparse_results,
        structure_results,
        weights=[0.5, 0.3, 0.2],        # learned or configured
    )

    # Re-rank with cross-encoder for precision
    re_ranked = cross_encoder.rerank(query.task, all_results)

    return re_ranked[:top_k]
```

### Step 4: Context Assembly

The retrieved items are assembled into a token-budget-optimized prompt block:

```python
class ContextAssembler:
    max_tokens: int = 2048               # budget for retrieved context

    # Per-source allocation (learned from usage stats)
    source_budget: dict[str, float] = {
        "skills": 0.30,                  # 30% of budget
        "trajectories": 0.25,            # 25%
        "failures": 0.20,                # 20%
        "constraints": 0.15,             # 15%
        "facts": 0.10,                   # 10%
    }

    def assemble(self, results: list[ScoredResult]) -> str:
        # Allocate tokens per source
        # Within each source, select items by score until budget exhausted
        # Format into a structured prompt block
        # Apply compression (compact mode for low-scoring items)
        ...
```

---

## Prompt Integration

Instead of a flat "Learned Skills" block, the prompt contains a dynamic cognition block:

```
RELEVANT PAST EXPERIENCE (retrieved for this task):
── Skills ──
  • search_logs_then_analyze: When investigating latency,
    first search logs for timeout patterns, then analyze
    response times. [97% confidence, 24 uses]

── Similar Successful Run ──
  Task: "API timeout on /v1/users endpoint"
  Sequence: search_logs → analyze_metrics → check_db_connections
    → restart_service → verify_recovery
  Note: The root cause was a connection pool exhaustion.
    Restarting without draining caused a brief second outage.
    Recommended: drain connections before restart.

── Related Failures ──
  • Error "connection_refused" appeared after restarting
    without draining (3 failures). Avoid: restart → immediate verify.
    Instead: drain → restart → wait → verify.

── Domain Constraints ──
  • Always drain active connections before service restart.
  • Never restart the primary without promoting a replica first.
```

---

## Index Management

### Embedding Models

| Tier | Model | Use |
|---|---|---|
| Online | `text-embedding-3-small` or `voyage-2` | Production retrieval |
| Offline | `BAAI/bge-small-en-v1.5` | Local development |
| Cross-encoder | `BAAI/bge-reranker-v2-m3` | Re-ranking precision |

### Index Updates

```
Event                    Action
───────────────────────  ─────────────────────────────
New skill extracted      Index immediately (priority)
New trace completed     Batch index every N runs or async
Skill deprecated        Remove from index or mark stale
Constraint discovered   Index immediately (priority)
Fact updated            Update embedding, re-index
```

### Index Structure

```python
class RetrievalIndex:
    # Dense index (vector DB or approximate nearest neighbors)
    dense: AnnoyIndex | FAISSIndex | HNSWIndex

    # Sparse index (inverted)
    sparse: BM25Index | SPLADEIndex

    # Metadata store
    metadata: dict[str, StoredItem]    # id → {source, content, score, timestamp}

    # Constraint matcher (structured)
    constraint_matcher: ConstraintIndex

    def search(self, query, top_k) -> list[ScoredResult]
    def upsert(self, item: StoredItem)
    def delete(self, item_id: str)
    def refresh(self)                    # rebuild from memory store
```

---

## Context Budget Management

Token budget is a first-class concept, not a post-hoc trim:

```
Total context window: 200K tokens (Claude) or 128K (GPT-4)

Allocation:
  ┌──────────────────────────────────────────┐
  │  System prompt (base)       ~500 tokens  │
  │  Objectives                 ~100 tokens  │
  │  Tool definitions           ~800 tokens  │
  │  Retrieved cognition        ~2K tokens   │ ← dynamic, this is the budget
  │  Conversation history       ~variable    │
  │  Current response           ~variable    │
  └──────────────────────────────────────────┘

Retrieved cognition budget (2K tokens):
  ┌──────────────────────────────────────────┐
  │  Skills (top-3, ~600 tokens)             │
  │  Trajectory (top-1, ~500 tokens)         │
  │  Failures (top-2, ~400 tokens)           │
  │  Constraints (top-3, ~300 tokens)        │
  │  Facts (top-2, ~200 tokens)              │
  └──────────────────────────────────────────┘
```

When budget is exceeded, items are:
1. Dropped lowest-score first
2. Or compressed: full → compact → title-only → dropped

---

## Retrieval Strategies by Task Type

Different task types benefit from different retrieval strategies:

| Task Type | Retrieval Strategy |
|---|---|
| **Debug/fix** (error occurred) | Prioritize failure episodes with matching error signatures. Retrieve trajectories that resolved similar errors. Demote skills — the agent needs to avoid failure, not repeat success |
| **New feature/build** | Prioritize successful trajectories from similar tasks. Retrieve relevant skills. Retrieve constraints that might apply |
| **Investigation** (unknown) | Prioritize semantic facts about the domain. Retrieve multiple diverse trajectories. Retrieve constraints to understand boundaries |
| **Routine operation** (known pattern) | Retrieve top-1 matching skill. Minimal context — the skill itself may be sufficient |
| **Recovery** (after failure) | Prioritize failure avoidance rules. Retrieve what NOT to do. Retrieve the last successful trajectory before failure |

```python
class RetrievalStrategySelector:
    def select(self, query: RetrievalQuery) -> RetrievalStrategy:
        if query.last_error:
            return FailureRecoveryStrategy()
        if is_novel_task(query.task, query.domain):
            return ExplorationStrategy()
        if is_routine_task(query.task, query.domain):
            return MinimalStrategy()
        return BalancedStrategy()
```

---

## Relevance Scoring

Not all retrieved items are equally useful. Scoring considers multiple signals:

```python
def score_item(item: StoredItem, query: RetrievalQuery) -> float:
    return (
        semantic_similarity(item, query) * 0.35 +

        recency_score(item, query) * 0.15 +
        # Recent items are more relevant

        specificity_score(item, query) * 0.20 +
        # Items that mention specific tools/errors in the query score higher

        confidence_score(item) * 0.15 +
        # Items with higher confidence/higher success rate win

        diversity_penalty(item, already_selected) * 0.15
        # Penalize items that are too similar to already-selected ones
    )
```

---

## Feedback Loop: Retrieval Improves with Usage

Every run provides implicit feedback on retrieval quality:

```python
class RetrievalFeedback:
    def record_outcome(
        self,
        query: RetrievalQuery,
        retrieved_items: list[ScoredResult],
        outcome: RunOutcome,
    ):
        if outcome.success:
            # Boost items that were retrieved and used
            for item in retrieved_items:
                if item.was_used_in_response(outcome):
                    item.boost_score(+0.05)
        else:
            # Penalize items that were retrieved but led to failure
            for item in retrieved_items:
                if item.was_used_in_response(outcome):
                    item.boost_score(-0.10)
                else:
                    # Retrieved but not used = may have been irrelevant
                    item.boost_score(-0.02)

        # Update retriever weights based on what worked
        self.update_retriever_weights(outcome)
```

---

## Implementation Status

| Phase | What | Status |
|---|---|---|
| 1 | Sparse retrieval (BM25) | **Done** — keyword-based top-k retrieval replaces static injection |
| 2 | Dense embeddings | **Done** — embedding index, hybrid sparse+dense, cross-encoder re-ranking |
| 3 | Multi-source retrieval | **Done** — indexes failures, trajectories, constraints, skills |
| 4 | Context assembler | **Done** — token budget management, per-source allocation, compact mode |
| 5 | Strategy selection | **Done** — route retrieval strategy by task type |
| 6 | Feedback loop | **Done** — outcome-driven score boost/penalty, retriever weight updates |
| 7 | Online indexing | **Done** — real-time index updates on new traces/skills/constraints |

---

## Key Differentiator

**Without RAC:**
The agent sees everything, all the time. More experience → more token bloat → more distraction → worse performance. Scale actively degrades quality.

**With RAC:**
The agent sees only what's relevant, right now. More experience → better retrieval candidates → more precise context → better performance. Scale improves quality.
