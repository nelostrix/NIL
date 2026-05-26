from __future__ import annotations

import concurrent.futures
import logging
import time
import traceback
from typing import Any, Callable, Optional

from nil.kernel.graph.graph_types import (
    EdgeType,
    ExecutionGraph,
    ExecutionNode,
    GraphOutcome,
    NodeResult,
    NodeStatus,
    NodeType,
)

logger = logging.getLogger("nil.kernel.graph.scheduler")

NodeExecutor = Callable[[ExecutionNode, ExecutionGraph], NodeResult]


class GraphScheduler:
    def __init__(
        self,
        max_concurrency: int = 5,
        max_cost_per_run: float = 2.0,
        timeout_per_node_ms: int = 60000,
    ):
        self.max_concurrency = max_concurrency
        self.max_cost_per_run = max_cost_per_run
        self.timeout_per_node_ms = timeout_per_node_ms
        self._executor_map: dict[NodeType, NodeExecutor] = {}

    def register_executor(self, node_type: NodeType, executor: NodeExecutor) -> None:
        self._executor_map[node_type] = executor

    def unregister_executor(self, node_type: NodeType) -> None:
        self._executor_map.pop(node_type, None)

    def execute(self, graph: ExecutionGraph, context: dict[str, Any] = None) -> GraphOutcome:
        start_time = time.time()
        graph.metadata.max_concurrency = min(
            graph.metadata.max_concurrency or self.max_concurrency,
            self.max_concurrency,
        )
        max_conc = graph.metadata.max_concurrency
        node_results: dict[str, NodeResult] = {}
        total_cost = 0.0
        context = context or {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_conc) as pool:
            in_flight: dict[concurrent.futures.Future, str] = {}
            fallback_chain: set[str] = set()

            while not graph.all_completed():
                ready = graph.find_ready_nodes(max_concurrency=max_conc)

                available = max_conc - len(in_flight)
                for node in ready[:available]:
                    if total_cost >= (graph.metadata.total_cost_budget or self.max_cost_per_run):
                        logger.warning("Cost budget exceeded, stopping new nodes")
                        node.status = NodeStatus.SKIPPED
                        continue

                    node.status = NodeStatus.RUNNING
                    future = pool.submit(self._execute_node, node, graph, context)
                    in_flight[future] = node.node_id

                if not in_flight:
                    if not graph.find_ready_nodes(max_concurrency=max_conc):
                        break
                    time.sleep(0.01)
                    continue

                done_futures = []
                for future in in_flight:
                    try:
                        result = future.result(timeout=0.01)
                        done_futures.append(future)
                        node_id = in_flight[future]

                        result.duration_ms = (time.time() - start_time) * 1000
                        node_results[node_id] = result
                        node = graph.get_node(node_id)

                        if result.success:
                            graph.set_node_output(node_id, result.output)
                            graph.mark_completed(node_id)
                        else:
                            total_cost += result.cost
                            if node and node.fallback_node:
                                graph.activate_fallback(node_id, node.fallback_node)
                            elif node and node.rollback_node:
                                graph.activate_rollback(node_id, node.rollback_node)
                            else:
                                graph.mark_failed(node_id, result.error or "Unknown error")
                    except concurrent.futures.TimeoutError:
                        continue
                    except Exception as e:
                        for f, nid in list(in_flight.items()):
                            if f == future:
                                done_futures.append(f)
                                node_results[nid] = NodeResult(
                                    node_id=nid, success=False, error=str(e)
                                )
                                graph.mark_failed(nid, str(e))
                                break

                for f in done_futures:
                    in_flight.pop(f, None)

        duration_ms = (time.time() - start_time) * 1000
        failed = [nid for nid, r in node_results.items() if not r.success]

        if failed and not self._has_recovery_path(graph, failed):
            first_error = node_results[failed[0]].error or "Execution failed"
            return GraphOutcome.failure(
                error=first_error,
                node_results=node_results,
                duration_ms=duration_ms,
            )

        output = graph.collect_outputs()
        return GraphOutcome.success(
            output=output,
            node_results=node_results,
            duration_ms=duration_ms,
        )

    def _execute_node(
        self, node: ExecutionNode, graph: ExecutionGraph, context: dict
    ) -> NodeResult:
        node_type = node.node_type
        executor = self._executor_map.get(node_type)

        if node_type == NodeType.MERGE:
            return self._execute_merge(node, graph)
        if node_type == NodeType.INPUT:
            return NodeResult(node_id=node.node_id, success=True, output=node.task)
        if node_type == NodeType.OUTPUT:
            outputs = graph.collect_outputs()
            return NodeResult(node_id=node.node_id, success=True, output=outputs)

        if executor:
            try:
                return executor(node, graph)
            except Exception as e:
                logger.error("Executor for %s failed: %s", node_type, e)
                return NodeResult(
                    node_id=node.node_id, success=False, error=f"{e}\n{traceback.format_exc()}"
                )

        return NodeResult(
            node_id=node.node_id,
            success=False,
            error=f"No executor registered for node type: {node_type}",
        )

    def _execute_merge(self, node: ExecutionNode, graph: ExecutionGraph) -> NodeResult:
        deps = graph.get_dependencies(node.node_id)
        merged = {}
        for dep_id in deps:
            output = graph.get_node_output(dep_id)
            if output is not None:
                if isinstance(output, dict):
                    merged.update(output)
                else:
                    merged[dep_id] = output
        return NodeResult(node_id=node.node_id, success=True, output=merged)

    def _has_recovery_path(self, graph: ExecutionGraph, failed_node_ids: list[str]) -> bool:
        for edge in graph.edges:
            if edge.edge_type in (EdgeType.CONDITIONAL,):
                if edge.source in failed_node_ids:
                    target = graph.get_node(edge.target)
                    if target and target.status == NodeStatus.PENDING:
                        return True
        return any(
            n.fallback_node or n.rollback_node
            for nid, n in graph.nodes.items()
            if nid in failed_node_ids
        )



