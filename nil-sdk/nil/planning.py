"""Explicit tool plans with schema validation, owned deadlines and resumable traces."""
import hashlib
import json
import importlib
import time
from pathlib import Path
from nil.lifecycle import atomic_json
from nil.tools.base import Tool


def _canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def _publish(path, state):
    if len(_canonical(state).encode()) > 1024*1024:
        raise ValueError('Journal exceeds 1 MiB')
    atomic_json(path, state)


def _invoke(tool, arguments):
    if isinstance(tool, dict):
        value = importlib.import_module(tool['module'])
        for part in tool['name'].split('.'):
            value = getattr(value, part)
        return value.invoke(arguments)
    return tool.invoke(arguments)


def _transfer(tool):
    # Decorated global functions are replaced by Tool objects, so their original
    # fn is not directly pickleable. Resolve the approved global Tool in worker.
    fn = tool.fn
    if fn is not None and '<locals>' not in getattr(fn, '__qualname__', ''):
        try:
            value = importlib.import_module(fn.__module__)
            for part in fn.__qualname__.split('.'):
                value = getattr(value, part)
            if value is tool:
                return dict(module=fn.__module__, name=fn.__qualname__)
        except (AttributeError, ImportError):
            pass
    return tool


def run(steps, tools, *, allowed, journal, tool_versions, max_steps=32, timeout=30., resume=False, cancel_event=None):
    """Run a declared ordered plan. Completed calls are never repeated on resume.

    Interrupted/failed calls require caller reconciliation; automatic retry is
    refused because a tool may already have performed an external side effect.
    Tools must be importable/picklable trusted local Tool objects.
    """
    from nil.execution import submit, Limits
    if type(max_steps) is not int or not 1<=max_steps<=1000:raise ValueError('Invalid step budget')
    if not isinstance(steps,list) or not 1<=len(steps)<=max_steps:raise ValueError('Plan exceeds step budget')
    if not isinstance(tools,dict) or any(not isinstance(t,Tool) for t in tools.values()):raise TypeError('tools must map names to Tool objects')
    if not isinstance(tool_versions,dict) or any(not isinstance(v,str) or not v for v in tool_versions.values()):raise ValueError('Explicit tool versions required')
    allowed=set(allowed);seen=set()
    for row in steps:
        if not isinstance(row,dict) or set(row)!={'id','tool','arguments'} or not isinstance(row['id'],str) or not row['id'].isidentifier() or row['id'] in seen:raise ValueError('Invalid/duplicate plan step')
        if row['tool'] not in tools or row['tool'] not in allowed or row['tool'] not in tool_versions:raise PermissionError('Plan requests an unavailable or unapproved tool')
        if not isinstance(row['arguments'],dict):raise ValueError('Tool arguments must be an object')
        def references(value):
            if isinstance(value, dict):
                if set(value) == {'$ref'}:
                    if not isinstance(value['$ref'], str) or value['$ref'] not in seen:
                        raise ValueError('References must point to prior plan steps')
                else:
                    for nested in value.values(): references(nested)
            elif isinstance(value, list):
                for nested in value: references(nested)
        references(row['arguments'])
        seen.add(row['id'])
    limits=Limits(timeout=timeout,payload_mb=1,log_bytes=8192)
    text=_canonical(dict(steps=steps,versions=tool_versions,allowed=sorted(allowed),timeout=timeout))
    if len(text)>65536:raise ValueError('Plan exceeds 64 KiB')
    identity=hashlib.sha256(text.encode()).hexdigest();path=Path(journal)
    if path.is_symlink():raise ValueError('Journal must not be a symlink')
    if resume:
        if path.stat().st_size>1024*1024:raise ValueError('Journal exceeds 1 MiB')
        state=json.loads(path.read_text())
        if state.get('schema_version')!=1 or state.get('plan_sha256')!=identity or not isinstance(state.get('completed'),list):raise ValueError('Plan/version mismatch')
        if state.get('status') not in {'ready','completed','cancelled'}:raise RuntimeError('Reconcile interrupted/failed tool execution before resuming')
    else:
        if path.exists():raise FileExistsError(path)
        state=dict(schema_version=1,plan_sha256=identity,status='ready',completed=[])
        _publish(path,state)
    if len(state['completed'])>len(steps):raise ValueError('Invalid completed prefix')
    for step,done in zip(steps,state['completed']):
        if not isinstance(done,dict) or set(done)!={'id','result'} or done['id']!=step['id']:raise ValueError('Invalid completed prefix')
    results={r['id']:r['result'] for r in state['completed']}
    def resolve(value):
        if isinstance(value,dict):
            if set(value)=={'$ref'}:
                if not isinstance(value['$ref'],str) or value['$ref'] not in results:raise ValueError('Reference requires a prior completed step ID')
                return results[value['$ref']]
            return {k:resolve(v) for k,v in value.items()}
        if isinstance(value,list):return [resolve(v) for v in value]
        return value
    for step in steps[len(state['completed']):]:
        if cancel_event is not None and cancel_event.is_set():
            state.update(status='cancelled',active_step=None);_publish(path,state);return json.loads(_canonical(state))
        arguments=resolve(step['arguments']);tools[step['tool']].validate(arguments)
        state.update(status='running',active_step=step['id']);_publish(path,state)
        job=None
        try:
            job=submit(_invoke,_transfer(tools[step['tool']]),arguments,limits=limits,job_root=path.parent/'tool_jobs')
            while job.status()['status'] not in {'completed','failed','timed_out','cancelled','interrupted','resource_limited'}:
                if cancel_event is not None and cancel_event.is_set():job.cancel()
                time.sleep(.02)
            result=job.result()
            if len(_canonical(result).encode())>16384:raise ValueError('Tool result exceeds 16 KiB')
        except BaseException:
            if job is not None and job.status()['status'] not in {'completed','failed','timed_out','cancelled','interrupted','resource_limited'}:
                job.cancel()
                try: job.result()
                except BaseException: pass
            state['status']='failed';_publish(path,state);raise
        results[step['id']]=result;state['completed'].append(dict(id=step['id'],result=result))
        state.update(status='ready',active_step=None);_publish(path,state)
    state['status']='completed';_publish(path,state)
    return json.loads(_canonical(state))


def shortest_path(start, goal, adjacency, *, max_states=10000):
    """Breadth-first shortest unweighted path over explicit string states."""
    from collections import deque
    if not isinstance(start, str) or not isinstance(goal, str) or not isinstance(adjacency, dict) or type(max_states) is not int or not 1 <= max_states <= 100000:
        raise ValueError('Invalid planning problem')
    queue = deque([start]); previous = {start: None}
    while queue:
        node = queue.popleft()
        if node == goal:
            path = []
            while node is not None:
                path.append(node); node = previous[node]
            return path[::-1]
        values = adjacency.get(node, [])
        if not isinstance(values, (list, tuple)) or any(not isinstance(v, str) for v in values):
            raise ValueError('Neighbors must be a sequence of string states')
        for neighbor in sorted(values):
            if neighbor not in previous:
                if len(previous) >= max_states: raise ValueError('Planner state budget exceeded')
                previous[neighbor] = node; queue.append(neighbor)
    return []
