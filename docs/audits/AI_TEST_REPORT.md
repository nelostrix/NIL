# AI Test Report

Audit date: 2026-11-14. Scope: NIL at `ed7f987ced82bbb59c8373c1d9bd3d9dd7cd8563` and adjacent NELO Studio IDE at `e517c6e02cd1e8bf59fccaa8c947143cae67537c`. Reports reside in NIL because that is the writable workspace. NELO Studio **is the IDE**; NIL is an agent SDK/backend, not the complete IDE. Existing audit documents were not treated as test evidence. No existing source code was modified.

Evidence is local source inspection and explicitly listed safe probes. No external API requests, dependency downloads, model downloads, cloud jobs, deployments or hardware access were performed. Optional installed runtimes and release bundles may differ from these checkouts; their functionality is not assumed.

## Environment and safety

Python 3.14.10 on this container. `node` is unavailable. pytest, NumPy, torch, TensorFlow, sklearn, OpenAI, Anthropic, google.genai and stacc were not discoverable in this Python environment. Importability was queried, not model availability or compatibility. No dependency installation was needed to establish these limitations, and none was attempted. No network, provider inference, physical interfaces, cloud execution or deployed serving was exercised.

Source scanning and temporary copied SDK probes were used to avoid modifying source. Studio unittest runs set PYTHONDONTWRITEBYTECODE=1. The packaging suite uses mocked downloads and temporary archives; its successful shell fixture is not an actual clangd release or live server session. Source copies were created at /tmp/nil-audit-x6513yuy; they may not survive environment cleanup.

## Commands and results

| ID | Command / method | Actual result | Meaning |
|---|---|---|---|
| E1 | `rg --files`, source reads, targeted symbol searches, `git rev-parse HEAD` in both repos | Revisions recorded in report header | Static evidence only; no build verification |
| E2 | `python --version`; `node --version`; importlib.util.find_spec probes | Python 3.14.10; node command not found; listed ML/test/provider packages absent | This environment cannot run IDE Node scripts or ML suites |
| E3 | AST parse of all 166 Python files under NIL | 165 parse; nelo_vision_mini.py:8 SyntaxError | Syntax check, not imports or semantic validation |
| E4 | `python -m pytest --version` | exit 1: No module named pytest | NIL pytest suites were not run; zero pytest passes claimed |
| E5 | `PYTHONPATH=/tmp/nil-audit-x6513yuy/nil-sdk PYTHONDONTWRITEBYTECODE=1 python -c 'import nil'` from the temporary copy | exit 1: No module named numpy, via nil/__init__.py → robotics → dmp.py | Public SDK cannot import here with advertised metadata alone |
| E6 | `PYTHONDONTWRITEBYTECODE=1 python ../NELO-Studio/tests/python/test_native_language_servers.py` | 3 tests, OK, 0.047s | Packaging checksum plumbing, extraction layout and traversal rejection only |
| E7 | `PYTHONDONTWRITEBYTECODE=1 python ../NELO-Studio/tests/python/test_training_artifacts.py` | exit 1 at import: No module named nelo_train | No artifact test body ran |
| E8 | `PYTHONDONTWRITEBYTECODE=1 python ../NELO-Studio/tests/python/test_onnx_serving.py` | exit 1 at import: No module named nelo_train | No ONNX/provider test body ran; existing ONNX test also mocks NumPy/ORT |
| E9 | `ls ../NELO-Studio/runtimes`; `rg --files` and runtime reference searches | Directory absent; runtimeDir defaults there; /runtimes/ is ignored | External installation/NELO_PY_RUNTIME could change this; default source checkout incomplete |
| T1 | Namespace-loaded ToolManager with `numeric(x: int)` and x='not-an-int' | output='not-an-int', error=None | Dispatch works; argument type enforcement is broken |
| T2 | `@tool` schema of `joints(q: List[float])` | q schema is type string | Typed joint arrays are misrepresented |
| T3 | Temporary SQLite trace write, close, reopen and trace-count read | count=1 | Narrow persistent-memory roundtrip verified |
| T4 | Direct call to legacy NILBridgeState.chat('run simulation') | tool MuJoCo.simulate(), status success without backend | Fabricated behavior reproduced; no HTTP server started |
| T5 | Attempt to import nil.agent after synthetic namespace setup | AttributeError: nil has no attribute PersistentMemory | Probe harness artifact; NOT a product defect or evidence for public-import failure (use E5) |
| T6 | Extracted root _resolve_llm with provider local and model test | ('openai', 'test') | Mapping verified; no model inference |
| T7 | Extracted root _resolve_llm with keyless OpenAI | ValueError: No API key is set for openai. | Missing-key check verified |

The three passing unittest cases are the only existing test-suite passes claimed. The custom probes are separate observations with explicit expectations, not a reported pytest suite. Failed test imports are not failed model predictions. Static defects (plan mode, zero-pose fallback, unused safety thresholds, missing Docker command target) were not executed through a live application.

## Reproducible custom probe

Create an isolated copy before executing this script; set `root` to that copy. The synthetic namespace intentionally bypasses nil/__init__.py so stdlib submodules can be inspected without installing NumPy. It does **not** prove the public package imports or an Agent runs. The script extracts a small AST subset for the pure resolver so nil_server.py cannot start, create repository databases or reach STACC. No secrets are supplied.

```python
import sys,types,pathlib,tempfile,ast,json,importlib.util
root=pathlib.Path(__file__).parent
# Namespace-only loading deliberately bypasses broken public dependency chain.
n=types.ModuleType('nil');n.__path__=[str(root/'nil-sdk/nil')];sys.modules['nil']=n
from nil.tools.base import tool
from nil.kernel.tool_manager import ToolManager
from nil.memory.store import PersistentMemory
from typing import List
@tool
def numeric(x:int): return x
@tool
def joints(q:List[float]): return q
m=ToolManager([numeric]); print('T1 type_validation',m.execute('numeric',{'x':'not-an-int'}))
print('T2 generic_schema',joints.to_schema())
with tempfile.TemporaryDirectory() as d:
 s=PersistentMemory(d);s.store_trace({'run_id':'r','domain':'d','created_at':1,'outcome_success':True});s.close()
 s=PersistentMemory(d);print('T3 sqlite_roundtrip',s.get_trace_count('d'));s.close()
spec=importlib.util.spec_from_file_location('legacy',root/'nil-sdk/nil/server.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
b=mod.NILBridgeState();print('T4 legacy_fake',b.chat('run simulation')['tool_call'])
try: import nil.agent
except Exception as e: print('T5 SDK dependency import',type(e).__name__,str(e))
# Compile only model resolution helpers: no module imports, server startup, or robot calls.
tree=ast.parse((root/'nil_server.py').read_text()); nodes=[x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.Assign)) and (getattr(x,'name','')=='_resolve_llm' or isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['_OPENAI_COMPATIBLE_BASE_URLS','_PROVIDER_KEY_ENV'] for t in x.targets))]
from typing import Dict,Any,Tuple
g={'Dict':Dict,'Any':Any,'Tuple':Tuple};exec(compile(ast.Module(body=nodes,type_ignores=[]),'extracted','exec'),g)
print('T6 local_selection',g['_resolve_llm']({'provider':'local','model':'test'})[:2])
try:g['_resolve_llm']({'provider':'openai','model':'test'})
except ValueError as e: print('T7 missing_key_rejected',str(e))
```

## Tests blocked or intentionally omitted

- NIL's 22 test files were inspected as available test assets, but not executed because pytest and NumPy are absent. Existing benchmark_report.json is historical evidence only; no benchmark result was adopted.
- Node typecheck, domain tests, Next production build and browser/Tauri tests were blocked by missing Node and no installed frontend dependencies. No dependency downloads or build claims.
- Actual LSP, DAP, PTY, Jupyter and extension processes were not launched. Packaging mocks do not verify these sessions.
- Training, optimizer/checkpoint recovery, TensorFlow/sklearn adapters, distributed ranks, local Ollama and cloud model contracts need a prepared isolated environment.
- ONNX Runtime, GPU/CUDA/TensorRT/Metal behavior, memory/throughput claims, quantization accuracy and model drift were not tested.
- No physical hardware or robotics daemon was contacted. Gatekeeper and STACC conclusions are static unless explicitly labeled otherwise.
- No exploit attempt, vulnerability-database scan, complete dependency SBOM, exhaustive dynamic reachability proof or cross-platform release certification was performed.

## Required follow-up validation

Use a clean installed-wheel environment and the restored versioned training runtime; run `python -m pytest nil-sdk/tests` with a forced stub provider and all provider keys unset. Add mocked provider contract tests before optional live tests. In a prepared Node >=22.16 environment run the IDE's typecheck/domain tests/build on a copy. Run tiny CPU training→checkpoint→resume→registry→serve→rollback tests and mock robotics action-policy tests. Only perform GPU, live provider, HIL and hardware acceptance with explicit resources and authorization.
