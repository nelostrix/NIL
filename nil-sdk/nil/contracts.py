"""Small backend-neutral protocols and metadata views; no numerical imports."""
from dataclasses import asdict, dataclass
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MapDataset(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Any: ...


@runtime_checkable
class Predictor(Protocol):
    def predict(self, inputs: Any, **options: Any) -> Any: ...


@runtime_checkable
class SingleModelResult(Protocol):
    model: Any
    status: str


@dataclass(frozen=True)
class ResultInfo:
    family: str
    status: str
    model_count: int
    history_length: int | None = None
    metric_names: tuple = ()
    checkpoint: str | None = None
    run_path: str | None = None
    requested_steps: int | None = None
    actual_steps: int | None = None
    schema_version: int = 1

    def to_dict(self):
        """Independent JSON-compatible metadata; never serialize model weights."""
        return asdict(self)


def result_info(result):
    """Describe a supported NIL outcome without loading its backend or changing it.

    Numerical payloads remain on the original result. Native fit keeps its mapping
    interface; centralized RL keeps its policy mapping. This view does not make
    their training, persistence or prediction APIs interchangeable.
    """
    if isinstance(result, dict) and {'model','history','steps'} <= set(result):
        if type(result['model']).__module__ != 'nil.native':
            raise TypeError('Expected a NIL training result, not an arbitrary mapping')
        return ResultInfo('native', 'completed', 1, history_length=len(result['history']), actual_steps=result['steps'])
    if getattr(result, '_nil_classical_result', False): family = 'classical'
    elif getattr(result, '_nil_training_result', False): family = 'neural'
    elif getattr(result, '_nil_rl_result', False):
        family = 'rl.offline' if type(result).__module__ == 'nil.rl._offline' else 'rl.control'
    elif getattr(result, '_nil_centralized_result', False): family = 'rl.centralized'
    else: raise TypeError('Expected a supported NIL training result')
    status = result.status
    if status not in ('completed','cancelled','failed','loaded'):
        raise ValueError('Unknown result lifecycle status')
    history = getattr(result, 'history', None)
    metrics = getattr(result, 'metrics', {})
    checkpoint = getattr(result, 'checkpoint', None)
    run_path = getattr(getattr(result, 'run', None), 'path', None)
    return ResultInfo(family, status,
        len(result.policies) if family == 'rl.centralized' else 1,
        history_length=len(history) if history is not None else None,
        metric_names=tuple(sorted(metrics)), checkpoint=str(checkpoint) if checkpoint is not None else None,
        run_path=str(run_path) if run_path is not None else None,
        requested_steps=getattr(result, 'requested_steps', None),actual_steps=getattr(result, 'actual_steps', None))
