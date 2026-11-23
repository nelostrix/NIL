"""Lazy agent signals and explicitly optional neural environment-training APIs."""

from importlib import import_module

_EXPORTS = {
    'RLConfig': ('nil.rl.agent', 'RLConfig'),
    'RLEnabledAgent': ('nil.rl.agent', 'RLEnabledAgent'),
    'ExplorationScheduler': ('nil.rl.exploration', 'ExplorationScheduler'),
    'BehavioralCloning': ('nil.rl.policy', 'BehavioralCloning'),
    'Policy': ('nil.rl.policy', 'Policy'),
    'DirectPreferenceOptimizer': ('nil.rl.preference', 'DirectPreferenceOptimizer'),
    'EloTrajectoryRating': ('nil.rl.trajectory', 'EloTrajectoryRating'),
    'ReplayBuffer': ('nil.rl.replay', 'ReplayBuffer'),
    'Transition': ('nil.rl.replay', 'Transition'),
    'DenseRewardModel': ('nil.rl.rewards', 'DenseRewardModel'),
    'InverseRLRewardModel': ('nil.rl.rewards', 'InverseRLRewardModel'),
    'ShapedRewardModel': ('nil.rl.rewards', 'ShapedRewardModel'),
    'StepRecord': ('nil.rl.rewards', 'StepRecord'),
    'TrajectoryScore': ('nil.rl.trajectory', 'TrajectoryScore'),
    'TrajectoryScorer': ('nil.rl.trajectory', 'TrajectoryScorer'),
    'AdvantageEstimator': ('nil.rl.value', 'AdvantageEstimator'),
    'ValueFunction': ('nil.rl.value', 'ValueFunction'),
}
for _name in ('algorithms', 'create', 'train', 'evaluate', 'save', 'load', 'make_env', 'her_replay_buffer', 'submit', 'RLTrainingResult'):
    _EXPORTS[_name] = ('nil.rl._control', _name)
for _name in ('offline_fit','offline_load','offline_save','collect_vector','collect_multiagent','multiagent_fit','centralized_fit','centralized_load'):
    _EXPORTS[_name]=('nil.rl._advanced',_name)
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f'module nil.rl has no attribute {name!r}')
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module), symbol)
    globals()[name] = value
    return value
