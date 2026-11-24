"""Lazy entry points for opt-in neural offline and structured rollout workflows."""
def offline_fit(*args,**kwargs):
    from nil.rl._offline import fit
    return fit(*args,**kwargs)
def offline_load(*args,**kwargs):
    from nil.rl._offline import load
    return load(*args,**kwargs)
def offline_save(*args,**kwargs):
    from nil.rl._offline import save
    return save(*args,**kwargs)
def collect_vector(*args,**kwargs):
    from nil.rl._rollouts import collect_vector as run
    return run(*args,**kwargs)
def collect_multiagent(*args,**kwargs):
    from nil.rl._rollouts import collect_multiagent as run
    return run(*args,**kwargs)
def multiagent_fit(*args,**kwargs):
    from nil.rl._rollouts import multiagent_fit as run
    return run(*args,**kwargs)
def centralized_fit(*args,**kwargs):
    from nil.rl._centralized import fit
    return fit(*args,**kwargs)
def centralized_load(*args,**kwargs):
    from nil.rl._centralized import load
    return load(*args,**kwargs)
