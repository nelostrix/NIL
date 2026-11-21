"""Small native neural RL train/evaluate/save/reload workflow; no score guarantee."""

import argparse
from pathlib import Path
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--algorithm', choices=['ppo', 'a2c', 'dqn', 'sac', 'td3', 'ddpg'], default='ppo')
    parser.add_argument('--steps', type=int, default=128)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()

    import numpy as np
    import torch
    import nil
    torch.set_num_threads(1)
    output = args.output or Path('./nil_runs') / ('rl-' + uuid.uuid4().hex)
    discrete = args.algorithm in {'ppo', 'a2c', 'dqn'}
    environment = 'CartPole-v1' if discrete else 'Pendulum-v1'
    options = {'policy_kwargs': {'net_arch': [32, 32]}}
    if args.algorithm == 'ppo':
        options.update(n_steps=32, batch_size=32, n_epochs=2)
    elif args.algorithm == 'a2c':
        options.update(n_steps=8)
    else:
        options.update(buffer_size=1000, learning_starts=32, batch_size=32)
    model = nil.rl.create(args.algorithm, environment, seed=7, **options)
    result = nil.rl.train(model, total_timesteps=args.steps, output_dir=output)
    env = nil.rl.make_env(environment)
    report = nil.evaluate(result, env, episodes=2, max_steps=200, seed=7)
    observation, _ = env.reset(seed=7)
    expected = nil.predict(result, observation, deterministic=True)[0]
    path = result.save(output / 'policy-v1')
    restored = nil.rl.load(path, env=nil.rl.make_env(environment), trusted=True, restore_rng=True)
    np.testing.assert_array_equal(nil.predict(restored, observation, deterministic=True)[0], expected)
    print({'algorithm': args.algorithm, 'actual_steps': result.actual_steps,
           'mean_return': report['mean_return'], 'bundle': str(path)})
    model.get_env().close()
    restored.get_env().close()
    env.close()


if __name__ == '__main__':
    main()
