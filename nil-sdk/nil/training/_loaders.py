"""Explicit epoch-boundary data contracts; no backend import at module import."""
import itertools
import random


def seed_worker(_worker_id):
    import torch
    random.seed(torch.initial_seed())
    try:
        import numpy
        numpy.random.seed(torch.initial_seed() % (2 ** 32))
    except ImportError:
        pass


def _kind(value):
    return f'{type(value).__module__}.{type(value).__qualname__}'


class LoaderContract:
    def __init__(self, data, config, torch, training=True):
        self.data, self.config, self.torch, self.training = data, config, torch, training
        self.external = isinstance(data, torch.utils.data.DataLoader)
        self.dataset = data.dataset if self.external else data
        self.iterable = isinstance(self.dataset, torch.utils.data.IterableDataset)
        workers = data.num_workers if self.external else config.num_workers
        if workers > 8:
            raise ValueError('NIL manages at most eight loader workers')
        persistent = data.persistent_workers if self.external else config.persistent_workers
        if persistent:
            raise ValueError('Persistent worker RNG continuation is unsupported; use persistent_workers=False')
        prefetch = data.prefetch_factor if self.external else config.prefetch_factor
        if workers and (prefetch is None or not 1 <= prefetch <= 8):
            raise ValueError('Worker prefetch_factor must be between one and eight')
        if self.iterable and workers:
            raise ValueError('Resumable iterable datasets require num_workers=0 to avoid unacknowledged prefetch')
        if self.iterable and config.max_batches_per_epoch is None:
            raise ValueError('Iterable datasets require max_batches_per_epoch')
        if self.iterable and not self.external and training and config.shuffle:
            raise ValueError('Iterable datasets require shuffle=False')
        if self.iterable and not all(callable(getattr(self.dataset, name, None)) for name in ('state_dict', 'validate_state_dict', 'load_state_dict')):
            raise ValueError('Iterable datasets require an explicit resumable state protocol')
        self.objects = {'dataset': self.dataset}
        self.generators = {}
        if self.external:
            self.objects['sampler'] = data.sampler
            self.objects['batch_sampler'] = data.batch_sampler
            for name, value in [('loader', data), ('sampler', data.sampler)]:
                generator = getattr(value, 'generator', None)
                if generator is not None:
                    self.generators[name] = generator
            for name in ('sampler', 'batch_sampler'):
                obj = self.objects[name]
                if not type(obj).__module__.startswith('torch.utils.data.') and not callable(getattr(obj, 'state_dict', None)):
                    raise ValueError('Custom samplers require state_dict/validate_state_dict/load_state_dict')
        self.stateful = {}
        for name, obj in self.objects.items():
            if callable(getattr(obj, 'state_dict', None)):
                if workers:
                    raise ValueError('Stateful datasets/samplers require num_workers=0 for exact state ownership')
                if not all(callable(getattr(obj, method, None)) for method in ('validate_state_dict', 'load_state_dict')):
                    raise ValueError('State protocol requires nonmutating validate_state_dict and load_state_dict')
                self.stateful[name] = obj
        try:
            length = len(self.dataset)
        except TypeError:
            length = None
        self.description = {
            'external': self.external, 'dataset': _kind(self.dataset), 'length': length,
            'iterable': self.iterable, 'workers': workers, 'prefetch': prefetch if workers else None,
            'pin_memory': data.pin_memory if self.external else config.pin_memory,
            'batch_size': data.batch_size if self.external else config.batch_size,
            'drop_last': data.drop_last if self.external else False,
            'sampler': _kind(data.sampler) if self.external else None,
            'batch_sampler': _kind(data.batch_sampler) if self.external else None,
            'sampler_options': {name: getattr(data.sampler, name, None) for name in ('replacement', 'num_samples', 'seed', 'shuffle', 'rank', 'num_replicas', 'drop_last')} if self.external else {},
            'batch_sampler_options': {name: getattr(data.batch_sampler, name, None) for name in ('batch_size', 'drop_last')} if self.external else {},
            'generator_names': sorted(self.generators), 'state_names': sorted(self.stateful),
            'max_batches_per_epoch': config.max_batches_per_epoch,
        }

    def state_dict(self):
        return {'description': self.description,
                'generators': {name: obj.get_state() for name, obj in self.generators.items()},
                'objects': {name: obj.state_dict() for name, obj in self.stateful.items()}}

    def validate_state_dict(self, saved):
        if saved.get('description') != self.description:
            raise ValueError('Resume requires the same DataLoader/dataset/sampler contract')
        if set(saved.get('generators', {})) != set(self.generators) or set(saved.get('objects', {})) != set(self.stateful):
            raise ValueError('Incompatible loader state')
        for name, state in saved['generators'].items():
            self.torch.Generator().set_state(state)
        for name, state in saved['objects'].items():
            self.stateful[name].validate_state_dict(state)

    def load_state_dict(self, saved):
        for name, state in saved['generators'].items():
            self.generators[name].set_state(state)
        for name, state in saved['objects'].items():
            self.stateful[name].load_state_dict(state)

    def loader(self, epoch):
        if self.external:
            loader = self.data
            sampler = loader.sampler
            if callable(getattr(sampler, 'set_epoch', None)):
                sampler.set_epoch(epoch)
        else:
            options = {}
            if self.config.num_workers:
                options['prefetch_factor'] = self.config.prefetch_factor
            loader = self.torch.utils.data.DataLoader(
                self.data, batch_size=self.config.batch_size,
                shuffle=self.config.shuffle if self.training else False,
                generator=self.torch.Generator().manual_seed(self.config.seed + epoch),
                num_workers=self.config.num_workers, pin_memory=self.config.pin_memory,
                worker_init_fn=seed_worker, **options)
        if self.config.max_batches_per_epoch is not None:
            return itertools.islice(loader, self.config.max_batches_per_epoch)
        return loader
