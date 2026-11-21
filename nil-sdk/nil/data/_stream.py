"""Loaded only when a caller explicitly constructs an optional Torch stream."""
import json
from pathlib import Path
from nil.runtime import require
from nil.lifecycle import sha256

torch = require('torch', 'torch')


class JsonlStream(torch.utils.data.IterableDataset):
    def __init__(self, path, transform=None):
        self.path = Path(path).resolve()
        self.transform = transform
        self.offset = 0
        self.digest = sha256(self.path)
        self.size = self.path.stat().st_size

    def __iter__(self):
        if torch.utils.data.get_worker_info() is not None:
            raise ValueError('JSONL cursor requires a single worker, num_workers=0')
        self.validate_state_dict(self.state_dict())
        with self.path.open('rb') as stream:
            stream.seek(self.offset)
            while line := stream.readline(8 * 1024 * 1024 + 1):
                if len(line) > 8 * 1024 * 1024:
                    raise ValueError('JSONL record exceeds 8 MiB')
                if not line.strip():
                    self.offset = stream.tell()
                    continue
                record = json.loads(line)
                value = self.transform(record) if self.transform else record
                self.offset = stream.tell()
                yield value

    def state_dict(self):
        if self.path.stat().st_size != self.size or sha256(self.path) != self.digest:
            raise ValueError('JSONL source changed; use an immutable source')
        return {'schema_version': 1, 'sha256': self.digest, 'size': self.size, 'offset': self.offset}

    def validate_state_dict(self, state):
        if state.get('schema_version') != 1 or state.get('sha256') != self.digest or state.get('size') != self.size or sha256(self.path) != self.digest:
            raise ValueError('JSONL resume requires the same immutable source')
        offset = state.get('offset')
        if isinstance(offset, bool) or not isinstance(offset, int) or not 0 <= offset <= self.size:
            raise ValueError('Invalid JSONL cursor')
        if offset and offset != self.size:
            with self.path.open('rb') as stream:
                stream.seek(offset - 1)
                if stream.read(1) != b'\n':
                    raise ValueError('JSONL cursor must be at a record boundary')

    def load_state_dict(self, state):
        self.validate_state_dict(state)
        self.offset = state['offset']
