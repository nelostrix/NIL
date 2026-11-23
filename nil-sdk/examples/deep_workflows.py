"""Tiny actual CNN/LSTM/Transformer training and checkpoint roundtrips on CPU."""

import argparse
from pathlib import Path
import uuid


def build(architecture):
    import torch
    import nil
    nn = nil.nn
    if architecture == 'cnn':
        return nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.ReLU(),
                             nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 1))
    if architecture == 'lstm':
        class SequenceModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.sequence = nn.LSTM(2, 8, batch_first=True)
                self.head = nn.Linear(8, 1)

            def forward(self, inputs):
                sequence, _ = self.sequence(inputs)
                return self.head(sequence[:, -1])

        return SequenceModel()
    if architecture == 'transformer':
        layer = nn.TransformerEncoderLayer(8, 2, dim_feedforward=16, dropout=0., batch_first=True)
        return nn.Sequential(nn.Linear(2, 8), nn.TransformerEncoder(layer, 1),
                             nn.Flatten(1), nn.Linear(32, 1))
    raise ValueError('Choose cnn, lstm or transformer')


def exercise(architecture, output, *, epochs=8):
    import torch
    import nil
    from nil.training import checkpoint

    torch.set_num_threads(1)
    nil.runtime.seed(7)
    model = build(architecture)
    X = torch.ones(8, 1, 4, 4) if architecture == 'cnn' else torch.ones(8, 4, 2)
    data = nil.data.tensors(X, torch.ones(8, 1))
    before = nil.evaluate(model, data, device_name='cpu')['mse']
    result = nil.train(model, data, epochs=epochs, batch_size=8, learning_rate=.03,
                        device='cpu', output_dir=output)
    after = nil.evaluate(model, data, device_name='cpu')['mse']
    if not after < before:
        raise AssertionError('Tiny learning sanity check did not improve')
    restored = build(architecture)
    checkpoint.load(result.checkpoint, restored)
    model.eval()
    restored.eval()
    with torch.inference_mode():
        torch.testing.assert_close(model(X), restored(X), rtol=0, atol=0)
    return {'architecture': architecture, 'before_mse': before, 'after_mse': after,
            'checkpoint': str(result.checkpoint)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--architecture', choices=['cnn', 'lstm', 'transformer'], default='cnn')
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    output = args.output or Path('./nil_runs') / ('deep-' + uuid.uuid4().hex)
    print(exercise(args.architecture, output))


if __name__ == '__main__':
    main()
