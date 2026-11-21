"""Held-out classification, resampling/search, uncertainty and streaming; no downloads."""

import argparse
import json
from pathlib import Path
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    output = args.output or Path('./nil_runs') / ('extended-' + uuid.uuid4().hex)

    import numpy as np
    import pandas as pd
    import nil
    from nil.sklearn.datasets import make_classification

    output.mkdir(parents=True, exist_ok=False)

    X, y = make_classification(n_samples=150, n_features=4, n_redundant=0,
                               weights=[.75, .25], random_state=7)
    frame = pd.DataFrame(X, columns=['a', 'b', 'c', 'd'])
    frame['kind'] = np.where(frame['a'] > 0, 'positive', 'negative')
    final = nil.split(frame, y, method='stratified', seed=7)
    development = nil.split(*final.train, method='stratified', seed=8, test_size=.25)
    pipeline = nil.imbalance.pipeline(nil.model('logistic_regression', random_state=7),
        preprocess=nil.preprocessing.auto(), resample=nil.imbalance.sampler('over', seed=7))
    selected = nil.tune(pipeline, *development.train,
        {'model__C': {'type': 'float', 'low': .1, 'high': 10., 'log': True}},
        method='bayesian', max_trials=3, cv=3, max_fits=10,
        storage='sqlite:///' + str(output.resolve()) + '-search.db', output_dir=output)
    conformal = nil.uncertainty.calibrate(selected, *development.test, alpha=.1)
    thresholded = nil.imbalance.threshold(selected, *development.test, candidates=101)
    artifact = nil.save(conformal, output / 'conformal-model')
    restored = nil.load(artifact, trusted=True)
    np.testing.assert_array_equal(conformal.prediction_sets(final.X_test), restored.prediction_sets(final.X_test))
    snapshot = nil.data.snapshot(frame.to_dict('records'), output / 'dataset-v1')
    online = nil.streaming.create(nil.model('sgd_classifier', loss='log_loss', random_state=7),
                                  preprocess=nil.preprocessing.scaler(), classes=[0, 1])
    online.partial_fit(X[:75], y[:75])
    checkpoint = online.checkpoint(output / 'online-v1')
    resumed = nil.streaming.resume(checkpoint, trusted=True)
    resumed.partial_fit(X[75:], y[75:])
    print(json.dumps({'best_params': selected.best_params,
        'threshold_accuracy': thresholded.score(final.X_test, final.y_test),
        'uncertainty': nil.uncertainty.evaluate(conformal, *final.test),
        'snapshot_rows': len(nil.data.load_snapshot(snapshot)),
        'streaming_batches': resumed.batch_count_, 'output': str(output)}, indent=2))


if __name__ == '__main__':
    main()
