"""Text classification with native sklearn APIs through NIL; no dataset downloads."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('./nil_runs/sklearn-mirror'))
    args = parser.parse_args()

    import nil
    from nil.sklearn.feature_extraction.text import TfidfVectorizer
    from nil.sklearn.linear_model import LogisticRegression
    from nil.sklearn.model_selection import GridSearchCV
    from nil.sklearn.pipeline import make_pipeline

    texts = ['good happy', 'great happy', 'good bright', 'great bright',
             'bad sad', 'awful sad', 'bad dark', 'awful dark']
    labels = [1, 1, 1, 1, 0, 0, 0, 0]
    model = make_pipeline(TfidfVectorizer(), LogisticRegression(random_state=7))
    search = GridSearchCV(model, {'logisticregression__C': [0.5, 1]}, cv=2, n_jobs=1)
    search.fit(texts, labels)
    result = nil.fit(search.best_estimator_, texts, labels,
                     validation=(['good happy', 'bad sad'], [1, 0]), output_dir=args.output)
    artifact = args.output / 'text-model'
    result.save(artifact)
    restored = nil.load(artifact, trusted=True)
    print({'backend': nil.sklearn.__version__, 'best_params': search.best_params_,
           'metrics': result.metrics,
           'predictions': restored.predict(['great bright', 'awful dark']).tolist(),
           'artifact': str(artifact)})


if __name__ == '__main__':
    main()
