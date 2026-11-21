"""Local classical pipeline acceptance example. Install the classical extra first."""
import argparse
import json

import nil


def main():
    parser = argparse.ArgumentParser(description="Train, validate, tune and reload a complete NIL classical pipeline")
    parser.add_argument("--output", default="./nil_runs")
    args = parser.parse_args()

    features = [[float(i), float("nan") if i % 9 == 0 else float(i % 4), "a" if i % 2 else "b"]
                for i in range(90)]
    targets = [2 * i + (3 if i % 2 else 0) for i in range(90)]
    data = nil.split(features, targets, seed=7)
    model = nil.pipeline(
        nil.model("ridge_regression"),
        preprocess=nil.preprocessing.tabular(numeric_columns=[0, 1], categorical_columns=[2]),
    )
    fitted = nil.fit(model, *data.train, validation=data.test, output_dir=args.output)
    cv = nil.cross_validate(model, *data.train, cv=3, seed=7)
    search = nil.tune(model, *data.train, {"model__alpha": [0.01, 0.1, 1.]},
                      method="grid", max_trials=3, cv=3, max_fits=10,
                      seed=7, output_dir=args.output)
    importance = nil.inspection.feature_importance(search, *data.test,
                                                   method="permutation", repeats=3, seed=7)
    curve = nil.inspection.learning_curve(model, *data.train, cv=3,
                                          train_sizes=(0.5, 1.), max_fits=6, seed=7)
    bundle = search.save(search.run.path / "model")
    loaded = nil.load(bundle, trusted=True)
    original, restored = search.predict(data.X_test), nil.predict(loaded, data.X_test)
    difference = max(abs(float(a) - float(b)) for a, b in zip(original, restored))
    if difference > 1e-10:
        raise RuntimeError("Saved pipeline did not reproduce predictions")
    print(json.dumps({
        "initial_metrics": fitted.metrics, "cv": cv.summary,
        "best_params": search.best_params, "held_out": search.evaluate(*data.test),
        "importance": importance, "learning_curve_fit_count": curve["fit_count"],
        "bundle": str(bundle), "roundtrip_max_difference": difference,
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
