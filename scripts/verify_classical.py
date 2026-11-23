"""Require a real classical backend before running acceptance (no skip-only success)."""
import argparse
import importlib
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Verify NIL's native classical workflows")
    parser.add_argument("--full", action="store_true", help="Also require pandas and XGBoost acceptance")
    parser.add_argument("--extended", action="store_true", help="Require pandas, imbalance and resumable Bayesian search")
    parser.add_argument("--specialized", action="store_true", help="Require actual LightGBM/CatBoost/survival and specialized workflows")
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    modules = ["sklearn", "numpy", "scipy", "joblib", "pytest"]
    if args.full:
        modules += ["pandas", "xgboost"]
    if args.extended:
        modules += ["pandas", "imblearn", "optuna"]
    if args.specialized:
        modules += ["lightgbm", "catboost", "lifelines", "pandas"]
    for name in dict.fromkeys(modules):
        try:
            module = importlib.import_module(name)
        except ImportError as error:
            print(f"Cannot verify native classical workflows: {name} is unavailable ({error}).", file=sys.stderr)
            print("Install .[classical,dev]; --full needs pandas and boosting or boosting-cpu, --extended needs tabular/imbalance/search.", file=sys.stderr)
            return 2
        print(f"{name}: {getattr(module, '__version__', 'available')}")
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(root / "nil-sdk"), os.environ.get("PYTHONPATH")]))}
    tests = sorted(str(path) for path in (root / "nil-sdk/tests").glob("test_classical_*.py"))
    tests.append(str(root / "nil-sdk/tests/test_sklearn_mirror.py"))
    if args.specialized:
        tests.extend(str(root / 'nil-sdk/tests' / name) for name in ('test_specialized_ml.py','test_typed_datasets.py','test_advanced_uncertainty.py','test_distributed_search.py'))
    return subprocess.call([sys.executable, "-m", "pytest", *tests, "-q"], cwd=root, env=env)


if __name__ == "__main__":
    sys.exit(main())
