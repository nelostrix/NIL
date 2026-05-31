#!/usr/bin/env python3
"""Generate the nil-sdk project structure."""

import os

BASE = "nil-sdk"

DIRS = [
    "nil/kernel",
    "nil/memory",
    "nil/evolution",
    "nil/optimization",
    "nil/providers",
    "nil/tools",
    "tests",
    "examples",
    ".github/workflows",
]

FILES = [
    "pyproject.toml",
    "README.md",
    ".github/workflows/ci.yml",
    "nil/__init__.py",
    "nil/agent.py",
    "nil/domain.py",
    "nil/config.py",
    "nil/kernel/__init__.py",
    "nil/kernel/runner.py",
    "nil/kernel/tracer.py",
    "nil/kernel/tool_manager.py",
    "nil/kernel/outcome.py",
    "nil/memory/__init__.py",
    "nil/memory/store.py",
    "nil/memory/experience.py",
    "nil/memory/skill_registry.py",
    "nil/memory/strategy_version.py",
    "nil/evolution/__init__.py",
    "nil/evolution/analyzer.py",
    "nil/evolution/skill_extractor.py",
    "nil/evolution/strategy_mutator.py",
    "nil/evolution/evaluator.py",
    "nil/evolution/drift_detector.py",
    "nil/optimization/__init__.py",
    "nil/optimization/model_router.py",
    "nil/optimization/chain_compressor.py",
    "nil/optimization/cache_manager.py",
    "nil/optimization/budget_controller.py",
    "nil/providers/__init__.py",
    "nil/providers/anthropic.py",
    "nil/providers/openai.py",
    "nil/providers/stub.py",
    "nil/tools/__init__.py",
    "nil/tools/base.py",
    "nil/tools/registry.py",
    "tests/test_kernel.py",
    "tests/test_memory.py",
    "tests/test_evolution.py",
    "examples/basic_agent.py",
]


def main():
    for d in DIRS:
        os.makedirs(os.path.join(BASE, d), exist_ok=True)
        print(f"  mkdir {d}/")

    for f in FILES:
        path = os.path.join(BASE, f)
        if not os.path.exists(path):
            open(path, "w").close()
        print(f"  touch {f}")

    print(f"\nDone. Structure created at ./{BASE}/")


if __name__ == "__main__":
    main()
