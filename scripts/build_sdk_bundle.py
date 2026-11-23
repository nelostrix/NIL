"""Build a local NIL wheel + docs/examples ZIP; never download dependencies."""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="dist")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = Path(args.output).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nil-bundle-") as temporary:
        wheels = Path(temporary)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--no-isolation",
                "--outdir",
                str(wheels),
                str(root),
            ],
            check=True,
        )
        (wheel,) = wheels.glob("*.whl")
        subprocess.run(
            [sys.executable, str(root / "scripts/verify_core_install.py"), str(wheel)],
            check=True,
        )
        files = {
            "PHASES.md": root / "PHASES.md",
            "docs/WINDOWS_EXECUTION.md": root / "docs/WINDOWS_EXECUTION.md",
            "docs/DEPLOYMENT.md": root / "docs/DEPLOYMENT.md",
            f"wheels/{wheel.name}": wheel,
            "README.md": root / "README.md",
            "TODO.md": root / "TODO.md",
            "docs/MODEL_DEVELOPMENT.md": root / "docs/MODEL_DEVELOPMENT.md",
            "docs/NEURAL_RL.md": root / "docs/NEURAL_RL.md",
            "docs/SHALLOW_DESIGN.md": root / "docs/SHALLOW_DESIGN.md",
            "docs/MODULE_REFERENCE.md": root / "docs/MODULE_REFERENCE.md",
            "docs/CLASSICAL_SPECIALIZED.md": root / "docs/CLASSICAL_SPECIALIZED.md",
            "docs/CLASSICAL_ML.md": root / "docs/CLASSICAL_ML.md",
            "docs/CLASSICAL_GAPS.md": root / "docs/CLASSICAL_GAPS.md",
            "docs/INSTALLATION.md": root / "docs/INSTALLATION.md",
            "docs/USERGUIDE.md": root / "docs/USERGUIDE.md",
            "pyproject.toml": root / "pyproject.toml",
            "scripts/verify_core_install.py": root / "scripts/verify_core_install.py",
        }
        files['docs/CPU_COMPLETION.md'] = root / 'docs/CPU_COMPLETION.md'
        files['scripts/benchmark_rl.py'] = root / 'scripts/benchmark_rl.py'
        files['scripts/benchmark_native.py'] = root / 'scripts/benchmark_native.py'
        files['scripts/verify_neural.py'] = root / 'scripts/verify_neural.py'
        for name in ('accept_serving.py','_tls_acceptance.py','accept_cgroup.py'):
            files['scripts/'+name]=root/'scripts'/name
        files['scripts/accept_gpu.py'] = root / 'scripts/accept_gpu.py'
        files['scripts/accept_execution.py'] = root / 'scripts/accept_execution.py'
        files['scripts/benchmark_workloads.py'] = root / 'scripts/benchmark_workloads.py'
        for path in sorted((root / 'docs/benchmarks').glob('*')):
            if path.is_file():
                files[path.relative_to(root).as_posix()] = path
        for path in sorted((root / "nil-sdk/examples").glob("*.py")):
            files[f"examples/{path.name}"] = path
        for path in sorted((root / "nil-sdk/nil/_help").glob("*/help.md")):
            files[path.relative_to(root).as_posix()] = path
        if (root / "SDK_IMPLEMENTATION_REPORT.md").is_file():
            files["SDK_IMPLEMENTATION_REPORT.md"] = root / "SDK_IMPLEMENTATION_REPORT.md"
        for path in sorted((root / "nil-sdk/tests").glob("*.py")):
            files[path.relative_to(root).as_posix()] = path
        for path in sorted((root / "benchmarks/budgets").glob("*.json")):
            files[path.relative_to(root).as_posix()] = path
        for path in sorted((root / ".github/workflows").glob("sdk*.yml")):
            files[path.relative_to(root).as_posix()] = path
        for path in sorted((root/"deploy").glob("*")):
            if path.is_file():files[path.relative_to(root).as_posix()]=path
        manifest = {
            "schema_version": 1,
            "wheel": wheel.name,
            "dependencies_bundled": False,
            "files": {
                name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()
            },
        }
        output = destination / f"{wheel.stem}-bundle.zip"
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, path in files.items():
                archive.write(path, name)
            archive.writestr("manifest.json", json.dumps(manifest, indent=2))
        print(output)


if __name__ == "__main__":
    main()
