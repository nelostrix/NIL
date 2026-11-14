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
        files = {
            f"wheels/{wheel.name}": wheel,
            "README.md": root / "docs/MODEL_DEVELOPMENT.md",
            "pyproject.toml": root / "pyproject.toml",
        }
        for path in sorted((root / "nil-sdk/examples").glob("*.py")):
            files[f"examples/{path.name}"] = path
        if (root / "SDK_IMPLEMENTATION_REPORT.md").is_file():
            files["SDK_IMPLEMENTATION_REPORT.md"] = root / "SDK_IMPLEMENTATION_REPORT.md"
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
