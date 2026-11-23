# cli

Use the installed command-line entry point for diagnostics, model listing, help and an explicit training demo.

## Usage

```python
import subprocess
import sys

subprocess.run([sys.executable, "-m", "nil", "models"], check=True)
```

## Parameters

Subcommands: doctor, models, help [topic], help --list, demo --output PATH.

## Output

Printed diagnostics/help or explicit demo run artifacts.

## Requirements

Core SDK for help/list/doctor. torch for demo.

## Limitations

Listing/help never download packages or load model weights. The training demo performs real compute and writes files.
