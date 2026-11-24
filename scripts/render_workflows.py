#!/usr/bin/env python3
"""Render the shipped passive inventory; --check detects documentation drift."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def render(data):
    lines = ['# Supported workflow inventory', '',
             f"Matrix {data['matrix_version']}; schema {data['schema_version']}; SDK {data['sdk_version']} ({data['release_channel']}).", '',
             'This is a bounded inventory, not a completeness score for the 32 roadmap areas.',
             'Recorded validation does not verify the current machine. Package detection does',
             'not verify binary compatibility, prerequisites or hardware. See [SDK contracts](SDK_CONTRACTS.md).', '',
             'Inspect offline with `nil workflows`, `nil workflows native.tensor --json`,',
             'or `nil.coverage.matrix()`. All APIs below resolve without optional backend imports.', '',
             '| Workflow | Implementation | Validation | Optional distributions |',
             '|---|---|---|---|']
    for row in data['workflows']:
        lines.append(f"| {row['id']} | {row['implementation']} | {row['validation']} | {', '.join(row['requires']) or 'none'} |")
    for row in data['workflows']:
        lines += ['', f"## {row['id']}", '', row['title'], '',
                  f"Areas: {', '.join(map(str,row['area_ids']))}. Offline help: `nil help {row['help']}`.", '',
                  'APIs: ' + (', '.join('`'+x+'`' for x in row['api']) or 'No accepted public workflow.'), '',
                  'Input: ' + row['inputs']['description'], '', 'Output: ' + row['outputs']['description'], '',
                  'Limits: ' + ' '.join(row['limits']), '',
                  'Prerequisites: ' + ('; '.join(row['prerequisites']) or 'No additional declared prerequisites.'), '',
                  'Targets: `' + json.dumps(row['targets'],sort_keys=True) + '`', '',
                  'Recorded profiles: ' + (', '.join(row['tested_profiles']) or 'See workflow-specific evidence; no backend-matrix acceptance claimed.'), '',
                  'Evidence: ' + ', '.join(f'[{Path(p).name}](../{p})' for p in row['evidence'])]
    lines += ['', '## Recorded backend profiles', '',
              'These profiles describe historical verification environments. Native workflows',
              'need no numerical backend even when their reference tests ran in a Torch environment.', '']
    for name, profile in data['profiles'].items():
        lines += [f"- **{name}**: Python {profile['python']}; {profile['platform']}; " +
                  ', '.join(f'{k}={v}' for k,v in profile['versions'].items()) +
                  f". [Evidence](../{profile['evidence']})."]
    return '\n'.join(lines)+'\n'


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    destination=ROOT/'docs/WORKFLOWS.md'
    content=render(json.loads((ROOT/'nil-sdk/nil/_workflows.json').read_text()))
    if args.check:
        if not destination.exists() or destination.read_text()!=content:
            raise SystemExit('Workflow documentation differs; run scripts/render_workflows.py')
    else:
        destination.write_text(content)
