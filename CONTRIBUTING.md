# Contributing to NIL

Thanks for helping improve NIL. Please read this guide before opening an issue
or pull request.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install '.[all,dev]'
python -m pytest nil-sdk/tests -q
```

Keep changes focused, document public API changes, and add a regression test
for behavior you change or introduce.

## Pull requests

1. Create a focused branch from `main`.
2. Run the relevant tests and formatting/lint checks locally.
3. Use a clear title and complete the pull request template.
4. Include tests, documentation, and release-note impact where applicable.
5. Keep generated files, credentials, local stores, and API keys out of commits.

Pull requests must pass required CI checks, receive review from a maintainer,
and be up to date with `main` before merging. Do not force-push shared branches.

## Commit messages

Use concise, imperative subjects such as `Add artifact registry validation`.
Keep unrelated changes in separate pull requests.

## Reporting security issues

Do not open a public issue for a vulnerability. Follow
[SECURITY.md](SECURITY.md).
