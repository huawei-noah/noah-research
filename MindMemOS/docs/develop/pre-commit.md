# Pre-commit Checks

This repository uses `pre-commit` to check code formatting and secret leaks before each commit.

## Install

Install `gitleaks` first.

macOS:

```bash
brew install gitleaks
```

Windows:

```powershell
winget install gitleaks
gitleaks version
```

Chocolatey and Scoop also work:

```powershell
choco install gitleaks
scoop install gitleaks
```

Linux:

```bash
curl -sSfL https://raw.githubusercontent.com/gitleaks/gitleaks/master/install.sh | sh -s -- -b /usr/local/bin
gitleaks version
```

Then install project dependencies and Git hooks from the repository root:

```bash
make dev-setup
```

If dependencies are already installed and you only need the hooks:

```bash
make hooks-install
```

## Use

Commit as usual:

```bash
git add .
git commit -m "your message"
```

Before the commit is created, the hook runs:

- `ruff check --fix`: fixes auto-fixable lint and import ordering issues
- `ruff format`: formats Python code consistently
- `gitleaks protect --staged --redact`: scans staged files for secrets

Run hooks manually against staged files with:

```bash
uv run pre-commit run
```

## Common Cases

If Ruff changes files, run `git add` again and retry the commit.

If Gitleaks reports a leaked secret, do not commit it. Remove it or replace it with a test placeholder. Add an allowlist entry only for confirmed false positives.
