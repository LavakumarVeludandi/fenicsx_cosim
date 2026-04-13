# Contributing to fenicsx-cosim

Thanks for your interest in contributing.

## Development Setup

```bash
git clone https://github.com/LavakumarVeludandi/fenicsx_cosim
cd fenicsx_cosim
pip install -e ".[dev,fenicsx]"
```

## Running Tests

```bash
export PYTHONPATH=src
pytest tests/ -v
```

## Branch Naming

- Branch from `main`
- Use `feat/your-feature` for features
- Use `fix/your-bugfix` for bug fixes

## Opening a Pull Request

- Keep changes focused and scoped
- Ensure tests pass before requesting review
- Add a clear summary of what changed and why
- Link related issues when available
