# Contributing to Judo 🥋

Thank you for your interest in contributing to `judo`! 🎉
We welcome bug reports, feature suggestions, documentation improvements, and pull requests of all kinds.

## 📌 Guidelines Overview

- **Bug reports**: Please include a minimal reproducible example and relevant log output.
- **Feature requests**: Keep them concise and focused. If you're unsure, open an issue for discussion first.
- **Pull requests**: Follow our style guide, write clear commit messages, and document any new features or changes.

## ⚙️ Development Setup

We recommend using either `conda` or [`pixi`](https://pixi.sh) to manage your development environment.

### Conda Setup

```bash
conda create -n judo python=3.13
conda activate judo
pip install -e .[dev]
pre-commit install
pybind11-stubgen mujoco -o typings/
```

### Pixi Setup

```bash
# Install pixi (once)
curl -fsSL https://pixi.sh/install.sh | sh

# Activate dev environment
pixi shell -e dev

# First-time setup
pre-commit install
pybind11-stubgen mujoco -o typings/
```

## 🧹 Code Style and Linting

We use the following tools:

| Tool       | Purpose                    |
|------------|----------------------------|
| Ruff       | Formatting and linting     |
| Pyright    | Static type checking       |
| Pre-commit | Auto-formatting and checks |
| Pytest     | Unit and integration tests |
| Sphinx     | Documentation              |
| Codecov    | Code coverage reporting    |

To check your code before committing:

```bash
pre-commit run --all-files   # style, formatting, etc.
pyright                       # type checks (not part of pre-commit)
pytest                        # run the test suite
```

All CI checks must pass before a PR can be merged.

---

## 🧪 Adding a New Task or Optimizer

If you add a new task or optimizer:

1. Register it with the appropriate entry point in the codebase.
2. Update [`judo/tasks/README.md`](judo/tasks/README.md) with:
   - Task name
   - Brief description
   - Default parameters
   - Known limitations or tips

## 📝 Submitting a Pull Request

Before opening a pull request, please:

- Ensure your code is tested and documented.
- Run all pre-commit and type checks.
- Keep PRs focused and concise (open multiple PRs if needed).
- Use clear and descriptive commit messages.

Example:
```
feat(task): add fr3_stack task with novel constraints
fix(mppi): guard against div-by-zero in cost calculation
```

## 💬 Discussions

Have questions or ideas?

- Use [GitHub Issues](https://github.com/bdaiinstitute/judo/issues) to report bugs or suggest features.
- For general discussion, clarifications, or help, open a "Discussion" or start a draft PR.

## 🙏 Thanks

Judo is an early-stage research project maintained by a small team. We greatly appreciate community feedback and help. Whether you fix a typo or add a new planning strategy, your contribution makes a difference!

Thank you for supporting open-source robotics. 🤖
