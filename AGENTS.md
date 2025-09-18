# Repository Guidelines

## Project Structure & Modules
- `orz/`: Core Python package (RL and PPO). Key submodules: `ppo/` (trainer, actors, models), `exps/` (experiment configs), `datasets/`, `exp_engine/`.
- `playground/`: Runnable training scripts (e.g., `orz_0p5b_ppo.py`, `orz_1p5b_ppo.py`, `orz_7b_ppo.py`).
- `data/`: Curated datasets referenced by playground scripts.
- `docker/`: CUDA-enabled Dockerfile and helper configs for reproducible environments.
- `figure/`: Project images for docs; `tamia/`: cluster/run shell helpers.
- `tests/`: Add your tests here (pytest discovers `tests/`).

## Setup, Build, and Dev Commands
- Create env and install: `python -m venv .venv && source .venv/bin/activate && pip install -U pip && pip install -e .[dev]`.
- Format: `black orz playground` and `isort orz playground`.
- Lint: `flake8 orz playground`.
- Run locally (examples):
  - Single GPU debug: `python -m playground.orz_0p5b_ppo_1gpu`
  - 0.5B on node: `python -m playground.orz_0p5b_ppo`
  - Multi‑node (master): `ray start --head` → then `python -m playground.orz_7b_ppo`

## Coding Style & Naming
- Python 3.10+. Indentation 4 spaces; max line length 120.
- Tools: Black (+isort profile), Flake8 (configured in `.flake8`/`pyproject.toml`).
- Naming: modules/functions `snake_case`, classes `CamelCase`, constants `UPPER_CASE`. Prefer type hints and docstrings for public APIs.

## Testing Guidelines
- Framework: `pytest` (+`pytest-cov`). Place tests in `tests/` as `test_*.py`.
- Run: `pytest -v`; coverage: `pytest --cov=orz --cov-report=term-missing`.
- Aim for coverage on new logic; include minimal fixtures for datasets/configs.

## Commits & Pull Requests
- History uses short verbs (e.g., "init", "upd"). Prefer clear, Conventional Commit style for new work: `feat: add PPO replay buffer`, `fix: guard None tokenizer`.
- PRs: include summary, motivation, linked issues, usage/run commands, before/after metrics or logs, and screenshots if UX-facing. Ensure tests/lint/format pass.

## Tips & Config
- Debug mode: `DEBUG_MODE=True python -m playground.orz_14m_ppo_mini`.
- Ray clusters: start with `ray start --head` (master) and `ray start --address='<ip>:<port>'` (workers).
- Keep large assets out of git; use dataset paths under `data/` and document any external credentials.
