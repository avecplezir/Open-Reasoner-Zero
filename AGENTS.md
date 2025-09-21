# Repository Guidelines

## Project Structure & Module Organization
- `orz/`: Core Python package for RL/PPO.
  - Key submodules: `ppo/` (trainer, actors, models), `exps/` (experiment configs), `datasets/`, `exp_engine/`.
- `playground/`: Runnable training scripts (e.g., `orz_0.5b_ppo.py`, `orz_1.5b_ppo.py`, `orz_7b_ppo.py`).
- `data/`: Curated datasets referenced by playground scripts.
- `docker/`: CUDA-enabled Dockerfile and helper configs.
- `figure/`: Project images; `tamia/`: cluster/run shell helpers.
- `tests/`: Pytest tests (`test_*.py`).

## Build, Test, and Development Commands
- Setup env: `python -m venv .venv && source .venv/bin/activate && pip install -U pip && pip install -e .[dev]`.
- Format: `black orz playground` and `isort orz playground`.
- Lint: `flake8 orz playground`.
- Run locally:
  - Single GPU debug: `python -m playground.orz_0p5b_ppo_1gpu`
  - 0.5B on node: `python -m playground.orz_0p5b_ppo`
  - Multi-node: `ray start --head` → `python -m playground.orz_7b_ppo`

## Coding Style & Naming Conventions
- Python 3.10+, 4-space indent, 120-char max line length.
- Naming: modules/functions `snake_case`, classes `CamelCase`, constants `UPPER_CASE`.
- Prefer type hints and docstrings for public APIs.
- Tools: Black (isort profile), isort, Flake8 (configured in repo).

## Testing Guidelines
- Framework: `pytest` (+`pytest-cov`).
- Location: `tests/` with files `test_*.py`.
- Run: `pytest -v`; coverage: `pytest --cov=orz --cov-report=term-missing`.
- Target coverage for new logic; use minimal fixtures for datasets/configs.

## Commit & Pull Request Guidelines
- Commits: Conventional style, e.g., `feat: add PPO replay buffer`, `fix: guard None tokenizer`.
- PRs: include summary, motivation, linked issues, usage/run commands, before/after metrics or logs; ensure format/lint/tests pass.

## Tips & Config
- Debug quickstart: `DEBUG_MODE=True python -m playground.orz_14m_ppo_mini`.
- Ray clusters: master `ray start --head`; workers `ray start --address='<ip>:<port>'`.
- Keep large assets out of git; reference datasets under `data/` and document external credentials.

